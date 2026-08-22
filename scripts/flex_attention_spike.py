# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DIAGNOSTIC (not part of training/eval): can FlexAttention fix the win28 OOM?

The SCRIPT attention mask is only ~2% dense at the shape that OOM'd
(audio_window_frames=28, chunk_size=2), yet ``build_script_mask`` materialises a
dense ``(B, 1, T, T)`` tensor and the model computes every one of the T^2 pairs.
FlexAttention expresses the same rule as a predicate and skips fully-masked
128x128 blocks instead.

This script answers three questions on real hardware, which CPU cannot:

  1. Does flex reproduce the dense-mask logits AND gradients to the parity bar?
  2. How much peak memory does each option actually use?
  3. Which options survive which batch size at the failing shape?

Four arms are compared: {dense, flex} x {no checkpointing, activation checkpointing}.

Run:  python scripts/flex_attention_spike.py --llm <path-or-name> [--seq_len 11198]
"""

import argparse
import gc
import traceback

import torch

from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX
from nemo.collections.speechlm2.parts.script import (
    ChunkSpec,
    build_packed_chunk_example,
    build_script_mask,
    collate_packed_chunk_examples,
)

VS, VE, EOT = 90, 91, 92


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# The failing shape
# ---------------------------------------------------------------------------


def build_examples(batch_size, chunk_frames, window_frames, n_chunks, words_every):
    chunks = []
    for i in range(n_chunks):
        tgt = [int(20 + (i % 7)), int(40 + (i % 5))] if (i % words_every == 0) else []
        chunks.append(ChunkSpec(chunk_frames, tgt))
    instr = list(range(5, 34))
    ex = build_packed_chunk_example(instr, chunks, VS, VE, EOT, audio_window_frames=window_frames)
    return [ex] * batch_size


def build_batch(batch_size, chunk_frames, window_frames, n_chunks, words_every, vocab):
    """Reproduce the training layout that OOM'd (54s clip, chunk 2, window 28)."""
    chunks = []
    for i in range(n_chunks):
        tgt = [int(20 + (i % 7)), int(40 + (i % 5))] if (i % words_every == 0) else []
        chunks.append(ChunkSpec(chunk_frames, tgt))
    instr = list(range(5, 34))  # 29-token instruction, as in the real recipe
    ex = build_packed_chunk_example(instr, chunks, VS, VE, EOT, audio_window_frames=window_frames)
    return collate_packed_chunk_examples([ex] * batch_size, pad_id=0)


def to_device(b_, device):
    """Move every tensor field of a BatchedPackedChunk onto ``device``.

    The mask_mod closes over these tensors and is evaluated with CUDA indices, so
    leaving any of them on CPU raises an index/device mismatch inside vmap.
    """
    return type(b_)(**{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vars(b_).items()})


def make_mask_mod(b_):
    seg, pos, pref, val = b_.seg_ids, b_.position_ids, b_.prefix_len, b_.valid

    def mask_mod(b, h, q, kv):
        qs, ks = seg[b, q], seg[b, kv]
        qp, kp = pos[b, q], pos[b, kv]
        q_spine, k_spine = qs == 0, ks == 0
        causal = kp <= qp
        return (
            (q_spine & k_spine & causal)
            | ((~q_spine) & k_spine & (kp < pref[b, q]))
            | ((qs == ks) & (~q_spine) & causal)
        ) & val[b, kv]

    return mask_mod


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_llm(path, attn_impl, dtype, device, lora_r, lora_alpha):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, attn_implementation=attn_impl, trust_remote_code=False
    )
    if lora_r > 0:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(task_type="CAUSAL_LM", r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0),
        )
    return model.to(device).train()


def set_checkpointing(model, on):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if on:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    elif hasattr(base, "gradient_checkpointing_disable"):
        base.gradient_checkpointing_disable()


def build_mask(kind, b_, dtype, device):
    """Return the object to pass as ``attention_mask`` for this arm."""
    if kind == "script":
        return None  # the structured backend reads its plan from the context
    if kind == "dense":
        return build_script_mask(b_.seg_ids, b_.position_ids, b_.prefix_len, b_.valid, dtype)
    from torch.nn.attention.flex_attention import create_block_mask

    B, T = b_.seg_ids.shape
    return create_block_mask(make_mask_mod(b_), B=B, H=None, Q_LEN=T, KV_LEN=T, device=device)


def step(model, b_, mask, embeds, do_backward):
    out = model(inputs_embeds=embeds, attention_mask=mask, position_ids=b_.position_ids, use_cache=False)
    logits = out.logits
    if do_backward:
        # Same reduction the real training step uses.
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1).float(), b_.target_ids.flatten(0, 1), ignore_index=-100
        )
        loss.backward()
    return logits


def measure(model, b_, kind, embeds, device, dtype, do_backward=True, plan=None, iters=3):
    """(peak MiB, median step seconds) for forward+backward, or (None, None) on OOM.

    Times AFTER a warmup iteration and with explicit synchronisation, since CUDA
    launches are async and the first call pays compilation/autotune costs.
    """
    import time

    from nemo.collections.speechlm2.parts.script_attention import script_attention_plan

    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        mask = build_mask(kind, b_, dtype, device)
        ctx = lambda: script_attention_plan(plan if kind == "script" else None)
        with ctx():  # warmup, excluded from timing
            step(model, b_, mask, embeds, do_backward)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / 2**20

        times = []
        for _ in range(iters):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with ctx():
                step(model, b_, mask, embeds, do_backward)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
        times.sort()
        secs = times[len(times) // 2]
    except torch.OutOfMemoryError:
        peak, secs = None, None
    finally:
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    return peak, secs


# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm", default="Qwen/Qwen3-1.7B")
    p.add_argument("--chunk_frames", type=int, default=2)
    p.add_argument("--window_frames", type=int, default=28)
    p.add_argument("--n_chunks", type=int, default=338, help="54s at chunk_size=2")
    p.add_argument("--words_every", type=int, default=2, help="1 in N chunks reveals words")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--lora_r", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=256)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument(
        "--parity_dtypes",
        nargs="+",
        default=["fp32", "bf16"],
        help="run the parity check in each of these; fp32 is the decisive one",
    )
    p.add_argument("--skip_memory", action="store_true")
    args = p.parse_args()

    assert torch.cuda.is_available(), "this diagnostic needs a GPU"
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    log(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | dtype={dtype}")

    probe = to_device(
        build_batch(1, args.chunk_frames, args.window_frames, args.n_chunks, args.words_every, 128), device
    )
    T = probe.input_ids.shape[1]
    dense_mb = T * T * 2 / 2**20
    log(f"layout: chunk={args.chunk_frames} window={args.window_frames} n_chunks={args.n_chunks} -> T={T}")
    log(f"dense mask alone would be {dense_mb:.0f} MiB/sample")

    def exs_for(bs):
        return build_examples(bs, args.chunk_frames, args.window_frames, args.n_chunks, args.words_every)

    globals()["exs_for"] = exs_for

    from nemo.collections.speechlm2.parts.script_attention import register_script_attention

    register_script_attention()

    from torch.nn.attention.flex_attention import create_block_mask

    bm = create_block_mask(make_mask_mod(probe), B=1, H=None, Q_LEN=T, KV_LEN=T, device=device)
    log(f"BlockMask: {bm.sparsity():.1f}% of blocks fully skipped")
    log("")

    # ---------------- 1. parity: logits and gradients ----------------
    log("=" * 74)
    log("1. PARITY  (flex BlockMask vs dense 4D mask)")
    log("=" * 74)
    small = to_device(build_batch(1, args.chunk_frames, args.window_frames, 24, args.words_every, 128), device)
    Ts = small.input_ids.shape[1]
    n_frames = int(small.audio_frame_index.max()) + 1

    # Run in fp32 as well as bf16. Two different attention kernels accumulating in
    # bf16 across 28 layers disagree by a visible amount even when they compute the
    # same function, so bf16 alone cannot distinguish "wrong mask" from "rounding".
    # fp32 is the decisive arm: a genuine mask difference stays large there.
    # Three arms, so we have a CONTROL. "eager" and "sdpa" both consume the same
    # dense SCRIPT mask and compute the same function, so eager-vs-sdpa measures
    # how far two different kernels drift through 28 layers on identical maths.
    # flex is only suspect if it drifts materially further than that.
    ARMS = {"eager": "eager", "sdpa": "sdpa", "flex": "flex_attention"}
    for pdtype_name in args.parity_dtypes:
        pdtype = torch.float32 if pdtype_name == "fp32" else torch.bfloat16
        logits, grads = {}, {}
        for kind, impl in ARMS.items():
            # Seed BEFORE constructing the model: get_peft_model draws lora_A
            # randomly, so both arms must be seeded identically or they are
            # different models (lora_B is zero-init, so this shows up in the
            # gradients rather than the forward).
            torch.manual_seed(0)
            m = load_llm(args.llm, impl, pdtype, device, args.lora_r, args.lora_alpha)
            torch.manual_seed(0)
            frames = torch.randn(n_frames, m.config.hidden_size, device=device, dtype=pdtype)
            ids = small.input_ids.clone()
            amask = ids == AUDIO_TOKEN_IDX
            ids[amask] = 0
            emb = m.get_input_embeddings()(ids)
            emb = torch.where(amask.unsqueeze(-1), frames[small.audio_frame_index.clamp(min=0)], emb)
            mask = build_mask("flex" if kind == "flex" else "dense", small, pdtype, device)
            lg = step(m, small, mask, emb, do_backward=True)
            logits[kind] = lg.detach().float().cpu()
            grads[kind] = [
                p.grad.detach().float().cpu() for _, p in sorted(m.named_parameters()) if p.grad is not None
            ]
            del m, emb, mask, lg, frames
            gc.collect()
            torch.cuda.empty_cache()

        def cmp(x, y):
            A, B = logits[x], logits[y]
            d = (A - B).abs().max().item()
            # If the MASK differed, the arms would disagree about which token comes
            # next, not merely about the last bits of a logit.
            return d, d / A.abs().max().item(), (A.argmax(-1) == B.argmax(-1)).float().mean().item()

        for x, y, note in [
            ("eager", "sdpa", "CONTROL: identical mask, different kernel"),
            ("eager", "flex", "flex under test"),
        ]:
            d, rel, agree = cmp(x, y)
            log(
                f"  [{pdtype_name}] T={Ts} {x:5s} vs {y:5s}  max|dlogit|={d:.3e}  rel={rel:.2e}  "
                f"argmax={agree*100:6.2f}%   ({note})"
            )
        if len({len(g) for g in grads.values()}) == 1 and grads["eager"]:
            gmax = max(u.abs().max().item() for u in grads["eager"])
            for x, y in [("eager", "sdpa"), ("eager", "flex")]:
                dg = max((u - v).abs().max().item() for u, v in zip(grads[x], grads[y]))
                log(f"  [{pdtype_name}]       {x:5s} vs {y:5s}  max|dgrad| ={dg:.3e}  rel={dg/max(gmax,1e-12):.2e}")
        # flex is fine if its drift is comparable to the control's; only a much
        # larger drift indicates it is computing a different function.
        de = cmp("eager", "sdpa")[0]
        df = cmp("eager", "flex")[0]
        ratio = df / max(de, 1e-12)
        log(
            f"  [{pdtype_name}] VERDICT: flex drift is {ratio:.2f}x the control -> "
            f"{'OK (kernel noise)' if ratio <= 3.0 else 'SUSPECT (beyond kernel noise)'}"
        )
        log("")
    log("")

    # ---------------- 2. memory at the failing shape ----------------
    if args.skip_memory:
        log("(memory section skipped)")
        return
    log("=" * 74)
    log(f"2. PEAK MEMORY at the OOM shape (T={T}), forward+backward")
    log("=" * 74)
    log(f"  {'arm':<22} " + " ".join(f"B={b:<13}" for b in args.batch_sizes))
    log("  (peak GiB / median step seconds, forward+backward, after warmup)")

    for kind in ("dense", "flex", "script"):
        impl = {"dense": "eager", "flex": "flex_attention", "script": "script"}[kind]
        for ckpt in (False, True):
            name = f"{kind}{' + ckpt' if ckpt else ''}"
            try:
                m = load_llm(args.llm, impl, dtype, device, args.lora_r, args.lora_alpha)
                set_checkpointing(m, ckpt)
            except Exception as e:
                log(f"  {name:<22} load failed: {type(e).__name__}: {e}")
                continue
            cells = []
            for B in args.batch_sizes:
                try:
                    b_ = build_batch(B, args.chunk_frames, args.window_frames, args.n_chunks, args.words_every, 128)
                    b_ = to_device(b_, device)
                    torch.manual_seed(0)
                    nf = int(b_.audio_frame_index.max()) + 1
                    frames = torch.randn(nf, m.config.hidden_size, device=device, dtype=dtype)
                    ids = b_.input_ids.clone()
                    am = ids == AUDIO_TOKEN_IDX
                    ids[am] = 0
                    emb = m.get_input_embeddings()(ids)
                    emb = torch.where(am.unsqueeze(-1), frames[b_.audio_frame_index.clamp(min=0)], emb)
                    plan = None
                    if kind == "script":
                        from nemo.collections.speechlm2.parts.script_attention import build_attention_plan

                        plan = build_attention_plan(exs_for(B)).to(device)
                    peak, secs = measure(m, b_, kind, emb, device, dtype, plan=plan)
                    cells.append(f"{peak/1024:.1f}G/{secs:.2f}s" if peak else "OOM")
                    del b_, emb, frames
                except torch.OutOfMemoryError:
                    cells.append("OOM")
                except Exception as e:
                    cells.append(f"ERR({type(e).__name__})")
                    traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()
            log(f"  {name:<22} " + " ".join(f"{c:<15}" for c in cells))
            del m
            gc.collect()
            torch.cuda.empty_cache()
    log("")
    log("done.")


if __name__ == "__main__":
    main()
