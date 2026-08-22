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


def measure(model, b_, kind, embeds, device, dtype, do_backward=True):
    """Peak allocated MiB for one forward(+backward), or None on OOM."""
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        mask = build_mask(kind, b_, dtype, device)
        step(model, b_, mask, embeds, do_backward)
        peak = torch.cuda.max_memory_allocated(device) / 2**20
    except torch.OutOfMemoryError:
        peak = None
    finally:
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    return peak


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
    for pdtype_name in args.parity_dtypes:
        pdtype = torch.float32 if pdtype_name == "fp32" else torch.bfloat16
        logits, grads = {}, {}
        for kind in ("dense", "flex"):
            impl = "eager" if kind == "dense" else "flex_attention"
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
            mask = build_mask(kind, small, pdtype, device)
            lg = step(m, small, mask, emb, do_backward=True)
            logits[kind] = lg.detach().float().cpu()
            grads[kind] = [
                p.grad.detach().float().cpu() for _, p in sorted(m.named_parameters()) if p.grad is not None
            ]
            del m, emb, mask, lg, frames
            gc.collect()
            torch.cuda.empty_cache()

        a, b = logits["dense"], logits["flex"]
        dl = (a - b).abs().max().item()
        rel = dl / a.abs().max().item()
        # If the mask differed, the models would disagree about WHICH token comes
        # next, not merely about the last few bits of the logit.
        agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
        tol = 1e-4 if pdtype is torch.float32 else 5e-2
        verdict = "PASS" if dl < tol else "FAIL"
        log(
            f"  [{pdtype_name}] T={Ts}  max|dlogit|={dl:.3e}  rel={rel:.2e}  "
            f"argmax agreement={agree*100:.2f}%  -> {verdict} (tol {tol:g})"
        )
        if len(grads["dense"]) == len(grads["flex"]) and grads["dense"]:
            dg = max((x - y).abs().max().item() for x, y in zip(grads["dense"], grads["flex"]))
            gmax = max(x.abs().max().item() for x in grads["dense"])
            log(f"  [{pdtype_name}] max|dgrad|={dg:.3e}  (grad scale {gmax:.3e}, rel {dg/max(gmax,1e-12):.2e})")
    log("")

    # ---------------- 2. memory at the failing shape ----------------
    if args.skip_memory:
        log("(memory section skipped)")
        return
    log("=" * 74)
    log(f"2. PEAK MEMORY at the OOM shape (T={T}), forward+backward")
    log("=" * 74)
    log(f"  {'arm':<22} " + " ".join(f"B={b:<9}" for b in args.batch_sizes))

    for kind in ("dense", "flex"):
        impl = "eager" if kind == "dense" else "flex_attention"
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
                    peak = measure(m, b_, kind, emb, device, dtype)
                    cells.append(f"{peak/1024:.1f}GiB" if peak else "OOM")
                    del b_, emb, frames
                except torch.OutOfMemoryError:
                    cells.append("OOM")
                except Exception as e:
                    cells.append(f"ERR({type(e).__name__})")
                    traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()
            log(f"  {name:<22} " + " ".join(f"{c:<11}" for c in cells))
            del m
            gc.collect()
            torch.cuda.empty_cache()
    log("")
    log("done.")


if __name__ == "__main__":
    main()
