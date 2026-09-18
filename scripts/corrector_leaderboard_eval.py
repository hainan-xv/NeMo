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
"""REAL decoding for the verifier/corrector -- no reference anywhere.

THIS IS NOT ``_corrected_wer``. That metric is an ORACLE: a rejected chunk is
replaced by the reference span, so it measures decision quality with a perfect
corrector and is minimised by rejecting everything. Here a rejected chunk is
replaced by what the model actually GENERATES, and the conditioning history is
the text the model has actually EMITTED. Nothing derived from the reference
enters the loop, so the WER this reports is a WER you could ship.

Expect it to be WORSE than val_corrected_wer, and that gap is the point: it is
the cost of the corrector having to write the fix rather than being handed it.

TRAIN/INFERENCE MISMATCH, stated because it is real and not fixed here. Training
conditions each chunk on the REFERENCE history (see
corrector_examples_for_utterance); this conditions on the emitted history. An
early error therefore propagates here in a way it never did during training.

  python scripts/corrector_leaderboard_eval.py \\
      --ckpt_path corrector_v5_step12000.ckpt \\
      --chat_nemo chat_banded1_nodelay_v2.nemo \\
      --pretrained_llm ./Qwen3-1.7B \\
      --pretrained_asr ./nemotron-speech-streaming-en-0.6b.nemo \\
      --cache_dir ~/leaderboard_cache --datasets ami_cleaned:test
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# The REPO ROOT too: "python scripts/x.py" puts scripts/ on sys.path but not
# the root, so "import nemo" silently resolves to the INSTALLED package,
# which has no script_corrector_model. Prepending keeps the checkout
# authoritative -- the model being evaluated must be this working tree's.
sys.path.insert(0, os.path.dirname(_HERE))

from leaderboard_common import (  # noqa: E402
    _log,
    aggregate_results,
    build_global_items,
    load_audio_batch,
    select_shard,
)


def load_model(args, device, dtype):
    """Build the corrector and load the trained weights.

    The checkpoint deliberately omits every ``chat.`` tensor -- the frozen CHAT
    is not the corrector's to save -- so the .nemo must be supplied separately
    and the load is necessarily non-strict. A checkpoint that matches NOTHING is
    fatal rather than silent: it would decode as a randomly-initialised model and
    simply report a bad WER.
    """
    from omegaconf import OmegaConf

    from nemo.collections.speechlm2.models.script_corrector_model import ScriptCorrectorModel

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("hyper_parameters", {}).get("cfg")
    if cfg is None:
        raise ValueError(f"no model cfg inside {args.ckpt_path}; cannot rebuild the model")
    cfg = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True) if not isinstance(cfg, dict) else dict(cfg)
    cfg["pretrained_llm"] = args.pretrained_llm
    cfg["pretrained_asr"] = args.pretrained_asr
    cfg["chat_nemo"] = args.chat_nemo

    model = ScriptCorrectorModel(cfg)
    sd = ckpt.get("state_dict", ckpt)
    tgt = model.state_dict()
    keep = {k: v for k, v in sd.items() if k in tgt and tgt[k].shape == v.shape}
    if not keep:
        raise ValueError(f"checkpoint matched ZERO parameters from {args.ckpt_path}")
    missing = [k for k in sd if k not in keep]
    model.load_state_dict(keep, strict=False)
    _log(f"loaded {len(keep)}/{len(tgt)} tensors ({len(missing)} unused from the ckpt)")

    model = model.to(device=device, dtype=dtype).eval()
    return model


@torch.no_grad()
def decode_batch(model, audios, audio_lens, args):
    """Free-running CHAT, then accept-or-REWRITE per chunk. Returns texts."""
    ids = model.ids
    dev = audios.device

    proc, proc_len = model.chat.preprocessor(input_signal=audios, length=audio_lens)
    enc, enc_len = model.chat.encoder(audio_signal=proc, length=proc_len)
    enc = enc.transpose(1, 2)
    cs = model.chat.joint.chunk_size
    n_chunks = [int((int(l) + cs - 1) // cs) for l in enc_len]
    B = len(n_chunks)

    # _chat_hypotheses reads only len() of this argument -- the hypotheses are
    # free-running and reference-free, which is what makes them usable here.
    hyp_i = model._chat_hypotheses(enc, enc_len, [None] * B, n_chunks)

    instr = model.tokenizer.text_to_ids(model.system_prompt + "\n")
    texts, stats = [], []
    for b in range(B):
        emitted, history = [], []
        n_acc = n_rej = n_would = 0
        for k in range(n_chunks[b]):
            hyp = hyp_i[b][k]
            alen = min(cs, max(0, int(enc_len[b]) - k * cs))
            from nemo.collections.speechlm2.parts.script_corrector import build_corrector_example

            ex = build_corrector_example(instr, history, alen, hyp, None, ids=ids)
            seq = list(ex.input_ids[: ex.prompt_len])
            emb = model._embed_tokens(torch.tensor([seq], dtype=torch.long, device=dev))
            if alen > 0:
                vs = seq.index(ids.vision_start)
                pos = torch.arange(vs + 1, vs + 1 + alen, device=dev)
                frm = torch.arange(k * cs, k * cs + alen, device=dev).clamp_(max=enc.shape[1] - 1)
                emb = emb.clone()
                emb[0, pos] = model.perception.proj(enc[b][frm].to(emb.dtype))

            cur = emb
            out_ids = []
            accepted = False
            would_accept = False
            if args.chat_only:
                # BASELINE on identical audio, chunking and scoring: emit CHAT's
                # chunk untouched. Without this the corrector's WER has nothing
                # to be better or worse THAN -- and since it accepts ~96% of
                # chunks, most of its output is CHAT's anyway.
                emitted += list(hyp)
                history = history + list(hyp)
                n_acc += 1
                continue
            for step in range(max(1, args.max_correction_tokens)):
                lg = model.llm(inputs_embeds=cur).logits[0, -1]
                if step == 0:
                    would_accept = int(lg.argmax()) == ids.accept
                    if args.force_reject:
                        # ABLATION: make it write the chunk whether it wanted to
                        # or not. Masking the token is the honest way to force
                        # this -- skipping the branch would leave ACCEPT as the
                        # argmax and emit the accept symbol AS the correction.
                        lg = lg.clone()
                        lg[ids.accept] = float("-inf")
                nxt = int(lg.argmax())
                if step == 0 and nxt == ids.accept:
                    accepted = True
                    break
                if nxt == ids.eot:
                    break
                out_ids.append(nxt)
                cur = torch.cat([cur, model._embed_tokens(torch.tensor([[nxt]], dtype=torch.long, device=dev))], dim=1)

            chunk_ids = list(hyp) if accepted else out_ids
            n_acc += accepted
            n_rej += not accepted
            n_would += would_accept
            emitted += chunk_ids
            # HISTORY IS WHAT WAS EMITTED -- not the reference. This is the only
            # honest choice at inference and it is NOT what training saw.
            history = history + chunk_ids

        # One detokenization of the whole utterance: per-chunk detokenization
        # splits words that straddle a boundary.
        texts.append(model.tokenizer.ids_to_text(emitted) if emitted else "")
        stats.append({"accept": n_acc, "reject": n_rej, "would_accept": n_would})
    return texts, stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--chat_nemo", required=True, help="the FROZEN CHAT; not stored in the corrector ckpt")
    p.add_argument("--pretrained_llm", required=True)
    p.add_argument("--pretrained_asr", required=True)
    p.add_argument("--cache_dir", default=os.path.expanduser("~/leaderboard_cache"))
    p.add_argument("--datasets", default="ami_cleaned:test")
    p.add_argument("--output_dir", default="./corrector_eval_out")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_eval_samples", type=int, default=0, help="cap per dataset; 0 = all")
    p.add_argument("--max_correction_tokens", type=int, default=48, help="cap per rejected chunk")
    p.add_argument("--pad_extra_seconds", type=float, default=0.0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--shuffle_seed", type=int, default=0)
    p.add_argument("--progress_interval", type=float, default=5.0)
    # Consumed by leaderboard_common.aggregate_results, not by this script.
    p.add_argument("--verbose", action="store_true", help="per-utterance scoring detail")
    p.add_argument("--chat_only", action="store_true", help="baseline: emit CHAT, never correct")
    p.add_argument(
        "--force_reject",
        action="store_true",
        help="ABLATION: never accept; make the model rewrite EVERY chunk. Isolates "
        "generation quality from decision quality -- a good WER here with a bad one "
        "in normal mode means the decision is the bottleneck, not the rewriting.",
    )
    args = p.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = load_model(args, device, dtype)

    items = build_global_items(args)
    shard = select_shard(items, args.num_shards, args.shard_index, args.shuffle_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"shard{args.shard_index}_of{args.num_shards}.generations.jsonl")
    _log(f"==> {len(shard)} utterances -> {out_path}")

    done, total = defaultdict(int), defaultdict(int)
    for it in shard:
        total[it["key"]] += 1
    acc_tot = rej_tot = would_tot = 0
    t0 = time.time()
    with open(out_path, "w") as fout:
        bar = tqdm(range(0, len(shard), args.batch_size), mininterval=args.progress_interval)
        for i in bar:
            batch = shard[i : i + args.batch_size]
            try:
                audios, alens = load_audio_batch([b["path"] for b in batch], args.pad_extra_seconds)
                hyps, stats = decode_batch(model, audios.to(device), alens.to(device), args)
            except Exception as e:  # one bad batch must not lose the run
                _log(f"    [WARN] batch at {i} failed ({type(e).__name__}: {e}); emitting empty hypotheses")
                hyps = [""] * len(batch)
                stats = [{"accept": 0, "reject": 0}] * len(batch)
            for b, hyp, stt in zip(batch, hyps, stats):
                fout.write(json.dumps({"key": b["key"], "reference": b["ref"], "hypothesis": hyp}) + "\n")
                done[b["key"]] += 1
                acc_tot += stt["accept"]
                rej_tot += stt["reject"]
                would_tot += stt.get("would_accept", 0)
            fout.flush()
            bar.set_postfix_str(f"acc={acc_tot} rej={rej_tot}")

    n = acc_tot + rej_tot
    if n:
        _log(
            f"==> {time.time() - t0:.1f}s | chunks: {n} | accept_frac={acc_tot / n:.4f} "
            f"| would_accept_frac={would_tot / n:.4f}"
        )
    else:
        _log("==> no chunks")
    aggregate_results(args)


if __name__ == "__main__":
    main()
