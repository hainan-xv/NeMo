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
"""Chunk-synchronous joint decoding of a CHAT and a SCRIPT model, scored on the
leaderboard cache.

    python scripts/joint_decode_eval.py \
        --chat_nemo  .../dfw_granary2_chat_banded1_nodelay_v2/averaged/top5-averaged.nemo \
        --script_ckpt .../dfw_granary2_script_banded1_nodelay_v2-averaged.ckpt \
        --cache_dir  .../leaderboard_cache \
        --datasets librispeech:test.clean --max_eval_samples 200 \
        --lam 0.0,0.25,0.5,0.75,1.0

Sweeps ``lam`` in ONE pass over the audio. Both models are encoded once per
utterance and only the fusion is repeated, so an N-point sweep costs barely more
than a single decode -- which matters because the joint decode is slow (see
PERFORMANCE below).

THE TWO ENDPOINTS ARE THE CONTROLS, and should always be included in a sweep:
``lam=1`` is CHAT decoding alone and ``lam=0`` is SCRIPT alone, through this
exact code path. If lam=1 does not land near the CHAT arm's own leaderboard
number, the harness is wrong and nothing in between means anything. That check
is cheap and catches the failure modes -- wrong chunk count, wrong instruction
string, mismatched padding -- that would otherwise look like "fusion doesn't
help".

PADDING IS NOT COSMETIC. ``--pad_extra_seconds`` (default 0.5, matching the
leaderboard evals) appends real trailing silence. For a CHAT model that changes
the CHUNK COUNT, which is its entire emission budget -- on AMI it moved plain
CHAT from 0.1316 to 0.1021 WER. Both models must see the SAME padded audio or
their chunk grids do not line up and the premise of joint decoding fails.

PERFORMANCE. This is a research driver, not the production eval. CHAT's
prediction network is recomputed from the full prefix at every step and SCRIPT
re-runs its per-chunk prompt for every token, with no KV reuse. Expect it to be
far slower per utterance than either model alone; run it on a few hundred
utterances, not the full 74,842.
"""

import argparse
import os
import sys
import time
from typing import List

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from leaderboard_common import load_audio_batch, parse_entries, read_cache_manifest  # noqa: E402
from leaderboard_wer import LeaderboardWER  # noqa: E402
from script_leaderboard_eval import _hubify  # noqa: E402

from nemo.collections.speechlm2.parts.joint_decode import chunk_sync_joint_decode  # noqa: E402
from nemo.collections.speechlm2.parts.joint_decode_adapters import (  # noqa: E402
    ChatChunkScorer,
    ScriptChunkScorer,
)

DEFAULT_PROMPT = (
    "You are doing streaming speech recognition. Given the transcript so far and the "
    "representation of the next audio chunk, output the words spoken in that chunk."
)


def load_chat(path: str, device: torch.device, tokenizer_dir: str = ""):
    """Restore a CHAT .nemo, repointing its tokenizer at something that exists here.

    A .nemo records the ABSOLUTE tokenizer path of the machine that trained it.
    Off that filesystem the directory is missing, and transformers then reads the
    path as a hub repo id and fails with "Repo id must be in the form
    'repo_name'". Same failure the DFW evals hit from a missing container mount --
    it is a path problem wearing a hub-API error message.

    ``_hubify`` turns ``.../huggingface/Qwen/Qwen3-1.7B`` into ``Qwen/Qwen3-1.7B``,
    which resolves from the local HF cache. ``--chat_tokenizer_dir`` overrides it
    outright when the cache is not an option.
    """
    from omegaconf import OmegaConf, open_dict

    from nemo.collections.asr.models.chat_bpe_models import EncDecCHATBPEModel

    cfg = EncDecCHATBPEModel.restore_from(restore_path=path, return_config=True)
    old_dir = str(cfg.tokenizer.get("dir", "") or "")
    # Hubify ONLY when the recorded path is gone. On the cluster that trained
    # the model the lustre path is REAL and resolvable; rewriting it to a hub
    # id there breaks it, because the container runs with HF_HUB_OFFLINE and
    # has no hub cache -- which is exactly how both fusion runs of job
    # 18751303 died ("couldn't connect to huggingface.co"). Locally the path
    # is absent and the hub id is what works. So the test is existence, not
    # the machine.
    if tokenizer_dir:
        new_dir = tokenizer_dir
    elif old_dir and not os.path.isdir(old_dir):
        new_dir = _hubify(old_dir)
    else:
        new_dir = old_dir
    if new_dir != old_dir:
        with open_dict(cfg):
            cfg.tokenizer.dir = new_dir
        print(f"    tokenizer: {old_dir}\n            -> {new_dir}", flush=True)

    m = EncDecCHATBPEModel.restore_from(restore_path=path, map_location=device, override_config_path=cfg)
    m.eval().to(device)
    return m


def load_script(ckpt: str, device: torch.device, dtype: torch.dtype):
    from script_leaderboard_eval import load_model

    m = load_model(ckpt, "nemo.collections.speechlm2.models.script_model.ScriptSTTModel", device, dtype)
    m.eval()
    return m


@torch.no_grad()
def decode_batch(
    chat_m, script_m, wav, wav_len, chunk_size, lams, prompt, max_new_tokens, taus=(float("inf"),), stats=None
):
    """Decode a BATCH at every lam, through SCRIPT's production decode loop.

    The fusion lives inside ``batched_stream_decode_script``, so this inherits its
    batching and its per-chunk KV cache. The standalone loop this replaced was
    mathematically identical and took 390 s for a single 5-second clip because it
    had neither.

    CHAT is encoded ONCE and reused across the whole lam sweep; only the mixing
    weight changes, so an N-point sweep costs N decodes rather than N encodes
    plus N decodes.
    """
    from nemo.collections.speechlm2.parts.chat_fusion import ChatFusionScorer

    proc, proc_len = chat_m.preprocessor(input_signal=wav, length=wav_len)
    enc, enc_len = chat_m.encoder(audio_signal=proc, length=proc_len)
    enc = enc.transpose(1, 2)  # joint_on_path wants RAW [B, T, D]; it chunks itself
    scorer = ChatFusionScorer(chat_m, enc, enc_len)

    out = {}
    for lam in lams:
        for tau in taus:
            out[(lam, tau)] = script_m.generate(
                wav,
                wav_len,
                system_prompt=prompt,
                max_new_tokens=max_new_tokens,
                chunk_size_override=chunk_size,
                chat_fusion=None if lam == 0.0 else scorer,
                fusion_lam=lam,
                fusion_margin_threshold=tau,
                fusion_stats=stats.get((lam, tau)) if stats else None,
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat_nemo", required=True)
    ap.add_argument("--script_ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--datasets", default="librispeech:test.clean")
    ap.add_argument("--max_eval_samples", type=int, default=200)
    ap.add_argument(
        "--max_duration",
        type=float,
        default=0.0,
        help="skip clips longer than this many seconds; 0 = no filter. Decode cost is "
        "linear in chunk count, so one 35 s clip costs as much as ten 3.5 s ones -- "
        "for a smoke test that is all spent proving the same thing.",
    )
    ap.add_argument("--lam", default="0.0,0.25,0.5,0.75,1.0", help="comma-separated CHAT weights")
    ap.add_argument("--chunk_size", type=int, default=14)
    ap.add_argument("--pad_extra_seconds", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument(
        "--margin_threshold",
        default="inf",
        help="comma-separated CHAT-confidence thresholds (nats). A step whose CHAT top1-top2 "
        "margin is at least tau is handed to CHAT outright instead of fused. Monotonic: "
        "0 = CHAT alone, inf = full fusion. Sweeping tau shows WHERE the ensemble earns its gain.",
    )
    ap.add_argument("--fusion_stats", action="store_true", help="report override rate by CHAT confidence")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument(
        "--chat_tokenizer_dir",
        default="",
        help="override the tokenizer dir baked into the CHAT .nemo (default: map it to its hub id)",
    )
    ap.add_argument("--system_prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    lams = [float(x) for x in args.lam.split(",") if x.strip()]
    taus = [float(x) for x in args.margin_threshold.split(",") if x.strip()]
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"==> loading CHAT   {args.chat_nemo}", flush=True)
    chat_m = load_chat(args.chat_nemo, device, args.chat_tokenizer_dir)
    print(f"==> loading SCRIPT {args.script_ckpt}", flush=True)
    script_m = load_script(args.script_ckpt, device, dtype)

    prompt_ids = script_m.tokenizer.text_to_ids(args.system_prompt + "\n")

    for ds, split in parse_entries(args.datasets):
        paths, refs, durs = read_cache_manifest(args.cache_dir, ds, split, 0)
        if args.max_duration > 0:
            keep = [i for i, d in enumerate(durs) if float(d) <= args.max_duration]
            paths = [paths[i] for i in keep]
            refs = [refs[i] for i in keep]
            durs = [durs[i] for i in keep]
        if args.max_eval_samples > 0:
            paths, refs, durs = (
                paths[: args.max_eval_samples],
                refs[: args.max_eval_samples],
                durs[: args.max_eval_samples],
            )
        total_s = sum(float(d) for d in durs)
        print(
            f"\n==> {ds}/{split}: {len(paths)} utts, {total_s/60:.1f} min audio, lam={lams}",
            flush=True,
        )

        keys = [(l, t) for l in lams for t in taus]
        stats = None
        if args.fusion_stats:
            from nemo.collections.speechlm2.parts.chat_fusion import FusionStats

            stats = {k: FusionStats() for k in keys}
        hyps = {k: [] for k in keys}
        t0 = time.time()
        for st in range(0, len(paths), args.batch_size):
            chunk_paths = paths[st : st + args.batch_size]
            wav, wav_len = load_audio_batch(chunk_paths, args.pad_extra_seconds)
            wav, wav_len = wav.to(device), wav_len.to(device)
            got = decode_batch(
                chat_m,
                script_m,
                wav,
                wav_len,
                args.chunk_size,
                lams,
                args.system_prompt,
                args.max_new_tokens,
                taus,
                stats,
            )
            for k in keys:
                hyps[k].extend(got[k])
            done = min(st + args.batch_size, len(paths))
            el = time.time() - t0
            print(
                f"   [{done}/{len(paths)}] {el/done:5.2f}s/utt  eta {(len(paths)-done)*el/done/60:5.1f} min",
                flush=True,
            )
            if args.verbose and st == 0:
                for j in range(min(2, len(chunk_paths))):
                    print(f"   ref: {refs[j]}")
                    for k in keys:
                        print(f"   lam={k[0]} tau={k[1]}: {got[k][j]}")

        print(f"\n  {'lam':>6}  {'tau':>6}  {'WER%':>7}")
        print(f"  {'-'*24}")
        for lam, tau in keys:
            # A FRESH scorer per lam: LeaderboardWER accumulates across update()
            # calls, so reusing one would pool every lam's hypotheses together
            # and report the same blended number for all of them.
            sc = LeaderboardWER()
            sc.update(f"{ds}/{split}", refs, hyps[(lam, tau)])
            wer = sc.compute()["wer"] * 100.0
            tag = "  <- CHAT alone" if (lam == 1.0 or tau == 0.0) else ("  <- SCRIPT alone" if lam == 0.0 else "")
            print(f"  {lam:>6}  {tau:>6}  {wer:>7.2f}{tag}")
        if stats:
            for k in keys:
                if stats[k].steps:
                    print(f"\n  lam={k[0]} tau={k[1]}\n  " + stats[k].report().replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
