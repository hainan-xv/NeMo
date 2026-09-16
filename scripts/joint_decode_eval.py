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

from nemo.collections.speechlm2.parts.joint_decode import chunk_sync_joint_decode  # noqa: E402
from nemo.collections.speechlm2.parts.joint_decode_adapters import (  # noqa: E402
    ChatChunkScorer,
    ScriptChunkScorer,
)

DEFAULT_PROMPT = (
    "You are doing streaming speech recognition. Given the transcript so far and the "
    "representation of the next audio chunk, output the words spoken in that chunk."
)


def load_chat(path: str, device: torch.device):
    from nemo.collections.asr.models.chat_bpe_models import EncDecCHATBPEModel

    m = EncDecCHATBPEModel.restore_from(restore_path=path, map_location=device)
    m.eval().to(device)
    return m


def load_script(ckpt: str, device: torch.device, dtype: torch.dtype):
    from script_leaderboard_eval import load_model

    m = load_model(ckpt, "nemo.collections.speechlm2.models.script_model.ScriptSTTModel", device, dtype)
    m.eval()
    return m


@torch.no_grad()
def decode_one(chat_m, script_m, wav, wav_len, chunk_size, lams, prompt_ids, device):
    """Decode ONE utterance at every lam, reusing a single pair of encodes."""
    # --- CHAT side: raw encoder output; joint_on_path chunks it itself ---
    proc, proc_len = chat_m.preprocessor(input_signal=wav, length=wav_len)
    enc, enc_len = chat_m.encoder(audio_signal=proc, length=proc_len)
    enc = enc.transpose(1, 2)  # [B, T, D] -- joint_on_path expects raw [B, T, D]

    # --- SCRIPT side: per-utterance encoder frames on the same grid ---
    frames = script_m.encode_frames(wav, wav_len, chunk_size)[0]

    # Both models must agree on the chunk count, or they are not indexing the
    # same audio. CHAT's count is authoritative: it owns the emission budget.
    n_chunks = int((enc_len[0].item() + chunk_size - 1) // chunk_size)
    n_chunks_script = int((frames.shape[0] + chunk_size - 1) // chunk_size)
    if n_chunks != n_chunks_script:
        n_chunks = min(n_chunks, n_chunks_script)

    chat_sc = ChatChunkScorer(chat_m, enc, enc_len)
    script_sc = ScriptChunkScorer(script_m, frames, prompt_ids, chunk_size, chat_sc.vocab_size)

    out = {}
    for lam in lams:
        ids = chunk_sync_joint_decode(chat_sc, script_sc, n_chunks, chat_sc.vocab_size, lam=lam)
        out[lam] = chat_m.tokenizer.ids_to_text(ids) if ids else ""
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat_nemo", required=True)
    ap.add_argument("--script_ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--datasets", default="librispeech:test.clean")
    ap.add_argument("--max_eval_samples", type=int, default=200)
    ap.add_argument("--lam", default="0.0,0.25,0.5,0.75,1.0", help="comma-separated CHAT weights")
    ap.add_argument("--chunk_size", type=int, default=14)
    ap.add_argument("--pad_extra_seconds", type=float, default=0.5)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--system_prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    lams = [float(x) for x in args.lam.split(",") if x.strip()]
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"==> loading CHAT   {args.chat_nemo}", flush=True)
    chat_m = load_chat(args.chat_nemo, device)
    print(f"==> loading SCRIPT {args.script_ckpt}", flush=True)
    script_m = load_script(args.script_ckpt, device, dtype)

    prompt_ids = script_m.tokenizer.text_to_ids(args.system_prompt + "\n")

    for ds, split in parse_entries(args.datasets):
        paths, refs, _ = read_cache_manifest(args.cache_dir, ds, split, args.max_eval_samples)
        print(f"\n==> {ds}/{split}: {len(paths)} utts, lam={lams}", flush=True)

        hyps = {lam: [] for lam in lams}
        t0 = time.time()
        for i, p in enumerate(paths):
            wav, wav_len = load_audio_batch([p], args.pad_extra_seconds)
            wav, wav_len = wav.to(device), wav_len.to(device)
            got = decode_one(chat_m, script_m, wav, wav_len, args.chunk_size, lams, prompt_ids, device)
            for lam in lams:
                hyps[lam].append(got[lam])
            if args.verbose and i < 3:
                print(f"   ref: {refs[i]}")
                for lam in lams:
                    print(f"   lam={lam}: {got[lam]}")
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"   {i+1}/{len(paths)}  {el/(i+1):.2f}s/utt", flush=True)

        print(f"\n  {'lam':>6}  {'WER%':>7}")
        print(f"  {'-'*16}")
        for lam in lams:
            # A FRESH scorer per lam: LeaderboardWER accumulates across update()
            # calls, so reusing one would pool every lam's hypotheses together
            # and report the same blended number for all of them.
            sc = LeaderboardWER()
            sc.update(f"{ds}/{split}", refs, hyps[lam])
            wer = sc.compute()["wer"] * 100.0
            tag = "  <- CHAT alone" if lam == 1.0 else ("  <- SCRIPT alone" if lam == 0.0 else "")
            print(f"  {lam:>6}  {wer:>7.2f}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
