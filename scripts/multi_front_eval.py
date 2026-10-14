#!/usr/bin/env python3
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
"""Multi-front vs single-chunk decode, on a full dataset, locally.

Scores four systems on the SAME utterances in ONE run, so nothing is compared
across pipelines or utterance subsets:

    single-2    plain decode at the fast chunk size
    single-14   plain decode at the slow chunk size
    mf-first    multi-front, what the reader sees FIRST (provisional)
    mf-final    multi-front, what the reader is LEFT WITH (authoritative)

The two comparisons that matter:
  * mf-first vs single-2  -- does resyncing the fast front to corrected text
    actually make the first paint better than a plain fast decode?
  * mf-final vs single-14 -- does the authoritative text match what a plain slow
    decode would have produced, i.e. does the reader give up nothing by having
    seen provisional text along the way?

Writes a JSONL trace (one row per utterance, including the serialised
<temp_start>...<temp_end> string) so the run can be re-analysed without decoding
again, and prints running WER so a long run can be watched.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="ami")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0, help="0 = whole split")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fast", type=int, default=2)
    ap.add_argument("--slow", type=int, default=14)
    ap.add_argument("--no_resync", action="store_true")
    ap.add_argument("--max_seconds", type=float, default=0.0, help="skip utterances longer than this")
    ap.add_argument("--out", default="", help="JSONL trace path")
    ap.add_argument("--report_every", type=int, default=25, help="batches between progress lines")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model_class", default="nemo.collections.speechlm2.models.script_model.ScriptSTTModel")
    ap.add_argument("--norm_root", default="/home/hainanx/Workplace/open_asr_leaderboard")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset

    from nemo.collections.speechlm2.parts.multi_front import (
        final_text,
        multi_front_decode,
        provisional_text,
        revision_stats,
    )
    from scripts.script_leaderboard_eval import load_model

    dev = torch.device(args.device)
    model = load_model(args.ckpt, args.model_class, dev, torch.bfloat16, prompt_control=True)
    model.eval()

    ds = load_dataset("hf-audio/open-asr-leaderboard", args.dataset, split=args.split)
    lo = args.index
    hi = len(ds) if not args.n else min(lo + args.n, len(ds))
    ds = ds.select(range(lo, hi))

    prompt_txt = (
        "You are doing streaming speech recognition. Given the transcript so far and the "
        "representation of the next audio chunk, output the words spoken in that chunk."
    )
    sys.path.insert(0, args.norm_root)
    from kaldialign import edit_distance
    from normalizer.normalizer import EnglishTextNormalizer

    norm = EnglishTextNormalizer()
    SYS = ["single_fast", "single_slow", "mf_first", "mf_first_raw", "mf_final"]
    err = {k: 0 for k in SYS}
    words = 0
    rev = {"blocks": 0, "corrected": 0, "prov_words": 0, "revised": 0}
    skipped = 0
    audio_s = 0.0

    out_f = open(args.out, "w") if args.out else None
    t0 = time.time()
    nb = 0
    for i in range(0, len(ds), args.batch):
        rows = ds[i : i + args.batch]
        auds, refs = [], []
        for a, t in zip(rows["audio"], rows.get("text", [""] * len(rows["audio"]))):
            dur = len(a["array"]) / a["sampling_rate"]
            if args.max_seconds and dur > args.max_seconds:
                skipped += 1
                continue
            auds.append(torch.tensor(a["array"], dtype=torch.float32))
            refs.append(t)
            audio_s += dur
        if not auds:
            continue
        n = len(auds)
        L = max(a.shape[0] for a in auds)
        batch_audio = torch.zeros(n, L)
        for j, a in enumerate(auds):
            batch_audio[j, : a.shape[0]] = a
        batch_audio = batch_audio.to(dev)
        lens = torch.tensor([a.shape[0] for a in auds], device=dev)

        with torch.inference_mode():
            s_fast = model.generate(batch_audio, lens, system_prompt=prompt_txt, chunk_size_override=args.fast)
            s_slow = model.generate(batch_audio, lens, system_prompt=prompt_txt, chunk_size_override=args.slow)
            mf = multi_front_decode(
                model,
                batch_audio,
                lens,
                system_prompt=prompt_txt,
                fast_chunk=args.fast,
                slow_chunk=args.slow,
                resync=not args.no_resync,
            )

        for j in range(n):
            ref = norm(refs[j]).split()
            if not ref:
                continue
            words += len(ref)
            hyp = {
                "single_fast": s_fast[j],
                "single_slow": s_slow[j],
                "mf_first": provisional_text(mf[j]),
                # raw concatenation too: it double-counts deferred words, and
                # keeping both makes that visible instead of hidden in a choice
                "mf_first_raw": provisional_text(mf[j], dedupe_boundary=False),
                "mf_final": final_text(mf[j]),
            }
            for k, v in hyp.items():
                err[k] += edit_distance(ref, norm(v).split())["total"]
            st = revision_stats(mf[j])
            rev["blocks"] += st.blocks
            rev["corrected"] += st.corrected_blocks
            rev["prov_words"] += st.provisional_words
            rev["revised"] += st.revised_words
            if out_f:
                out_f.write(json.dumps({"i": lo + i + j, "ref": refs[j], "serialized": mf[j], **hyp}) + "\n")
        if out_f:
            out_f.flush()

        nb += 1
        if nb % args.report_every == 0 and words:
            el = time.time() - t0
            done = i + args.batch
            rate = audio_s / el
            eta = (len(ds) - done) / max(done, 1) * el
            print(
                f"[{done}/{len(ds)}] "
                + "  ".join(f"{k} {100*err[k]/words:5.2f}" for k in SYS)
                + f"   {rate:.1f}x realtime   eta {eta/3600:.1f}h",
                flush=True,
            )

    if out_f:
        out_f.close()
    el = time.time() - t0
    print(f"\n{'='*92}")
    print(
        f"{args.dataset}/{args.split}  {len(ds)} utts  ({audio_s/3600:.1f}h audio, {skipped} skipped)  "
        f"fronts {args.fast}/{args.slow}  resync={not args.no_resync}"
    )
    print(f"decoded in {el/3600:.2f}h  ({audio_s/el:.1f}x realtime)\n")
    if not words:
        print("no scorable utterances")
        return
    print(f"  {'system':<12}{'WER':>8}   what it is")
    print(f"  {'-'*58}")
    print(f"  {'single-2':<12}{100*err['single_fast']/words:>8.2f}   plain fast decode")
    print(f"  {'mf-first':<12}{100*err['mf_first']/words:>8.2f}   multi-front, shown FIRST (deferrals deduped)")
    print(
        f"  {'mf-first-raw':<12}{100*err['mf_first_raw']/words:>8.2f}   same, raw concatenation (double-counts deferrals)"
    )
    print(f"  {'single-14':<12}{100*err['single_slow']/words:>8.2f}   plain slow decode")
    print(f"  {'mf-final':<12}{100*err['mf_final']/words:>8.2f}   multi-front, left WITH")
    print()
    d1 = 100 * (err["mf_first"] - err["single_fast"]) / words
    d2 = 100 * (err["mf_final"] - err["single_slow"]) / words
    print(f"  resync effect on the first paint : {d1:+.2f} WER  (mf-first vs single-2)")
    print(f"  cost of having shown provisionals: {d2:+.2f} WER  (mf-final vs single-14)")
    if rev["blocks"]:
        print(
            f"  blocks corrected {rev['corrected']}/{rev['blocks']} "
            f"({100*rev['corrected']/rev['blocks']:.1f}%), "
            f"words revised {rev['revised']}/{rev['prov_words']} "
            f"({100*rev['revised']/max(rev['prov_words'],1):.1f}%)"
        )


if __name__ == "__main__":
    main()
