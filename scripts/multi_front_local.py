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
"""Run multi-front (self-correcting) decode on a few utterances, locally.

Two fronts from ONE prompt-controlled checkpoint: a fast one that paints text
immediately and a slow one that replaces it a block later. Prints the serialised
``<temp_start>…<temp_end>…`` string per utterance plus what a reader actually
experiences -- what was shown first, what they were left with, and how much moved
under them.

    python scripts/multi_front_local.py \
        --ckpt /home/hainanx/Workplace/ckpts/granary2_script_promptctl-averaged.ckpt \
        --n 5 --fast 2 --slow 14

The checkpoint must list BOTH chunk sizes in model.chunk_size; the script checks
and refuses otherwise, since a front at an untrained chunk size silently decodes
at an operating point the model never saw.
"""

import argparse
import os
import sys
import time

# Run the REPO's nemo, not whatever is installed in site-packages: this script
# exercises code that may not be installed yet.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=5, help="utterances to decode")
    ap.add_argument("--index", type=int, default=0, help="index of the first utterance to decode")
    ap.add_argument("--fast", type=int, default=2, help="provisional front chunk size")
    ap.add_argument("--slow", type=int, default=14, help="authoritative front chunk size")
    ap.add_argument("--no_resync", action="store_true", help="let the fast front keep its own history")
    ap.add_argument(
        "--print",
        dest="print_mode",
        default="plain",
        choices=["plain", "stream", "dual"],
        help="plain: serialised string + stats; stream: one stream, temporary tail erased "
        "and rewritten; dual: fast front on one line, slow front beneath it",
    )
    ap.add_argument(
        "--realtime",
        action="store_true",
        help="with --print stream, pace playback at the real block duration instead of dumping instantly",
    )
    ap.add_argument("--dataset", default="librispeech")
    ap.add_argument("--split", default="test.clean")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model_class", default="nemo.collections.speechlm2.models.script_model.ScriptSTTModel")
    ap.add_argument("--norm_root", default="/home/hainanx/Workplace/open_asr_leaderboard")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from scripts.script_leaderboard_eval import load_model

    from nemo.collections.speechlm2.parts.multi_front import (
        final_text,
        multi_front_decode,
        parse,
        provisional_text,
        render_dual,
        render_stream,
        revision_stats,
    )

    dev = torch.device(args.device)
    model = load_model(args.ckpt, args.model_class, dev, torch.bfloat16, prompt_control=True)
    model.eval()

    trained = list(getattr(model.core_cfg, "chunk_size", []) or [])
    if isinstance(trained, int):
        trained = [trained]
    missing = [c for c in (args.fast, args.slow) if trained and c not in trained]
    if missing:
        # A front at an untrained chunk size still produces fluent text, so this
        # would look like it worked while measuring an operating point the model
        # was never taught.
        raise SystemExit(f"checkpoint was trained on chunk sizes {trained}; fronts {missing} are not among them")

    ds = load_dataset("hf-audio/open-asr-leaderboard", args.dataset, split=args.split)
    lo = max(0, args.index)
    hi = min(lo + args.n, len(ds))
    if lo >= len(ds):
        raise SystemExit(f"--index {args.index} is past the end of {args.dataset}/{args.split} ({len(ds)} utts)")
    ds = ds.select(range(lo, hi))

    prompt_txt = (
        "You are doing streaming speech recognition. Given the transcript so far and the "
        "representation of the next audio chunk, output the words spoken in that chunk."
    )

    sys.path.insert(0, args.norm_root)
    from kaldialign import edit_distance
    from normalizer.normalizer import EnglishTextNormalizer

    norm = EnglishTextNormalizer()

    def wer(ref, hyp):
        r, h = norm(ref).split(), norm(hyp).split()
        if not r:
            return None, 0
        return edit_distance(r, h)["total"], len(r)

    frame_s = 0.08  # one encoder frame
    tot = {"prov_e": 0, "fin_e": 0, "w": 0, "blocks": 0, "corr": 0, "pw": 0, "rw": 0}

    for i in range(len(ds)):
        row = ds[i]
        audio = torch.tensor(row["audio"]["array"], dtype=torch.float32)
        ref = row.get("text") or row.get("sentence") or ""
        auds = audio.unsqueeze(0).to(dev)
        lens = torch.tensor([audio.shape[0]], device=dev)

        t0 = time.time()
        with torch.inference_mode():
            s = multi_front_decode(
                model,
                auds,
                lens,
                system_prompt=prompt_txt,
                fast_chunk=args.fast,
                slow_chunk=args.slow,
                resync=not args.no_resync,
            )[0]
        dt = time.time() - t0

        blocks = parse(s)
        prov, fin = provisional_text(s), final_text(s)
        st = revision_stats(s)
        pe, w = wer(ref, prov)
        fe, _ = wer(ref, fin)

        print(
            f"\n{'='*100}\nutt {lo + i}   {len(blocks)} blocks x {args.slow} frames "
            f"({args.slow*frame_s:.2f}s each)   decode {dt:.1f}s"
        )
        print(f"  REF         {norm(ref)}")
        if args.print_mode == "stream":
            print("  ONE STREAM (temporary text is erased and rewritten as each block closes):")
            print("  ", end="")
            render_stream(s, delay=0.0, block_seconds=args.slow * frame_s if args.realtime else 0.0)
        elif args.print_mode == "dual":
            print("  TWO STREAMS (fast front above, slow front below; ^ marks what changed):")
            print(render_dual(s, width=110))
        else:
            print(f"  shown first {norm(prov)}")
            print(f"  left with   {norm(fin)}")
            print(f"  serialised  {s}")
        if w:
            print(
                f"  WER  provisional {100*pe/w:5.1f}   final {100*fe/w:5.1f}   "
                f"(corrections: {st.corrected_blocks}/{st.blocks} blocks, "
                f"{100*st.word_revision_rate:.0f}% of shown words revised)"
            )
            tot["prov_e"] += pe
            tot["fin_e"] += fe
            tot["w"] += w
        tot["blocks"] += st.blocks
        tot["corr"] += st.corrected_blocks
        tot["pw"] += st.provisional_words
        tot["rw"] += st.revised_words

        for j, b in enumerate(blocks):
            if b.corrected:
                print(f"    block {j} [{j*args.slow*frame_s:5.2f}s] " f"{b.provisional!r} -> {b.final!r}")

    print(
        f"\n{'='*100}\nTOTALS over {len(ds)} utterances "
        f"(fronts {args.fast}/{args.slow}, resync={not args.no_resync})"
    )
    if tot["w"]:
        print(
            f"  WER shown-first {100*tot['prov_e']/tot['w']:.2f}   "
            f"WER left-with {100*tot['fin_e']/tot['w']:.2f}   "
            f"(the gap is what the second front buys)"
        )
    if tot["blocks"]:
        print(f"  blocks corrected {tot['corr']}/{tot['blocks']} ({100*tot['corr']/tot['blocks']:.0f}%)")
    if tot["pw"]:
        print(f"  words revised    {tot['rw']}/{tot['pw']} ({100*tot['rw']/tot['pw']:.0f}% of what was shown moved)")
    print(f"  latency: first paint {args.fast*frame_s:.2f}s, settles at {args.slow*frame_s:.2f}s")


if __name__ == "__main__":
    main()
