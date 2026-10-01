#!/usr/bin/env python3
"""Emission latency from eval shard JSONLs: when was each word produced?

Needs runs made with --emit_chunk_ids (EMIT_CHUNK_IDS=1), which adds
``"chunks": [[chunk_index, text], ...]`` per utterance. Without it the JSONL holds
only the final concatenated hypothesis and latency cannot be recovered.

Reports, per (run, dataset):

  avg_chunk   mean chunk index over all emitted words, weighted by words.
              This is what was asked for, but it is NOT comparable across chunk
              sizes: a 2-frame chunk fits ~7x more chunks into the same audio, so
              its indices are inflated by construction.

  avg_sec     the same quantity in SECONDS. Chunk k spans frames [k*w, (k+1)*w),
  so a word emitted during chunk k is available only at (k+1)*w*frame_sec.
              which IS comparable across chunk sizes, and is the number that
              corresponds to how long a listener waits.

Both are emission TIME, not delay relative to when the word was spoken; that would
need per-word alignments on the reference.

Usage:
  python scripts/emission_latency.py --run DIR [DIR ...]
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict


def chunk_size_of(run_dir: str) -> int:
    m = re.search(r"chunk(\d+)", os.path.basename(run_dir.rstrip("/")))
    return int(m.group(1)) if m else 0


def scan(run_dir: str, frame_sec: float):
    cs = chunk_size_of(run_dir)
    per = defaultdict(lambda: {"words": 0, "chunkw": 0, "utts": 0, "nochunks": 0})
    shards = os.path.join(run_dir, "shards")
    files = [os.path.join(shards, f) for f in sorted(os.listdir(shards))] if os.path.isdir(shards) else []
    for fp in files:
        if not fp.endswith(".jsonl"):
            continue
        with open(fp) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                d = per[r.get("key", "?")]
                d["utts"] += 1
                ch = r.get("chunks")
                if not ch:
                    d["nochunks"] += 1
                    continue
                for idx, text in ch:
                    n = len(str(text).split())
                    if n:
                        d["words"] += n
                        d["chunkw"] += int(idx) * n
    return cs, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="eval result dirs (the ones holding shards/)")
    ap.add_argument("--frame_sec", type=float, default=0.08, help="encoder frame length")
    args = ap.parse_args()

    print(f"{'run':<46}{'chunk':>6}{'dataset':>28}{'words':>10}{'avg_chunk':>11}{'avg_sec':>9}")
    print("-" * 110)
    for run in args.run:
        cs, per = scan(run, args.frame_sec)
        tot_w = tot_cw = 0
        for key in sorted(per):
            d = per[key]
            if not d["words"]:
                print(f"{os.path.basename(run)[:45]:<46}{cs:>6}{key:>28}{0:>10}{'-':>11}{'-':>9}"
                      f"   ({d['nochunks']}/{d['utts']} utts lack 'chunks' -- rerun with EMIT_CHUNK_IDS=1)")
                continue
            ac = d["chunkw"] / d["words"]
            tot_w += d["words"]; tot_cw += d["chunkw"]
            print(f"{os.path.basename(run)[:45]:<46}{cs:>6}{key:>28}{d['words']:>10}{ac:>11.2f}{(ac+1)*cs*args.frame_sec:>9.2f}")
        if tot_w:
            ac = tot_cw / tot_w
            print(f"{os.path.basename(run)[:45]:<46}{cs:>6}{'ALL':>28}{tot_w:>10}{ac:>11.2f}{(ac+1)*cs*args.frame_sec:>9.2f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
