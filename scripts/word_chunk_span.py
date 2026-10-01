#!/usr/bin/env python3
"""How many chunks does a single word's subword tokens span?

``--emit_chunk_ids`` stores, per utterance, ``[[chunk_id, text], ...]`` where each
text is that chunk's emitted tokens decoded on their own. Qwen's byte-level BPE
keeps the word-start marker as a literal leading space, so concatenating the
groups in chunk order reproduces the hypothesis exactly and a word continues
across a chunk boundary iff the next chunk's text does not begin with a space.
That lets us attribute every character -- and so every word -- to the chunk(s)
that emitted it, without needing the raw token ids.

Restricted to words of >=2 subwords, since a 1-subword word can never span.
"""
import argparse
import glob
import json
import os
from collections import Counter

_SUB = {}


def _nsub(word, tok):
    """Subword count for a word, memoised -- the unique vocabulary is tiny next to
    the ~1.6M word tokens per run, so this turns 1.6M encodes into ~50k."""
    n = _SUB.get(word)
    if n is None:
        n = _SUB[word] = len(tok(" " + word, add_special_tokens=False)["input_ids"])
    return n


def spans(rec, tok):
    """Yield (n_subwords, n_chunks_spanned) for each word of one utterance."""
    chunks = rec.get("chunks") or []
    text, owner = "", []  # owner[i] = chunk id that emitted character i
    for k, t in chunks:
        text += t
        owner.extend([int(k)] * len(t))
    out = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        j = i
        while j < n and not text[j].isspace():
            j += 1
        if j > i:
            word = text[i:j]
            ks = owner[i:j]
            nsub = _nsub(word, tok)
            out.append((nsub, len(set(ks))))
        i = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="result dirs with shard jsonls")
    ap.add_argument("--tokenizer", default="/home/hainanx/corrector_local/Qwen3-1.7B")
    ap.add_argument("--min_subwords", type=int, default=2)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    for run in args.run:
        hist, allw, multi, no_chunks = Counter(), 0, 0, 0
        for f in sorted(glob.glob(os.path.join(run, "**", "*.jsonl"), recursive=True)):
            for line in open(f):
                rec = json.loads(line)
                if not rec.get("chunks"):
                    no_chunks += 1
                    continue
                for nsub, nk in spans(rec, tok):
                    allw += 1
                    if nsub >= args.min_subwords:
                        multi += 1
                        hist[nk] += 1
        print(f"=== {os.path.basename(run.rstrip('/'))}")
        if no_chunks:
            print(f"    (skipped {no_chunks} records with no chunks field)")
        if not multi:
            print("    no multi-subword words found\n")
            continue
        print(f"    words total {allw}, with >={args.min_subwords} subwords: {multi} ({100*multi/max(allw,1):.1f}%)")
        for nk in sorted(hist):
            print(f"      spans {nk} chunk(s): {hist[nk]:>9}  {100*hist[nk]/multi:6.2f}%")
        print()


if __name__ == "__main__":
    main()
