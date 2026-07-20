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

"""Per-condition ASR error-pattern analysis from leaderboard eval dumps.

Reads the per-utterance JSONL files written by
``scripts/asr_leaderboard_eval.py --dump_dir DIR`` (one ``<dataset>.jsonl`` per
ESB set = per acoustic condition) and, for EACH dataset separately, decomposes the
errors so the failure modes can be compared across conditions:

  * WER and its substitution / insertion / deletion breakdown (as % of ref words)
  * insertion-vs-deletion balance (over- vs under-generation; >100% WER == I-heavy)
  * hyp/ref length ratio, empty-hyp rate, exact-match rate
  * WER by reference-length bucket (short vs long utterances)
  * top substitution pairs (ref -> hyp confusions), top inserted / deleted words

Writes a machine-readable ``error_analysis.json`` (all datasets) for downstream
plotting / a canvas, and prints a compact per-dataset report.

Usage
-----
python scripts/analyze_asr_errors.py --dump_dir ~/leaderboard_run/dumps/<EXP>
python scripts/analyze_asr_errors.py --dump_dir DIR --field raw   # score PnC/raw text
python scripts/analyze_asr_errors.py --dumps a.jsonl,b.jsonl --topk 30
"""

import argparse
import glob
import json
import os
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dump_dir", help="Directory of <dataset>.jsonl dumps from the eval --dump_dir.")
    src.add_argument("--dumps", help="Comma-separated / glob list of dump JSONL files.")
    p.add_argument(
        "--field",
        choices=["norm", "raw"],
        default="norm",
        help="Which text to score: 'norm' (ref_norm/hyp_norm, matches leaderboard WER) or 'raw' (PnC).",
    )
    p.add_argument("--topk", type=int, default=20, help="How many top confusion/insert/delete items to report.")
    p.add_argument("--out", default=None, help="Path for the combined JSON report (default: <dump_dir>/error_analysis.json).")
    p.add_argument(
        "--length_buckets",
        default="1-5,6-10,11-20,21-40,41-",
        help="Reference-length (word count) buckets for WER-by-length, comma-separated 'lo-hi' ('41-' = 41+).",
    )
    return p.parse_args()


def align_words(ref, hyp):
    """Word-level Levenshtein alignment -> list of ('C'|'S'|'D'|'I', ref_word, hyp_word).

    Backtrace prefers match/substitution over delete over insert for a stable,
    conventional WER alignment.
    """
    R, H = len(ref), len(hyp)
    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
    dp = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(1, R + 1):
        dp[i][0] = i
    for j in range(1, H + 1):
        dp[0][j] = j
    for i in range(1, R + 1):
        ri = ref[i - 1]
        for j in range(1, H + 1):
            if ri == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    # Backtrace.
    ops = []
    i, j = R, H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("C", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def _bucketize(n, buckets):
    for lo, hi, label in buckets:
        if n >= lo and (hi is None or n <= hi):
            return label
    return buckets[-1][2]


def parse_buckets(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        lo = int(lo)
        hi = int(hi) if hi else None
        label = f"{lo}-{hi}" if hi is not None else f"{lo}+"
        out.append((lo, hi, label))
    return out


def analyze_file(path, field, topk, buckets):
    ref_key = "ref_norm" if field == "norm" else "ref"
    hyp_key = "hyp_norm" if field == "norm" else "hyp"

    n_utts = 0
    ref_words = 0
    hyp_words = 0
    C = S = D = I = 0
    empty_hyp = 0
    exact = 0
    unk_hyp = 0  # SentencePiece <unk> surface (U+2047) emitted by the model
    sub_pairs = Counter()
    ins_words = Counter()
    del_words = Counter()
    # WER-by-length: per bucket accumulate ref words and errors.
    bucket_ref = Counter()
    bucket_err = Counter()
    dataset_name = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            dataset_name = dataset_name or rec.get("dataset", os.path.basename(path))
            ref = (rec.get(ref_key) or "").split()
            hyp = (rec.get(hyp_key) or "").split()
            # The model's <unk> surface (⁇, U+2047) is not a real word (it marks a
            # char the tokenizer can't represent, e.g. quotes). Count it separately
            # and drop it so it doesn't masquerade as an insertion error.
            unk_hyp += hyp.count("\u2047")
            hyp = [w for w in hyp if w != "\u2047"]
            ref = [w for w in ref if w != "\u2047"]
            n_utts += 1
            ref_words += len(ref)
            hyp_words += len(hyp)
            if not hyp:
                empty_hyp += 1

            ops = align_words(ref, hyp)
            s = sum(1 for o in ops if o[0] == "S")
            d = sum(1 for o in ops if o[0] == "D")
            ins = sum(1 for o in ops if o[0] == "I")
            c = sum(1 for o in ops if o[0] == "C")
            C += c
            S += s
            D += d
            I += ins
            if s + d + ins == 0:
                exact += 1
            for op, rw, hw in ops:
                if op == "S":
                    sub_pairs[(rw, hw)] += 1
                elif op == "I":
                    ins_words[hw] += 1
                elif op == "D":
                    del_words[rw] += 1

            label = _bucketize(len(ref), buckets)
            bucket_ref[label] += len(ref)
            bucket_err[label] += s + d + ins

    denom = max(ref_words, 1)
    wer = (S + D + I) / denom
    result = {
        "dataset": dataset_name,
        "file": os.path.basename(path),
        "n_utts": n_utts,
        "ref_words": ref_words,
        "hyp_words": hyp_words,
        "hyp_ref_word_ratio": hyp_words / denom,
        "wer": wer,
        "sub": S,
        "del": D,
        "ins": I,
        "cor": C,
        "sub_rate": S / denom,
        "del_rate": D / denom,
        "ins_rate": I / denom,
        "ins_minus_del_rate": (I - D) / denom,  # >0 => over-generation (insertion-heavy)
        "empty_hyp": empty_hyp,
        "empty_hyp_rate": empty_hyp / max(n_utts, 1),
        "unk_hyp": unk_hyp,  # count of <unk>/⁇ tokens the model emitted (excluded from WER)
        "unk_per_100utt": 100.0 * unk_hyp / max(n_utts, 1),
        "exact_match": exact,
        "exact_match_rate": exact / max(n_utts, 1),
        "wer_by_length": {
            label: (bucket_err[label] / bucket_ref[label] if bucket_ref[label] else None)
            for (_, _, label) in buckets
        },
        "top_substitutions": [
            {"ref": r, "hyp": h, "count": c} for (r, h), c in sub_pairs.most_common(topk)
        ],
        "top_insertions": [{"word": w, "count": c} for w, c in ins_words.most_common(topk)],
        "top_deletions": [{"word": w, "count": c} for w, c in del_words.most_common(topk)],
    }
    return result


def print_report(res):
    d = res
    print(f"\n================ {d['dataset']} ================")
    print(f"utts={d['n_utts']}  ref_words={d['ref_words']}  hyp/ref word ratio={d['hyp_ref_word_ratio']:.3f}")
    print(f"WER={d['wer'] * 100:.2f}%   (S={d['sub_rate'] * 100:.2f}  D={d['del_rate'] * 100:.2f}  I={d['ins_rate'] * 100:.2f}  as % of ref words)")
    print(f"insertion-minus-deletion={d['ins_minus_del_rate'] * 100:+.2f}%  "
          f"(>0 => over-generation / insertion-heavy)")
    print(f"empty_hyp={d['empty_hyp']} ({d['empty_hyp_rate'] * 100:.2f}%)   "
          f"exact_match={d['exact_match']} ({d['exact_match_rate'] * 100:.2f}%)")
    print(f"<unk>/⁇ emitted={d['unk_hyp']} ({d['unk_per_100utt']:.2f} per 100 utts; excluded from WER)")
    wl = "  ".join(
        f"{lab}:{(v * 100):.1f}%" if v is not None else f"{lab}:-" for lab, v in d["wer_by_length"].items()
    )
    print(f"WER by ref length (words): {wl}")
    if d["top_substitutions"]:
        subs = ", ".join(f"{s['ref']}->{s['hyp']}({s['count']})" for s in d["top_substitutions"][:10])
        print(f"top subs: {subs}")
    if d["top_insertions"]:
        ins = ", ".join(f"{s['word']}({s['count']})" for s in d["top_insertions"][:10])
        print(f"top inserted: {ins}")
    if d["top_deletions"]:
        dels = ", ".join(f"{s['word']}({s['count']})" for s in d["top_deletions"][:10])
        print(f"top deleted: {dels}")


def main():
    args = parse_args()
    buckets = parse_buckets(args.length_buckets)

    if args.dump_dir:
        files = sorted(glob.glob(os.path.join(args.dump_dir, "*.jsonl")))
    else:
        files = []
        for part in args.dumps.split(","):
            part = part.strip()
            if part:
                files.extend(sorted(glob.glob(part)) if any(c in part for c in "*?[") else [part])
    if not files:
        raise SystemExit("No dump JSONL files found.")

    results = []
    for path in files:
        try:
            res = analyze_file(path, args.field, args.topk, buckets)
        except FileNotFoundError:
            print(f"WARNING: not found, skipping: {path}")
            continue
        results.append(res)
        print_report(res)

    # Cross-condition summary table.
    print("\n" + "=" * 78)
    print(f"{'dataset':<24}{'WER%':>7}{'S%':>7}{'D%':>7}{'I%':>7}{'I-D%':>8}{'empty%':>8}{'exact%':>8}")
    print("-" * 78)
    for d in results:
        print(
            f"{str(d['dataset']):<24}{d['wer']*100:>7.2f}{d['sub_rate']*100:>7.2f}{d['del_rate']*100:>7.2f}"
            f"{d['ins_rate']*100:>7.2f}{d['ins_minus_del_rate']*100:>+8.2f}"
            f"{d['empty_hyp_rate']*100:>8.2f}{d['exact_match_rate']*100:>8.2f}"
        )
    print("=" * 78)

    out = args.out or (os.path.join(args.dump_dir, "error_analysis.json") if args.dump_dir else "error_analysis.json")
    payload = json.dumps({"field": args.field, "datasets": results}, indent=2, ensure_ascii=False)
    try:
        with open(out, "w", encoding="utf-8") as f:
            f.write(payload)
    except OSError:
        out = os.path.join(os.getcwd(), "error_analysis.json")
        with open(out, "w", encoding="utf-8") as f:
            f.write(payload)
    print(f"\nWrote combined report -> {out}")


if __name__ == "__main__":
    main()
