#!/usr/bin/env python3
"""Average WER as a function of audio length (1-second buckets).

Reads the per-utterance manifests that run_eval_sslm.py writes to ./results/
(one JSONL per dataset; each line has {duration, text=ref, pred_text=hyp}, both
already whisper-normalized) and reports the micro-averaged WER
(= sum of word edits / sum of reference words) within each length bucket.

By default it analyzes the most recent LOCAL-checkpoint run (MODEL_checkpoints_*
manifests, e.g. the imend_12 chunks=2 eval). Select a different run with
--model <MODEL_label> or point at explicit files with --glob.

Examples:
  python wer_by_length.py                       # pooled over all datasets, 1s buckets
  python wer_by_length.py --per_dataset         # + a per-dataset breakdown
  python wer_by_length.py --bucket 2 --max_sec 30
  python wer_by_length.py --model ord_bidirHainan_parakeetTdt0p6b...   # a different run
  python wer_by_length.py --glob 'results/MODEL_checkpoints_*librispeech*'
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict


def edit_ops(ref_words, hyp_words):
    """Word-level Levenshtein op counts -> (substitutions, deletions, insertions).

    deletion = word in ref but not hyp (model dropped it);
    insertion = word in hyp but not ref (model hallucinated it).
    """
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return 0, 0, m
    if m == 0:
        return 0, n, 0
    # Full DP table (utterances are short; max ~a few hundred words).
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri = ref_words[i - 1]
        di, dim1 = d[i], d[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp_words[j - 1] else 1
            di[j] = min(dim1[j] + 1, di[j - 1] + 1, dim1[j - 1] + cost)
    # Backtrace to classify each edit.
    i, j = n, m
    sub = dele = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i -= 1; j -= 1                       # match
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            sub += 1; i -= 1; j -= 1             # substitution
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1; i -= 1                    # deletion (ref word dropped)
        else:
            ins += 1; j -= 1                     # insertion (extra hyp word)
    return sub, dele, ins


def dataset_of(path):
    """Pull the '<name>_<split>' tail out of a manifest filename for labeling."""
    b = os.path.basename(path)
    m = re.search(r"sorted_(.+)\.jsonl$", b)
    return m.group(1) if m else b


def analyze(files, bucket, max_sec):
    """Return (per_bucket, per_dataset_bucket, overall) accumulators.

    Each accumulator maps key -> [sub, del, ins, ref_words, n_utt].
    """
    per_bucket = defaultdict(lambda: [0, 0, 0, 0, 0])
    per_ds = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0]))
    overall = [0, 0, 0, 0, 0]
    for f in files:
        ds = dataset_of(f)
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                dur = d.get("duration")
                if dur is None:
                    continue
                ref = (d.get("text") or "").split()
                hyp = (d.get("pred_text") or "").split()
                sub, dele, ins = edit_ops(ref, hyp)
                nref = len(ref)
                b = int(dur // bucket)
                if max_sec is not None and dur >= max_sec:
                    b = int(max_sec // bucket)  # tail lump
                for acc in (per_bucket[b], per_ds[ds][b], overall):
                    acc[0] += sub
                    acc[1] += dele
                    acc[2] += ins
                    acc[3] += nref
                    acc[4] += 1
    return per_bucket, per_ds, overall


def _rates(acc):
    """(WER%, sub%, del%, ins%) micro-averaged over the bucket."""
    sub, dele, ins, nref, _ = acc
    if not nref:
        return (float("nan"),) * 4
    return (100.0 * (sub + dele + ins) / nref,
            100.0 * sub / nref, 100.0 * dele / nref, 100.0 * ins / nref)


def print_table(title, buckets, bucket, max_sec):
    print(f"\n{title}")
    print(f"{'audio length (s)':>18}{'#utt':>9}{'ref words':>12}"
          f"{'WER%':>8}{'sub%':>7}{'del%':>7}{'ins%':>7}")
    print("-" * 68)
    tot = [0, 0, 0, 0, 0]
    for b in sorted(buckets):
        acc = buckets[b]
        for k in range(5):
            tot[k] += acc[k]
        lo = b * bucket
        hi = lo + bucket
        if max_sec is not None and math.isclose(lo, max_sec):
            rng = f"{lo:g}+"
        else:
            rng = f"[{lo:g}, {hi:g})"
        wer, s, dl, i = _rates(acc)
        def fmt(x):
            return f"{x:.2f}" if not math.isnan(x) else "n/a"
        print(f"{rng:>18}{acc[4]:>9}{acc[3]:>12}"
              f"{fmt(wer):>8}{fmt(s):>7}{fmt(dl):>7}{fmt(i):>7}")
    print("-" * 68)
    wer, s, dl, i = _rates(tot)
    def fmt(x):
        return f"{x:.2f}" if not math.isnan(x) else "n/a"
    print(f"{'ALL':>18}{tot[4]:>9}{tot[3]:>12}"
          f"{fmt(wer):>8}{fmt(s):>7}{fmt(dl):>7}{fmt(i):>7}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--model", default="checkpoints",
                    help="MODEL_<label> prefix to select (default: 'checkpoints' = local-ckpt runs like imend_12)")
    ap.add_argument("--glob", default=None, help="explicit glob for manifests (overrides --model/--results_dir)")
    ap.add_argument("--bucket", type=float, default=1.0, help="bucket width in seconds (default 1)")
    ap.add_argument("--max_sec", type=float, default=None, help="lump all utterances >= this into a final 'N+' bucket")
    ap.add_argument("--per_dataset", action="store_true", help="also print a per-dataset table")
    args = ap.parse_args()

    if args.glob:
        files = sorted(glob.glob(args.glob))
    else:
        files = sorted(glob.glob(os.path.join(args.results_dir, f"MODEL_{args.model}_*.jsonl")))
    if not files:
        raise SystemExit(f"No manifests matched (model={args.model!r}, dir={args.results_dir!r}). "
                         f"Use --glob to specify explicitly.")

    print("Manifests analyzed:")
    for f in files:
        print(f"  - {dataset_of(f):<22} {f}")

    per_bucket, per_ds, overall = analyze(files, args.bucket, args.max_sec)

    print_table(f"WER vs audio length (pooled over {len(files)} dataset file(s)), "
                f"{args.bucket:g}s buckets", per_bucket, args.bucket, args.max_sec)

    if args.per_dataset:
        for ds in sorted(per_ds):
            print_table(f"[{ds}]", per_ds[ds], args.bucket, args.max_sec)

    wer, s, dl, i = _rates(overall)
    print(f"\nPooled overall WER: {wer:.2f}%  (sub {s:.2f} / del {dl:.2f} / ins {i:.2f})  "
          f"({overall[4]} utts, {overall[3]} ref words)")
    print("Note: pooling mixes datasets of differing difficulty AND length "
          "distributions (e.g. librispeech is short+clean, ami/earnings are long). "
          "Use --per_dataset to disentangle length from dataset difficulty.")


if __name__ == "__main__":
    main()
