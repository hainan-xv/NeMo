"""
Per-dataset utterance-length statistics over the ASR result manifests.

For every test set under ``results/`` this reports the distribution of:
  * audio length  -- the manifest ``duration`` field (seconds);
  * text length   -- number of words in the reference text;
  * speaking rate -- words per second (text length / audio length).

Audio duration and reference text are properties of the dataset, not the
model, so each dataset is summarized once using its most complete manifest
(the variant with the most lines).

Examples
--------
python utterance_length_stats.py
python utterance_length_stats.py --use-normalized        # count normalized words
python utterance_length_stats.py --output utt_length_stats.txt
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("numpy is required: pip install numpy")

PREFIX = "hf-audio-esb-datasets-test-only-sorted_"


def dataset_name(path):
    b = os.path.basename(path)
    m = re.search(r"_DATASET_(.+)\.jsonl$", b)
    name = m.group(1) if m else b
    return name[len(PREFIX):] if name.startswith(PREFIX) else name


def pick_manifests(results_dir, pattern):
    """One manifest per dataset: the variant with the most lines."""
    by_ds = defaultdict(list)
    for f in glob.glob(os.path.join(results_dir, pattern)):
        n = sum(1 for _ in open(f, encoding="utf-8"))
        by_ds[dataset_name(f)].append((n, f))
    return {ds: max(v)[1] for ds, v in by_ds.items()}


def load_utts(path, use_normalized):
    """Yield (duration_sec, n_words) per utterance with a usable duration."""
    text_key = "text_normalized" if use_normalized else "text"
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            dur = d.get("duration")
            txt = d.get(text_key) or d.get("text") or d.get("text_normalized") or ""
            nw = len(txt.split())
            if dur and dur > 0:
                out.append((float(dur), nw))
    return out


def summarize(durs, words):
    durs = np.asarray(durs, dtype=float)
    words = np.asarray(words, dtype=float)
    # per-utterance speaking rate (guard against zero-duration already filtered)
    wps = words / durs
    return {
        "n": len(durs),
        "audio_hours": durs.sum() / 3600.0,
        "dur_mean": durs.mean(), "dur_med": np.median(durs),
        "dur_p10": np.percentile(durs, 10), "dur_p90": np.percentile(durs, 90),
        "dur_min": durs.min(), "dur_max": durs.max(),
        "w_mean": words.mean(), "w_med": np.median(words),
        "w_min": words.min(), "w_max": words.max(),
        "wps_mean": wps.mean(), "wps_med": np.median(wps),
        "wps_p10": np.percentile(wps, 10), "wps_p90": np.percentile(wps, 90),
        "corpus_wps": words.sum() / durs.sum(),
    }


def fmt(rows, total):
    L = []
    p = L.append
    p("=" * 118)
    p("Utterance-length statistics by dataset")
    p("=" * 118)

    p("")
    p("Audio duration (seconds):")
    p(f"  {'dataset':<24} {'n_utt':>8} {'hours':>7} {'mean':>7} {'med':>7} "
      f"{'p10':>7} {'p90':>7} {'min':>7} {'max':>8}")
    for ds, s in rows:
        p(f"  {ds:<24} {s['n']:>8} {s['audio_hours']:>7.1f} {s['dur_mean']:>7.2f} "
          f"{s['dur_med']:>7.2f} {s['dur_p10']:>7.2f} {s['dur_p90']:>7.2f} "
          f"{s['dur_min']:>7.2f} {s['dur_max']:>8.2f}")
    s = total
    p(f"  {'TOTAL/ALL':<24} {s['n']:>8} {s['audio_hours']:>7.1f} {s['dur_mean']:>7.2f} "
      f"{s['dur_med']:>7.2f} {s['dur_p10']:>7.2f} {s['dur_p90']:>7.2f} "
      f"{s['dur_min']:>7.2f} {s['dur_max']:>8.2f}")

    p("")
    p("Text length (words per utterance):")
    p(f"  {'dataset':<24} {'n_utt':>8} {'mean':>7} {'med':>7} {'min':>7} {'max':>8}")
    for ds, s in rows:
        p(f"  {ds:<24} {s['n']:>8} {s['w_mean']:>7.1f} {s['w_med']:>7.1f} "
          f"{s['w_min']:>7.0f} {s['w_max']:>8.0f}")
    s = total
    p(f"  {'TOTAL/ALL':<24} {s['n']:>8} {s['w_mean']:>7.1f} {s['w_med']:>7.1f} "
      f"{s['w_min']:>7.0f} {s['w_max']:>8.0f}")

    p("")
    p("Speaking rate (words per second):")
    p(f"  {'dataset':<24} {'mean':>7} {'med':>7} {'p10':>7} {'p90':>7} {'corpus':>8}")
    p("  (corpus = total_words / total_seconds; mean/med are over utterances)")
    for ds, s in rows:
        p(f"  {ds:<24} {s['wps_mean']:>7.2f} {s['wps_med']:>7.2f} {s['wps_p10']:>7.2f} "
          f"{s['wps_p90']:>7.2f} {s['corpus_wps']:>8.2f}")
    s = total
    p(f"  {'TOTAL/ALL':<24} {s['wps_mean']:>7.2f} {s['wps_med']:>7.2f} {s['wps_p10']:>7.2f} "
      f"{s['wps_p90']:>7.2f} {s['corpus_wps']:>8.2f}")
    p("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--pattern", default="*.jsonl")
    ap.add_argument("--use-normalized", action="store_true",
                    help="Count words in text_normalized instead of the raw text.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    chosen = pick_manifests(args.results_dir, args.pattern)
    if not chosen:
        sys.exit(f"No manifests in {args.results_dir!r} matching {args.pattern!r}.")

    rows, all_d, all_w = [], [], []
    for ds in sorted(chosen):
        utts = load_utts(chosen[ds], args.use_normalized)
        if not utts:
            continue
        durs = [u[0] for u in utts]
        words = [u[1] for u in utts]
        rows.append((ds, summarize(durs, words)))
        all_d.extend(durs)
        all_w.extend(words)
        print(f"  {ds}: {os.path.basename(chosen[ds])}", file=sys.stderr)

    total = summarize(all_d, all_w)
    text = fmt(rows, total)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n[written to {args.output}]", file=sys.stderr)


if __name__ == "__main__":
    main()
