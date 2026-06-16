"""
Relate ASR word-level errors to lexical / positional features.

Where ``analyze_decoding_errors.py`` summarizes *what* the errors are, this
script asks *which kinds of words* get missed.  The unit of analysis is the
reference word: from the jiwer alignment every reference word is a hit
(``equal``), a substitution, or a deletion, so it carries a clean binary
outcome ``error = 1`` (sub or del) vs ``0`` (hit).  Insertions have no
reference word and are excluded here -- so the headline number is a
*per-reference-word error rate*, which differs slightly from WER (WER also
counts insertions in the numerator).

For every reference word we compute text-only features and then report, per
feature, an error-rate-by-bucket table (with Wilson 95% CIs) plus a
point-biserial correlation (Pearson of the feature against the 0/1 outcome)
so the features can be ranked by how strongly they track errors.

Features
--------
  * frequency rank   -- position in a frequency-ordered word list (OOV = rarer
                        than the rarest listed word); also used as log10(rank).
  * word length      -- number of characters.
  * syllable count   -- heuristic vowel-group estimate.
  * utterance position (relative, deciles) -- boundary effects (esp. streaming).
  * utterance length -- number of reference words in the utterance.
  * is-number        -- number word or digit string.
  * has-apostrophe   -- contraction / possessive.

Features that would likely help but need more than ref/hyp text (NOT computed):
acoustic duration / speaking rate, SNR / noise level, homophone confusability
(needs a pronunciation dict), proper-noun / NER flags (needs casing or NER).

Examples
--------
# Aggregate every manifest from one model, all datasets:
python analyze_word_features.py --model ord_chunkedaligner_chunkedaligner_c12

# One dataset, save the report:
python analyze_word_features.py --dataset librispeech_test.clean \
    --output word_features_lsclean.txt
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

try:
    import jiwer
except ImportError:  # pragma: no cover
    sys.exit("jiwer is required: pip install jiwer")

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("numpy is required: pip install numpy")


NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "first", "second", "third",
}

DEFAULT_FREQ_LIST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "google-10000-english-usa.txt",
)


# --------------------------------------------------------------------------- #
# Lexical helpers
# --------------------------------------------------------------------------- #
def freq_word_key(word):
    """Normalize a token for frequency-list membership (lower, strip edge punct)."""
    return word.lower().strip(".,!?;:\"'()[]{}")


def load_freq_ranks(path):
    """Map each word to its 1-indexed rank in a frequency-ordered list.

    Returns ``(ranks, n_listed)`` where ``ranks[word] = rank`` (smallest is
    most frequent) and ``n_listed`` is the number of distinct listed words.
    """
    ranks = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = freq_word_key(line.strip())
            if w and w not in ranks:
                ranks[w] = len(ranks) + 1
    return ranks, len(ranks)


_VOWELS = set("aeiouy")


def count_syllables(word):
    """Cheap English syllable estimate: count vowel groups, drop silent -e."""
    w = "".join(ch for ch in word.lower() if ch.isalpha())
    if not w:
        return 0
    groups = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            groups += 1
        prev_vowel = is_vowel
    if w.endswith("e") and groups > 1:
        groups -= 1
    return max(1, groups)


def wilson(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def find_manifests(results_dir, model=None, dataset=None, pattern=None):
    files = sorted(glob.glob(os.path.join(results_dir, pattern or "*.jsonl")))
    out = []
    for f in files:
        base = os.path.basename(f)
        if model and model not in base:
            continue
        if dataset and dataset not in base:
            continue
        out.append(f)
    return out


def load_pairs(path, use_formatted=False):
    """Return list of (ref, hyp) string pairs from one manifest."""
    ref_key = "text" if use_formatted else "text_normalized"
    hyp_key = "pred_text" if use_formatted else "pred_text_normalized"
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ref = d.get(ref_key)
            hyp = d.get(hyp_key)
            if ref is None:
                ref = d.get("text") or d.get("text_normalized") or ""
            if hyp is None:
                hyp = d.get("pred_text") or d.get("pred_text_normalized") or ""
            pairs.append((ref or "", hyp or ""))
    return pairs


# --------------------------------------------------------------------------- #
# Per-word record collection
# --------------------------------------------------------------------------- #
class Records:
    """Column-oriented store of per-reference-word features + error outcome."""

    FIELDS = ("error", "rank", "length", "syllables", "pos", "utt_len",
              "is_number", "has_apostrophe")

    def __init__(self):
        for f in self.FIELDS:
            setattr(self, f, [])
        self.n_words = 0
        self.n_utt = 0

    def add(self, **kw):
        for f in self.FIELDS:
            getattr(self, f).append(kw[f])
        self.n_words += 1

    def arrays(self):
        return {f: np.asarray(getattr(self, f), dtype=float) for f in self.FIELDS}


def collect(pairs, ranks, n_listed, rec):
    """Word-align pairs and append one record per reference word to ``rec``."""
    refs = [r for r, _ in pairs]
    hyps = [h for _, h in pairs]
    if not refs:
        return
    oov_rank = n_listed + 1  # treat unseen words as rarer than the rarest listed
    out = jiwer.process_words(refs, hyps)
    for i, chunks in enumerate(out.alignments):
        ref_words = out.references[i]
        n_ref = len(ref_words)
        if n_ref == 0:
            continue
        rec.n_utt += 1
        outcome = [None] * n_ref
        for c in chunks:
            if c.type == "equal":
                for idx in range(c.ref_start_idx, c.ref_end_idx):
                    outcome[idx] = 0
            elif c.type == "substitute":
                for idx in range(c.ref_start_idx, c.ref_end_idx):
                    outcome[idx] = 1
            elif c.type == "delete":
                for idx in range(c.ref_start_idx, c.ref_end_idx):
                    outcome[idx] = 1
            # insert: no reference word
        for idx, word in enumerate(ref_words):
            if outcome[idx] is None:
                continue
            key = freq_word_key(word)
            rank = ranks.get(key, oov_rank)
            pos = idx / (n_ref - 1) if n_ref > 1 else 0.0
            rec.add(
                error=outcome[idx],
                rank=rank,
                length=len(key) if key else len(word),
                syllables=count_syllables(key),
                pos=pos,
                utt_len=n_ref,
                is_number=1 if (key in NUMBER_WORDS or key.isdigit()) else 0,
                has_apostrophe=1 if "'" in word else 0,
            )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def bucket_table(p, title, labels, masks, error_arr, max_bar=34):
    """Emit an error-rate-by-bucket table given boolean masks per bucket."""
    p("")
    p(title)
    p(f"  {'bucket':<16} {'n_words':>10} {'%':>6} {'errors':>9} {'err_rate':>9} {'95% CI':>16}")
    total = len(error_arr)
    rates = []
    for lab, m in zip(labels, masks):
        n = int(m.sum())
        if n == 0:
            continue
        k = int(error_arr[m].sum())
        rate = k / n
        lo, hi = wilson(k, n)
        pct = 100 * n / total if total else 0.0
        bar = "#" * int(round(max_bar * rate))
        p(f"  {lab:<16} {n:>10} {pct:>5.1f}% {k:>9} {100*rate:>8.2f}% "
          f"[{100*lo:>5.1f},{100*hi:>5.1f}]  {bar}")
        rates.append((lab, rate, n))
    return rates


def point_biserial(feat, error_arr):
    """Pearson correlation of a feature with the 0/1 error outcome."""
    if len(feat) < 2 or np.std(feat) == 0 or np.std(error_arr) == 0:
        return float("nan")
    return float(np.corrcoef(feat, error_arr)[0, 1])


def fmt_report(title, rec, ranks_size, top_corr=True):
    L = []
    p = L.append
    A = rec.arrays()
    err = A["error"]
    n = len(err)

    p("=" * 92)
    p(title)
    p("=" * 92)
    if n == 0:
        p("(no reference words)")
        return "\n".join(L)
    overall = err.mean()
    p(f"utterances                  : {rec.n_utt}")
    p(f"reference words             : {n}")
    p(f"per-ref-word error rate     : {100*overall:.2f} %   "
      f"({int(err.sum())} words sub'd or deleted; insertions excluded)")

    # log-rank for correlation (monotone in rarity)
    log_rank = np.log10(A["rank"])

    # ---- correlation ranking --------------------------------------------- #
    corr_feats = {
        "log10(freq rank)": log_rank,
        "word length (chars)": A["length"],
        "syllable count": A["syllables"],
        "position in utt (0=start,1=end)": A["pos"],
        "utterance length (words)": A["utt_len"],
        "is number/digit": A["is_number"],
        "has apostrophe": A["has_apostrophe"],
    }
    corrs = [(name, point_biserial(v, err)) for name, v in corr_feats.items()]
    corrs.sort(key=lambda t: (-abs(t[1]) if not math.isnan(t[1]) else 0))
    p("")
    p("Point-biserial correlation with per-word error (sorted by |r|):")
    p(f"  {'feature':<34} {'r':>8}")
    for name, r in corrs:
        p(f"  {name:<34} {r:>8.3f}")
    p("  (r>0 means the feature increases error probability; |r|>=0.1 is")
    p("   already notable for a noisy per-word binary outcome.)")

    # ---- frequency rank --------------------------------------------------- #
    rank = A["rank"]
    oov = ranks_size + 1
    freq_labels = ["1-100", "101-300", "301-1000", "1001-2000",
                   "2001-5000", "5001-%d" % ranks_size, "OOV (not listed)"]
    freq_masks = [
        (rank >= 1) & (rank <= 100),
        (rank > 100) & (rank <= 300),
        (rank > 300) & (rank <= 1000),
        (rank > 1000) & (rank <= 2000),
        (rank > 2000) & (rank <= 5000),
        (rank > 5000) & (rank < oov),
        (rank >= oov),
    ]
    bucket_table(p, "Error rate by word-frequency rank (lower rank = more common):",
                 freq_labels, freq_masks, err)

    # ---- word length ------------------------------------------------------ #
    length = A["length"]
    len_labels, len_masks = [], []
    for ln in range(1, 13):
        len_labels.append(str(ln))
        len_masks.append(length == ln)
    len_labels.append("13+")
    len_masks.append(length >= 13)
    bucket_table(p, "Error rate by word length (characters):",
                 len_labels, len_masks, err)

    # ---- syllables -------------------------------------------------------- #
    syl = A["syllables"]
    syl_labels, syl_masks = [], []
    for s in range(1, 5):
        syl_labels.append(str(s))
        syl_masks.append(syl == s)
    syl_labels.append("5+")
    syl_masks.append(syl >= 5)
    bucket_table(p, "Error rate by (estimated) syllable count:",
                 syl_labels, syl_masks, err)

    # ---- position in utterance ------------------------------------------- #
    pos = A["pos"]
    pos_labels, pos_masks = [], []
    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        pos_labels.append(f"[{int(lo_e*100)}-{int(min(hi_e,1.0)*100)}%)")
        pos_masks.append((pos >= lo_e) & (pos < hi_e))
    bucket_table(p, "Error rate by relative position in utterance "
                    "(start -> end; single-word utts -> 0%):",
                 pos_labels, pos_masks, err)

    # ---- utterance length ------------------------------------------------- #
    ulen = A["utt_len"]
    ul_labels = ["1-3", "4-7", "8-15", "16-30", "31-60", "61+"]
    ul_edges = [(1, 3), (4, 7), (8, 15), (16, 30), (31, 60), (61, 10**9)]
    ul_masks = [(ulen >= a) & (ulen <= b) for a, b in ul_edges]
    bucket_table(p, "Error rate by utterance length (number of reference words):",
                 ul_labels, ul_masks, err)

    # ---- flags ------------------------------------------------------------ #
    bucket_table(p, "Error rate for number words / digits:",
                 ["not number", "number/digit"],
                 [A["is_number"] == 0, A["is_number"] == 1], err)
    bucket_table(p, "Error rate for apostrophe words (contractions/possessives):",
                 ["no apostrophe", "has apostrophe"],
                 [A["has_apostrophe"] == 0, A["has_apostrophe"] == 1], err)

    # ---- interaction: frequency x length --------------------------------- #
    common = rank <= 5000
    short = length <= 6
    p("")
    p("Interaction: common-vs-rare x short-vs-long (does length matter beyond rarity?)")
    p("  (common = freq rank <= 5000; short = <= 6 chars)")
    p(f"  {'cell':<24} {'n_words':>10} {'errors':>9} {'err_rate':>9}")
    for clab, cmask in [("common", common), ("rare", ~common)]:
        for slab, smask in [("short", short), ("long", ~short)]:
            m = cmask & smask
            nn = int(m.sum())
            if nn == 0:
                continue
            kk = int(err[m].sum())
            p(f"  {clab + ' + ' + slab:<24} {nn:>10} {kk:>9} {100*kk/nn:>8.2f}%")

    p("")
    p("Note: features needing audio/timing (acoustic duration, speaking rate,")
    p("SNR) or extra resources (homophone confusability via a pronunciation")
    p("dict, proper-noun/NER flags) are not computed here.")
    p("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results", help="Directory with *.jsonl manifests.")
    ap.add_argument("--pattern", default=None, help="Glob within results-dir (default: *.jsonl).")
    ap.add_argument("--model", default=None, help="Substring filter on the MODEL_... part of filenames.")
    ap.add_argument("--dataset", default=None, help="Substring filter (e.g. 'librispeech_test.clean').")
    ap.add_argument("--per-file", action="store_true", help="One report per manifest (default: aggregate).")
    ap.add_argument("--use-formatted", action="store_true",
                    help="Analyze casing/punct text instead of the normalized (WER) text.")
    ap.add_argument("--freq-list", default=DEFAULT_FREQ_LIST,
                    help="Frequency-ordered word list (one word/line).")
    ap.add_argument("--output", default=None, help="Also write the report to this file.")
    args = ap.parse_args()

    if not os.path.exists(args.freq_list):
        sys.exit(f"freq list {args.freq_list!r} not found (pass --freq-list).")
    ranks, n_listed = load_freq_ranks(args.freq_list)
    print(f"Loaded {n_listed} ranked words from {os.path.basename(args.freq_list)}.",
          file=sys.stderr)

    manifests = find_manifests(args.results_dir, args.model, args.dataset, args.pattern)
    if not manifests:
        sys.exit(f"No manifests matched in {args.results_dir!r} "
                 f"(model={args.model!r}, dataset={args.dataset!r}, pattern={args.pattern!r}).")
    print(f"Matched {len(manifests)} manifest(s):", file=sys.stderr)
    for m in manifests:
        print(f"  {os.path.basename(m)}", file=sys.stderr)

    reports = []
    if args.per_file:
        for m in manifests:
            rec = Records()
            collect(load_pairs(m, args.use_formatted), ranks, n_listed, rec)
            reports.append(fmt_report(os.path.basename(m), rec, n_listed))
    else:
        rec = Records()
        for m in manifests:
            collect(load_pairs(m, args.use_formatted), ranks, n_listed, rec)
        title = "AGGREGATE over %d manifest(s)" % len(manifests)
        if args.model:
            title += f"  [model~={args.model}]"
        if args.dataset:
            title += f"  [dataset~={args.dataset}]"
        reports.append(fmt_report(title, rec, n_listed))

    text = "\n".join(reports)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n[report written to {args.output}]", file=sys.stderr)


if __name__ == "__main__":
    main()
