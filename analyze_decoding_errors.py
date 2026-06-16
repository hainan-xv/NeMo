"""
Analyze ASR decoding results and surface error patterns for inspection.

Consumes the per-(model, dataset, split) result manifests written by
``run_eval_asr.py`` into ``./results/`` (one JSON object per line with the
fields ``text`` / ``pred_text`` -- formatted ref/hyp -- and
``text_normalized`` / ``pred_text_normalized`` -- the whisper-normalized
strings actually used for leaderboard WER).

For each manifest (or an aggregate over several), it word-aligns every
reference/hypothesis pair (via ``jiwer``) and reports:

  * the headline WER plus its substitution / insertion / deletion breakdown;
  * the most frequent substitution pairs (ref -> hyp), deletions, insertions;
  * a categorization of substitutions (near-miss spelling, number words,
    word split/merge, casing/punct-only seen after normalization, other);
  * a per-utterance WER distribution and the worst-scoring utterances, with
    their word-level alignment laid out so you can eyeball what went wrong.

Examples
--------
# Aggregate over every manifest from one model, all datasets:
python analyze_decoding_errors.py \
    --model ord_chat_chat_llmvocab_qwen3_1p7b_fullctx64_c16

# One dataset, dump the 40 worst utterances with full alignments:
python analyze_decoding_errors.py \
    --model ord_chat_chat_llmvocab_qwen3_1p7b --dataset librispeech_test.clean \
    --worst 40 --show-alignments

# Compare every file separately (one report block per manifest) and save:
python analyze_decoding_errors.py --per-file --output error_report.txt
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

try:
    import jiwer
except ImportError:  # pragma: no cover
    sys.exit("jiwer is required: pip install jiwer")


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


def freq_word_key(word):
    """Normalize a token for frequency-list membership (lower, strip edge punct)."""
    return word.lower().strip(".,!?;:\"'()[]{}")


def load_freq_words(path, topn):
    """Load the first ``topn`` words of a frequency-ordered word list (one/line)."""
    words = set()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if topn is not None and i >= topn:
                break
            w = line.strip()
            if w:
                words.add(freq_word_key(w))
    return words


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
            # Fall back to the other field set if the requested one is absent.
            if ref is None:
                ref = d.get("text") or d.get("text_normalized") or ""
            if hyp is None:
                hyp = d.get("pred_text") or d.get("pred_text_normalized") or ""
            pairs.append((ref or "", hyp or ""))
    return pairs


# --------------------------------------------------------------------------- #
# Alignment + aggregation
# --------------------------------------------------------------------------- #
def char_similarity(a, b):
    """1 - normalized Levenshtein distance between two words (0..1)."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[lb] / max(la, lb)


def categorize_sub(ref_w, hyp_w):
    """Coarse bucket for a substitution pair."""
    if ref_w.replace("'", "") == hyp_w.replace("'", ""):
        return "apostrophe"
    if ref_w in NUMBER_WORDS or hyp_w in NUMBER_WORDS or ref_w.isdigit() or hyp_w.isdigit():
        return "number"
    if char_similarity(ref_w, hyp_w) >= 0.6:
        return "near-miss (spelling)"
    return "other"


class Stats:
    def __init__(self):
        self.n_utt = 0
        self.n_ref_words = 0
        self.hits = 0
        self.subs = 0
        self.ins = 0
        self.dels = 0
        self.sub_pairs = Counter()
        self.del_words = Counter()
        self.ins_words = Counter()
        self.sub_categories = Counter()
        self.utt_wer = []          # (wer, n_ref, ref, hyp, alignment_str)
        self.wer_buckets = Counter()
        # Reference-word accounting split by frequency bucket
        # ("common" = in top-N freq list, "rare" = not in list).
        self.freq_hits = {"common": 0, "rare": 0}
        self.freq_subs = {"common": 0, "rare": 0}
        self.freq_dels = {"common": 0, "rare": 0}
        self.rare_err_words = Counter()   # rare ref words that were sub'd/deleted

    def merge(self, other):
        self.n_utt += other.n_utt
        self.n_ref_words += other.n_ref_words
        self.hits += other.hits
        self.subs += other.subs
        self.ins += other.ins
        self.dels += other.dels
        self.sub_pairs.update(other.sub_pairs)
        self.del_words.update(other.del_words)
        self.ins_words.update(other.ins_words)
        self.sub_categories.update(other.sub_categories)
        self.utt_wer.extend(other.utt_wer)
        self.wer_buckets.update(other.wer_buckets)
        for k in ("common", "rare"):
            self.freq_hits[k] += other.freq_hits[k]
            self.freq_subs[k] += other.freq_subs[k]
            self.freq_dels[k] += other.freq_dels[k]
        self.rare_err_words.update(other.rare_err_words)

    @property
    def wer(self):
        denom = self.hits + self.subs + self.dels
        return (self.subs + self.dels + self.ins) / denom if denom else 0.0


def bucket_for(wer):
    if wer == 0:
        return "0% (perfect)"
    if wer <= 0.05:
        return "(0%, 5%]"
    if wer <= 0.10:
        return "(5%, 10%]"
    if wer <= 0.20:
        return "(10%, 20%]"
    if wer <= 0.50:
        return "(20%, 50%]"
    return "(50%, 100%+]"


def alignment_to_str(chunks, ref_words, hyp_words, max_ops=60):
    """Render a per-utterance alignment as readable [S]/[D]/[I] op tokens."""
    ops = []
    for c in chunks:
        if c.type == "equal":
            continue
        if c.type == "substitute":
            for r, h in zip(ref_words[c.ref_start_idx:c.ref_end_idx],
                            hyp_words[c.hyp_start_idx:c.hyp_end_idx]):
                ops.append(f"[S] {r} -> {h}")
        elif c.type == "delete":
            for r in ref_words[c.ref_start_idx:c.ref_end_idx]:
                ops.append(f"[D] {r}")
        elif c.type == "insert":
            for h in hyp_words[c.hyp_start_idx:c.hyp_end_idx]:
                ops.append(f"[I] {h}")
    if len(ops) > max_ops:
        ops = ops[:max_ops] + [f"... (+{len(ops) - max_ops} more ops)"]
    return " | ".join(ops)


def analyze(pairs, freq_words=None):
    """Word-align every pair and accumulate Stats.

    If ``freq_words`` (a set of normalized common words) is given, every
    reference word is bucketed as ``common`` (in the set) or ``rare`` (not),
    and per-bucket hit/sub/del counts are tracked so we can compare error
    rates on frequent vs. infrequent words.
    """
    st = Stats()
    refs = [r for r, _ in pairs]
    hyps = [h for _, h in pairs]
    if not refs:
        return st

    def bucket(word):
        return "common" if freq_word_key(word) in freq_words else "rare"

    out = jiwer.process_words(refs, hyps)
    for i, chunks in enumerate(out.alignments):
        ref_words = out.references[i]
        hyp_words = out.hypotheses[i]
        n_ref = len(ref_words)
        s = d = ins = hit = 0
        for c in chunks:
            if c.type == "equal":
                hit += c.ref_end_idx - c.ref_start_idx
                if freq_words is not None:
                    for r in ref_words[c.ref_start_idx:c.ref_end_idx]:
                        st.freq_hits[bucket(r)] += 1
            elif c.type == "substitute":
                for r, h in zip(ref_words[c.ref_start_idx:c.ref_end_idx],
                                hyp_words[c.hyp_start_idx:c.hyp_end_idx]):
                    st.sub_pairs[(r, h)] += 1
                    st.sub_categories[categorize_sub(r, h)] += 1
                    s += 1
                    if freq_words is not None:
                        b = bucket(r)
                        st.freq_subs[b] += 1
                        if b == "rare":
                            st.rare_err_words[r] += 1
            elif c.type == "delete":
                for r in ref_words[c.ref_start_idx:c.ref_end_idx]:
                    st.del_words[r] += 1
                    d += 1
                    if freq_words is not None:
                        b = bucket(r)
                        st.freq_dels[b] += 1
                        if b == "rare":
                            st.rare_err_words[r] += 1
            elif c.type == "insert":
                for h in hyp_words[c.hyp_start_idx:c.hyp_end_idx]:
                    st.ins_words[h] += 1
                    ins += 1

        st.n_utt += 1
        st.n_ref_words += n_ref
        st.hits += hit
        st.subs += s
        st.dels += d
        st.ins += ins

        denom = hit + s + d
        uwer = (s + d + ins) / denom if denom else (1.0 if ins else 0.0)
        st.wer_buckets[bucket_for(uwer)] += 1
        st.utt_wer.append((uwer, n_ref, refs[i], hyps[i],
                           alignment_to_str(chunks, ref_words, hyp_words)))
    return st


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def fmt_report(title, st, top, worst, show_alignments, min_count=2, freq_label=None):
    L = []
    p = L.append
    p("=" * 88)
    p(title)
    p("=" * 88)
    total_err = st.subs + st.ins + st.dels
    p(f"utterances           : {st.n_utt}")
    p(f"reference words       : {st.n_ref_words}")
    p(f"WER                   : {100 * st.wer:.2f} %   ({total_err} errors)")
    if total_err:
        p(f"  substitutions       : {st.subs:>7}  ({100 * st.subs / total_err:.1f}% of errors)")
        p(f"  deletions           : {st.dels:>7}  ({100 * st.dels / total_err:.1f}% of errors)")
        p(f"  insertions          : {st.ins:>7}  ({100 * st.ins / total_err:.1f}% of errors)")

    p("")
    p("Per-utterance WER distribution:")
    order = ["0% (perfect)", "(0%, 5%]", "(5%, 10%]", "(10%, 20%]", "(20%, 50%]", "(50%, 100%+]"]
    for b in order:
        c = st.wer_buckets.get(b, 0)
        if c:
            bar = "#" * int(round(40 * c / max(1, st.n_utt)))
            p(f"  {b:<14}: {c:>6}  {bar}")

    if st.subs:
        p("")
        p("Substitution categories:")
        for cat, c in st.sub_categories.most_common():
            p(f"  {cat:<22}: {c:>6}  ({100 * c / st.subs:.1f}%)")

    freq_total = sum(st.freq_hits.values()) + sum(st.freq_subs.values()) + sum(st.freq_dels.values())
    if freq_total:
        p("")
        p("Reference-word error rate by frequency bucket" + (f"  [{freq_label}]" if freq_label else "") + ":")
        p(f"  {'bucket':<8} {'ref_words':>10} {'%ofref':>8} {'hits':>9} {'subs':>8} {'dels':>8} {'err_rate':>9}")
        rates = {}
        for b in ("common", "rare"):
            ref_w = st.freq_hits[b] + st.freq_subs[b] + st.freq_dels[b]
            err = st.freq_subs[b] + st.freq_dels[b]
            rate = err / ref_w if ref_w else 0.0
            rates[b] = rate
            pct = 100 * ref_w / freq_total if freq_total else 0.0
            p(f"  {b:<8} {ref_w:>10} {pct:>7.1f}% {st.freq_hits[b]:>9} "
              f"{st.freq_subs[b]:>8} {st.freq_dels[b]:>8} {100 * rate:>8.2f}%")
        if rates["common"] > 0:
            p(f"  -> rare-word error rate is {rates['rare'] / rates['common']:.2f}x the common-word rate")
        p("")
        p(f"Top {top} rare (out-of-list) reference words that were mis-recognized:")
        shown = 0
        for w, c in st.rare_err_words.most_common(top):
            if c < min_count:
                break
            p(f"  {c:>5}  {w!r}")
            shown += 1
        if not shown:
            p("  (none above the min-count threshold)")

    p("")
    p(f"Top {top} substitution pairs (ref -> hyp):")
    for (r, h), c in st.sub_pairs.most_common(top):
        if c < min_count:
            break
        p(f"  {c:>5}  {r!r} -> {h!r}")

    p("")
    p(f"Top {top} deleted reference words (model dropped these):")
    for w, c in st.del_words.most_common(top):
        if c < min_count:
            break
        p(f"  {c:>5}  {w!r}")

    p("")
    p(f"Top {top} inserted words (model hallucinated / added these):")
    for w, c in st.ins_words.most_common(top):
        if c < min_count:
            break
        p(f"  {c:>5}  {w!r}")

    p("")
    p(f"Worst {worst} utterances by WER (n_ref words >= 3):")
    worst_sorted = sorted([u for u in st.utt_wer if u[1] >= 3],
                          key=lambda u: (u[0], u[1]), reverse=True)[:worst]
    for uwer, n_ref, ref, hyp, align in worst_sorted:
        p(f"  - WER {100 * uwer:6.1f}%  ({n_ref} ref words)")
        if show_alignments:
            p(f"      REF: {ref}")
            p(f"      HYP: {hyp}")
            p(f"      OPS: {align}")
        else:
            p(f"      OPS: {align}")
    p("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results", help="Directory with *.jsonl manifests.")
    ap.add_argument("--pattern", default=None, help="Glob within results-dir (default: *.jsonl).")
    ap.add_argument("--model", default=None, help="Substring filter on the MODEL_... part of filenames.")
    ap.add_argument("--dataset", default=None, help="Substring filter (e.g. 'librispeech_test.clean').")
    ap.add_argument("--per-file", action="store_true", help="One report per manifest (default: aggregate).")
    ap.add_argument("--use-formatted", action="store_true",
                    help="Analyze casing/punct text instead of the normalized (WER) text.")
    ap.add_argument("--top", type=int, default=30, help="How many top patterns to list.")
    ap.add_argument("--worst", type=int, default=20, help="How many worst utterances to show.")
    ap.add_argument("--min-count", type=int, default=2, help="Hide patterns occurring fewer than N times.")
    ap.add_argument("--show-alignments", action="store_true",
                    help="Print full REF/HYP text for each worst utterance.")
    ap.add_argument("--freq-list", default=DEFAULT_FREQ_LIST,
                    help="Frequency-ordered word list (one word/line) for common-vs-rare analysis.")
    ap.add_argument("--freq-topn", type=int, default=5000,
                    help="Treat the first N words of --freq-list as 'common' (default 5000).")
    ap.add_argument("--no-freq", action="store_true",
                    help="Disable the common-vs-rare reference-word error-rate analysis.")
    ap.add_argument("--output", default=None, help="Also write the report to this file.")
    args = ap.parse_args()

    freq_words = None
    freq_label = None
    if not args.no_freq:
        if not os.path.exists(args.freq_list):
            print(f"[warn] freq list {args.freq_list!r} not found; skipping common-vs-rare analysis "
                  f"(use --no-freq to silence).", file=sys.stderr)
        else:
            freq_words = load_freq_words(args.freq_list, args.freq_topn)
            freq_label = f"top-{args.freq_topn} of {os.path.basename(args.freq_list)}"
            print(f"Loaded {len(freq_words)} common words ({freq_label}).", file=sys.stderr)

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
            st = analyze(load_pairs(m, args.use_formatted), freq_words)
            reports.append(fmt_report(os.path.basename(m), st, args.top, args.worst,
                                      args.show_alignments, args.min_count, freq_label))
    else:
        agg = Stats()
        for m in manifests:
            agg.merge(analyze(load_pairs(m, args.use_formatted), freq_words))
        title = "AGGREGATE over %d manifest(s)" % len(manifests)
        if args.model:
            title += f"  [model~={args.model}]"
        if args.dataset:
            title += f"  [dataset~={args.dataset}]"
        reports.append(fmt_report(title, agg, args.top, args.worst,
                                  args.show_alignments, args.min_count, freq_label))

    text = "\n".join(reports)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n[report written to {args.output}]", file=sys.stderr)


if __name__ == "__main__":
    main()
