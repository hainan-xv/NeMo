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
"""Compare leaderboard eval runs and decompose WHERE the errors come from.

Consumes the ``shard*_of*.generations.jsonl`` files that
``script_leaderboard_eval.py`` / ``nemotron_leaderboard_eval.py`` write, so it
works on any pair of runs without re-decoding. It answers two questions:

1. **Is the headline number right?** It re-scores each run from the raw
   generations with the same normalizer and the same ``kaldialign`` call the
   aggregate uses, and cross-checks that the runs cover the SAME utterance set --
   a WER gap means nothing if the two runs scored different utterances (a shard
   that crashed, a stale ``aggregate.log`` from an earlier decode, a different
   ``--max_eval_samples``).

2. **What KIND of errors?** WER alone cannot distinguish a model that drops words
   from one that repeats them, and those have opposite fixes. The breakdown
   below is chosen so the plausible failure modes separate cleanly:

   ============================  ==================================================
   signal                        what a spike in it means
   ============================  ==================================================
   del% >> ins%                  under-emission: chunks emit blank/<eot> too eagerly
   ins% >> del%                  over-emission: words emitted again in a later chunk
   ``rep%`` (repeated n-grams)   specifically re-emission/looping, not random ins
   ``len_ratio`` << 1            systematic truncation (tail drop, early stop)
   WER rising with ref length    error propagation -- the text history drifts and
                                 never recovers, the signature failure of a model
                                 conditioned on its OWN past output
   WER flat in ref length        a per-chunk acoustic problem, not drift
   ``err_pos`` >> 0.5            errors concentrated late = drift or tail drop
   ``empty%``                    utterances that decoded to nothing at all
   ============================  ==================================================

   For a chunk-size sweep of one model, reading ``del%``/``ins%``/``rep%`` across
   chunk sizes localises the regression far faster than staring at WER.

Usage::

    python scripts/analyze_eval_errors.py \\
        script_c2=/path/to/exp/leaderboard_eval_c2 \\
        script_c14=/path/to/exp/leaderboard_eval_c14 \\
        nemotron_c2=/path/to/nemotron/leaderboard_eval_c2

    # per-dataset table, and dump the worst utterances for eyeballing
    python scripts/analyze_eval_errors.py --per-dataset --examples 15 a=DIR_A b=DIR_B

A directory argument may be the eval dir itself or any parent -- the shard files
are found recursively.
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EPS = "*"  # alignment gap symbol


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def find_shard_files(root: str) -> List[str]:
    """Locate shard JSONLs under ``root`` (which may be the eval dir or a parent)."""
    direct = sorted(glob.glob(os.path.join(root, "shard*_of*.generations.jsonl")))
    if direct:
        return direct
    return sorted(glob.glob(os.path.join(root, "**", "shard*_of*.generations.jsonl"), recursive=True))


def load_run(root: str) -> List[dict]:
    """Read every shard record under ``root``.

    Shards partition the utterance list, so duplicates across files mean a shard
    was decoded twice (e.g. a rerun into the same dir) -- that would double-count
    in scoring, so it is reported rather than silently merged.
    """
    files = find_shard_files(root)
    if not files:
        raise SystemExit(f"ERROR: no shard*_of*.generations.jsonl under {root}")
    recs, seen, dupes = [], set(), 0
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                uid = (rec.get("key", ""), rec.get("reference", ""))
                if uid in seen:
                    dupes += 1
                    continue
                seen.add(uid)
                recs.append(rec)
    if dupes:
        print(f"    [warn] {dupes} duplicate records ignored (overlapping shard files in {root})")
    return recs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def get_normalizer(enable: bool):
    if not enable:
        return lambda s: s
    from leaderboard_normalizer import EnglishTextNormalizer

    return EnglishTextNormalizer()


def align(ref: Sequence[str], hyp: Sequence[str]) -> List[Tuple[str, str]]:
    """Word alignment as (ref_or_EPS, hyp_or_EPS) pairs.

    Uses ``kaldialign.align`` when present -- the same library the scorer uses --
    and otherwise a plain Levenshtein backtrace, which gives identical counts.
    """
    try:
        from kaldialign import align as _ka

        return _ka(list(ref), list(hyp), EPS)
    except Exception:
        pass

    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            out.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append((ref[i - 1], EPS))
            i -= 1
        else:
            out.append((EPS, hyp[j - 1]))
            j -= 1
    return out[::-1]


def repeated_run_words(hyp: Sequence[str], ref: Sequence[str], max_n: int = 4) -> int:
    """Words belonging to an n-gram repeated back-to-back MORE than the reference does.

    Targets the specific failure where a chunk re-emits what a previous chunk
    already emitted ("the cat the cat sat"). Legitimate repetition in the
    reference ("very very good") is subtracted, so this counts only spurious
    looping. Longer n-grams are matched first, and matched spans are consumed, so
    a repeated 3-gram is not also counted as three repeated unigrams.
    """

    def spurious(seq: Sequence[str], n: int, taken: List[bool]) -> int:
        count, i = 0, 0
        while i + 2 * n <= len(seq):
            if any(taken[i : i + 2 * n]):
                i += 1
                continue
            if list(seq[i : i + n]) == list(seq[i + n : i + 2 * n]):
                count += n  # the SECOND copy is the spurious one
                for k in range(i, i + 2 * n):
                    taken[k] = True
                i += 2 * n
            else:
                i += 1
        return count

    total = 0
    taken_h = [False] * len(hyp)
    taken_r = [False] * len(ref)
    for n in range(max_n, 0, -1):
        total += spurious(hyp, n, taken_h) - spurious(ref, n, taken_r)
    return max(0, total)


class RunStats:
    """Per-run aggregate counters plus per-utterance rows for the length analysis."""

    def __init__(self, label: str):
        self.label = label
        self.ins = self.dele = self.sub = self.ref_words = self.hyp_words = 0
        self.n = self.empty = 0
        self.rep = 0
        self.err_pos_sum = 0.0
        self.err_pos_n = 0
        self.rows: List[dict] = []  # {key, ref_len, errors, ins, del, sub, ref, hyp}
        self.by_key: Dict[str, List[int]] = defaultdict(lambda: [0, 0])  # key -> [errors, ref_words]

    @property
    def wer(self) -> float:
        return 100.0 * (self.ins + self.dele + self.sub) / max(1, self.ref_words)

    def add(self, key: str, ref: List[str], hyp: List[str]) -> None:
        pairs = align(ref, hyp)
        i = d = s = 0
        pos_sum, pos_n, seen_ref = 0.0, 0, 0
        for r, h in pairs:
            if r == EPS:
                i += 1
            elif h == EPS:
                d += 1
                seen_ref += 1
            elif r != h:
                s += 1
                seen_ref += 1
            else:
                seen_ref += 1
                continue
            if ref:
                pos_sum += seen_ref / len(ref)
                pos_n += 1

        self.ins += i
        self.dele += d
        self.sub += s
        self.ref_words += len(ref)
        self.hyp_words += len(hyp)
        self.n += 1
        self.empty += 1 if not hyp else 0
        self.rep += repeated_run_words(hyp, ref)
        self.err_pos_sum += pos_sum
        self.err_pos_n += pos_n
        errs = i + d + s
        self.rows.append(
            {
                "key": key,
                "ref_len": len(ref),
                "errors": errs,
                "ins": i,
                "del": d,
                "sub": s,
                "ref": " ".join(ref),
                "hyp": " ".join(hyp),
            }
        )
        self.by_key[key][0] += errs
        self.by_key[key][1] += len(ref)


def leaderboard_wer(refs: List[str], hyps: List[str]) -> Optional[float]:
    """The exact scorer the aggregate uses, for cross-checking the headline number."""
    try:
        from leaderboard_wer import WER

        w = WER(normalize=False)  # already normalized upstream
        w.update("all", refs=refs, hyps=hyps)
        return float(w.compute()["wer"]) * 100.0
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] leaderboard_wer unavailable ({type(e).__name__}: {e}); using raw edit distance")
        return None


def parse_aggregate_log(root: str) -> Dict[str, float]:
    """Pull ``RESULT\\t<key>\\t<wer>`` rows out of an aggregate.log, if one exists."""
    out: Dict[str, float] = {}
    for path in glob.glob(os.path.join(root, "**", "aggregate.log"), recursive=True):
        with open(path, errors="replace") as f:
            for line in f:
                m = re.match(r"RESULT\t([^\t]+)\t([0-9.]+)", line)
                if m:
                    out[m.group(1)] = float(m.group(2))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def bucket_by_ref_len(rows: List[dict], n_buckets: int = 4) -> List[Tuple[str, float, int]]:
    """WER within equal-count buckets of reference length.

    A model whose text history drifts gets monotonically worse as utterances get
    longer; a per-chunk acoustic problem stays flat. That contrast is the point.
    """
    usable = sorted((r for r in rows if r["ref_len"] > 0), key=lambda r: r["ref_len"])
    if not usable:
        return []
    out, size = [], max(1, len(usable) // n_buckets)
    for b in range(n_buckets):
        lo = b * size
        hi = len(usable) if b == n_buckets - 1 else min(len(usable), lo + size)
        chunk = usable[lo:hi]
        if not chunk:
            continue
        errs = sum(r["errors"] for r in chunk)
        words = sum(r["ref_len"] for r in chunk)
        label = f"{chunk[0]['ref_len']}-{chunk[-1]['ref_len']}w"
        out.append((label, 100.0 * errs / max(1, words), len(chunk)))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="label=DIR pairs (DIR may be the eval dir or a parent)")
    p.add_argument("--no-normalize", action="store_true", help="skip the leaderboard normalizer")
    p.add_argument("--per-dataset", action="store_true", help="also print a per-dataset WER table")
    p.add_argument("--buckets", type=int, default=4, help="reference-length buckets (default 4)")
    p.add_argument("--examples", type=int, default=0, help="dump this many worst utterances per run")
    p.add_argument("--common-only", action="store_true", help="score only utterances present in EVERY run")
    args = p.parse_args()

    parsed = []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"ERROR: expected label=DIR, got {spec!r}")
        label, root = spec.split("=", 1)
        parsed.append((label, root))

    norm = get_normalizer(not args.no_normalize)

    # ---- load + normalize ----
    runs: Dict[str, Dict[str, dict]] = {}
    reported: Dict[str, Dict[str, float]] = {}
    for label, root in parsed:
        print(f"==> {label}: {root}")
        recs = load_run(root)
        table = {}
        for r in recs:
            ref = norm(r.get("reference", "") or "")
            hyp = norm(r.get("hypothesis", "") or "")
            table[f"{r.get('key','')}||{ref}"] = {"key": r.get("key", ""), "ref": ref, "hyp": hyp}
        runs[label] = table
        reported[label] = parse_aggregate_log(root)
        print(f"    {len(table)} utterances")

    # ---- coverage check: a WER gap is only meaningful over the same utterances ----
    keysets = {lab: set(t) for lab, t in runs.items()}
    common = set.intersection(*keysets.values()) if keysets else set()
    union = set.union(*keysets.values()) if keysets else set()
    print()
    print(f"==> coverage: {len(common)} utterances common to all runs, {len(union)} in the union")
    ragged = False
    for lab in runs:
        missing = len(union) - len(keysets[lab])
        if missing:
            ragged = True
            print(f"    [warn] {lab} is missing {missing} utterance(s) the other runs have")
    if ragged and not args.common_only:
        print("    -> rerun with --common-only to compare on the identical utterance set")

    scored_keys = common if args.common_only else None

    # ---- per-run stats ----
    stats: Dict[str, RunStats] = {}
    for lab, table in runs.items():
        st = RunStats(lab)
        refs, hyps = [], []
        for uid, rec in table.items():
            if scored_keys is not None and uid not in scored_keys:
                continue
            st.add(rec["key"], rec["ref"].split(), rec["hyp"].split())
            refs.append(rec["ref"])
            hyps.append(rec["hyp"])
        st.official = leaderboard_wer(refs, hyps)
        stats[lab] = st

    # ---- headline verification ----
    print()
    print("=" * 108)
    print("VERIFICATION -- recomputed from the raw generations")
    print("=" * 108)
    print(f"  {'run':<20} {'N':>7} {'WER%':>8} {'kaldialign%':>12} {'aggregate.log%':>15}  {'agree':>6}")
    print("  " + "-" * 104)
    for lab in runs:
        st = stats[lab]
        off = st.official
        rep = reported[lab].get("Average")
        agree = "-"
        if off is not None and rep is not None:
            agree = "yes" if abs(off - rep) < 0.15 else "NO"
        print(
            f"  {lab:<20} {st.n:>7} {st.wer:>8.2f} "
            f"{('%.2f' % off) if off is not None else '-':>12} "
            f"{('%.2f' % rep) if rep is not None else '-':>15}  {agree:>6}"
        )
    print()
    print("  WER% is a plain edit distance; kaldialign% adds compound merging and is the")
    print("  number the leaderboard reports. 'agree' compares it with the run's aggregate.log")
    print("  -- a 'NO' means the log is stale or was produced from different generations.")

    # ---- error decomposition ----
    print()
    print("=" * 108)
    print("ERROR DECOMPOSITION -- all rates are % of reference words")
    print("=" * 108)
    print(
        f"  {'run':<20} {'WER%':>7} {'sub%':>7} {'del%':>7} {'ins%':>7} "
        f"{'rep%':>7} {'len_ratio':>10} {'empty%':>8} {'err_pos':>8}"
    )
    print("  " + "-" * 104)
    for lab in runs:
        st = stats[lab]
        rw = max(1, st.ref_words)
        print(
            f"  {lab:<20} {st.wer:>7.2f} {100.0*st.sub/rw:>7.2f} {100.0*st.dele/rw:>7.2f} "
            f"{100.0*st.ins/rw:>7.2f} {100.0*st.rep/rw:>7.2f} {st.hyp_words/rw:>10.3f} "
            f"{100.0*st.empty/max(1,st.n):>8.2f} "
            f"{(st.err_pos_sum/st.err_pos_n if st.err_pos_n else 0.0):>8.3f}"
        )
    print()
    print("  rep%     = words in an n-gram repeated back-to-back more often than the reference")
    print("             does (spurious looping / re-emission), n<=4.")
    print("  len_ratio= hypothesis words / reference words. <1 means words are being dropped.")
    print("  err_pos  = mean position of an error within the utterance, 0=start 1=end.")
    print("             0.5 is uniform; well above 0.5 means errors pile up toward the end.")

    # ---- drift check ----
    print()
    print("=" * 108)
    print("WER BY REFERENCE LENGTH -- rising with length => error propagation / drift")
    print("=" * 108)
    for lab in runs:
        buckets = bucket_by_ref_len(stats[lab].rows, args.buckets)
        if not buckets:
            continue
        cells = "  ".join(f"{name:>12}: {wer:6.2f} (n={n})" for name, wer, n in buckets)
        trend = ""
        if len(buckets) >= 2 and buckets[0][1] > 0:
            trend = f"   last/first = {buckets[-1][1] / buckets[0][1]:.2f}x"
        print(f"  {lab:<20} {cells}{trend}")

    # ---- per-dataset ----
    if args.per_dataset:
        print()
        print("=" * 108)
        print("PER-DATASET WER%")
        print("=" * 108)
        all_keys = sorted({k for st in stats.values() for k in st.by_key})
        print(f"  {'dataset':<34}" + "".join(f"{lab:>16}" for lab in runs))
        print("  " + "-" * 104)
        for key in all_keys:
            cells = ""
            for lab in runs:
                e, w = stats[lab].by_key.get(key, [0, 0])
                cells += f"{(100.0*e/w if w else float('nan')):>16.2f}"
            print(f"  {key:<34}{cells}")

    # ---- worst examples ----
    if args.examples:
        for lab in runs:
            print()
            print("=" * 108)
            print(f"WORST {args.examples} UTTERANCES -- {lab}  (ranked by errors, ties by ref length)")
            print("=" * 108)
            worst = sorted(stats[lab].rows, key=lambda r: (-r["errors"], -r["ref_len"]))[: args.examples]
            for r in worst:
                print(
                    f"  [{r['key']}] ref_len={r['ref_len']} errors={r['errors']} "
                    f"(sub={r['sub']} del={r['del']} ins={r['ins']})"
                )
                print(f"    REF: {r['ref']}")
                print(f"    HYP: {r['hyp']}")
                print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
