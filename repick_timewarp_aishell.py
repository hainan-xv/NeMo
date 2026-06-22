#!/usr/bin/env python3
"""
Fast re-picking of time-warp hypotheses from cached JSONL scores.

The expensive part (decoding every warp factor + extracting per-token log-probs)
is already done by ``likelihood_timewarp_aishell.py`` and saved to
``likelihood_results_aishell/likelihood_<set>.jsonl``.  Each line stores, per
factor, the predicted text and selection score.

This script reloads those files and re-applies a *selection rule* without
touching the model, so you can iterate on picking strategies in seconds.

The default rule is "epsilon switching": stick with x1.0 unless another factor's
score beats x1.0 by at least ``--epsilon``.  Add your own rules in ``pick_factor``.

Usage:
    # Default epsilon picking on all three sets
    python repick_timewarp_aishell.py

    # Pick specific files / epsilon
    python repick_timewarp_aishell.py --epsilon 0.02 \
        likelihood_results_aishell/likelihood_test_android.jsonl

    # Sweep epsilon to find the best value
    python repick_timewarp_aishell.py --sweep 0,0.005,0.01,0.02,0.05,0.1
"""
import argparse
import glob
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ── CER (matches NeMo word_error_rate(use_cer=True): corpus-level char edits) ──

def _edit_distance(hyp: List[str], ref: List[str]) -> int:
    n, m = len(hyp), len(ref)
    if n == 0:
        return m
    if m == 0:
        return n
    prev_row = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            cur[j] = min(prev_row[j] + 1, cur[j - 1] + 1, prev_row[j - 1] + cost)
        prev_row = cur
    return prev_row[m]


def corpus_cer(hyps: List[str], refs: List[str]) -> float:
    """Corpus CER (%) = total char edits / total ref chars."""
    edits = 0
    ref_chars = 0
    for h, r in zip(hyps, refs):
        h_chars = list(h)
        r_chars = list(r)
        edits += _edit_distance(h_chars, r_chars)
        ref_chars += len(r_chars)
    return 100.0 * edits / max(ref_chars, 1)


# ── data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("{"):
                records.append(json.loads(line))
    return records


def factor_keys(record: dict) -> List[str]:
    """Sorted factor keys, e.g. ['0.9', '1.0', '1.1']."""
    return sorted(record["per_factor"].keys(), key=lambda k: float(k))


def baseline_key(record: dict) -> str:
    """The no-warp key (1.0)."""
    for k in record["per_factor"]:
        if abs(float(k) - 1.0) < 1e-9:
            return k
    # fall back to factor closest to 1.0
    return min(record["per_factor"], key=lambda k: abs(float(k) - 1.0))


# ── selection rules ───────────────────────────────────────────────────────────
#
# A picker takes a single record and returns the chosen factor key (string).
# Add new rules here and select them with --rule.

def pick_argmax(record: dict, **kw) -> str:
    """Highest selection_score wins (ties → nearest to 1.0, preferring 1.0)."""
    pf = record["per_factor"]
    best = max(pf[k]["selection_score"] for k in pf)
    tied = [k for k in pf if pf[k]["selection_score"] == best]
    return _tie_break(tied)


def pick_epsilon(record: dict, epsilon: float = 0.01, **kw) -> str:
    """Stay on x1.0 unless another factor beats it by >= epsilon."""
    pf = record["per_factor"]
    base = baseline_key(record)
    base_score = pf[base]["selection_score"]
    eligible = [k for k in pf if k == base or pf[k]["selection_score"] - base_score >= epsilon]
    best = max(pf[k]["selection_score"] for k in eligible)
    tied = [k for k in eligible if pf[k]["selection_score"] == best]
    return _tie_break(tied)


def _tie_break(tied: List[str]) -> str:
    """Prefer 1.0, else the factor nearest to 1.0."""
    for k in tied:
        if abs(float(k) - 1.0) < 1e-9:
            return k
    return min(tied, key=lambda k: abs(float(k) - 1.0))


RULES = {
    "argmax": pick_argmax,
    "epsilon": pick_epsilon,
}


# ── oracle (for diagnostics only) ─────────────────────────────────────────────

def pick_oracle(record: dict) -> str:
    """Lowest-CER factor (cheating). Ties → nearest to 1.0."""
    pf = record["per_factor"]
    best_cer = min(pf[k]["cer"] for k in pf)
    tied = [k for k in pf if pf[k]["cer"] == best_cer]
    return _tie_break(tied)


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_set(records: List[dict], picker, picker_kwargs: dict) -> dict:
    refs = [r["text_normalized"] for r in records]
    fkeys = factor_keys(records[0])

    selected_preds, oracle_preds = [], []
    selected_factors, oracle_factors = [], []
    per_factor_preds: Dict[str, List[str]] = {k: [] for k in fkeys}

    for r in records:
        pf = r["per_factor"]
        sel = picker(r, **picker_kwargs)
        orc = pick_oracle(r)
        selected_factors.append(sel)
        oracle_factors.append(orc)
        selected_preds.append(pf[sel]["pred_text_normalized"])
        oracle_preds.append(pf[orc]["pred_text_normalized"])
        for k in fkeys:
            per_factor_preds[k].append(pf[k]["pred_text_normalized"])

    n = len(records)
    base = baseline_key(records[0])
    factor_cer = {k: corpus_cer(per_factor_preds[k], refs) for k in fkeys}
    return {
        "n": n,
        "fkeys": fkeys,
        "baseline_key": base,
        "factor_cer": factor_cer,
        "best_fixed_key": min(factor_cer, key=factor_cer.get),
        "selected_cer": corpus_cer(selected_preds, refs),
        "oracle_cer": corpus_cer(oracle_preds, refs),
        "selected_counts": Counter(selected_factors),
        "oracle_counts": Counter(oracle_factors),
        "agreement": sum(s == o for s, o in zip(selected_factors, oracle_factors)) / n,
    }


def pick_dist_str(counts: Counter, fkeys: List[str], n: int) -> str:
    return ",".join(f"x{k}:{100 * counts.get(k, 0) / n:.1f}" for k in fkeys)


# ── reporting ─────────────────────────────────────────────────────────────────

def print_summary(set_results: Dict[str, dict], rule: str, picker_kwargs: dict) -> None:
    any_res = next(iter(set_results.values()))
    fkeys = any_res["fkeys"]

    kw_str = " ".join(f"{k}={v}" for k, v in picker_kwargs.items())
    print()
    print("=" * 92)
    print(f"  RE-PICK TIME-WARP CER  |  rule={rule}  {kw_str}".rstrip())
    print("=" * 92)

    header = f"  {'set':<22} {'scored':>7}"
    for k in fkeys:
        header += f" {'x'+k:>8}"
    header += f" {'SELECT':>8} {'ORACLE':>8} {'agree%':>8}  selected_pick_%"
    print(header)

    sums = {k: 0.0 for k in fkeys}
    sum_sel = sum_orc = 0.0
    cnt = 0
    for set_name, res in set_results.items():
        row = f"  {set_name:<22} {res['n']:>7}"
        for k in fkeys:
            row += f" {res['factor_cer'][k]:>8.2f}"
        row += (f" {res['selected_cer']:>8.2f} {res['oracle_cer']:>8.2f}"
                f" {100 * res['agreement']:>8.2f}  "
                f"{pick_dist_str(res['selected_counts'], fkeys, res['n'])}")
        print(row)
        for k in fkeys:
            sums[k] += res["factor_cer"][k]
        sum_sel += res["selected_cer"]
        sum_orc += res["oracle_cer"]
        cnt += 1

    if cnt > 1:
        sep = f"  {'----':<22} {'----':>7}"
        for _ in fkeys:
            sep += f" {'----':>8}"
        sep += f" {'----':>8} {'----':>8}"
        print(sep)
        avg = f"  {'AVERAGE':<22} {'':>7}"
        for k in fkeys:
            avg += f" {sums[k] / cnt:>8.2f}"
        avg += f" {sum_sel / cnt:>8.2f} {sum_orc / cnt:>8.2f}"
        print(avg)
    print("=" * 92)


def run_sweep(set_records: Dict[str, List[dict]], rule: str, eps_values: List[float]) -> None:
    """Compare several epsilon values side by side (average CER over sets)."""
    print()
    print("=" * 72)
    print(f"  EPSILON SWEEP  |  rule={rule}")
    print("=" * 72)
    print(f"  {'epsilon':>10} | " + " | ".join(f"{s:>14}" for s in set_records) + " |   AVG")
    print("  " + "-" * 68)

    for eps in eps_values:
        row_cers = []
        for set_name, recs in set_records.items():
            res = evaluate_set(recs, RULES[rule], {"epsilon": eps})
            row_cers.append(res["selected_cer"])
        avg = sum(row_cers) / len(row_cers)
        label = f"{eps:.4f}"
        print(f"  {label:>10} | " + " | ".join(f"{c:>14.4f}" for c in row_cers) + f" | {avg:.4f}")

    # Reference rows: baseline (x1.0 only) and oracle
    base_cers, orc_cers = [], []
    for set_name, recs in set_records.items():
        refs = [r["text_normalized"] for r in recs]
        bk = baseline_key(recs[0])
        base_cers.append(corpus_cer([r["per_factor"][bk]["pred_text_normalized"] for r in recs], refs))
        orc_cers.append(corpus_cer([r["per_factor"][pick_oracle(r)]["pred_text_normalized"] for r in recs], refs))
    print("  " + "-" * 68)
    print(f"  {'x1.0 base':>10} | " + " | ".join(f"{c:>14.4f}" for c in base_cers) + f" | {sum(base_cers)/len(base_cers):.4f}")
    print(f"  {'oracle':>10} | " + " | ".join(f"{c:>14.4f}" for c in orc_cers) + f" | {sum(orc_cers)/len(orc_cers):.4f}")
    print("=" * 72)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*",
                    help="JSONL files (default: likelihood_results_aishell/likelihood_test_*.jsonl)")
    ap.add_argument("--rule", default="epsilon", choices=list(RULES.keys()),
                    help="Selection rule (default: epsilon).")
    ap.add_argument("--epsilon", type=float, default=0.01,
                    help="Switch from x1.0 only if another factor's score beats it by >= epsilon (default 0.01).")
    ap.add_argument("--sweep", default=None,
                    help="Comma-separated epsilon values to compare side by side, e.g. 0,0.005,0.01,0.02,0.05.")
    args = ap.parse_args()

    files = args.files or sorted(glob.glob("likelihood_results_aishell/likelihood_test_*.jsonl"))
    if not files:
        ap.error("no JSONL files found; pass paths explicitly or run from the repo root.")

    set_records: Dict[str, List[dict]] = {}
    for path in files:
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        tag = os.path.basename(path).replace("likelihood_", "").replace(".jsonl", "")
        recs = load_jsonl(path)
        if recs:
            set_records[tag] = recs
        else:
            print(f"WARNING: {path} had no records, skipping")

    if not set_records:
        ap.error("no usable records loaded.")

    if args.sweep is not None:
        eps_values = [float(x) for x in args.sweep.split(",") if x.strip()]
        run_sweep(set_records, "epsilon", eps_values)
        return

    picker_kwargs = {"epsilon": args.epsilon} if args.rule == "epsilon" else {}
    set_results = {name: evaluate_set(recs, RULES[args.rule], picker_kwargs)
                   for name, recs in set_records.items()}
    print_summary(set_results, args.rule, picker_kwargs)


if __name__ == "__main__":
    main()
