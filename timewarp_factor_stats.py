#!/usr/bin/env python3
"""
Per-utterance time-warp factor statistics from a cached likelihood JSONL file.

For each utterance and each warp factor it prints:
    * total audio length   (warped duration = original_duration / factor for the
                            "speed" method, which is what these runs use)
    * output # tokens      (exact if the JSONL stores ``num_tokens``; otherwise
                            approximated by the normalized-text character count,
                            which for Mandarin is ~1 token/char)
    * total likelihood     (sum of per-token log-probs; exact if ``total_logprob``
                            is stored, else per_token_logprob * num_tokens)
    * per-token likelihood (the ``selection_score`` field = mean log-prob/token)
and marks which factor gets PICKED by the selection rule (epsilon switching).

Usage:
    python timewarp_factor_stats.py likelihood_results_aishell/likelihood_test_ios.jsonl
    python timewarp_factor_stats.py FILE --limit 30          # first 30 utterances
    python timewarp_factor_stats.py FILE --all               # every utterance
    python timewarp_factor_stats.py FILE --epsilon 0.0       # pure argmax pick
    python timewarp_factor_stats.py FILE --summary-only      # skip per-utterance
"""
import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional


def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("{"):
                records.append(json.loads(line))
    return records


def factor_keys(record: dict) -> List[str]:
    return sorted(record["per_factor"].keys(), key=lambda k: float(k))


def baseline_key(record: dict) -> str:
    for k in record["per_factor"]:
        if abs(float(k) - 1.0) < 1e-9:
            return k
    return min(record["per_factor"], key=lambda k: abs(float(k) - 1.0))


def warped_duration(orig_duration: float, factor: str, v: Optional[dict] = None) -> float:
    """Warped length = original / factor (factor>1 => shorter). Uses the stored
    ``warped_duration`` field if present."""
    if v is not None and v.get("warped_duration") is not None:
        return float(v["warped_duration"])
    f = float(factor)
    return orig_duration / f if f > 0 else orig_duration


def num_tokens(v: dict) -> int:
    """Exact token count if stored, else fall back to normalized char count."""
    if "num_tokens" in v and v["num_tokens"] is not None:
        return int(v["num_tokens"])
    return len(v.get("pred_text_normalized", "") or "")


def total_logprob(v: dict) -> float:
    """Total log-likelihood; exact if stored, else per-token * num_tokens."""
    if "total_logprob" in v and v["total_logprob"] is not None:
        return float(v["total_logprob"])
    return float(v["selection_score"]) * num_tokens(v)


def per_token_logprob(v: dict) -> float:
    return float(v["selection_score"])


def pick_factor(record: dict, epsilon: float) -> str:
    """Stay on x1.0 unless another factor's per-token score beats it by >= epsilon.
    epsilon=0 reduces to pure argmax. Ties prefer 1.0, then nearest to 1.0."""
    pf = record["per_factor"]
    base = baseline_key(record)
    base_score = pf[base]["selection_score"]
    if epsilon > 0.0:
        eligible = [k for k in pf if k == base or pf[k]["selection_score"] - base_score >= epsilon]
    else:
        eligible = list(pf.keys())
    best = max(pf[k]["selection_score"] for k in eligible)
    tied = [k for k in eligible if pf[k]["selection_score"] == best]
    for k in tied:
        if abs(float(k) - 1.0) < 1e-9:
            return k
    return min(tied, key=lambda k: abs(float(k) - 1.0))


def all_hyps_identical(record: dict) -> bool:
    """True if every warp factor produced the same normalized hypothesis."""
    texts = {(v.get("pred_text_normalized", "") or "") for v in record["per_factor"].values()}
    return len(texts) <= 1


def oracle_factor(record: dict) -> str:
    pf = record["per_factor"]
    best_cer = min(pf[k]["cer"] for k in pf)
    tied = [k for k in pf if pf[k]["cer"] == best_cer]
    for k in tied:
        if abs(float(k) - 1.0) < 1e-9:
            return k
    return min(tied, key=lambda k: abs(float(k) - 1.0))


def print_utterance(record: dict, idx: int, fkeys: List[str], picked: str, oracle: str,
                    approx_tokens: bool, show_tokens: bool = False) -> None:
    pf = record["per_factor"]
    dur = record.get("duration", 0.0) or 0.0
    ap = os.path.basename(record.get("audio_filepath", ""))
    tok_label = "tokens~" if approx_tokens else "tokens"

    print(f"\n[{idx}] {ap}   orig_dur={dur:.2f}s")
    print(f"     ref : {record.get('text_normalized','')}")
    print(f"     {'factor':>7} {'audio_s':>8} {tok_label:>8} {'total_LL':>11} "
          f"{'LL/tok':>9} {'CER':>7}  pick")
    for k in fkeys:
        v = pf[k]
        wdur = warped_duration(dur, k, v)
        nt = num_tokens(v)
        tll = total_logprob(v)
        ptl = per_token_logprob(v)
        cer = v.get("cer", float("nan"))
        marks = []
        if k == picked:
            marks.append("<PICK")
        if k == oracle:
            marks.append("ORACLE")
        mark = " ".join(marks)
        print(f"     x{float(k):<6.2f} {wdur:8.2f} {nt:8d} {tll:11.3f} "
              f"{ptl:9.4f} {cer:7.4f}  {mark}")
    for k in fkeys:
        v = pf[k]
        hyp = v.get("pred_text_normalized", "") or ""
        print(f"     x{float(k):<6.2f} hyp: {hyp}")

    if show_tokens and pf[fkeys[0]].get("token_logprobs") is not None:
        for k in fkeys:
            v = pf[k]
            tlp = v.get("token_logprobs") or []
            ttx = v.get("token_texts") or [""] * len(tlp)
            parts = [f"{t}:{lp:+.2f}" for t, lp in zip(ttx, tlp)]
            print(f"     x{float(k):<6.2f} tok: " + "  ".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", help="Input likelihood JSONL file.")
    ap.add_argument("--limit", "-n", type=int, default=20,
                    help="Max number of utterances to print (default 20). Use --all to override.")
    ap.add_argument("--all", action="store_true", help="Print every utterance.")
    ap.add_argument("--epsilon", type=float, default=0.01,
                    help="Switch from x1.0 only if another factor beats it by >= epsilon (default 0.01; 0 = pure argmax).")
    ap.add_argument("--summary-only", action="store_true", help="Skip per-utterance tables, print only the summary.")
    ap.add_argument("--include-identical", action="store_true",
                    help="Also show utterances where all factors produced the same hypothesis "
                         "(by default these are filtered out of the per-utterance view).")
    ap.add_argument("--tokens", action="store_true",
                    help="Also print the per-token (text:logprob) sequence for each factor "
                         "(requires token_logprobs in the JSONL).")
    args = ap.parse_args()

    if not os.path.exists(args.jsonl):
        ap.error(f"file not found: {args.jsonl}")

    records = load_jsonl(args.jsonl)
    if not records:
        ap.error("no records found in file.")

    fkeys = factor_keys(records[0])
    approx_tokens = "num_tokens" not in records[0]["per_factor"][fkeys[0]]

    print("=" * 78)
    print(f"  TIME-WARP FACTOR STATS  |  {os.path.basename(args.jsonl)}")
    print(f"  {len(records)} utterances  |  factors {['x'+k for k in fkeys]}  |  epsilon={args.epsilon}")
    if approx_tokens:
        print("  NOTE: token counts approximated by normalized char length "
              "(JSONL has no num_tokens).")
    print("=" * 78)

    # Optionally filter out utterances where all factors agree on the text.
    display_idx = list(range(len(records)))
    n_identical = sum(all_hyps_identical(r) for r in records)
    if not args.include_identical:
        display_idx = [i for i in display_idx if not all_hyps_identical(records[i])]
        print(f"  Filtered out {n_identical} utterances with identical hyps across all factors; "
              f"{len(display_idx)} differ (use --include-identical to show all).")
        print("=" * 78)

    # Per-utterance tables
    if not args.summary_only:
        n_show = len(display_idx) if args.all else min(args.limit, len(display_idx))
        for shown, i in enumerate(display_idx[:n_show]):
            rec = records[i]
            picked = pick_factor(rec, args.epsilon)
            oracle = oracle_factor(rec)
            print_utterance(rec, i, fkeys, picked, oracle, approx_tokens, args.tokens)
        if n_show < len(display_idx):
            print(f"\n  ... {len(display_idx) - n_show} more (differing) utterances "
                  f"(use --all or --limit N to see them).")

    # Aggregate summary
    pick_counts: Counter = Counter()
    oracle_counts: Counter = Counter()
    agg = {k: {"audio_s": 0.0, "tokens": 0, "total_ll": 0.0, "ll_per_tok_sum": 0.0, "n": 0}
           for k in fkeys}
    agree = 0
    for rec in records:
        picked = pick_factor(rec, args.epsilon)
        oracle = oracle_factor(rec)
        pick_counts[picked] += 1
        oracle_counts[oracle] += 1
        agree += (picked == oracle)
        dur = rec.get("duration", 0.0) or 0.0
        for k in fkeys:
            v = rec["per_factor"][k]
            agg[k]["audio_s"] += warped_duration(dur, k, v)
            agg[k]["tokens"] += num_tokens(v)
            agg[k]["total_ll"] += total_logprob(v)
            agg[k]["ll_per_tok_sum"] += per_token_logprob(v)
            agg[k]["n"] += 1

    n = len(records)
    tok_label = "tot_tok~" if approx_tokens else "tot_tok"
    print("\n" + "=" * 78)
    print("  AGGREGATE (summed / averaged over all utterances)")
    print("=" * 78)
    print(f"  {'factor':>7} {'tot_audio_s':>12} {tok_label:>10} {'tot_LL':>14} "
          f"{'mean_LL/tok':>12} {'pick%':>7} {'oracle%':>8}")
    for k in fkeys:
        a = agg[k]
        mean_ptl = a["ll_per_tok_sum"] / a["n"] if a["n"] else 0.0
        print(f"  x{float(k):<6.2f} {a['audio_s']:12.1f} {a['tokens']:10d} "
              f"{a['total_ll']:14.1f} {mean_ptl:12.4f} "
              f"{100*pick_counts.get(k,0)/n:7.1f} {100*oracle_counts.get(k,0)/n:8.1f}")
    print("-" * 78)
    print(f"  picked-vs-oracle agreement: {100*agree/n:.2f}%   "
          f"(epsilon={args.epsilon})")
    print("=" * 78)


if __name__ == "__main__":
    main()
