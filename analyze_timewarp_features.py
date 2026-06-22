#!/usr/bin/env python3
"""
Analyze per-utterance features to find the best discriminator for time-warp factor selection.

For each utterance we have three factor hypotheses (x0.9, x1.0, x1.1).
We want to know: which feature best identifies the oracle (lowest-CER) hypothesis?

Usage:
    python analyze_timewarp_features.py likelihood_results_aishell/likelihood_test_*.jsonl
"""

import sys
import json
import math
import collections
import glob
from typing import List, Dict, Tuple, Optional


# ── helpers ─────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("{"):
                records.append(json.loads(line))
    return records


def best_factors_by_cer(per_factor: dict) -> List[str]:
    """Return the factor(s) with the lowest CER (may be a tie)."""
    min_cer = min(v["cer"] for v in per_factor.values())
    return [k for k, v in per_factor.items() if v["cer"] == min_cer]


def edit_distance(a: str, b: str) -> int:
    """Simple character edit distance."""
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (0 if a[i-1] == b[j-1] else 1))
    return d[m]


# ── feature extraction ───────────────────────────────────────────────────────

def extract_features(record: dict) -> Dict[str, Dict[str, float]]:
    """
    For each factor return a dict of candidate selection features (higher = prefer).
    """
    pf = record["per_factor"]
    factors = list(pf.keys())
    duration = record.get("duration", 1.0) or 1.0

    # factor_numeric: "0.9" → 0.9, "x0.9" → 0.9
    def _fnum(fk: str) -> float:
        return float(fk.lstrip("x"))

    feats_by_factor: Dict[str, Dict[str, float]] = {}
    for fk in factors:
        v    = pf[fk]
        raw  = v["raw_score"]
        sel  = v["selection_score"]       # logprob/tok: mean log-softmax per emitted token
        text = v["pred_text_normalized"] or ""
        clen = max(len(text), 1)
        fnum = _fnum(fk)

        total_lp = sel * clen             # total log-prob proxy

        feats_by_factor[fk] = {
            "selection_score":   sel,
            "raw_score":         raw,
            "raw_per_char":      raw / clen,
            "raw_per_dur":       raw / duration,
            "total_logprob":     total_lp,
            "char_len":          float(clen),
            "neg_char_len":      float(-clen),
            "char_per_dur":      clen / duration,
            # Duration-adjusted: normalize total logprob by WARPED audio duration.
            # warped_dur = original_dur / factor => total_lp / warped_dur = total_lp * factor / duration
            # For comparing same utterance across factors, this is ∝ total_lp * factor.
            # Since total_lp is negative, ×0.9 < ×1.0 < ×1.1 in magnitude,
            # so this penalises x1.1 relative to x0.9.
            "lp_per_warped_dur": total_lp * fnum / duration,
            # Partial length penalty (between logprob/tok and total_logprob):
            "lp_sqrt_char":      sel * (clen ** 0.5),
            "lp_cbrt_char":      sel * (clen ** (1.0/3.0)),
        }

    # Majority-vote feature: pick the factor whose predicted text is most central
    # (closest in edit distance to the other two).
    texts = {fk: (pf[fk]["pred_text_normalized"] or "") for fk in factors}
    for fk in factors:
        others = [texts[ok] for ok in factors if ok != fk]
        avg_dist = sum(edit_distance(texts[fk], ot) for ot in others) / max(len(others), 1)
        feats_by_factor[fk]["neg_avg_edit_dist"] = -avg_dist   # higher = more central

    # Similarity to x1.0 baseline text
    base_key = "1.0" if "1.0" in pf else "x1.0"
    base_text = texts.get(base_key, "")
    for fk in factors:
        ed = edit_distance(texts[fk], base_text)
        norm = max(len(base_text), 1)
        feats_by_factor[fk]["neg_edit_to_1_0"] = -ed / norm

    return feats_by_factor


# ── evaluation utilities ─────────────────────────────────────────────────────

class AccStats:
    """Track accuracy with proper tie handling:
    - A pick is 'correct' only when it uniquely resolves to an oracle factor.
    - When all candidates tie, the utterance is counted as a random pick
      (credit = 1/n_tied if oracle is in tie set, else 0).
    """
    def __init__(self):
        self.correct = 0.0
        self.total   = 0

    def add(self, oracle_set: set, picked_set: set):
        self.total += 1
        if len(picked_set) == 1:
            fk = next(iter(picked_set))
            self.correct += 1.0 if fk in oracle_set else 0.0
        else:
            # Tie: credit proportional to oracle overlap
            self.correct += len(picked_set & oracle_set) / len(picked_set)

    @property
    def acc(self) -> float:
        return self.correct / self.total if self.total else 0.0


def evaluate_feature(records: List[dict], feat_name: str):
    """
    Returns dict with keys:
      acc_all, acc_disc, acc_disc_nontie, cer_corr, cer_delta_vs_1_0
    """
    all_acc    = AccStats()
    disc_acc   = AccStats()
    disc_nt_acc = AccStats()   # discriminative + feature non-tied

    agree = 0; pairs = 0       # for CER rank correlation
    cer_deltas = []

    for rec in records:
        pf = rec["per_factor"]
        cers = {k: pf[k]["cer"] for k in pf}
        discriminative = len(set(cers.values())) > 1
        oracle = set(best_factors_by_cer(pf))
        feats  = extract_features(rec)

        best_val = max(feats[fk][feat_name] for fk in feats)
        picked   = {fk for fk in feats if feats[fk][feat_name] == best_val}
        non_tied_feature = len(picked) == 1

        all_acc.add(oracle, picked)
        if discriminative:
            disc_acc.add(oracle, picked)
            if non_tied_feature:
                disc_nt_acc.add(oracle, picked)

        # CER rank correlation: for each pair of factors check if feature rank
        # agrees with CER rank (higher feature ↔ lower CER)
        factor_list = list(pf.keys())
        for i in range(len(factor_list)):
            for j in range(i + 1, len(factor_list)):
                fi, fj = factor_list[i], factor_list[j]
                if cers[fi] == cers[fj] or feats[fi][feat_name] == feats[fj][feat_name]:
                    continue
                pairs += 1
                if (feats[fi][feat_name] > feats[fj][feat_name]) == (cers[fi] < cers[fj]):
                    agree += 1

        # CER delta: compare actually-selected (single best) vs 1.0 baseline
        baseline_cer = cers.get("1.0", cers.get("x1.0", None))
        if baseline_cer is None:
            continue
        picked_list = sorted(picked)           # deterministic tie-break
        picked_cer  = cers[picked_list[0]]
        cer_deltas.append(baseline_cer - picked_cer)

    cer_corr  = agree / pairs if pairs else 0.0
    cer_delta = sum(cer_deltas) / len(cer_deltas) if cer_deltas else 0.0
    return {
        "acc_all":          all_acc.acc,
        "acc_disc":         disc_acc.acc,
        "acc_disc_nontie":  disc_nt_acc.acc,
        "n_disc_nontie":    disc_nt_acc.total,
        "cer_corr":         cer_corr,
        "cer_delta":        cer_delta,
    }


def factor_pick_dist(records: List[dict], feat_name: str) -> Dict[str, float]:
    counts = collections.Counter()
    for rec in records:
        feats = extract_features(rec)
        best_val = max(feats[fk][feat_name] for fk in feats)
        picked   = [fk for fk in feats if feats[fk][feat_name] == best_val]
        for p in picked:
            counts[p] += 1.0 / len(picked)
    total = sum(counts.values())
    return {k: v / total * 100 for k, v in sorted(counts.items())}


def oracle_dist(records: List[dict]) -> Dict[str, float]:
    counts = collections.Counter()
    for r in records:
        for f in best_factors_by_cer(r["per_factor"]):
            counts[f] += 1.0 / len(best_factors_by_cer(r["per_factor"]))
    total = sum(counts.values())
    return {k: v / total * 100 for k, v in sorted(counts.items())}


# ── main ─────────────────────────────────────────────────────────────────────

FEATURES = [
    ("selection_score",   "logprob/tok  (current method)"),
    ("total_logprob",     "logprob/tok × char_len  (total logprob)"),
    ("lp_per_warped_dur", "total_logprob / warped_dur  (dur-adjusted)"),
    ("lp_sqrt_char",      "logprob/tok × sqrt(char_len)"),
    ("lp_cbrt_char",      "logprob/tok × cbrt(char_len)"),
    ("raw_score",         "raw cumul logit sum"),
    ("raw_per_char",      "raw / char_len"),
    ("raw_per_dur",       "raw / audio_duration"),
    ("char_len",          "+char_len  (longer → better?)"),
    ("neg_char_len",      "−char_len  (shorter → better?)"),
    ("char_per_dur",      "chars/sec  (speech rate match?)"),
    ("neg_avg_edit_dist", "majority vote (neg avg edit dist to others)"),
    ("neg_edit_to_1_0",   "similarity to x1.0 text  (neg edit dist)"),
]


def main(paths: List[str]) -> None:
    all_records: List[dict] = []
    set_records: Dict[str, List[dict]] = {}
    for path in paths:
        tag  = path.split("/")[-1].replace("likelihood_", "").replace(".jsonl", "")
        recs = load_jsonl(path)
        set_records[tag] = recs
        all_records.extend(recs)

    n_total  = len(all_records)
    n_disc   = sum(1 for r in all_records
                   if len(set(r["per_factor"][k]["cer"] for k in r["per_factor"])) > 1)

    print(f"\n{'='*76}")
    print(f"  TIME-WARP FACTOR SELECTION FEATURE ANALYSIS")
    print(f"  {n_total} utterances  |  {n_disc} discriminative ({n_disc/n_total*100:.1f}%)")
    print(f"{'='*76}")

    # ── disagreement stats per set
    print("\n── Disagreement statistics ──────────────────────────────────────────────")
    for tag, recs in set_records.items():
        disc_recs = [r for r in recs if len(set(r["per_factor"][k]["cer"] for k in r["per_factor"])) > 1]
        n = len(disc_recs)
        if not n:
            continue
        spreads     = [max(r["per_factor"][k]["cer"] for k in r["per_factor"])
                       - min(r["per_factor"][k]["cer"] for k in r["per_factor"]) for r in disc_recs]
        base_key    = "1.0" if "1.0" in disc_recs[0]["per_factor"] else "x1.0"
        oracle_gains = [r["per_factor"][base_key]["cer"]
                        - min(r["per_factor"][k]["cer"] for k in r["per_factor"]) for r in disc_recs]
        print(f"  [{tag}]  disc={n}/{len(recs)}={n/len(recs)*100:.1f}%  "
              f"mean_spread={sum(spreads)/n:.4f}  mean_oracle_gain_vs_1.0={sum(oracle_gains)/n:.4f}")

    # ── feature accuracy table
    print("\n── Feature accuracy ─────────────────────────────────────────────────────")
    print(f"\n  Note: 'disc_nontie' = discriminative utterances where feature value is UNIQUE")
    print(f"        (not all tied) — this is the acid test for discriminative power.\n")

    hdr = (f"  {'Feature':<42} {'acc_all':>7} {'acc_disc':>8} "
           f"{'acc_disc_nt':>11} {'n_disc_nt':>9} {'cer_corr':>8} {'CER_Δ':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    results = []
    for feat_name, feat_desc in FEATURES:
        s = evaluate_feature(all_records, feat_name)
        results.append((feat_name, feat_desc, s))

    results.sort(key=lambda x: -x[2]["acc_disc_nontie"])

    for feat_name, feat_desc, s in results:
        print(f"  {feat_desc:<42} "
              f"{s['acc_all']*100:6.2f}% "
              f"{s['acc_disc']*100:7.2f}% "
              f"{s['acc_disc_nontie']*100:10.2f}% "
              f"{s['n_disc_nontie']:9d} "
              f"{s['cer_corr']*100:7.2f}% "
              f"{s['cer_delta']*100:+7.4f}pp")

    print(f"\n  ORACLE ceiling:  acc_disc=100%  acc_disc_nt=100%  CER_Δ=+"
          f"{_oracle_cer_delta(all_records)*100:.4f}pp")

    # ── factor pick distributions
    print("\n── Factor-pick distributions ────────────────────────────────────────────")
    factor_keys = sorted(all_records[0]["per_factor"].keys())
    print(f"\n  {'Feature':<42} " + "  ".join(f"{k:>6}" for k in factor_keys))
    print("  " + "-" * (44 + 9 * len(factor_keys)))
    for feat_name, feat_desc, _ in results[:5]:
        dist = factor_pick_dist(all_records, feat_name)
        print(f"  {feat_desc:<42} " + "  ".join(f"{dist.get(k,0):6.1f}%" for k in factor_keys))
    od = oracle_dist(all_records)
    print(f"\n  {'Oracle factor dist':<42} " + "  ".join(f"{od.get(k,0):6.1f}%" for k in factor_keys))

    # ── deep-dive: logprob_token miss analysis on discriminative non-tie utterances
    print("\n── Deep-dive: logprob_token miss analysis (disc, feature non-tied) ──────")
    feat_name = "selection_score"

    miss_recs = []
    for rec in all_records:
        pf   = rec["per_factor"]
        cers = {k: pf[k]["cer"] for k in pf}
        if len(set(cers.values())) <= 1:
            continue
        feats    = extract_features(rec)
        vals     = {fk: feats[fk][feat_name] for fk in feats}
        best_val = max(vals.values())
        if len([fk for fk in vals if vals[fk] == best_val]) > 1:
            continue   # feature tied → skip
        oracle   = set(best_factors_by_cer(pf))
        picked   = {fk for fk in vals if vals[fk] == best_val}
        if picked & oracle:
            continue   # correct pick
        pik = next(iter(picked))
        ork = sorted(oracle)[0]
        miss_recs.append(dict(rec=rec, oracle=ork, picked=pik,
                              cers=cers, feats=feats, pf=pf))

    n_disc_nt = sum(1 for rec in all_records
                    if (len(set(rec["per_factor"][k]["cer"] for k in rec["per_factor"])) > 1)
                    and len({fk for fk in (extract_features(rec))
                              if extract_features(rec)[fk][feat_name]
                                 == max(extract_features(rec)[fk2][feat_name] for fk2 in extract_features(rec))}) == 1)

    print(f"\n  Total misses: {len(miss_recs)} / {n_disc_nt} disc-nontie = {len(miss_recs)/max(n_disc_nt,1)*100:.1f}%")

    # Miss direction frequencies
    miss_dir = collections.Counter((m["oracle"], m["picked"]) for m in miss_recs)
    print("\n  Miss direction frequencies:")
    for (ork, pik), cnt in sorted(miss_dir.items(), key=lambda x: -x[1]):
        print(f"    oracle={ork:<4}  picked={pik:<4}  {cnt:5d}  ({cnt/len(miss_recs)*100:.1f}%)")

    # In miss cases: what do other features look like?
    print("\n  Mean feature value (oracle factor vs wrongly-picked factor) in miss cases:")
    print(f"  {'Feature':<42} {'mean(oracle)':>13} {'mean(picked)':>13} {'gap':>10}")
    print("  " + "-" * 80)
    for fn, fd in FEATURES:
        ov = [m["feats"][m["oracle"]][fn] for m in miss_recs]
        pv = [m["feats"][m["picked"]][fn] for m in miss_recs]
        mo, mp = sum(ov)/len(ov), sum(pv)/len(pv)
        flag = " <-- FLIP" if (mo > mp) else ""   # feature says wrong direction
        print(f"  {fd:<42} {mo:13.5f} {mp:13.5f} {mo-mp:10.5f}{flag}")

    # In miss cases: would majority vote have helped?
    mv_correct = sum(1 for m in miss_recs
                     if m["feats"][m["oracle"]]["neg_avg_edit_dist"]
                        > m["feats"][m["picked"]]["neg_avg_edit_dist"])
    print(f"\n  Majority vote corrects {mv_correct}/{len(miss_recs)} = {mv_correct/len(miss_recs)*100:.1f}% "
          f"of logprob_token misses")

    # ── per-oracle-category accuracy for logprob_token
    print("\n── logprob_token accuracy broken down by oracle factor ──────────────────")
    _per_oracle_accuracy(all_records, "selection_score")

    # ── combined feature: logprob_token + majority vote boost
    print("\n── Combined feature: logprob_token × majority-vote agreement ────────────")
    _combined_accuracy(all_records)


def _per_oracle_accuracy(records: List[dict], feat_name: str) -> None:
    """Show per-oracle-category: how often does the feature pick correctly?"""
    cat_stats: Dict[str, AccStats] = collections.defaultdict(AccStats)
    cat_feat_vals: Dict[str, Dict[str, list]] = collections.defaultdict(lambda: {"oracle": [], "other": []})

    for rec in records:
        pf   = rec["per_factor"]
        cers = {k: pf[k]["cer"] for k in pf}
        if len(set(cers.values())) <= 1:
            continue
        oracle = set(best_factors_by_cer(pf))
        feats  = extract_features(rec)
        best_val = max(feats[fk][feat_name] for fk in feats)
        picked   = {fk for fk in feats if feats[fk][feat_name] == best_val}

        for ork in oracle:
            cat_stats[ork].add(oracle, picked)
            cat_feat_vals[ork]["oracle"].append(feats[ork][feat_name])
        for fk in pf:
            if fk not in oracle:
                cat_feat_vals[ork]["other"].append(feats[fk][feat_name])

    print(f"\n  {'Oracle factor':<14} {'n_disc':>7} {'acc_disc':>9}  "
          f"{'mean_feat(oracle)':>18}  {'mean_feat(other)':>16}  gap")
    print("  " + "-" * 75)
    for cat in sorted(cat_stats):
        st = cat_stats[cat]
        ov = cat_feat_vals[cat]["oracle"]
        other_v = cat_feat_vals[cat]["other"]
        mo = sum(ov)/len(ov) if ov else 0
        mo_other = sum(other_v)/len(other_v) if other_v else 0
        print(f"  {cat:<14} {st.total:>7} {st.acc*100:>8.2f}%  "
              f"{mo:>18.5f}  {mo_other:>16.5f}  {mo-mo_other:>7.5f}")


def _oracle_cer_delta(records: List[dict]) -> float:
    """CER delta of oracle vs 1.0 baseline."""
    deltas = []
    for rec in records:
        pf = rec["per_factor"]
        base_key = "1.0" if "1.0" in pf else "x1.0"
        if base_key not in pf:
            continue
        oracle_cer = min(pf[k]["cer"] for k in pf)
        deltas.append(pf[base_key]["cer"] - oracle_cer)
    return sum(deltas) / len(deltas) if deltas else 0.0


def _combined_accuracy(records: List[dict]) -> None:
    """
    Try combining logprob_token with majority-vote:
    1. If majority-vote consensus exists (2 factors agree on text), use it.
    2. Otherwise fall back to logprob_token.
    """
    all_acc  = AccStats()
    disc_acc = AccStats()
    all_acc_base  = AccStats()
    disc_acc_base = AccStats()
    cer_deltas = []
    cer_deltas_base = []

    for rec in records:
        pf    = rec["per_factor"]
        cers  = {k: pf[k]["cer"] for k in pf}
        oracle = set(best_factors_by_cer(pf))
        disc   = len(set(cers.values())) > 1
        feats  = extract_features(rec)
        factor_list = list(pf.keys())

        # Baseline: logprob_token
        sel_scores = {fk: feats[fk]["selection_score"] for fk in feats}
        best_sel   = max(sel_scores.values())
        picked_base = {fk for fk in feats if sel_scores[fk] == best_sel}

        # Majority vote: 2 factors agree?
        texts = {fk: (pf[fk]["pred_text_normalized"] or "") for fk in factor_list}
        majority = None
        for i in range(len(factor_list)):
            for j in range(i + 1, len(factor_list)):
                fi, fj = factor_list[i], factor_list[j]
                if texts[fi] == texts[fj]:
                    majority = {fi, fj}
                    break
            if majority:
                break

        # Combined: use majority if available, else logprob_token
        if majority:
            # Among majority-agreeing factors, pick by logprob_token
            best_sel_maj = max(sel_scores[fk] for fk in majority)
            picked_combined = {fk for fk in majority if sel_scores[fk] == best_sel_maj}
        else:
            picked_combined = picked_base

        all_acc.add(oracle, picked_combined)
        all_acc_base.add(oracle, picked_base)
        if disc:
            disc_acc.add(oracle, picked_combined)
            disc_acc_base.add(oracle, picked_base)

        base_key = "1.0" if "1.0" in pf else "x1.0"
        if base_key in pf:
            b_cer = pf[base_key]["cer"]
            p_combined = sorted(picked_combined)[0]
            p_base     = sorted(picked_base)[0]
            cer_deltas.append(b_cer - cers[p_combined])
            cer_deltas_base.append(b_cer - cers[p_base])

    print(f"\n  {'Method':<45} {'acc_all':>7} {'acc_disc':>8} {'CER_Δ':>8}")
    print("  " + "-" * 70)
    print(f"  {'logprob_token (baseline)':<45} "
          f"{all_acc_base.acc*100:6.2f}% "
          f"{disc_acc_base.acc*100:7.2f}% "
          f"{sum(cer_deltas_base)/len(cer_deltas_base)*100:+7.4f}pp")
    print(f"  {'majority_vote → logprob_token fallback':<45} "
          f"{all_acc.acc*100:6.2f}% "
          f"{disc_acc.acc*100:7.2f}% "
          f"{sum(cer_deltas)/len(cer_deltas)*100:+7.4f}pp")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        paths = sorted(glob.glob("likelihood_results_aishell/likelihood_test_*.jsonl"))
        if not paths:
            print("Usage: python analyze_timewarp_features.py <jsonl_file> ...")
            sys.exit(1)
    else:
        paths = sys.argv[1:]
    main(paths)
