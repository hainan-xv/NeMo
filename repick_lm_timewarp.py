#!/usr/bin/env python3
"""
Re-pick time-warp hypotheses by **external language-model score**.

The expensive ASR decoding for every warp factor is already cached in JSONL by
``likelihood_timewarp_aishell.py`` / ``oracle_timewarp_leaderboard.py`` (and the
leaderboard ``eval_oracle_timewarp*`` scripts). Each record stores, per factor,
the predicted text. This script reloads those files, scores each candidate
transcript with a (lightweight) causal LM, and picks the highest-scoring warp --
no model / audio needed, so you can iterate on the idea in one GPU pass.

Score per candidate = mean token log-prob under the LM (length-normalized, i.e.
negative perplexity). Highest score wins. Optionally:
  * --epsilon : stay on x1.0 unless another factor beats it by >= epsilon
  * --alpha   : shallow fusion, final = lm_meanlp + alpha * acoustic_selection_score
                (acoustic score only available in likelihood_*.jsonl)

Works on both JSONL flavours:
  * likelihood_results_aishell/likelihood_*.jsonl   (Mandarin, char/CER)
  * oracle_results/oracle_*.jsonl                   (leaderboard, word/WER)
Unit (char vs word) is auto-detected from the per-factor metric key (cer/wer),
override with --unit.

Examples:
    # English leaderboard sets, distilgpt2 (word-level WER)
    python repick_lm_timewarp.py --lm_model distilgpt2 oracle_results/oracle_*.jsonl

    # Mandarin aishell, locally-cached Qwen3 (char-level CER)
    python repick_lm_timewarp.py --lm_model Qwen/Qwen3-1.7B \
        likelihood_results_aishell/likelihood_test_*.jsonl

    # Add epsilon bias toward x1.0
    python repick_lm_timewarp.py --lm_model distilgpt2 --epsilon 0.05 oracle_results/oracle_*.jsonl
"""
import argparse
import glob
import json
import os
from collections import Counter
from typing import Dict, List, Optional


# ── error rate (corpus-level, char or word units) ─────────────────────────────

def _edit_distance(hyp: List[str], ref: List[str]) -> int:
    n, m = len(hyp), len(ref)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        hi = hyp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if hi == ref[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def _tokenize(s: str, unit: str) -> List[str]:
    return list(s) if unit == "char" else s.split()


def corpus_err(hyps: List[str], refs: List[str], unit: str) -> float:
    edits = 0
    ref_units = 0
    for h, r in zip(hyps, refs):
        ht, rt = _tokenize(h, unit), _tokenize(r, unit)
        edits += _edit_distance(ht, rt)
        ref_units += len(rt)
    return 100.0 * edits / max(ref_units, 1)


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
    return sorted(record["per_factor"].keys(), key=lambda k: float(k))


def baseline_key(record: dict) -> str:
    for k in record["per_factor"]:
        if abs(float(k) - 1.0) < 1e-9:
            return k
    return min(record["per_factor"], key=lambda k: abs(float(k) - 1.0))


def detect_unit(records: List[dict]) -> str:
    pf = records[0]["per_factor"]
    any_factor = next(iter(pf.values()))
    if "cer" in any_factor:
        return "char"
    if "wer" in any_factor:
        return "word"
    # fall back: CJK chars present -> char
    sample = records[0].get("text_normalized", "")
    if any("\u4e00" <= c <= "\u9fff" for c in sample):
        return "char"
    return "word"


def pred_text(factor_entry: dict) -> str:
    return factor_entry.get("pred_text_normalized") or factor_entry.get("pred_text") or ""


def oracle_metric_key(records: List[dict]) -> Optional[str]:
    pf = records[0]["per_factor"]
    any_factor = next(iter(pf.values()))
    if "cer" in any_factor:
        return "cer"
    if "wer" in any_factor:
        return "wer"
    return None


def _tie_break(tied: List[str]) -> str:
    for k in tied:
        if abs(float(k) - 1.0) < 1e-9:
            return k
    return min(tied, key=lambda k: abs(float(k) - 1.0))


# ── LM scoring ─────────────────────────────────────────────────────────────────

class LMScorer:
    """Mean per-token log-prob of a string under a HF causal LM (higher = better)."""

    def __init__(self, model_name: str, device: int = 0, batch_size: int = 16,
                 dtype: str = "auto", max_length: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = f"cuda:{device}" if torch.cuda.is_available() and device >= 0 else "cpu"
        self.batch_size = batch_size
        self.max_length = max_length

        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or self.tok.unk_token
        td = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16,
              "fp32": torch.float32}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=td if td != "auto" else None)
        self.model.to(self.device).eval()

    def score_batch(self, texts: List[str]) -> List[float]:
        """Return mean log-prob per token for each text (empty -> very low score)."""
        torch = self.torch
        out: List[Optional[float]] = [None] * len(texts)
        idx_nonempty = [i for i, t in enumerate(texts) if t.strip()]
        for i in range(len(texts)):
            if not texts[i].strip():
                out[i] = -1e9
        for s in range(0, len(idx_nonempty), self.batch_size):
            chunk_idx = idx_nonempty[s:s + self.batch_size]
            chunk = [texts[i] for i in chunk_idx]
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_length)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            logp = torch.log_softmax(logits.float(), dim=-1)
            ids = enc["input_ids"]
            mask = enc["attention_mask"]
            # next-token prediction: logits[t] predicts ids[t+1]
            tgt = ids[:, 1:]
            lp = logp[:, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            m = mask[:, 1:].float()
            tok_logp = (lp * m).sum(dim=1)
            ntok = m.sum(dim=1).clamp(min=1.0)
            mean_lp = (tok_logp / ntok).tolist()
            for j, gi in enumerate(chunk_idx):
                out[gi] = float(mean_lp[j])
        return [o if o is not None else -1e9 for o in out]


# ── selection ──────────────────────────────────────────────────────────────────

def pick_lm(record: dict, lm_scores: Dict[str, float], epsilon: float,
            alpha: float) -> str:
    pf = record["per_factor"]
    fused = {}
    for k in pf:
        s = lm_scores[k]
        if alpha != 0.0 and "selection_score" in pf[k]:
            s = s + alpha * float(pf[k]["selection_score"])
        fused[k] = s
    base = baseline_key(record)
    base_s = fused[base]
    eligible = [k for k in pf if k == base or fused[k] - base_s >= epsilon]
    best = max(fused[k] for k in eligible)
    tied = [k for k in eligible if fused[k] == best]
    return _tie_break(tied)


def pick_oracle(record: dict, metric: str) -> str:
    pf = record["per_factor"]
    best = min(pf[k][metric] for k in pf)
    tied = [k for k in pf if pf[k][metric] == best]
    return _tie_break(tied)


# ── evaluation ──────────────────────────────────────────────────────────────────

def evaluate_set(records: List[dict], scorer: LMScorer, unit: str,
                 metric: Optional[str], epsilon: float, alpha: float) -> dict:
    refs = [r["text_normalized"] for r in records]
    fkeys = factor_keys(records[0])
    base = baseline_key(records[0])

    # Flatten all candidate texts for one batched LM pass.
    flat_texts: List[str] = []
    span = []  # (record_idx -> {factor: pos})
    for r in records:
        pos = {}
        for k in fkeys:
            pos[k] = len(flat_texts)
            flat_texts.append(pred_text(r["per_factor"][k]))
        span.append(pos)
    flat_scores = scorer.score_batch(flat_texts)

    selected_preds, oracle_preds = [], []
    selected_factors, oracle_factors = [], []
    per_factor_preds: Dict[str, List[str]] = {k: [] for k in fkeys}

    for r, pos in zip(records, span):
        lm_scores = {k: flat_scores[pos[k]] for k in fkeys}
        sel = pick_lm(r, lm_scores, epsilon, alpha)
        selected_factors.append(sel)
        selected_preds.append(pred_text(r["per_factor"][sel]))
        if metric:
            orc = pick_oracle(r, metric)
            oracle_factors.append(orc)
            oracle_preds.append(pred_text(r["per_factor"][orc]))
        for k in fkeys:
            per_factor_preds[k].append(pred_text(r["per_factor"][k]))

    n = len(records)
    factor_err = {k: corpus_err(per_factor_preds[k], refs, unit) for k in fkeys}
    res = {
        "n": n,
        "fkeys": fkeys,
        "unit": unit,
        "baseline_key": base,
        "factor_err": factor_err,
        "best_fixed_key": min(factor_err, key=factor_err.get),
        "selected_err": corpus_err(selected_preds, refs, unit),
        "selected_counts": Counter(selected_factors),
    }
    if metric:
        res["oracle_err"] = corpus_err(oracle_preds, refs, unit)
        res["oracle_counts"] = Counter(oracle_factors)
        res["agreement"] = sum(s == o for s, o in zip(selected_factors, oracle_factors)) / n
    return res


def pick_dist_str(counts: Counter, fkeys: List[str], n: int) -> str:
    return ",".join(f"x{k}:{100 * counts.get(k, 0) / n:.1f}" for k in fkeys)


def print_summary(set_results: Dict[str, dict], lm_model: str, epsilon: float, alpha: float) -> None:
    any_res = next(iter(set_results.values()))
    fkeys = any_res["fkeys"]
    unit = any_res["unit"]
    metric_name = "CER" if unit == "char" else "WER"
    has_oracle = "oracle_err" in any_res

    print()
    print("=" * 100)
    print(f"  LM RE-PICK TIME-WARP {metric_name}  |  lm={lm_model}  epsilon={epsilon}  alpha={alpha}")
    print("=" * 100)

    header = f"  {'set':<26} {'scored':>7}"
    for k in fkeys:
        header += f" {'x'+k:>8}"
    header += f" {'LM-SEL':>8}"
    if has_oracle:
        header += f" {'ORACLE':>8} {'agree%':>8}"
    header += "  selected_pick_%"
    print(header)

    sums = {k: 0.0 for k in fkeys}
    sum_sel = sum_orc = 0.0
    cnt = 0
    for set_name, res in set_results.items():
        row = f"  {set_name:<26} {res['n']:>7}"
        for k in fkeys:
            row += f" {res['factor_err'][k]:>8.2f}"
        row += f" {res['selected_err']:>8.2f}"
        if has_oracle:
            row += f" {res['oracle_err']:>8.2f} {100 * res['agreement']:>8.2f}"
        row += "  " + pick_dist_str(res["selected_counts"], fkeys, res["n"])
        print(row)
        for k in fkeys:
            sums[k] += res["factor_err"][k]
        sum_sel += res["selected_err"]
        sum_orc += res.get("oracle_err", 0.0)
        cnt += 1

    if cnt > 1:
        print("  " + "-" * 96)
        avg = f"  {'AVERAGE':<26} {'':>7}"
        for k in fkeys:
            avg += f" {sums[k] / cnt:>8.2f}"
        avg += f" {sum_sel / cnt:>8.2f}"
        if has_oracle:
            avg += f" {sum_orc / cnt:>8.2f}"
        print(avg)
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="JSONL files to re-pick.")
    ap.add_argument("--lm_model", default="distilgpt2",
                    help="HF causal LM id or local path (default distilgpt2).")
    ap.add_argument("--device", type=int, default=0, help="GPU id (-1 for CPU).")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--max_length", type=int, default=256, help="LM truncation length.")
    ap.add_argument("--epsilon", type=float, default=0.0,
                    help="Stay on x1.0 unless another factor's LM score beats it by >= epsilon.")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="Shallow fusion weight on acoustic selection_score (likelihood jsonl only).")
    ap.add_argument("--unit", default="auto", choices=["auto", "char", "word"])
    ap.add_argument("--max_records", type=int, default=None, help="Cap records per file (debug).")
    args = ap.parse_args()

    files = args.files or (sorted(glob.glob("oracle_results/oracle_*.jsonl"))
                           or sorted(glob.glob("likelihood_results_aishell/likelihood_test_*.jsonl")))
    if not files:
        ap.error("no JSONL files found; pass paths explicitly.")

    set_records: Dict[str, List[dict]] = {}
    for path in files:
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        recs = load_jsonl(path)
        if args.max_records:
            recs = recs[:args.max_records]
        tag = os.path.basename(path).replace(".jsonl", "")
        if recs:
            set_records[tag] = recs
        else:
            print(f"WARNING: {path} had no records, skipping")
    if not set_records:
        ap.error("no usable records loaded.")

    print(f"==> loading LM: {args.lm_model} (device={args.device}, dtype={args.dtype})")
    scorer = LMScorer(args.lm_model, device=args.device, batch_size=args.batch_size,
                      dtype=args.dtype, max_length=args.max_length)

    set_results = {}
    for name, recs in set_records.items():
        unit = detect_unit(recs) if args.unit == "auto" else args.unit
        metric = oracle_metric_key(recs)
        print(f"==> scoring {name}  (n={len(recs)}, unit={unit})")
        set_results[name] = evaluate_set(recs, scorer, unit, metric, args.epsilon, args.alpha)

    print_summary(set_results, args.lm_model, args.epsilon, args.alpha)


if __name__ == "__main__":
    main()
