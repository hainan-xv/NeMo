"""
Oracle time-warp ("speed cheat") CER experiment for Mandarin ASR checkpoints.

Chinese sibling of ``oracle_timewarp_eval.py``.  For each utterance we synthesize
several time-WARPED copies of the audio (a list of tempo/speed factors), decode
every copy independently with the SAME model, record all hypotheses, and then --
in scoring only -- pick, per utterance, the warp that gives the lowest CER
against the reference.

This is deliberately CHEATING (it peeks at the reference to choose the warp), so
the resulting CER is an *oracle upper bound*: "if we could always pick the best
of N time-warps, how good could decoding get?".  Useful to gauge headroom on the
AISHELL test sets and to see whether fast/short utterances are rescued by
slowing them down.

Everything except the warping + oracle selection is reused from
``eval_aishell_cer.py`` (manifest read + audio-path remap, the per-architecture
transcribe dispatch, whitespace-only normalization, and CER) so the numbers are
directly comparable to a normal ``eval_aishell.sh`` run.  ``warp_audio`` itself
is shared with the English ``oracle_timewarp_eval.py``.

Input: a local NeMo-style JSON manifest (one object per line) with at least
``audio_filepath`` and ``text``; ``duration`` is used for batching if present.

Examples
--------
python oracle_timewarp_aishell.py \
    --model /checkpoints/.../model-averaged.nemo \
    --manifest /.../aishell/manifests/test_android.json \
    --audio_src_prefix /data/mandarin/aishell2/evaluation/aishell \
    --audio_dst_prefix /home/hainanx/Workplace/data/aishell_eval/aishell \
    --factors 0.9,1.0,1.1 --method time_stretch \
    --output oracle_results_aishell/oracle_test_android.jsonl
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile
import torch
from tqdm import tqdm

try:
    import librosa  # noqa: F401  (used transitively by warp_audio)
except ImportError:  # pragma: no cover
    sys.exit("librosa is required: pip install librosa")

# Reuse the exact model-loading / transcribe / normalization / CER from the
# AISHELL eval, and the waveform warp from the English oracle experiment.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import run_eval_asr as R  # noqa: E402
import eval_aishell_cer as A  # noqa: E402
from oracle_timewarp_eval import warp_audio  # noqa: E402

from nemo.collections.asr.metrics.wer import word_error_rate  # noqa: E402


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def per_utt_cer(ref, hyp):
    """Single-utterance CER; ``ref`` is guaranteed non-empty after filtering."""
    return word_error_rate(
        hypotheses=[hyp if hyp.strip() else " "], references=[ref], use_cer=True
    )


# --------------------------------------------------------------------------- #
# IO  (mirrors eval_aishell_cer.main's manifest handling)
# --------------------------------------------------------------------------- #
def load_aishell_manifest(args):
    """Read (audio, raw_ref, duration) items, remapping audio onto the local tree."""
    rows = A.read_manifest(args.manifest)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        rows = rows[: args.max_eval_samples]

    items, missing = [], 0
    for r in rows:
        ref = r.get(args.text_key, r.get("text", ""))
        if ref is None or not str(ref).strip():
            continue
        ap = A.remap_audio_path(r["audio_filepath"], args.audio_src_prefix, args.audio_dst_prefix)
        if not os.path.exists(ap):
            missing += 1
            if missing <= 5:
                print(f"  WARNING: missing audio, skipping: {ap}")
            continue
        items.append({"audio": ap, "ref_raw": str(ref), "duration": float(r.get("duration", 0.0) or 0.0)})
    if missing:
        print(f"  ({missing} entries skipped: audio file not found)")
    # Longest-first for efficient batching; all factors decode in this same order.
    items.sort(key=lambda d: d["duration"], reverse=True)
    return items


def make_transcriber(model, is_multistream, batch_size):
    def _t(paths):
        return A.transcribe(model, is_multistream, paths, batch_size)

    loss_type = getattr(model, "loss_type", None)
    kind = (
        "multistream"
        if is_multistream
        else ("aligner/chunked" if loss_type in ("aligner", "chunked_aligner") else "tdt")
    )
    return _t, kind


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    factors = [float(x) for x in args.factors.split(",") if x.strip()]
    if 1.0 not in factors:
        factors = [1.0] + factors
    factors = sorted(set(factors))

    torch.set_float32_matmul_precision("medium")
    device = torch.device(
        f"cuda:{args.device}"
        if (args.device is not None and args.device >= 0 and torch.cuda.is_available())
        else "cpu"
    )

    model, is_multistream = R.load_model(args.model, device, tokenizer_dir=args.tokenizer_dir)
    if hasattr(model, "use_cer"):
        model.use_cer = True
    if args.max_symbols_per_step is not None and hasattr(model, "ms_greedy"):
        model.ms_greedy.max_symbols = args.max_symbols_per_step
    transcribe, kind = make_transcriber(model, is_multistream, args.batch_size)

    items = load_aishell_manifest(args)
    n = len(items)
    if n == 0:
        print("ERROR: nothing to evaluate (check --audio_src_prefix/--audio_dst_prefix).")
        sys.exit(1)
    refs = [A.normalize(it["ref_raw"], args.keep_spaces) for it in items]
    set_name = args.set_name or os.path.splitext(os.path.basename(args.manifest))[0]
    print(f"Manifest: {args.manifest}")
    print(
        f"Loaded {n} utterances; decoding {len(factors)} warp(s) {factors} "
        f"with method={args.method!r} ({kind})."
    )

    # factor -> aligned lists of formatted / normalized hypotheses
    hyps_fmt = {f: None for f in factors}
    hyps_norm = {f: None for f in factors}
    decode_time = {}
    work = tempfile.mkdtemp(prefix="oracle_warp_aishell_")
    try:
        for f in factors:
            fdir = os.path.join(work, f"f_{f}")
            os.makedirs(fdir, exist_ok=True)
            paths = []
            for i, it in enumerate(tqdm(items, desc=f"warp x{f}", leave=False)):
                y, sr = soundfile.read(it["audio"], dtype="float32", always_2d=False)
                if y.ndim == 2:
                    y = y.mean(axis=1)
                yw = warp_audio(y, f, args.method)
                op = os.path.join(fdir, f"{i:07d}.wav")
                soundfile.write(op, yw, sr)
                paths.append(op)
            t0 = time.time()
            out = transcribe(paths)
            decode_time[f] = time.time() - t0
            fmt = [h.strip() for h in out]
            hyps_fmt[f] = fmt
            hyps_norm[f] = [A.normalize(h, args.keep_spaces) for h in fmt]
            if not args.keep_warped:
                shutil.rmtree(fdir, ignore_errors=True)
    finally:
        if not args.keep_warped:
            shutil.rmtree(work, ignore_errors=True)

    # ---- scoring ---------------------------------------------------------- #
    # Per-factor corpus CER (what a single fixed warp would score).
    factor_corpus = {
        f: 100 * word_error_rate(hypotheses=hyps_norm[f], references=refs, use_cer=True) for f in factors
    }

    # Oracle: per utterance, pick the warp with the lowest per-utt CER.
    identity = 1.0 if 1.0 in factors else None
    chosen_factor, oracle_preds, details = [], [], []
    for i in range(n):
        cand = [(per_utt_cer(refs[i], hyps_norm[f][i]), f) for f in factors]
        best_w = min(c[0] for c in cand)
        tied = [f for (w, f) in cand if w == best_w]
        # tie-break: prefer identity (1.0), else the factor closest to 1.0
        if identity is not None and identity in tied:
            best_f = identity
        else:
            best_f = min(tied, key=lambda f: abs(f - 1.0))
        chosen_factor.append(best_f)
        oracle_preds.append(hyps_norm[best_f][i])
        details.append(
            {
                "audio_filepath": items[i]["audio"],
                "duration": items[i]["duration"],
                "text": items[i]["ref_raw"],
                "text_normalized": refs[i],
                "per_factor": {
                    str(f): {
                        "pred_text": hyps_fmt[f][i],
                        "pred_text_normalized": hyps_norm[f][i],
                        "cer": round(per_utt_cer(refs[i], hyps_norm[f][i]), 4),
                    }
                    for f in factors
                },
                "chosen_factor": best_f,
                "oracle_pred_text_normalized": oracle_preds[i],
                "oracle_cer": round(per_utt_cer(refs[i], oracle_preds[i]), 4),
            }
        )

    oracle_corpus = 100 * word_error_rate(hypotheses=oracle_preds, references=refs, use_cer=True)
    best_single_f = min(factor_corpus, key=factor_corpus.get)
    chosen_counts = Counter(chosen_factor)
    baseline = factor_corpus.get(1.0, float("nan"))

    # ---- human-readable report -------------------------------------------- #
    lines = []
    p = lines.append
    p("=" * 78)
    p(f"ORACLE TIME-WARP (CER)  |  set={set_name}  method={args.method}")
    p("=" * 78)
    p(f"model      : {args.model}")
    p(f"manifest   : {args.manifest}")
    p(f"utterances : {n}")
    p(f"factors    : {factors}")
    p("")
    p("Per-factor corpus CER (each warp used alone):")
    for f in factors:
        tag = "  (baseline/no-warp)" if f == 1.0 else ""
        p(f"  x{f:<5}: {factor_corpus[f]:6.2f} %   [decode {decode_time[f]:.1f}s]{tag}")
    p("")
    p(f"baseline (x1.0)                 : {baseline:6.2f} %")
    p(f"best single fixed warp (x{best_single_f}) : {factor_corpus[best_single_f]:6.2f} %")
    p(f"ORACLE best-of-{len(factors)} (cheating)     : {oracle_corpus:6.2f} %")
    p(f"  -> abs. gain vs no-warp       : {baseline - oracle_corpus:6.2f} pts "
      f"({100 * (baseline - oracle_corpus) / max(baseline, 1e-9):.1f}% rel.)")
    p(f"  -> abs. gain vs best fixed    : {factor_corpus[best_single_f] - oracle_corpus:6.2f} pts")
    p("")
    p("How often each warp was the oracle pick:")
    for f in factors:
        c = chosen_counts.get(f, 0)
        bar = "#" * int(round(40 * c / max(1, n)))
        p(f"  x{f:<5}: {c:>6}  ({100 * c / n:5.1f}%)  {bar}")
    report = "\n".join(lines)
    print(report)

    # ---- machine-readable summary line (parsed by the driver) ------------- #
    pick_summary = ",".join(f"x{f}:{100 * chosen_counts.get(f, 0) / n:.1f}" for f in factors)
    factor_cer_summary = ",".join(f"x{f}:{factor_corpus[f]:.2f}" for f in factors)
    print(
        f"ORACLE_SUMMARY {set_name} scored={n} total={n} "
        f"baseline={baseline:.2f} best_fixed={factor_corpus[best_single_f]:.2f} "
        f"oracle={oracle_corpus:.2f} factor_cers={factor_cer_summary} picks={pick_summary}"
    )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write("# " + report.replace("\n", "\n# ") + "\n")
            for d in details:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"[per-utterance report written to {args.output}]", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Path to a .nemo or Lightning .ckpt file.")
    ap.add_argument("--manifest", required=True, help="Local NeMo manifest (audio_filepath + text per line).")
    ap.add_argument("--set_name", default=None, help="Label for the summary line (default: manifest basename).")
    ap.add_argument("--tokenizer_dir", default=None, help="Override tokenizer dir for .ckpt loads.")
    ap.add_argument("--audio_src_prefix", default="/data/mandarin/aishell2/evaluation/aishell",
                    help="Manifest audio path prefix to rewrite (set empty to disable remap).")
    ap.add_argument("--audio_dst_prefix", default="",
                    help="Local audio root to substitute for --audio_src_prefix.")
    ap.add_argument("--text_key", default="text", help="Reference text field in the manifest.")
    ap.add_argument("--factors", default="0.9,1.0,1.1",
                    help="Comma-separated warp factors (>1 faster/shorter, <1 slower/longer, 1.0 = identity). "
                         "1.0 is auto-added if absent.")
    ap.add_argument("--method", default="time_stretch", choices=["time_stretch", "speed"],
                    help="time_stretch = pitch-preserving tempo warp; speed = resample (pitch shifts).")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_eval_samples", type=int, default=None, help="Limit to the first N manifest entries.")
    ap.add_argument("--max_symbols_per_step", type=int, default=None,
                    help="Override greedy symbols-per-step (multistream only).")
    ap.add_argument("--keep_spaces", action="store_true",
                    help="Do NOT collapse whitespace before CER (default: collapse, AISHELL convention).")
    ap.add_argument("--keep_warped", action="store_true", help="Do not delete the temp warped wavs.")
    ap.add_argument("--output", default=None, help="Write a per-utterance JSONL report (with header summary).")
    main(ap.parse_args())
