"""
Oracle time-warp ("speed cheat") inference experiment.

For each utterance we synthesize several time-WARPED copies of the audio (a list
of tempo/speed factors), decode every copy independently with the SAME model,
record all hypotheses, and then -- in scoring only -- pick, per utterance, the
warp that gives the lowest WER against the reference.

This is deliberately CHEATING (it peeks at the reference to choose the warp), so
the resulting WER is an *oracle upper bound*: "if we could always pick the best
of N time-warps, how good could decoding get?". Useful to gauge headroom and to
see whether short/fast utterances (which the error analysis flagged) are rescued
by slowing them down.

Model loading, the per-architecture transcribe paths, whisper normalization and
the (leaderboard) WER metric are all reused from ``run_eval_asr.py`` so numbers
are directly comparable to a normal eval.

Input: a NeMo-style JSON manifest (one object per line) with at least
``audio_filepath`` and ``text``; ``duration`` is used if present. The cached
16 kHz wavs under ``audio_cache/<dataset>/<split>/`` (written by run_eval_asr)
already work -- just build a manifest pointing at them.

Examples
--------
python oracle_timewarp_eval.py \
    --model /checkpoints/.../model.nemo \
    --manifest my_audio.jsonl \
    --factors 0.9,1.0,1.1 \
    --method time_stretch \
    --output oracle_report.jsonl

# Purely-stretched (no identity copy), resample-based speed change:
python oracle_timewarp_eval.py --model ... --manifest ... \
    --factors 0.85,0.95,1.1 --method speed
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
    import librosa
except ImportError:  # pragma: no cover
    sys.exit("librosa is required: pip install librosa")

try:
    import jiwer
except ImportError:  # pragma: no cover
    sys.exit("jiwer is required: pip install jiwer")

# Reuse the exact model loading / transcribe / normalization / WER from the eval.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import run_eval_asr as R  # noqa: E402


# --------------------------------------------------------------------------- #
# Time warping
# --------------------------------------------------------------------------- #
def warp_audio(y, factor, method):
    """Return a time-warped copy of mono waveform ``y``.

    Convention (both methods): ``factor > 1`` => faster / shorter audio,
    ``factor < 1`` => slower / longer (stretched) audio, ``factor == 1`` => copy.

    * ``time_stretch``: phase-vocoder tempo change, pitch PRESERVED (a pure time
      warp). ``factor`` is the librosa ``rate``.
    * ``speed``: linear-resample speed change, pitch ALSO shifts (classic ASR
      speed perturbation). New length = len(y) / factor.
    """
    if factor == 1.0:
        return np.asarray(y, dtype=np.float32)
    if method == "time_stretch":
        return librosa.effects.time_stretch(np.asarray(y, dtype=np.float32), rate=factor)
    if method == "speed":
        n = max(1, int(round(len(y) / factor)))
        x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
        return np.interp(x_new, x_old, y).astype(np.float32)
    raise ValueError(f"unknown --method {method!r} (expected time_stretch|speed)")


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_manifest(path, max_samples=None):
    """Read (audio_filepath, normalized_ref, raw_ref, duration) tuples."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ap = d.get("audio_filepath")
            if ap is None:
                continue
            ref_raw = None
            for k in ("text", "sentence", "normalized_text", "transcript", "transcription"):
                if k in d and d[k] is not None:
                    ref_raw = d[k]
                    break
            if ref_raw is None:
                continue
            ref = R.text_normalizer(ref_raw)
            if not ref.strip() or ref.strip() == "ignore time segment in scoring":
                continue
            items.append({"audio": ap, "ref": ref, "ref_raw": ref_raw, "duration": d.get("duration")})
            if max_samples is not None and len(items) >= max_samples > 0:
                break
    return items


def make_transcriber(model, is_multistream, batch_size):
    loss_type = getattr(model, "loss_type", None)
    is_aligner = loss_type in ("aligner", "chunked_aligner")

    def _t(paths):
        if is_multistream:
            return R.transcribe_multistream(model, paths, batch_size)
        if is_aligner:
            return R.transcribe_aligner_like(model, paths, batch_size)
        return R.transcribe_tdt(model, paths, batch_size)

    kind = "multistream" if is_multistream else ("aligner/chunked" if is_aligner else "tdt")
    return _t, kind


def per_utt_wer(ref, hyp):
    """Standard WER for a single (ref, hyp) pair; ref is guaranteed non-empty."""
    return jiwer.wer(ref, hyp if hyp.strip() else " ")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    factors = [float(x) for x in args.factors.split(",") if x.strip()]
    if not factors:
        sys.exit("Provide at least one --factors value.")
    torch.set_float32_matmul_precision("medium")
    device = torch.device(
        f"cuda:{args.device}" if (args.device is not None and args.device >= 0 and torch.cuda.is_available()) else "cpu"
    )

    model, is_multistream = R.load_model(args.model, device, tokenizer_dir=args.tokenizer_dir)
    if args.max_symbols_per_step is not None and hasattr(model, "ms_greedy"):
        model.ms_greedy.max_symbols = args.max_symbols_per_step
    transcribe, kind = make_transcriber(model, is_multistream, args.batch_size)

    items = load_manifest(args.manifest, args.max_samples)
    n = len(items)
    if n == 0:
        sys.exit(f"No usable samples in manifest {args.manifest!r}.")
    refs = [it["ref"] for it in items]
    print(f"Loaded {n} utterances; decoding {len(factors)} warp(s) {factors} "
          f"with method={args.method!r} ({kind}).")

    # factor -> aligned lists of formatted/normalized hypotheses
    hyps_fmt = {f: None for f in factors}
    hyps_norm = {f: None for f in factors}
    work = tempfile.mkdtemp(prefix="oracle_warp_")
    decode_time = {}
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
            hyps_norm[f] = [R.text_normalizer(h) for h in fmt]
            if not args.keep_warped:
                shutil.rmtree(fdir, ignore_errors=True)
    finally:
        if not args.keep_warped:
            shutil.rmtree(work, ignore_errors=True)

    # ---- scoring ---------------------------------------------------------- #
    # Per-factor corpus WER (what a single fixed warp would score).
    factor_corpus = {f: 100 * R.wer_metric.compute(references=refs, predictions=hyps_norm[f]) for f in factors}

    # Oracle: per utterance, pick the warp with the lowest per-utt WER.
    identity = 1.0 if 1.0 in factors else None
    chosen_factor = []
    oracle_preds = []
    details = []
    for i in range(n):
        cand = []
        for f in factors:
            w = per_utt_wer(refs[i], hyps_norm[f][i])
            cand.append((w, f))
        # tie-break: prefer identity (1.0) if available, else the factor closest to 1.0
        best_w = min(c[0] for c in cand)
        tied = [f for (w, f) in cand if w == best_w]
        if identity is not None and identity in tied:
            best_f = identity
        else:
            best_f = min(tied, key=lambda f: abs(f - 1.0))
        chosen_factor.append(best_f)
        oracle_preds.append(hyps_norm[best_f][i])
        details.append({
            "audio_filepath": items[i]["audio"],
            "duration": items[i]["duration"],
            "text": items[i]["ref_raw"],
            "text_normalized": refs[i],
            "per_factor": {
                str(f): {
                    "pred_text": hyps_fmt[f][i],
                    "pred_text_normalized": hyps_norm[f][i],
                    "wer": round(per_utt_wer(refs[i], hyps_norm[f][i]), 4),
                }
                for f in factors
            },
            "chosen_factor": best_f,
            "oracle_pred_text_normalized": oracle_preds[i],
            "oracle_wer": round(per_utt_wer(refs[i], oracle_preds[i]), 4),
        })

    oracle_corpus = 100 * R.wer_metric.compute(references=refs, predictions=oracle_preds)
    best_single_f = min(factor_corpus, key=factor_corpus.get)
    chosen_counts = Counter(chosen_factor)

    # ---- report ----------------------------------------------------------- #
    lines = []
    p = lines.append
    p("=" * 78)
    p("ORACLE TIME-WARP EXPERIMENT")
    p("=" * 78)
    p(f"model      : {args.model}")
    p(f"manifest   : {args.manifest}")
    p(f"utterances : {n}")
    p(f"method     : {args.method}   factors: {factors}")
    p("")
    p("Per-factor corpus WER (each warp used alone):")
    for f in factors:
        tag = "  (identity/no-warp)" if f == 1.0 else ""
        p(f"  x{f:<5}: {factor_corpus[f]:6.2f} %   [decode {decode_time[f]:.1f}s]{tag}")
    p("")
    if identity is not None:
        p(f"Baseline (no warp, x1.0)        : {factor_corpus[1.0]:6.2f} %")
    p(f"Best single fixed warp (x{best_single_f}) : {factor_corpus[best_single_f]:6.2f} %")
    p(f"ORACLE best-of-{len(factors)} (cheating)     : {oracle_corpus:6.2f} %")
    if identity is not None:
        p(f"  -> abs. gain vs no-warp       : {factor_corpus[1.0] - oracle_corpus:6.2f} pts "
          f"({100 * (factor_corpus[1.0] - oracle_corpus) / max(factor_corpus[1.0], 1e-9):.1f}% rel.)")
    p(f"  -> abs. gain vs best fixed warp : {factor_corpus[best_single_f] - oracle_corpus:6.2f} pts")
    p("")
    p("How often each warp was the oracle pick:")
    for f in factors:
        c = chosen_counts.get(f, 0)
        bar = "#" * int(round(40 * c / max(1, n)))
        p(f"  x{f:<5}: {c:>6}  ({100 * c / n:5.1f}%)  {bar}")
    p("")
    report = "\n".join(lines)
    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write("# " + report.replace("\n", "\n# ") + "\n")
            for d in details:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"[per-utterance report written to {args.output}]", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Path to a .nemo or Lightning .ckpt file.")
    ap.add_argument("--manifest", required=True, help="JSON manifest (audio_filepath + text per line).")
    ap.add_argument("--tokenizer_dir", default=None, help="Override tokenizer dir for .ckpt loads.")
    ap.add_argument("--factors", default="0.9,1.0,1.1",
                    help="Comma-separated warp factors (>1 faster/shorter, <1 slower/longer, 1.0 = identity).")
    ap.add_argument("--method", default="time_stretch", choices=["time_stretch", "speed"],
                    help="time_stretch = pitch-preserving tempo warp; speed = resample (pitch shifts).")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_samples", type=int, default=None, help="Limit to the first N manifest entries.")
    ap.add_argument("--max_symbols_per_step", type=int, default=None,
                    help="Override greedy symbols-per-step (multistream only).")
    ap.add_argument("--keep_warped", action="store_true", help="Do not delete the temp warped wavs.")
    ap.add_argument("--output", default=None, help="Write a per-utterance JSONL report (with header summary).")
    main(ap.parse_args())
