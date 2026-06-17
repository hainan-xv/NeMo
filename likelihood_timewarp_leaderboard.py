"""
Likelihood-selected time-warp eval over the Open-ASR-Leaderboard datasets.

For each utterance, decode several time-warped copies (for example x0.9, x1.0,
x1.1), then choose the hypothesis whose greedy decoder score is highest. This
is a deployable alternative to oracle_timewarp_leaderboard.py: it never looks at
the reference when selecting the warp. References are used only to report WER.
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

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import oracle_timewarp_leaderboard as O  # noqa: E402
import run_eval_asr as R  # noqa: E402


def _hyp_text(model, hyp):
    text = getattr(hyp, "text", None)
    if text is not None:
        return str(text).strip()
    seq = getattr(hyp, "y_sequence", None)
    if isinstance(seq, torch.Tensor):
        seq = seq.tolist()
    if seq is not None and hasattr(model, "tokenizer"):
        return model.tokenizer.ids_to_text([int(x) for x in seq]).strip()
    return str(hyp).strip()


def _hyp_token_len(hyp, text):
    seq = getattr(hyp, "y_sequence", None)
    if isinstance(seq, torch.Tensor):
        return int(seq.numel())
    if seq is not None:
        return len(seq)
    return max(1, len(text.split()))


def _hyp_score(hyp):
    score = getattr(hyp, "score", None)
    if score is None:
        return None
    if isinstance(score, torch.Tensor):
        return float(score.detach().cpu().item())
    return float(score)


def normalize_score(raw_score, token_len, text, norm):
    if norm == "none":
        return raw_score
    if norm == "token":
        return raw_score / max(token_len, 1)
    if norm == "word":
        return raw_score / max(len(text.split()), 1)
    if norm == "char":
        return raw_score / max(len(text), 1)
    raise ValueError(f"unknown score_norm={norm!r}")


@torch.inference_mode()
def transcribe_scored(model, is_multistream, audio_files, batch_size, score_norm):
    """Return formatted text, normalized text, raw scores, and selection scores."""
    loss_type = getattr(model, "loss_type", None)
    is_aligner = loss_type in ("aligner", "chunked_aligner")

    hypotheses = []
    if is_multistream:
        dloader = R.setup_dloader(audio_files, batch_size=batch_size)
        for batch in tqdm(dloader, desc="Transcribing scored (multistream)"):
            audios = batch["audios"].to(model.device, non_blocking=True)
            audio_lens = batch["audio_lens"].to(model.device, non_blocking=True)
            encoded, encoded_len = model.forward(input_signal=audios, input_signal_length=audio_lens)
            hypotheses.extend(model.ms_greedy(encoder_output=encoded, encoded_lengths=encoded_len)[0])
    elif is_aligner:
        dloader = torch.utils.data.DataLoader(
            R.AudioFileDataset(audio_files),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=R._collate_audio,
        )
        for signal, signal_len in tqdm(dloader, desc="Transcribing scored (aligner/chunked)"):
            signal = signal.to(model.device, non_blocking=True)
            signal_len = signal_len.to(model.device, non_blocking=True)
            outputs = model._transcribe_forward((signal, signal_len), trcfg=None)
            hypotheses.extend(model._transcribe_output_processing(outputs, trcfg=None))
    else:
        with torch.inference_mode():
            try:
                out = model.transcribe(
                    audio_files, batch_size=batch_size, verbose=True, return_hypotheses=True
                )
            except TypeError:
                sys.exit("model.transcribe() does not support return_hypotheses=True; cannot score hypotheses")
        if isinstance(out, tuple):
            out = out[0]
        hypotheses = out

    texts, norm_texts, raw_scores, select_scores = [], [], [], []
    missing_scores = 0
    for hyp in hypotheses:
        text = _hyp_text(model, hyp)
        raw_score = _hyp_score(hyp)
        if raw_score is None:
            missing_scores += 1
            raw_score = float("-inf")
        token_len = _hyp_token_len(hyp, text)
        texts.append(text)
        norm_texts.append(R.text_normalizer(text))
        raw_scores.append(raw_score)
        select_scores.append(normalize_score(raw_score, token_len, text, score_norm))

    if missing_scores:
        sys.exit(f"{missing_scores}/{len(hypotheses)} hypotheses did not expose a decoder score")
    return texts, norm_texts, raw_scores, select_scores


def warp_decode_scored(model, is_ms, wavs, factor, method, batch_size, score_norm, keep=False):
    work = tempfile.mkdtemp(prefix=f"likelihood_warp_{factor}_")
    try:
        paths = []
        for i, w in enumerate(tqdm(wavs, desc=f"warp x{factor}", leave=False)):
            y, sr = soundfile.read(w, dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1)
            op = os.path.join(work, f"{i:07d}.wav")
            soundfile.write(op, O.warp_audio(y, factor, method), sr)
            paths.append(op)
        return transcribe_scored(model, is_ms, paths, batch_size, score_norm)
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


def tie_break_factor(candidates, prefer_one=True):
    best = max(score for score, _factor in candidates)
    tied = [factor for score, factor in candidates if score == best]
    if prefer_one and 1.0 in tied:
        return 1.0
    return min(tied, key=lambda factor: abs(factor - 1.0))


def oracle_factor(ref, hyps_norm, factors, idx):
    cand = [(O.per_utt_wer(ref, hyps_norm[f][idx]), f) for f in factors]
    best_w = min(w for w, _factor in cand)
    tied = [factor for w, factor in cand if w == best_w]
    if 1.0 in tied:
        return 1.0
    return min(tied, key=lambda factor: abs(factor - 1.0))


def pct_summary(counts, factors, denom):
    return ",".join(f"x{f}:{100*counts.get(f, 0)/denom:.1f}" for f in factors)


def main(args):
    factors = [float(x) for x in args.factors.split(",") if x.strip()]
    if 1.0 not in factors:
        factors = [1.0] + factors
    factors = sorted(set(factors))

    torch.set_float32_matmul_precision("medium")
    device = torch.device(
        f"cuda:{args.device}" if (args.device is not None and args.device >= 0 and torch.cuda.is_available()) else "cpu"
    )

    data = O.load_leaderboard_audio_refs(args)
    wavs = data["wavs"]
    if not wavs:
        sys.exit(f"no scorable samples for {args.dataset!r}")
    refs = data["refs"]
    refs_raw = data["refs_raw"]
    sample_ids = data["sample_ids"]
    total_rows = data["total_rows"]
    scored = len(wavs)

    model, is_ms = R.load_model(args.model, device, tokenizer_dir=args.tokenizer_dir)
    kind = "multistream" if is_ms else ("aligner/chunked" if getattr(model, "loss_type", None) in ("aligner", "chunked_aligner") else "tdt")
    print(f"[{args.dataset}] {scored}/{total_rows} wavs | decode={kind} | factors={factors} | score_norm={args.score_norm}")

    hyps_fmt, hyps_norm, raw_scores, select_scores, decode_time = {}, {}, {}, {}, {}
    t0 = time.time()
    hyps_fmt[1.0], hyps_norm[1.0], raw_scores[1.0], select_scores[1.0] = transcribe_scored(
        model, is_ms, wavs, args.batch_size, args.score_norm
    )
    decode_time[1.0] = time.time() - t0

    for factor in factors:
        if factor == 1.0:
            continue
        t0 = time.time()
        hyps_fmt[factor], hyps_norm[factor], raw_scores[factor], select_scores[factor] = warp_decode_scored(
            model, is_ms, wavs, factor, args.method, args.batch_size, args.score_norm, keep=args.keep_warped
        )
        decode_time[factor] = time.time() - t0

    factor_corpus = {f: 100 * R.wer_metric.compute(references=refs, predictions=hyps_norm[f]) for f in factors}
    best_fixed = min(factor_corpus, key=factor_corpus.get)

    selected_factors, oracle_factors = [], []
    selected_preds, oracle_preds, details = [], [], []
    for i in range(scored):
        selected = tie_break_factor([(select_scores[f][i], f) for f in factors])
        oracle = oracle_factor(refs[i], hyps_norm, factors, i)
        selected_factors.append(selected)
        oracle_factors.append(oracle)
        selected_preds.append(hyps_norm[selected][i])
        oracle_preds.append(hyps_norm[oracle][i])
        details.append(
            {
                "audio_filepath": wavs[i],
                "sample_id": sample_ids[i],
                "text": refs_raw[i],
                "text_normalized": refs[i],
                "selected_factor": selected,
                "oracle_factor": oracle,
                "selected_wer": round(O.per_utt_wer(refs[i], selected_preds[i]), 4),
                "oracle_wer": round(O.per_utt_wer(refs[i], oracle_preds[i]), 4),
                "per_factor": {
                    str(f): {
                        "pred_text": hyps_fmt[f][i],
                        "pred_text_normalized": hyps_norm[f][i],
                        "raw_score": raw_scores[f][i],
                        "selection_score": select_scores[f][i],
                        "wer": round(O.per_utt_wer(refs[i], hyps_norm[f][i]), 4),
                    }
                    for f in factors
                },
            }
        )

    selected_wer = 100 * R.wer_metric.compute(references=refs, predictions=selected_preds)
    oracle_wer = 100 * R.wer_metric.compute(references=refs, predictions=oracle_preds)
    selected_counts = Counter(selected_factors)
    oracle_counts = Counter(oracle_factors)
    agreement = sum(s == o for s, o in zip(selected_factors, oracle_factors)) / scored

    lines = []
    p = lines.append
    p("=" * 88)
    p(f"LIKELIHOOD TIME-WARP  |  dataset={args.dataset}  method={args.method}  score_norm={args.score_norm}")
    p("=" * 88)
    p(f"scored utterances     : {scored}/{total_rows}")
    p(f"factors               : {factors}")
    p("")
    p("Per-factor corpus WER (each warp alone):")
    for f in factors:
        tag = "  (baseline/no-warp)" if f == 1.0 else ""
        p(f"  x{f:<5}: {factor_corpus[f]:6.2f} %   [decode {decode_time[f]:.0f}s]{tag}")
    p("")
    p(f"baseline (x1.0)             : {factor_corpus[1.0]:6.2f} %")
    p(f"best single fixed (x{best_fixed})    : {factor_corpus[best_fixed]:6.2f} %")
    p(f"LIKELIHOOD selected         : {selected_wer:6.2f} %")
    p(f"ORACLE best-of-{len(factors)}           : {oracle_wer:6.2f} %")
    p(f"selector/oracle agreement   : {100*agreement:6.2f} %")
    p("")
    p("likelihood pick distribution:")
    for f in factors:
        c = selected_counts.get(f, 0)
        p(f"  x{f:<5}: {c:>6}  ({100*c/scored:5.1f}%)  {'#'*int(round(40*c/scored))}")
    p("")
    p("oracle pick distribution:")
    for f in factors:
        c = oracle_counts.get(f, 0)
        p(f"  x{f:<5}: {c:>6}  ({100*c/scored:5.1f}%)  {'#'*int(round(40*c/scored))}")
    report = "\n".join(lines)
    print(report)

    selected_pick_summary = pct_summary(selected_counts, factors, scored)
    oracle_pick_summary = pct_summary(oracle_counts, factors, scored)
    factor_wer_summary = ",".join(f"x{f}:{factor_corpus[f]:.2f}" for f in factors)
    print(
        f"LIKELIHOOD_SUMMARY {args.dataset} scored={scored} total={total_rows} "
        f"baseline={factor_corpus[1.0]:.2f} best_fixed={factor_corpus[best_fixed]:.2f} "
        f"selected={selected_wer:.2f} oracle={oracle_wer:.2f} agreement={100*agreement:.2f} "
        f"factor_wers={factor_wer_summary} picks={selected_pick_summary} oracle_picks={oracle_pick_summary}"
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write("# " + report.replace("\n", "\n# ") + "\n")
            for d in details:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"[per-utterance report -> {args.output}]", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, help="e.g. librispeech_test.clean")
    ap.add_argument("--dataset_path", default="hf-audio/esb-datasets-test-only-sorted")
    ap.add_argument("--dataset_revision", default=None)
    ap.add_argument("--audio_dir", default=None, help="override audio cache dir for this dataset/split")
    ap.add_argument("--tokenizer_dir", default=None)
    ap.add_argument("--factors", default="0.9,1.0,1.1")
    ap.add_argument("--method", default="time_stretch", choices=["time_stretch", "speed"])
    ap.add_argument("--score_norm", default="token", choices=["none", "token", "word", "char"])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--keep_warped", action="store_true")
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-streaming", dest="streaming", action="store_false")
    ap.set_defaults(streaming=True)
    main(ap.parse_args())
