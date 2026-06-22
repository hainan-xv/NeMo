"""
Likelihood-selected time-warp CER experiment for Mandarin ASR checkpoints.

For each utterance, decode several time-warped copies of the audio and choose
the hypothesis with the highest greedy decoder score. This is the deployable
counterpart to ``oracle_timewarp_aishell.py``: references are not used for
selection, only for reporting CER and for the optional oracle diagnostic.
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
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import eval_aishell_cer as A  # noqa: E402
import oracle_timewarp_aishell as O  # noqa: E402
import run_eval_asr as R  # noqa: E402
from oracle_timewarp_eval import warp_audio  # noqa: E402

from nemo.collections.asr.metrics.wer import word_error_rate  # noqa: E402


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
    return max(1, len(text))


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
    if norm == "char":
        return raw_score / max(len(text), 1)
    raise ValueError(f"unknown score_norm={norm!r}")


def _decoder_blank_id(model):
    decoding = getattr(model, "decoding", None)
    for obj in (decoding, getattr(decoding, "decoding", None), model):
        for attr in ("blank_id", "_blank_index"):
            value = getattr(obj, attr, None)
            if value is not None:
                return int(value)
    return None


def _set_preserve_alignments(model, enabled):
    """Enable alignment capture on the already-constructed greedy decoder."""
    changed = []
    seen = set()
    queue = [getattr(model, "decoding", None), getattr(model, "ms_greedy", None)]
    while queue:
        obj = queue.pop(0)
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        if hasattr(obj, "preserve_alignments"):
            changed.append((obj, getattr(obj, "preserve_alignments")))
            setattr(obj, "preserve_alignments", enabled)
        queue.append(getattr(obj, "decoding", None))
    return changed


def _restore_attrs(changed):
    for obj, value in changed:
        setattr(obj, "preserve_alignments", value)


def _clone_cfg(cfg):
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def _enable_alignment_decoding(model):
    """Rebuild RNNT/TDT decoding with alignment capture enabled, returning old config."""
    if not hasattr(model, "change_decoding_strategy") or not hasattr(model, "cfg"):
        return None
    old_cfg = _clone_cfg(model.cfg.decoding)
    new_cfg = _clone_cfg(model.cfg.decoding)
    with open_dict(new_cfg):
        new_cfg.preserve_alignments = True
        if "greedy" in new_cfg and new_cfg.greedy is not None:
            new_cfg.greedy.preserve_alignments = True
        if "beam" in new_cfg and new_cfg.beam is not None:
            new_cfg.beam.preserve_alignments = True
    model.change_decoding_strategy(new_cfg, verbose=False)
    return old_cfg


def _restore_decoding(model, old_cfg):
    if old_cfg is not None:
        model.change_decoding_strategy(old_cfg, verbose=False)


def _alignment_token_logprob(hyp, blank_id):
    """Return (sum log P(emitted token), emitted token count) from preserved alignments."""
    alignments = getattr(hyp, "alignments", None)
    if alignments is None or blank_id is None:
        return None

    total = 0.0
    count = 0
    for frame in alignments:
        for item in frame:
            if not item:
                continue
            logits, label = item[:2]
            if isinstance(label, torch.Tensor):
                label = int(label.detach().cpu().item())
            else:
                label = int(label)
            if label == blank_id:
                continue
            logits = logits.detach().float().cpu()
            if label < 0 or label >= logits.numel():
                continue
            total += float(torch.log_softmax(logits, dim=-1)[label].item())
            count += 1
    return total, count


@torch.inference_mode()
def transcribe_scored(model, is_multistream, audio_files, batch_size, score_norm, keep_spaces):
    """Return formatted text, AISHELL-normalized text, raw scores, and selection scores."""
    loss_type = getattr(model, "loss_type", None)
    is_aligner_like = loss_type in ("aligner", "chunked_aligner")

    hypotheses = []
    old_decoding_cfg = _enable_alignment_decoding(model) if score_norm == "logprob_token" and not is_multistream else None
    changed = _set_preserve_alignments(model, score_norm == "logprob_token")
    try:
        if is_multistream:
            dloader = R.setup_dloader(audio_files, batch_size=batch_size)
            for batch in tqdm(dloader, desc="Transcribing scored (multistream)"):
                audios = batch["audios"].to(model.device, non_blocking=True)
                audio_lens = batch["audio_lens"].to(model.device, non_blocking=True)
                encoded, encoded_len = model.forward(input_signal=audios, input_signal_length=audio_lens)
                hypotheses.extend(model.ms_greedy(encoder_output=encoded, encoded_lengths=encoded_len)[0])
        elif is_aligner_like:
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
            try:
                out = model.transcribe(audio_files, batch_size=batch_size, verbose=True, return_hypotheses=True)
            except TypeError:
                sys.exit("model.transcribe() does not support return_hypotheses=True; cannot score hypotheses")
            if isinstance(out, tuple):
                out = out[0]
            hypotheses = out
    finally:
        _restore_attrs(changed)
        _restore_decoding(model, old_decoding_cfg)

    texts, norm_texts, raw_scores, select_scores = [], [], [], []
    missing_scores = 0
    missing_alignments = 0
    blank_id = _decoder_blank_id(model)
    for hyp in hypotheses:
        text = _hyp_text(model, hyp)
        raw_score = _hyp_score(hyp)
        if raw_score is None:
            missing_scores += 1
            raw_score = float("-inf")
        token_len = _hyp_token_len(hyp, text)
        if score_norm == "logprob_token":
            aligned = _alignment_token_logprob(hyp, blank_id)
            if aligned is None or aligned[1] == 0:
                missing_alignments += 1
                select_score = float("-inf")
            else:
                select_score = aligned[0] / aligned[1]
        else:
            select_score = normalize_score(raw_score, token_len, text, score_norm)
        texts.append(text)
        norm_texts.append(A.normalize(text, keep_spaces))
        raw_scores.append(raw_score)
        select_scores.append(select_score)

    if score_norm != "logprob_token" and missing_scores:
        sys.exit(f"{missing_scores}/{len(hypotheses)} hypotheses did not expose a decoder score")
    if missing_alignments:
        sys.exit(
            f"{missing_alignments}/{len(hypotheses)} hypotheses did not expose emitted-token alignments; "
            "cannot use score_norm=logprob_token"
        )
    return texts, norm_texts, raw_scores, select_scores


def decode_factor(model, is_ms, items, factor, args):
    work = tempfile.mkdtemp(prefix=f"likelihood_warp_aishell_{factor}_")
    try:
        paths = []
        for i, it in enumerate(tqdm(items, desc=f"warp x{factor}", leave=False)):
            y, sr = soundfile.read(it["audio"], dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1)
            op = os.path.join(work, f"{i:07d}.wav")
            soundfile.write(op, warp_audio(np.asarray(y, dtype=np.float32), factor, args.method), sr)
            paths.append(op)
        return transcribe_scored(model, is_ms, paths, args.batch_size, args.score_norm, args.keep_spaces)
    finally:
        if not args.keep_warped:
            shutil.rmtree(work, ignore_errors=True)


def tie_break_factor(candidates, epsilon=0.0):
    """Pick the factor with the highest score, defaulting to 1.0.

    When epsilon > 0 (the recommended default), a non-1.0 factor is chosen only
    if its score exceeds the x1.0 score by at least epsilon.  This prevents
    marginal score differences from overriding the safe 1.0 baseline.

    Tie-breaking always prefers 1.0, then the factor nearest to 1.0.
    """
    score_1 = next((s for s, f in candidates if f == 1.0), None)
    if epsilon > 0.0 and score_1 is not None:
        # Only consider factors that beat 1.0 by >= epsilon
        eligible = [(s, f) for s, f in candidates if f == 1.0 or s - score_1 >= epsilon]
    else:
        eligible = candidates

    best = max(score for score, _factor in eligible)
    tied = [factor for score, factor in eligible if score == best]
    if 1.0 in tied:
        return 1.0
    return min(tied, key=lambda factor: abs(factor - 1.0))


def oracle_factor(ref, hyps_norm, factors, idx):
    cand = [(O.per_utt_cer(ref, hyps_norm[f][idx]), f) for f in factors]
    best_cer = min(cer for cer, _factor in cand)
    tied = [factor for cer, factor in cand if cer == best_cer]
    if 1.0 in tied:
        return 1.0
    return min(tied, key=lambda factor: abs(factor - 1.0))


def pct_summary(counts, factors, denom):
    return ",".join(f"x{f}:{100 * counts.get(f, 0) / denom:.1f}" for f in factors)


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

    model, is_ms = R.load_model(args.model, device, tokenizer_dir=args.tokenizer_dir)
    if hasattr(model, "use_cer"):
        model.use_cer = True
    if args.max_symbols_per_step is not None and hasattr(model, "ms_greedy"):
        model.ms_greedy.max_symbols = args.max_symbols_per_step

    items = O.load_aishell_manifest(args)
    n = len(items)
    if n == 0:
        print("ERROR: nothing to evaluate (check --audio_src_prefix/--audio_dst_prefix).")
        sys.exit(1)

    refs = [A.normalize(it["ref_raw"], args.keep_spaces) for it in items]
    set_name = args.set_name or os.path.splitext(os.path.basename(args.manifest))[0]
    kind = "multistream" if is_ms else ("aligner/chunked" if getattr(model, "loss_type", None) in ("aligner", "chunked_aligner") else "tdt")
    print(
        f"Loaded {n} utterances; decoding {len(factors)} warp(s) {factors} "
        f"with method={args.method!r}, score_norm={args.score_norm!r} ({kind})."
    )

    # logprob_token_dur decodes with logprob_token then multiplies by (char_len × factor)
    # to normalise by expected warped-audio duration rather than actual token count.
    _decode_norm = "logprob_token" if args.score_norm == "logprob_token_dur" else args.score_norm

    hyps_fmt, hyps_norm, raw_scores, select_scores, decode_time = {}, {}, {}, {}, {}
    for factor in factors:
        t0 = time.time()
        # Temporarily override score_norm for the decode call if needed
        _orig_norm = args.score_norm
        args.score_norm = _decode_norm
        hyps_fmt[factor], hyps_norm[factor], raw_scores[factor], select_scores[factor] = decode_factor(
            model, is_ms, items, factor, args
        )
        args.score_norm = _orig_norm
        decode_time[factor] = time.time() - t0

    # Duration-adjusted: multiply logprob/tok by (char_len × factor).  Both the
    # char_len (output length) and factor account for how much speech content the
    # warped audio contains; their product is ∝ warped audio duration × factor²,
    # which empirically removes the x1.1 bias seen in plain logprob_token.
    if args.score_norm == "logprob_token_dur":
        for factor in factors:
            select_scores[factor] = [
                s * max(len(t), 1) * factor
                for s, t in zip(select_scores[factor], hyps_norm[factor])
            ]

    factor_corpus = {
        f: 100 * word_error_rate(hypotheses=hyps_norm[f], references=refs, use_cer=True) for f in factors
    }
    best_fixed = min(factor_corpus, key=factor_corpus.get)

    selected_factors, oracle_factors = [], []
    selected_preds, oracle_preds, details = [], [], []
    for i in range(n):
        selected = tie_break_factor([(select_scores[f][i], f) for f in factors],
                                    epsilon=args.score_epsilon)
        oracle = oracle_factor(refs[i], hyps_norm, factors, i)
        selected_factors.append(selected)
        oracle_factors.append(oracle)
        selected_preds.append(hyps_norm[selected][i])
        oracle_preds.append(hyps_norm[oracle][i])
        details.append(
            {
                "audio_filepath": items[i]["audio"],
                "duration": items[i]["duration"],
                "text": items[i]["ref_raw"],
                "text_normalized": refs[i],
                "selected_factor": selected,
                "oracle_factor": oracle,
                "selected_cer": round(O.per_utt_cer(refs[i], selected_preds[i]), 4),
                "oracle_cer": round(O.per_utt_cer(refs[i], oracle_preds[i]), 4),
                "per_factor": {
                    str(f): {
                        "pred_text": hyps_fmt[f][i],
                        "pred_text_normalized": hyps_norm[f][i],
                        "raw_score": raw_scores[f][i],
                        "selection_score": select_scores[f][i],
                        "cer": round(O.per_utt_cer(refs[i], hyps_norm[f][i]), 4),
                    }
                    for f in factors
                },
            }
        )

    selected_corpus = 100 * word_error_rate(hypotheses=selected_preds, references=refs, use_cer=True)
    oracle_corpus = 100 * word_error_rate(hypotheses=oracle_preds, references=refs, use_cer=True)
    selected_counts = Counter(selected_factors)
    oracle_counts = Counter(oracle_factors)
    agreement = sum(s == o for s, o in zip(selected_factors, oracle_factors)) / n

    lines = []
    p = lines.append
    p("=" * 88)
    p(f"LIKELIHOOD TIME-WARP (CER)  |  set={set_name}  method={args.method}  score_norm={args.score_norm}")
    p("=" * 88)
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
    p(f"baseline (x1.0)             : {factor_corpus[1.0]:6.2f} %")
    p(f"best single fixed (x{best_fixed})    : {factor_corpus[best_fixed]:6.2f} %")
    p(f"LIKELIHOOD selected         : {selected_corpus:6.2f} %")
    p(f"ORACLE best-of-{len(factors)}           : {oracle_corpus:6.2f} %")
    p(f"selector/oracle agreement   : {100 * agreement:6.2f} %")
    p("")
    p("likelihood pick distribution:")
    for f in factors:
        c = selected_counts.get(f, 0)
        p(f"  x{f:<5}: {c:>6}  ({100 * c / n:5.1f}%)  {'#' * int(round(40 * c / n))}")
    p("")
    p("oracle pick distribution:")
    for f in factors:
        c = oracle_counts.get(f, 0)
        p(f"  x{f:<5}: {c:>6}  ({100 * c / n:5.1f}%)  {'#' * int(round(40 * c / n))}")
    report = "\n".join(lines)
    print(report)

    selected_pick_summary = pct_summary(selected_counts, factors, n)
    oracle_pick_summary = pct_summary(oracle_counts, factors, n)
    factor_cer_summary = ",".join(f"x{f}:{factor_corpus[f]:.2f}" for f in factors)
    print(
        f"LIKELIHOOD_SUMMARY {set_name} scored={n} total={n} "
        f"baseline={factor_corpus[1.0]:.2f} best_fixed={factor_corpus[best_fixed]:.2f} "
        f"selected={selected_corpus:.2f} oracle={oracle_corpus:.2f} agreement={100 * agreement:.2f} "
        f"factor_cers={factor_cer_summary} picks={selected_pick_summary} oracle_picks={oracle_pick_summary}"
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
    ap.add_argument("--audio_src_prefix", default="/data/mandarin/aishell2/evaluation/aishell")
    ap.add_argument("--audio_dst_prefix", default="")
    ap.add_argument("--text_key", default="text", help="Reference text field in the manifest.")
    ap.add_argument("--factors", default="0.9,1.0,1.1", help="Comma-separated warp factors; 1.0 is auto-added.")
    ap.add_argument("--method", default="speed", choices=["speed", "time_stretch"])
    ap.add_argument("--score_norm", default="logprob_token",
                    choices=["logprob_token", "logprob_token_dur", "none", "token", "char"])
    ap.add_argument("--score_epsilon", type=float, default=0.01,
                    help="A non-1.0 factor is selected only if its score exceeds the x1.0 score "
                         "by at least this value. Set 0 to disable (pure argmax). "
                         "Empirically 0.01 is a good default for logprob_token scores. (default: 0.01)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_eval_samples", type=int, default=None)
    ap.add_argument("--max_symbols_per_step", type=int, default=None)
    ap.add_argument("--keep_spaces", action="store_true")
    ap.add_argument("--keep_warped", action="store_true")
    ap.add_argument("--output", default=None)
    main(ap.parse_args())
