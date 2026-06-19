"""
Character Error Rate (CER) eval for Mandarin ASR checkpoints on a local
NeMo-format manifest (e.g. the AISHELL-2 test sets under
``/home/hainanx/Workplace/data/aishell_eval/aishell/manifests``).

This is the Mandarin sibling of ``run_eval_asr.py``.  Two important differences
from the English leaderboard driver:

  * it reads a *local* NeMo manifest (``audio_filepath`` + ``text``) instead of an
    HF dataset, and it can remap the manifest's stored ``audio_filepath`` prefix
    onto wherever the audio actually lives on this box; and
  * it scores with **CER** (``nemo ... word_error_rate(use_cer=True)``) and does
    *no* English/whisper normalization -- for Mandarin we only collapse
    whitespace (configurable), which matches the usual AISHELL CER convention.

Decoding reuses the checkpoint's *own* embedded decoding config via the model's
``transcribe`` (TDT durations, CHAT cross-attention joint, ...), so the same
script handles both the ``tdt`` and ``chat`` Mandarin recipes without per-model
decoding overrides.  Model loading / transcription dispatch is reused from
``run_eval_asr.py`` so behaviour stays in sync with the English path.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Import the eval helpers that live next to this script so we reuse the exact
# same model-loading + transcription code paths as the English leaderboard.
_EVAL_ROOT = Path(__file__).resolve().parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))

import run_eval_asr as R  # noqa: E402

from nemo.collections.asr.metrics.wer import word_error_rate  # noqa: E402


def read_manifest(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def remap_audio_path(p, src_prefix, dst_prefix):
    """Rewrite a manifest audio path onto the local audio tree.

    The manifests were authored on a cluster, so ``audio_filepath`` points at a
    container mount (e.g. ``/data/mandarin/.../aishell``).  If ``dst_prefix`` is
    given and ``p`` starts with ``src_prefix``, swap the prefix; otherwise return
    ``p`` unchanged (it may already resolve locally).
    """
    if dst_prefix and src_prefix and p.startswith(src_prefix):
        rel = p[len(src_prefix):].lstrip("/")
        return os.path.join(dst_prefix, rel)
    return p


def collapse_ws(s):
    return "".join(s.split())


def normalize(s, keep_spaces):
    s = s.strip()
    if not keep_spaces:
        s = collapse_ws(s)
    return s


def transcribe(model, is_multistream, audio_files, batch_size, consistency=False, consistency_weights=None):
    """Dispatch to the right decode path (mirrors run_eval_asr.main)."""
    loss_type = getattr(model, "loss_type", None)
    is_aligner_like = loss_type in ("aligner", "chunked_aligner")
    if consistency:
        if not getattr(model, "multi_target_enabled", False):
            raise SystemExit("--consistency requires a multi_target model (token + pronunciation heads).")
        return R.transcribe_consistency(model, audio_files, batch_size, head_weights=consistency_weights)
    if is_multistream:
        return R.transcribe_multistream(model, audio_files, batch_size)
    if is_aligner_like:
        return R.transcribe_aligner_like(model, audio_files, batch_size)
    return R.transcribe_tdt(model, audio_files, batch_size)


def main(args):
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

    rows = read_manifest(args.manifest)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        rows = rows[: args.max_eval_samples]

    audio_files, references, durations = [], [], []
    missing = 0
    for r in rows:
        ref = r.get(args.text_key, r.get("text", ""))
        if ref is None or not str(ref).strip():
            continue
        ap = remap_audio_path(r["audio_filepath"], args.audio_src_prefix, args.audio_dst_prefix)
        if not os.path.exists(ap):
            missing += 1
            if missing <= 5:
                print(f"  WARNING: missing audio, skipping: {ap}")
            continue
        audio_files.append(ap)
        references.append(str(ref))
        durations.append(float(r.get("duration", 0.0) or 0.0))

    if missing:
        print(f"  ({missing} entries skipped: audio file not found)")
    n = len(audio_files)
    print(f"Manifest: {args.manifest}")
    print(f"Total scorable samples: {n}")
    if n == 0:
        print("ERROR: nothing to evaluate (check --audio_src_prefix/--audio_dst_prefix).")
        return None

    # Sort by duration (desc) for efficient batching; transcribe paths preserve order.
    order = sorted(range(n), key=lambda k: durations[k], reverse=True)
    audio_files = [audio_files[i] for i in order]
    references = [references[i] for i in order]
    durations = [durations[i] for i in order]

    consistency_weights = None
    if args.consistency_weights:
        consistency_weights = [float(w) for w in args.consistency_weights.replace(",", " ").split()]

    if args.agreement:
        print(f"Scoring cross-head agreement on {n} samples (decoded-context, top-1)...")
        rates = R.head_agreement_rates(model, audio_files, args.batch_size)
        print("\n" + "=" * 70)
        print("Cross-head top-1 agreement (over cells where both heads emit non-blank):")
        order_keys = ["token_notone", "token_tone", "notone_tone", "all3"]
        for pair in order_keys + [k for k in rates if k not in order_keys]:
            if pair not in rates:
                continue
            rate, agree, denom = rates[pair]
            print(f"  {pair:14s}: {100 * rate:6.2f} %  ({agree}/{denom})")
        print("=" * 70)
        return rates

    print(f"Transcribing {n} samples...{' [consistency decode]' if args.consistency else ''}")
    start = time.time()
    hyps = transcribe(
        model,
        is_multistream,
        audio_files,
        args.batch_size,
        consistency=args.consistency,
        consistency_weights=consistency_weights,
    )
    total_time = time.time() - start

    if len(hyps) != n:
        print(f"ERROR: got {len(hyps)} hypotheses for {n} inputs; aborting.")
        return None

    refs_n = [normalize(r, args.keep_spaces) for r in references]
    hyps_n = [normalize(h, args.keep_spaces) for h in hyps]

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for ap, ref, hyp, dur in zip(audio_files, references, hyps, durations):
                f.write(
                    json.dumps(
                        {"audio_filepath": ap, "duration": dur, "text": ref, "pred_text": hyp},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print("Per-utterance results:", os.path.abspath(args.output))

    if args.verbose:
        print("\n" + "=" * 70)
        for i in range(n):
            print(f"[{i}] REF: {references[i]}")
            print(f"[{i}] HYP: {hyps[i]}\n")
        print("=" * 70)

    cer = word_error_rate(hypotheses=hyps_n, references=refs_n, use_cer=True)
    wer = word_error_rate(hypotheses=hyps_n, references=refs_n, use_cer=False)
    cer = round(100 * cer, 2)
    wer = round(100 * wer, 2)
    audio_secs = sum(durations)
    rtfx = round(audio_secs / total_time, 2) if total_time > 0 else 0.0

    print(f"Dataset: {os.path.basename(args.manifest)}")
    print(f"RTFX: {rtfx}")
    print(f"WER: {wer} %")
    print(f"CER: {cer} %")
    return cer, wer, rtfx


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Mandarin CER eval on a local NeMo manifest")
    p.add_argument("--model", required=True, help="Path to a .nemo or Lightning .ckpt file")
    p.add_argument("--manifest", required=True, help="Local NeMo manifest (audio_filepath + text)")
    p.add_argument("--tokenizer_dir", default=None, help="Override tokenizer dir for .ckpt loads (unused for char models)")
    p.add_argument("--audio_src_prefix", default="/data/mandarin/aishell2/evaluation/aishell",
                   help="Manifest audio path prefix to rewrite (set empty to disable remap)")
    p.add_argument("--audio_dst_prefix", default="",
                   help="Local audio root to substitute for --audio_src_prefix")
    p.add_argument("--text_key", default="text", help="Reference text field in the manifest")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--max_symbols_per_step", type=int, default=None)
    p.add_argument("--consistency", action="store_true",
                   help="Multi-target only: consistency-maintaining decode (combine token + "
                        "pronunciation head log-probs per char).")
    p.add_argument("--consistency_weights", default=None,
                   help="Per-head weights for --consistency, ordered 'token,notone[,tone]' "
                        "(comma/space separated). Default: all 1.0.")
    p.add_argument("--agreement", action="store_true",
                   help="Multi-target only: instead of CER, report cross-head top-1 agreement "
                        "rates (token/notone/tone) over the model's own decoded output.")
    p.add_argument("--keep_spaces", action="store_true",
                   help="Do NOT collapse whitespace before CER (default: collapse, AISHELL convention)")
    p.add_argument("--output", default=None, help="Optional per-utterance results manifest (jsonl)")
    p.add_argument("--verbose", action="store_true")
    main(p.parse_args())
