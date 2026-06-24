#!/usr/bin/env python3
"""Backward (right-to-left) inference for a bidirectional HAINAN / TDT model.

A ``EncDecBidirectionalTDTBPEModel`` has a shared encoder plus TWO (prediction-net,
joint) pairs: a forward (L2R) branch -- ``model.decoder`` / ``model.joint`` /
``model.decoding``, used by the normal ``model.transcribe`` -- and a backward (R2L)
branch -- ``model.decoder_bwd`` / ``model.joint_bwd`` / ``model.decoding_bwd``.

The backward branch was trained on the **time-reversed** encoder output aligned to the
**reversed** labels. So backward inference is:

    encoder(audio) -> reverse along time -> greedy-decode with decoding_bwd
                   -> un-reverse the predicted token ids -> detokenize

which yields normal forward-order text, directly comparable to the forward branch.
This mirrors ``EncDecBidirectionalTDTBPEModel._backward_wer_counts`` (training-time
backward WER), just wrapped with audio loading + batching for standalone inference.

Usage
-----
    # From a NeMo manifest (audio_filepath [, text]); reports WER if `text` present.
    python backward_inference.py --model /path/model.nemo --manifest val.json --device 0

    # From explicit audio files (no refs -> just prints transcripts).
    python backward_inference.py --model /path/model.nemo --audio a.wav b.wav

    # Also decode the forward branch for a side-by-side comparison.
    python backward_inference.py --model /path/model.nemo --manifest val.json --compare_forward
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import run_eval_asr as R  # noqa: E402  (load_model + EnglishTextNormalizer)

try:
    import jiwer
except ImportError:  # pragma: no cover
    jiwer = None


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_inputs(args):
    """Return parallel lists (audio_paths, refs) from a manifest and/or --audio."""
    audio_paths, refs = [], []
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                path = d.get("audio_filepath") or d.get("audio")
                if not path:
                    continue
                if not os.path.isabs(path) and args.audio_root:
                    path = os.path.join(args.audio_root, path)
                audio_paths.append(path)
                refs.append(d.get("text"))
    for a in args.audio or []:
        audio_paths.append(a)
        refs.append(None)
    if args.max_samples:
        audio_paths = audio_paths[: args.max_samples]
        refs = refs[: args.max_samples]
    return audio_paths, refs


class _AudioDS(torch.utils.data.Dataset):
    """Loads audio, downmixes to mono, resamples to the model sample rate."""

    def __init__(self, audio_files, target_sr):
        self.audio_files = audio_files
        self.target_sr = target_sr

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio, sr = soundfile.read(self.audio_files[idx], dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != self.target_sr:
            import torchaudio

            audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, self.target_sr).numpy()
        audio = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        return audio, torch.tensor(audio.numel(), dtype=torch.long)


def _collate(batch):
    audios, lens = zip(*batch)
    lens = torch.stack(lens)
    max_len = int(lens.max().item())
    padded = torch.zeros(len(audios), max_len, dtype=torch.float32)
    for i, a in enumerate(audios):
        padded[i, : a.numel()] = a
    return padded, lens


# --------------------------------------------------------------------------- #
# Backward decode
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def decode_backward(model, audio_files, batch_size, target_sr):
    """Greedy backward (R2L) decode -> forward-order text, in input order."""
    loader = torch.utils.data.DataLoader(
        _AudioDS(audio_files, target_sr),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    texts = []
    for signal, signal_len in tqdm(loader, desc="Backward decode"):
        signal = signal.to(model.device, non_blocking=True)
        signal_len = signal_len.to(model.device, non_blocking=True)
        encoded, encoded_len = model.forward(input_signal=signal, input_signal_length=signal_len)
        # encoded is [B, D, T] (channel-first); reverse valid frames per-sample.
        enc_rev = model._reverse_time(encoded, encoded_len)
        hyps = model.decoding_bwd.rnnt_decoder_predictions_tensor(
            encoder_output=enc_rev, encoded_lengths=encoded_len, return_hypotheses=True
        )
        if isinstance(hyps, tuple):
            hyps = hyps[0]
        for hyp in hyps:
            ids = hyp.y_sequence
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            ids = list(ids)[::-1]  # un-reverse R2L prediction -> forward token order
            texts.append(model.decoding_bwd.decode_ids_to_str(ids))
    return texts


def corpus_wer(refs, hyps):
    """Corpus WER (%) on whitespace tokens; jiwer if available, else local DP."""
    pairs = [(r, h) for r, h in zip(refs, hyps) if r is not None]
    if not pairs:
        return None, 0
    rs = [r for r, _ in pairs]
    hs = [h if h.strip() else " " for _, h in pairs]
    if jiwer is not None:
        return 100.0 * jiwer.wer(rs, hs), len(pairs)

    def _ed(a, b):
        n, m = len(a), len(b)
        if n == 0:
            return m
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            cur = [i] + [0] * m
            for j in range(1, m + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            prev = cur
        return prev[m]

    edits = words = 0
    for r, h in zip(rs, hs):
        rt, ht = r.split(), h.split()
        edits += _ed(ht, rt)
        words += len(rt)
    return 100.0 * edits / max(words, 1), len(pairs)


def main(args):
    torch.set_float32_matmul_precision("medium")
    device = torch.device(
        f"cuda:{args.device}" if (args.device is not None and args.device >= 0 and torch.cuda.is_available()) else "cpu"
    )

    model, _ = R.load_model(args.model, device, tokenizer_dir=args.tokenizer_dir)
    if not hasattr(model, "decoding_bwd"):
        sys.exit(
            f"ERROR: {type(model).__name__} has no backward branch (decoding_bwd). "
            "This script requires a bidirectional model (EncDecBidirectionalTDTBPEModel)."
        )
    if args.max_symbols is not None and hasattr(model.decoding_bwd, "decoding"):
        if hasattr(model.decoding_bwd.decoding, "max_symbols"):
            model.decoding_bwd.decoding.max_symbols = args.max_symbols

    try:
        target_sr = int(model.cfg.preprocessor.sample_rate)
    except Exception:  # pragma: no cover - defensive
        target_sr = 16000

    audio_files, refs = load_inputs(args)
    if not audio_files:
        sys.exit("No inputs: pass --manifest and/or --audio.")
    print(f"Backward inference: {len(audio_files)} utt(s) | sr={target_sr} | device={device}")

    t0 = time.time()
    bwd_texts = decode_backward(model, audio_files, args.batch_size, target_sr)
    bwd_time = time.time() - t0

    fwd_texts = None
    if args.compare_forward:
        t0 = time.time()
        fwd_texts = R.transcribe_tdt(model, audio_files, args.batch_size)
        fwd_time = time.time() - t0

    # Normalize for (leaderboard-comparable) WER; keep raw text in the output file.
    norm = R.text_normalizer
    refs_norm = [norm(r) if r is not None else None for r in refs]
    bwd_norm = [norm(t) for t in bwd_texts]
    bwd_wer, scored = corpus_wer(refs_norm, bwd_norm)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            for i, path in enumerate(audio_files):
                rec = {"audio_filepath": path, "pred_text_backward": bwd_texts[i]}
                if refs[i] is not None:
                    rec["text"] = refs[i]
                if fwd_texts is not None:
                    rec["pred_text_forward"] = fwd_texts[i]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[per-utterance output -> {os.path.abspath(args.output)}]")

    if args.verbose:
        print("\n" + "=" * 70)
        for i in range(len(audio_files)):
            if refs[i] is not None:
                print(f"[{i}] REF : {refs[i]}")
            print(f"[{i}] BWD : {bwd_texts[i]}")
            if fwd_texts is not None:
                print(f"[{i}] FWD : {fwd_texts[i]}")
            print()
        print("=" * 70)

    print("\n================== Backward inference summary ==================")
    print(f"model      : {os.path.basename(args.model)}")
    print(f"utterances : {len(audio_files)}  (scored against refs: {scored})")
    print(f"backward   : {bwd_time:.1f}s")
    if bwd_wer is not None:
        print(f"BACKWARD WER : {bwd_wer:.2f} %")
    if fwd_texts is not None:
        fwd_norm = [norm(t) for t in fwd_texts]
        fwd_wer, _ = corpus_wer(refs_norm, fwd_norm)
        print(f"forward      : {fwd_time:.1f}s")
        if fwd_wer is not None:
            print(f"FORWARD  WER : {fwd_wer:.2f} %")
    print("================================================================")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Path to a bidirectional .nemo (or .ckpt).")
    ap.add_argument("--manifest", default=None, help="NeMo manifest jsonl (audio_filepath [, text]).")
    ap.add_argument("--audio", nargs="*", default=None, help="Explicit audio file(s).")
    ap.add_argument("--audio_root", default=None, help="Prefix for relative audio_filepath in the manifest.")
    ap.add_argument("--tokenizer_dir", default=None, help="External tokenizer dir (for .ckpt inputs).")
    ap.add_argument("--device", type=int, default=0, help="GPU id (-1 for CPU).")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_symbols", type=int, default=None, help="Cap greedy symbols/step on the backward decoder.")
    ap.add_argument("--compare_forward", action="store_true", help="Also run the forward branch for comparison.")
    ap.add_argument("--output", default=None, help="Write per-utterance predictions to this jsonl.")
    ap.add_argument("--verbose", action="store_true")
    main(ap.parse_args())
