# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Self-contained Open-ASR-Leaderboard evaluator for a NeMo RNNT/CHAT ASR model.

Loads a single ASR model (``.nemo`` or Lightning ``.ckpt``), pulls the ESB
Open-ASR-Leaderboard test sets from HuggingFace (caching 16 kHz wavs so each set
is fetched only once), transcribes them, and reports WER per dataset.

This is the ASR counterpart of the SpeechLM ``run_eval_sslm.py`` driver and is
driven by ``eval_leaderboard_chat.sh``. It is intentionally small and dependency
light so it is easy to run locally for debugging (e.g. ``--max_eval_samples 10
--verbose`` prints reference/hypothesis pairs per utterance).

The model is loaded ONCE and reused for every dataset.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def _log(msg: str) -> None:
    print(msg, flush=True)


# Open ASR Leaderboard references are normalized (lowercase, no punctuation).
# Our CHAT model emits punctuation + capitalization, so normalize BOTH sides
# before scoring, matching the leaderboard convention. Disable with --no_normalize.
_PUNCT = str.maketrans({c: " " for c in ".,?!;:\"()[]{}<>/\\|`~@#$%^&*_=+"})


def normalize_text(text: str) -> str:
    """Lowercase, strip most punctuation (keep intra-word apostrophes/hyphens), collapse spaces."""
    if text is None:
        return ""
    text = text.lower().strip()
    text = text.replace("’", "'")
    # SentencePiece <unk> surface (U+2047). The model emits <unk> where the tokenizer
    # lacks a character (e.g. curly quotes in PnC text); it is not a real word, so
    # drop it rather than let it count as an insertion error.
    text = text.replace("\u2047", " ")
    text = text.translate(_PUNCT)
    # drop stray apostrophes/hyphens that are not intra-word
    text = " ".join(tok.strip("'-") for tok in text.split())
    return " ".join(text.split())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True, help="Path to the ASR model (.nemo preferred, or Lightning .ckpt).")
    p.add_argument(
        "--manifest",
        default=None,
        help="Evaluate a NeMo manifest (audio_filepath+text) directly INSTEAD of the ESB sets. "
        "Use this to reproduce the training-time validation locally (e.g. the dev manifest).",
    )
    p.add_argument(
        "--dataset_path",
        default="hf-audio/esb-datasets-test-only-sorted",
        help="HuggingFace dataset repo id for the ESB leaderboard sets.",
    )
    p.add_argument(
        "--datasets",
        default="ami:test,earnings22:test,gigaspeech:test,librispeech:test.clean,"
        "librispeech:test.other,spgispeech:test,tedlium:test,voxpopuli:test",
        help="Comma-separated <config>:<split> entries to evaluate.",
    )
    p.add_argument("--audio_cache_dir", default=None, help="Where to cache 16 kHz wavs + manifests per split.")
    p.add_argument("--device", default="0", help="CUDA device index, or 'cpu'.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_eval_samples", type=int, default=None, help="Cap utterances per dataset (fast iteration).")
    p.add_argument("--gt_text_field", default=None, help="Reference text field in the HF dataset (auto-detected).")
    p.add_argument("--verbose", action="store_true", help="Print per-utterance reference/hypothesis pairs.")
    p.add_argument("--no_normalize", action="store_true", help="Do NOT normalize text before WER (score raw PnC).")
    p.add_argument(
        "--dump_dir",
        default=None,
        help="If set, write per-utterance {wav, ref, hyp, ref_norm, hyp_norm} JSONL per dataset here "
        "(for offline error-pattern analysis, e.g. scripts/analyze_asr_errors.py).",
    )
    p.add_argument("--refresh_cache", action="store_true", help="Ignore cached wavs and re-fetch from HuggingFace.")
    # Optional decoding / streaming overrides (useful for CHAT debugging).
    p.add_argument("--max_symbols", type=int, default=None, help="Override greedy max symbols per (chunk) step.")
    p.add_argument("--att_context_size", default=None, help="Override encoder att context, e.g. '[70,13]'.")
    p.add_argument(
        "--chunk_size", type=int, default=None, help="Override CHAT joint chunk_size (for full-context models)."
    )
    return p.parse_args()


def load_model(model_path: str, device: str):
    import torch
    from nemo.collections.asr.models import ASRModel, EncDecRNNTBPEModel

    map_location = "cpu"
    _log(f"==> Loading ASR model: {model_path}")
    if model_path.endswith(".nemo"):
        model = ASRModel.restore_from(model_path, map_location=map_location)
    elif model_path.endswith(".ckpt"):
        # Lightning checkpoint: relies on the model cfg being stored in hparams and
        # the tokenizer being reachable. Prefer a .nemo when possible.
        model = EncDecRNNTBPEModel.load_from_checkpoint(model_path, map_location=map_location)
    else:
        raise ValueError(f"Unsupported model file (expected .nemo or .ckpt): {model_path}")

    if device != "cpu" and torch.cuda.is_available():
        model = model.to(f"cuda:{device}")
    else:
        model = model.to("cpu")
    model.eval()
    return model


def maybe_override_decoding(model, args) -> None:
    """Apply optional att_context / chunk_size / max_symbols overrides."""
    # Encoder attention context (cache-aware streaming models).
    if args.att_context_size is not None and hasattr(model.encoder, "set_default_att_context_size"):
        import ast

        ctx = ast.literal_eval(args.att_context_size)
        model.encoder.set_default_att_context_size(ctx)
        _log(f"    override att_context_size={ctx}")

    # CHAT joint chunk_size (explicit for full-context models).
    if args.chunk_size is not None and hasattr(model, "joint") and hasattr(model.joint, "chunk_size"):
        model.joint.chunk_size = args.chunk_size
        _log(f"    override joint.chunk_size={args.chunk_size}")

    # Greedy max symbols per step (per chunk in CHAT).
    if args.max_symbols is not None:
        try:
            from omegaconf import open_dict

            decoding_cfg = model.cfg.decoding
            with open_dict(decoding_cfg):
                decoding_cfg.greedy.max_symbols = args.max_symbols
            model.change_decoding_strategy(decoding_cfg)
            _log(f"    override greedy.max_symbols={args.max_symbols}")
        except Exception as e:  # noqa: BLE001
            _log(f"    WARNING: could not override max_symbols: {e}")


def prepare_split(dataset_path, config, split, cache_dir, max_samples, gt_field, refresh):
    """Materialize a split to 16 kHz wavs + return (wav_paths, references)."""
    import soundfile as sf

    split_cache = os.path.join(cache_dir, config, split.replace(".", "_"))
    os.makedirs(split_cache, exist_ok=True)
    done_marker = os.path.join(split_cache, f".done_{max_samples if max_samples else 'all'}")

    # Reuse cache if complete (manifest of paths + refs written).
    manifest = os.path.join(split_cache, "manifest.tsv")
    if os.path.exists(done_marker) and os.path.exists(manifest) and not refresh:
        wav_paths, refs = [], []
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                path, ref = line.rstrip("\n").split("\t", 1)
                wav_paths.append(path)
                refs.append(ref)
        _log(f"    reusing cached {config}/{split}: {len(wav_paths)} utts")
        return wav_paths, refs

    import io

    import numpy as np
    from datasets import Audio, load_dataset

    _log(f"    downloading {config}/{split} from {dataset_path} ...")
    ds = load_dataset(dataset_path, config, split=split, streaming=True)
    # Disable HuggingFace's built-in audio decoding (which pulls in torchcodec +
    # FFmpeg and crashes when those libs are missing). We decode the raw bytes
    # ourselves with soundfile (libsndfile handles WAV/FLAC without FFmpeg).
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:  # noqa: BLE001 - column may already be undecoded
        pass

    def _to_16k_mono(data, sr):
        data = np.asarray(data, dtype="float32")
        if data.ndim > 1:  # stereo -> mono
            data = data.mean(axis=1)
        if sr != 16000:
            try:
                import librosa

                data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            except Exception:  # noqa: BLE001 - fall back to scipy polyphase resample
                from math import gcd

                from scipy.signal import resample_poly

                g = gcd(int(sr), 16000)
                data = resample_poly(data, 16000 // g, int(sr) // g).astype("float32")
        return data

    wav_paths, refs = [], []
    n = 0
    for sample in ds:
        if max_samples is not None and n >= max_samples:
            break
        # Reference field auto-detection.
        field = gt_field
        if field is None:
            for cand in ("text", "norm_text", "sentence", "transcription", "transcript"):
                if cand in sample and isinstance(sample[cand], str):
                    field = cand
                    break
        ref = sample.get(field, "") if field else ""
        audio = sample["audio"]
        if isinstance(audio, dict) and audio.get("bytes") is not None:
            wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        elif isinstance(audio, dict) and audio.get("path"):
            wav, sr = sf.read(audio["path"], dtype="float32")
        elif isinstance(audio, dict) and audio.get("array") is not None:
            wav, sr = audio["array"], audio["sampling_rate"]
        else:
            raise ValueError(f"Unrecognized audio entry for {config}/{split}: keys={list(audio) if isinstance(audio, dict) else type(audio)}")
        wav = _to_16k_mono(wav, sr)
        wav_path = os.path.join(split_cache, f"{n:06d}.wav")
        sf.write(wav_path, wav, 16000)
        wav_paths.append(wav_path)
        refs.append(ref)
        n += 1

    with open(manifest, "w", encoding="utf-8") as f:
        for path, ref in zip(wav_paths, refs):
            f.write(f"{path}\t{ref}\n")
    open(done_marker, "w").close()
    _log(f"    cached {len(wav_paths)} utts -> {split_cache}")
    return wav_paths, refs


def prepare_manifest(manifest_path, max_samples, gt_field):
    """Read a NeMo manifest (audio_filepath + text) -> (audio_paths, references).

    Mirrors the training-time validation input so the val WER can be reproduced
    locally. Assumes audio_filepath is readable from this box.
    """
    import json

    audio_paths, refs = [], []
    field = gt_field or "text"
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_samples is not None and len(audio_paths) >= max_samples:
                break
            data = json.loads(line)
            path = data.get("audio_filepath")
            if path is None:
                continue
            audio_paths.append(path)
            refs.append(data.get(field, data.get("text", "")))
    _log(f"    manifest {os.path.basename(manifest_path)}: {len(audio_paths)} utts")
    return audio_paths, refs


def transcribe(model, wav_paths, batch_size):
    hyps = model.transcribe(wav_paths, batch_size=batch_size, num_workers=0)
    # NeMo may return a list, or a tuple (best, all) for some models.
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    texts = []
    for h in hyps:
        if isinstance(h, str):
            texts.append(h)
        elif hasattr(h, "text"):
            texts.append(h.text)
        else:
            texts.append(str(h))
    return texts


def main() -> int:
    args = parse_args()
    if args.audio_cache_dir is None:
        args.audio_cache_dir = os.path.join(os.path.expanduser("~"), "leaderboard_run", "cache")
    os.makedirs(args.audio_cache_dir, exist_ok=True)

    from nemo.collections.asr.metrics.wer import word_error_rate

    model = load_model(args.model_path, args.device)
    maybe_override_decoding(model, args)

    # Build the list of (key, loader) tasks. --manifest overrides the ESB sets.
    if args.manifest:
        entries = [("manifest:" + os.path.basename(args.manifest), None, None)]
    else:
        entries = []
        for e in (x.strip() for x in args.datasets.split(",")):
            if not e:
                continue
            config, split = e.split(":", 1) if ":" in e else (e, "test")
            entries.append((f"{config}/{split}", config, split))

    results = {}
    for key, config, split in entries:
        _log(f"\n==> {key}")
        t0 = time.time()
        try:
            if args.manifest:
                wav_paths, refs = prepare_manifest(args.manifest, args.max_eval_samples, args.gt_text_field)
            else:
                wav_paths, refs = prepare_split(
                    args.dataset_path,
                    config,
                    split,
                    args.audio_cache_dir,
                    args.max_eval_samples,
                    args.gt_text_field,
                    args.refresh_cache,
                )
            if not wav_paths:
                _log(f"    no utterances for {key}; skipping")
                results[key] = None
                continue
            hyps = transcribe(model, wav_paths, args.batch_size)

            if args.no_normalize:
                refs_s, hyps_s = refs, hyps
            else:
                refs_s = [normalize_text(r) for r in refs]
                hyps_s = [normalize_text(h) for h in hyps]

            if args.verbose:
                for i, (r, h) in enumerate(zip(refs_s, hyps_s)):
                    _log(f"    [{i}] REF: {r}")
                    _log(f"    [{i}] HYP: {h}")

            # Per-utterance dump for offline error-pattern analysis (per condition).
            if args.dump_dir:
                os.makedirs(args.dump_dir, exist_ok=True)
                safe_key = key.replace("/", "__").replace(":", "_")
                dump_path = os.path.join(args.dump_dir, f"{safe_key}.jsonl")
                with open(dump_path, "w", encoding="utf-8") as df:
                    for wp, r_raw, h_raw, r_n, h_n in zip(wav_paths, refs, hyps, refs_s, hyps_s):
                        df.write(
                            json.dumps(
                                {
                                    "dataset": key,
                                    "wav": os.path.basename(str(wp)),
                                    "ref": r_raw,
                                    "hyp": h_raw,
                                    "ref_norm": r_n,
                                    "hyp_norm": h_n,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                _log(f"    dumped per-utterance results -> {dump_path}")

            wer = word_error_rate(hypotheses=hyps_s, references=refs_s, use_cer=False)
            dt = time.time() - t0
            results[key] = wer
            _log(f"    {key}  WER={wer * 100:.2f}%  ({len(wav_paths)} utts, {dt:.1f}s)")
        except Exception as e:  # noqa: BLE001
            import traceback

            _log(f"    ERROR on {key}: {e}")
            traceback.print_exc()
            results[key] = None

    # Summary table.
    _log("\n" + "=" * 44)
    _log(f"  {'Dataset':<28} {'WER (%)':>8}")
    _log(f"  {'-' * 28} {'-' * 8}")
    total, n = 0.0, 0
    for key, wer in results.items():
        if wer is None:
            _log(f"  {key:<28} {'ERR':>8}")
        else:
            _log(f"  {key:<28} {wer * 100:>8.2f}")
            total += wer * 100
            n += 1
    if n:
        _log(f"  {'-' * 28} {'-' * 8}")
        _log(f"  {'Average':<28} {total / n:>8.2f}")
    _log("=" * 44)
    return 0


if __name__ == "__main__":
    sys.exit(main())
