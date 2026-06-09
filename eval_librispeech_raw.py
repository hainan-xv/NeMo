"""
LibriSpeech test-clean eval that stores RAW (un-normalized) text.

Unlike ``run_eval_asr.py`` (which whisper-normalizes both the reference and the
hypothesis before writing the manifest), this driver keeps the model's raw
decoded text and the dataset's raw reference verbatim, so you can inspect
casing / punctuation. It reuses the model loading + transcription helpers from
``run_eval_asr.py``.

Invoked via ``eval_librispeech_clean.sh``.
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile
import torch
from tqdm import tqdm

# Match run_eval_asr.py: force soundfile decoding before importing datasets.
os.environ["HF_AUDIO_DECODER"] = "soundfile"

from datasets import Audio, load_dataset  # noqa: E402

_EVAL_ROOT = Path(__file__).resolve().parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))

# Reuse the existing, battle-tested helpers (auto-detect model class, decode).
from run_eval_asr import (  # noqa: E402
    get_text,
    load_model,
    transcribe_multistream,
    transcribe_tdt,
)


def main(args):
    data_cache_dir = os.path.join(os.getcwd(), "audio_cache")
    cache_dir = os.path.join(data_cache_dir, args.dataset, args.split)
    os.makedirs(cache_dir, exist_ok=True)

    torch.set_float32_matmul_precision("medium")
    device = torch.device(
        f"cuda:{args.device}"
        if (args.device is not None and args.device >= 0 and torch.cuda.is_available())
        else "cpu"
    )

    model, is_multistream = load_model(args.model, device, tokenizer_dir=args.tokenizer_dir)

    print(f"Loading dataset: {args.dataset_path}/{args.dataset} split={args.split}")
    dataset = load_dataset(
        path=args.dataset_path,
        name=args.dataset,
        split=args.split,
        streaming=args.streaming,
        token=True,
        trust_remote_code=True,
    )
    dataset = dataset.cast_column("audio", Audio(decode=False))

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling to first {args.max_eval_samples} samples")
        dataset = dataset.take(args.max_eval_samples)

    print("Downloading and caching audio samples...")
    all_data = {"audio_filepaths": [], "durations": [], "references": [], "ids": []}
    for sample in tqdm(dataset, desc="Processing samples"):
        # RAW reference: no normalization.
        ref = get_text(sample)
        if not ref.strip():
            continue
        raw_audio = sample["audio"]
        sample_id = sample["id"].replace("/", "_").removesuffix(".wav")
        audio_path = os.path.join(cache_dir, f"{sample_id}.wav")
        if not os.path.exists(audio_path):
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            if isinstance(raw_audio, dict):
                if "array" in raw_audio and raw_audio["array"] is not None:
                    audio_array = np.float32(raw_audio["array"])
                    sr = raw_audio.get("sampling_rate", 16000)
                elif "bytes" in raw_audio and raw_audio["bytes"]:
                    with io.BytesIO(raw_audio["bytes"]) as f:
                        audio_array, sr = soundfile.read(f, dtype="float32")
                elif "path" in raw_audio and raw_audio["path"] and os.path.exists(raw_audio["path"]):
                    audio_array, sr = soundfile.read(raw_audio["path"], dtype="float32")
                else:
                    print(f"  WARNING: cannot decode audio for {sample_id}, skipping")
                    continue
            else:
                print(f"  WARNING: unexpected audio format for {sample_id}: {type(raw_audio)}, skipping")
                continue
            if sr != 16000:
                import torchaudio

                t = (
                    torch.from_numpy(audio_array).unsqueeze(0)
                    if audio_array.ndim == 1
                    else torch.from_numpy(audio_array)
                )
                t = torchaudio.functional.resample(t, sr, 16000)
                audio_array = t.squeeze(0).numpy()
            soundfile.write(audio_path, audio_array, 16000)
        info = soundfile.info(audio_path)
        all_data["audio_filepaths"].append(audio_path)
        all_data["durations"].append(info.duration)
        all_data["references"].append(ref)
        all_data["ids"].append(sample_id)

    # Sort by duration (desc) for efficient batching; both decode paths preserve order.
    sorted_idx = sorted(range(len(all_data["durations"])), key=lambda k: all_data["durations"][k], reverse=True)
    for key in all_data:
        all_data[key] = [all_data[key][i] for i in sorted_idx]

    n_samples = len(all_data["references"])
    print(f"Total samples: {n_samples}")
    if n_samples == 0:
        print("ERROR: No samples to evaluate!")
        return

    print(f"Transcribing {n_samples} samples ({'multistream' if is_multistream else 'tdt'})...")
    start = time.time()
    if is_multistream:
        transcriptions = transcribe_multistream(model, all_data["audio_filepaths"], args.batch_size)
    else:
        transcriptions = transcribe_tdt(model, all_data["audio_filepaths"], args.batch_size)
    total_time = time.time() - start

    # RAW predictions: strip whitespace only, no whisper normalization.
    predictions = [pred.strip() for pred in transcriptions]

    results = []
    for i in range(n_samples):
        results.append(
            {
                "id": all_data["ids"][i],
                "audio_filepath": all_data["audio_filepaths"][i],
                "duration": all_data["durations"][i],
                "ref": all_data["references"][i],
                "hyp": predictions[i],
            }
        )

    payload = {
        "model": os.path.abspath(args.model),
        "dataset_path": args.dataset_path,
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": n_samples,
        "normalized": False,
        "results": results,
    }

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    audio_length = sum(all_data["durations"])
    rtfx = round(audio_length / total_time, 2) if total_time > 0 else float("nan")
    print(f"RTFX: {rtfx}")
    print("Raw results saved at:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LibriSpeech test-clean eval storing RAW text")
    parser.add_argument("--model", type=str, required=True, help="Path to a .nemo or Lightning .ckpt file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path for raw ref/hyp pairs")
    parser.add_argument("--tokenizer_dir", type=str, default=None, help="Override tokenizer dir for .ckpt loads")
    parser.add_argument("--dataset_path", type=str, default="hf-audio/esb-datasets-test-only-sorted")
    parser.add_argument("--dataset", type=str, default="librispeech")
    parser.add_argument("--split", type=str, default="test.clean")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.set_defaults(streaming=True)
    main(parser.parse_args())
