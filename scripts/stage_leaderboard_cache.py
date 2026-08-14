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
"""Stage the Open ASR Leaderboard test sets into the on-disk cache our eval reads.

Downloads each requested config from the consolidated hub dataset
``hf-audio/open-asr-leaderboard`` and materializes it as 16 kHz mono wavs plus a
``_cache_manifest.jsonl`` in the EXACT layout ``speechlm_leaderboard_eval.py``
expects:

    <cache_dir>/<dataset>/<split>/_cache_manifest.jsonl   # {audio_filepath,duration,reference}
    <cache_dir>/<dataset>/<split>/000000.wav ...          # 16 kHz mono

Run this ONCE (on a node with internet + HF access, e.g. a login node or inside
the eval container) to populate the shared lustre cache; the SLURM eval then runs
fully offline off that cache. Idempotent: a split whose ``.done`` marker matches
the requested sample count is skipped (use --refresh to re-stage).

Default suite = the current public leaderboard set (cleaned AMI/GigaSpeech/VoxPopuli,
TED-LIUM dropped), matching ``speechlm_leaderboard_eval.DEFAULT_DATASETS``.
"""
import argparse
import io
import json
import os
import sys
import time

# The current PUBLIC leaderboard suite as "name:split" (config:split on the hub).
DEFAULT_DATASETS = [
    "librispeech:test.clean",
    "librispeech:test.other",
    "ami_cleaned:test",
    "earnings22:test",
    "gigaspeech_cleaned:test",
    "spgispeech:test",
    "voxpopuli_cleaned_aa:test",
]

_REF_FIELDS = ("text", "norm_text", "sentence", "transcription", "transcript", "normalized_text")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _to_16k_mono(data, sr):
    import numpy as np

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


def _reference(sample, gt_field):
    if gt_field:
        return sample.get(gt_field, "") or ""
    for cand in _REF_FIELDS:
        v = sample.get(cand)
        if isinstance(v, str):
            return v
    return ""


def stage_split(dataset_path, dataset, split, cache_dir, max_samples, gt_field, refresh, token):
    """Materialize one (dataset, split) -> 16 kHz wavs + _cache_manifest.jsonl."""
    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    out_dir = os.path.join(cache_dir, dataset, split)  # keep the dot in test.clean
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "_cache_manifest.jsonl")
    done_marker = os.path.join(out_dir, f".done_{max_samples if max_samples else 'all'}")

    if os.path.exists(done_marker) and os.path.exists(manifest_path) and not refresh:
        with open(manifest_path) as f:
            n = sum(1 for line in f if line.strip())
        _log(f"  [skip] {dataset}/{split}: already staged ({n} utts). Use --refresh to rebuild.")
        return n

    _log(f"  downloading {dataset}/{split} from {dataset_path} (streaming) ...")
    ds = load_dataset(dataset_path, dataset, split=split, streaming=True, token=token)
    # Disable HF's built-in audio decode (pulls torchcodec/FFmpeg); decode bytes
    # ourselves with soundfile (libsndfile handles WAV/FLAC without FFmpeg).
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:  # noqa: BLE001 - column may already be undecoded
        pass

    n = 0
    t0 = time.time()
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for sample in ds:
            if max_samples and n >= max_samples:
                break
            ref = _reference(sample, gt_field)
            audio = sample["audio"]
            if isinstance(audio, dict) and audio.get("bytes") is not None:
                wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            elif isinstance(audio, dict) and audio.get("path"):
                wav, sr = sf.read(audio["path"], dtype="float32")
            elif isinstance(audio, dict) and audio.get("array") is not None:
                wav, sr = audio["array"], audio["sampling_rate"]
            else:
                raise ValueError(
                    f"Unrecognized audio entry for {dataset}/{split}: "
                    f"keys={list(audio) if isinstance(audio, dict) else type(audio)}"
                )
            wav = _to_16k_mono(wav, sr)
            wav_path = os.path.join(out_dir, f"{n:06d}.wav")
            sf.write(wav_path, np.ascontiguousarray(wav), 16000)
            mf.write(
                json.dumps(
                    {"audio_filepath": wav_path, "duration": round(len(wav) / 16000.0, 3), "reference": ref}
                )
                + "\n"
            )
            n += 1
            if n % 200 == 0:
                _log(f"    {dataset}/{split}: {n} utts ({time.time() - t0:.0f}s)")

    open(done_marker, "w").close()
    _log(f"  staged {n} utts -> {out_dir} ({time.time() - t0:.0f}s)")
    return n


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", required=True, help="Cache root (writes <dataset>/<split>/ under here).")
    p.add_argument("--dataset_path", default="hf-audio/open-asr-leaderboard", help="Hub dataset to pull from.")
    p.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma/space-separated 'name:split' list (default = current public suite).",
    )
    p.add_argument("--max_samples", type=int, default=0, help="Cap utts per split (0 = all; use e.g. 10 for a smoke test).")
    p.add_argument("--gt_field", default=None, help="Force the reference column (default: auto-detect).")
    p.add_argument("--refresh", action="store_true", help="Re-stage even if a split is already cached.")
    return p.parse_args()


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or True  # use env token, else stored HF login
    entries = []
    for e in (x.strip() for x in args.datasets.replace(",", " ").split()):
        if not e:
            continue
        name, _, split = e.partition(":")
        entries.append((name, split or "test"))

    _log(f"Staging {len(entries)} splits from {args.dataset_path} -> {args.cache_dir}")
    total = 0
    failures = []
    for dataset, split in entries:
        try:
            total += stage_split(
                args.dataset_path, dataset, split, args.cache_dir, args.max_samples, args.gt_field, args.refresh, token
            )
        except Exception as ex:  # noqa: BLE001 - keep staging the rest
            _log(f"  ERROR staging {dataset}/{split}: {type(ex).__name__}: {ex}")
            failures.append(f"{dataset}/{split}")
    _log(f"\nDone. {total} utts across {len(entries) - len(failures)}/{len(entries)} splits.")
    if failures:
        _log("FAILED: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
