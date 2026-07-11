#!/usr/bin/env python3
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
Add word-level alignments to NeMo manifests using QwenForcedAligner.

Usage:
    # Single manifest
    python scripts/speechlm2/align_manifest.py \
        --input /path/to/manifest.json \
        --batch-size 8

    # Multiple manifests (comma-separated)
    python scripts/speechlm2/align_manifest.py \
        --input /path/to/train.json,/path/to/dev.json,/path/to/test.json \
        --batch-size 8

    # Multi-GPU: shard each manifest across N GPUs (one worker process per GPU)
    python scripts/speechlm2/align_manifest.py \
        --input /path/to/train_960.json \
        --batch-size 8 \
        --num-gpus 2

Reads each line of the input manifest (JSON-lines with ``audio_filepath``,
``text``, ``duration``), runs forced alignment in batches, and writes a new
manifest with an ``-aligned`` suffix containing an additional ``alignments``
field per utterance:

    {"audio_filepath": "...", "text": "...", "duration": ...,
     "alignments": [{"text": "hello", "start_time": 0.12, "end_time": 0.36}, ...]}
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Prefer this checkout over any separately installed NeMo package.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nemo.collections.common.parts.preprocessing.manifest import get_full_path  # noqa: E402
from nemo.collections.speechlm2.modules.qwen_forced_aligner import QwenForcedAligner  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def read_manifest(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_audio(audio_path: str) -> np.ndarray:
    """Load audio and resample to 16 kHz mono float32."""
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def get_output_path(input_path: str) -> str:
    """Derive output path by adding '-aligned' suffix before the extension."""
    p = Path(input_path).absolute()
    return str(p.with_name(f"{p.stem}-aligned{p.suffix}"))


def align_manifest(
    input_path: str,
    output_path: str,
    aligner: QwenForcedAligner,
    batch_size: int,
    manifest_label: str = "",
):
    """Align a single manifest file and write the output."""
    log.info("%sProcessing: %s -> %s", manifest_label, input_path, output_path)
    entries = read_manifest(input_path)

    n_aligned = 0
    n_failed = 0

    pbar = tqdm(
        range(0, len(entries), batch_size),
        desc=f"{manifest_label}{Path(input_path).name}",
        unit="batch",
        total=(len(entries) + batch_size - 1) // batch_size,
    )

    with open(output_path, "w") as out_f:
        for batch_start in pbar:
            batch = entries[batch_start : batch_start + batch_size]

            audio_arrays = []
            texts = []
            valid_indices = []

            for i, entry in enumerate(batch):
                audio_path = get_full_path(entry["audio_filepath"], manifest_file=input_path)
                text = entry.get("text", "")
                if not text:
                    log.warning("Skipping entry %d: empty text", batch_start + i)
                    continue
                try:
                    audio = load_audio(audio_path)
                except Exception as e:
                    log.warning("Skipping entry %d: failed to load audio %s: %s", batch_start + i, audio_path, e)
                    continue

                audio_arrays.append(audio)
                texts.append(text)
                valid_indices.append(i)

            alignment_map = {}
            if audio_arrays:
                try:
                    batch_alignments = aligner.align_numpy(audio_arrays, texts)
                except Exception as e:
                    log.warning(
                        "Alignment failed for batch starting at %d: %s. Writing entries without alignments.",
                        batch_start,
                        e,
                    )
                    batch_alignments = [[] for _ in valid_indices]
                    n_failed += len(valid_indices)

                for idx, aligns in zip(valid_indices, batch_alignments):
                    alignment_map[idx] = [asdict(a) for a in aligns]
                    n_aligned += 1

            for i, entry in enumerate(batch):
                out_entry = dict(entry)
                out_entry["alignments"] = alignment_map.get(i, [])
                out_f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")

            pbar.set_postfix(aligned=n_aligned, failed=n_failed)

    log.info("%sDone. Aligned: %d, Failed: %d, Total: %d", manifest_label, n_aligned, n_failed, len(entries))


def resolve_device_ids(num_gpus: int) -> list[str]:
    """Pick the physical GPU ids each worker should see.

    Honors an externally set ``CUDA_VISIBLE_DEVICES`` (splitting those ids
    among workers); otherwise falls back to ``0..num_gpus-1``.
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        ids = [x.strip() for x in env.split(",") if x.strip() != ""]
        if len(ids) < num_gpus:
            log.warning(
                "CUDA_VISIBLE_DEVICES lists %d device(s) but --num-gpus=%d; using %d worker(s).",
                len(ids),
                num_gpus,
                len(ids),
            )
        return ids[:num_gpus] if len(ids) >= num_gpus else ids
    return [str(i) for i in range(num_gpus)]


def split_contiguous(entries: list[dict], n: int) -> list[list[dict]]:
    """Split *entries* into at most *n* contiguous shards (order preserved)."""
    n = max(1, min(n, len(entries)))
    base, extra = divmod(len(entries), n)
    shards = []
    start = 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        shards.append(entries[start : start + size])
        start += size
    return shards


def align_manifest_multi_gpu(
    input_path: str,
    output_path: str,
    args,
    device_ids: list[str],
    manifest_label: str = "",
):
    """Shard a manifest across GPUs, run one worker per shard, then merge in order."""
    entries = read_manifest(input_path)
    if not entries:
        log.warning("%s%s is empty; writing empty output.", manifest_label, input_path)
        open(output_path, "w").close()
        return

    num_workers = max(1, min(len(device_ids), len(entries)))
    shards = split_contiguous(entries, num_workers)

    # Resolve audio paths to absolute so shard files can live anywhere without
    # breaking relative ``audio_filepath`` resolution in the worker.
    for shard in shards:
        for entry in shard:
            entry["audio_filepath"] = get_full_path(entry["audio_filepath"], manifest_file=input_path)

    out_p = Path(output_path).absolute()
    shard_dir = out_p.with_name(f".align_shards_{out_p.stem}")
    shard_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "%sSharding %d entries across %d GPU worker(s) [devices: %s]",
        manifest_label,
        len(entries),
        len(shards),
        ", ".join(device_ids[: len(shards)]),
    )

    procs = []
    shard_out_paths = []
    script_path = str(Path(__file__).resolve())
    for i, shard in enumerate(shards):
        shard_in = shard_dir / f"shard_{i}.in.json"
        shard_out = shard_dir / f"shard_{i}.out.json"
        with open(shard_in, "w") as f:
            for entry in shard:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device_ids[i]
        cmd = [
            sys.executable,
            script_path,
            "--input",
            str(shard_in),
            "--output",
            str(shard_out),
            "--model",
            args.model,
            "--language",
            args.language,
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--num-gpus",
            "1",
        ]
        log.info("%s  worker %d -> GPU %s (%d entries)", manifest_label, i, device_ids[i], len(shard))
        procs.append((subprocess.Popen(cmd, env=env), shard_out, i, device_ids[i]))
        shard_out_paths.append(shard_out)

    failed = []
    for proc, _shard_out, i, dev in procs:
        returncode = proc.wait()
        if returncode != 0:
            failed.append((i, dev, returncode))
            log.error("%s  worker %d (GPU %s) failed with exit code %d", manifest_label, i, dev, returncode)

    if failed:
        raise RuntimeError(
            f"{manifest_label}{len(failed)} shard worker(s) failed: "
            + ", ".join(f"shard {i} (GPU {dev}, rc={rc})" for i, dev, rc in failed)
            + f". Shard files kept for inspection at {shard_dir}"
        )

    with open(output_path, "w") as out_f:
        for shard_out in shard_out_paths:
            with open(shard_out) as in_f:
                shutil.copyfileobj(in_f, out_f)

    shutil.rmtree(shard_dir, ignore_errors=True)
    log.info("%sDone (multi-GPU). Merged %d entries -> %s", manifest_label, len(entries), output_path)


def main():
    parser = argparse.ArgumentParser(description="Add word-level alignments to NeMo manifests.")
    parser.add_argument(
        "--input",
        required=True,
        help="Comma-separated paths to input NeMo manifests (JSON-lines).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Comma-separated output paths (one per input). Defaults to <input-stem>-aligned.json.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-ForcedAligner-0.6B", help="Pretrained aligner model.")
    parser.add_argument("--language", default="English", help="Language for alignment.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for alignment.")
    parser.add_argument("--device", default="cuda", help="Device for the aligner model.")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to shard each manifest across (one worker process per GPU).",
    )
    args = parser.parse_args()

    if args.num_gpus < 1:
        parser.error(f"--num-gpus must be >= 1, got {args.num_gpus}")

    input_paths = [p.strip() for p in args.input.split(",")]
    if args.output is not None:
        output_paths = [p.strip() for p in args.output.split(",")]
        if len(output_paths) != len(input_paths):
            parser.error(
                f"Number of --output paths ({len(output_paths)}) must match "
                f"number of --input paths ({len(input_paths)})."
            )
    else:
        output_paths = [get_output_path(p) for p in input_paths]

    n_manifests = len(input_paths)

    # Multi-GPU: the parent only shards work and spawns single-GPU workers; it
    # never loads the aligner itself (avoids creating a CUDA context here).
    if args.num_gpus > 1:
        device_ids = resolve_device_ids(args.num_gpus)
        for mi, (input_path, output_path) in enumerate(zip(input_paths, output_paths), 1):
            label = f"[{mi}/{n_manifests}] "
            align_manifest_multi_gpu(input_path, output_path, args, device_ids, manifest_label=label)
        log.info("All %d manifest(s) processed.", n_manifests)
        return

    log.info("Loading aligner: %s", args.model)
    aligner = QwenForcedAligner(
        pretrained_model=args.model,
        language=args.language,
        device=args.device,
    )

    for mi, (input_path, output_path) in enumerate(zip(input_paths, output_paths), 1):
        label = f"[{mi}/{n_manifests}] "
        align_manifest(input_path, output_path, aligner, args.batch_size, manifest_label=label)

    log.info("All %d manifest(s) processed.", n_manifests)


if __name__ == "__main__":
    main()
