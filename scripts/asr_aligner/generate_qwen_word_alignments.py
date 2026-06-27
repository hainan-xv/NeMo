# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Offline word-level forced alignment with the Qwen3 ForcedAligner (Option B).

Runs the heavy Qwen aligner ONCE over a NeMo ASR manifest and writes per-word
**start times** (seconds) keyed by ``file_id`` (audio basename without extension).
Training then reads this file via
:class:`~nemo.collections.asr.parts.submodules.external_word_aligner.PrecomputedWordForcedAligner`
(``loss_type=chunkwise_aligner`` with ``external_aligner.backend=precomputed``),
so no ``qwen_asr`` / pinned ``transformers`` runs inside the training process.

This avoids the per-step Qwen forward that makes the live (Option A) backend slow.

Run it in an environment that has ``qwen_asr`` (e.g. the same image used for
Option A). Shard across GPUs with ``--num-shards`` / ``--shard-index`` and point
the trainer's ``alignments_path`` at the output *directory* to merge shards.

Example
-------
    python scripts/asr_aligner/generate_qwen_word_alignments.py \
        --manifest /data/librispeech/train_960.json \
        --output   /results/qwen_word_aligns/train_960 \
        --aligner  /aligner_qwen \
        --language English --dtype bfloat16 --device cuda:0 \
        --batch-size 16 --num-shards 8 --shard-index 0
"""

import argparse
import json
import os
from typing import List

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', required=True, help='NeMo manifest (.json/.jsonl): audio_filepath, text, duration.')
    p.add_argument(
        '--output',
        required=True,
        help='Output path. With --num-shards>1, a directory is created and each shard writes shard_XXXX.json into it; '
        'point the trainer alignments_path at that directory. With a single shard, writes <output>.json.',
    )
    p.add_argument('--aligner', default='Qwen/Qwen3-ForcedAligner-0.6B', help='Qwen aligner repo id or local dir.')
    p.add_argument('--language', default='English')
    p.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--sample-rate', type=int, default=16000)
    p.add_argument('--max-duration', type=float, default=None, help='Skip utterances longer than this (seconds).')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--text-key', default='text')
    p.add_argument('--audio-key', default='audio_filepath')
    p.add_argument('--limit', type=int, default=None, help='Process at most N (post-shard) utterances (debug).')
    return p.parse_args()


def _file_id(path: str) -> str:
    return os.path.splitext(os.path.basename(str(path)))[0]


def read_manifest(path: str) -> List[dict]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_audio(path: str, target_sr: int) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype='float32', always_2d=False)
    if audio.ndim > 1:  # downmix to mono
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(audio, dtype=np.float32)


def main():
    args = parse_args()

    from nemo.collections.speechlm2.modules.qwen_forced_aligner import QwenForcedAligner

    dtype_map = {'bfloat16': 'bfloat16', 'float16': 'float16', 'float32': 'float32'}
    import torch

    torch_dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[dtype_map[args.dtype]]

    items = read_manifest(args.manifest)
    # Deterministic round-robin sharding across parallel jobs.
    items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_index]
    if args.limit is not None:
        items = items[: args.limit]
    print(f"[gen-align] shard {args.shard_index}/{args.num_shards}: {len(items)} utterances to align.")

    aligner = QwenForcedAligner(
        pretrained_model=args.aligner, language=args.language, device=args.device, dtype=torch_dtype
    )

    results = {}
    n_ok = n_skip = 0
    buf_audio: List[np.ndarray] = []
    buf_text: List[str] = []
    buf_key: List[str] = []

    def flush():
        nonlocal n_ok, n_skip
        if not buf_audio:
            return
        try:
            batch_aligns = aligner.align_numpy(buf_audio, buf_text)
        except Exception as e:  # noqa: BLE001
            print(f"[gen-align] WARNING: batch of {len(buf_audio)} failed ({e!r}); skipping.")
            n_skip += len(buf_audio)
            buf_audio.clear(); buf_text.clear(); buf_key.clear()
            return
        for key, words in zip(buf_key, batch_aligns):
            results[key] = [round(float(w.start_time), 3) for w in words]
            n_ok += 1
        buf_audio.clear(); buf_text.clear(); buf_key.clear()

    for it in items:
        apath = it.get(args.audio_key)
        text = it.get(args.text_key, '')
        dur = it.get('duration', None)
        if apath is None or not text:
            n_skip += 1
            continue
        if args.max_duration is not None and dur is not None and float(dur) > args.max_duration:
            n_skip += 1
            continue
        try:
            audio = load_audio(apath, args.sample_rate)
        except Exception as e:  # noqa: BLE001
            print(f"[gen-align] WARNING: failed to read '{apath}' ({e!r}); skipping.")
            n_skip += 1
            continue
        buf_audio.append(audio)
        buf_text.append(text)
        buf_key.append(_file_id(apath))
        if len(buf_audio) >= args.batch_size:
            flush()
        if (n_ok + n_skip) and (n_ok + n_skip) % 1000 == 0:
            print(f"[gen-align] processed ~{n_ok + n_skip} (ok={n_ok}, skip={n_skip})")
    flush()

    if args.num_shards > 1:
        os.makedirs(args.output, exist_ok=True)
        out_path = os.path.join(args.output, f"shard_{args.shard_index:04d}.json")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
        out_path = args.output if args.output.endswith('.json') else args.output + '.json'

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f)
    print(f"[gen-align] DONE shard {args.shard_index}: wrote {len(results)} alignments to {out_path} "
          f"(ok={n_ok}, skip={n_skip}).")


if __name__ == '__main__':
    main()
