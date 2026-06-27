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

Two input modes
---------------
* **non-tarred** (default): ``--manifest`` lines carry ``audio_filepath`` to real
  audio files; sharding is round-robin over manifest entries.
* **tarred** (``--tarred-audio-filepaths``): reads audio directly from NeMo tar
  shards (use this for tarred / speed-perturbed training data so file_ids AND
  durations match exactly what training sees). ``--manifest`` is the tarred
  manifest (member name + text); sharding is round-robin over tar shards.

Examples
--------
    # tarred (matches tarred training data) -- one shard per GPU on a single node
    python scripts/asr_aligner/generate_qwen_word_alignments.py \
        --manifest /data/tarred_train/tarred_audio_manifest.json \
        --tarred-audio-filepaths "/data/tarred_train/audio__OP_0..511_CL_.tar" \
        --output   /results/.../qwen_word_aligns/train_960 \
        --aligner  /aligner_qwen --device cuda:0 \
        --num-shards 8 --shard-index 0
"""

import argparse
import io
import json
import os
import tarfile
from typing import Dict, Iterator, List, Tuple

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', required=True, help='NeMo manifest (.json/.jsonl): audio_filepath, text, duration.')
    p.add_argument(
        '--output',
        required=True,
        help='Output DIRECTORY. Each shard writes shard_XXXX.json into it; point the trainer '
        'external_aligner.alignments_path at this directory (shards are merged on load).',
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
    p.add_argument(
        '--tarred-audio-filepaths',
        default=None,
        help='If set, read audio from these NeMo tar shards (brace pattern or comma-separated). '
        'Use for tarred/speed-perturbed training data. --manifest is then the tarred manifest.',
    )
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


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return np.ascontiguousarray(audio, dtype=np.float32)
    import librosa

    return np.ascontiguousarray(librosa.resample(audio, orig_sr=sr, target_sr=target_sr), dtype=np.float32)


def load_audio(path: str, target_sr: int) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype='float32', always_2d=False)
    if audio.ndim > 1:  # downmix to mono
        audio = audio.mean(axis=1)
    return _resample(audio, sr, target_sr)


def _decode_audio_bytes(raw: bytes, target_sr: int) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return _resample(audio, sr, target_sr)


def _expand_tar_pattern(pattern: str) -> List[str]:
    """Expand a NeMo tar filepath spec into a sorted list of tar paths.

    Accepts a comma-separated list and/or NeMo's ``_OP_a..b_CL_`` brace syntax
    (equivalent to ``{a..b}``).
    """
    from braceexpand import braceexpand

    out: List[str] = []
    for part in str(pattern).split(','):
        part = part.strip()
        if not part:
            continue
        part = part.replace('_OP_', '{').replace('_CL_', '}')
        out.extend(braceexpand(part))
    return sorted(out)


def iter_nontarred(items: List[dict], args) -> Iterator[Tuple[str, str, np.ndarray]]:
    items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_index]
    if args.limit is not None:
        items = items[: args.limit]
    print(f"[gen-align] shard {args.shard_index}/{args.num_shards}: {len(items)} utterances (non-tarred).")
    for it in items:
        apath = it.get(args.audio_key)
        text = it.get(args.text_key, '')
        dur = it.get('duration', None)
        if apath is None or not text:
            continue
        if args.max_duration is not None and dur is not None and float(dur) > args.max_duration:
            continue
        try:
            audio = load_audio(apath, args.sample_rate)
        except Exception as e:  # noqa: BLE001
            print(f"[gen-align] WARNING: failed to read '{apath}' ({e!r}); skipping.")
            continue
        yield _file_id(apath), text, audio


def iter_tarred(items: List[dict], args) -> Iterator[Tuple[str, str, np.ndarray]]:
    # file_id -> (text, duration) from the tarred manifest.
    id2meta: Dict[str, Tuple[str, float]] = {}
    for it in items:
        apath = it.get(args.audio_key)
        text = it.get(args.text_key, '')
        if apath is None or not text:
            continue
        id2meta[_file_id(apath)] = (text, it.get('duration', None))

    all_tars = _expand_tar_pattern(args.tarred_audio_filepaths)
    my_tars = [t for i, t in enumerate(all_tars) if i % args.num_shards == args.shard_index]
    print(
        f"[gen-align] shard {args.shard_index}/{args.num_shards}: {len(my_tars)}/{len(all_tars)} tar shards "
        f"({len(id2meta)} manifest entries, tarred)."
    )
    n = 0
    n_bad_tars = 0
    for tpath in my_tars:
        try:
            tf = tarfile.open(tpath, 'r')
        except Exception as e:  # noqa: BLE001
            n_bad_tars += 1
            print(f"[gen-align] WARNING: cannot open tar '{tpath}' ({e!r}); skipping this shard file.")
            continue
        with tf:
            # Iterate defensively: a truncated/corrupt shard raises tarfile.ReadError
            # mid-stream. Skip the remainder of that shard and move on (same policy as
            # the training dataloader's tarred_audio_skip_handler), so a damaged tar
            # never stalls or kills the run.
            it = iter(tf)
            while True:
                try:
                    member = next(it)
                except StopIteration:
                    break
                except Exception as e:  # noqa: BLE001  (tarfile.ReadError etc.)
                    n_bad_tars += 1
                    print(
                        f"[gen-align] WARNING: damaged tar '{tpath}' ({e!r}); "
                        f"skipping the rest of this shard. These utterances get NO alignment "
                        f"and are dropped at train time too."
                    )
                    break
                if not member.isfile():
                    continue
                fid = _file_id(member.name)
                meta = id2meta.get(fid)
                if meta is None:
                    continue
                text, dur = meta
                if args.max_duration is not None and dur is not None and float(dur) > args.max_duration:
                    continue
                try:
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    audio = _decode_audio_bytes(f.read(), args.sample_rate)
                except Exception as e:  # noqa: BLE001
                    print(f"[gen-align] WARNING: failed to decode '{member.name}' in '{tpath}' ({e!r}); skipping.")
                    continue
                n += 1
                if args.limit is not None and n > args.limit:
                    return
                yield fid, text, audio
    if n_bad_tars:
        print(f"[gen-align] shard {args.shard_index}: {n_bad_tars} damaged/unreadable tar shard(s) skipped.")


def main():
    args = parse_args()

    import torch

    from nemo.collections.speechlm2.modules.qwen_forced_aligner import QwenForcedAligner

    torch_dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[args.dtype]

    items = read_manifest(args.manifest)
    sample_iter = iter_tarred(items, args) if args.tarred_audio_filepaths else iter_nontarred(items, args)

    aligner = QwenForcedAligner(
        pretrained_model=args.aligner, language=args.language, device=args.device, dtype=torch_dtype
    )

    results: Dict[str, List[float]] = {}
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

    for fid, text, audio in sample_iter:
        buf_audio.append(audio)
        buf_text.append(text)
        buf_key.append(fid)
        if len(buf_audio) >= args.batch_size:
            flush()
        if (n_ok + n_skip) and (n_ok + n_skip) % 1000 == 0:
            print(f"[gen-align] processed ~{n_ok + n_skip} (ok={n_ok}, skip={n_skip})")
    flush()

    # Always write a per-shard file INTO the output directory so the trainer's
    # directory-merge works uniformly for any shard count (incl. a single shard).
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"shard_{args.shard_index:04d}.json")

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f)
    print(f"[gen-align] DONE shard {args.shard_index}: wrote {len(results)} alignments to {out_path} "
          f"(ok={n_ok}, skip={n_skip}).")


if __name__ == '__main__':
    main()
