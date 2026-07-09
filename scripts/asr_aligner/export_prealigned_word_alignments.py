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

"""Export word-level alignments from offline/pre-aligned Lhotse manifests.

Reads the same ``input_cfg`` YAML used by streaming_stt / imend (cuts carry
``custom.alignments`` with per-word start/end times) and writes the
``{file_id: {"starts": [...], "ends": [...]}}`` JSON shards expected by
:class:`~nemo.collections.asr.parts.submodules.external_word_aligner.PrecomputedWordForcedAligner`.

This avoids re-running Qwen forced alignment when the training manifest is
already Qwen-pre-aligned.

Example (single process):
    python scripts/asr_aligner/export_prealigned_word_alignments.py \\
        --input-cfg /lustre/fsw/.../v1p1_recombined_original_layout_iad_s3_audio.yaml \\
        --output /results/chat_extaligner_ce_granary/qwen_word_aligns/v1p1_train

Sharded (8-way) export on a single node:
    python ... --num-shards 8 --shard-index \$SLURM_PROCID
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterator, List, Optional, Tuple

from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-cfg', required=True, help='Lhotse input_cfg YAML (same as imend data.train_ds.input_cfg).')
    p.add_argument(
        '--output',
        required=True,
        help='Output DIRECTORY. Writes shard_XXXX.json; point external_aligner.alignments_path here.',
    )
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--limit', type=int, default=None, help='Process at most N cuts (debug).')
    p.add_argument('--sample-rate', type=int, default=16000, help='Resample cuts to this rate before iteration.')
    return p.parse_args()


def _file_id(path: str) -> str:
    return os.path.splitext(os.path.basename(str(path)))[0]


def _cut_file_id(cut) -> str:
    if cut.recording is not None and cut.recording.sources:
        src = cut.recording.sources[0].source
        if src:
            return _file_id(src)
    return _file_id(cut.id)


def _iter_cut_alignments(cut) -> Optional[Tuple[List[float], List[float]]]:
    custom = cut.custom or {}
    raw = custom.get('alignments', [])
    if not raw:
        return None
    starts: List[float] = []
    ends: List[float] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if 'start_time' not in item:
            continue
        starts.append(float(item['start_time']))
        ends.append(float(item.get('end_time', item['start_time'])))
    if not starts:
        return None
    return starts, ends


def main():
    args = parse_args()

    from nemo.collections.common.data.lhotse import read_cutset_from_config

    config = OmegaConf.create({'input_cfg': args.input_cfg, 'sample_rate': args.sample_rate})
    cuts, _ = read_cutset_from_config(config)

    out: Dict[str, Dict[str, List[float]]] = {}
    n_seen = 0
    n_written = 0
    n_missing = 0

    for i, cut in enumerate(cuts):
        if args.num_shards > 1 and (i % args.num_shards) != args.shard_index:
            continue
        if args.limit is not None and n_seen >= args.limit:
            break
        n_seen += 1
        times = _iter_cut_alignments(cut)
        if times is None:
            n_missing += 1
            continue
        starts, ends = times
        out[_cut_file_id(cut)] = {'starts': starts, 'ends': ends}
        n_written += 1

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f'shard_{args.shard_index:04d}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f)

    print(
        f"[export-prealigned] shard {args.shard_index}/{args.num_shards}: "
        f"wrote {n_written} alignments ({n_missing} cuts missing alignments, {n_seen} scanned) -> {out_path}"
    )


if __name__ == '__main__':
    main()
