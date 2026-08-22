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
"""Model-agnostic pieces of the Open-ASR-Leaderboard harness.

Shared by ``script_leaderboard_eval.py`` (the SCRIPT SpeechLM) and
``nemotron_leaderboard_eval.py`` (a plain NeMo ASR model) so that the dataset
list, the shard partition and the scoring are *identical* by construction. Two
systems evaluated through this module differ only in how they turn audio into
text -- which is the entire point of comparing them.
"""

import glob
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import soundfile
import torch

# leaderboard_wer / leaderboard_normalizer sit next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leaderboard_wer import WER  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


_LEADERBOARD_SR = 16000

# The 2026-08 public suite (TED-LIUM dropped; "cleaned" variants where published).
DEFAULT_DATASETS = [
    "librispeech:test.clean",
    "librispeech:test.other",
    "ami_cleaned:test",
    "earnings22:test",
    "gigaspeech_cleaned:test",
    "spgispeech:test",
    "voxpopuli_cleaned_aa:test",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def parse_entries(spec: str) -> List[Tuple[str, str]]:
    """``"librispeech:test.clean,ami_cleaned:test"`` -> ``[(name, split), ...]``."""
    out = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, _, split = entry.partition(":")
        out.append((name, split or "test"))
    return out


def read_cache_manifest(cache_dir: str, dataset: str, split: str, max_samples: int = 0):
    """Read one staged split: ``<cache_dir>/<dataset>/<split>/_cache_manifest.jsonl``."""
    ds_dir = os.path.join(cache_dir, dataset, split)
    manifest = os.path.join(ds_dir, "_cache_manifest.jsonl")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(
            f"No staged cache for {dataset}:{split} at {manifest}.\n"
            f"Stage it first:  sbatch launch/stage_leaderboard_cache.sh"
        )
    paths, refs, durs = [], [], []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fp = rec["audio_filepath"]
            if not os.path.exists(fp):
                # Cache re-staged under a different root: retry by basename.
                fp = os.path.join(ds_dir, os.path.basename(fp))
                if not os.path.exists(fp):
                    continue
            paths.append(fp)
            refs.append(rec.get("reference", rec.get("text", "")))
            durs.append(float(rec.get("duration", 0.0)))
            if max_samples and len(paths) >= max_samples:
                break
    return paths, refs, durs


def load_audio_batch(paths: List[str], pad_extra_seconds: float = 0.0):
    """Load a batch of 16 kHz mono wavs into ``(audios, audio_lens)``.

    ``pad_extra_seconds`` of REAL trailing silence is appended and counted in
    ``audio_lens``, mirroring the training dataloader's ``pad_extra_duration``.
    The encoder therefore sees the same end-of-utterance context it was trained
    on; dropping it changes exactly the final chunk, where delayed words land.
    """
    pad = int(round(pad_extra_seconds * _LEADERBOARD_SR))
    waves = []
    for p in paths:
        w, sr = soundfile.read(p, dtype="float32")
        if w.ndim > 1:
            w = w.mean(axis=1)
        if sr != _LEADERBOARD_SR:  # staging writes 16 kHz; guard against a stale cache
            raise ValueError(f"{p} has sample rate {sr}, expected {_LEADERBOARD_SR}")
        waves.append(w)
    lens = [len(w) + pad for w in waves]
    max_len = max(lens) if lens else 0
    out = np.zeros((len(waves), max_len), dtype=np.float32)
    for i, w in enumerate(waves):
        out[i, : len(w)] = w
    return torch.from_numpy(out), torch.tensor(lens, dtype=torch.long)


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------


def build_global_items(args) -> List[dict]:
    """Pool every utterance of every dataset into one flat list."""
    items: List[dict] = []
    for name, split in parse_entries(args.datasets):
        paths, refs, durs = read_cache_manifest(args.cache_dir, name, split, args.max_eval_samples)
        key = f"{name}/{split}"
        items.extend({"key": key, "path": p, "ref": r, "dur": d} for p, r, d in zip(paths, refs, durs))
        _log(f"    {key}: {len(paths)} utts")
    return items


def select_shard(items: List[dict], num_shards: int, shard_index: int, seed: int) -> List[dict]:
    """Deterministic, length-balanced shard.

    The shuffle uses a seed shared by every process so all shards agree on the
    same permutation and their union is exactly the pooled list. Sorting by
    duration afterwards keeps each batch homogeneous, which matters because a
    batch is padded to its longest clip.

    Both evaluators call this with the same seed and dataset list, so a given
    utterance lands in the same shard for every system under comparison.
    """
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    shard = [items[j] for pos, j in enumerate(order) if pos % num_shards == shard_index]
    shard.sort(key=lambda it: it["dur"])
    return shard


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_results(args) -> int:
    files = sorted(glob.glob(os.path.join(args.output_dir, "shard*_of*.generations.jsonl")))
    if not files:
        _log(f"ERROR: no shard files under {args.output_dir}")
        return 1
    _log(f"==> aggregating {len(files)} shard file(s)")

    groups: Dict[str, dict] = defaultdict(lambda: {"refs": [], "hyps": []})
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                groups[rec["key"]]["refs"].append(rec.get("reference", ""))
                groups[rec["key"]]["hyps"].append(rec.get("hypothesis", ""))

    results = []
    for key in sorted(groups):
        g = groups[key]
        wer = WER(normalize=True, verbose=args.verbose)
        wer.update(key, refs=g["refs"], hyps=g["hyps"])
        val = float(wer.compute()["wer"]) * 100.0
        results.append((key, val, len(g["refs"])))
        # Machine-readable row; the launcher greps these for the wandb report.
        _log(f"RESULT\t{key}\t{val:.2f}\t0.0\t{len(g['refs'])}")

    _log("")
    _log("  {:<30} {:>8} {:>10}".format("Dataset", "WER(%)", "N"))
    _log("  " + "-" * 50)
    for key, val, n in results:
        _log("  {:<30} {:>8.2f} {:>10d}".format(key, val, n))
    _log("  " + "-" * 50)
    macro = sum(v for _, v, _ in results) / len(results) if results else 0.0
    _log("  {:<30} {:>8.2f}".format("Average (macro)", macro))
    _log(f"RESULT\tAverage\t{macro:.2f}\t0.0\t{sum(n for _, _, n in results)}")
    return 0
