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
"""Open-ASR-Leaderboard evaluator for a STANDARD NeMo ASR model (e.g. Parakeet
TDT v2), reading the SAME pre-staged wav/manifest cache as the SpeechLM evaluator.

This is a sanity/reference driver: it deliberately shares the pooled-shard logic,
the on-disk cache format, the shard-generation JSONL schema, and the
leaderboard-faithful scorer (vendored normalizer + kaldialign merge_compounds)
with scripts/speechlm_leaderboard_eval.py. The ONLY difference is the decode call:
a plain ``ASRModel.transcribe()`` instead of the SpeechLM ``model.generate()``.

Because the data + scoring are byte-for-byte the same code path, running a known
public model (Parakeet TDT 0.6B v2) through this driver should reproduce that
model's published Open-ASR-Leaderboard WER -- validating that our cache staging
and our scorer match the board. Any residual gap is the model, not the pipeline.

It loads ONE ``.nemo`` file, shards the pooled utterances across GPUs identically
to the SpeechLM driver (--num_shards N --shard_index <gpu>, same seed), and writes
a dataset-tagged generations JSONL; a final --aggregate reduces per-dataset WER.
"""
import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from typing import List

import torch
from tqdm import tqdm

# Reuse the SpeechLM driver's data plumbing + scoring so the pipeline is IDENTICAL.
# Its module-level imports are light (numpy/soundfile/torch/tqdm); the heavy NeMo
# imports there are all lazy (inside functions), so importing it costs nothing.
# When launched as `python /code/scripts/parakeet_leaderboard_eval.py`, Python puts
# /code/scripts on sys.path[0], so these sibling imports resolve.
from speechlm_leaderboard_eval import (  # noqa: E402
    DEFAULT_DATASETS,
    _log,
    _parse_entries,
    aggregate_results,
    build_global_items,
    select_shard,
)


def load_asr_model(nemo_path: str, device: torch.device):
    """Restore a standard NeMo ASR model from a local ``.nemo`` (offline, no Hub)."""
    from nemo.collections.asr.models import ASRModel

    _log(f"Loading NeMo ASR model: {nemo_path}")
    model = ASRModel.restore_from(restore_path=nemo_path, map_location=device)
    model = model.to(device).eval()
    _log(f"  Loaded {type(model).__name__} on {device}")
    return model


def _hyps_to_text(out) -> List[str]:
    """Normalize NeMo transcribe() output to a flat list of strings.

    transcribe() has returned different shapes across versions/decoders: a list of
    strings, a list of Hypothesis (``.text``), or a (best, all) tuple for RNNT/TDT.
    """
    if isinstance(out, tuple):
        out = out[0]
    texts: List[str] = []
    for h in out:
        if hasattr(h, "text"):
            texts.append(h.text or "")
        else:
            texts.append(str(h))
    return texts


def _transcribe_with_oom_backoff(model, paths: List[str], batch_size: int, min_bs: int = 1) -> List[str]:
    """model.transcribe with batch-halving on CUDA OOM. Retries the SAME block at a
    smaller batch (transcribe batches internally, so memory scales with batch_size).
    """
    bs = max(min_bs, int(batch_size))
    try:
        with torch.inference_mode():
            try:
                out = model.transcribe(paths, batch_size=bs, verbose=False)
            except TypeError as te:
                # ONLY the "old signature lacks verbose=" case; any other internal
                # TypeError (e.g. a NeMo version mismatch in the decoder) must
                # surface, not be disguised as a verbose retry.
                if "verbose" not in str(te):
                    raise
                out = model.transcribe(paths, batch_size=bs)
        return _hyps_to_text(out)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower() or bs <= min_bs:
            raise
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    new_bs = max(min_bs, bs // 2)
    _log(f"  CUDA OOM at batch_size={bs}; retrying block at {new_bs}.")
    return _transcribe_with_oom_backoff(model, paths, new_bs, min_bs)


def evaluate_shard(model, args, device: torch.device) -> int:
    """Decode this GPU's pooled slice (a mix of all datasets) and write a
    shard-unique, dataset-tagged generations JSONL for the reduce step. Identical
    output contract to speechlm_leaderboard_eval.evaluate_shard."""
    entries = _parse_entries(args.datasets)
    items = build_global_items(args.cache_dir, entries, args.max_eval_samples)
    if not items:
        _log("shard: no samples found across any dataset; nothing to do")
        return 0
    shard = select_shard(items, args.shard_index, args.num_shards, args.shuffle_seed)
    total = len(shard)
    _log(
        f"shard {args.shard_index}/{args.num_shards}: {total}/{len(items)} pooled utts "
        f"across {len(entries)} datasets (seed={args.shuffle_seed})"
    )
    if total == 0:
        return 0

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"shard{args.shard_index}_of{args.num_shards}.generations.jsonl")

    per_ds_total = Counter(it["key"] for it in shard)
    per_ds_done: Counter = Counter()

    def _postfix() -> str:
        return " ".join(f"{k.split('/')[0]}:{per_ds_done[k]}/{per_ds_total[k]}" for k in sorted(per_ds_total))

    # Decode in blocks so we can write incrementally (live tail -f) and bound an OOM
    # retry to one block. transcribe() batches internally at --batch_size.
    block = max(int(args.block_size), int(args.batch_size))
    start = time.time()
    pbar = tqdm(
        total=total, desc=f"shard{args.shard_index}", unit="utt", ncols=140,
        mininterval=float(args.progress_interval), file=sys.stdout,
    )
    pbar.set_postfix_str(_postfix())
    with open(out_path, "w") as fout:
        for i in range(0, total, block):
            batch = shard[i : i + block]
            paths = [it["path"] for it in batch]
            try:
                hyps = _transcribe_with_oom_backoff(model, paths, args.batch_size, args.min_batch_size)
            except Exception as ex:  # noqa: BLE001 - non-fatal: keep other datasets intact
                _log(f"  WARN: block [{i}:{i + len(batch)}] failed ({type(ex).__name__}: {ex}); emitting empty hyps")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                hyps = [""] * len(batch)
            if len(hyps) != len(batch):  # defensive: keep alignment with refs
                hyps = (hyps + [""] * len(batch))[: len(batch)]
            for it, h in zip(batch, hyps):
                fout.write(json.dumps({"key": it["key"], "reference": it["ref"], "hypothesis": h}) + "\n")
                per_ds_done[it["key"]] += 1
            fout.flush()
            pbar.update(len(batch))
            pbar.set_postfix_str(_postfix())
    pbar.close()
    _log(f"shard {args.shard_index}: decoded {total} utts in {time.time() - start:.1f}s -> {out_path}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nemo_model", help="Path to the .nemo ASR model (required unless --aggregate).")
    p.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated 'name:split' list (default = full leaderboard suite).",
    )
    p.add_argument("--cache_dir", help="Pre-staged cache root (<dataset>/<split>/_cache_manifest.jsonl).")
    p.add_argument("--device", type=int, default=0, help="GPU index (cuda:N).")
    # --- pooled-shard mode (balance work across GPUs regardless of dataset size) ---
    p.add_argument("--num_shards", type=int, default=1, help="If >1, evaluate only this GPU's 1/num_shards slice.")
    p.add_argument("--shard_index", type=int, default=0, help="This shard's index in [0, num_shards).")
    p.add_argument("--shuffle_seed", type=int, default=1234, help="Seed for the global shuffle (stable across shards).")
    p.add_argument(
        "--aggregate",
        action="store_true",
        help="Reduce mode: read all shard generation JSONLs under --output_dir and print per-dataset WER "
        "(delegates to the SpeechLM driver's aggregator so scoring is identical).",
    )
    p.add_argument("--batch_size", type=int, default=32, help="transcribe() batch size (OOM auto-halves).")
    p.add_argument("--min_batch_size", type=int, default=1, help="Lower bound for the OOM batch-halving.")
    p.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="Utterances per incremental write block (>= batch_size). Bounds OOM retries + gives live progress.",
    )
    p.add_argument("--max_eval_samples", type=int, default=0, help="Cap samples per dataset (0 = all).")
    p.add_argument("--output_dir", type=str, default=None, help="Where to write shard generations JSONL.")
    p.add_argument(
        "--progress_interval",
        type=float,
        default=5.0,
        help="Min seconds between progress-bar refreshes in the log (tail -f friendly).",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Reduce mode needs no GPU/model: reuse the SpeechLM aggregator verbatim so the
    # normalizer + WER are byte-for-byte identical to the SpeechLM eval.
    if args.aggregate:
        if not args.output_dir:
            _log("ERROR: --aggregate requires --output_dir (where shard JSONLs live).")
            return 1
        return aggregate_results(args)

    if not args.nemo_model or not args.cache_dir:
        _log("ERROR: --nemo_model and --cache_dir are required for evaluation.")
        return 1
    if not os.path.isfile(args.nemo_model):
        _log(f"ERROR: .nemo model not found: {args.nemo_model}")
        return 1

    if not torch.cuda.is_available():
        _log("WARNING: CUDA not available; running on CPU (very slow).")
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

    model = load_asr_model(args.nemo_model, device)

    if not (args.num_shards and args.num_shards > 1):
        # Single-process convenience: treat the whole suite as one shard.
        args.num_shards, args.shard_index = 1, 0
    if not args.output_dir:
        _log("ERROR: --output_dir is required for shard generations.")
        return 1
    return evaluate_shard(model, args, device)


if __name__ == "__main__":
    raise SystemExit(main())
