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
"""SINGLE-process, SINGLE-GPU Parakeet leaderboard eval for DESKTOP sanity checks.

This is the debugging counterpart of scripts/parakeet_leaderboard_eval.py. It
strips out EVERY harness variable except the scoring:

  * NO multi-GPU fan-out (one process, one device),
  * NO cross-dataset pooling, NO seeded global shuffle, NO duration sort,
  * each dataset is transcribed INDEPENDENTLY, in the manifest's original order,
    with a plain ``model.transcribe()`` loop -- i.e. the same shape as the public
    leaderboard's own per-dataset run_eval.py.

It still reads the SAME on-disk cache (``read_cache_manifest``) and scores with
the SAME leaderboard-faithful WER (``leaderboard_wer.WER``: vendored normalizer +
kaldialign merge_compounds) as the grid eval. So if the grid (sharded) numbers and
these (unsharded) numbers agree, the fan-out is exonerated; if THESE match the
public board but the grid ones don't, the bug is in the sharding/aggregation.

It is ALSO self-staging: for each dataset it downloads + materializes the cache
(reusing scripts/stage_leaderboard_cache.stage_split) the FIRST time, then reuses
it on later runs. Download and eval ALTERNATE per dataset -- dataset i is fetched,
evaluated, and only then is dataset i+1 fetched -- so you're never waiting for all
7 downloads before the first number, and a crash mid-suite still leaves earlier
datasets cached. First run needs internet + HF access to the gated dataset
``hf-audio/open-asr-leaderboard`` (``huggingface-cli login`` or export HF_TOKEN,
and accept the dataset terms once on the Hub). Use --offline to eval cache-only.

Usage (on the desktop, from the clean repo root, in the env that has nemo.asr):
    # first run downloads each dataset to --cache_dir, then evals it:
    python scripts/parakeet_eval_local.py \
        --nemo_model /path/to/parakeet-tdt-0.6b-v2.nemo \
        --cache_dir  ~/leaderboard_cache \
        --output_dir ./parakeet_local_out               # optional (dumps generations)
    # quick check first (downloads only 50 utts/dataset):
    python scripts/parakeet_eval_local.py --nemo_model ... --cache_dir ~/leaderboard_cache --max_eval_samples 50
    # later, reuse the cache with no network:
    python scripts/parakeet_eval_local.py --nemo_model ... --cache_dir ~/leaderboard_cache --offline
"""
import argparse
import gc
import json
import os
import time
from typing import List

import torch
from tqdm import tqdm

# Same-dir imports (run as `python scripts/parakeet_eval_local.py`, which puts
# scripts/ on sys.path[0]). Reuse the cache reader + scorer so the ONLY difference
# vs the grid eval is the removed sharding/pooling, and reuse the SAME staging
# routine so a self-staged local cache is byte-identical to the grid's.
from leaderboard_wer import WER
from quiet_logs import silence as _silence_logs
from speechlm_leaderboard_eval import DEFAULT_DATASETS, _log, _parse_entries, read_cache_manifest
from stage_leaderboard_cache import stage_split


def load_asr_model(nemo_path: str, device: torch.device):
    """Restore a standard NeMo ASR model from a local ``.nemo`` (offline)."""
    from nemo.collections.asr.models import ASRModel

    _log(f"Loading NeMo ASR model: {nemo_path}")
    model = ASRModel.restore_from(restore_path=nemo_path, map_location=device)
    model = model.to(device).eval()
    _log(f"  Loaded {type(model).__name__} on {device}")
    return model


def _hyps_to_text(out) -> List[str]:
    """Normalize transcribe() output (list[str] | list[Hypothesis] | (best, all))."""
    if isinstance(out, tuple):
        out = out[0]
    texts: List[str] = []
    for h in out:
        texts.append(h.text or "" if hasattr(h, "text") else str(h))
    return texts


def _transcribe_with_oom_backoff(model, paths: List[str], batch_size: int, min_bs: int = 1) -> List[str]:
    """model.transcribe with batch-halving on CUDA OOM (same policy as the grid driver)."""
    bs = max(min_bs, int(batch_size))
    try:
        with torch.inference_mode():
            try:
                out = model.transcribe(paths, batch_size=bs, verbose=False)
            except TypeError as te:
                # ONLY treat this as the "old signature lacks verbose=" case; any
                # other internal TypeError (e.g. a NeMo version mismatch in the
                # decoder) must surface, not be disguised as a verbose retry.
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
    _log(f"  CUDA OOM at batch_size={bs}; retrying at {new_bs}.")
    return _transcribe_with_oom_backoff(model, paths, new_bs, min_bs)


def evaluate_dataset(model, args, dataset: str, split: str) -> dict:
    key = f"{dataset}/{split}"
    paths, refs, _durs = read_cache_manifest(args.cache_dir, dataset, split, args.max_eval_samples)
    if not paths:
        _log(f"{key}: no samples found; skipping")
        return None

    hyps: List[str] = []
    bs = max(1, int(args.batch_size))
    total = len(paths)
    start = time.time()
    pbar = tqdm(total=total, desc=key, unit="utt", ncols=100)
    for i in range(0, total, bs):
        batch = paths[i : i + bs]
        try:
            hyps.extend(_transcribe_with_oom_backoff(model, batch, args.batch_size, args.min_batch_size))
        except Exception as ex:  # noqa: BLE001 - keep going; empty hyps for this batch
            _log(f"  WARN: batch [{i}:{i + len(batch)}] failed ({type(ex).__name__}: {ex}); empty hyps")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            hyps.extend([""] * len(batch))
        pbar.update(len(batch))
    pbar.close()
    if len(hyps) != total:  # defensive alignment
        hyps = (hyps + [""] * total)[:total]
    elapsed = time.time() - start

    wer = WER(normalize=True, verbose=args.verbose)
    wer.update(key, refs=refs, hyps=hyps)
    wer_val = float(wer.compute()["wer"]) * 100.0

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, f"{dataset}_{split}_generations.jsonl".replace("/", "_"))
        with open(out, "w") as f:
            for p, r, h in zip(paths, refs, hyps):
                f.write(json.dumps({"audio_filepath": p, "reference": r, "hypothesis": h}) + "\n")

    _log(f"RESULT\t{key}\t{wer_val:.2f}\t{elapsed:.1f}\t{len(paths)}")
    return {"key": key, "wer": wer_val, "time": elapsed, "n": len(paths)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nemo_model", required=True, help="Path to the .nemo ASR model.")
    p.add_argument(
        "--cache_dir",
        required=True,
        help="Cache root (<dataset>/<split>/_cache_manifest.jsonl). Auto-staged on first run; reused after.",
    )
    p.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated 'name:split' list (default = full public leaderboard suite).",
    )
    p.add_argument(
        "--gpu",
        "--device",
        dest="gpu",
        type=int,
        default=0,
        help="CUDA device index (cuda:N); ignored if no CUDA. Pin different N per run to share the box.",
    )
    p.add_argument("--batch_size", type=int, default=16, help="transcribe() batch size (OOM auto-halves).")
    p.add_argument("--min_batch_size", type=int, default=1, help="Lower bound for the OOM batch-halving.")
    p.add_argument("--max_eval_samples", type=int, default=0, help="Cap samples per dataset (0 = all; use e.g. 50 for a quick check).")
    p.add_argument("--output_dir", type=str, default=None, help="If set, dump per-dataset generations JSONL for inspection.")
    # --- self-staging (download) knobs; ignored under --offline ---
    p.add_argument("--dataset_path", default="hf-audio/open-asr-leaderboard", help="Hub dataset to download from.")
    p.add_argument("--gt_field", default=None, help="Force the reference column when staging (default: auto-detect).")
    p.add_argument("--refresh", action="store_true", help="Re-download/re-stage even if a split is already cached.")
    p.add_argument(
        "--offline",
        action="store_true",
        help="Do NOT download: eval only splits already present in --cache_dir (fails a split if it's missing).",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.isfile(args.nemo_model):
        _log(f"ERROR: .nemo model not found: {args.nemo_model}")
        return 1
    if args.offline and not os.path.isdir(args.cache_dir):
        _log(f"ERROR: --offline set but cache_dir not found: {args.cache_dir}")
        return 1
    os.makedirs(args.cache_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        _log("WARNING: CUDA not available; running on CPU (slow -- use --max_eval_samples for a quick check).")
        device = torch.device("cpu")

    # env token, else fall back to a stored `huggingface-cli login` (True).
    token = os.environ.get("HF_TOKEN") or True

    _silence_logs()
    model = load_asr_model(args.nemo_model, device)
    _silence_logs()  # restore_from re-inits NeMo logging; re-apply after load.

    entries = _parse_entries(args.datasets)
    results = []
    for dataset, split in entries:
        # Stage (download) THIS dataset first, then eval it -- alternating per
        # dataset so the first WER lands without waiting on all downloads, and a
        # mid-suite crash still leaves earlier datasets cached for the next run.
        if not args.offline:
            try:
                stage_split(
                    args.dataset_path, dataset, split, args.cache_dir,
                    args.max_eval_samples, args.gt_field, args.refresh, token,
                )
            except Exception as ex:  # noqa: BLE001 - skip this dataset, keep the rest
                _log(f"RESULT\t{dataset}/{split}\tERR\t0.0\t0  (staging failed: {type(ex).__name__}: {ex})")
                continue
        try:
            r = evaluate_dataset(model, args, dataset, split)
            if r is not None:
                results.append(r)
        except Exception as ex:  # noqa: BLE001
            _log(f"RESULT\t{dataset}/{split}\tERR\t0.0\t0  ({type(ex).__name__}: {ex})")

    if results:
        _log("\n  {:<28} {:>8} {:>10}".format("Dataset", "WER(%)", "N"))
        _log("  " + "-" * 48)
        tot = 0.0
        for r in results:
            _log(f"  {r['key']:<28} {r['wer']:>8.2f} {r['n']:>10d}")
            tot += r["wer"]
        _log("  " + "-" * 48)
        _log(f"  {'Average (macro)':<28} {tot / len(results):>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
