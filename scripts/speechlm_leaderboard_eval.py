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
"""Self-contained Open-ASR-Leaderboard evaluator for a SpeechLM (StreamingSTT /
Script) model, reading a PRE-STAGED wav/manifest cache on disk.

This is the SpeechLM counterpart of scripts/asr_leaderboard_eval.py, and unlike
the local ``run_eval_sslm.py`` driver it does NOT download from HuggingFace: it
reads each dataset from ``<cache_dir>/<dataset>/<split>/_cache_manifest.jsonl``
(the format written by run_eval_sslm's local cache — lines of
``{"audio_filepath","duration","reference"}`` pointing at 16 kHz mono wavs). This
makes it usable on OCI compute nodes that have no internet: pre-stage the cache
on lustre once, then evaluate there.

It loads ONE Lightning ``.ckpt`` into a configurable model class (default the
SCRIPT model), transcribes each dataset via ``model.generate`` (batched,
with CUDA-OOM batch-halving), and reports WER per dataset using the Open ASR
Leaderboard's OWN scoring (vendored normalizer + kaldialign merge_compounds; see
scripts/leaderboard_wer.py) so the numbers line up with the public board -- this
differs from training's whisper-normalized ``val_wer``.

Designed to eval ONE dataset per process/GPU (``--datasets ami:test --device 3``)
so a SLURM job can fan the leaderboard across a node's 8 GPUs
(see launch/eval_leaderboard.sh). It also accepts a comma-separated list to run
several datasets in one process (model loaded once).
"""
import argparse
import gc
import glob
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import soundfile
import torch
from tqdm import tqdm

# Default Open-ASR-Leaderboard entries (name:split), matching the CURRENT public
# suite scored by the board (github.com/huggingface/open_asr_leaderboard
# normalizer/eval_utils.py, 2026-08): the CLEANED AMI/GigaSpeech/VoxPopuli variants,
# and TED-LIUM dropped from the default macro-average. Cache the 7 configs from the
# consolidated hub dataset `hf-audio/open-asr-leaderboard` under these <name>/<split>
# dirs (that is the source the leaderboard run scripts now load).
DEFAULT_DATASETS = [
    "librispeech:test.clean",
    "librispeech:test.other",
    "ami_cleaned:test",
    "earnings22:test",
    "gigaspeech_cleaned:test",
    "spgispeech:test",
    "voxpopuli_cleaned_aa:test",
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _hubify(path: str) -> str:
    """Map an absolute ``.../huggingface/<org>/<name>[/file]`` path to ``<org>/<name>``.

    Only used as a fallback when the checkpoint's pretrained path does NOT exist
    locally (e.g. running off-cluster). On OCI the absolute lustre paths exist and
    are kept as-is (no network needed).
    """
    if not isinstance(path, str) or not path.startswith("/"):
        return path
    parts = [p for p in path.rstrip("/").split("/") if p]
    anchor = parts.index("huggingface") + 1 if "huggingface" in parts else 0
    rem = [p for p in parts[anchor:] if not p.endswith((".nemo", ".ckpt", ".bin"))]
    if len(rem) >= 2:
        return f"{rem[-2]}/{rem[-1]}"
    return path


def load_model(ckpt_path: str, model_class_path: str, device: torch.device, dtype: torch.dtype):
    """Instantiate ``model_class_path`` from a Lightning checkpoint's hyper-params
    and load its weights (strict=False). Pretrained LLM/ASR paths that exist on
    disk are kept (offline); otherwise mapped to Hub ids."""
    from nemo.utils.model_utils import import_class_by_path

    _log(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["hyper_parameters"]["cfg"]
    state_dict = ckpt["state_dict"]

    cfg["load_llm_weights"] = False  # LLM weights come from the checkpoint
    for key in ("pretrained_llm", "pretrained_asr"):
        p = cfg.get(key, "")
        if p and not os.path.exists(p):
            new = _hubify(p)
            if new != p:
                _log(f"  {key}: {p} -> {new} (not found locally; using Hub id)")
                cfg[key] = new

    # Newer PEFT needs explicit LoRA target_modules; detect from the ckpt keys.
    if "lora" in cfg and "target_modules" not in cfg["lora"]:
        mods = set()
        for k in state_dict:
            if ".lora_A." in k:
                parts = k.split(".")
                i = parts.index("lora_A")
                if i > 0:
                    mods.add(parts[i - 1])
        cfg["lora"]["target_modules"] = sorted(mods) if mods else "all-linear"
        _log(f"  LoRA target_modules: {cfg['lora']['target_modules']}")

    cls = import_class_by_path(model_class_path)
    _log(f"  Constructing {cls.__name__} ...")
    model = cls(cfg=cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        _log(f"  Missing keys ({len(missing)}): {missing[:4]}{'...' if len(missing) > 4 else ''}")
    if unexpected:
        _log(f"  Unexpected keys ({len(unexpected)}): {unexpected[:4]}{'...' if len(unexpected) > 4 else ''}")
    del ckpt, state_dict
    model = model.eval().to(dtype).to(device)
    return model


def read_cache_manifest(
    cache_dir: str, dataset: str, split: str, max_samples: int = 0
) -> Tuple[List[str], List[str], List[float]]:
    """Read a pre-staged split: returns (audio_filepaths, references, durations)."""
    path = os.path.join(cache_dir, dataset, split, "_cache_manifest.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No pre-staged manifest for {dataset}/{split} at {path}. Stage the leaderboard "
            f"cache on this filesystem (the run_eval_sslm cache layout: "
            f"<cache_dir>/<dataset>/<split>/_cache_manifest.jsonl + 16kHz wavs)."
        )
    ds_dir = os.path.join(cache_dir, dataset, split)
    paths, refs, durs = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fp = rec["audio_filepath"]
            # The manifest stores ABSOLUTE paths from where the cache was built.
            # After staging the cache to a different root (e.g. rsync to lustre),
            # those paths no longer exist -> reconstruct under cache_dir by basename.
            if not os.path.exists(fp):
                alt = os.path.join(ds_dir, os.path.basename(fp))
                if os.path.exists(alt):
                    fp = alt
            if not os.path.exists(fp):
                continue
            paths.append(fp)
            refs.append(rec.get("reference", rec.get("text", "")))
            durs.append(float(rec.get("duration", 0.0) or 0.0))
            if max_samples and len(paths) >= max_samples:
                break
    return paths, refs, durs


def load_audio_batch(paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a batch of (mono, 16 kHz) wavs into a right-padded (B, T) tensor + lengths."""
    arrs = []
    for p in paths:
        a, _sr = soundfile.read(p, dtype="float32")
        if a.ndim == 2:
            a = a.mean(axis=1)
        arrs.append(a)
    lens = [len(a) for a in arrs]
    T = max(lens) if lens else 1
    x = torch.zeros(len(arrs), T, dtype=torch.float32)
    for i, a in enumerate(arrs):
        x[i, : len(a)] = torch.from_numpy(np.ascontiguousarray(a))
    return x, torch.tensor(lens, dtype=torch.long)


def _generate_with_oom_backoff(model, audios, audio_lens, gen_kwargs, min_bs=1):
    """model.generate with recursive batch-halving on CUDA OOM. The retry happens
    OUTSIDE the except block so the failed forward's memory (pinned by the
    exception traceback) is released before the smaller retry."""
    B = int(audios.shape[0])
    try:
        return model.generate(audios=audios, audio_lens=audio_lens, **gen_kwargs)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower() or B <= min_bs:
            raise
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mid = B // 2
    _log(f"  CUDA OOM on batch of {B}; retrying as {mid}+{B - mid}.")
    left = _generate_with_oom_backoff(model, audios[:mid], audio_lens[:mid], gen_kwargs, min_bs)
    right = _generate_with_oom_backoff(model, audios[mid:], audio_lens[mid:], gen_kwargs, min_bs)
    return left + right


def evaluate_dataset(model, args, dataset: str, split: str, device: torch.device):
    from transformers import GenerationConfig

    # Leaderboard-faithful WER (vendored normalizer + kaldialign merge_compounds),
    # NOT the training-time whisper-normalizer val_wer. See scripts/leaderboard_wer.py.
    from leaderboard_wer import WER

    key = f"{dataset}/{split}"
    paths, refs, _durs = read_cache_manifest(args.cache_dir, dataset, split, args.max_eval_samples)
    if not paths:
        _log(f"{key}: no samples found; skipping")
        return None

    gen_kwargs = dict(
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        generation_config=GenerationConfig(do_sample=False),
        chunk_size_override=(args.chunk_size if args.chunk_size and args.chunk_size > 0 else None),
    )
    # Only forward when set, so non-redecode models never see an unexpected kwarg.
    if args.self_correct:
        gen_kwargs["self_correct"] = True
    if args.use_state_machine:
        gen_kwargs["use_state_machine_inference"] = True
    # SCRIPT-only kwarg; guard so non-SCRIPT model classes never see it.
    if args.force_word_start and "script" in args.model_class.lower():
        gen_kwargs["force_chunk_word_start"] = True

    hyps: List[str] = []
    start = time.time()
    bs = max(1, int(args.batch_size))
    total = len(paths)
    # Progress bar goes to stdout, which the SLURM launcher redirects into this
    # dataset's .log -> `tail -f <log>` shows a live bar. mininterval throttles
    # how often it rewrites so the log stays small over a long decode.
    pbar = tqdm(
        total=total, desc=key, unit="utt", ncols=100,
        mininterval=float(args.progress_interval), file=sys.stdout,
    )
    for i in range(0, total, bs):
        audios, audio_lens = load_audio_batch(paths[i : i + bs])
        audios = audios.to(device, non_blocking=True)
        audio_lens = audio_lens.to(device, non_blocking=True)
        with torch.inference_mode():
            hyps.extend(_generate_with_oom_backoff(model, audios, audio_lens, gen_kwargs, args.min_batch_size))
        pbar.update(min(bs, total - i))
    pbar.close()
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

    # Machine-parseable row (the SLURM launcher greps this).
    _log(f"RESULT\t{key}\t{wer_val:.2f}\t{elapsed:.1f}\t{len(paths)}")
    return {"key": key, "wer": wer_val, "time": elapsed, "n": len(paths)}


def build_global_items(cache_dir: str, entries: List[Tuple[str, str]], max_samples: int) -> List[dict]:
    """Pool every utterance across all datasets into one flat list.

    Each item carries its dataset/split so per-dataset WER can be reduced later.
    ``max_samples`` still caps PER dataset (so smoke tests stay small)."""
    items: List[dict] = []
    for dataset, split in entries:
        paths, refs, durs = read_cache_manifest(cache_dir, dataset, split, max_samples)
        key = f"{dataset}/{split}"
        for p, r, d in zip(paths, refs, durs):
            items.append({"key": key, "path": p, "ref": r, "dur": d})
    return items


def select_shard(items: List[dict], shard_index: int, num_shards: int, seed: int) -> List[dict]:
    """Deterministically pick this GPU's slice: seeded global shuffle for load
    balancing, round-robin assignment to shards, then sort the slice by duration
    so each decode batch holds similar-length clips (minimal padding waste)."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    shard = [items[j] for pos, j in enumerate(order) if pos % num_shards == shard_index]
    shard.sort(key=lambda it: it["dur"])
    return shard


def evaluate_shard(model, args, device: torch.device) -> int:
    """Decode this GPU's pooled slice of utterances (a mix of all datasets) and
    write a shard-unique, dataset-tagged generations JSONL for the reduce step.

    Per-batch decode errors are non-fatal: the batch's hyps are left empty and we
    keep going, so one bad clip can't corrupt every dataset's WER."""
    from transformers import GenerationConfig

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

    gen_kwargs = dict(
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        generation_config=GenerationConfig(do_sample=False),
        chunk_size_override=(args.chunk_size if args.chunk_size and args.chunk_size > 0 else None),
    )
    # Only forward when set, so non-redecode models never see an unexpected kwarg.
    if args.self_correct:
        gen_kwargs["self_correct"] = True
    if args.use_state_machine:
        gen_kwargs["use_state_machine_inference"] = True
    # SCRIPT-only kwarg; guard so non-SCRIPT model classes never see it.
    if args.force_word_start and "script" in args.model_class.lower():
        gen_kwargs["force_chunk_word_start"] = True

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"shard{args.shard_index}_of{args.num_shards}.generations.jsonl")

    per_ds_total = Counter(it["key"] for it in shard)
    per_ds_done: Counter = Counter()

    def _postfix() -> str:
        # Compact per-dataset done/total, ordered like the leaderboard suite.
        return " ".join(f"{k.split('/')[0]}:{per_ds_done[k]}/{per_ds_total[k]}" for k in sorted(per_ds_total))

    bs = max(1, int(args.batch_size))
    start = time.time()
    pbar = tqdm(
        total=total, desc=f"shard{args.shard_index}", unit="utt", ncols=140,
        mininterval=float(args.progress_interval), file=sys.stdout,
    )
    pbar.set_postfix_str(_postfix())
    with open(out_path, "w") as fout:
        for i in range(0, total, bs):
            batch = shard[i : i + bs]
            paths = [it["path"] for it in batch]
            audios, audio_lens = load_audio_batch(paths)
            audios = audios.to(device, non_blocking=True)
            audio_lens = audio_lens.to(device, non_blocking=True)
            try:
                with torch.inference_mode():
                    hyps = _generate_with_oom_backoff(model, audios, audio_lens, gen_kwargs, args.min_batch_size)
            except Exception as ex:  # noqa: BLE001 - non-fatal: keep other datasets intact
                _log(f"  WARN: batch [{i}:{i + len(batch)}] failed ({type(ex).__name__}: {ex}); emitting empty hyps")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                hyps = [""] * len(batch)
            for it, h in zip(batch, hyps):
                fout.write(json.dumps({"key": it["key"], "reference": it["ref"], "hypothesis": h}) + "\n")
                per_ds_done[it["key"]] += 1
            fout.flush()
            pbar.update(len(batch))
            pbar.set_postfix_str(_postfix())
    pbar.close()
    _log(f"shard {args.shard_index}: decoded {total} utts in {time.time() - start:.1f}s -> {out_path}")
    return 0


def aggregate_results(args) -> int:
    """Reduce step: pool every shard's dataset-tagged generations and compute WER
    per dataset (identical to non-sharded WER, since it's additive over utts)."""
    # Leaderboard-faithful WER (vendored normalizer + kaldialign merge_compounds),
    # NOT the training-time whisper-normalizer val_wer. See scripts/leaderboard_wer.py.
    from leaderboard_wer import WER

    files = sorted(glob.glob(os.path.join(args.output_dir, "shard*_of*.generations.jsonl")))
    if not files:
        _log(f"aggregate: no shard generation files under {args.output_dir}")
        return 1
    groups: Dict[str, Dict[str, list]] = defaultdict(lambda: {"refs": [], "hyps": []})
    for fn in files:
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                g = groups[rec["key"]]
                g["refs"].append(rec.get("reference", ""))
                g["hyps"].append(rec.get("hypothesis", ""))
    _log(f"aggregate: {len(files)} shard files -> {len(groups)} datasets")

    results = []
    for key in sorted(groups):
        g = groups[key]
        wer = WER(normalize=True, verbose=args.verbose)
        wer.update(key, refs=g["refs"], hyps=g["hyps"])
        wer_val = float(wer.compute()["wer"]) * 100.0
        # time field is meaningless post-shard; keep the RESULT schema for the launcher grep.
        _log(f"RESULT\t{key}\t{wer_val:.2f}\t0.0\t{len(g['refs'])}")
        results.append({"key": key, "wer": wer_val, "n": len(g["refs"])})

    if results:
        _log("\n  {:<28} {:>8} {:>10}".format("Dataset", "WER(%)", "N"))
        _log("  " + "-" * 48)
        tot = 0.0
        for r in results:
            _log(f"  {r['key']:<28} {r['wer']:>8.2f} {r['n']:>10d}")
            tot += r["wer"]
        _log("  " + "-" * 48)
        _log(f"  {'Average':<28} {tot / len(results):>8.2f}")
    return 0


def _parse_entries(datasets_arg: str) -> List[Tuple[str, str]]:
    entries = []
    for e in (x.strip() for x in datasets_arg.split(",")):
        if not e:
            continue
        name, _, split = e.partition(":")
        entries.append((name, split or "test"))
    return entries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_path", help="Lightning .ckpt to evaluate (required unless --aggregate).")
    p.add_argument(
        "--model_class",
        default="nemo.collections.speechlm2.models.script_model.ScriptSTTModel",
        help="Dotted path of the model class to load.",
    )
    p.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated 'name:split' list (default = full leaderboard). Pass one for per-GPU eval.",
    )
    p.add_argument("--cache_dir", help="Pre-staged cache root (<dataset>/<split>/_cache_manifest.jsonl).")
    p.add_argument("--device", type=int, default=0, help="GPU index (cuda:N).")
    # --- pooled-shard mode (balance work across GPUs regardless of dataset size) ---
    p.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="If >1, pool ALL datasets' utts, shuffle, and evaluate only this GPU's 1/num_shards slice "
        "(writes a dataset-tagged generations JSONL; run --aggregate afterwards for WER). Default 1 = "
        "legacy per-dataset inline-WER mode.",
    )
    p.add_argument("--shard_index", type=int, default=0, help="This shard's index in [0, num_shards).")
    p.add_argument("--shuffle_seed", type=int, default=1234, help="Seed for the global shuffle (stable across shards).")
    p.add_argument(
        "--aggregate",
        action="store_true",
        help="Reduce mode: read all shard generation JSONLs under --output_dir and print per-dataset WER.",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--min_batch_size", type=int, default=1, help="Lower bound for the OOM batch-halving.")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--chunk_size", type=int, default=None, help="Decode chunk size override (encoder frames).")
    p.add_argument(
        "--self_correct",
        action="store_true",
        help="SCRIPT redecode models only: emit the self-corrected LOCKED stream "
        "(re-decode each chunk with lookahead). Default (off) is the non-corrective "
        "j=0 stream (decode each chunk once, append). Ignored by other models.",
    )
    p.add_argument(
        "--use_state_machine",
        action="store_true",
        help="SCRIPT models only: decode via the streaming state machine "
        "(incremental cache-aware encode + spine/branch decode) instead of the "
        "up-front offline encode. Plain SCRIPT only (not redecode/last_layer/shared_audio).",
    )
    p.add_argument(
        "--force_word_start",
        dest="force_word_start",
        action="store_true",
        default=True,
        help="SCRIPT models only (ON by default): mask each chunk's FIRST decoded token "
        "to a word-start (leading-space BPE) or eot, so a chunk cannot merge onto the "
        "previous word (e.g. 'border ruffian' -> 'bordereruffian').",
    )
    p.add_argument(
        "--no_force_word_start",
        dest="force_word_start",
        action="store_false",
        help="Disable the chunk word-start enforcement (see --force_word_start).",
    )
    p.add_argument("--system_prompt", type=str, default="Transcribe the audio into text.")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--max_eval_samples", type=int, default=0, help="Cap samples per dataset (0 = all).")
    p.add_argument("--output_dir", type=str, default=None, help="If set, dump per-dataset generations JSONL.")
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

    # Reduce mode needs no GPU/model: just pool shard JSONLs and score.
    if args.aggregate:
        if not args.output_dir:
            _log("ERROR: --aggregate requires --output_dir (where shard JSONLs live).")
            return 1
        return aggregate_results(args)

    if not args.ckpt_path or not args.cache_dir:
        _log("ERROR: --ckpt_path and --cache_dir are required for evaluation.")
        return 1

    if not torch.cuda.is_available():
        _log("WARNING: CUDA not available; running on CPU (very slow).")
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    model = load_model(args.ckpt_path, args.model_class, device, dtype)

    # Pooled-shard mode: this process decodes only its balanced slice of all
    # datasets and writes a tagged JSONL; WER comes from the later --aggregate.
    if args.num_shards and args.num_shards > 1:
        if not args.output_dir:
            _log("ERROR: --num_shards > 1 requires --output_dir for shard generations.")
            return 1
        return evaluate_shard(model, args, device)

    # Legacy per-dataset mode (one or more full datasets in this process).
    entries = _parse_entries(args.datasets)
    results = []
    for dataset, split in entries:
        try:
            r = evaluate_dataset(model, args, dataset, split, device)
            if r is not None:
                results.append(r)
        except Exception as ex:  # noqa: BLE001
            _log(f"RESULT\t{dataset}/{split}\tERR\t0.0\t0  ({type(ex).__name__}: {ex})")

    if results:
        _log("\n  {:<28} {:>8} {:>10}".format("Dataset", "WER(%)", "Time(s)"))
        _log("  " + "-" * 48)
        tot = 0.0
        for r in results:
            _log(f"  {r['key']:<28} {r['wer']:>8.2f} {r['time']:>10.1f}")
            tot += r["wer"]
        _log("  " + "-" * 48)
        _log(f"  {'Average':<28} {tot / len(results):>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
