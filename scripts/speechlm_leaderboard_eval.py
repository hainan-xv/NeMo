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
ChunkCompletion) model, reading a PRE-STAGED wav/manifest cache on disk.

This is the SpeechLM counterpart of scripts/asr_leaderboard_eval.py, and unlike
the local ``run_eval_sslm.py`` driver it does NOT download from HuggingFace: it
reads each dataset from ``<cache_dir>/<dataset>/<split>/_cache_manifest.jsonl``
(the format written by run_eval_sslm's local cache — lines of
``{"audio_filepath","duration","reference"}`` pointing at 16 kHz mono wavs). This
makes it usable on OCI compute nodes that have no internet: pre-stage the cache
on lustre once, then evaluate there.

It loads ONE Lightning ``.ckpt`` into a configurable model class (default the
chunk-completion model), transcribes each dataset via ``model.generate`` (batched,
with CUDA-OOM batch-halving), and reports WER per dataset using the same
whisper-normalized metric as training's ``val_wer``.

Designed to eval ONE dataset per process/GPU (``--datasets ami:test --device 3``)
so a SLURM job can fan the leaderboard across a node's 8 GPUs
(see oci/eval_leaderboard_slurm.sh). It also accepts a comma-separated list to run
several datasets in one process (model loaded once).
"""
import argparse
import gc
import json
import os
import time
from typing import List, Tuple

import numpy as np
import soundfile
import torch

# Default Open-ASR-Leaderboard entries (name:split).
DEFAULT_DATASETS = [
    "librispeech:test.clean",
    "librispeech:test.other",
    "ami:test",
    "earnings22:test",
    "gigaspeech:test",
    "spgispeech:test",
    "tedlium:test",
    "voxpopuli:test",
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


def read_cache_manifest(cache_dir: str, dataset: str, split: str, max_samples: int = 0) -> Tuple[List[str], List[str]]:
    """Read a pre-staged split: returns (audio_filepaths, references)."""
    path = os.path.join(cache_dir, dataset, split, "_cache_manifest.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No pre-staged manifest for {dataset}/{split} at {path}. Stage the leaderboard "
            f"cache on this filesystem (the run_eval_sslm cache layout: "
            f"<cache_dir>/<dataset>/<split>/_cache_manifest.jsonl + 16kHz wavs)."
        )
    ds_dir = os.path.join(cache_dir, dataset, split)
    paths, refs = [], []
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
            if max_samples and len(paths) >= max_samples:
                break
    return paths, refs


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

    from nemo.collections.speechlm2.parts.metrics.wer import WER

    key = f"{dataset}/{split}"
    paths, refs = read_cache_manifest(args.cache_dir, dataset, split, args.max_eval_samples)
    if not paths:
        _log(f"{key}: no samples found; skipping")
        return None

    gen_kwargs = dict(
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        generation_config=GenerationConfig(do_sample=False),
        chunk_size_override=(args.chunk_size if args.chunk_size and args.chunk_size > 0 else None),
    )

    hyps: List[str] = []
    start = time.time()
    bs = max(1, int(args.batch_size))
    for i in range(0, len(paths), bs):
        audios, audio_lens = load_audio_batch(paths[i : i + bs])
        audios = audios.to(device, non_blocking=True)
        audio_lens = audio_lens.to(device, non_blocking=True)
        with torch.inference_mode():
            hyps.extend(_generate_with_oom_backoff(model, audios, audio_lens, gen_kwargs, args.min_batch_size))
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_path", required=True, help="Lightning .ckpt to evaluate.")
    p.add_argument(
        "--model_class",
        default="nemo.collections.speechlm2.models.chunk_completion_model.ChunkCompletionSTTModel",
        help="Dotted path of the model class to load.",
    )
    p.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated 'name:split' list (default = full leaderboard). Pass one for per-GPU eval.",
    )
    p.add_argument("--cache_dir", required=True, help="Pre-staged cache root (<dataset>/<split>/_cache_manifest.jsonl).")
    p.add_argument("--device", type=int, default=0, help="GPU index (cuda:N).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--min_batch_size", type=int, default=1, help="Lower bound for the OOM batch-halving.")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--chunk_size", type=int, default=None, help="Decode chunk size override (encoder frames).")
    p.add_argument("--system_prompt", type=str, default="Transcribe the audio into text.")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--max_eval_samples", type=int, default=0, help="Cap samples per dataset (0 = all).")
    p.add_argument("--output_dir", type=str, default=None, help="If set, dump per-dataset generations JSONL.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        _log("WARNING: CUDA not available; running on CPU (very slow).")
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    entries = []
    for e in (x.strip() for x in args.datasets.split(",")):
        if not e:
            continue
        name, _, split = e.partition(":")
        entries.append((name, split or "test"))

    model = load_model(args.ckpt_path, args.model_class, device, dtype)

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
