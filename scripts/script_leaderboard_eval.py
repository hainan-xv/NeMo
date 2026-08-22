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
"""Open-ASR-Leaderboard evaluation for the SCRIPT streaming SpeechLM.

Reads a PRE-STAGED on-disk dataset cache (see ``scripts/stage_leaderboard_cache.py``)
rather than downloading anything: compute nodes run with ``HF_HUB_OFFLINE=1``, and
eight processes racing to populate an HF cache is both slow and unsafe.

Two modes:

* **decode** (default) --- decode one shard of the pooled utterance list and append
  ``{"key", "reference", "hypothesis"}`` records to a per-shard JSONL. Every
  dataset is pooled and globally shuffled with a shared seed before being split
  ``pos % num_shards``, so each GPU gets a length-balanced mix rather than whole
  datasets; within a shard, utterances are sorted by duration so batches are
  homogeneous. Run one process per GPU.
* **aggregate** (``--aggregate``) --- reduce every shard JSONL under
  ``--output_dir`` into per-dataset WER plus an unweighted macro average. Needs no
  GPU and no model.

Scoring uses the vendored Open-ASR-Leaderboard normalizer and ``kaldialign``'s
compound-aware error rate (``scripts/leaderboard_wer.py``), so numbers are
comparable with the public leaderboard.

Usage (decode one shard):

    python scripts/script_leaderboard_eval.py \\
        --ckpt_path /path/to/model.ckpt --cache_dir /path/to/leaderboard_cache \\
        --num_shards 8 --shard_index 0 --device 0 --output_dir /path/to/shards

Usage (reduce):

    python scripts/script_leaderboard_eval.py --aggregate --output_dir /path/to/shards
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import List

import torch
from tqdm import tqdm

# leaderboard_common sits next to this file; sys.path[0] is the script's own
# directory, so a plain import works when run as a script. Sharing these with
# nemotron_leaderboard_eval.py is what makes the two systems' numbers comparable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leaderboard_common import (  # noqa: E402
    DEFAULT_DATASETS,
    _log,
    aggregate_results,
    build_global_items,
    load_audio_batch,
    select_shard,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _hubify(path: str) -> str:
    """Map a local ``.../huggingface/<org>/<name>`` path to its hub id.

    Checkpoints record the absolute paths of the machine that trained them; on a
    different filesystem those do not exist, but the hub id usually resolves from
    the local HF cache.
    """
    parts = path.rstrip("/").split("/")
    if "huggingface" in parts:
        i = parts.index("huggingface")
        if len(parts) >= i + 3:
            return "/".join(parts[i + 1 : i + 3])
    return path


def load_model(ckpt_path: str, model_class_path: str, device: torch.device, dtype: torch.dtype):
    """Rebuild the model from a Lightning checkpoint's own hyper-parameters.

    Deliberately not ``load_from_checkpoint``: we want to override a few config
    entries (skip re-downloading base LLM weights that the checkpoint already
    carries, repair stale absolute paths) before instantiating.
    """
    from nemo.utils.model_utils import import_class_by_path

    _log(f"==> Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["hyper_parameters"]["cfg"]
    state_dict = ckpt["state_dict"]

    # The checkpoint holds the adapted LLM weights, so skip loading base weights.
    cfg["load_llm_weights"] = False
    for key in ("pretrained_llm", "pretrained_asr"):
        p = cfg.get(key, "")
        if p and not os.path.exists(p):
            new = _hubify(p)
            if new != p:
                _log(f"    {key}: {p} -> {new}")
                cfg[key] = new

    # Older checkpoints may omit lora.target_modules; recover it from the weights.
    if "lora" in cfg and cfg["lora"] and "target_modules" not in cfg["lora"]:
        mods = sorted({k.split(".")[-4] for k in state_dict if ".lora_A." in k})
        cfg["lora"]["target_modules"] = mods or "all-linear"
        _log(f"    recovered lora.target_modules: {cfg['lora']['target_modules']}")

    cls = import_class_by_path(model_class_path)
    model = cls(cfg=cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        _log(f"    [warn] {len(missing)} missing keys (first 5): {list(missing)[:5]}")
    if unexpected:
        _log(f"    [warn] {len(unexpected)} unexpected keys (first 5): {list(unexpected)[:5]}")
    return model.eval().to(dtype).to(device)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _generate(model, audios, audio_lens, gen_kwargs, min_batch_size: int):
    """Decode one batch, halving on CUDA OOM down to ``min_batch_size``."""
    n = int(audios.shape[0])
    try:
        with torch.inference_mode():
            return model.generate(audios=audios, audio_lens=audio_lens, **gen_kwargs)
    except torch.cuda.OutOfMemoryError:
        if n <= min_batch_size:
            raise
    # Retry OUTSIDE the except block, so the failed batch's traceback -- and the
    # tensors it still references -- are released before we allocate again.
    torch.cuda.empty_cache()
    half = max(min_batch_size, n // 2)
    _log(f"    [oom] retrying {n} utts as sub-batches of {half}")
    out: List[str] = []
    for i in range(0, n, half):
        sl = slice(i, min(i + half, n))
        sub_lens = audio_lens[sl]
        # Trim to the sub-batch's own longest clip; keeping the parent batch's
        # padding width would defeat the point of splitting.
        sub_audios = audios[sl, : int(sub_lens.max().item())]
        out.extend(_generate(model, sub_audios, sub_lens, gen_kwargs, min_batch_size))
    return out


def build_gen_kwargs(args) -> dict:
    from transformers import GenerationConfig

    kwargs = dict(
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        generation_config=GenerationConfig(do_sample=False),
        chunk_size_override=(args.chunk_size if args.chunk_size and args.chunk_size > 0 else None),
        force_word_start=args.force_word_start,
    )
    if args.max_history_tokens > 0:
        kwargs["max_history_tokens"] = args.max_history_tokens
    return kwargs


def evaluate_shard(model, args, device) -> None:
    items = build_global_items(args)
    shard = select_shard(items, args.num_shards, args.shard_index, args.shuffle_seed)
    _log(f"==> shard {args.shard_index}/{args.num_shards}: {len(shard)} of {len(items)} pooled utts")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"shard{args.shard_index}_of{args.num_shards}.generations.jsonl")
    gen_kwargs = build_gen_kwargs(args)

    done = defaultdict(int)
    total = defaultdict(int)
    for it in shard:
        total[it["key"]] += 1

    t0 = time.time()
    with open(out_path, "w") as fout:
        bar = tqdm(
            range(0, len(shard), args.batch_size), mininterval=args.progress_interval, desc=f"shard{args.shard_index}"
        )
        for i in bar:
            batch = shard[i : i + args.batch_size]
            paths = [b["path"] for b in batch]
            try:
                audios, audio_lens = load_audio_batch(paths, args.pad_extra_seconds)
                hyps = _generate(model, audios.to(device), audio_lens.to(device), gen_kwargs, args.min_batch_size)
            except Exception as e:  # a bad batch must not kill the whole shard
                _log(f"    [WARN] batch at {i} failed ({type(e).__name__}: {e}); emitting empty hypotheses")
                hyps = [""] * len(batch)
            for b, hyp in zip(batch, hyps):
                fout.write(json.dumps({"key": b["key"], "reference": b["ref"], "hypothesis": hyp}) + "\n")
                done[b["key"]] += 1
            fout.flush()
            bar.set_postfix_str(" ".join(f"{k.split('/')[0]}:{done[k]}/{total[k]}" for k in sorted(total)))

    _log(f"==> shard {args.shard_index} wrote {out_path} in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--ckpt_path", type=str, default=None, help="Lightning .ckpt to evaluate")
    p.add_argument(
        "--model_class",
        type=str,
        default="nemo.collections.speechlm2.models.script_model.ScriptSTTModel",
        help="dotted path of the model class",
    )
    p.add_argument("--cache_dir", type=str, default=None, help="pre-staged leaderboard cache root")
    p.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS), help="comma-separated name:split list")
    p.add_argument("--output_dir", type=str, default=None, help="where shard JSONLs are written/read")

    p.add_argument("--device", type=int, default=0, help="GPU index (each shard sees one GPU as cuda:0)")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")

    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument(
        "--shuffle_seed",
        type=int,
        default=1234,
        help="must be identical across shards or the union is not the full set",
    )

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--min_batch_size", type=int, default=1, help="floor for OOM batch halving")
    p.add_argument("--max_eval_samples", type=int, default=0, help="cap per dataset; 0 = all")

    # Decode knobs. These mirror ScriptSTTModel.generate.
    p.add_argument(
        "--system_prompt",
        type=str,
        default="Transcribe the audio into text.",
        help="MUST match the model's training instruction; drift is out-of-distribution",
    )
    p.add_argument("--max_new_tokens", type=int, default=64, help="per-chunk decode cap")
    p.add_argument(
        "--chunk_size", type=int, default=None, help="decode chunk size in encoder frames; omit for the model default"
    )
    p.add_argument(
        "--max_history_tokens", type=int, default=0, help="cap conditioning history; 0 = model config default"
    )
    p.add_argument(
        "--force_word_start",
        dest="force_word_start",
        action="store_true",
        default=True,
        help="insert a word-start token when a chunk's first token is a continuation (default on)",
    )
    p.add_argument("--no_force_word_start", dest="force_word_start", action="store_false")
    p.add_argument(
        "--pad_extra_seconds",
        type=float,
        default=0.5,
        help="trailing silence appended per clip; match training's pad_extra_duration",
    )

    p.add_argument("--aggregate", action="store_true", help="reduce shard JSONLs; no GPU or model needed")
    p.add_argument("--progress_interval", type=float, default=5.0, help="tqdm mininterval (log-friendly)")
    p.add_argument("--verbose", action="store_true", help="print sample normalized ref/hyp pairs")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.aggregate:
        if not args.output_dir:
            _log("ERROR: --aggregate requires --output_dir")
            return 1
        return aggregate_results(args)

    for required in ("ckpt_path", "cache_dir", "output_dir"):
        if not getattr(args, required):
            _log(f"ERROR: --{required} is required for decoding")
            return 1
    if not (0 <= args.shard_index < args.num_shards):
        _log(f"ERROR: --shard_index {args.shard_index} out of range for --num_shards {args.num_shards}")
        return 1

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        _log("WARNING: CUDA unavailable; falling back to CPU (this will be very slow)")
        device = torch.device("cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    model = load_model(args.ckpt_path, args.model_class, device, dtype)
    _log(f"==> system_prompt: {args.system_prompt!r}")
    _log(
        f"==> chunk_size={args.chunk_size} force_word_start={args.force_word_start} "
        f"pad_extra_seconds={args.pad_extra_seconds}"
    )
    evaluate_shard(model, args, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
