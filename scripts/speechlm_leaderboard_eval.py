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
"""Open-ASR-Leaderboard evaluation for the ORIGINAL streaming SpeechLM.

The third driver alongside ``script_leaderboard_eval.py`` (SCRIPT) and
``nemotron_leaderboard_eval.py`` (cache-aware RNNT). All three import
``leaderboard_common``, so the dataset list, the shard partition and the scoring
are identical by construction -- a given utterance lands in the same shard and is
scored by the same normalizer for every system. The only thing that differs is
how audio becomes text, which is the entire point of the comparison.

WHY A SEPARATE DRIVER FROM SCRIPT'S
-----------------------------------
``StreamingSTTModel.generate`` returns a ``StreamingSTTGenerateResult`` dataclass
(``.texts`` plus optional alignments / gate scores / debug logs), where
``ScriptSTTModel.generate`` returns a plain ``list[str]``. Feeding the dataclass
into the SCRIPT driver would silently iterate the wrong object. It also takes a
different set of decode knobs, so keeping the two apart is clearer than
special-casing one driver.

ENCODING MODE
-------------
``--offline_embs`` (default) builds the per-chunk embeddings in one pass.
``generate`` pins ``att_context_size = [left, chunk-1]`` for the call either way,
so a frame never depends on audio past its own chunk boundary: the dependency
structure is the streaming one even though the computation is batched. This is
the matched comparison against SCRIPT (whose evaluation encodes the same way) and
against ``nemotron_leaderboard_eval.py --mode offline``, and it is much faster.

``--streaming_embs`` runs the true cache-aware streaming perception instead. That
is the honest deployment number and exercises cache-boundary effects. The two
should agree closely; a large gap points at a streaming-path problem rather than
a model quality difference.

Usage (decode one shard):

    python scripts/speechlm_leaderboard_eval.py \\
        --ckpt_path /path/to/step=200000-last.ckpt \\
        --cache_dir /path/to/leaderboard_cache --chunk_size 14 \\
        --num_shards 8 --shard_index 0 --device 0 --output_dir /path/to/shards

Usage (reduce):

    python scripts/speechlm_leaderboard_eval.py --aggregate --output_dir /path/to/shards
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
    """Rebuild the model from its Lightning checkpoint's own hyper-parameters.

    Deliberately not ``load_from_checkpoint``: a few config entries need
    overriding first (skip re-downloading base LLM weights the checkpoint already
    carries, repair stale absolute paths).

    Missing/unexpected key counts are printed rather than swallowed. Evaluating
    someone else's checkpoint against this repo's copy of the model class is
    exactly where a silent architecture drift would show up, and a WER that looks
    merely "a bit poor" is a much worse outcome than a loud warning.
    """
    from nemo.utils.model_utils import import_class_by_path

    _log(f"==> Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["hyper_parameters"]["cfg"]
    state_dict = ckpt["state_dict"]

    cfg["load_llm_weights"] = False
    for key in ("pretrained_llm", "pretrained_asr"):
        p = cfg.get(key, "")
        if p and not os.path.exists(p):
            new = _hubify(p)
            if new != p:
                _log(f"    {key}: {p} -> {new}")
                cfg[key] = new

    if "lora" in cfg and cfg["lora"] and "target_modules" not in cfg["lora"]:
        mods = sorted({k.split(".")[-4] for k in state_dict if ".lora_A." in k})
        cfg["lora"]["target_modules"] = mods or "all-linear"
        _log(f"    recovered lora.target_modules: {cfg['lora']['target_modules']}")

    for k in ("chunk_size", "att_context_size", "blank_token", "compact_template", "num_delay_frames"):
        if k in cfg:
            _log(f"    cfg.{k} = {cfg[k]!r}")

    cls = import_class_by_path(model_class_path)
    model = cls(cfg=cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        _log(f"    [WARN] {len(missing)} missing keys (first 5): {list(missing)[:5]}")
    if unexpected:
        _log(f"    [WARN] {len(unexpected)} unexpected keys (first 5): {list(unexpected)[:5]}")
    if missing or unexpected:
        _log("    [WARN] a large count here means this repo's model class has drifted from the")
        _log("           one that trained the checkpoint -- treat the WER as unreliable.")
    return model.eval().to(dtype).to(device)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _texts_of(result) -> List[str]:
    """Pull the transcripts out of whatever ``generate`` returned.

    ``StreamingSTTModel`` returns a ``StreamingSTTGenerateResult``; older or
    simpler variants return a plain list of strings. Accept both rather than
    assuming, because getting this wrong yields hypotheses that are the repr of a
    dataclass and a WER near 100% that looks like a model failure.
    """
    texts = getattr(result, "texts", None)
    if texts is not None:
        return list(texts)
    if isinstance(result, (list, tuple)):
        return [t if isinstance(t, str) else (getattr(t, "text", "") or "") for t in result]
    raise TypeError(f"generate() returned {type(result).__name__}, which has no .texts and is not a list")


def _generate(model, audios, audio_lens, gen_kwargs, min_batch_size: int) -> List[str]:
    """Decode one batch, halving on CUDA OOM down to ``min_batch_size``."""
    n = int(audios.shape[0])
    try:
        with torch.inference_mode():
            return _texts_of(model.generate(audios=audios, audio_lens=audio_lens, **gen_kwargs))
    except torch.cuda.OutOfMemoryError:
        if n <= min_batch_size:
            raise
    # Retry OUTSIDE the except block so the failed batch's traceback -- and the
    # tensors it still references -- are released before we allocate again.
    torch.cuda.empty_cache()
    half = max(min_batch_size, n // 2)
    _log(f"    [oom] retrying {n} utts as sub-batches of {half}")
    out: List[str] = []
    for i in range(0, n, half):
        sl = slice(i, min(i + half, n))
        sub_lens = audio_lens[sl]
        sub_audios = audios[sl, : int(sub_lens.max().item())]
        out.extend(_generate(model, sub_audios, sub_lens, gen_kwargs, min_batch_size))
    return out


def build_gen_kwargs(args) -> dict:
    from transformers import GenerationConfig

    return dict(
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        generation_config=GenerationConfig(do_sample=False),
        chunk_size_override=(args.chunk_size if args.chunk_size and args.chunk_size > 0 else None),
        use_offline_embs=args.offline_embs,
        emit_delay_frames=args.emit_delay_frames,
        # Both are expensive and unused for WER; alignments are on by default.
        return_alignments=False,
        return_debug_logs=False,
    )


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
            if len(hyps) != len(batch):
                _log(f"    [WARN] got {len(hyps)} hypotheses for {len(batch)} utts; padding")
                hyps = (list(hyps) + [""] * len(batch))[: len(batch)]
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
        default="nemo.collections.speechlm2.models.streaming_stt_model.StreamingSTTModel",
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
        "--shuffle_seed", type=int, default=1234, help="must be identical across shards, and across systems compared"
    )

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--min_batch_size", type=int, default=1, help="floor for OOM batch halving")
    p.add_argument("--max_eval_samples", type=int, default=0, help="cap per dataset; 0 = all")

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
        "--emit_delay_frames", type=int, default=0, help="inference-time emission delay in frames (0 = model default)"
    )
    p.add_argument(
        "--offline_embs",
        dest="offline_embs",
        action="store_true",
        default=True,
        help="batch the per-chunk embeddings (default). Attention is still chunk-limited, so the "
        "dependency structure is the streaming one -- this is the matched comparison against SCRIPT.",
    )
    p.add_argument(
        "--streaming_embs",
        dest="offline_embs",
        action="store_false",
        help="true cache-aware streaming perception instead; slower, the honest deployment number",
    )
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
        f"==> chunk_size={args.chunk_size} offline_embs={args.offline_embs} "
        f"emit_delay_frames={args.emit_delay_frames} pad_extra_seconds={args.pad_extra_seconds}"
    )
    evaluate_shard(model, args, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
