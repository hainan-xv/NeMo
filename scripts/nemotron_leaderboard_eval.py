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
"""Open-ASR-Leaderboard evaluation for a cache-aware streaming NeMo ASR model.

The baseline counterpart to ``script_leaderboard_eval.py``. Both drivers import
``leaderboard_common``, so the dataset list, the shard partition and the scoring
are identical by construction: a given utterance lands in the same shard and is
scored by the same normalizer for both systems. The only difference is how audio
becomes text, which is exactly what the comparison is about.

TWO DECODE MODES
----------------

``--mode offline`` (default) encodes each utterance in one pass with the
encoder's attention restricted to ``att_context_size = [left, chunk-1]``. Because
the model's ``att_context_style`` is ``chunked_limited``, that look-ahead does not
compound across layers, so a frame never depends on audio past its own chunk
boundary -- the dependency structure is the streaming one even though the
computation is batched. This is the matched comparison against SCRIPT, whose
evaluation encodes the same way, and it is far faster.

``--mode streaming`` runs true cache-aware streaming: mel chunks are fed through
``conformer_stream_step`` with the KV/conv caches carried between steps. This
additionally exercises cache-boundary effects, ``pre_encode_cache_size`` and
``valid_out_len`` truncation. It is the honest deployment number, and it is
slower. Cache-aware streaming is fp32-only in NeMo, so ``--dtype`` is ignored here.

The two normally agree closely; a large gap points at a streaming-path problem
rather than a model quality difference.

LATENCY POINTS
--------------
The model publishes the look-aheads it was trained for as
``encoder.att_context_size_all``. Requesting anything else silently degrades
accuracy (NeMo only warns), so this script validates ``--chunk_size`` against
that list and refuses unsupported values, printing the ones that are supported.

Usage:

    python scripts/nemotron_leaderboard_eval.py \\
        --model_path /path/to/nemotron-speech-streaming-en-0.6b.nemo \\
        --cache_dir /path/to/leaderboard_cache --chunk_size 14 \\
        --num_shards 8 --shard_index 0 --device 0 --output_dir /path/to/shards

    python scripts/nemotron_leaderboard_eval.py --aggregate --output_dir /path/to/shards
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

# Shared with script_leaderboard_eval.py -- this is what makes the two
# systems' numbers comparable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leaderboard_common import (  # noqa: E402
    DEFAULT_DATASETS,
    _log,
    aggregate_results,
    build_global_items,
    select_shard,
)

# Encoder frame duration: window_stride (10 ms) x subsampling_factor (8).
_FRAME_SECONDS = 0.08


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_model(model_path: str, device: torch.device, dtype: torch.dtype):
    """Restore a ``.nemo`` ASR model through its own concrete class.

    Mirrors NeMo's ``setup_model``: read the config first to discover
    ``target``, then restore via that class so model-specific logic runs.
    """
    from nemo.collections.asr.models import ASRModel
    from nemo.utils import model_utils

    _log(f"==> Restoring model: {model_path}")
    model_cfg = ASRModel.restore_from(restore_path=model_path, return_config=True)
    cls = model_utils.import_class_by_path(model_cfg.target)
    _log(f"    class: {cls.__name__}")
    model = cls.restore_from(restore_path=model_path, map_location=device)
    return model.to(dtype).to(device).eval()


def resolve_att_context(model, chunk_size: int, left_context: int = None) -> List[int]:
    """Map a chunk size in encoder frames to a supported ``att_context_size``.

    A chunk of ``C`` frames means the encoder may look ahead ``C-1`` frames, i.e.
    exactly to its own chunk boundary. ``set_default_att_context_size`` only warns
    on unsupported values, which would silently produce a bad number, so validate
    here and fail loudly instead.
    """
    supported = [list(x) for x in getattr(model.encoder, "att_context_size_all", [])]
    if not supported:
        raise ValueError("Model exposes no att_context_size_all; it is not a multi-lookahead cache-aware model.")

    default_left = int(model.encoder.att_context_size[0])
    left = int(left_context) if left_context is not None else default_left
    want = [left, int(chunk_size) - 1]

    if want not in supported:
        opts = ", ".join(f"chunk={c[1] + 1} ({(c[1] + 1) * _FRAME_SECONDS:.2f}s) -> {c}" for c in supported)
        raise ValueError(
            f"att_context_size {want} (chunk_size={chunk_size}) is not one this model was trained for.\n"
            f"Supported: {opts}\n"
            f"Using an untrained look-ahead silently degrades accuracy, so refusing."
        )
    return want


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _texts_from(hyps) -> List[str]:
    """``transcribe`` / ``conformer_stream_step`` return Hypothesis objects or str."""
    out = []
    for h in hyps:
        if isinstance(h, str):
            out.append(h)
        else:
            out.append(getattr(h, "text", "") or "")
    return out


def transcribe_offline(model, paths: List[str], batch_size: int) -> List[str]:
    """One full-utterance forward per clip, with chunk-limited attention."""
    with torch.inference_mode():
        hyps = model.transcribe(paths, batch_size=batch_size, verbose=False)
    # n-best configs return (best, all); take the best list.
    if isinstance(hyps, tuple) and len(hyps) == 2:
        hyps = hyps[0]
    return _texts_from(hyps)


def _per_step_emissions(step_texts, n_utts):
    """Per-utterance [[step_index, text_emitted_at_that_step], ...].

    ``step_texts[t][u]`` is utterance u's CUMULATIVE transcript after step t, so
    what step t emitted is the suffix beyond step t-1. Compared on the common
    word prefix rather than by string slicing: RNN-T may revise its tail between
    steps, and a raw suffix diff would then attribute the rewritten words to the
    wrong step.
    """
    out = []
    for u in range(n_utts):
        prev, per = [], []
        for t, texts in enumerate(step_texts):
            cur = (texts[u] if u < len(texts) else "").split()
            c = 0
            while c < len(prev) and c < len(cur) and prev[c] == cur[c]:
                c += 1
            new = cur[c:]
            if new:
                per.append([t, " ".join(new)])
            prev = cur
        out.append(per)
    return out


def transcribe_streaming(model, paths: List[str], pad_and_drop_preencoded: bool = False, per_step: bool = False):
    """True cache-aware streaming: feed mel chunks, carry the caches across steps.

    Mirrors NeMo's ``speech_to_text_cache_aware_streaming_infer.py``. Note the
    batch is static: every stream runs for as many chunks as the longest one, so
    duration-sorted batches (which ``select_shard`` produces) matter a lot here.
    """
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    # The buffer snapshots encoder.streaming_cfg at construction, so the
    # att_context_size must already be set by the time we get here.
    buf = CacheAwareStreamingAudioBuffer(
        model=model, online_normalization=False, pad_and_drop_preencoded=pad_and_drop_preencoded
    )
    for p in paths:
        buf.append_audio_file(p, stream_id=-1)

    batch_size = len(buf.streams_length)
    cache_last_channel, cache_last_time, cache_last_channel_len = model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )
    previous_hypotheses = None
    pred_out_stream = None
    transcribed = []
    # Cumulative transcript after each streaming step. Differencing consecutive
    # entries gives what THAT step emitted -- the RNN-T analogue of SCRIPT's
    # per-chunk emission, without touching the decoder.
    step_texts: List[List[str]] = []

    for step_num, (chunk_audio, chunk_lengths) in enumerate(buf):
        # Step 0 has no cache yet, so nothing to drop after pre-encoding.
        drop = (
            0
            if (step_num == 0 and not pad_and_drop_preencoded)
            else model.encoder.streaming_cfg.drop_extra_pre_encoded
        )
        with torch.inference_mode():
            (
                pred_out_stream,
                transcribed,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                previous_hypotheses,
            ) = model.conformer_stream_step(
                processed_signal=chunk_audio,
                processed_signal_length=chunk_lengths,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                # True only on the final chunk; otherwise outputs are trimmed to valid_out_len.
                keep_all_outputs=buf.is_buffer_empty(),
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=pred_out_stream,
                drop_extra_pre_encoded=drop,
                return_transcription=True,
            )
        if per_step:
            step_texts.append(list(_texts_from(transcribed)))
    buf.reset_buffer()
    if per_step:
        return _texts_from(transcribed), step_texts
    return _texts_from(transcribed)


def evaluate_shard(model, args) -> None:
    items = build_global_items(args)
    shard = select_shard(items, args.num_shards, args.shard_index, args.shuffle_seed)
    _log(f"==> shard {args.shard_index}/{args.num_shards}: {len(shard)} of {len(items)} pooled utts")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"shard{args.shard_index}_of{args.num_shards}.generations.jsonl")

    done, total = defaultdict(int), defaultdict(int)
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
                if args.mode == "streaming":
                    if args.emit_chunk_ids:
                        hyps, step_texts = transcribe_streaming(
                            model, paths, args.pad_and_drop_preencoded, per_step=True
                        )
                        chunks = _per_step_emissions(step_texts, len(paths))
                    else:
                        hyps = transcribe_streaming(model, paths, args.pad_and_drop_preencoded)
                else:
                    hyps = transcribe_offline(model, paths, len(paths))
            except Exception as e:  # one bad batch must not kill the shard
                _log(f"    [WARN] batch at {i} failed ({type(e).__name__}: {e}); emitting empty hypotheses")
                hyps = [""] * len(batch)
                chunks = [None] * len(batch)
            if len(hyps) != len(batch):
                _log(f"    [WARN] got {len(hyps)} hyps for {len(batch)} utts; padding")
                hyps = (list(hyps) + [""] * len(batch))[: len(batch)]
            if not args.emit_chunk_ids or args.mode != "streaming":
                chunks = [None] * len(batch)
            for b, hyp, ch in zip(batch, hyps, chunks):
                rec = {"key": b["key"], "reference": b["ref"], "hypothesis": hyp}
                if ch is not None:
                    rec["chunks"] = ch
                fout.write(json.dumps(rec) + "\n")
                done[b["key"]] += 1
            fout.flush()
            bar.set_postfix_str(" ".join(f"{k.split('/')[0]}:{done[k]}/{total[k]}" for k in sorted(total)))

    _log(f"==> shard {args.shard_index} wrote {out_path} in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--model_path", type=str, default=None, help="path to the .nemo model")
    p.add_argument("--cache_dir", type=str, default=None, help="pre-staged leaderboard cache root")
    p.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    p.add_argument("--output_dir", type=str, default=None)

    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="offline mode only; cache-aware streaming is fp32-only in NeMo",
    )

    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument(
        "--shuffle_seed",
        type=int,
        default=1234,
        help="keep identical to the SCRIPT eval so both see the same partition",
    )

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_eval_samples", type=int, default=0, help="cap per dataset; 0 = all")

    p.add_argument(
        "--mode",
        choices=["offline", "streaming"],
        default="offline",
        help="offline = chunk-limited full-utterance encode (matches the SCRIPT eval); "
        "streaming = true cache-aware chunk-by-chunk decode",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=14,
        help="encoder frames per chunk (0.08s each); validated against the model's trained look-aheads",
    )
    p.add_argument(
        "--left_context", type=int, default=None, help="left attention context in frames; default = the model's own"
    )
    p.add_argument(
        "--pad_and_drop_preencoded",
        action="store_true",
        help="streaming mode: also cache/drop pre-encoded frames on step 0",
    )

    p.add_argument(
        "--emit_chunk_ids",
        action="store_true",
        help="record which streaming step emitted each word (adds 'chunks'); --mode streaming only",
    )
    p.add_argument("--aggregate", action="store_true", help="reduce shard JSONLs; no GPU or model needed")
    p.add_argument("--progress_interval", type=float, default=5.0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.aggregate:
        if not args.output_dir:
            _log("ERROR: --aggregate requires --output_dir")
            return 1
        return aggregate_results(args)

    for required in ("model_path", "cache_dir", "output_dir"):
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

    # NeMo's cache-aware streaming path only supports fp32.
    dtype = torch.float32
    if args.mode == "offline" and args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.mode == "streaming" and args.dtype != "fp32":
        _log("==> streaming mode is fp32-only in NeMo; ignoring --dtype bf16")

    model = load_model(args.model_path, device, dtype)

    try:
        att = resolve_att_context(model, args.chunk_size, args.left_context)
    except ValueError as e:
        _log(f"ERROR: {e}")
        return 1
    # Also recomputes encoder.streaming_cfg, which the streaming buffer reads.
    model.encoder.set_default_att_context_size(att)
    _log(
        f"==> mode={args.mode} chunk_size={args.chunk_size} frames "
        f"({args.chunk_size * _FRAME_SECONDS:.2f}s) att_context_size={att} dtype={dtype}"
    )
    _log(f"==> streaming_cfg: {model.encoder.streaming_cfg}")

    evaluate_shard(model, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
