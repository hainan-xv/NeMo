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
"""Per-shard ASR (RNNT / CHAT) decode for the pooled Open-ASR-Leaderboard eval.

This is the ASR counterpart of the heh decode engine
(examples/speechlm2/streaming_stt_generate.py) in the pooled multi-GPU OCI eval
(oci/eval_leaderboard_chat_slurm.sh). It plugs into the SAME balancing machinery:

  scripts/leaderboard_heh_shards.py build      -> shard{k}_of{N}.json  (this reads)
  THIS SCRIPT (one process per GPU)             -> shard{k}_of{N}.generations.jsonl
  scripts/leaderboard_heh_shards.py aggregate  -> per-dataset WER       (reads that)

It loads ONE NeMo ASR model (.nemo or .ckpt), transcribes a single pooled shard
manifest ({audio_filepath, duration, text, dataset_key}), and writes one
generations row per utterance with the fields the aggregator expects:
``{"dataset_key", "text" (normalized ref), "pred_text" (normalized hyp)}``.

Both reference and hypothesis are normalized with the SAME normalizer heh uses
(whisper EnglishTextNormalizer by default), so per-dataset WER is directly
comparable to the SpeechLM leaderboard runs reduced by the same aggregator.
"""
import argparse
import json
import os
import sys
import time

# Reuse the model loading / decoding-override / transcribe helpers from the
# self-contained ASR leaderboard evaluator (same scripts/ directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_leaderboard_eval import (  # noqa: E402
    load_model,
    maybe_override_decoding,
    normalize_text as _builtin_normalize,
    transcribe,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def get_normalizer(kind: str):
    """Return a text normalizer callable matching heh's ``use_normalizer``.

    'english' (default) and 'basic' use whisper_normalizer to match
    streaming_stt_generate exactly; 'none' disables normalization; if
    whisper_normalizer is unavailable we fall back to the builtin light
    normalizer from asr_leaderboard_eval (lowercase + strip punctuation).
    """
    kind = (kind or "english").lower()
    if kind == "none":
        return lambda x: (x or "")
    try:
        if kind == "basic":
            from whisper_normalizer.basic import BasicTextNormalizer

            return BasicTextNormalizer()
        from whisper_normalizer.english import EnglishTextNormalizer

        return EnglishTextNormalizer()
    except Exception as e:  # noqa: BLE001
        _log(f"WARNING: whisper_normalizer unavailable ({e}); using builtin normalizer.")
        return _builtin_normalize


def read_shard(path: str, gt_field: str = "text"):
    """Read a pooled shard NeMo manifest -> (audio_paths, refs, dataset_keys)."""
    audio_paths, refs, keys = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fp = rec.get("audio_filepath")
            if not fp:
                continue
            audio_paths.append(fp)
            refs.append(rec.get(gt_field, rec.get("text", "")) or "")
            keys.append(rec.get("dataset_key", "unknown"))
    return audio_paths, refs, keys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard_manifest", required=True, help="Pooled shard manifest (shard{k}_of{N}.json).")
    p.add_argument("--model_path", required=True, help="ASR model (.nemo preferred, or Lightning .ckpt).")
    p.add_argument("--output", required=True, help="Output generations JSONL (shard{k}_of{N}.generations.jsonl).")
    p.add_argument("--device", default="0", help="CUDA device index (with CUDA_VISIBLE_DEVICES set, use 0), or 'cpu'.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gt_text_field", default="text", help="Reference text field in the shard manifest.")
    p.add_argument(
        "--use_normalizer",
        default="english",
        help="Text normalizer for BOTH ref and hyp: english (default) | basic | none. "
        "Matches heh's streaming_stt_generate so WER is comparable.",
    )
    # Optional decoding / streaming overrides (consumed by maybe_override_decoding).
    p.add_argument("--max_symbols", type=int, default=None, help="Override greedy max symbols per (chunk) step.")
    p.add_argument("--att_context_size", default=None, help="Override encoder att context, e.g. '[70,13]'.")
    p.add_argument("--chunk_size", type=int, default=None, help="Override CHAT joint chunk_size (full-context models).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    audio_paths, refs, keys = read_shard(args.shard_manifest, args.gt_text_field)
    _log(f"==> shard {os.path.basename(args.shard_manifest)}: {len(audio_paths)} utts")
    if not audio_paths:
        # Write an empty file so the aggregator's glob still succeeds.
        open(args.output, "w").close()
        _log("    empty shard; wrote empty generations file.")
        return 0

    model = load_model(args.model_path, args.device)
    maybe_override_decoding(model, args)

    t0 = time.time()
    hyps = transcribe(model, audio_paths, args.batch_size)
    dt = time.time() - t0
    _log(f"    transcribed {len(hyps)} utts in {dt:.1f}s ({len(hyps) / max(dt, 1e-6):.1f} utt/s)")

    normalizer = get_normalizer(args.use_normalizer)

    n = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for fp, ref_raw, hyp_raw, key in zip(audio_paths, refs, hyps, keys):
            ref_norm = normalizer(ref_raw) if ref_raw is not None else ""
            hyp_norm = normalizer(hyp_raw) if hyp_raw is not None else ""
            out.write(
                json.dumps(
                    {
                        "audio_filepath": fp,
                        "dataset_key": key,
                        # Fields consumed by leaderboard_heh_shards.py aggregate:
                        "text": ref_norm,
                        "pred_text": hyp_norm,
                        # Raw text kept for offline inspection / error analysis.
                        "ref_raw": ref_raw,
                        "hyp_raw": hyp_raw,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    _log(f"    wrote {n} generations -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
