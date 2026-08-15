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
"""NeMo-NATIVE per-GPU shard driver for the Open-ASR-Leaderboard eval.

This is the ``BACKEND=nemo`` twin of scripts/parakeet_leaderboard_eval.py: it does
the EXACT same pooled-shard work division (``build_global_items`` + ``select_shard``
with the same seed) and writes the EXACT same dataset-tagged generations JSONL
(``shard{idx}_of{N}.generations.jsonl`` with ``{key, reference, hypothesis}``), so
the identical ``--aggregate`` reduce + leaderboard scorer applies unchanged. The
ONLY thing that differs is the decode: instead of our own ``model.transcribe()``
loop, this shells out to NeMo's OWN maintained ``examples/asr/speech_to_text_eval.py``
(which wraps ``transcribe_speech.py``: duration presort, amp autocast, its own
batching). Running both backends over the same shards on the same GPUs isolates the
decoder harness as the only variable when comparing per-dataset WER.

Flow (one process per GPU, GPU pinned by the launcher via CUDA_VISIBLE_DEVICES):
  1. pool + shard the suite  -> this GPU's balanced slice (mix of all datasets)
  2. write that slice as a NeMo manifest, carrying a ``key`` field per utt
  3. subprocess speech_to_text_eval.py on the shard manifest (cuda=0 -> the one
     visible GPU); it preserves input order + all input fields and adds pred_text
  4. rewrite its output as the shared generations JSONL ({key, reference,
     hypothesis=pred_text}) for the common aggregator
"""
import argparse
import json
import os
import subprocess
import sys
from typing import List

# Reuse the SpeechLM driver's pooled-shard plumbing so the work division is
# byte-for-byte identical to the transcribe backend. Sibling import: running
# `python scripts/parakeet_nemo_shard_eval.py` puts scripts/ on sys.path[0].
from speechlm_leaderboard_eval import (
    DEFAULT_DATASETS,
    _log,
    _parse_entries,
    build_global_items,
    select_shard,
)

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
_ASR_EXAMPLES_DIR = os.path.join(_REPO_ROOT, "examples", "asr")


def _write_shard_manifest(shard: List[dict], path: str) -> None:
    """Dump this shard's utts as a NeMo manifest, carrying ``key`` so the dataset
    identity survives speech_to_text_eval's write-back (it copies every input field
    and only ADDS pred_text)."""
    with open(path, "w", encoding="utf-8") as f:
        for it in shard:
            f.write(
                json.dumps(
                    {
                        "audio_filepath": it["path"],
                        "duration": it["dur"],
                        "reference": it["ref"],
                        "key": it["key"],
                    }
                )
                + "\n"
            )


def _run_speech_to_text_eval(args, shard_manifest: str, preds_path: str) -> None:
    """Decode the shard manifest with NeMo's stock examples/asr/speech_to_text_eval.py.

    Run as a subprocess (not runpy) so Hydra's global state / CWD change / logging
    re-init stay isolated from this driver. The launcher pins the GPU with
    CUDA_VISIBLE_DEVICES, so cuda=0 == this shard's single visible device. amp +
    text_processing flags mirror the validated desktop NeMo-native run exactly (the
    leaderboard rescore re-normalizes anyway; these only affect NeMo's own printout)."""
    cmd = [
        sys.executable,
        "speech_to_text_eval.py",
        f"model_path={args.nemo_model}",
        f"dataset_manifest={shard_manifest}",
        f"output_filename={preds_path}",
        "gt_text_attr_name=reference",
        "cuda=0",
        f"batch_size={args.batch_size}",
        "amp=True",
        "use_cer=False",
        "text_processing.do_lowercase=true",
        "text_processing.rm_punctuation=true",
    ]
    _log(f"shard {args.shard_index}: decoding via speech_to_text_eval.py -> {preds_path}")
    # cwd=examples/asr so the script's `import transcribe_speech` resolves (it sits on
    # sys.path[0]); PYTHONPATH (repo root -> `import nemo`) and CUDA_VISIBLE_DEVICES are
    # inherited from the launcher's environment.
    subprocess.run(cmd, cwd=_ASR_EXAMPLES_DIR, check=True)


def _to_generations(preds_path: str, gen_path: str) -> int:
    """Rewrite speech_to_text_eval's output manifest as the shared generations JSONL.

    ``key`` + ``reference`` are carried through from the shard manifest (write_transcription
    copies every input field), so grouping/scoring is byte-identical to the transcribe
    backend; ``pred_text`` becomes the hypothesis (empty if the decode dropped it)."""
    n = 0
    with open(preds_path, encoding="utf-8") as fin, open(gen_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fout.write(
                json.dumps(
                    {
                        "key": rec["key"],
                        "reference": rec.get("reference", ""),
                        "hypothesis": rec.get("pred_text", "") or "",
                    }
                )
                + "\n"
            )
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nemo_model", required=True, help="Path to the .nemo ASR model.")
    p.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated 'name:split' list (default = full leaderboard suite).",
    )
    p.add_argument(
        "--cache_dir", required=True, help="Pre-staged cache root (<dataset>/<split>/_cache_manifest.jsonl)."
    )
    # --- pooled-shard mode: SAME semantics/seed as parakeet_leaderboard_eval.py ---
    p.add_argument("--num_shards", type=int, default=1, help="If >1, evaluate only this GPU's 1/num_shards slice.")
    p.add_argument("--shard_index", type=int, default=0, help="This shard's index in [0, num_shards).")
    p.add_argument(
        "--shuffle_seed", type=int, default=1234, help="Seed for the global shuffle (stable across shards)."
    )
    p.add_argument("--batch_size", type=int, default=32, help="speech_to_text_eval batch size.")
    p.add_argument("--max_eval_samples", type=int, default=0, help="Cap samples per dataset (0 = all).")
    p.add_argument("--output_dir", required=True, help="Where to write shard generations JSONL (+ temp manifests).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.isfile(args.nemo_model):
        _log(f"ERROR: .nemo model not found: {args.nemo_model}")
        return 1
    if not (args.num_shards and args.num_shards > 1):
        args.num_shards, args.shard_index = 1, 0

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

    os.makedirs(args.output_dir, exist_ok=True)
    gen_path = os.path.join(args.output_dir, f"shard{args.shard_index}_of{args.num_shards}.generations.jsonl")
    if total == 0:
        open(gen_path, "w").close()  # keep the aggregator's glob happy for empty shards
        return 0

    shard_manifest = os.path.join(args.output_dir, f"shard{args.shard_index}_of{args.num_shards}.manifest.jsonl")
    preds_path = os.path.join(args.output_dir, f"shard{args.shard_index}_of{args.num_shards}.preds.json")
    _write_shard_manifest(shard, shard_manifest)
    _run_speech_to_text_eval(args, shard_manifest, preds_path)
    n = _to_generations(preds_path, gen_path)
    _log(f"shard {args.shard_index}: wrote {n} generations -> {gen_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
