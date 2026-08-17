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
"""Open-ASR-Leaderboard evaluator for the PROMPT-CONTROLLED SCRIPT model.

This is a thin wrapper around ``scripts/speechlm_leaderboard_eval.py``: it reuses
that driver's model loading, pre-staged cache reader, batched CUDA-OOM-backoff
decode, pooled-shard fan-out, and leaderboard-faithful WER unchanged. The ONLY
thing it adds is that the decode ``system_prompt`` is *built from the model's
prompt-control knobs* instead of being passed verbatim, so the eval prompt is
byte-identical to what :class:`ScriptSTTDataset` renders at training time for the
same operating point (that is what keeps the eval in-distribution).

Knobs (all configurable; defaults in brackets):

  * ``--chunk_size``  [12]   frames per chunk. Sets BOTH the decode chunk size
                             (``chunk_size_override``) AND, when the chunk clause
                             is enabled, the "...chunks of N frames." wording.
  * ``--delay``       [3]    emission delay in frames, rendered as "...delay of N
                             frames." (must be within the model's trained range).
  * ``--cap/--no-cap`` [cap] capitalization on/off  -> selects the format clause.
  * ``--punct/--no-punct`` [punct] punctuation on/off -> selects the format clause.

The prompt is assembled EXACTLY like ``ScriptSTTDataset._build_exact_prompt`` +
``_append_chunk_clause``:

    <prompt_template with {delay} and {format_clause} filled>  (stripped)
    + " " + <chunk_size_prompt_template with {chunk_size} filled>   (if enabled)

Defaults for the template / clauses mirror the code defaults
(``ScriptSTTDataset._DEFAULT_PROMPT_TEMPLATE`` / ``_DEFAULT_FORMAT_CLAUSES`` and
the standard chunk clause), which is what the committed promptctl recipes use.
IMPORTANT: some recipes differ and you MUST match your model's training render:
  * ``_promptctl_d8`` / ``_d8_shared`` / ``_d8_win14`` / ``_promptctl_recover``
    do NOT state the chunk size -> pass ``--chunk_size_prompt_template ""``.
  * ``launch/script_promptctl.sh`` uses a slightly different template wording
    ("Emit the words of each chunk ...") -> pass ``--prompt_template "..."``.
  * ``_promptctl_all`` also appends a self-correction ON/OFF clause -> pass it via
    ``--prompt_suffix "Do not go back and change words you already wrote."``.
The fully-rendered prompt is PRINTED at startup; eyeball it against your recipe's
``data.dataset.system_prompt`` (or launcher ``SYSTEM_PROMPT``) before trusting the
numbers.

Everything else (``--datasets``, ``--cache_dir``, ``--num_shards``/``--aggregate``,
``--batch_size``, ``--max_new_tokens``, ``--dtype``, ...) is identical to
``speechlm_leaderboard_eval.py`` and can be driven by the same SLURM launcher
(``launch/eval_leaderboard.sh``) by swapping the script name and adding the knobs.

Examples:
    # single dataset on GPU 3, default operating point (12 fpc, cap+punct, delay 3)
    python scripts/speechlm_promptctl_eval.py --ckpt_path model.ckpt \
        --cache_dir /lustre/.../leaderboard_cache --datasets ami_cleaned:test --device 3

    # d8-style model (no chunk clause in the prompt), lowercase no-punct, delay 5
    python scripts/speechlm_promptctl_eval.py --ckpt_path model.ckpt \
        --cache_dir /lustre/.../leaderboard_cache \
        --chunk_size 7 --delay 5 --no-cap --no-punct --chunk_size_prompt_template ""
"""
import argparse
import os
import sys

import torch

# Sibling scripts (leaderboard_wer, speechlm_leaderboard_eval) are imported by bare
# module name; make that work regardless of the caller's CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import speechlm_leaderboard_eval as base  # noqa: E402

# --- Prompt templates / clauses ---------------------------------------------
# KEEP IN SYNC with nemo/collections/speechlm2/data/script_dataset.py
# (ScriptSTTDataset._DEFAULT_PROMPT_TEMPLATE / _DEFAULT_FORMAT_CLAUSES). Copied
# here so the eval can build the prompt without importing the (heavy) model stack.
DEFAULT_PROMPT_TEMPLATE = (
    "You are doing streaming speech recognition. Given the transcript so far and "
    "the next audio chunk, output the words spoken in that chunk. Emit each chunk's "
    "words with a fixed delay of {delay} frames. {format_clause}"
)
DEFAULT_FORMAT_CLAUSES = {
    "cap_punct": "Write the text with normal capitalization and punctuation.",
    "cap_nopunct": "Write the text with normal capitalization but no punctuation.",
    "nocap_punct": "Write the text in all lowercase, keeping punctuation.",
    "nocap_nopunct": "Write the text in all lowercase with no punctuation.",
}
DEFAULT_CHUNK_CLAUSE = "Process the audio in chunks of {chunk_size} frames."


def _repr_key(cap: bool, punct: bool) -> str:
    return f"{'cap' if cap else 'nocap'}_{'punct' if punct else 'nopunct'}"


def build_promptctl_system_prompt(
    *,
    delay: int,
    cap: bool,
    punct: bool,
    chunk_size: int,
    prompt_template: str,
    format_clause: str,
    chunk_size_prompt_template: str,
    prompt_suffix: str,
) -> str:
    """Render the prompt-control system prompt for one operating point.

    Mirrors ``ScriptSTTDataset._build_exact_prompt`` (template.format(...).strip())
    followed by ``_append_chunk_clause`` (append the chunk clause) and an optional
    trailing suffix (e.g. the self-correction ON/OFF clause). Order matches the
    dataset: template(+format) -> chunk clause -> suffix.
    """
    prompt = prompt_template.format(delay=int(delay), format_clause=format_clause).strip()
    if chunk_size_prompt_template:
        clause = chunk_size_prompt_template.format(chunk_size=int(chunk_size)).strip()
        if clause:
            prompt = (prompt.rstrip() + " " + clause).strip()
    if prompt_suffix:
        prompt = (prompt.rstrip() + " " + prompt_suffix.strip()).strip()
    return prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- prompt-control knobs (the only additions over speechlm_leaderboard_eval) ---
    g = p.add_argument_group("prompt-control knobs")
    g.add_argument("--chunk_size", type=int, default=12, help="Frames per chunk (decode + prompt clause). [12]")
    g.add_argument("--delay", type=int, default=3, help="Emission delay in frames, rendered into the prompt. [3]")
    g.add_argument("--cap", action=argparse.BooleanOptionalAction, default=True, help="Capitalization on/off. [--cap]")
    g.add_argument(
        "--punct", action=argparse.BooleanOptionalAction, default=True, help="Punctuation on/off. [--punct]"
    )
    g.add_argument(
        "--prompt_template",
        type=str,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template with {delay} and {format_clause} placeholders (default = code default).",
    )
    g.add_argument(
        "--format_clause",
        type=str,
        default=None,
        help="Override the (cap, punct) format clause text (default = the built-in clause for the chosen combo).",
    )
    g.add_argument(
        "--chunk_size_prompt_template",
        type=str,
        default=DEFAULT_CHUNK_CLAUSE,
        help="Chunk-size clause with a {chunk_size} placeholder, appended to the prompt. Pass \"\" to disable "
        "(for models trained WITHOUT a chunk clause, e.g. _promptctl_d8). [\"%s\"]" % DEFAULT_CHUNK_CLAUSE,
    )
    g.add_argument(
        "--prompt_suffix",
        type=str,
        default="",
        help="Extra clause appended verbatim at the very end (e.g. the self-correction ON/OFF clause for "
        "_promptctl_all). Default empty.",
    )
    g.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="If given, use this VERBATIM and skip prompt building (escape hatch to paste an exact training prompt).",
    )

    # --- everything below mirrors speechlm_leaderboard_eval.py (same semantics) ---
    p.add_argument("--ckpt_path", help="Lightning .ckpt to evaluate (required unless --aggregate).")
    p.add_argument(
        "--model_class",
        default="nemo.collections.speechlm2.models.script_model.ScriptSTTModel",
        help="Dotted path of the model class to load.",
    )
    p.add_argument(
        "--datasets",
        default=",".join(base.DEFAULT_DATASETS),
        help="Comma-separated 'name:split' list (default = full leaderboard). Pass one for per-GPU eval.",
    )
    p.add_argument("--cache_dir", help="Pre-staged cache root (<dataset>/<split>/_cache_manifest.jsonl).")
    p.add_argument("--device", type=int, default=0, help="GPU index (cuda:N).")
    p.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="If >1, pool ALL datasets' utts and evaluate only this GPU's 1/num_shards slice (write a tagged "
        "generations JSONL; run --aggregate afterwards). Default 1 = per-dataset inline-WER mode.",
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
    p.add_argument(
        "--self_correct",
        action="store_true",
        help="SCRIPT redecode models only: emit the self-corrected LOCKED stream. Ignored by other models.",
    )
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--max_eval_samples", type=int, default=0, help="Cap samples per dataset (0 = all).")
    p.add_argument("--output_dir", type=str, default=None, help="If set, dump generations JSONL (required for shards).")
    p.add_argument(
        "--progress_interval",
        type=float,
        default=5.0,
        help="Min seconds between progress-bar refreshes in the log (tail -f friendly).",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def resolve_system_prompt(args: argparse.Namespace) -> str:
    """Verbatim ``--system_prompt`` if given, else the prompt built from the knobs."""
    if args.system_prompt is not None:
        return args.system_prompt
    clause = args.format_clause
    if clause is None:
        clause = DEFAULT_FORMAT_CLAUSES[_repr_key(args.cap, args.punct)]
    return build_promptctl_system_prompt(
        delay=args.delay,
        cap=args.cap,
        punct=args.punct,
        chunk_size=args.chunk_size,
        prompt_template=args.prompt_template,
        format_clause=clause,
        chunk_size_prompt_template=args.chunk_size_prompt_template,
        prompt_suffix=args.prompt_suffix,
    )


def main() -> int:
    args = parse_args()

    # Reduce mode needs no GPU/model/prompt: just pool shard JSONLs and score.
    if args.aggregate:
        if not args.output_dir:
            base._log("ERROR: --aggregate requires --output_dir (where shard JSONLs live).")
            return 1
        return base.aggregate_results(args)

    if not args.ckpt_path or not args.cache_dir:
        base._log("ERROR: --ckpt_path and --cache_dir are required for evaluation.")
        return 1

    # Build the decode prompt from the knobs and inject it into the shared driver's
    # args (base.evaluate_* read args.system_prompt / args.chunk_size verbatim).
    args.system_prompt = resolve_system_prompt(args)
    base._log("=" * 72)
    base._log("[promptctl-eval] operating point:")
    base._log(f"  chunk_size = {args.chunk_size} frames | delay = {args.delay} | cap = {args.cap} | punct = {args.punct}")
    base._log(f"  system_prompt: {args.system_prompt!r}")
    base._log("  ^ ensure this EXACTLY matches your model's TRAINING prompt render (see --help).")
    base._log("=" * 72)
    # A non-positive chunk size means "model default" (no override / no stated size).
    if args.chunk_size is not None and args.chunk_size <= 0:
        args.chunk_size = None

    if not torch.cuda.is_available():
        base._log("WARNING: CUDA not available; running on CPU (very slow).")
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    model = base.load_model(args.ckpt_path, args.model_class, device, dtype)

    # Pooled-shard mode: decode only this GPU's balanced slice; WER via --aggregate.
    if args.num_shards and args.num_shards > 1:
        if not args.output_dir:
            base._log("ERROR: --num_shards > 1 requires --output_dir for shard generations.")
            return 1
        return base.evaluate_shard(model, args, device)

    # Per-dataset mode (one or more full datasets in this process).
    entries = base._parse_entries(args.datasets)
    results = []
    for dataset, split in entries:
        try:
            r = base.evaluate_dataset(model, args, dataset, split, device)
            if r is not None:
                results.append(r)
        except Exception as ex:  # noqa: BLE001
            base._log(f"RESULT\t{dataset}/{split}\tERR\t0.0\t0  ({type(ex).__name__}: {ex})")

    if results:
        base._log("\n  {:<28} {:>8} {:>10}".format("Dataset", "WER(%)", "Time(s)"))
        base._log("  " + "-" * 48)
        tot = 0.0
        for r in results:
            base._log(f"  {r['key']:<28} {r['wer']:>8.2f} {r['time']:>10.1f}")
            tot += r["wer"]
        base._log("  " + "-" * 48)
        base._log(f"  {'Average':<28} {tot / len(results):>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
