#!/usr/bin/env bash
# Submit a SLURM batch script to the "interactive" partition on a single node,
# with a decode/train job-name suffix so interactive runs don't collide with
# full-queue jobs -- without breaking checkpoint lookup for evals.
#
#   ./launch_with_interactive.sh <scriptname.sh> [script args...]
#
# This is `sbatch <scriptname.sh> [args...]` with:
#   --partition=interactive   (overrides the script's #SBATCH -p ...)
#   --nodes=1                 (overrides the script's #SBATCH -N ...)
# plus an ephemeral rewritten copy of the script:
#
#   training (non-eval_*):
#     every EXP_NAME=... line gets an `_interactive` suffix so smoke training
#     writes to a distinct results dir / wandb name (same idea as make_tmp.sh).
#
#   eval (eval_*):
#     EXP_NAME is LEFT ALONE -- it names the *trained* checkpoint tree.
#     WANDB_RUN_NAME=... and EVAL_TAG=... get `_interactive` so the *decoding*
#     job is tagged distinctly (wandb run + RESULTS_DIR), while ckpt paths
#     still resolve under the real trained EXP_NAME.
#
# The original file is never modified.
#
# Examples (run from the launch/ dir):
#   ./launch_with_interactive.sh script_baseline.sh
#   ./launch_with_interactive.sh eval_script_promptctl.sh cap punct 2 14
#
# Optional env:
#   INTERACTIVE_PARTITION=<name>   target partition (default: interactive)
#   SBATCH_EXTRA="<flags>"         extra sbatch flags, e.g. SBATCH_EXTRA="-t 00:30:00"
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <scriptname.sh> [script args...]" >&2
    exit 1
fi

PARTITION="${INTERACTIVE_PARTITION:-interactive}"
SCRIPT="$1"
shift

if [[ ! -f "$SCRIPT" ]]; then
    echo "WARNING: '$SCRIPT' is not a file in the current directory; passing to sbatch as-is (no name rewrite)." >&2
    echo "==> sbatch --partition=${PARTITION} --nodes=1 ${SBATCH_EXTRA:-} ${SCRIPT} $*"
    exec sbatch --partition="${PARTITION}" --nodes=1 ${SBATCH_EXTRA:-} "$SCRIPT" "$@"
fi

BASE="$(basename "$SCRIPT")"
# Ephemeral rewrite. Slurm reads the batch script at submit time, so the temp
# file can be removed as soon as sbatch returns.
TMP="$(mktemp --tmpdir "${BASE}.XXXXXX.sh")"
trap 'rm -f "$TMP"' EXIT

if [[ "$BASE" == eval_* ]]; then
    # Decode job identity only -- do not rewrite EXP_NAME (trained model).
    sed -E \
        -e 's/^([[:space:]]*WANDB_RUN_NAME=.*)$/\1_interactive/' \
        -e 's/^([[:space:]]*EVAL_TAG=.*)$/\1_interactive/' \
        "$SCRIPT" > "$TMP"
    REWRITE_NOTE="eval: WANDB_RUN_NAME/EVAL_TAG+=_interactive (EXP_NAME unchanged)"
else
    # Training: EXP_NAME is the run identity / results folder.
    sed -E \
        -e 's/^([[:space:]]*EXP_NAME=.*)$/\1_interactive/' \
        "$SCRIPT" > "$TMP"
    REWRITE_NOTE="train: EXP_NAME+=_interactive"
fi
chmod +x "$TMP"

# ${SBATCH_EXTRA:-} is intentionally unquoted so multiple flags word-split; "$@"
# keeps the script args exactly as given.
echo "==> sbatch --partition=${PARTITION} --nodes=1 ${SBATCH_EXTRA:-} ${SCRIPT} (${REWRITE_NOTE}) $*"
sbatch --partition="${PARTITION}" --nodes=1 ${SBATCH_EXTRA:-} "$TMP" "$@"
