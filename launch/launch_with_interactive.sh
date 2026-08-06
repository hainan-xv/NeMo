#!/usr/bin/env bash
# Submit a SLURM batch script to the "interactive" partition on a single node,
# with EXP_NAME suffixed so interactive smoke runs don't collide with full jobs.
#
#   ./launch_with_interactive.sh <scriptname.sh> [script args...]
#
# This is `sbatch <scriptname.sh> [args...]` with:
#   --partition=interactive   (overrides the script's #SBATCH -p ...)
#   --nodes=1                 (overrides the script's #SBATCH -N ...)
# plus a rewritten copy of the script where every EXP_NAME=... assignment gets
# an `_interactive` suffix (same idea as make_tmp.sh's `_tmp` suffix). The
# original file is never modified.
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
    echo "WARNING: '$SCRIPT' is not a file in the current directory; passing to sbatch as-is (no EXP_NAME rewrite)." >&2
    echo "==> sbatch --partition=${PARTITION} --nodes=1 ${SBATCH_EXTRA:-} ${SCRIPT} $*"
    exec sbatch --partition="${PARTITION}" --nodes=1 ${SBATCH_EXTRA:-} "$SCRIPT" "$@"
fi

# Ephemeral rewrite: append _interactive to every EXP_NAME=... line so results
# land under a distinct folder / wandb name. Slurm reads the batch script at
# submit time, so the temp file can be removed as soon as sbatch returns.
TMP="$(mktemp --tmpdir "${SCRIPT##*/}.XXXXXX.sh")"
trap 'rm -f "$TMP"' EXIT
sed -E -e 's/^([[:space:]]*EXP_NAME=.*)$/\1_interactive/' "$SCRIPT" > "$TMP"
chmod +x "$TMP"

# ${SBATCH_EXTRA:-} is intentionally unquoted so multiple flags word-split; "$@"
# keeps the script args exactly as given.
echo "==> sbatch --partition=${PARTITION} --nodes=1 ${SBATCH_EXTRA:-} ${SCRIPT} (EXP_NAME+=_interactive) $*"
sbatch --partition="${PARTITION}" --nodes=1 ${SBATCH_EXTRA:-} "$TMP" "$@"
