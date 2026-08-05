#!/usr/bin/env bash
# Submit a SLURM batch script to the "interactive" partition instead of its own
# default, with everything else left unchanged.
#
#   ./launch_with_interactive.sh <scriptname.sh> [script args...]
#
# This is exactly `sbatch <scriptname.sh> [args...]` with an added
# `--partition=interactive`. A command-line --partition OVERRIDES the script's own
# `#SBATCH -p ...` directive (ours: batch_block1,batch_block3,batch_block4), so the
# target file is never modified -- only the partition it lands on changes.
#
# Examples:
#   ./launch_with_interactive.sh launch/script_baseline.sh
#   ./launch_with_interactive.sh launch/eval_script_promptctl.sh cap punct 2 14
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

if [[ ! -f "$1" ]]; then
    echo "WARNING: '$1' is not a file in the current directory; passing to sbatch as-is." >&2
fi

# ${SBATCH_EXTRA:-} is intentionally unquoted so multiple flags word-split; "$@"
# keeps the script path + its args exactly as given.
echo "==> sbatch --partition=${PARTITION} ${SBATCH_EXTRA:-} $*"
exec sbatch --partition="${PARTITION}" ${SBATCH_EXTRA:-} "$@"
