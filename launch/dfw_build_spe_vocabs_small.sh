#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-build-spe-vocabs-small
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 04:00:00
#SBATCH --time-min 01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# Build the SMALL CHAT SentencePiece vocabularies: 1k, 2k, 4k.
#
#   sbatch launch/dfw_build_spe_vocabs_small.sh      <- no arguments
#
# The low end of the sweep that dfw_build_spe_vocabs.sh covers at 8k/16k/32k.
# It exists as its own script rather than as VOCABS="1024 2048 4096" on that
# one so a bare sbatch is enough and the sizes cannot be lost to a forgotten
# environment variable.
#
# WHY THE LOW END IS THE INTERESTING END. Measured so far, at max_symbols=15
# on the 7-dataset macro:
#
#     spe8k   4.96      spe16k  5.01      spe32k  5.05
#
# Smaller has won every eval, three in a row, and 8k is the smallest size built
# -- so the sweep is currently bounded by what exists, not by where the optimum
# is. 1k also happens to be parakeet-tdt-0.6b-v2's own vocabulary size, which
# makes it the one point in the sweep where OUR vocabulary and the DONOR's are
# the same size.
#
# Note the direction of the cost, which is the opposite of the usual worry: a
# SMALLER vocabulary means MORE tokens per word, so more inner decoder
# iterations per chunk. That is exactly the axis max_symbols probes -- raising
# the cap 10->15 was worth ~0.09 to the SPE arms and 0.00 to the Qwen ones --
# so a 1k arm should be evaluated at a generous max_symbols from the start.
#
# THE CORPUS IS SHARED AND REUSED, and that is the point: these sizes are cut
# from the SAME corpus.txt as 8k/16k/32k, so the sweep varies vocabulary size
# and nothing else. The builder skips corpus extraction when the file exists.
# ============================================================================
set -uo pipefail

export VOCABS="${VOCABS:-1024 2048 4096}"

# Under sbatch $0 is a spool copy, so dirname "$0" does not find siblings; and
# on a requeue SLURM_SUBMIT_DIR points at the scratch root rather than the
# submit directory, so the absolute fallback is load-bearing, not belt-and-braces.
find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/dfw_build_spe_vocabs.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/dfw_build_spe_vocabs.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/dfw_build_spe_vocabs.sh" ]] && { echo "${here}"; return; }
    local repo="${DFW_CODE_DIR:-/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/hainanx/NeMo_SCRIPT_cc}"
    [[ -f "${repo}/launch/dfw_build_spe_vocabs.sh" ]] && { echo "${repo}/launch"; return; }
    echo "ERROR: cannot locate dfw_build_spe_vocabs.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
exec bash "$(find_launch_dir)/dfw_build_spe_vocabs.sh" "$@"
