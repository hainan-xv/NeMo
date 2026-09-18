#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-eval-parakeet-tdt
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# Leaderboard eval of parakeet-tdt-0.6b-v2 -- the DONOR whose encoder the
# full-context CHAT arm was initialised from.
#
#   sbatch launch/dfw_eval_parakeet_tdt.sh
#
# WHY THIS NUMBER MATTERS. The full-context arm scores 5.08 against the streaming
# arms' 5.42-5.47, and that gain confounds TWO changes: an unconstrained
# receptive field, AND a stronger donor encoder. Scoring the donor alone
# separates them -- if parakeet is already at ~5, most of the gain is the donor;
# if it is well above, the chunked-emission objective and full context are doing
# the work.
#
# It is NOT a like-for-like system: parakeet is a plain offline TDT model over its
# own 1024-piece vocabulary, with no chunked emission at all. It is a reference
# point for the ENCODER, not a competitor on the streaming task.
#
# FULL_CONTEXT=1 because parakeet is non-causal (att_context_size_all
# [[-1,-1]]); without it the shared backend forces [left, 13] to decode at chunk
# 14 and the model correctly refuses. Its joint has no chunk_size attribute, so
# the joint-chunking step is skipped automatically.
#
# Decoding goes through model.transcribe(), i.e. the model's OWN decoding config
# -- which is what makes TDT's duration outputs work here without special casing.
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-${DFW}/hainanx}"
export CODE_DIR="${CODE_DIR:-${OUTPUT_PREFIX}/NeMo_SCRIPT_cc}"
export CONTAINER="${CONTAINER:-${DFW}/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
export CACHE_DIR="${CACHE_DIR:-${OUTPUT_PREFIX}/leaderboard_cache}"
export H_DIR="${H_DIR:-${DFW}/users/heh}"
export PROJECT="${PROJECT:-SpeechlmDFW}"
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-${DFW}:${DFW}}"

export MODEL_PATH="${OUTPUT_PREFIX}/pretrained_models/nvidia/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo"
export EXP_NAME="parakeet_tdt_0.6b_v2"
export MODE=offline
export CHUNK_SIZE=14
export FULL_CONTEXT=1
export EVAL_TAG="donor"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: parakeet donor not found: ${MODEL_PATH}" >&2
    exit 1
fi
echo "==> parakeet-tdt-0.6b-v2 leaderboard eval (DONOR reference, offline TDT)"
echo "    model: ${MODEL_PATH}"

resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_nemotron.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_nemotron.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_nemotron.sh" ]] && { echo "${here}"; return; }
    [[ -f "${CODE_DIR}/launch/eval_nemotron.sh" ]] && { echo "${CODE_DIR}/launch"; return; }
    echo "ERROR: cannot locate eval_nemotron.sh" >&2
    exit 1
}
exec bash "$(resolve_launch_dir)/eval_nemotron.sh"
