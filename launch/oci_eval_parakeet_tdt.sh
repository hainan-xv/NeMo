#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:oci-eval-parakeet-tdt
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Reference-model eval: nvidia/parakeet-tdt-0.6b-v2
#
#   sbatch launch/oci_eval_parakeet_tdt.sh
#
# The offline TDT reference. Full-context model, so pad 0 and offline mode --
# it has no streaming look-ahead to respect.
#
# Batch partition, not interactive: the interactive QOS caps concurrent jobs per
# user, which serialised an earlier five-way eval fan-out instead of running it
# in parallel.
#
# Same seven datasets and same aggregate scorer as the OCI arm evals, so these
# are directly comparable with them. NOT comparable with DFW's macro-7, which
# scores the chunked earnings22 and uses the official kaldialign scorer.
# ============================================================================
set -euo pipefail
mkdir -p slurm_out

OCI=/lustre/fsw/portfolios/nemotron/users/hainanx
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-${OCI}}"
export CODE_DIR="${CODE_DIR:-${OCI}/NeMo_SCRIPT_cc}"
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
export CACHE_DIR="${CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache}"
export H_DIR="${H_DIR:-/lustre/fsw/portfolios/llmservice/users/heh}"
export PROJECT="${PROJECT:-SpeechlmOCI}"
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-/lustre/fsw/portfolios/llmservice:/lustre/fsw/portfolios/llmservice}"

export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/heh/pretrained_models/nemo_asr/parakeet-tdt-0.6b-v2.nemo}"
export EXP_NAME="${EXP_NAME:-parakeet_tdt_0.6b_v2_baseline}"
export MODE="${MODE:-offline}"
export CHUNK_SIZE="${CHUNK_SIZE:-14}"
export PAD_EXTRA_SECONDS="${PAD_EXTRA_SECONDS:-0}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: model not found: ${MODEL_PATH}" >&2
    exit 1
fi

find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_nemotron.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_nemotron.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_nemotron.sh" ]] && { echo "${here}"; return; }
    local repo="${OCI}/NeMo_SCRIPT_cc"
    [[ -f "${repo}/launch/eval_nemotron.sh" ]] && { echo "${repo}/launch"; return; }
    echo "ERROR: cannot locate eval_nemotron.sh" >&2; exit 1
}
exec bash "$(find_launch_dir)/eval_nemotron.sh"
