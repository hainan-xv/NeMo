#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:eval-script-multi-cs14
#SBATCH -p interactive
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
# Leaderboard eval: granary2_script_banded1_both_multilookahead @ chunk 14
#
#   sbatch launch/oci_eval_script_multilookahead_cs14.sh
#
# Multi-look-ahead arm decoded at chunk 14 (~1.12 s look-ahead).
# Directly comparable with oci_eval_script_banded1_both.sh, which is the
# same recipe trained at chunk 14 only.
# ============================================================================
set -euo pipefail
mkdir -p slurm_out

OCI=/lustre/fsw/portfolios/nemotron/users/hainanx
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-${OCI}}"
export CODE_DIR="${CODE_DIR:-${OCI}/NeMo_SCRIPT_cc}"
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
# The staged leaderboard cache (43 G). Compute nodes run HF_HUB_OFFLINE=1 and
# download nothing, so this must exist or the eval dies at startup.
export CACHE_DIR="${CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache}"
export H_DIR="${H_DIR:-/lustre/fsw/portfolios/llmservice/users/heh}"
# Both the OCI CHAT and SCRIPT arms of this era report into SpeechlmOCI.
export PROJECT="${PROJECT:-SpeechlmOCI}"
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-/lustre/fsw/portfolios/llmservice:/lustre/fsw/portfolios/llmservice}"

if [[ ! -d "${CACHE_DIR}" ]]; then
    echo "ERROR: leaderboard cache missing at ${CACHE_DIR}" >&2
    exit 1
fi

find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_script.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_script.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_script.sh" ]] && { echo "${here}"; return; }
    local repo="${OCI}/NeMo_SCRIPT_cc"
    [[ -f "${repo}/launch/eval_script.sh" ]] && { echo "${repo}/launch"; return; }
    echo "ERROR: cannot locate eval_script.sh" >&2
    exit 1
}

export EVAL_TAG="${EVAL_TAG:-granary2_script_banded1_both_multilookahead_cs14}"
# MATCH DFW'S MACRO-7. The backends default to plain earnings22:test; DFW's
# macro-7 scores the ArtificialAnalysis CHUNKED variant and excludes the plain
# one. Same seven slots otherwise. Stage it first:
#   sbatch launch/oci_stage_earnings22_chunked.sh
export DATASETS="${DATASETS:-librispeech:test.clean librispeech:test.other ami_cleaned:test earnings22_cleaned_aa_chunked:test gigaspeech_cleaned:test spgispeech:test voxpopuli_cleaned_aa:test}"

exec bash "$(find_launch_dir)/eval_script.sh" granary2_script_banded1_both_multilookahead 14
