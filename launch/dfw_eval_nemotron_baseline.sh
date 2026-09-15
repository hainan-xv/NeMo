#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-nemotron-base-eval
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
# pool0-01815 carries an old NVIDIA driver (12020) and fails torch's CUDA init
# outright; it killed job 18686485 in 103s. pool0-00407 is the reference recipe's
# known-bad node.
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# Leaderboard eval of the DONOR nemotron streaming RNN-T, on the CW DFW cluster.
#
#   sbatch launch/dfw_eval_nemotron_baseline.sh      <- no arguments
#
# WHY THIS EXISTS. Every arm in this project is initialised from this model and
# measured against it, but the nemotron reference macro we quote (5.81) was
# produced on OCI. That leaves the one number all the others are judged against
# as the only number in the table from different hardware and a different
# filesystem. This reproduces it on DFW, through the SAME backend
# (eval_nemotron.sh) that produced every DFW arm's number, so the whole
# comparison finally rests on one pipeline on one cluster.
#
# WHICH MODEL. nemotron-speech-streaming-en-0.6b -- the .nemo the arms actually
# initialise from (it is PRETRAINED_ASR in dfw_script_*.sh and the donor whose
# SentencePiece the 1k-vocab CHAT arm extracts). NOTE: there is no "3.5" RNN-T
# ASR checkpoint anywhere on this cluster. The whole project tree was searched;
# the only nemotron .nemo files present are this 0.6b streaming RNN-T, a
# diarization preview and an encoder-only artifact, and the sole "3.5" directory
# -- NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 -- is a text MoE LLM, not an ASR
# model. If a different RNN-T is meant, set MODEL_PATH to it; everything else
# here is model-agnostic.
#
# chunk_size 14 matches the arms' emission grid. The donor's .nemo already
# defaults to att_context_size [70, 13], so this changes nothing for it and
# simply makes the setting explicit and identical across the table.
#
# ENV
#   MODEL_PATH   a different .nemo to evaluate (default: the 0.6b donor)
#   CHUNK_SIZE   decode chunk size in encoder frames (default 14)
#   MODE         offline | streaming (default offline, as the arms were scored)
# ============================================================================
set -uo pipefail

mkdir -p slurm_out

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-${DFW}/hainanx}"
export CODE_DIR="${CODE_DIR:-${OUTPUT_PREFIX}/NeMo_SCRIPT_cc}"
export CONTAINER="${CONTAINER:-${DFW}/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
export CACHE_DIR="${CACHE_DIR:-${OUTPUT_PREFIX}/leaderboard_cache}"
# Mounted so the donor .nemo -- which lives under users/heh on DFW, not on an
# fs12-style second root as on OCI -- is visible inside the container.
export H_DIR="${H_DIR:-${DFW}/users/heh}"
export PROJECT="${PROJECT:-SpeechlmDFW}"

export MODEL_PATH="${MODEL_PATH:-${DFW}/users/heh/pretrained_models/huggingface/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo}"
export EXP_NAME="${EXP_NAME:-dfw_nemotron_streaming_0.6b_baseline}"
export MODE="${MODE:-offline}"
export CHUNK_SIZE="${CHUNK_SIZE:-14}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: donor model not found: ${MODEL_PATH}" >&2
    echo "       Override with MODEL_PATH=/path/to/model.nemo" >&2
    exit 1
fi
if [[ ! -d "${CACHE_DIR}" ]] || [[ -z "$(ls -A "${CACHE_DIR}" 2>/dev/null)" ]]; then
    echo "ERROR: leaderboard cache is empty or missing at ${CACHE_DIR}" >&2
    echo "       Stage it first:  sbatch launch/dfw_stage_leaderboard_cache.sh" >&2
    exit 1
fi

echo "==> DFW nemotron donor leaderboard eval"
echo "    model:      ${MODEL_PATH}"
echo "    mode:       ${MODE}   chunk_size: ${CHUNK_SIZE}"
echo "    cache:      ${CACHE_DIR}"

# Under sbatch $0 is a spool copy, and on a REQUEUE SLURM_SUBMIT_DIR comes back
# as the scratch root rather than the submit dir -- which silently broke the CHAT
# arms for three hours. Hence the absolute fallback.
resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_nemotron.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_nemotron.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_nemotron.sh" ]] && { echo "${here}"; return; }
    [[ -f "${CODE_DIR}/launch/eval_nemotron.sh" ]] && { echo "${CODE_DIR}/launch"; return; }
    echo "ERROR: cannot locate eval_nemotron.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}, CODE_DIR=${CODE_DIR})" >&2
    exit 1
}
exec bash "$(resolve_launch_dir)/eval_nemotron.sh"
