#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-script-lb-eval
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
# pool0-01815 carries an old NVIDIA driver (12020) and fails torch's CUDA init
# outright; it killed job 18686485 in 103s. pool0-00407 is the reference recipe's
# known-bad node.
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# Open-ASR-Leaderboard eval for a SCRIPT model on the CW DFW cluster.
#
#   sbatch launch/dfw_eval_script.sh dfw_granary2_script_baseline
#   sbatch launch/dfw_eval_script.sh dfw_granary2_script_banded1 14
#
# The DFW port of eval_script.sh. It sets DFW paths and then execs the SAME
# shared backend (eval_leaderboard.sh) the OCI evals use, so a DFW number is
# produced by exactly the same code path -- averaging, sharding, normalisation
# and scoring -- as every number already collected on OCI. Only the paths differ,
# which is what keeps the two clusters' results comparable.
#
# The only change the backend needed was making H_DIR overridable: it defaulted
# to an OCI-only directory, and container mounts fail outright on a path that
# does not exist.
#
# WHY interactive: DFW's batch partition is 1850 nodes and rarely contended, but
# eval is single-node and short, and interactive has the same 4h limit here. The
# one-job-per-user rule that made this awkward on OCI does not appear to apply.
#
# POSITIONAL ARGS
#   $1  EXP_NAME    experiment to evaluate  (REQUIRED)
#   $2  CHUNK_SIZE  decode chunk size in encoder frames (default 14)
#
# NOTE ON CHUNK SIZE: use one the model actually trained on. The banded arms are
# pinned to 14; the baseline draws from [2,7,10,14], so it can be evaluated at
# any of those -- but compare arms at the SAME size or the number means nothing.
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-${DFW}/hainanx}"
export CODE_DIR="${CODE_DIR:-${OUTPUT_PREFIX}/NeMo_SCRIPT_cc}"
export CONTAINER="${CONTAINER:-${DFW}/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
export CACHE_DIR="${CACHE_DIR:-${OUTPUT_PREFIX}/leaderboard_cache}"
# DFW's stand-in for the OCI llmservice/users/heh tree: it holds the pretrained
# models and data configs, and must be mounted for a restored .nemo to resolve
# anything it recorded at train time.
export H_DIR="${H_DIR:-${DFW}/users/heh}"
# The DFW-era wandb project, matching the training runs.
export PROJECT="${PROJECT:-SpeechlmDFW}"

EXP_NAME="${1:-${EXP_NAME:-}}"
if [[ -z "$EXP_NAME" ]]; then
    echo "ERROR: no experiment name given." >&2
    echo "usage: sbatch launch/dfw_eval_script.sh <exp_name> [chunk_size]" >&2
    echo "" >&2
    echo "Experiments under ${OUTPUT_PREFIX}/results/${PROJECT}:" >&2
    ls -1 "${OUTPUT_PREFIX}/results/${PROJECT}" 2>/dev/null | sed 's/^/  /' >&2 || echo "  (none found)" >&2
    exit 1
fi
export EXP_NAME
export CHUNK_SIZE="${2:-${CHUNK_SIZE:-14}}"
export MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"

# Decode prompt -- MUST be byte-for-byte the training instruction. Drift here is
# silently out of distribution.
export SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are doing streaming speech recognition. Given the transcript so far and the representation of the next audio chunk, output the words spoken in that chunk.}"
export EVAL_TAG="${EVAL_TAG:-${EXP_NAME}}"

if [[ ! -d "${CACHE_DIR}" ]] || [[ -z "$(ls -A "${CACHE_DIR}" 2>/dev/null)" ]]; then
    echo "ERROR: leaderboard cache is empty or missing at ${CACHE_DIR}" >&2
    echo "       Stage it first:  sbatch launch/dfw_stage_leaderboard_cache.sh" >&2
    exit 1
fi

CKPT_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/${EXP_NAME}/checkpoints"
echo "==> DFW SCRIPT leaderboard eval"
echo "    exp:        ${EXP_NAME}"
echo "    chunk_size: ${CHUNK_SIZE}"
echo "    ckpt dir:   ${CKPT_DIR}"
echo "    cache:      ${CACHE_DIR}"
[[ -d "$CKPT_DIR" ]] || echo "WARNING: no checkpoints directory at ${CKPT_DIR}" >&2

# Under sbatch $0 is a copy in Slurm's spool directory, and on a REQUEUE
# SLURM_SUBMIT_DIR comes back as the scratch root rather than the submit dir --
# which silently broke the CHAT arms for three hours. Hence the absolute fallback.
resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_leaderboard.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_leaderboard.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_leaderboard.sh" ]] && { echo "${here}"; return; }
    [[ -f "${CODE_DIR}/launch/eval_leaderboard.sh" ]] && { echo "${CODE_DIR}/launch"; return; }
    echo "ERROR: cannot locate eval_leaderboard.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}, CODE_DIR=${CODE_DIR})" >&2
    exit 1
}
exec bash "$(resolve_launch_dir)/eval_leaderboard.sh"
