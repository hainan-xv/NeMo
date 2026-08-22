#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:script-lb-baseline
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
# Open-ASR-Leaderboard eval for the BASELINE SCRIPT model
# (granary2_script_baseline, trained by launch/script_baseline.sh).
#
# A thin positional wrapper around launch/eval_leaderboard.sh: it pins the decode
# prompt and chunk size to the training operating point, then hands off. Because
# it `exec`s inside the same allocation, the backend's #SBATCH headers are inert
# and there is one job / one node / 8 GPUs.
#
# POSITIONAL ARGS
#   $1  EXP_NAME    which run to evaluate  (default: granary2_script_baseline)
#   $2  CHUNK_SIZE  decode chunk size in frames (default: 14)
#
# USAGE (from the repo root on the OCI login node)
#   sbatch launch/eval_script_baseline.sh
#   sbatch launch/eval_script_baseline.sh granary2_script_baseline
#   sbatch launch/eval_script_baseline.sh granary2_script_baseline 28
#   for c in 2 7 14 28; do sbatch launch/eval_script_baseline.sh my_model $c; done
#
# Or from your laptop, syncing first:
#   ./oci_launch.sh launch/eval_script_baseline.sh
#   ./oci_launch_interactive.sh MAX_EVAL_SAMPLES=32 launch/eval_script_baseline.sh
#
# CHUNK SIZE is the only inference-time latency knob for this model: the 3-frame
# emission delay it was trained with is baked into the weights. Valid values are
# the ones the model actually saw in training -- {2, 4, 7, 10, 14, 28}. Anything
# else is out of distribution.
#
# All other knobs (RUN_AVERAGING, CKPT/STEP/USE_LAST, DATASETS, BATCH_SIZE,
# MAX_NEW_TOKENS, MAX_EVAL_SAMPLES, FORCE_WORD_START, wandb, ...) are env vars
# handled by eval_leaderboard.sh -- see its header.
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

EXP_NAME="${1:-${EXP_NAME:-granary2_script_baseline}}"
CHUNK_SIZE="${2:-${CHUNK_SIZE:-14}}"
PROJECT="${PROJECT:-SpeechlmScriptCC}"
MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"

# Decode prompt -- MUST be byte-for-byte the training instruction from
# launch/script_baseline.sh (SYSTEM_PROMPT there) and from the recipe's
# data.dataset.system_prompt. The model keys its behaviour on this exact text;
# any drift is out-of-distribution and silently costs WER.
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are doing streaming speech recognition. Given the transcript so far and the representation of the next audio chunk, output the words spoken in that chunk.}"

EVAL_TAG="${EVAL_TAG:-baseline}"

export EXP_NAME PROJECT MODEL_CLASS SYSTEM_PROMPT CHUNK_SIZE EVAL_TAG

echo "==> baseline SCRIPT leaderboard eval"
echo "    exp:        ${EXP_NAME}"
echo "    chunk_size: ${CHUNK_SIZE}"
echo "    prompt:     ${SYSTEM_PROMPT}"

# Locate the backend. Under sbatch, Slurm COPIES this script into a spool dir, so
# BASH_SOURCE is unreliable -- prefer SLURM_SUBMIT_DIR (cwd at submit time).
resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_leaderboard.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_leaderboard.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_leaderboard.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate eval_leaderboard.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
LAUNCH_DIR="$(resolve_launch_dir)"
cd "${SLURM_SUBMIT_DIR:-$LAUNCH_DIR}"

# Already inside the allocation: run the backend as a plain script.
exec bash "${LAUNCH_DIR}/eval_leaderboard.sh" "${EXP_NAME}"
