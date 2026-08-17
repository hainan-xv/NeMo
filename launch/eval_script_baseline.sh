#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:sslm-lb-script-baseline
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1   # ONE task; fans 8 python procs across the 8 GPUs
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Open-ASR-Leaderboard eval for the BASELINE SCRIPT model
# (granary2_script_baseline, trained by launch/script_baseline.sh).
#
# Thin wrapper that pins the baseline's fixed operating point -- the verbatim
# training instruction (SYSTEM_PROMPT), the SCRIPT model class, and a decode
# chunk size (default 14) -- then execs the shared pooled-shard backend
# launch/eval_leaderboard.sh on the SAME allocation. The backend pools every
# utterance across all datasets, deals an even 1/8 slice to each GPU, decodes
# with the model's own generate() (same path as val_wer), and reduces to
# per-dataset + macro WER.
#
# The baseline is NOT prompt-controlled: its delay (3 frames) is baked into the
# weights (a training-time alignment shift, not stated in the prompt) and its
# targets are normal caps+punct. So there is exactly ONE decode prompt -- the
# verbatim training instruction from launch/script_baseline.sh -- and the only
# operating-point knob at inference is the chunk size (must be one the model saw:
# {2, 7, 14, 28}; default 14). Sweep it by resubmitting with CHUNK_SIZE=.
#
# Usage (from the clean repo root on OCI):
#   sbatch launch/eval_script_baseline.sh                       # default model, averaged ckpt
#   sbatch launch/eval_script_baseline.sh granary2_script_myrun # $1 = model/exp name
#   RUN_AVERAGING=0 sbatch launch/eval_script_baseline.sh <exp> # eval the single best ckpt
#   FORCE_AVERAGE=1 sbatch launch/eval_script_baseline.sh <exp> # recompute the averaged ckpt
#   USE_LAST=1      sbatch launch/eval_script_baseline.sh <exp> # eval the rolling -last.ckpt
#   CHUNK_SIZE=7    sbatch launch/eval_script_baseline.sh <exp>
#   for c in 2 7 14 28; do CHUNK_SIZE=$c sbatch launch/eval_script_baseline.sh; done
#   MAX_EVAL_SAMPLES=10 sbatch launch/eval_script_baseline.sh   # smoke test, 10 utts/ds
#
# NOTE: $1 only swaps WHICH checkpoint folder is evaluated; the baseline SYSTEM_PROMPT
# / MODEL_CLASS / CHUNK_SIZE below stay fixed. Only point $1 at models trained with
# the SAME training instruction (SCRIPT baseline variants) -- a different prompt is
# out-of-distribution. To eval a differently-prompted model, set SYSTEM_PROMPT= too.
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

# --- Model + run identity (MUST match launch/script_baseline.sh) ---
# $1 selects the experiment/model name (the results/<PROJECT>/<EXP_NAME>/ folder
# whose checkpoints we eval); falls back to $EXP_NAME then the baseline default.
EXP_NAME="${1:-${EXP_NAME:-granary2_script_baseline}}"
PROJECT="${PROJECT:-SpeechlmScriptClean}"
MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"
CHUNK_SIZE="${CHUNK_SIZE:-14}"

# Baseline decode prompt -- MUST be byte-for-byte the training instruction set in
# launch/script_baseline.sh (SYSTEM_PROMPT there). The model keys behavior on this
# exact text; any drift is out-of-distribution.
SYSTEM_PROMPT="You are doing streaming speech recognition. You are given the text history so far, followed by the audio representation of the next chunk; output the words spoken in that chunk. The text history is:"

EVAL_TAG="${EVAL_TAG:-baseline}"
# USE_STATE_MACHINE (backend knob) is forwarded. The SCRIPT streaming state
# machine is ON by default (see eval_leaderboard.sh); set USE_STATE_MACHINE=0 to
# fall back to the offline-encode decode.
export SYSTEM_PROMPT MODEL_CLASS CHUNK_SIZE EXP_NAME PROJECT EVAL_TAG USE_STATE_MACHINE

echo "==> baseline SCRIPT leaderboard eval"
echo "    exp_name:      ${EXP_NAME}"
echo "    project:       ${PROJECT}"
echo "    chunk_size:    ${CHUNK_SIZE}"
echo "    system_prompt: ${SYSTEM_PROMPT}"

# Locate eval_leaderboard.sh. Under sbatch, Slurm COPIES this script into a spool
# dir, so BASH_SOURCE is unreliable -- prefer SLURM_SUBMIT_DIR (cwd at submit time).
# Accept submit-from-repo-root (`sbatch launch/...`, backend at launch/) or
# submit-from-launch/ (`sbatch ./...`, backend alongside).
resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        if [[ -f "${SLURM_SUBMIT_DIR}/eval_leaderboard.sh" ]]; then
            echo "${SLURM_SUBMIT_DIR}"; return
        fi
        if [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_leaderboard.sh" ]]; then
            echo "${SLURM_SUBMIT_DIR}/launch"; return
        fi
    fi
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "${here}/eval_leaderboard.sh" ]]; then
        echo "${here}"; return
    fi
    echo "ERROR: cannot locate eval_leaderboard.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
LAUNCH_DIR="$(resolve_launch_dir)"
# Keep cwd at the submit dir so relative paths (slurm_out/) land where expected.
cd "${SLURM_SUBMIT_DIR:-$LAUNCH_DIR}"

# Already inside the sbatch allocation: run the shared pooled-shard body as a
# normal bash script (its own #SBATCH headers are ignored). $1 = EXP_NAME.
exec bash "${LAUNCH_DIR}/eval_leaderboard.sh" "${EXP_NAME}"
