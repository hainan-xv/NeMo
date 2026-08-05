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
# Uses the SAME pooled-shard, work-BALANCED backend as every other SpeechLM eval
# (launch/eval_leaderboard_slurm.sh): all utterances across all datasets are
# pooled, globally shuffled with a fixed seed, dealt into 8 duration-sorted
# shards, and one decode process per GPU handles its 1/8 slice, so wall time is
# ~= sum(all)/8 instead of the single largest dataset. This wrapper only pins the
# baseline's fixed operating point, then execs that backend on the SAME allocation.
#
# The baseline is NOT prompt-controlled: its delay (3 frames) is baked into the
# weights (a training-time alignment shift, not stated in the prompt) and its
# targets are normal caps+punct. So there is exactly ONE decode prompt -- the
# verbatim training instruction from launch/script_baseline.sh -- and the only
# operating-point knob at inference is the chunk size (must be one the model saw:
# {2, 7, 14, 28}; default 14). Sweep it by resubmitting with CHUNK_SIZE=.
#
# Usage (from the NeMo79 repo root on OCI):
#   sbatch launch/eval_script_baseline.sh
#   CHUNK_SIZE=7  sbatch launch/eval_script_baseline.sh
#   for c in 2 7 14 28; do CHUNK_SIZE=$c sbatch launch/eval_script_baseline.sh; done
#   sbatch launch/eval_script_baseline.sh --quick_run        # smoke test, 10 utts/ds
#
# Flags:
#   --quick_run[=N]   decode only the first N (default 10) utts of EACH dataset for
#                     a fast smoke test; tags RESULTS_DIR + wandb run with _quick.
#
# Optional env (forwarded to eval_leaderboard_slurm.sh):
#   CHUNK_SIZE=14 (default)     BACKEND=heh|sslm     BATCH_SIZE=...
#   PROJECT=SpeechlmRefactored (default; the training project)
#   RUN_AVERAGING=1 (default) / USE_LAST=1 / STEP=n / CKPT=path
#   DATASETS="..."   MAX_EVAL_SAMPLES=n (smoke test)   SHUFFLE_SEED=1234
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

# --- Flags (--quick_run) --------------------------------------------------
# --quick_run caps decoding to the first N (default 10) utts/dataset via the
# backend's MAX_EVAL_SAMPLES (first-N per dataset on both backends).
QUICK_RUN=0
QUICK_N=10
for _arg in "$@"; do
    case "$_arg" in
        --quick_run|--quick-run) QUICK_RUN=1 ;;
        --quick_run=*|--quick-run=*) QUICK_RUN=1; QUICK_N="${_arg#*=}" ;;
        *) echo "WARNING: ignoring unrecognized argument '${_arg}' (baseline eval takes no positional args)." >&2 ;;
    esac
done
QUICK_SUFFIX=""
if (( QUICK_RUN )); then
    export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-$QUICK_N}"
    QUICK_SUFFIX="_quick"
fi

# --- Model + run identity ---
EXP_NAME="${EXP_NAME:-granary2_script_baseline}"
# The training project (launch/script_baseline.sh: PROJECT_NAME=SpeechlmRefactored).
# eval_leaderboard_slurm.sh resolves checkpoints under results/<PROJECT>/<EXP>/<EXP>/,
# so this MUST match how the model was trained.
PROJECT="${PROJECT:-SpeechlmRefactored}"
MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"
CHUNK_SIZE="${CHUNK_SIZE:-14}"

# Baseline decode prompt -- MUST be byte-for-byte the training instruction set in
# launch/script_baseline.sh (SYSTEM_PROMPT there). The model keys behavior on this
# exact text; any drift is out-of-distribution.
SYSTEM_PROMPT="You are doing streaming speech recognition. You are given the text history so far, followed by the audio representation of the next chunk; output the words spoken in that chunk. The text history is:"

# Distinguishes this run's RESULTS_DIR (the backend also appends _chunk<CHUNK_SIZE>).
EVAL_TAG="baseline${QUICK_SUFFIX}"
# wandb run name encodes the decode config (delay is baked; targets are caps+punct),
# so the only knob to surface is the chunk size. Logged to WANDB_EVAL_PROJECT.
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXP_NAME}_chunk${CHUNK_SIZE}${QUICK_SUFFIX}}"

export SYSTEM_PROMPT MODEL_CLASS CHUNK_SIZE EXP_NAME PROJECT EVAL_TAG WANDB_RUN_NAME

echo "==> baseline SCRIPT leaderboard eval"
echo "    exp_name:      ${EXP_NAME}"
echo "    project:       ${PROJECT}"
echo "    eval_tag:      ${EVAL_TAG}"
echo "    chunk_size:    ${CHUNK_SIZE}"
(( QUICK_RUN )) && echo "    quick_run:     first ${QUICK_N} utts/dataset (MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES})"
echo "    wandb_run:     ${WANDB_RUN_NAME} (+_<launch-time> appended by backend)"
echo "    system_prompt: ${SYSTEM_PROMPT}"

# Locate eval_leaderboard_slurm.sh. Under sbatch, Slurm COPIES this script into a
# spool dir, so BASH_SOURCE is useless -- prefer SLURM_SUBMIT_DIR (cwd at submit
# time). Accept submit-from-repo-root (`sbatch launch/...`, backend at launch/)
# or submit-from-launch/ (`sbatch ./...`, backend alongside).
resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        if [[ -f "${SLURM_SUBMIT_DIR}/eval_leaderboard_slurm.sh" ]]; then
            echo "${SLURM_SUBMIT_DIR}"; return
        fi
        if [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_leaderboard_slurm.sh" ]]; then
            echo "${SLURM_SUBMIT_DIR}/launch"; return
        fi
    fi
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "${here}/eval_leaderboard_slurm.sh" ]]; then
        echo "${here}"; return
    fi
    echo "ERROR: cannot locate eval_leaderboard_slurm.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
LAUNCH_DIR="$(resolve_launch_dir)"
# Keep cwd at the submit dir so relative paths (slurm_out/) land where expected.
cd "${SLURM_SUBMIT_DIR:-$LAUNCH_DIR}"

# Already inside the sbatch allocation: run the shared pooled-shard body as a
# normal bash script (its own #SBATCH headers are ignored). $1 = EXP_NAME.
exec bash "${LAUNCH_DIR}/eval_leaderboard_slurm.sh" "${EXP_NAME}"
