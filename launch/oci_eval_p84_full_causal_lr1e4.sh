#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:oci-eval-p84-full-causal-lr1e4
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
# ---------------------------------------------------------------------------
# Chunk sweep for granary2_script_promptctl84_causal on OCI, all 7 datasets.
#
# Uses OUR driver, not the OASR harness: run_eval.py has no prompt_control
# support at all (no references to prompt_control / num_delay_frames /
# capitalization / punctuation). These checkpoints do not record the flag, so
# that path would skip render_control_prompt and decode on a bare instruction
# the model never trained on -- healthy-looking and meaningless.
#
# Data: the leaderboard cache staged on OCI lives in a DIFFERENT portfolio
# (llmservice, not nemotron). Verified complete: ami_cleaned 7805,
# gigaspeech_cleaned 18768, librispeech 2620/2939, spgispeech 39341.
#
# One allocation for all five chunk sizes: the first eval in a job pays
# container start plus the checkpoint and cache read cold, measured at 1503s
# first vs 160s second on identical work.
# ---------------------------------------------------------------------------
set -euo pipefail
resolve_launch_dir() {
    [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/launch/eval_leaderboard.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    [[ -f "${CODE_DIR:-}/launch/eval_leaderboard.sh" ]] && { echo "${CODE_DIR}/launch"; return; }
    echo "ERROR: cannot locate eval_leaderboard.sh" >&2; exit 1
}
MY=/lustre/fsw/portfolios/nemotron/users/hainanx
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-$MY}"
export CODE_DIR="${CODE_DIR:-$MY/NeMo_SCRIPT_cc}"
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
export CACHE_DIR="${CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache}"
export H_DIR="${H_DIR:-/lustre/fsw/portfolios/llmservice/users/heh}"
export PROJECT="${PROJECT:-SpeechlmPromptCtl84}"
# The FULL-FINETUNE arm at LR 1e-4 -- strictly causal encoder (chunk 1).
# Not the LoRA run of the same shape: no adapters, all 2.3B params trained.
EXP_NAME="${EXP_NAME:-granary2_script_promptctl84_full_causal_lr1e4}"; export EXP_NAME
export MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"
export SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are doing streaming speech recognition. Given the transcript so far and the representation of the next audio chunk, output the words spoken in that chunk.}"
# MANDATORY for these checkpoints -- see the header.
export PROMPT_CONTROL=1
export NUM_DELAY_FRAMES="${NUM_DELAY_FRAMES:-3}"
export CAPITALIZATION="${CAPITALIZATION:-1}"
export PUNCTUATION="${PUNCTUATION:-1}"
# Forced-loss arm: words are emitted whole, matching training.
export FORCE_WORD_START="${FORCE_WORD_START:-1}"
# Only ONE checkpoint exists so far (step 132000); nothing to average over.
# Every arm now has >=5 checkpoints, so average them rather than scoring a
# single -last. FORCE_AVERAGE=1 on the FIRST pass only: a cached average from
# the earlier step-136000 sweep would otherwise be reused silently and look
# like a fresh result, and rebuilding per chunk size would repeat a 5 GB
# read/write of a bit-identical file five times.
export RUN_AVERAGING="${RUN_AVERAGING:-1}"
export USE_LAST="${USE_LAST:-0}"
# These runs are STILL TRAINING, so a fixed suffix would make two sweeps taken at
# different points overwrite each other and silently mix step counts. Derive it
# from the highest checkpoint step instead: results are self-labelling, the
# script stays re-runnable at any point, and nothing lands in the v2 namespace
# where it could be mistaken for a converged result.
_CKPT_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/${EXP_NAME}/checkpoints"
_MAX_STEP="$(ls "${_CKPT_DIR}"/*.ckpt 2>/dev/null | grep -oE 'step=[0-9]+' | cut -d= -f2 | sort -n | tail -1)"
export RESULTS_SUFFIX="${RESULTS_SUFFIX:-d3c1p1_lr1e4_step${_MAX_STEP:-unknown}}"
echo "### averaging over: $(ls "${_CKPT_DIR}"/*.ckpt 2>/dev/null | grep -v -- '-last\.ckpt$' | grep -v -- '-averaged\.ckpt$' | grep -oE 'step=[0-9]+' | tr '\n' ' ')"
echo "### RESULTS_SUFFIX=${RESULTS_SUFFIX}"
BACKEND="$(resolve_launch_dir)/eval_leaderboard.sh"
echo "### node: $(hostname)   exp: ${EXP_NAME}"
CHUNK_LIST="${CHUNK_LIST:-14 28 42 7 2}"
first=1
for cs in ${CHUNK_LIST}; do
    echo ""
    echo "############ ${EXP_NAME} chunk_size=${cs} (promptctl d3 c1 p1) ############"
    t0=$(date +%s)
    CHUNK_SIZE="$cs" FORCE_AVERAGE="${first}" bash "$BACKEND" \
        || echo "WARNING: chunk ${cs} failed; continuing" >&2
    first=0
    echo "#### chunk=${cs} wall=$(( $(date +%s) - t0 ))s"
done
