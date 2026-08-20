#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:sslm-lb-script-flush
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
# Open-ASR-Leaderboard eval for a <flush>-trained PROMPT-CONTROLLED SCRIPT model
# (e.g. granary2_script_promptctl_flush, trained by launch/script_promptctl_flush.sh).
#
# The flush behavior is baked into the checkpoint config (++model.flush=true was set
# at train time), so generate() AUTOMATICALLY appends the <flush> control token on
# each stream's final chunk to drain delay-held tail words -- no eval-side flag is
# needed. Everything else (prompt building from chunk/delay/cap/punct, pooled 8-GPU
# shards, per-dataset + macro WER) is identical to the non-flush prompt-control eval,
# so this is a thin POSITIONAL wrapper around launch/eval_promptctl.sh.
#
# POSITIONAL ARGS:
#   $1  MODEL_NAME  the exp/model name to eval  (default: granary2_script_promptctl_flush)
#   $2  CHUNK_SIZE  frames/chunk; also fills "...chunks of N frames." (default: 14)
#   $3  DELAY       emission delay in frames "...delay of N frames." (default: 3)
#   $4  CAP         capitalization on/off, 1|0                       (default: 1)
#   $5  PUNCT       punctuation on/off, 1|0                          (default: 1)
#
# Any positional arg left off falls back to its env var (CHUNK_SIZE/DELAY/CAP/PUNCT)
# and then the default above, so `CHUNK_SIZE=28 sbatch ... mymodel` still works.
#
# Usage (from the clean repo root on OCI):
#   sbatch launch/eval_flush_model.sh                                   # defaults
#   sbatch launch/eval_flush_model.sh granary2_script_promptctl_flush   # $1 = model
#   sbatch launch/eval_flush_model.sh granary2_script_promptctl_flush 14 6      # chunk 14, delay 6
#   sbatch launch/eval_flush_model.sh granary2_script_promptctl_flush 14 3 0 0  # lowercase, no punct
#   for d in 1 3 6; do sbatch launch/eval_flush_model.sh my_flush_model 14 $d; done  # delay sweep
#
# All other knobs (RUN_AVERAGING, CKPT/STEP/USE_LAST, DATASETS, BATCH_SIZE,
# MAX_NEW_TOKENS, USE_STATE_MACHINE, FORCE_WORD_START, MAX_EVAL_SAMPLES, wandb, ...)
# are env vars handled by eval_promptctl.sh / eval_leaderboard.sh -- see their
# headers.
#
# NOTE: the final-chunk flush is ALWAYS ON here (it comes from the checkpoint's
# model.flush=true; generate() adds flush=False support but the leaderboard backend
# does not expose that toggle). To measure the WITHOUT-flush baseline, eval the
# non-flush prompt-control model with launch/eval_promptctl.sh instead.
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

# --- Map positional args -> the knob env vars eval_promptctl.sh consumes ---
# $1 = model name; $2.. = the regular prompt-control parameters (chunk/delay/cap/punct).
MODEL_NAME="${1:-${EXP_NAME:-granary2_script_promptctl_flush}}"
CHUNK_SIZE="${2:-${CHUNK_SIZE:-14}}"
DELAY="${3:-${DELAY:-3}}"
CAP="${4:-${CAP:-1}}"
PUNCT="${5:-${PUNCT:-1}}"

# eval_promptctl.sh reads these from the environment; export so they survive the exec.
export CHUNK_SIZE DELAY CAP PUNCT

# Default to OFFLINE encode (USE_STATE_MACHINE=0). WER evals must match TRAINING,
# which always encodes offline; the cache-aware STREAMING encoder used by the state
# machine does NOT bit-reproduce the offline encode (measured per-frame cosine ~0.9,
# off-by-one frame count, worst at the tail), which corrupts exactly the tail/delayed
# words and masks the <flush> benefit. The launch backend otherwise defaults plain
# SCRIPT to the state machine (_sm); pin it off here. Override with USE_STATE_MACHINE=1
# only for bounded-memory long-form checks.
export USE_STATE_MACHINE="${USE_STATE_MACHINE:-0}"

# Tag results/wandb so flush runs are distinguishable at a glance (results already
# land under the flush model's own exp dir, but this keeps the run label explicit).
export EVAL_TAG="${EVAL_TAG:-flush_c${CHUNK_SIZE}_d${DELAY}_cap${CAP}_punct${PUNCT}}"

echo "==> flush-model leaderboard eval (delegating to eval_promptctl.sh)"
echo "    model:      ${MODEL_NAME}"
echo "    chunk_size: ${CHUNK_SIZE}   delay: ${DELAY}   cap: ${CAP}   punct: ${PUNCT}"
echo "    eval_tag:   ${EVAL_TAG}"

# Locate eval_promptctl.sh. Under sbatch, Slurm COPIES this script into a spool dir,
# so BASH_SOURCE is unreliable -- prefer SLURM_SUBMIT_DIR (cwd at submit time).
# Accept submit-from-repo-root (`sbatch launch/...`, sibling at launch/) or
# submit-from-launch/ (`sbatch ./...`, alongside).
resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        if [[ -f "${SLURM_SUBMIT_DIR}/eval_promptctl.sh" ]]; then
            echo "${SLURM_SUBMIT_DIR}"; return
        fi
        if [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_promptctl.sh" ]]; then
            echo "${SLURM_SUBMIT_DIR}/launch"; return
        fi
    fi
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "${here}/eval_promptctl.sh" ]]; then
        echo "${here}"; return
    fi
    echo "ERROR: cannot locate eval_promptctl.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
LAUNCH_DIR="$(resolve_launch_dir)"
# Keep cwd at the submit dir so relative paths (slurm_out/) land where expected.
cd "${SLURM_SUBMIT_DIR:-$LAUNCH_DIR}"

# Already inside the sbatch allocation: run eval_promptctl.sh as a normal bash script
# (its own #SBATCH headers are ignored). $1 = model/exp name; the knob env vars above
# drive the prompt build. It in turn execs the shared pooled-shard backend.
exec bash "${LAUNCH_DIR}/eval_promptctl.sh" "${MODEL_NAME}"
