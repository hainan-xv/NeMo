#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:sslm-lb-script-flushctx
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
# WITH a delay-sized LEFT AUDIO CONTEXT (e.g. granary2_script_promptctl_flush_ctx,
# trained by launch/script_promptctl_flush_ctx.sh).
#
# Two behaviors are baked into the checkpoint config (both set at train time), so
# generate() applies them AUTOMATICALLY -- no eval-side flag is needed:
#   * ++model.flush=true                    -> appends <flush> on each stream's final
#                                              chunk to drain delay-held tail words.
#   * ++model.audio_left_context_frames=N   -> each branch window is extended left by
#                                              N frames, so every chunk sees
#                                              (chunk_size + N) frames (its own chunk
#                                              plus N frames of pre-chunk history).
# Everything else (prompt building from chunk/delay/cap/punct, pooled 8-GPU shards,
# per-dataset + macro WER) is identical to the plain prompt-control eval, so this is
# a thin POSITIONAL wrapper around launch/eval_promptctl.sh.
#
# POSITIONAL ARGS:
#   $1  EXP_NAME    the exp/model name to eval  (default: granary2_script_promptctl_flush_ctx)
#   $2  CHUNK_SIZE  frames/chunk; also fills "...chunks of N frames." (default: 14)
#   $3  DELAY       emission delay in frames "...delay of N frames." (default: 3)
#   $4  CAP         capitalization on/off, 1|0                       (default: 1)
#   $5  PUNCT       punctuation on/off, 1|0                          (default: 1)
#
# Any positional arg left off falls back to its env var (CHUNK_SIZE/DELAY/CAP/PUNCT)
# and then the default above, so `CHUNK_SIZE=28 sbatch ... mymodel` still works.
#
# Usage (from the clean repo root on OCI):
#   sbatch launch/eval_flush_ctx_model.sh                                       # defaults
#   sbatch launch/eval_flush_ctx_model.sh granary2_script_promptctl_flush_ctx   # $1 = exp
#   sbatch launch/eval_flush_ctx_model.sh granary2_script_promptctl_flush_ctx 14 6      # chunk 14, delay 6
#   sbatch launch/eval_flush_ctx_model.sh granary2_script_promptctl_flush_ctx 14 3 0 0  # lowercase, no punct
#   for d in 1 3 6; do sbatch launch/eval_flush_ctx_model.sh my_ctx_model 14 $d; done   # delay sweep
#
# The left-context win vs the plain flush model shows up as HIGHER delays no longer
# degrading WER (d6 ~ d3), so a 1/3/6 delay sweep is the natural comparison to the
# launch/eval_flush_model.sh numbers.
#
# All other knobs (RUN_AVERAGING, CKPT/STEP/USE_LAST, DATASETS, BATCH_SIZE,
# MAX_NEW_TOKENS, USE_STATE_MACHINE, FORCE_WORD_START, MAX_EVAL_SAMPLES, wandb, ...)
# are env vars handled by eval_promptctl.sh / eval_leaderboard.sh -- see their
# headers.
#
# NOTE: both flush and the left context are ALWAYS ON here (they come from the
# checkpoint cfg; the leaderboard backend exposes no toggle). To measure the model
# WITHOUT the left context, eval the plain flush model with
# launch/eval_flush_model.sh; without flush at all, use launch/eval_promptctl.sh.
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

# --- Map positional args -> the knob env vars eval_promptctl.sh consumes ---
# $1 = exp/model name; $2.. = the regular prompt-control parameters (chunk/delay/cap/punct).
MODEL_NAME="${1:-${EXP_NAME:-granary2_script_promptctl_flush_ctx}}"
CHUNK_SIZE="${2:-${CHUNK_SIZE:-14}}"
DELAY="${3:-${DELAY:-3}}"
CAP="${4:-${CAP:-1}}"
PUNCT="${5:-${PUNCT:-1}}"

# eval_promptctl.sh reads these from the environment; export so they survive the exec.
export CHUNK_SIZE DELAY CAP PUNCT

# Tag results/wandb so flush-ctx runs are distinguishable at a glance (results land
# under the model's own exp dir, but this keeps the run label explicit).
export EVAL_TAG="${EVAL_TAG:-flushctx_c${CHUNK_SIZE}_d${DELAY}_cap${CAP}_punct${PUNCT}}"

echo "==> flush+left-context leaderboard eval (delegating to eval_promptctl.sh)"
echo "    model:      ${MODEL_NAME}"
echo "    chunk_size: ${CHUNK_SIZE}   delay: ${DELAY}   cap: ${CAP}   punct: ${PUNCT}"
echo "    eval_tag:   ${EVAL_TAG}"
echo "    (flush + audio_left_context_frames are baked into the checkpoint cfg)"

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
