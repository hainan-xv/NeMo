#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:script-lb-eval
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
# Open-ASR-Leaderboard eval for ANY SCRIPT model -- pass the experiment name.
#
#   sbatch launch/eval_script.sh granary2_script_win28
#   sbatch launch/eval_script.sh granary2_script_win28 7        # chunk size 7
#   sbatch launch/eval_script.sh granary2_script_win14          # a future variant
#
# Or from your laptop:
#   ./oci_launch.sh launch/eval_script.sh granary2_script_win28
#   for c in 2 7 14; do ./oci_launch.sh launch/eval_script.sh granary2_script_win28 $c; done
#
# WHY ONE SCRIPT COVERS ALL THE VARIANTS
#   Every SCRIPT recipe (baseline, flex, win28, twod, ...) shares the same
#   system prompt and the same chunk-size set -- verified identical -- and the
#   settings that DO differ ride inside the checkpoint config and are rebuilt
#   automatically by the model at load time:
#       audio_window_frames      (win28 = 28, baseline = 0)
#       audio_history_chunks
#       attn_backend / activation_checkpointing   (training-only; decode uses SDPA)
#   So a window model needs nothing special here. If you ever train with a
#   DIFFERENT prompt, you must pass SYSTEM_PROMPT= to match it, or the decode is
#   out of distribution.
#
# POSITIONAL ARGS
#   $1  EXP_NAME    experiment to evaluate  (REQUIRED)
#   $2  CHUNK_SIZE  decode chunk size in encoder frames (default: 14)
#
# CHUNK SIZE is the inference-time latency knob. Use a size the model actually
# trained on -- {2, 4, 7, 10, 14, 28} for the current recipes. Anything else is
# out of distribution. Frames are 0.08s, so 2 -> 0.16s, 14 -> 1.12s.
#
# PROMPT-CONTROLLED MODELS
#   A model trained with model.prompt_control=true also accepts an operating
#   point, requested as env vars:
#       NUM_DELAY_FRAMES=6 CAPITALIZATION=1 PUNCTUATION=0 \
#           ./oci_launch.sh launch/eval_script.sh granary2_script_promptctl 7
#   Omit them to decode at the checkpoint's val_* defaults. A model trained
#   WITHOUT prompt control rejects them rather than ignoring them, so a request
#   the checkpoint cannot honour fails loudly instead of quietly.
#   Note the leaderboard normalizer lowercases and strips punctuation before
#   scoring, so CAPITALIZATION/PUNCTUATION will not move the reported WER.
#
# All other knobs (RUN_AVERAGING, CKPT/STEP/USE_LAST, DATASETS, BATCH_SIZE,
# MAX_NEW_TOKENS, MAX_EVAL_SAMPLES, FORCE_WORD_START, wandb, ...) are env vars
# handled by eval_leaderboard.sh -- see its header.
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

PROJECT="${PROJECT:-SpeechlmScriptCC}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"

EXP_NAME="${1:-${EXP_NAME:-}}"
if [[ -z "$EXP_NAME" ]]; then
    echo "ERROR: no experiment name given." >&2
    echo "usage: sbatch launch/eval_script.sh <exp_name> [chunk_size]" >&2
    echo "" >&2
    echo "Experiments under ${OUTPUT_PREFIX}/results/${PROJECT}:" >&2
    ls -1 "${OUTPUT_PREFIX}/results/${PROJECT}" 2>/dev/null | sed 's/^/  /' >&2 || echo "  (none found)" >&2
    exit 1
fi

CHUNK_SIZE="${2:-${CHUNK_SIZE:-14}}"
MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"

# Decode prompt -- MUST be byte-for-byte the training instruction (identical
# across every current SCRIPT recipe). Override only if the model was trained
# with a different one; drift here is silently out-of-distribution.
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are doing streaming speech recognition. Given the transcript so far and the representation of the next audio chunk, output the words spoken in that chunk.}"

# Tag results/wandb by model+chunk so a sweep is readable at a glance.
EVAL_TAG="${EVAL_TAG:-${EXP_NAME}}"

# Named here (not just read by the backend) so oci_launch.sh's knob discovery,
# which scans this script for ${NAME:-...}, forwards them from your shell.
# FSM decode (see scripts/script_fsm.py). Independent switches.
STATE_MACHINE="${STATE_MACHINE:-}"       # 1 = per-stream state machine decode
STREAMING_ENCODE="${STREAMING_ENCODE:-}" # 1 = cache-aware streaming perception

# Declared here, not just documented, because oci_launch.sh only forwards a var
# it can SEE as ${NAME:-...} in this file. Left empty so eval_leaderboard.sh keeps
# its own defaults; a comment-only mention would be dropped silently, and for a
# full-context model that means generation truncated at 64 tokens with nothing
# in the log to say so.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"

NUM_DELAY_FRAMES="${NUM_DELAY_FRAMES:-}"
CAPITALIZATION="${CAPITALIZATION:-}"
PUNCTUATION="${PUNCTUATION:-}"

export EXP_NAME PROJECT MODEL_CLASS SYSTEM_PROMPT CHUNK_SIZE EVAL_TAG OUTPUT_PREFIX
export NUM_DELAY_FRAMES CAPITALIZATION PUNCTUATION STATE_MACHINE STREAMING_ENCODE
export MAX_NEW_TOKENS BATCH_SIZE

CKPT_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/${EXP_NAME}/checkpoints"
echo "==> SCRIPT leaderboard eval"
echo "    exp:        ${EXP_NAME}"
echo "    chunk_size: ${CHUNK_SIZE} frames ($(python3 -c "print(f'{${CHUNK_SIZE}*0.08:.2f}')" 2>/dev/null || echo '?')s)"
echo "    ckpt dir:   ${CKPT_DIR}"
if [[ ! -d "$CKPT_DIR" ]]; then
    echo "WARNING: no checkpoints directory at ${CKPT_DIR}" >&2
    echo "         (the run may not have saved a checkpoint yet)" >&2
fi

# Locate the shared backend. Under sbatch, Slurm COPIES this script into a spool
# dir, so BASH_SOURCE is unreliable -- prefer SLURM_SUBMIT_DIR.
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
