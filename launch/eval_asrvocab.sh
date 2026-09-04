#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:asrvocab-lb-eval
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
# Open-ASR-Leaderboard eval for a SCRIPT model trained on the ASR ENCODER'S
# TEXT VOCABULARY (model.text_vocab_from_asr=true).
#
#   sbatch launch/eval_asrvocab.sh granary2_script_asrvocab_fix
#   sbatch launch/eval_asrvocab.sh granary2_script_asrvocab_fix 2      # chunk 2
#
#   ./oci_launch.sh launch/eval_asrvocab.sh granary2_script_asrvocab_fix
#   for c in 2 7 14; do ./oci_launch.sh launch/eval_asrvocab.sh granary2_script_asrvocab_fix $c; done
#
# WHY A SEPARATE SCRIPT FROM eval_script.sh
#   Decoding needs no special handling: `text_vocab_from_asr` rides in the
#   checkpoint config, so the model rebuilds the 1,050-piece SentencePiece
#   tokenizer from `pretrained_asr` at load time and every marker resolves by
#   string exactly as before. What DOES change is the generation budget.
#
#   MAX_NEW_TOKENS is a per-chunk cap counted in TOKENS, and this vocabulary is
#   ~1.62x more granular than Qwen's (1.877 vs 1.160 tokens/word on AMI). The
#   shared default of 64 therefore buys only ~33 words here against ~64 with
#   Qwen -- the same number silently means half the headroom. So it defaults to
#   128 below, restoring the original word-level allowance.
#
#   Everything else is deliberately identical to eval_script.sh, so a number
#   from here is comparable to a baseline SCRIPT number.
#
# POSITIONAL ARGS
#   $1  EXP_NAME    experiment to evaluate  (REQUIRED)
#   $2  CHUNK_SIZE  decode chunk size in encoder frames (default: 14)
#
# WHERE THIS VOCABULARY IS EXPECTED TO HURT: small chunks. The token inflation
# lengthens every branch, and branches are proportionally longest when chunks
# are small, so chunk 2 is the informative cell -- not chunk 14, where every
# variant in this project has historically tied.
#
# All other knobs (RUN_AVERAGING, CKPT/STEP/USE_LAST, DATASETS, BATCH_SIZE,
# MAX_EVAL_SAMPLES, FORCE_WORD_START, wandb, ...) are env vars handled by
# eval_leaderboard.sh -- see its header.
#
# NOTE ON FORCE_WORD_START: it is ON by default and, unlike under the old
# tokenisation bug, it now actually fires. While chunk targets began with a bare
# word-start token the guard's condition was permanently satisfied and it never
# triggered; with that fixed it again protects against a chunk merging onto the
# previous chunk's last word.
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

PROJECT="${PROJECT:-SpeechlmScriptCC}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"

EXP_NAME="${1:-${EXP_NAME:-}}"
if [[ -z "$EXP_NAME" ]]; then
    echo "ERROR: no experiment name given." >&2
    echo "usage: sbatch launch/eval_asrvocab.sh <exp_name> [chunk_size]" >&2
    echo "" >&2
    echo "Experiments under ${OUTPUT_PREFIX}/results/${PROJECT}:" >&2
    ls -1 "${OUTPUT_PREFIX}/results/${PROJECT}" 2>/dev/null | grep -i asrvocab | sed 's/^/  /' >&2 || echo "  (none found)" >&2
    exit 1
fi

CHUNK_SIZE="${2:-${CHUNK_SIZE:-14}}"
MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"

# Byte-for-byte the training instruction, as for every other SCRIPT recipe. The
# ASR vocabulary tokenises it differently, but the STRING must still match or
# the decode is out of distribution.
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are doing streaming speech recognition. Given the transcript so far and the representation of the next audio chunk, output the words spoken in that chunk.}"

# See the header: 128 here is the word-level equivalent of 64 under Qwen.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
BATCH_SIZE="${BATCH_SIZE:-}"
DATASETS="${DATASETS:-}"

EVAL_TAG="${EVAL_TAG:-${EXP_NAME}}"

# Named here (not just read by the backend) so oci_launch.sh's knob discovery,
# which scans this script for ${NAME:-...}, forwards them from your shell.
STATE_MACHINE="${STATE_MACHINE:-}"
STREAMING_ENCODE="${STREAMING_ENCODE:-}"
NUM_DELAY_FRAMES="${NUM_DELAY_FRAMES:-}"
CAPITALIZATION="${CAPITALIZATION:-}"
PUNCTUATION="${PUNCTUATION:-}"

export EXP_NAME PROJECT MODEL_CLASS SYSTEM_PROMPT CHUNK_SIZE EVAL_TAG OUTPUT_PREFIX
export NUM_DELAY_FRAMES CAPITALIZATION PUNCTUATION STATE_MACHINE STREAMING_ENCODE
export MAX_NEW_TOKENS BATCH_SIZE DATASETS

CKPT_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/${EXP_NAME}/checkpoints"
echo "==> ASR-vocabulary SCRIPT leaderboard eval"
echo "    exp:            ${EXP_NAME}"
echo "    chunk_size:     ${CHUNK_SIZE} frames ($(python3 -c "print(f'{${CHUNK_SIZE}*0.08:.2f}')" 2>/dev/null || echo '?')s)"
echo "    max_new_tokens: ${MAX_NEW_TOKENS} (per chunk; ~1.62x more tokens/word than Qwen)"
echo "    ckpt dir:       ${CKPT_DIR}"
if [[ ! -d "$CKPT_DIR" ]]; then
    echo "WARNING: no checkpoints directory at ${CKPT_DIR}" >&2
    echo "         (the run may not have saved a checkpoint yet)" >&2
fi

# Under sbatch, Slurm COPIES this script into a spool dir, so BASH_SOURCE is
# unreliable -- prefer SLURM_SUBMIT_DIR.
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
