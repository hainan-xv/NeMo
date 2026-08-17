#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:sslm-lb-script-promptctl
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
# Open-ASR-Leaderboard eval for the PROMPT-CONTROLLED SCRIPT model
# (granary2_script_promptctl, trained by launch/script_promptctl.sh).
#
# Unlike the baseline (one fixed operating point baked into the weights), this
# model is CONFIGURED AT INFERENCE via the system prompt: chunk size, emission
# delay, and capitalization/punctuation are all stated in the prompt. This
# wrapper BUILDS that prompt from knob env vars -- byte-identically to what
# ScriptSTTDataset renders at training time (see nemo/.../data/script_dataset.py
# and scripts/speechlm_promptctl_eval.py, the standalone single-GPU counterpart)
# -- then execs the shared pooled-shard backend launch/eval_leaderboard.sh on the
# SAME allocation (8-GPU pooled shards -> per-dataset + macro WER).
#
# Default operating point (all overridable):
#   CHUNK_SIZE=14  frames/chunk (decode chunk AND the "...chunks of N frames." clause;
#                  a TRAINED size in [2,7,14,28], matching the baseline eval + val_wer)
#   DELAY=3        emission delay in frames ("...delay of N frames.")
#   CAP=1 PUNCT=1  capitalization + punctuation ON -> picks the format clause
#
# Usage (from the clean repo root on OCI):
#   sbatch launch/eval_promptctl.sh                         # 14 fpc, cap+punct, delay 3
#   sbatch launch/eval_promptctl.sh granary2_script_myrun   # $1 = model/exp name
#   CHUNK_SIZE=28 DELAY=0 sbatch launch/eval_promptctl.sh    # different operating point
#   CAP=0 PUNCT=0 sbatch launch/eval_promptctl.sh            # lowercase, no punctuation
#   for c in 2 7 14 28; do CHUNK_SIZE=$c sbatch launch/eval_promptctl.sh; done  # sweep chunk
#   MAX_EVAL_SAMPLES=10 sbatch launch/eval_promptctl.sh      # smoke test, 10 utts/ds
#
# Prompt-shape knobs (defaults match launch/script_promptctl.sh's training render):
#   PROMPT_TEMPLATE     template with {delay} + {format_clause} placeholders
#   CHUNK_CLAUSE        chunk clause with a {chunk_size} placeholder
#   STATE_CHUNK_SIZE=1  1 -> append CHUNK_CLAUSE; 0 -> omit it (e.g. _promptctl_d8)
#   FORMAT_CLAUSE       override the (CAP, PUNCT)-selected format clause text
#   PROMPT_SUFFIX       extra trailing clause (e.g. the self-correction clause for _promptctl_all)
#   SYSTEM_PROMPT       if set, used VERBATIM (skips prompt building) -- escape hatch
#
# All other knobs (CHUNK_SIZE decode override, RUN_AVERAGING, CKPT/STEP/USE_LAST,
# DATASETS, BATCH_SIZE, MAX_NEW_TOKENS, wandb, ...) are handled by the backend
# launch/eval_leaderboard.sh -- see its header. Note CHUNK_SIZE here feeds BOTH the
# decode override and the "...chunks of N frames." wording, so they can't drift.
#
# NOTE: chunk sizes the model actually SAW in training were [2,7,14,28] (delay
# ~ U[0,6]); requesting an unseen size like 12 exercises the model's prompt-control
# generalization -- fine to try, but compare against a trained size if in doubt.
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

# --- Model + run identity (MUST match launch/script_promptctl.sh) ---
EXP_NAME="${1:-${EXP_NAME:-granary2_script_promptctl}}"
PROJECT="${PROJECT:-SpeechlmScriptClean}"
MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"

# --- Prompt-control operating point (the headline knobs) ---
CHUNK_SIZE="${CHUNK_SIZE:-14}"
DELAY="${DELAY:-3}"
CAP="${CAP:-1}"
PUNCT="${PUNCT:-1}"

# --- Prompt shape (defaults = the granary2_script_promptctl TRAINING render) ---
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-You are doing streaming speech recognition. Given the transcript so far and the next audio chunk, output the words spoken in that chunk. Emit the words of each chunk with a fixed delay of {delay} frames. {format_clause}}"
CHUNK_CLAUSE="${CHUNK_CLAUSE:-Process the audio in chunks of {chunk_size} frames.}"
STATE_CHUNK_SIZE="${STATE_CHUNK_SIZE:-1}"
FORMAT_CLAUSE="${FORMAT_CLAUSE:-}"
PROMPT_SUFFIX="${PROMPT_SUFFIX:-}"

# Trim leading/trailing ASCII whitespace (mirrors python str.strip()).
trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

# Select the (CAP, PUNCT) format clause (mirrors ScriptSTTDataset._DEFAULT_FORMAT_CLAUSES).
if [[ -n "$FORMAT_CLAUSE" ]]; then
    CLAUSE="$FORMAT_CLAUSE"
elif [[ "$CAP" == 1 && "$PUNCT" == 1 ]]; then
    CLAUSE="Write the text with normal capitalization and punctuation."
elif [[ "$CAP" == 1 && "$PUNCT" == 0 ]]; then
    CLAUSE="Write the text with normal capitalization but no punctuation."
elif [[ "$CAP" == 0 && "$PUNCT" == 1 ]]; then
    CLAUSE="Write the text in all lowercase, keeping punctuation."
else
    CLAUSE="Write the text in all lowercase with no punctuation."
fi

# Build the decode prompt from the knobs, EXACTLY like _build_exact_prompt +
# _append_chunk_clause: fill {delay}/{format_clause}, strip, then (optionally)
# append the chunk clause and any suffix. SYSTEM_PROMPT (if pre-set) wins verbatim.
if [[ -n "${SYSTEM_PROMPT:-}" ]]; then
    BUILT_PROMPT="$SYSTEM_PROMPT"
else
    BUILT_PROMPT="${PROMPT_TEMPLATE//\{delay\}/$DELAY}"
    BUILT_PROMPT="${BUILT_PROMPT//\{format_clause\}/$CLAUSE}"
    BUILT_PROMPT="$(trim "$BUILT_PROMPT")"
    if [[ "$STATE_CHUNK_SIZE" == 1 && -n "$CHUNK_CLAUSE" ]]; then
        CC="${CHUNK_CLAUSE//\{chunk_size\}/$CHUNK_SIZE}"
        CC="$(trim "$CC")"
        [[ -n "$CC" ]] && BUILT_PROMPT="$(trim "$BUILT_PROMPT") $CC"
    fi
    if [[ -n "$PROMPT_SUFFIX" ]]; then
        BUILT_PROMPT="$(trim "$BUILT_PROMPT") $(trim "$PROMPT_SUFFIX")"
    fi
fi

# Hand the built prompt + decode chunk size to the shared backend. EVAL_TAG labels
# the results dir / wandb run with this operating point.
SYSTEM_PROMPT="$BUILT_PROMPT"
EVAL_TAG="${EVAL_TAG:-promptctl_c${CHUNK_SIZE}_d${DELAY}_cap${CAP}_punct${PUNCT}}"
export SYSTEM_PROMPT MODEL_CLASS CHUNK_SIZE EXP_NAME PROJECT EVAL_TAG

echo "==> prompt-controlled SCRIPT leaderboard eval"
echo "    exp_name:      ${EXP_NAME}"
echo "    project:       ${PROJECT}"
echo "    chunk_size:    ${CHUNK_SIZE}   delay: ${DELAY}   cap: ${CAP}   punct: ${PUNCT}"
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
