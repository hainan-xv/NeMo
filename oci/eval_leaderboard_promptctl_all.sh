#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:sslm-lb-promptctl-all
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
# Leaderboard eval for the UNIFIED prompt-controlled chunk-completion model
# (granary2_chunkcompletion_promptctl_all).
#
# Builds the SYSTEM_PROMPT exactly as training does and hands off to
# oci/eval_leaderboard_slurm.sh. The prompt states ALL of the model's per-batch
# knobs, in the SAME append order the dataset uses:
#     base (delay + cap/punct format clause)   -- from _promptctl
#   + chunk-size clause                         -- "Process the audio in chunks of N frames."
#   + self-correction ON/OFF clause             -- may-correct vs must-not
#
# The chunk size is stated in the prompt AND used as the decode chunk size (they
# must match, as in training). The delete-last-word mechanism itself is handled by
# ChunkCompletionSTTModel.generate (config-driven: self_correction=true enables the
# <del> token + word-pop); the ON/OFF clause is what tells the model whether to use
# it. Both the heh and sslm backends route through model.generate.
#
# Usage (from the NeMo79 repo root on OCI):
#   sbatch oci/eval_leaderboard_promptctl_all.sh <cap> <punct> <delay> <chunk> <correct>
#
#   <cap>     : cap | nocap        (aliases: true/1/yes , false/0/no)
#   <punct>   : punct | nopunct    (aliases: true/1/yes , false/0/no)
#   <delay>   : integer frames in [0,4]
#   <chunk>   : chunk size (frames), one the model trained on: 2 4 7 10 14 28
#   <correct> : correct | nocorrect  (self-correction on/off; aliases true/1/yes,false/0/no)
#
# Examples:
#   sbatch oci/eval_leaderboard_promptctl_all.sh cap   punct   2 14 correct
#   sbatch oci/eval_leaderboard_promptctl_all.sh cap   punct   2 14 nocorrect
#   sbatch oci/eval_leaderboard_promptctl_all.sh nocap nopunct 0 7  correct
#
# Overridable via env (match the recipe's clauses if you changed them):
#   EXP_NAME=...  SC_ON_SUFFIX="..."  SC_OFF_SUFFIX="..."  CHUNK_TEMPLATE="... {chunk} ..."
#
# Checkpoint AVERAGING is ON by default (eval_leaderboard_slurm.sh RUN_AVERAGING=1):
# the top-k (by val_wer) checkpoints of this exp are averaged into
# <EXP>-averaged.ckpt (cached + reused, locked across concurrent jobs). Disable with
# RUN_AVERAGING=0, or select a specific ckpt with USE_LAST=1 / STEP=n / CKPT=path.
#
# Concurrent jobs for the same model share the averaged ckpt + HF convert (mkdir
# locks). Each run gets its own RESULTS_DIR via
# EVAL_TAG=d<delay>_c<chunk>_<cap>_<punct>_<corr> + slurm job id.
#
# Optional env (forwarded to eval_leaderboard_slurm.sh):
#   BACKEND=heh|sslm   BATCH_SIZE=...   SAVE_RAW=true|false   MAX_EVAL_SAMPLES=...
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

usage() {
    echo "usage: sbatch oci/eval_leaderboard_promptctl_all.sh <cap> <punct> <delay> <chunk> <correct>" >&2
    echo "  cap     : cap | nocap" >&2
    echo "  punct   : punct | nopunct" >&2
    echo "  delay   : integer frames (0-4)" >&2
    echo "  chunk   : chunk size in frames (2 4 7 10 14 28)" >&2
    echo "  correct : correct | nocorrect" >&2
    exit 1
}

[[ $# -ge 5 ]] || usage

CAP_ARG="$1"
PUNCT_ARG="$2"
DELAY_ARG="$3"
CHUNK_ARG="$4"
CORRECT_ARG="$5"

# Fixed model (unified prompt-ctl, exact_max_delay=4). Override EXP_NAME to eval a
# differently-named run.
EXP_NAME="${EXP_NAME:-granary2_chunkcompletion_promptctl_all}"
MAX_DELAY="${MAX_DELAY:-4}"

# Prompt clauses -- MUST match the recipe (data.dataset.*) so eval is in-distribution.
CHUNK_TEMPLATE="${CHUNK_TEMPLATE:-Process the audio in chunks of {chunk} frames.}"
SC_ON_SUFFIX="${SC_ON_SUFFIX:-If a word you already wrote turns out to be wrong given the new audio, delete it and write the correct word.}"
SC_OFF_SUFFIX="${SC_OFF_SUFFIX:-Do not go back and change words you already wrote.}"

# ---- cap / punct -> bool ----
parse_bool() {
    # $1 = value, $2 = positive name (cap|punct), $3 = negative name (nocap|nopunct)
    local v
    v="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    case "$v" in
        "$2"|true|1|yes|on)  echo 1 ;;
        "$3"|false|0|no|off) echo 0 ;;
        *)
            echo "ERROR: expected '$2' or '$3' (got '$1')" >&2
            exit 1
            ;;
    esac
}
CAP="$(parse_bool "$CAP_ARG" cap nocap)"
PUNCT="$(parse_bool "$PUNCT_ARG" punct nopunct)"
CORRECT="$(parse_bool "$CORRECT_ARG" correct nocorrect)"

# ---- delay ----
if ! [[ "$DELAY_ARG" =~ ^[0-9]+$ ]]; then
    echo "ERROR: delay must be a non-negative integer (got '$DELAY_ARG')" >&2
    exit 1
fi
DELAY=$((10#$DELAY_ARG))
if (( DELAY < 0 || DELAY > MAX_DELAY )); then
    echo "ERROR: delay=$DELAY out of range for ${EXP_NAME} (allowed 0..${MAX_DELAY})" >&2
    exit 1
fi

# ---- chunk size (stated in the prompt AND used as the decode chunk size) ----
if ! [[ "$CHUNK_ARG" =~ ^[0-9]+$ ]]; then
    echo "ERROR: chunk must be a positive integer (got '$CHUNK_ARG')" >&2
    exit 1
fi
CHUNK=$((10#$CHUNK_ARG))
(( CHUNK >= 1 )) || { echo "ERROR: chunk must be >= 1 (got '$CHUNK_ARG')" >&2; exit 1; }

# ---- format clause (verbatim from ChunkCompletionSTTDataset._DEFAULT_FORMAT_CLAUSES) ----
if (( CAP )) && (( PUNCT )); then
    FORMAT_CLAUSE="Write the text with normal capitalization and punctuation."
    REPR_TAG=cap_punct
elif (( CAP )) && ! (( PUNCT )); then
    FORMAT_CLAUSE="Write the text with normal capitalization but no punctuation."
    REPR_TAG=cap_nopunct
elif ! (( CAP )) && (( PUNCT )); then
    FORMAT_CLAUSE="Write the text in all lowercase, keeping punctuation."
    REPR_TAG=nocap_punct
else
    FORMAT_CLAUSE="Write the text in all lowercase with no punctuation."
    REPR_TAG=nocap_nopunct
fi

# ---- chunk-size + self-correction clauses (verbatim from the recipe) ----
CHUNK_CLAUSE="${CHUNK_TEMPLATE/\{chunk\}/$CHUNK}"
if (( CORRECT )); then
    CORR_CLAUSE="$SC_ON_SUFFIX"
    CORR_TAG=correct
else
    CORR_CLAUSE="$SC_OFF_SUFFIX"
    CORR_TAG=nocorrect
fi

# Full prompt, assembled in the SAME order as the dataset:
#   base (delay + format clause) -> chunk-size clause -> self-correction clause.
SYSTEM_PROMPT="You are doing streaming speech recognition. Given the transcript so far and the next audio chunk, output the words spoken in that chunk. Emit each chunk's words with a fixed delay of ${DELAY} frames. ${FORMAT_CLAUSE} ${CHUNK_CLAUSE} ${CORR_CLAUSE}"

MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.chunk_completion_model.ChunkCompletionSTTModel}"
# The stated chunk size IS the decode chunk size (prompt + decode must match).
CHUNK_SIZE="$CHUNK"
# Dump the raw <del>-annotated emission stream (raw_text) so you can see where the
# model self-corrected. On by default; set SAVE_RAW=false to skip.
SAVE_RAW="${SAVE_RAW:-true}"
# Distinguishes concurrent evals in RESULTS_DIR (shared avg/HF are locked).
EVAL_TAG="d${DELAY}_c${CHUNK}_${REPR_TAG}_${CORR_TAG}"

export SYSTEM_PROMPT MODEL_CLASS CHUNK_SIZE EXP_NAME EVAL_TAG SAVE_RAW

echo "==> unified prompt-controlled leaderboard eval"
echo "    exp_name:      ${EXP_NAME}"
echo "    eval_tag:      ${EVAL_TAG}"
echo "    cap/punct:     ${REPR_TAG}"
echo "    delay:         ${DELAY} frames (max trained ${MAX_DELAY})"
echo "    chunk_size:    ${CHUNK_SIZE} (stated in prompt AND used for decode)"
echo "    self-correct:  ${CORR_TAG} (delete-last-word is config-driven in generate; clause turns it on/off)"
echo "    averaging:     RUN_AVERAGING=${RUN_AVERAGING:-1} (best-checkpoint average unless CKPT/STEP/USE_LAST set)"
echo "    save_raw:      ${SAVE_RAW} (raw_text field with <del> markers in generations.jsonl)"
echo "    system_prompt: ${SYSTEM_PROMPT}"

# Locate oci/eval_leaderboard_slurm.sh. Under sbatch, Slurm COPIES this script
# into /cm/local/apps/slurm/var/spool/job<id>/, so BASH_SOURCE is useless —
# use SLURM_SUBMIT_DIR (cwd at sbatch time) instead. Accepts submit-from-repo-
# root (`sbatch oci/...`) or submit-from-oci/ (`sbatch ./...`).
resolve_oci_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        if [[ -f "${SLURM_SUBMIT_DIR}/eval_leaderboard_slurm.sh" ]]; then
            echo "${SLURM_SUBMIT_DIR}"
            return
        fi
        if [[ -f "${SLURM_SUBMIT_DIR}/oci/eval_leaderboard_slurm.sh" ]]; then
            echo "${SLURM_SUBMIT_DIR}/oci"
            return
        fi
    fi
    # Interactive / non-sbatch fallback (script not spool-copied).
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "${here}/eval_leaderboard_slurm.sh" ]]; then
        echo "${here}"
        return
    fi
    echo "ERROR: cannot locate eval_leaderboard_slurm.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
OCI_DIR="$(resolve_oci_dir)"
# Keep cwd at the submit dir so relative paths (slurm_out/) land where expected.
cd "${SLURM_SUBMIT_DIR:-$OCI_DIR}"

# Already inside the sbatch allocation: run the shared eval body as a normal
# bash script (its #SBATCH headers are ignored). $1 = EXP_NAME.
exec bash "${OCI_DIR}/eval_leaderboard_slurm.sh" "${EXP_NAME}"
