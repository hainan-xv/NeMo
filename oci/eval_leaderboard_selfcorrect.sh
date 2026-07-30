#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:sslm-lb-selfcorrect
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
# Leaderboard eval for the SELF-CORRECTION chunk-completion model
# (granary2_chunkcompletion_selfcorrect).
#
# Builds the SYSTEM_PROMPT exactly as training does -- the prompt-controlled
# exact-delay + cap/punct format clause (from _promptctl) PLUS the self-correction
# suffix -- then hands off to oci/eval_leaderboard_slurm.sh.
#
# The delete-last-word correction mechanism itself is handled automatically by
# ChunkCompletionSTTModel.generate: the checkpoint's model config has
# self_correction=true, so generate() enables the <del> token and its word-pop in
# the batched decoder (both the heh and sslm eval backends route through
# model.generate). Nothing extra is needed here beyond the matching prompt.
#
# Usage (from the NeMo79 repo root on OCI):
#   sbatch oci/eval_leaderboard_selfcorrect.sh <cap> <punct> <delay>
#
#   <cap>    : cap | nocap     (aliases: true/1/yes , false/0/no)
#   <punct>  : punct | nopunct (aliases: true/1/yes , false/0/no)
#   <delay>  : integer frames in [0,4]  (built on _promptctl, exact_max_delay=4)
#
# Examples:
#   sbatch oci/eval_leaderboard_selfcorrect.sh cap   punct   2
#   sbatch oci/eval_leaderboard_selfcorrect.sh nocap nopunct 0
#
# Override the exact exp name / delete instruction via env:
#   EXP_NAME=... SC_SUFFIX="..." sbatch oci/eval_leaderboard_selfcorrect.sh cap punct 2
#
# Concurrent jobs for the same model share the averaged ckpt + HF convert
# (serialized by mkdir locks in eval_leaderboard_slurm.sh). Each run gets its own
# RESULTS_DIR via EVAL_TAG=d<delay>_<cap>_<punct>_sc + slurm job id.
#
# Optional env (forwarded to eval_leaderboard_slurm.sh):
#   CHUNK_SIZE=14 (default)   BACKEND=heh|sslm   BATCH_SIZE=...
#   USE_LAST=1 / STEP=n / CKPT=path / RUN_AVERAGING=0 / MAX_EVAL_SAMPLES=...
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

usage() {
    echo "usage: sbatch oci/eval_leaderboard_selfcorrect.sh <cap> <punct> <delay>" >&2
    echo "  cap   : cap | nocap" >&2
    echo "  punct : punct | nopunct" >&2
    echo "  delay : integer frames (0-4)" >&2
    exit 1
}

[[ $# -ge 3 ]] || usage

CAP_ARG="$1"
PUNCT_ARG="$2"
DELAY_ARG="$3"

# Fixed model (built on _promptctl, exact_max_delay=4). Override EXP_NAME to eval a
# differently-named self-correction run.
EXP_NAME="${EXP_NAME:-granary2_chunkcompletion_selfcorrect}"
MAX_DELAY="${MAX_DELAY:-4}"

# Self-correction prompt suffix -- MUST match data.dataset.self_correction_prompt_suffix
# in the recipe so eval is in-distribution.
SC_SUFFIX="${SC_SUFFIX:-If a word you already wrote turns out to be wrong given the new audio, delete it and write the correct word.}"

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

# Prompt = the promptctl template (exact_delay + vary_text_repr) + the
# self-correction suffix, matching how the selfcorrect recipe builds it in training.
SYSTEM_PROMPT="You are doing streaming speech recognition. Given the transcript so far and the next audio chunk, output the words spoken in that chunk. Emit each chunk's words with a fixed delay of ${DELAY} frames. ${FORMAT_CLAUSE} ${SC_SUFFIX}"

MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.chunk_completion_model.ChunkCompletionSTTModel}"
CHUNK_SIZE="${CHUNK_SIZE:-14}"
# Distinguishes concurrent selfcorrect evals in RESULTS_DIR (shared avg/HF are locked).
EVAL_TAG="d${DELAY}_${REPR_TAG}_sc"

export SYSTEM_PROMPT MODEL_CLASS CHUNK_SIZE EXP_NAME EVAL_TAG

echo "==> self-correction leaderboard eval"
echo "    exp_name:      ${EXP_NAME}"
echo "    eval_tag:      ${EVAL_TAG}"
echo "    cap/punct:     ${REPR_TAG}"
echo "    delay:         ${DELAY} frames (max trained ${MAX_DELAY})"
echo "    chunk_size:    ${CHUNK_SIZE}"
echo "    self-correct:  delete-last-word handled by ChunkCompletionSTTModel.generate (config-driven)"
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
