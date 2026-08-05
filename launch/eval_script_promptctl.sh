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
# Uses the SAME pooled-shard, work-BALANCED backend as every other SpeechLM eval
# (launch/eval_leaderboard_slurm.sh): pool all utterances, global shuffle (fixed
# seed), 8 duration-sorted shards, one decode process per GPU -> wall time
# ~= sum(all)/8. This wrapper only picks the operating point, builds the prompt
# exactly as ScriptSTTDataset renders it at training, then execs that backend on
# the SAME allocation.
#
# The model's operating point is chosen ENTIRELY through the prompt, so the eval
# just states the desired (delay, capitalization, punctuation, chunk size). The
# prompt is assembled from three verbatim pieces the dataset uses:
#   base template  (ScriptSTTDataset._DEFAULT_PROMPT_TEMPLATE, {delay} filled)
#   + format clause (_DEFAULT_FORMAT_CLAUSES[cap/punct])
#   + chunk clause  ("Process the audio in chunks of {chunk} frames.")
# The chunk clause number MUST equal the decode CHUNK_SIZE (both set here).
#
# Usage (from the NeMo79 repo root on OCI):
#   sbatch launch/eval_script_promptctl.sh [cap] [punct] [delay] [chunk]
#     cap    : cap | nocap       (default cap)      aliases: true/1/yes, false/0/no
#     punct  : punct | nopunct   (default punct)    aliases: true/1/yes, false/0/no
#     delay  : integer frames 0..4 (default 2)      (exact_max_delay=4 in the recipe)
#     chunk  : encoder frames/chunk (default 14)    (one the model saw: {2,7,14,28})
#
# Examples:
#   sbatch launch/eval_script_promptctl.sh                      # cap punct, delay 2, chunk 14
#   sbatch launch/eval_script_promptctl.sh cap   punct   0  7   # low-latency, chunk 7
#   sbatch launch/eval_script_promptctl.sh nocap nopunct 4  28  # high-accuracy, chunk 28
#   for d in 0 2 4; do sbatch launch/eval_script_promptctl.sh cap punct $d 14; done  # delay sweep
#
# Each run gets its own RESULTS_DIR via EVAL_TAG=d<delay>_<cap>_<punct> + the
# backend's _chunk<chunk> tag + slurm job id; concurrent runs for the same exp
# share the averaged ckpt / HF convert (mkdir-locked in the backend).
#
# Optional env (forwarded to eval_leaderboard_slurm.sh):
#   BACKEND=heh|sslm   BATCH_SIZE=...   PROJECT=SpeechlmRefactored (default)
#   RUN_AVERAGING=1 (default) / USE_LAST=1 / STEP=n / CKPT=path
#   DATASETS="..."   MAX_EVAL_SAMPLES=n (smoke test)   SHUFFLE_SEED=1234
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

# --- Operating point (positional, all optional with defaults) ---
CAP_ARG="${1:-cap}"
PUNCT_ARG="${2:-punct}"
DELAY_ARG="${3:-2}"
CHUNK_ARG="${4:-14}"

MAX_DELAY=4   # recipe exact_max_delay=4 (delay ~ Uniform[0,4] at training)

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
    echo "ERROR: delay=$DELAY out of range for granary2_script_promptctl (allowed 0..${MAX_DELAY})" >&2
    exit 1
fi

# ---- chunk size ----
if ! [[ "$CHUNK_ARG" =~ ^[0-9]+$ ]] || (( CHUNK_ARG < 1 )); then
    echo "ERROR: chunk must be a positive integer (got '$CHUNK_ARG')" >&2
    exit 1
fi
CHUNK_SIZE=$((10#$CHUNK_ARG))

# ---- format clause (verbatim from ScriptSTTDataset._DEFAULT_FORMAT_CLAUSES) ----
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

# Prompt assembled EXACTLY as ScriptSTTDataset builds it for this recipe:
#   _build_exact_prompt(): DEFAULT_PROMPT_TEMPLATE.format(delay, format_clause).strip()
#   _append_chunk_clause(): + " " + "Process the audio in chunks of <chunk> frames."
# (chunk_size_prompt_template in streaming_stt_granary2_lora_script_promptctl.yaml).
SYSTEM_PROMPT="You are doing streaming speech recognition. Given the transcript so far and the next audio chunk, output the words spoken in that chunk. Emit each chunk's words with a fixed delay of ${DELAY} frames. ${FORMAT_CLAUSE} Process the audio in chunks of ${CHUNK_SIZE} frames."

EXP_NAME="${EXP_NAME:-granary2_script_promptctl}"
# The training project (launch/script_promptctl.sh: PROJECT_NAME=SpeechlmRefactored).
PROJECT="${PROJECT:-SpeechlmRefactored}"
MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"
# Distinguishes concurrent promptctl evals in RESULTS_DIR (backend adds _chunk<n>).
EVAL_TAG="d${DELAY}_${REPR_TAG}"

export SYSTEM_PROMPT MODEL_CLASS CHUNK_SIZE EXP_NAME PROJECT EVAL_TAG

echo "==> prompt-controlled SCRIPT leaderboard eval"
echo "    exp_name:      ${EXP_NAME}"
echo "    project:       ${PROJECT}"
echo "    eval_tag:      ${EVAL_TAG}"
echo "    cap/punct:     ${REPR_TAG}"
echo "    delay:         ${DELAY} frames (max trained ${MAX_DELAY})"
echo "    chunk_size:    ${CHUNK_SIZE}"
echo "    system_prompt: ${SYSTEM_PROMPT}"

# Locate eval_leaderboard_slurm.sh (see note in eval_script_baseline.sh).
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
cd "${SLURM_SUBMIT_DIR:-$LAUNCH_DIR}"

# Already inside the sbatch allocation: run the shared pooled-shard body as a
# normal bash script (its own #SBATCH headers are ignored). $1 = EXP_NAME.
exec bash "${LAUNCH_DIR}/eval_leaderboard_slurm.sh" "${EXP_NAME}"
