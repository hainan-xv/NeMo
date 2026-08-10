#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:sslm-lb-script-redecode
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
# Open-ASR-Leaderboard eval for the WINDOWED RE-DECODING SCRIPT model
# (granary2_script_redecode, trained by launch/script_redecode.sh).
#
# Uses the SAME pooled-shard, work-BALANCED backend as every other SpeechLM eval
# (launch/eval_leaderboard_slurm.sh): pool all utterances, global shuffle (fixed
# seed), 8 duration-sorted shards, one decode process per GPU -> wall time
# ~= sum(all)/8. This wrapper only picks the operating point, builds the prompt
# exactly as ScriptSTTDataset renders it at training, then execs that backend on
# the SAME allocation.
#
# The redecode model runs at the BASELINE operating point: delay is FIXED at 3
# frames (a training-time alignment shift, NOT stated in the prompt), so there is
# NO delay knob here. What IS stated in the prompt (the "newer" design, matching
# training render) is the text format (cap x punct, from vary_text_repr) and the
# per-batch chunk size, ending with the "The text history is:" connector:
#   base template  (ScriptSTTDataset train_system_prompt)
#   + format clause (_DEFAULT_FORMAT_CLAUSES[cap/punct])
#   + chunk clause  ("Process the audio in chunks of {chunk} frames. The text history is:")
# The chunk clause number MUST equal the decode CHUNK_SIZE (both set here).
#
# Windowed re-decoding is auto-enabled from the checkpoint's cfg (model.redecode);
# generate() emits the LOCKED transcript (the one WER is scored on) and the eval
# needs no special flags.
#
# Usage (from the NeMo79 repo root on OCI):
#   sbatch launch/eval_script_redecode.sh [cap] [punct] [chunk]
#     cap    : cap | nocap       (default cap)      aliases: true/1/yes, false/0/no
#     punct  : punct | nopunct   (default punct)    aliases: true/1/yes, false/0/no
#     chunk  : encoder frames/chunk (default 14)    (one the model saw: {2,7,10,14})
#
# Examples:
#   sbatch launch/eval_script_redecode.sh                    # cap punct, chunk 14
#   sbatch launch/eval_script_redecode.sh cap   punct   7    # chunk 7
#   sbatch launch/eval_script_redecode.sh nocap nopunct 10   # lowercase/no-punct, chunk 10
#   for c in 2 7 10 14; do sbatch launch/eval_script_redecode.sh cap punct $c; done  # chunk sweep
#   CHUNK_SIZE=7 sbatch launch/eval_script_redecode.sh       # env form of the same knob
#   sbatch launch/eval_script_redecode.sh --quick_run cap punct 14   # smoke test
#
# Flags (may appear anywhere among the positional args):
#   --quick_run[=N]   decode only the first N (default 10) utts of EACH dataset for
#                     a fast smoke test; tags RESULTS_DIR + wandb run with _quick.
#   --interactive     locate the INTERACTIVELY-TRAINED checkpoint, i.e. append
#                     "_interactive" to EXP_NAME (granary2_script_redecode ->
#                     granary2_script_redecode_interactive). Use this to eval the
#                     interactive smoke-training run while the full node=8 run is
#                     still queued. DEFAULT (no flag) locates the non-interactive
#                     (full-queue) EXP_NAME.
#
# NOTE: --interactive (WHICH checkpoint to decode) is INDEPENDENT of
# launch/launch_with_interactive.sh (WHERE the decode job runs). The latter submits
# this eval to the `interactive` partition and tags only the decode job identity
# (WANDB_RUN_NAME/EVAL_TAG += _interactive) while leaving EXP_NAME alone. Combine
# them to decode the interactive-trained checkpoint ON interactive nodes:
#   ./launch_with_interactive.sh eval_script_redecode.sh --interactive
#
# Each run gets its own RESULTS_DIR via EVAL_TAG=<cap>_<punct> + the backend's
# _chunk<chunk> tag + slurm job id; concurrent runs for the same exp share the
# averaged ckpt / HF convert (mkdir-locked in the backend).
#
# Optional env:
#   CAP / PUNCT / CHUNK_SIZE          defaults for the three positional args above.
#   INTERACTIVE=1                     env form of the --interactive flag.
# Optional env (forwarded to eval_leaderboard_slurm.sh):
#   BACKEND=heh|sslm   BATCH_SIZE=...   PROJECT=SpeechlmRefactored (default)
#   RUN_AVERAGING=1 (default) / USE_LAST=1 / STEP=n / CKPT=path
#   DATASETS="..."   MAX_EVAL_SAMPLES=n (smoke test)   SHUFFLE_SEED=1234
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

# --- Flags (--quick_run, --interactive) -----------------------------------
# Strip flags out of the argument list FIRST so the positional cap/punct/chunk
# parsing below is unaffected by flag position. --quick_run caps decoding to the
# first N (default 10) utts/dataset via MAX_EVAL_SAMPLES; --interactive selects
# the interactively-trained checkpoint (EXP_NAME += _interactive).
QUICK_RUN=0
QUICK_N=10
INTERACTIVE="${INTERACTIVE:-0}"
POSITIONAL=()
for _arg in "$@"; do
    case "$_arg" in
        --quick_run|--quick-run) QUICK_RUN=1 ;;
        --quick_run=*|--quick-run=*) QUICK_RUN=1; QUICK_N="${_arg#*=}" ;;
        --interactive) INTERACTIVE=1 ;;
        *) POSITIONAL+=("$_arg") ;;
    esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"
QUICK_SUFFIX=""
if (( QUICK_RUN )); then
    export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-$QUICK_N}"
    QUICK_SUFFIX="_quick"
fi

# --- Operating point (positional, all optional with defaults) ---
# Env vars act as the defaults so `CHUNK_SIZE=7 sbatch ...` works like it does for
# eval_script_baseline.sh; an explicit positional arg still wins. No delay arg:
# the redecode model's delay is FIXED at 3 (baked), not prompt-controlled.
CAP_ARG="${1:-${CAP:-cap}}"
PUNCT_ARG="${2:-${PUNCT:-punct}}"
CHUNK_ARG="${3:-${CHUNK_SIZE:-14}}"

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
#   train base (train_system_prompt)
#   + " " + format clause (_append_format_clause, vary_text_repr)
#   + " " + "Process the audio in chunks of <chunk> frames. The text history is:"
#           (chunk_size_prompt_template, with the connector folded on the end)
# in streaming_stt_granary2_lora_script_redecode.yaml. NO delay clause (fixed 3).
SYSTEM_PROMPT="You are doing streaming speech recognition. Given the transcript so far and the representation of the next audio chunk, output the words spoken in that chunk. ${FORMAT_CLAUSE} Process the audio in chunks of ${CHUNK_SIZE} frames. The text history is:"

# --- Model + run identity ---
# Default locates the NON-interactive (full node=8) run. --interactive (or
# INTERACTIVE=1) appends _interactive to locate the interactively-trained tree.
EXP_NAME="${EXP_NAME:-granary2_script_redecode}"
if (( INTERACTIVE )) && [[ "$EXP_NAME" != *_interactive ]]; then
    EXP_NAME="${EXP_NAME}_interactive"
fi
# The training project (launch/script_redecode.sh: PROJECT_NAME=SpeechlmRefactored).
# eval_leaderboard_slurm.sh resolves checkpoints under results/<PROJECT>/<EXP>/<EXP>/,
# so this MUST match how the model was trained.
PROJECT="${PROJECT:-SpeechlmRefactored}"
MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"
# Distinguishes concurrent redecode evals in RESULTS_DIR (backend adds _chunk<n>).
EVAL_TAG="${REPR_TAG}${QUICK_SUFFIX}"
# wandb run name encodes the decode config (delay baked). Logged to WANDB_EVAL_PROJECT.
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXP_NAME}_${REPR_TAG}_chunk${CHUNK_SIZE}${QUICK_SUFFIX}}"

export SYSTEM_PROMPT MODEL_CLASS CHUNK_SIZE EXP_NAME PROJECT EVAL_TAG WANDB_RUN_NAME

echo "==> windowed re-decoding SCRIPT leaderboard eval"
echo "    exp_name:      ${EXP_NAME}$( (( INTERACTIVE )) && echo '  (interactive-trained checkpoint)')"
echo "    project:       ${PROJECT}"
echo "    eval_tag:      ${EVAL_TAG}"
echo "    cap/punct:     ${REPR_TAG}"
echo "    delay:         3 frames (FIXED / baked -- not a knob)"
echo "    chunk_size:    ${CHUNK_SIZE}"
(( QUICK_RUN )) && echo "    quick_run:     first ${QUICK_N} utts/dataset (MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES})"
echo "    wandb_run:     ${WANDB_RUN_NAME} (+_<launch-time> appended by backend)"
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
