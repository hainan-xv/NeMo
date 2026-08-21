#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:sslm-lb-promptctl-all-legacy
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
# LEGACY eval for the BEST prior prompt-controlled model, on the ORIGINAL 8-set
# suite -- runs the OLD code + OLD model. This reproduces the 6.93 macro WER
# recorded in chat history for:
#   granary2_chunkcompletion_promptctl_all, chunk 14, delay 4, cap, nopunct,
#   nocorrect  (the "_all" recipe = audio_history_chunks=1; best-at-longest-delay)
#
# It LIVES in the clean repo (launch/) so it syncs to OCI with it, but runs
# NOTHING from the clean repo: it builds the prompt exactly like the previous
# repo's launch/eval_leaderboard_promptctl_all.sh and execs that repo's
# pooled-shard backend (launch/eval_leaderboard_slurm.sh), mounting the previous
# repo's checkout as /code and pointing at the previous model's checkpoints.
#
# ORIGINAL suite = the old backend's default DATASETS: 8 sets, UNCLEANED
# ami/gigaspeech/voxpopuli, INCLUDING tedlium. For the CLEANED 7-set suite that
# is comparable to the clean repo's 6.06, use eval_promptctl_all_legacy_cleaned.sh.
#
# PREREQ: sync the OLD repo to OCI first (its own sync_to_oci.sh), so the old
# code lands at OLD_CODE_DIR and its backend at
# ${OLD_CODE_DIR}/launch/eval_leaderboard_slurm.sh. The old model checkpoints
# must already be on lustre at
#   ${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/${EXP_NAME}/checkpoints/.
#
# Usage (from the clean repo root on OCI):
#   sbatch launch/eval_promptctl_all_legacy.sh                       # BEST: cap nopunct 4 14 nocorrect
#   sbatch launch/eval_promptctl_all_legacy.sh <cap> <punct> <delay> <chunk> <correct>
#
#   <cap>     : cap | nocap        (aliases: true/1/yes , false/0/no)
#   <punct>   : punct | nopunct    (aliases: true/1/yes , false/0/no)
#   <delay>   : integer frames in [0,4]
#   <chunk>   : chunk size (frames), a trained size: 2 4 7 10 14 28
#   <correct> : correct | nocorrect  (self-correction on/off)
#
# Examples (reproduce the recorded delay sweep, chunk 14, cap nopunct nocorrect):
#   for d in 3 4; do sbatch launch/eval_promptctl_all_legacy.sh cap nopunct $d 14 nocorrect; done
#
# Where the OLD code / model live (all overridable via env):
#   OLD_CODE_DIR   OCI checkout of the previous repo, mounted as /code
#                  (default matches NeMo_ord_sync_d146_current/sync_to_oci.sh)
#   EXP_NAME       previous model exp name (default granary2_chunkcompletion_promptctl_all)
#   PROJECT        previous results project dir (default Speechlm79)
#   OUTPUT_PREFIX  results root (default = old backend's own default: nemotron)
#   LEGACY_BACKEND explicit path to the old eval_leaderboard_slurm.sh
#   DATASETS / CACHE_DIR  suite override (used by eval_promptctl_all_legacy_cleaned.sh)
#   SUITE_TAG      extra label appended to EVAL_TAG (e.g. 'cleaned')
#
# Optional env forwarded to the OLD backend (see its header):
#   BACKEND=heh|sslm  BATCH_SIZE=...  RUN_AVERAGING=0  CKPT=/STEP=/USE_LAST=1
#   MAX_EVAL_SAMPLES=...  SAVE_RAW=true|false
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

# Defaults reproduce the BEST recorded config (cap nopunct, delay 4, chunk 14, nocorrect).
CAP_ARG="${1:-cap}"
PUNCT_ARG="${2:-nopunct}"
DELAY_ARG="${3:-4}"
CHUNK_ARG="${4:-14}"
CORRECT_ARG="${5:-nocorrect}"

# Fixed model (best prior prompt-ctl, "_all" recipe, exact_max_delay=4).
EXP_NAME="${EXP_NAME:-granary2_chunkcompletion_promptctl_all}"
MAX_DELAY="${MAX_DELAY:-4}"

# Prompt clauses -- MUST match the recipe (data.dataset.*) so eval is in-distribution.
CHUNK_TEMPLATE="${CHUNK_TEMPLATE:-Process the audio in chunks of {chunk} frames.}"
SC_ON_SUFFIX="${SC_ON_SUFFIX:-If a word you already wrote turns out to be wrong given the new audio, delete it and write the correct word.}"
SC_OFF_SUFFIX="${SC_OFF_SUFFIX:-Do not go back and change words you already wrote.}"

# ---- cap / punct / correct -> bool ----
parse_bool() {
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

MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"
CHUNK_SIZE="$CHUNK"
SAVE_RAW="${SAVE_RAW:-true}"
EVAL_TAG="d${DELAY}_c${CHUNK}_${REPR_TAG}_${CORR_TAG}${SUITE_TAG:+_${SUITE_TAG}}"

# ---- point /code + results at the OLD tree/model ----
OLD_CODE_DIR="${OLD_CODE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/NeMo79}"
export CODE_DIR="$OLD_CODE_DIR"
PROJECT="${PROJECT:-Speechlm79}"

export SYSTEM_PROMPT MODEL_CLASS CHUNK_SIZE EXP_NAME EVAL_TAG SAVE_RAW PROJECT

echo "==> LEGACY promptctl_all leaderboard eval (OLD code + OLD model)"
echo "    old_code_dir:  ${OLD_CODE_DIR}  (mounted as /code)"
echo "    project:       ${PROJECT}"
echo "    exp_name:      ${EXP_NAME}"
echo "    eval_tag:      ${EVAL_TAG}"
echo "    cap/punct:     ${REPR_TAG}"
echo "    delay:         ${DELAY} frames (max trained ${MAX_DELAY})"
echo "    chunk_size:    ${CHUNK_SIZE} (stated in prompt AND used for decode)"
echo "    self-correct:  ${CORR_TAG}"
echo "    suite:         ${SUITE_TAG:-original (old backend default: 8-set, uncleaned + tedlium)}"
[[ -n "${DATASETS:-}" ]] && echo "    datasets:      ${DATASETS}"
[[ -n "${CACHE_DIR:-}" ]] && echo "    cache_dir:     ${CACHE_DIR}"
echo "    system_prompt: ${SYSTEM_PROMPT}"

# Locate the OLD backend under OLD_CODE_DIR. LEGACY_BACKEND overrides.
LEGACY_BACKEND="${LEGACY_BACKEND:-}"
if [[ -z "$LEGACY_BACKEND" ]]; then
    if [[ -f "${OLD_CODE_DIR}/launch/eval_leaderboard_slurm.sh" ]]; then
        LEGACY_BACKEND="${OLD_CODE_DIR}/launch/eval_leaderboard_slurm.sh"
    elif [[ -f "${OLD_CODE_DIR}/oci/eval_leaderboard_slurm.sh" ]]; then
        LEGACY_BACKEND="${OLD_CODE_DIR}/oci/eval_leaderboard_slurm.sh"
    fi
fi
if [[ -z "$LEGACY_BACKEND" || ! -f "$LEGACY_BACKEND" ]]; then
    echo "ERROR: cannot find the OLD backend eval_leaderboard_slurm.sh under ${OLD_CODE_DIR}." >&2
    echo "       Sync the previous repo to OCI first (its sync_to_oci.sh), or set" >&2
    echo "       OLD_CODE_DIR=/path/to/old/checkout (or LEGACY_BACKEND=/full/path)." >&2
    exit 1
fi
echo "    legacy_backend: ${LEGACY_BACKEND}"

# Keep cwd at the submit dir so relative paths (slurm_out/) land where expected.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Already inside the sbatch allocation: run the OLD pooled-shard backend as a
# normal bash script (its own #SBATCH headers are ignored). $1 = EXP_NAME.
exec bash "$LEGACY_BACKEND" "${EXP_NAME}"
