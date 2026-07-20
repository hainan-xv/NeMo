#!/bin/bash
set -e
set -o pipefail

# ============================================================================
# Download a CHAT / RNNT ASR checkpoint from the OCI grid (draco-oci, IAD) and
# evaluate it on the Open ASR Leaderboard (ESB) datasets LOCALLY (this box's GPU).
#
# This is the ASR (EncDecRNNTBPEModel / CHAT) counterpart of
# eval_leaderboard_oci.sh (which targets the SpeechLM StreamingSTTModel). It
# resolves a checkpoint on OCI, rsync's it into $LEADERBOARD_RUN/ckpts/<EXP>/,
# then runs scripts/asr_leaderboard_eval.py (this repo) over the ESB suite.
#
# It PREFERS a `.nemo` (bundles the tokenizer + config, cleanest to restore).
# If no `.nemo` exists on OCI it falls back to the best/last `.ckpt` and loads
# it via EncDecRNNTBPEModel.load_from_checkpoint (needs cfg+tokenizer in ckpt).
#
# Usage:
#   ./eval_leaderboard_chat.sh [--last] [--gpu N] <EXP_NAME> [STEP]
#
# Examples:
#   ./eval_leaderboard_chat.sh oci_chat_nemotron06b_rnnt_g2_ctx70_13_lr5.0_n8
#   QUICK_TEST=1 ./eval_leaderboard_chat.sh oci_chat_nemotron06b_rnnt_g2_ctx70_13_lr5.0_n8
#   ONLY=librispeech ./eval_leaderboard_chat.sh <EXP>
#   MAX_EVAL_SAMPLES=50 ./eval_leaderboard_chat.sh <EXP>
#
# Env overrides:
#   PROJECT              OCI project dir under REMOTE_RESULTS_ROOT (default Speechlm79)
#   REMOTE_HOST          OCI file-transfer/login host
#   REMOTE_USER          OCI user (default hainanx)
#   SSH_KEY              ssh key (default ~/.ssh/draco-rno)
#   REMOTE_RESULTS_ROOT  results root on OCI (nemotron users/hainanx/results)
#   BATCH_SIZE           eval batch size (default 32)
#   QUICK_TEST           1 -> 10 utterances from ami/test only, verbose ref/hyp
#   MAX_EVAL_SAMPLES     cap utterances per dataset (fast iteration)
#   ONLY                 comma-separated dataset filter, e.g. "librispeech,ami"
#   NO_NORMALIZE         1 -> score raw PnC text (do NOT normalize before WER)
#   MAX_SYMBOLS          override greedy max symbols per (chunk) step
#   ATT_CONTEXT_SIZE     override encoder att context, e.g. "[70,13]"
#   CHUNK_SIZE           override CHAT joint chunk_size (full-context models)
#   PREFER_CKPT          1 -> use the .ckpt even if a .nemo exists
#   FORCE_DOWNLOAD       1 -> re-pull even if a local copy exists
#   ASSUME_YES           1 -> auto-"yes" to the re-sync prompt
#   USE_LAST             1 -> use -last.ckpt instead of the best snapshot
#   LEADERBOARD_RUN      base dir for downloaded ckpts/cache/results
#                        (default ~/leaderboard_run); keeps the NeMo repo clean
#   AUDIO_CACHE_DIR      shared 16k-wav cache (default $LEADERBOARD_RUN/cache)
#   REFRESH_CACHE        1 -> ignore audio cache and re-fetch from HuggingFace
# ============================================================================

# ---------- OCI connection ----------
REMOTE_HOST="${REMOTE_HOST:-draco-oci-dc-03.draco-oci-iad.nvidia.com}"
REMOTE_USER="${REMOTE_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/lustre/fsw/portfolios/nemotron/users/hainanx/results}"
PROJECT_CANDIDATES_DEFAULT=("Chat79" "Speechlm79")
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no"

# ---------- Local paths ----------
NEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LEADERBOARD_RUN="${LEADERBOARD_RUN:-$HOME/leaderboard_run}"
LOCAL_CKPT_DIR="${LEADERBOARD_RUN}/ckpts"
RUN_EVAL_PY="${NEMO_ROOT}/scripts/asr_leaderboard_eval.py"
if [ ! -f "$RUN_EVAL_PY" ]; then
    echo "ERROR: eval driver not found: $RUN_EVAL_PY" >&2
    exit 1
fi
# Force the eval to import nemo from THIS checkout (needed for RNNTAttJoint / CHAT).
export PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH:-}"

# HF token for the gated ESB datasets.
if [ -z "${HF_TOKEN:-}" ] && [ -f "${NEMO_ROOT}/.hf_token" ]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${NEMO_ROOT}/.hf_token")"
    export HF_TOKEN
fi

# ---------- Arguments ----------
USE_LAST="${USE_LAST:-0}"
DEVICE_ID="${DEVICE_ID:-0}"
_POS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --last)               USE_LAST=1; shift ;;
        --gpu|--device)       DEVICE_ID="$2"; shift 2 ;;
        --gpu=*|--device=*)   DEVICE_ID="${1#*=}"; shift ;;
        -h|--help)            echo "Usage: $0 [--last] [--gpu N] <EXP_NAME> [STEP]"; exit 0 ;;
        --)                   shift; while [ $# -gt 0 ]; do _POS+=("$1"); shift; done ;;
        -*)                   echo "ERROR: unknown option: $1" >&2; exit 1 ;;
        *)                    _POS+=("$1"); shift ;;
    esac
done
EXP_NAME="${_POS[0]:-}"
STEP="${_POS[1]:-}"
[ -n "${_POS[2]:-}" ] && DEVICE_ID="${_POS[2]}"
[ -n "$EXP_NAME" ] || { echo "Usage: $0 [--last] [--gpu N] <EXP_NAME> [STEP]" >&2; exit 1; }
BATCH_SIZE="${BATCH_SIZE:-32}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
QUICK_TEST="${QUICK_TEST:-0}"
ONLY="${ONLY:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"

# ---------- Resolve project dir on OCI ----------
if [ -n "${PROJECT:-}" ]; then
    PROJECT_CANDIDATES=("$PROJECT")
else
    PROJECT_CANDIDATES=("${PROJECT_CANDIDATES_DEFAULT[@]}")
fi
REMOTE_CKPT_DIR=""
for proj in "${PROJECT_CANDIDATES[@]}"; do
    candidate="${REMOTE_RESULTS_ROOT}/${proj}/${EXP_NAME}/${EXP_NAME}/$EXP_NAME/checkpoints"
    if ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "[ -d '${candidate}' ]" 2>/dev/null; then
        REMOTE_CKPT_DIR="$candidate"
        echo "==> Resolved project: ${proj}"
        break
    fi
done
if [ -z "$REMOTE_CKPT_DIR" ]; then
    echo "ERROR: experiment '${EXP_NAME}' not found under any of:" >&2
    for proj in "${PROJECT_CANDIDATES[@]}"; do
        echo "       ${REMOTE_RESULTS_ROOT}/${proj}/${EXP_NAME}/${EXP_NAME}/checkpoints" >&2
    done
    echo "       Override with PROJECT=<name>." >&2
    exit 1
fi

# ---------- Step 1: resolve the model file on OCI (.nemo preferred) ----------
REMOTE_MODEL_PATH=""
MODEL_EXT=""
if [ "${PREFER_CKPT:-0}" != "1" ] && [ -z "$STEP" ]; then
    # Look for a bundled .nemo (exp_manager always_save_nemo=True).
    NEMO_NAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
        "ls -t ${REMOTE_CKPT_DIR}/*.nemo 2>/dev/null | head -1 | xargs -r basename" || true)
    if [ -n "$NEMO_NAME" ]; then
        REMOTE_MODEL_PATH="${REMOTE_CKPT_DIR}/${NEMO_NAME}"
        MODEL_EXT="nemo"
        MODEL_FILENAME="$NEMO_NAME"
        echo "==> Found bundled .nemo: ${NEMO_NAME}"
    fi
fi

if [ -z "$REMOTE_MODEL_PATH" ]; then
    # Fall back to a .ckpt (best-WER, --last, or explicit STEP).
    if [ -z "$STEP" ]; then
        if [ "${USE_LAST:-0}" = "1" ]; then
            echo "==> USE_LAST=1: finding most recent -last.ckpt on OCI..."
            REMOTE_LIST_CMD="ls -t ${REMOTE_CKPT_DIR}/*-last.ckpt"
        else
            echo "==> Finding best (non '-last') checkpoint on OCI..."
            REMOTE_LIST_CMD="ls -t ${REMOTE_CKPT_DIR}/*.ckpt 2>/dev/null | grep -v -- '-last\.ckpt$'"
        fi
        CKPT_FILENAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "${REMOTE_LIST_CMD} | head -1 | xargs -r basename")
        if [ -z "$CKPT_FILENAME" ] && [ "${USE_LAST:-0}" != "1" ]; then
            echo "    No best-WER checkpoint found; falling back to -last.ckpt..."
            CKPT_FILENAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
                "ls -t ${REMOTE_CKPT_DIR}/*-last.ckpt 2>/dev/null | head -1 | xargs -r basename")
        fi
    else
        CKPT_FILENAME="step=${STEP}.ckpt"
    fi
    if [ -z "$CKPT_FILENAME" ]; then
        echo "ERROR: No .nemo or .ckpt found in ${REMOTE_CKPT_DIR}" >&2
        exit 1
    fi
    REMOTE_MODEL_PATH="${REMOTE_CKPT_DIR}/${CKPT_FILENAME}"
    MODEL_EXT="ckpt"
    MODEL_FILENAME="$CKPT_FILENAME"
    echo "    Using checkpoint: ${CKPT_FILENAME}"
fi

LOCAL_MODEL_PATH="${LOCAL_CKPT_DIR}/${EXP_NAME}/${MODEL_FILENAME}"
mkdir -p "$(dirname "$LOCAL_MODEL_PATH")"

# ---------- Step 1b: cache policy (download / reuse / compare + prompt) ----------
do_download() {
    echo "==> Syncing model from OCI via rsync..."
    echo "    Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_MODEL_PATH}"
    echo "    Local:  ${LOCAL_MODEL_PATH}"
    if ! rsync -vh --times --partial -e "ssh $SSH_OPTS" \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_MODEL_PATH}" "$LOCAL_MODEL_PATH"; then
        [ -f "$LOCAL_MODEL_PATH" ] && [ ! -s "$LOCAL_MODEL_PATH" ] && rm -f "$LOCAL_MODEL_PATH"
        echo "ERROR: rsync failed for ${REMOTE_MODEL_PATH}" >&2
        exit 1
    fi
    echo "==> Download complete ($(du -h "$LOCAL_MODEL_PATH" | cut -f1))"
}

if [ "${FORCE_DOWNLOAD:-0}" = "1" ]; then
    echo "==> FORCE_DOWNLOAD=1: re-pulling model."
    do_download
elif [ ! -f "$LOCAL_MODEL_PATH" ] || [ ! -s "$LOCAL_MODEL_PATH" ]; then
    echo "==> No local copy of ${MODEL_FILENAME}; downloading."
    do_download
else
    REMOTE_STAT=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "stat -c '%s %Y' '${REMOTE_MODEL_PATH}'" 2>/dev/null || true)
    REMOTE_SIZE="${REMOTE_STAT%% *}"
    REMOTE_MTIME="${REMOTE_STAT##* }"
    LOCAL_SIZE=$(stat -c '%s' "$LOCAL_MODEL_PATH" 2>/dev/null || echo "")
    LOCAL_MTIME=$(stat -c '%Y' "$LOCAL_MODEL_PATH" 2>/dev/null || echo "")
    if [ -z "$REMOTE_STAT" ]; then
        echo "WARNING: could not stat remote model; reusing local copy." >&2
    elif [ "$REMOTE_SIZE" = "$LOCAL_SIZE" ] && [ "$REMOTE_MTIME" = "$LOCAL_MTIME" ]; then
        echo "==> Local model matches grid (size=${LOCAL_SIZE}); reusing."
    else
        echo "==> Local model DIFFERS from grid:"
        echo "    local : size=${LOCAL_SIZE:-?} mtime=${LOCAL_MTIME:-?}"
        echo "    grid  : size=${REMOTE_SIZE:-?} mtime=${REMOTE_MTIME:-?}"
        if [ "${ASSUME_YES:-0}" = "1" ]; then
            echo "    ASSUME_YES=1 -> re-syncing."
            do_download
        elif [ -t 0 ]; then
            read -r -p "    Re-sync from grid? [y/N] " _ans
            case "$_ans" in
                [yY]|[yY][eE][sS]) do_download ;;
                *) echo "    Keeping local copy (not re-synced)." ;;
            esac
        else
            echo "    Non-interactive shell and ASSUME_YES!=1 -> keeping local copy." >&2
        fi
    fi
fi

# ---------- Step 2: build the dataset list ----------
DATASETS="ami:test,earnings22:test,gigaspeech:test,librispeech:test.clean,librispeech:test.other,spgispeech:test,tedlium:test,voxpopuli:test"

if [ -n "$ONLY" ]; then
    IFS=',' read -r -a _only <<< "$ONLY"
    IFS=',' read -r -a _all <<< "$DATASETS"
    _filtered=()
    for entry in "${_all[@]}"; do
        ename="${entry%%:*}"
        for want in "${_only[@]}"; do
            [ "$ename" = "$want" ] && _filtered+=("$entry")
        done
    done
    DATASETS=$(IFS=,; echo "${_filtered[*]}")
fi

EXTRA_ARGS=()
if [ "$QUICK_TEST" = "1" ]; then
    DATASETS="ami:test"
    EXTRA_ARGS+=(--max_eval_samples 10 --verbose)
    echo "==> QUICK TEST: 10 utterances from ami/test only (verbose ref/hyp)"
fi
# MANIFEST=<path> reproduces training-time validation locally: evaluate a NeMo
# manifest (e.g. the dev set) directly instead of the ESB sets.
[ -n "${MANIFEST:-}" ] && EXTRA_ARGS+=(--manifest "$MANIFEST")
[ -n "$MAX_EVAL_SAMPLES" ] && EXTRA_ARGS+=(--max_eval_samples "$MAX_EVAL_SAMPLES")
[ "${NO_NORMALIZE:-0}" = "1" ] && EXTRA_ARGS+=(--no_normalize)
[ "${REFRESH_CACHE:-0}" = "1" ] && EXTRA_ARGS+=(--refresh_cache)
[ -n "${MAX_SYMBOLS:-}" ] && EXTRA_ARGS+=(--max_symbols "$MAX_SYMBOLS")
[ -n "${ATT_CONTEXT_SIZE:-}" ] && EXTRA_ARGS+=(--att_context_size "$ATT_CONTEXT_SIZE")
[ -n "${CHUNK_SIZE:-}" ] && EXTRA_ARGS+=(--chunk_size "$CHUNK_SIZE")

AUDIO_CACHE_DIR="${AUDIO_CACHE_DIR:-${LEADERBOARD_RUN}/cache}"
mkdir -p "$AUDIO_CACHE_DIR"
RESULTS_DIR="${LEADERBOARD_RUN}/results/${EXP_NAME}_chat"
mkdir -p "$RESULTS_DIR"

# Per-utterance dumps for the offline per-condition error analysis
# (scripts/analyze_asr_errors.py). Set DUMP_DIR= to disable.
DUMP_DIR="${DUMP_DIR:-${RESULTS_DIR}/dumps}"
[ -n "$DUMP_DIR" ] && EXTRA_ARGS+=(--dump_dir "$DUMP_DIR")

echo ""
echo "==> Running CHAT/RNNT leaderboard evaluation (model loaded once for all sets)"
echo "    Model:       $LOCAL_MODEL_PATH (${MODEL_EXT})"
echo "    Device:      cuda:${DEVICE_ID}   Batch: ${BATCH_SIZE}"
echo "    Datasets:    ${DATASETS}"
echo "    Audio cache: ${AUDIO_CACHE_DIR}"
echo "    Results:     ${RESULTS_DIR}"
echo ""

set +e
python "$RUN_EVAL_PY" \
    --model_path "$LOCAL_MODEL_PATH" \
    --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
    --datasets "$DATASETS" \
    --audio_cache_dir "$AUDIO_CACHE_DIR" \
    --device "$DEVICE_ID" \
    --batch_size "$BATCH_SIZE" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${RESULTS_DIR}/eval_log.txt"
rc=${PIPESTATUS[0]}
set -e
if [ "$rc" -ne 0 ]; then
    echo "WARNING: eval process exited with code ${rc} (see log above)." | tee -a "${RESULTS_DIR}/eval_log.txt"
fi
echo ""
echo "======================================================================"
echo "Evaluation complete. Results in: ${RESULTS_DIR}/eval_log.txt"
echo "======================================================================"
