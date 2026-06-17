#!/bin/bash
set -e
set -o pipefail

# ============================================================================
# Download/resolve an ASR checkpoint from ORD, then run likelihood-selected
# time-warp eval over the Open-ASR-Leaderboard datasets.
#
# This is the likelihood-selector sibling of eval_asr_ord.sh. It reuses the same
# local checkpoint cache, ORD layout, optional checkpoint averaging, and tokenizer
# download logic, then delegates the actual leaderboard loop to:
#   ./eval_likelihood_timewarp.sh
#
# Usage:
#   ./eval_likelihood_timewarp_ord.sh <EXP_NAME> [STEP] [DEVICE_ID]
#     STEP=last (or USE_LAST=1) -> use the rolling -last.ckpt
#     no STEP                  -> prefer a .nemo, else best non-last .ckpt
#
#   MODEL=/abs/path/model.nemo ./eval_likelihood_timewarp_ord.sh
#
# Common env overrides:
#   FACTORS=0.9,1.0,1.1
#   METHOD=time_stretch|speed
#   SCORE_NORM=none|token|word|char
#   BATCH_SIZE=32
#   ONLY=ami_test,librispeech_test.clean
#   MAX_EVAL_SAMPLES=100
#   RUN_AVERAGING=1
# ============================================================================

# ---------- ORD connection ----------
REMOTE_HOST="${REMOTE_HOST:-cs-oci-ord-login-01.nvidia.com}"
REMOTE_USER="${REMOTE_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/lustre/fsw/portfolios/llmservice/users/hainanx/results}"
PROJECT_CANDIDATES_DEFAULT=("Streaming_SLM_ORD3" "Streaming_SLM_ord2")
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no"

# ---------- Local paths ----------
NEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOCAL_CKPT_DIR="${NEMO_ROOT}/checkpoints"
LOCAL_DRIVER="${NEMO_ROOT}/eval_likelihood_timewarp.sh"
AVG_SCRIPT="${NEMO_ROOT}/scripts/checkpoint_averaging/checkpoint_averaging.py"
export PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${NEMO_ROOT}/.cache/flashinfer}"

if [ ! -x "$LOCAL_DRIVER" ]; then
    echo "ERROR: cannot execute likelihood driver at ${LOCAL_DRIVER}" >&2
    exit 1
fi

# ---------- Arguments ----------
EXP_NAME="${1:-}"
STEP="${2:-}"
DEVICE_ID="${3:-${DEVICE:-0}}"

if [ -n "$STEP" ]; then
    _step_lc=$(echo "$STEP" | tr '[:upper:]' '[:lower:]')
    if [ "$_step_lc" = "last" ]; then
        echo "==> STEP='${STEP}' -> routing through USE_LAST=1"
        USE_LAST=1
        STEP=""
    fi
fi

LOCAL_CKPT_PATH=""
LOCAL_TOK_DIR=""
REMOTE_EXP_DIR=""

if [ -n "${MODEL:-}" ]; then
    LOCAL_CKPT_PATH="$MODEL"
    [ -z "$EXP_NAME" ] && EXP_NAME="$(basename "$(dirname "$MODEL")")"
    echo "==> Using explicit local checkpoint: $LOCAL_CKPT_PATH"
    [ -n "${TOKENIZER_DIR:-}" ] && LOCAL_TOK_DIR="$TOKENIZER_DIR"
elif [ -n "$EXP_NAME" ] && [ -e "$EXP_NAME" ]; then
    LOCAL_CKPT_PATH="$EXP_NAME"
    EXP_NAME="$(basename "$(dirname "$LOCAL_CKPT_PATH")")"
    echo "==> Using positional local checkpoint: $LOCAL_CKPT_PATH"
    [ -n "${TOKENIZER_DIR:-}" ] && LOCAL_TOK_DIR="$TOKENIZER_DIR"
else
    if [ -z "$EXP_NAME" ]; then
        echo "Usage: $0 <EXP_NAME> [STEP] [DEVICE_ID]   (or MODEL=/abs/path $0)" >&2
        exit 1
    fi

    # Resolve which project dir hosts this experiment.
    if [ -n "${PROJECT:-}" ]; then
        PROJECT_CANDIDATES=("$PROJECT")
    else
        PROJECT_CANDIDATES=("${PROJECT_CANDIDATES_DEFAULT[@]}")
    fi
    REMOTE_CKPT_DIR=""
    FOUND_EXP_WITHOUT_CKPT=""
    for proj in "${PROJECT_CANDIDATES[@]}"; do
        candidate="${REMOTE_RESULTS_ROOT}/${proj}/${EXP_NAME}/${EXP_NAME}/checkpoints"
        if ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "[ -d '${candidate}' ]" 2>/dev/null; then
            REMOTE_CKPT_DIR="$candidate"
            REMOTE_EXP_DIR="${REMOTE_RESULTS_ROOT}/${proj}/${EXP_NAME}"
            echo "==> Resolved project: ${proj}"
            break
        fi
        exp_candidate="${REMOTE_RESULTS_ROOT}/${proj}/${EXP_NAME}"
        if ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "[ -d '${exp_candidate}' ]" 2>/dev/null; then
            FOUND_EXP_WITHOUT_CKPT="$exp_candidate"
        fi
    done
    if [ -z "$REMOTE_CKPT_DIR" ]; then
        if [ -n "$FOUND_EXP_WITHOUT_CKPT" ]; then
            echo "ERROR: experiment directory exists, but expected checkpoint directory is missing:" >&2
            echo "       ${FOUND_EXP_WITHOUT_CKPT}/${EXP_NAME}/checkpoints" >&2
            echo "       Current remote layout:" >&2
            ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
                "ls -la '${FOUND_EXP_WITHOUT_CKPT}' '${FOUND_EXP_WITHOUT_CKPT}/${EXP_NAME}' 2>&1" >&2 || true
            echo "       This usually means training failed before checkpointing, used a different exp_manager.name, or never reached the first save/validation point." >&2
            exit 1
        fi
        echo "ERROR: experiment '${EXP_NAME}' not found under any of:" >&2
        for proj in "${PROJECT_CANDIDATES[@]}"; do
            echo "       ${REMOTE_RESULTS_ROOT}/${proj}/${EXP_NAME}/${EXP_NAME}/checkpoints" >&2
        done
        echo "       Override with PROJECT=<name>." >&2
        exit 1
    fi

    if [ "${RUN_AVERAGING:-0}" = "1" ]; then
        [ -n "$STEP" ] && echo "WARNING: RUN_AVERAGING=1 ignores explicit STEP=$STEP" >&2 && STEP=""
        AVG_DIR="${LOCAL_CKPT_DIR}/${EXP_NAME}/avg_inputs"

        if [ "${REUSE_AVG:-0}" = "1" ] && [ "${FORCE_AVERAGE:-0}" != "1" ] && [ "${FORCE_DOWNLOAD:-0}" != "1" ]; then
            existing_avg=$(ls -t "${AVG_DIR}"/*-averaged.nemo 2>/dev/null | head -1)
            if [ -n "$existing_avg" ]; then
                LOCAL_CKPT_PATH="$existing_avg"
                echo "==> REUSE_AVG=1: reusing existing averaged model: $LOCAL_CKPT_PATH"
            else
                echo "ERROR: REUSE_AVG=1 but no *-averaged.nemo in ${AVG_DIR}" >&2
                exit 1
            fi
        fi

        if [ -z "$LOCAL_CKPT_PATH" ]; then
            if [ ! -f "$AVG_SCRIPT" ]; then
                echo "ERROR: NeMo averaging script not found at $AVG_SCRIPT" >&2
                exit 1
            fi
            mkdir -p "$AVG_DIR"

            NEMO_FNAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
                "ls -t ${REMOTE_CKPT_DIR}/*.nemo 2>/dev/null | grep -v -- '-averaged' | head -1 | xargs -r basename")
            if [ -z "$NEMO_FNAME" ]; then
                echo "ERROR: RUN_AVERAGING needs a .nemo export in ${REMOTE_CKPT_DIR}" >&2
                exit 1
            fi

            LOCAL_NEMO="${AVG_DIR}/${NEMO_FNAME}"
            if [ -f "$LOCAL_NEMO" ] && [ "${FORCE_DOWNLOAD:-0}" != "1" ]; then
                echo "==> Cached base model: ${NEMO_FNAME}"
            else
                echo "==> Downloading base model ${NEMO_FNAME}..."
                scp $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_DIR}/${NEMO_FNAME}" "$LOCAL_NEMO"
            fi

            echo "==> Listing non '-last' checkpoints on ORD..."
            REMOTE_AVG_LIST_CMD="ls ${REMOTE_CKPT_DIR}/*.ckpt 2>/dev/null | grep -v -- '-last\\.ckpt$' | xargs -r -n1 basename"
            REMOTE_CKPT_FILES=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "$REMOTE_AVG_LIST_CMD")
            if [ -z "$REMOTE_CKPT_FILES" ]; then
                echo "ERROR: No non '-last' checkpoints found in ${REMOTE_CKPT_DIR}" >&2
                exit 1
            fi
            NUM_CKPTS=$(echo "$REMOTE_CKPT_FILES" | wc -l)
            echo "    Found ${NUM_CKPTS} checkpoint(s) to average:"
            echo "$REMOTE_CKPT_FILES" | sed 's/^/      - /'

            NEWEST_INPUT="$LOCAL_NEMO"
            while IFS= read -r fname; do
                [ -z "$fname" ] && continue
                local_path="${AVG_DIR}/${fname}"
                if [ -f "$local_path" ] && [ "${FORCE_DOWNLOAD:-0}" != "1" ]; then
                    echo "==> Cached: ${fname} ($(du -h "$local_path" | cut -f1))"
                else
                    echo "==> Downloading ${fname}..."
                    scp $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_DIR}/${fname}" "$local_path"
                fi
                [ "$local_path" -nt "$NEWEST_INPUT" ] && NEWEST_INPUT="$local_path"
            done <<< "$REMOTE_CKPT_FILES"

            LOCAL_CKPT_PATH="${AVG_DIR}/${NEMO_FNAME%.nemo}-averaged.nemo"
            if [ -f "$LOCAL_CKPT_PATH" ] && [ "${FORCE_AVERAGE:-0}" != "1" ] && [ ! "$NEWEST_INPUT" -nt "$LOCAL_CKPT_PATH" ]; then
                echo "==> Using cached averaged model: $LOCAL_CKPT_PATH (FORCE_AVERAGE=1 to recompute)"
            else
                echo "==> Averaging ${NUM_CKPTS} checkpoint(s) via checkpoint_averaging.py ..."
                rm -f "$LOCAL_CKPT_PATH"
                PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH}" python "$AVG_SCRIPT" "$LOCAL_NEMO"
                if [ ! -f "$LOCAL_CKPT_PATH" ]; then
                    echo "ERROR: averaging did not produce $LOCAL_CKPT_PATH" >&2
                    exit 1
                fi
                echo "==> Average complete ($(du -h "$LOCAL_CKPT_PATH" | cut -f1))"
            fi
        fi
    else
        if [ -n "$STEP" ]; then
            echo "==> Looking up STEP=${STEP} checkpoint on ORD..."
            CKPT_FILENAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
                "ls ${REMOTE_CKPT_DIR}/*step=${STEP}*.ckpt 2>/dev/null | grep -v -- '-last' | head -1 | xargs -r basename")
            [ -z "$CKPT_FILENAME" ] && CKPT_FILENAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
                "ls ${REMOTE_CKPT_DIR}/*step=${STEP}*.ckpt 2>/dev/null | head -1 | xargs -r basename")
        elif [ "${USE_LAST:-0}" = "1" ]; then
            echo "==> Finding most recent -last.ckpt on ORD..."
            CKPT_FILENAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
                "ls -t ${REMOTE_CKPT_DIR}/*-last.ckpt 2>/dev/null | head -1 | xargs -r basename")
        else
            CKPT_FILENAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
                "ls -t ${REMOTE_CKPT_DIR}/*.nemo 2>/dev/null | head -1 | xargs -r basename")
            if [ -z "$CKPT_FILENAME" ]; then
                echo "==> No .nemo found, picking best (non '-last') .ckpt by val_wer..."
                REMOTE_PICK_CMD="\
                    files=\$(ls ${REMOTE_CKPT_DIR}/*.ckpt 2>/dev/null | grep -v -- '-last\\.ckpt$'); \
                    wer_files=\$(echo \"\$files\" | grep -E 'val_wer=[0-9]+\\.[0-9]+' || true); \
                    if [ -n \"\$wer_files\" ]; then \
                        echo \"\$wer_files\" | awk -F'val_wer=' '{ print \$2, \$0 }' | sort -k1,1n | head -1 | awk '{ print \$2 }'; \
                    else \
                        echo \"\$files\" | head -1; \
                    fi"
                CKPT_FILENAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "${REMOTE_PICK_CMD} | xargs -r basename")
            fi
        fi
        if [ -z "$CKPT_FILENAME" ]; then
            echo "ERROR: No matching checkpoint in ${REMOTE_CKPT_DIR} (STEP='${STEP}', USE_LAST='${USE_LAST:-0}')." >&2
            exit 1
        fi
        echo "    Found: ${CKPT_FILENAME}"

        REMOTE_CKPT_PATH="${REMOTE_CKPT_DIR}/${CKPT_FILENAME}"
        LOCAL_CKPT_PATH="${LOCAL_CKPT_DIR}/${EXP_NAME}/${CKPT_FILENAME}"
        if [ -f "$LOCAL_CKPT_PATH" ] && [ "${FORCE_DOWNLOAD:-0}" != "1" ]; then
            echo "==> Using cached local checkpoint (FORCE_DOWNLOAD=1 to refresh): $LOCAL_CKPT_PATH"
        else
            echo "==> Downloading checkpoint from ORD..."
            echo "    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_PATH}"
            mkdir -p "$(dirname "$LOCAL_CKPT_PATH")"
            scp $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_PATH}" "$LOCAL_CKPT_PATH"
            echo "==> Download complete ($(du -h "$LOCAL_CKPT_PATH" | cut -f1))"
        fi
    fi

    if [[ "$LOCAL_CKPT_PATH" == *.ckpt ]]; then
        REMOTE_TOK_DIR="${REMOTE_EXP_DIR}/tokenizer"
        if ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "[ -d '${REMOTE_TOK_DIR}' ]" 2>/dev/null; then
            LOCAL_TOK_DIR="${LOCAL_CKPT_DIR}/${EXP_NAME}/tokenizer"
            if [ -d "$LOCAL_TOK_DIR" ] && [ "${FORCE_DOWNLOAD:-0}" != "1" ]; then
                echo "==> Using cached tokenizer dir: $LOCAL_TOK_DIR"
            else
                echo "==> Downloading tokenizer dir from ORD..."
                rm -rf "$LOCAL_TOK_DIR"
                mkdir -p "$(dirname "$LOCAL_TOK_DIR")"
                scp -r $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TOK_DIR}" "$(dirname "$LOCAL_TOK_DIR")/"
            fi
        else
            echo "WARNING: .ckpt selected but no tokenizer dir at ${REMOTE_TOK_DIR}." >&2
        fi
    fi
fi

if [ ! -e "$LOCAL_CKPT_PATH" ]; then
    echo "ERROR: checkpoint not found at $LOCAL_CKPT_PATH" >&2
    exit 1
fi

echo "==> Starting likelihood time-warp leaderboard eval..."
echo "    checkpoint: $LOCAL_CKPT_PATH"
[ -n "$LOCAL_TOK_DIR" ] && echo "    tokenizer : $LOCAL_TOK_DIR"

DEVICE="$DEVICE_ID" MODEL="$LOCAL_CKPT_PATH" TOKENIZER_DIR="$LOCAL_TOK_DIR" "$LOCAL_DRIVER"
