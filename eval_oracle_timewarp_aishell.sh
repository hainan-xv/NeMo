#!/bin/bash
set -e
set -o pipefail

# ============================================================================
# Oracle time-warp ("speed cheat") CER eval for a Mandarin ASR checkpoint over
# the local AISHELL-2 test manifests.
#
# This is the time-warp sibling of eval_aishell.sh: it resolves + downloads the
# checkpoint from RNO exactly like eval_aishell.sh (checkpoint averaging ON by
# default), then for each test set decodes several time-warped copies of every
# utterance and reports, per set:
#   * per-factor corpus CER (each warp used alone),
#   * the best single fixed warp, and
#   * the ORACLE best-of-N CER (cheating: per-utterance pick the lowest-CER warp).
# Scoring goes through oracle_timewarp_aishell.py, which reuses eval_aishell_cer.py
# so the x1.0 (baseline) column matches a normal eval_aishell.sh run.
#
# Usage:
#   ./eval_oracle_timewarp_aishell.sh <EXP_NAME> [STEP] [DEVICE_ID]
#   MODEL=/abs/path/model.nemo ./eval_oracle_timewarp_aishell.sh
#
# Env overrides (superset of eval_aishell.sh):
#   FACTORS        warp factors (default 0.9,1.0,1.1; 1.0 auto-added if absent)
#   METHOD         time_stretch | speed   (default time_stretch)
#   RUN_AVERAGING  1 (default) -> average non '-last' ckpts into *-averaged.nemo
#   FORCE_AVERAGE / REUSE_AVG / FORCE_DOWNLOAD   (as in eval_aishell.sh)
#   PROJECT / REMOTE_* / REMOTE_RESULTS_ROOT     RNO checkpoint location
#   MANIFEST_DIR / AUDIO_SRC_PREFIX / AUDIO_DST_PREFIX   dataset paths
#   SETS / ONLY    test sets (default "test_android test_ios test_mic")
#   BATCH_SIZE     dataloader batch size (default 32)
#   MAX_EVAL_SAMPLES   cap samples per set (fast iteration)
#   KEEP_SPACES    1 -> do NOT collapse whitespace before CER
#   OUT_DIR        per-utterance reports + log (default oracle_results_aishell/)
# ============================================================================

# ---------- RNO connection ----------
REMOTE_HOST="${REMOTE_HOST:-draco-rno-login}"
REMOTE_USER="${REMOTE_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/gpfs/fs1/projects/ent_aiapps/users/hainanx/results}"
PROJECT_CANDIDATES_DEFAULT=("Mandarin_202606_enc512l18_enginit")
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no"

# ---------- Local paths ----------
NEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOCAL_CKPT_DIR="${NEMO_ROOT}/checkpoints"
EVAL_PY="${NEMO_ROOT}/oracle_timewarp_aishell.py"
AVG_SCRIPT="${NEMO_ROOT}/scripts/checkpoint_averaging/checkpoint_averaging.py"
RUN_AVERAGING="${RUN_AVERAGING:-1}"
export PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba}"
if [ ! -f "$EVAL_PY" ]; then
    echo "ERROR: cannot find oracle driver at ${EVAL_PY}" >&2
    exit 1
fi

# ---------- Dataset ----------
MANIFEST_DIR="${MANIFEST_DIR:-/home/hainanx/Workplace/data/aishell_eval/aishell/manifests}"
AUDIO_SRC_PREFIX="${AUDIO_SRC_PREFIX:-/data/mandarin/aishell2/evaluation/aishell}"
AUDIO_DST_PREFIX="${AUDIO_DST_PREFIX:-$(dirname "$MANIFEST_DIR")}"

if [ -n "${ONLY:-}" ]; then
    SETS="${ONLY//,/ }"
fi
SETS="${SETS:-test_android test_ios test_mic}"
SETS="${SETS//,/ }"

# ---------- Time-warp config ----------
FACTORS="${FACTORS:-0.9,1.0,1.1}"
METHOD="${METHOD:-time_stretch}"
OUT_DIR="${OUT_DIR:-${NEMO_ROOT}/oracle_results_aishell}"

# ---------- Arguments ----------
EXP_NAME="${1:-}"
STEP="${2:-}"
DEVICE_ID="${3:-0}"

if [ -n "$STEP" ]; then
    _step_lc=$(echo "$STEP" | tr '[:upper:]' '[:lower:]')
    if [ "$_step_lc" = "last" ]; then
        echo "==> STEP='${STEP}' -> routing through USE_LAST=1"
        USE_LAST=1
        STEP=""
    fi
fi

BATCH_SIZE="${BATCH_SIZE:-32}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"

# ============================================================================
# Resolve / download the checkpoint  (ported verbatim from eval_aishell.sh)
# ============================================================================
LOCAL_CKPT_PATH=""

if [ -n "${MODEL:-}" ]; then
    LOCAL_CKPT_PATH="$MODEL"
    [ -z "$EXP_NAME" ] && EXP_NAME="$(basename "$(dirname "$MODEL")")"
    echo "==> Using explicit local checkpoint: $LOCAL_CKPT_PATH"
else
    if [ -z "$EXP_NAME" ]; then
        echo "Usage: $0 <EXP_NAME> [STEP] [DEVICE_ID]   (or MODEL=/abs/path $0)" >&2
        exit 1
    fi

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
            exit 1
        fi
        echo "ERROR: experiment '${EXP_NAME}' not found under any of:" >&2
        for proj in "${PROJECT_CANDIDATES[@]}"; do
            echo "       ${REMOTE_RESULTS_ROOT}/${proj}/${EXP_NAME}/${EXP_NAME}/checkpoints" >&2
        done
        echo "       Override with PROJECT=<name> or REMOTE_RESULTS_ROOT=<path>." >&2
        exit 1
    fi

    if [ "${RUN_AVERAGING}" = "1" ]; then
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
                echo "       Re-run with RUN_AVERAGING=0 to eval a single .ckpt instead." >&2
                exit 1
            fi
            LOCAL_NEMO="${AVG_DIR}/${NEMO_FNAME}"
            if [ -f "$LOCAL_NEMO" ] && [ "${FORCE_DOWNLOAD:-0}" != "1" ]; then
                echo "==> Cached base model: ${NEMO_FNAME}"
            else
                echo "==> Downloading base model ${NEMO_FNAME}..."
                scp $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_DIR}/${NEMO_FNAME}" "$LOCAL_NEMO" || exit 1
            fi

            echo "==> Listing non '-last' checkpoints on RNO..."
            REMOTE_AVG_LIST_CMD="ls ${REMOTE_CKPT_DIR}/*.ckpt 2>/dev/null | grep -v -- '-last\\.ckpt$' | xargs -r -n1 basename"
            REMOTE_CKPT_FILES=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "$REMOTE_AVG_LIST_CMD")
            if [ -z "$REMOTE_CKPT_FILES" ]; then
                echo "ERROR: No non '-last' checkpoints found in ${REMOTE_CKPT_DIR}" >&2
                echo "       Re-run with RUN_AVERAGING=0 to eval the .nemo export directly." >&2
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
                    if ! scp $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_DIR}/${fname}" "$local_path"; then
                        echo "ERROR: scp failed for ${fname}" >&2
                        [ -f "$local_path" ] && [ ! -s "$local_path" ] && rm -f "$local_path"
                        exit 1
                    fi
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
    # Pick a single checkpoint filename on the remote (prefer .nemo).
    if [ -n "$STEP" ]; then
        echo "==> Looking up STEP=${STEP} checkpoint on RNO..."
        CKPT_FILENAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
            "ls ${REMOTE_CKPT_DIR}/*step=${STEP}*.ckpt 2>/dev/null | grep -v -- '-last' | head -1 | xargs -r basename")
        [ -z "$CKPT_FILENAME" ] && CKPT_FILENAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
            "ls ${REMOTE_CKPT_DIR}/*step=${STEP}*.ckpt 2>/dev/null | head -1 | xargs -r basename")
    elif [ "${USE_LAST:-0}" = "1" ]; then
        echo "==> Finding most recent -last.ckpt on RNO..."
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
        echo "==> Downloading checkpoint from RNO..."
        echo "    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_PATH}"
        mkdir -p "$(dirname "$LOCAL_CKPT_PATH")"
        if ! scp $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_PATH}" "$LOCAL_CKPT_PATH"; then
            echo "ERROR: scp failed for ${REMOTE_CKPT_PATH}." >&2
            [ -f "$LOCAL_CKPT_PATH" ] && [ ! -s "$LOCAL_CKPT_PATH" ] && rm -f "$LOCAL_CKPT_PATH"
            exit 1
        fi
        echo "==> Download complete ($(du -h "$LOCAL_CKPT_PATH" | cut -f1))"
    fi
    fi  # end RUN_AVERAGING if/else
fi

if [ ! -e "$LOCAL_CKPT_PATH" ]; then
    echo "ERROR: checkpoint not found at $LOCAL_CKPT_PATH" >&2
    exit 1
fi

# ============================================================================
# Run the oracle time-warp eval over the AISHELL test sets
# ============================================================================
EXTRA_ARGS=()
[ -n "$AUDIO_SRC_PREFIX" ] && EXTRA_ARGS+=(--audio_src_prefix "$AUDIO_SRC_PREFIX")
[ -n "$AUDIO_DST_PREFIX" ] && EXTRA_ARGS+=(--audio_dst_prefix "$AUDIO_DST_PREFIX")
[ -n "$MAX_EVAL_SAMPLES" ] && EXTRA_ARGS+=(--max_eval_samples "$MAX_EVAL_SAMPLES")
[ "${KEEP_SPACES:-0}" = "1" ] && EXTRA_ARGS+=(--keep_spaces)

mkdir -p "$OUT_DIR"
EVAL_LOG="${OUT_DIR}/oracle_timewarp_aishell_$(basename "${LOCAL_CKPT_PATH}").log"
: > "$EVAL_LOG"
echo "==> factors=${FACTORS}  method=${METHOD}  sets='${SETS}'  device=${DEVICE_ID}"
echo "==> Logging to $EVAL_LOG"

IFS=',' read -r -a FACTOR_LIST <<< "$FACTORS"
# Make sure 1.0 is represented in the table columns (python auto-adds it).
_has_one=0
for f in "${FACTOR_LIST[@]}"; do [ "$f" = "1.0" ] && _has_one=1; done
if [ "$_has_one" -eq 0 ]; then
    FACTOR_LIST=("1.0" "${FACTOR_LIST[@]}")
fi

declare -A BASELINE BESTFIX ORACLE SCORED PICKS FACTOR_CERS
for set_name in $SETS; do
    manifest="${MANIFEST_DIR}/${set_name}.json"
    if [ ! -f "$manifest" ]; then
        echo "  ${set_name}: manifest not found at ${manifest} -- skipping"
        BASELINE[$set_name]="NOFILE"; BESTFIX[$set_name]="NOFILE"; ORACLE[$set_name]="NOFILE"; SCORED[$set_name]="?"
        continue
    fi

    echo "running oracle time-warp on ${set_name}..."
    {
        echo "==================== ${set_name} ===================="
        echo "[$(date '+%H:%M:%S')] Evaluating: ${set_name} (${manifest})"
    } >> "$EVAL_LOG"

    run_log=$(mktemp)
    set +e
    python "$EVAL_PY" \
        --model "$LOCAL_CKPT_PATH" \
        --manifest "$manifest" \
        --set_name "$set_name" \
        --factors "$FACTORS" \
        --method "$METHOD" \
        --device "$DEVICE_ID" \
        --batch_size "$BATCH_SIZE" \
        --output "${OUT_DIR}/oracle_${set_name}.jsonl" \
        "${EXTRA_ARGS[@]}" \
        > "$run_log" 2>&1
    rc=$?
    set -e
    cat "$run_log" >> "$EVAL_LOG"

    if [ "$rc" -ne 0 ]; then
        echo "  ${set_name} FAILED (exit ${rc}) -- see ${EVAL_LOG}"
        BASELINE[$set_name]="FAIL"; BESTFIX[$set_name]="FAIL"; ORACLE[$set_name]="FAIL"; SCORED[$set_name]="?"; PICKS[$set_name]="?"; FACTOR_CERS[$set_name]="?"
    else
        line=$(grep '^ORACLE_SUMMARY' "$run_log" | tail -1)
        SCORED[$set_name]=$(echo "$line" | grep -oE 'scored=[0-9]+' | cut -d= -f2)
        BASELINE[$set_name]=$(echo "$line" | grep -oE 'baseline=[0-9.]+' | cut -d= -f2)
        BESTFIX[$set_name]=$(echo "$line" | grep -oE 'best_fixed=[0-9.]+' | cut -d= -f2)
        ORACLE[$set_name]=$(echo "$line" | grep -oE 'oracle=[0-9.]+' | cut -d= -f2)
        FACTOR_CERS[$set_name]=$(echo "$line" | tr ' ' '\n' | awk -F= '$1=="factor_cers"{print $2}')
        PICKS[$set_name]=$(echo "$line" | tr ' ' '\n' | awk -F= '$1=="picks"{print $2}')
        echo "  ${set_name}: factor_CERs=${FACTOR_CERS[$set_name]} best_fixed=${BESTFIX[$set_name]} oracle=${ORACLE[$set_name]} (scored ${SCORED[$set_name]}; picks ${PICKS[$set_name]})"
    fi
    rm -f "$run_log"
done

# ---------- Summary ----------
echo ""
echo "==================== AISHELL oracle time-warp CER summary ===================="
echo "exp    : ${EXP_NAME}"
echo "ckpt   : $(basename "${LOCAL_CKPT_PATH}")"
echo "factors: ${FACTORS}   method: ${METHOD}   metric: CER (%)"
printf "  %-22s %8s" "set" "scored"
for f in "${FACTOR_LIST[@]}"; do printf " %8s" "x${f}"; done
printf " %8s %8s  %s\n" "BESTFIX" "ORACLE" "oracle_pick_%"

declare -A FACTOR_SUMS
sb=0; so=0; cnt=0
for set_name in $SETS; do
    printf "  %-22s %8s" "$set_name" "${SCORED[$set_name]:-?}"
    for f in "${FACTOR_LIST[@]}"; do
        fc=$(echo "${FACTOR_CERS[$set_name]:-}" | tr ',' '\n' | awk -F: -v k="x${f}" '$1==k{print $2}')
        printf " %8s" "${fc:-NA}"
    done
    printf " %8s %8s  %s\n" "${BESTFIX[$set_name]:-NA}" "${ORACLE[$set_name]:-NA}" "${PICKS[$set_name]:-?}"
    if [[ "${BASELINE[$set_name]}" =~ ^[0-9.]+$ ]] && [[ "${ORACLE[$set_name]}" =~ ^[0-9.]+$ ]]; then
        sb=$(echo "$sb + ${BASELINE[$set_name]}" | bc -l)
        so=$(echo "$so + ${ORACLE[$set_name]}" | bc -l)
        for f in "${FACTOR_LIST[@]}"; do
            fc=$(echo "${FACTOR_CERS[$set_name]:-}" | tr ',' '\n' | awk -F: -v k="x${f}" '$1==k{print $2}')
            if [[ "$fc" =~ ^[0-9.]+$ ]]; then
                FACTOR_SUMS[$f]=$(echo "${FACTOR_SUMS[$f]:-0} + $fc" | bc -l)
            fi
        done
        cnt=$((cnt+1))
    fi
done
if [ "$cnt" -gt 0 ]; then
    printf "  %-22s %8s" "----" ""
    for _f in "${FACTOR_LIST[@]}"; do printf " %8s" "----"; done
    printf " %8s %8s  %s\n" "----" "----" ""

    printf "  %-22s %8s" "AVERAGE" ""
    for f in "${FACTOR_LIST[@]}"; do
        if [ -n "${FACTOR_SUMS[$f]:-}" ]; then
            printf " %8.2f" "$(echo "${FACTOR_SUMS[$f]}/$cnt" | bc -l)"
        else
            printf " %8s" "NA"
        fi
    done
    printf " %8s %8.2f  %s\n" "" "$(echo "$so/$cnt" | bc -l)" ""
fi
echo "=============================================================================="
echo "Full log: ${EVAL_LOG}"
