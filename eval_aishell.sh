#!/bin/bash
set -e
set -o pipefail

# ============================================================================
# Eval a Mandarin ASR checkpoint (the rno/ tdt or chat char recipes) on the
# local AISHELL-2 test manifests, reporting CER.
#
# Like eval_asr_ord.sh, this runs LOCALLY: it downloads the checkpoint from the
# RNO cluster via ssh, then evaluates on THIS box's GPU with eval_aishell_cer.py
# over the AISHELL test sets (android / ios / mic by default). The driver
# auto-detects the model class from the checkpoint and decodes with the model's
# own embedded decoding config (TDT durations, CHAT att-joint, ...), so one
# script handles both rno/zh_tdt.sh and rno/zh_chat.sh models.
#
# Usage:
#   ./eval_aishell.sh <EXP_NAME> [STEP] [DEVICE_ID]
#     STEP=last (or USE_LAST=1) -> use the rolling -last.ckpt
#     (no STEP)                 -> prefer a .nemo, else best (lowest val_wer) .ckpt
#
#   MODEL=/abs/path/model.nemo ./eval_aishell.sh        # eval an explicit local file
#
# Checkpoint averaging is ON by default (these recipes save top-k ckpts +
# always_save_nemo): it downloads the .nemo export + every non '-last' (top-k
# val_wer) ckpt and writes a self-contained checkpoints/<EXP>/avg_inputs/
# <name>-averaged.nemo via scripts/checkpoint_averaging/checkpoint_averaging.py,
# then evaluates that. Disable with RUN_AVERAGING=0 to eval a single checkpoint.
#
# Env overrides:
#   RUN_AVERAGING  1 (default) -> average non '-last' ckpts into *-averaged.nemo;
#                  0 -> eval a single checkpoint (.nemo, or STEP / USE_LAST .ckpt)
#   FORCE_AVERAGE  1 -> recompute the averaged .nemo (else a cached one is reused)
#   REUSE_AVG      1 -> reuse an existing local *-averaged.nemo with no remote access
#   REMOTE_HOST / REMOTE_USER / SSH_KEY   RNO ssh connection (defaults below)
#   REMOTE_RESULTS_ROOT / PROJECT         where the training recipe saved results
#   MANIFEST_DIR        dir holding test_android.json / test_ios.json / test_mic.json
#   AUDIO_SRC_PREFIX    manifest audio prefix to rewrite (cluster path)
#   AUDIO_DST_PREFIX    local audio root (default: parent of MANIFEST_DIR)
#   SETS                space/comma list of manifest basenames (no .json)
#                       default: "test_android test_ios test_mic"
#   ONLY                alias for SETS (comma-separated)
#   BATCH_SIZE          dataloader batch size (default 32)
#   MAX_EVAL_SAMPLES    cap samples per set (fast iteration)
#   KEEP_SPACES         1 -> do NOT collapse whitespace before CER
#   FORCE_DOWNLOAD      1 -> re-scp even if a local copy exists
# ============================================================================

# ---------- RNO connection ----------
REMOTE_HOST="${REMOTE_HOST:-draco-rno-login}"
REMOTE_USER="${REMOTE_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/gpfs/fs1/projects/ent_aiapps/users/hainanx/results}"
# The Mandarin tdt/chat recipes save under this project (see rno/zh_tdt.sh).
PROJECT_CANDIDATES_DEFAULT=("Mandarin_202606_enc512l18_enginit")
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no"

# ---------- Local paths ----------
NEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOCAL_CKPT_DIR="${NEMO_ROOT}/checkpoints"
EVAL_PY="${NEMO_ROOT}/eval_aishell_cer.py"
AVG_SCRIPT="${NEMO_ROOT}/scripts/checkpoint_averaging/checkpoint_averaging.py"
RUN_AVERAGING="${RUN_AVERAGING:-1}"
export PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH}"
if [ ! -f "$EVAL_PY" ]; then
    echo "ERROR: cannot find eval driver at ${EVAL_PY}" >&2
    exit 1
fi

# ---------- Dataset ----------
MANIFEST_DIR="${MANIFEST_DIR:-/home/hainanx/Workplace/data/aishell_eval/aishell/manifests}"
AUDIO_SRC_PREFIX="${AUDIO_SRC_PREFIX:-/data/mandarin/aishell2/evaluation/aishell}"
# By convention the manifests live at <aishell_root>/manifests and the audio at
# <aishell_root>/aishell2_test_data/..., so the local audio root is the parent.
AUDIO_DST_PREFIX="${AUDIO_DST_PREFIX:-$(dirname "$MANIFEST_DIR")}"

# Which test sets to score.
if [ -n "${ONLY:-}" ]; then
    SETS="${ONLY//,/ }"
fi
SETS="${SETS:-test_android test_ios test_mic}"
SETS="${SETS//,/ }"

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

# ---------- Resolve / download the checkpoint ----------
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

    # Checkpoints double-nest: <root>/<proj>/<EXP>/<EXP>/checkpoints
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
            echo "       Current remote layout:" >&2
            ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
                "ls -la '${FOUND_EXP_WITHOUT_CKPT}' '${FOUND_EXP_WITHOUT_CKPT}/${EXP_NAME}' 2>&1" >&2 || true
            echo "       (training may have failed before checkpointing, or used a different exp_manager.name.)" >&2
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
        # ---- Weight averaging via NeMo's checkpoint_averaging.py ----
        # Averages all non '-last' .ckpt files that sit next to a .nemo and writes
        # a self-contained <name>-averaged.nemo (model class + labels embedded), so
        # the result needs no separate tokenizer download.
        [ -n "$STEP" ] && echo "WARNING: RUN_AVERAGING=1 ignores explicit STEP=$STEP" >&2 && STEP=""
        AVG_DIR="${LOCAL_CKPT_DIR}/${EXP_NAME}/avg_inputs"

        # REUSE_AVG=1: reuse a previously-built *-averaged.nemo with no remote access.
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

            # The averaging script needs a self-contained .nemo alongside the .ckpts
            # (it supplies the model class + char labels).
            NEMO_FNAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
                "ls -t ${REMOTE_CKPT_DIR}/*.nemo 2>/dev/null | grep -v -- '-averaged' | head -1 | xargs -r basename")
            if [ -z "$NEMO_FNAME" ]; then
                echo "ERROR: RUN_AVERAGING needs a .nemo export in ${REMOTE_CKPT_DIR}" >&2
                echo "       (it supplies the model class + labels for checkpoint_averaging.py)." >&2
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

            # Download every non '-last' .ckpt into the .nemo's folder (the script
            # auto-discovers them and ignores -last.ckpt).
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

            # checkpoint_averaging.py writes <nemo_basename>-averaged.nemo next to the inputs.
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

# ---------- Run evaluation over the AISHELL test sets ----------
EXTRA_ARGS=()
[ -n "$AUDIO_SRC_PREFIX" ] && EXTRA_ARGS+=(--audio_src_prefix "$AUDIO_SRC_PREFIX")
[ -n "$AUDIO_DST_PREFIX" ] && EXTRA_ARGS+=(--audio_dst_prefix "$AUDIO_DST_PREFIX")
[ -n "$MAX_EVAL_SAMPLES" ] && EXTRA_ARGS+=(--max_eval_samples "$MAX_EVAL_SAMPLES")
[ "${KEEP_SPACES:-0}" = "1" ] && EXTRA_ARGS+=(--keep_spaces)
# Multi-target only: consistency-maintaining decode (token + pronunciation heads).
# CONSISTENCY=1 ./eval_aishell.sh ...   (optionally CONSISTENCY_WEIGHTS="token,notone,tone")
[ "${CONSISTENCY:-0}" = "1" ] && EXTRA_ARGS+=(--consistency)
[ -n "${CONSISTENCY_WEIGHTS:-}" ] && EXTRA_ARGS+=(--consistency_weights "$CONSISTENCY_WEIGHTS")

EVAL_LOG="${LOCAL_CKPT_DIR}/${EXP_NAME}/eval_aishell_$(basename "${LOCAL_CKPT_PATH}").log"
mkdir -p "$(dirname "$EVAL_LOG")"
: > "$EVAL_LOG"
echo "==> Logging to $EVAL_LOG"

declare -A CER_RESULTS
for set_name in $SETS; do
    manifest="${MANIFEST_DIR}/${set_name}.json"
    if [ ! -f "$manifest" ]; then
        echo "  ${set_name}: manifest not found at ${manifest} -- skipping"
        CER_RESULTS[$set_name]="NOFILE"
        continue
    fi

    echo "running inference on ${set_name}..."
    {
        echo "----------------------------------------------------------------------"
        echo "[$(date '+%H:%M:%S')] Evaluating: ${set_name} (${manifest})"
        echo "----------------------------------------------------------------------"
    } >> "$EVAL_LOG"

    out_manifest="${LOCAL_CKPT_DIR}/${EXP_NAME}/aishell_${set_name}.jsonl"
    run_log=$(mktemp)
    set +e
    python "$EVAL_PY" \
        --model "$LOCAL_CKPT_PATH" \
        --manifest "$manifest" \
        --device "$DEVICE_ID" \
        --batch_size "$BATCH_SIZE" \
        --output "$out_manifest" \
        "${EXTRA_ARGS[@]}" \
        > "$run_log" 2>&1
    rc=$?
    set -e
    cat "$run_log" >> "$EVAL_LOG"

    if [ "$rc" -ne 0 ]; then
        echo "  ${set_name} FAILED (exit ${rc}) -- see ${EVAL_LOG}"
        CER_RESULTS[$set_name]="FAIL"
    else
        cer=$(grep -oE 'CER:[[:space:]]*[0-9.]+' "$run_log" | tail -1 | grep -oE '[0-9.]+$')
        rtfx=$(grep -oE 'RTFX:[[:space:]]*[0-9.]+' "$run_log" | tail -1 | grep -oE '[0-9.]+$')
        if [ -n "$cer" ]; then
            echo "  ${set_name} CER: ${cer} | RTFx: ${rtfx:-?}"
            CER_RESULTS[$set_name]="$cer"
        else
            echo "  ${set_name}: no CER parsed -- see ${EVAL_LOG}"
            CER_RESULTS[$set_name]="N/A"
        fi
    fi
    rm -f "$run_log"
done

# ---------- Summary ----------
echo ""
echo "================ AISHELL CER summary ================"
echo "exp    : ${EXP_NAME}"
echo "ckpt   : $(basename "${LOCAL_CKPT_PATH}")"
echo "metric : CER (%)"
sum=0; cnt=0
for set_name in $SETS; do
    v="${CER_RESULTS[$set_name]:-N/A}"
    printf "  %-22s %s\n" "$set_name" "$v"
    if [[ "$v" =~ ^[0-9.]+$ ]]; then sum=$(echo "$sum + $v" | bc -l); cnt=$((cnt+1)); fi
done
if [ "$cnt" -gt 0 ]; then
    printf "  %-22s %s\n" "----" "----"
    printf "  %-22s %.2f\n" "AVERAGE" "$(echo "$sum / $cnt" | bc -l)"
fi
echo "===================================================="
echo "Full log: ${EVAL_LOG}"
