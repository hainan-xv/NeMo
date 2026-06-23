#!/bin/bash
set -e
set -o pipefail

# ============================================================================
# Eval a (multistream-)TDT ASR checkpoint on the Open ASR Leaderboard datasets.
#
# This is the ASR sibling of eval_ord2.sh: it runs LOCALLY (downloads the
# checkpoint from ORD via ssh, then evaluates on THIS box's GPU with
# run_eval_asr.py over the HF ESB / Open-ASR-Leaderboard suite).  The driver
# auto-detects the model class from the checkpoint, so one script handles BOTH:
#   * the regular TDT model (EncDecRNNTBPEModel, e.g. fine-tuned parakeet-tdt), and
#   * the 2-stream spelling+capitalization model (EncDecMultiStreamTDTBPEModel).
#
# Usage:
#   ./eval_asr_ord.sh <EXP_NAME> [STEP] [DEVICE_ID]
#     STEP=last (or USE_LAST=1) -> use the rolling -last.ckpt
#     (no STEP)                 -> prefer a .nemo, else best (non '-last') .ckpt
#
#   MODEL=/abs/path/model.nemo ./eval_asr_ord.sh    # eval an explicit local file
#
#   # Weight averaging (uses NeMo's scripts/checkpoint_averaging/checkpoint_averaging.py):
#   # downloads the .nemo export + every non '-last' (top-k val_wer) ckpt and writes a
#   # self-contained checkpoints/<EXP>/avg_inputs/<name>-averaged.nemo, then evals it.
#   # Requires a .nemo export in the remote checkpoints dir.
#   RUN_AVERAGING=1 ./eval_asr_ord.sh <EXP_NAME>
#
# Env overrides:
#   PROJECT        force a results project dir (default: try the candidates below)
#   BATCH_SIZE     dataloader batch size (default 32)
#   USE_CER        1 -> report CER instead of WER
#   ONLY           comma-separated "dataset[:split]" filter (default: full suite)
#   MAX_EVAL_SAMPLES  cap samples per dataset (fast iteration)
#   QUICK_TEST     1 -> 10 samples from ami/test only
#   FORCE_DOWNLOAD 1 -> re-scp even if a local copy exists
#   MAX_SYMBOLS    override greedy symbols-per-step (multistream only)
#   RUN_AVERAGING  1 -> average all non '-last' ckpts into a *-averaged.nemo before eval
#   FORCE_AVERAGE  1 -> recompute the averaged .nemo (else the cached one is reused)
#   REUSE_AVG      1 -> reuse an existing local *-averaged.nemo with no remote access
#   REUSE          1 -> reuse any cached local copy without prompting (skip the interactive ask)
#   LM_FUSION_ALPHA  override the GPT-LM fusion weight (TDT+GPT fusion models only); 0 = LM disabled (TDT-only)
#
# When a checkpoint is already cached locally, the script asks whether to reuse it or
# re-download a fresh copy (unless FORCE_DOWNLOAD=1 / REUSE=1, or it is run non-interactively).
# ============================================================================

# ---------- ORD connection ----------
REMOTE_HOST="${REMOTE_HOST:-cs-oci-ord-login-01.nvidia.com}"
REMOTE_USER="${REMOTE_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/lustre/fsw/portfolios/llmservice/users/hainanx/results}"
# The multistream/TDT recipes save under Streaming_SLM_ORD3; keep ord2 as a fallback.
PROJECT_CANDIDATES_DEFAULT=("Streaming_SLM_ORD3" "Streaming_SLM_ord2")
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no"

# ---------- Local paths ----------
NEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOCAL_CKPT_DIR="${NEMO_ROOT}/checkpoints"
RUN_EVAL_PY="${NEMO_ROOT}/run_eval_asr.py"
export PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH}"
if [ ! -f "$RUN_EVAL_PY" ]; then
    echo "ERROR: cannot find eval driver at ${RUN_EVAL_PY}" >&2
    exit 1
fi

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
QUICK_TEST="${QUICK_TEST:-0}"
ONLY="${ONLY:-}"
# TED-LIUM was removed from hf-audio/open-asr-leaderboard on 2026-05-27; pin to
# the last commit that still has the parquet so historical numbers reproduce.
TEDLIUM_REVISION="${TEDLIUM_REVISION:-20a009a}"

# ---------- Cached-checkpoint reuse prompt ----------
# When a local copy already exists, ask once whether to reuse it or pull a fresh copy.
# A "fresh" answer flips FORCE_DOWNLOAD/FORCE_AVERAGE so every downstream artifact is re-fetched
# consistently. Honors explicit overrides and stays non-interactive-safe (defaults to reuse).
_REUSE_ASKED=0
_REUSE_ANS=""
maybe_reuse() {
    local path="$1"
    [ "${FORCE_DOWNLOAD:-0}" = "1" ] && return 1   # already forced fresh
    [ "${REUSE:-0}" = "1" ] && return 0            # forced reuse, no prompt
    if [ "$_REUSE_ASKED" = "1" ]; then
        [ "$_REUSE_ANS" = "reuse" ] && return 0 || return 1
    fi
    # No controlling terminal -> preserve old behavior (silently reuse the cache).
    if [ ! -e /dev/tty ]; then return 0; fi
    _REUSE_ASKED=1
    printf '\n==> A locally cached copy already exists:\n      %s\n' "$path" >&2
    local ans=""
    read -r -p "    Reuse it? [Y/n]  (n = re-download a fresh copy) " ans </dev/tty || ans=""
    case "$ans" in
        [Nn]*) _REUSE_ANS="fresh"; FORCE_DOWNLOAD=1; FORCE_AVERAGE=1; echo "    -> re-downloading fresh." >&2; return 1 ;;
        *)     _REUSE_ANS="reuse"; echo "    -> reusing cached copy." >&2; return 0 ;;
    esac
}

# ---------- Resolve / download the checkpoint ----------
LOCAL_CKPT_PATH=""
LOCAL_TOK_DIR=""

if [ -n "${MODEL:-}" ]; then
    # Explicit local checkpoint: skip all remote resolution.
    LOCAL_CKPT_PATH="$MODEL"
    [ -z "$EXP_NAME" ] && EXP_NAME="$(basename "$(dirname "$MODEL")")"
    echo "==> Using explicit local checkpoint: $LOCAL_CKPT_PATH"
    if [ -n "${TOKENIZER_DIR:-}" ]; then
        LOCAL_TOK_DIR="$TOKENIZER_DIR"
    fi
else
    if [ -z "$EXP_NAME" ]; then
        echo "Usage: $0 <EXP_NAME> [STEP] [DEVICE_ID]   (or MODEL=/abs/path $0)" >&2
        exit 1
    fi

    # Resolve which project dir hosts this experiment (checkpoints double-nest:
    # <root>/<proj>/<EXP>/<EXP>/checkpoints; tokenizer at <root>/<proj>/<EXP>/tokenizer).
    if [ -n "${PROJECT:-}" ]; then
        PROJECT_CANDIDATES=("$PROJECT")
    else
        PROJECT_CANDIDATES=("${PROJECT_CANDIDATES_DEFAULT[@]}")
    fi
    REMOTE_CKPT_DIR=""
    REMOTE_EXP_DIR=""
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
        # ---- Weight averaging via NeMo's checkpoint_averaging.py ----
        # That script averages all non '-last' .ckpt files that sit next to a
        # .nemo and writes a self-contained <name>-averaged.nemo (model class +
        # tokenizer embedded), so the result needs NO separate tokenizer download.
        [ -n "$STEP" ] && echo "WARNING: RUN_AVERAGING=1 ignores explicit STEP=$STEP" >&2 && STEP=""
        AVG_SCRIPT="${NEMO_ROOT}/scripts/checkpoint_averaging/checkpoint_averaging.py"
        AVG_DIR="${LOCAL_CKPT_DIR}/${EXP_NAME}/avg_inputs"

        # The *-averaged.nemo is the actual model we run inference with. If one is already cached,
        # reuse it (and skip ALL download + averaging) -- either silently (REUSE_AVG=1) or after an
        # interactive prompt. FORCE_AVERAGE=1 / FORCE_DOWNLOAD=1 always rebuild it from scratch.
        if [ "${FORCE_AVERAGE:-0}" != "1" ] && [ "${FORCE_DOWNLOAD:-0}" != "1" ]; then
            existing_avg=$(ls -t "${AVG_DIR}"/*-averaged.nemo 2>/dev/null | head -1)
            if [ "${REUSE_AVG:-0}" = "1" ]; then
                if [ -n "$existing_avg" ]; then
                    LOCAL_CKPT_PATH="$existing_avg"
                    echo "==> REUSE_AVG=1: reusing existing averaged model: $LOCAL_CKPT_PATH"
                else
                    echo "ERROR: REUSE_AVG=1 but no *-averaged.nemo in ${AVG_DIR}" >&2
                    exit 1
                fi
            elif [ -n "$existing_avg" ] && maybe_reuse "$existing_avg"; then
                LOCAL_CKPT_PATH="$existing_avg"
                echo "==> Reusing existing averaged model (no re-download/averaging): $LOCAL_CKPT_PATH"
            fi
        fi

        if [ -z "$LOCAL_CKPT_PATH" ]; then
            if [ ! -f "$AVG_SCRIPT" ]; then
                echo "ERROR: NeMo averaging script not found at $AVG_SCRIPT" >&2
                exit 1
            fi
            mkdir -p "$AVG_DIR"

            # The averaging script needs a self-contained .nemo alongside the
            # .ckpts (it supplies the model class + tokenizer).
            NEMO_FNAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
                "ls -t ${REMOTE_CKPT_DIR}/*.nemo 2>/dev/null | grep -v -- '-averaged' | head -1 | xargs -r basename")
            if [ -z "$NEMO_FNAME" ]; then
                echo "ERROR: RUN_AVERAGING needs a .nemo export in ${REMOTE_CKPT_DIR}" >&2
                echo "       (it supplies the model class + tokenizer for checkpoint_averaging.py)." >&2
                exit 1
            fi
            LOCAL_NEMO="${AVG_DIR}/${NEMO_FNAME}"
            # The base .nemo only supplies the model class/config/tokenizer; checkpoint_averaging.py
            # OVERWRITES its weights with the average of the .ckpt files (load_state_dict, strict=True).
            # So its weights are irrelevant and it never needs a "fresh" re-download -- reuse any cached
            # copy and fetch it only when missing/empty (even on a fresh-checkpoint re-download).
            if [ -s "$LOCAL_NEMO" ]; then
                echo "==> Cached base model (weights unused; supplies arch+tokenizer only): ${NEMO_FNAME}"
            else
                echo "==> Downloading base model ${NEMO_FNAME}..."
                scp $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_DIR}/${NEMO_FNAME}" "$LOCAL_NEMO" || exit 1
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

            # Sync EXACTLY the current top-k non '-last' .ckpt set into the .nemo's folder via rsync
            # (resumable, skips byte-identical files), then PRUNE any stale .ckpt left from a previous
            # run. This is critical: checkpoint_averaging.py averages EVERY *.ckpt in the folder
            # (os.listdir, ignoring only -last.ckpt), so a leftover older checkpoint would silently be
            # folded into the average. Pruning guarantees we average ONLY these ${NUM_CKPTS}.
            CKPT_LIST_FILE=$(mktemp)
            printf '%s\n' "$REMOTE_CKPT_FILES" > "$CKPT_LIST_FILE"
            RSYNC_OPTS=(-vh --times --partial --files-from="$CKPT_LIST_FILE" -e "ssh $SSH_OPTS")
            # A "fresh re-download" forces rsync to re-transfer even byte-identical files.
            [ "${FORCE_DOWNLOAD:-0}" = "1" ] && RSYNC_OPTS+=(--ignore-times)
            echo "==> Syncing ${NUM_CKPTS} checkpoint(s) from ORD via rsync..."
            if ! rsync "${RSYNC_OPTS[@]}" \
                    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_DIR}/" "$AVG_DIR/"; then
                echo "ERROR: rsync of averaging checkpoints failed." >&2
                rm -f "$CKPT_LIST_FILE"
                exit 1
            fi
            rm -f "$CKPT_LIST_FILE"

            # Prune any local non '-last' .ckpt that is NOT in the current remote top-k set, so the
            # averaging input set == the current ${NUM_CKPTS} checkpoints (no stale ones).
            NEWEST_INPUT="$LOCAL_NEMO"
            PRUNED_STALE=0
            shopt -s nullglob
            for f in "$AVG_DIR"/*.ckpt; do
                base=$(basename "$f")
                case "$base" in *-last.ckpt) continue ;; esac
                if printf '%s\n' "$REMOTE_CKPT_FILES" | grep -qxF "$base"; then
                    [ "$f" -nt "$NEWEST_INPUT" ] && NEWEST_INPUT="$f"
                else
                    echo "==> Pruning stale checkpoint (not in current top-${NUM_CKPTS}): $base"
                    rm -f "$f"
                    PRUNED_STALE=1
                fi
            done
            shopt -u nullglob

            # Sanity: the folder must now hold exactly the current set of non '-last' ckpts.
            LOCAL_NUM_CKPTS=$(find "$AVG_DIR" -maxdepth 1 -name '*.ckpt' ! -name '*-last.ckpt' | wc -l)
            if [ "$LOCAL_NUM_CKPTS" -ne "$NUM_CKPTS" ]; then
                echo "ERROR: expected ${NUM_CKPTS} checkpoint(s) to average but ${AVG_DIR} has ${LOCAL_NUM_CKPTS}." >&2
                exit 1
            fi
            echo "==> Averaging input set: ${LOCAL_NUM_CKPTS} checkpoint(s) (stale ones pruned: ${PRUNED_STALE})"

            # checkpoint_averaging.py writes <nemo_basename>-averaged.nemo next to the inputs.
            LOCAL_CKPT_PATH="${AVG_DIR}/${NEMO_FNAME%.nemo}-averaged.nemo"
            # If we pruned a stale ckpt the input set changed even if timestamps didn't, so force recompute.
            [ "$PRUNED_STALE" = "1" ] && FORCE_AVERAGE=1
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
        # ---- Single checkpoint: pick a filename on the remote ----
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
            # Prefer a self-contained .nemo; else best (non '-last') ckpt by val_wer.
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

        # Download the checkpoint (cache-first).
        REMOTE_CKPT_PATH="${REMOTE_CKPT_DIR}/${CKPT_FILENAME}"
        LOCAL_CKPT_PATH="${LOCAL_CKPT_DIR}/${EXP_NAME}/${CKPT_FILENAME}"
        if [ -f "$LOCAL_CKPT_PATH" ] && maybe_reuse "$LOCAL_CKPT_PATH"; then
            echo "==> Using cached local checkpoint: $LOCAL_CKPT_PATH"
        else
            echo "==> Downloading checkpoint from ORD..."
            echo "    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_PATH}"
            mkdir -p "$(dirname "$LOCAL_CKPT_PATH")"
            if ! scp $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_PATH}" "$LOCAL_CKPT_PATH"; then
                echo "ERROR: scp failed for ${REMOTE_CKPT_PATH}." >&2
                [ -f "$LOCAL_CKPT_PATH" ] && [ ! -s "$LOCAL_CKPT_PATH" ] && rm -f "$LOCAL_CKPT_PATH"
                exit 1
            fi
            echo "==> Download complete ($(du -h "$LOCAL_CKPT_PATH" | cut -f1))"
        fi
    fi

    # A .ckpt needs the exported tokenizer dir; pull it (recursively) if present.
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
                # Copy the remote 'tokenizer' dir into the parent (avoids scp's
                # "unexpected filename: ." error from a trailing /. on newer OpenSSH).
                scp -r $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TOK_DIR}" "$(dirname "$LOCAL_TOK_DIR")/"
            fi
        else
            echo "WARNING: .ckpt selected but no tokenizer dir at ${REMOTE_TOK_DIR}." >&2
            echo "         The .ckpt load may fail; consider evaluating a .nemo export." >&2
        fi
    fi
fi

if [ ! -e "$LOCAL_CKPT_PATH" ]; then
    echo "ERROR: checkpoint not found at $LOCAL_CKPT_PATH" >&2
    exit 1
fi

# ---------- Run evaluation on the leaderboard datasets ----------
DATASETS=(
    "ami:test"
    "earnings22:test"
    "gigaspeech:test"
    "librispeech:test.clean"
    "librispeech:test.other"
    "spgispeech:test"
#    "tedlium:test"
    "voxpopuli:test"
)

# Optional ONLY filter (dataset[:split], comma-separated).
if [ -n "$ONLY" ]; then
    IFS=',' read -r -a _only <<< "$ONLY"
    _filtered=()
    for want in "${_only[@]}"; do
        for entry in "${DATASETS[@]}"; do
            ename="${entry%%:*}"
            if [ "$ename" = "${want%%:*}" ]; then
                if [[ "$want" == *:* ]]; then _filtered+=("$want"); else _filtered+=("$entry"); fi
            fi
        done
    done
    DATASETS=("${_filtered[@]}")
fi
if [ "$QUICK_TEST" = "1" ]; then
    DATASETS=("ami:test")
    MAX_EVAL_SAMPLES=10
    echo "==> QUICK TEST: 10 samples from ami/test only"
fi

EXTRA_ARGS=()
[ -n "$LOCAL_TOK_DIR" ] && EXTRA_ARGS+=(--tokenizer_dir "$LOCAL_TOK_DIR")
[ "${USE_CER:-0}" = "1" ] && EXTRA_ARGS+=(--use_cer)
[ -n "${MAX_SYMBOLS:-}" ] && EXTRA_ARGS+=(--max_symbols_per_step "$MAX_SYMBOLS")
[ -n "$MAX_EVAL_SAMPLES" ] && EXTRA_ARGS+=(--max_eval_samples "$MAX_EVAL_SAMPLES")
[ -n "${LM_FUSION_ALPHA:-}" ] && EXTRA_ARGS+=(--lm_fusion_alpha "$LM_FUSION_ALPHA")

# Tag the log file with the fusion alpha so TDT-only (alpha=0) and fused runs don't clobber logs.
LOG_TAG=""
[ -n "${LM_FUSION_ALPHA:-}" ] && LOG_TAG="_lmAlpha${LM_FUSION_ALPHA}"
EVAL_LOG="${LOCAL_CKPT_DIR}/${EXP_NAME}/eval_$(basename "${LOCAL_CKPT_PATH}")${LOG_TAG}.log"
mkdir -p "$(dirname "$EVAL_LOG")"
: > "$EVAL_LOG"
echo "==> Logging to $EVAL_LOG"

declare -A WER_RESULTS
for ds_entry in "${DATASETS[@]}"; do
    IFS=':' read -r DATASET SPLIT <<< "$ds_entry"
    DATASET_EXTRA_ARGS=()
    if [ "$DATASET" = "tedlium" ] && [ -n "$TEDLIUM_REVISION" ]; then
        DATASET_EXTRA_ARGS+=(--dataset_revision "$TEDLIUM_REVISION")
    fi

    echo "running inference on ${DATASET}/${SPLIT}..."
    {
        echo "----------------------------------------------------------------------"
        echo "[$(date '+%H:%M:%S')] Evaluating: ${DATASET}/${SPLIT}"
        echo "----------------------------------------------------------------------"
    } >> "$EVAL_LOG"

    run_log=$(mktemp)
    set +e
    python "$RUN_EVAL_PY" \
        --model "$LOCAL_CKPT_PATH" \
        --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --device "$DEVICE_ID" \
        --batch_size "$BATCH_SIZE" \
        "${EXTRA_ARGS[@]}" \
        "${DATASET_EXTRA_ARGS[@]}" \
        > "$run_log" 2>&1
    rc=$?
    set -e
    cat "$run_log" >> "$EVAL_LOG"

    if [ "$rc" -ne 0 ]; then
        echo "  ${DATASET} FAILED (exit ${rc}) -- see ${EVAL_LOG}"
        WER_RESULTS[$ds_entry]="FAIL"
    else
        wer=$(grep -oE 'WER:[[:space:]]*[0-9.]+' "$run_log" | tail -1 | grep -oE '[0-9.]+$')
        rtfx=$(grep -oE 'RTFX:[[:space:]]*[0-9.]+' "$run_log" | tail -1 | grep -oE '[0-9.]+$')
        if [ -n "$wer" ]; then
            echo "  ${DATASET} WER: ${wer} | RTFx: ${rtfx:-?}"
            WER_RESULTS[$ds_entry]="$wer"
        else
            echo "  ${DATASET}: no WER parsed -- see ${EVAL_LOG}"
            WER_RESULTS[$ds_entry]="N/A"
        fi
    fi
    rm -f "$run_log"
done

# ---------- Summary ----------
echo ""
echo "================ Evaluation summary ================"
echo "exp    : ${EXP_NAME}"
echo "ckpt   : $(basename "${LOCAL_CKPT_PATH}")"
echo "metric : $([ "${USE_CER:-0}" = "1" ] && echo CER || echo WER)"
[ -n "${LM_FUSION_ALPHA:-}" ] && echo "lm_alpha: ${LM_FUSION_ALPHA}$([ "${LM_FUSION_ALPHA}" = "0" ] && echo '  (LM disabled: TDT-only)')"
sum=0; cnt=0
for ds_entry in "${DATASETS[@]}"; do
    v="${WER_RESULTS[$ds_entry]:-N/A}"
    printf "  %-26s %s\n" "$ds_entry" "$v"
    if [[ "$v" =~ ^[0-9.]+$ ]]; then sum=$(echo "$sum + $v" | bc -l); cnt=$((cnt+1)); fi
done
if [ "$cnt" -gt 0 ]; then
    printf "  %-26s %s\n" "----" "----"
    printf "  %-26s %.2f\n" "AVERAGE" "$(echo "$sum / $cnt" | bc -l)"
fi
echo "===================================================="
echo "Full log: ${EVAL_LOG}"
