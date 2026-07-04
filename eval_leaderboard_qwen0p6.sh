#!/bin/bash
set -uo pipefail

# ============================================================================
# Average + leaderboard-eval EVERY Streaming_SLM_Qwen1p7B model, and print a
# single clean WER table: one ROW per leaderboard dataset, one COLUMN per model.
#
#   dataset                        imend_loss   imend_weighted   imend_rnddelay
#   --------------------------------------------------------------------------
#   ami/test                            18.42            18.10            18.77
#   earnings22/test                     ...
#   ...
#   --------------------------------------------------------------------------
#   AVERAGE                             ...
#
# Pipeline:
#   Phase 1 (per model): resolve + rsync top-3 (save_top_k=3) val_wer ckpts and
#     average them into <EXP>-averaged.ckpt   [reuses eval_leaderboard.sh
#     PREPARE_ONLY, so all the rsync/averaging/freshness/caching logic is shared].
#   Phase 2: evaluate DATASET-major (outer loop = dataset, inner = models) and
#     STREAM each CELL to STDOUT the instant its eval finishes -- the row label
#     prints first, then each model's WER appears one-by-one across the row, so
#     AMI fills in left-to-right, then earnings22, etc., without waiting for the
#     whole run. (Each model is reloaded per dataset; that reload cost is the
#     price of a live datasets-as-rows / models-as-columns table.)
#   The AVERAGE row (numeric cells only) prints at the end.
#
# Output discipline: STDOUT is ONLY the table (header + cells as they complete +
# AVERAGE). The verbose per-model eval logs go to files under
# eval_results/_leaderboard_*/; only post-row error notes go to STDERR. So
# `./eval_leaderboard_qwen0p6.sh > table.txt` captures just the table.
#
# Runs LOCALLY (this box's GPU) and pulls checkpoints FROM OCI. Needs the gated
# HF token in ./.hf_token (same as eval_leaderboard.sh).
#
# Usage:
#   ./eval_leaderboard_qwen0p6.sh                     # every model under the project
#   ./eval_leaderboard_qwen0p6.sh imend_loss imend_weighted   # only these models
#   ./eval_leaderboard_qwen0p6.sh --gpu 1 imend_loss_sub2
#   QUICK_TEST=1 ./eval_leaderboard_qwen0p6.sh        # 10-sample smoke over all models
#
# LOCAL_ONLY_IF_EXIST=1 -> always reuse the already-averaged checkpoint under
#   ./checkpoints/<EXP>/<EXP>-averaged.ckpt when it exists (NO grid check, so it
#   won't re-download even if training has since written newer checkpoints); it
#   only pulls from OCI for models that have no local average yet. When no models
#   are named it discovers them from the local cache. Use this for the "complete"
#   run right after the quick test pulled + averaged the ckpts.
#     LOCAL_ONLY_IF_EXIST=1 ./eval_leaderboard_qwen0p6.sh
#   (Default, without this flag, re-checks the grid and pulls newer top-k ckpts.)
#
# Env overrides: BATCH_SIZE (default 256; auto-halved on CUDA OOM down to
#   MIN_BATCH_SIZE, default 1), MAX_NEW_TOKENS, MAX_EVAL_SAMPLES, ONLY,
#   QUICK_TEST, FORCE_AVERAGE, FORCE_DOWNLOAD, LOCAL_ONLY_IF_EXIST, GPU, ...
# ============================================================================

NEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
EVAL="${NEMO_ROOT}/eval_leaderboard.sh"
LOCAL_CKPT_DIR="${NEMO_ROOT}/checkpoints"
LOCAL_ONLY_IF_EXIST="${LOCAL_ONLY_IF_EXIST:-0}"
[ -f "$EVAL" ] || { echo "ERROR: sibling driver not found at $EVAL" >&2; exit 1; }

PROJECT="Streaming_SLM_Qwen1p7B"

# ---------- OCI connection (must match eval_leaderboard.sh) ----------
REMOTE_HOST="${REMOTE_HOST:-draco-oci-login-01.draco-oci-iad.nvidia.com}"
REMOTE_USER="${REMOTE_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/lustre/fsw/portfolios/llmservice/users/hainanx/results}"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no"
REMOTE_PROJECT_DIR="${REMOTE_RESULTS_ROOT}/${PROJECT}"

# ---------- Arguments ----------
GPU="${GPU:-0}"
EXPS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu|--device)     GPU="$2"; shift 2 ;;
        --gpu=*|--device=*) GPU="${1#*=}"; shift ;;
        -h|--help)          sed -n '3,40p' "$0"; exit 0 ;;
        -*)                 echo "ERROR: unknown option: $1" >&2; exit 1 ;;
        *)                  EXPS+=("$1"); shift ;;
    esac
done

# ---------- Discover experiments if none given ----------
if [ "${#EXPS[@]}" -eq 0 ]; then
    if [ "$LOCAL_ONLY_IF_EXIST" = "1" ]; then
        echo "==> LOCAL_ONLY_IF_EXIST=1: discovering models from local cache ${LOCAL_CKPT_DIR}/*/ ..."
        shopt -s nullglob
        for d in "${LOCAL_CKPT_DIR}"/*/; do
            e="$(basename "$d")"
            [ -f "${d}${e}-averaged.ckpt" ] && EXPS+=("$e")
        done
        shopt -u nullglob
        IFS=$'\n' EXPS=($(printf '%s\n' "${EXPS[@]}" | sort)); unset IFS
        if [ "${#EXPS[@]}" -eq 0 ]; then
            echo "ERROR: LOCAL_ONLY_IF_EXIST=1 but no averaged checkpoints found under ${LOCAL_CKPT_DIR}/<EXP>/<EXP>-averaged.ckpt" >&2
            echo "       Run once without LOCAL_ONLY_IF_EXIST to download+average first." >&2
            exit 1
        fi
    else
        echo "==> Discovering models under ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR} ..."
        mapfile -t EXPS < <(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
            "for d in ${REMOTE_PROJECT_DIR}/*/; do e=\$(basename \"\$d\"); [ -d \"\${d}\${e}/checkpoints\" ] && echo \"\$e\"; done" \
            2>/dev/null | sort)
        if [ "${#EXPS[@]}" -eq 0 ]; then
            echo "ERROR: no experiments with a <EXP>/<EXP>/checkpoints dir found under ${REMOTE_PROJECT_DIR}" >&2
            echo "       (Is the run far enough along to have written checkpoints? Check the path / SSH.)" >&2
            exit 1
        fi
    fi
fi

# Everything below prints progress/errors to STDERR; STDOUT is reserved for the
# final table only.
echo "==> Will average-top-3 + leaderboard-eval ${#EXPS[@]} model(s) from ${PROJECT} (gpu ${GPU}):" >&2
printf '      - %s\n' "${EXPS[@]}" >&2

# ---------- Eval environment (direct run_eval_sslm.py calls) ----------
RUN_EVAL_PY="${NEMO_ROOT}/run_eval_sslm.py"
[ -f "$RUN_EVAL_PY" ] || { echo "ERROR: eval driver not found at $RUN_EVAL_PY" >&2; exit 1; }
export PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH:-}"
export STREAMING_STT_MODEL_ROOT="${NEMO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${NEMO_ROOT}/.hf_token" ]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${NEMO_ROOT}/.hf_token")"; export HF_TOKEN
fi

# Start large; run_eval_sslm.py halves the batch on CUDA OOM (down to
# MIN_BATCH_SIZE) so an over-large start just backs off instead of crashing.
BATCH_SIZE="${BATCH_SIZE:-256}"
MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-4}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
QUICK_TEST="${QUICK_TEST:-0}"
ONLY="${ONLY:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"

# ---------- Dataset list (rows of the table) ----------
DATASETS=(
    "ami:test" "earnings22:test" "gigaspeech:test"
    "librispeech:test.clean" "librispeech:test.other"
    "spgispeech:test" "tedlium:test" "voxpopuli:test"
)
if [ -n "$ONLY" ]; then
    IFS=',' read -r -a _only <<< "$ONLY"; _f=()
    for e in "${DATASETS[@]}"; do for w in "${_only[@]}"; do [ "${e%%:*}" = "$w" ] && _f+=("$e"); done; done
    DATASETS=("${_f[@]}")
fi

COMMON_ARGS=(--device "$GPU" --batch_size "$BATCH_SIZE" --min_batch_size "$MIN_BATCH_SIZE"
             --max_new_tokens "$MAX_NEW_TOKENS"
             --no_repeat_ngram_size "$NO_REPEAT_NGRAM_SIZE"
             --dataset_path "hf-audio/esb-datasets-test-only-sorted")
[ -n "$SYSTEM_PROMPT" ] && COMMON_ARGS+=(--system_prompt "$SYSTEM_PROMPT")
[ -n "$MAX_EVAL_SAMPLES" ] && COMMON_ARGS+=(--max_eval_samples "$MAX_EVAL_SAMPLES")
if [ "$QUICK_TEST" = "1" ]; then DATASETS=("ami:test"); COMMON_ARGS+=(--max_eval_samples 10); fi

LOGDIR="${NEMO_ROOT}/eval_results/_leaderboard_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
echo "==> Logs: ${LOGDIR}" >&2

# ---------- Phase 1: prepare (rsync + average) each model's checkpoint ----------
declare -A CKPT
READY=()
FAILED=()
for exp in "${EXPS[@]}"; do
    plog="${LOGDIR}/prep_${exp}.log"
    echo "==> [prep] ${exp}" >&2
    if PROJECT="$PROJECT" RUN_AVERAGING=1 LOCAL_ONLY_IF_EXIST="$LOCAL_ONLY_IF_EXIST" \
        PREPARE_ONLY=1 "$EVAL" "$exp" >"$plog" 2>&1; then
        ck="$(grep '^PREPARED_CKPT=' "$plog" | tail -1 | cut -d= -f2-)"
        if [ -n "$ck" ] && [ -f "$ck" ]; then
            CKPT["$exp"]="$ck"; READY+=("$exp")
        else
            echo "!! ${exp}: prep succeeded but no checkpoint path (see ${plog})" >&2
            FAILED+=("$exp")
        fi
    else
        echo "!! ${exp}: FAILED to prepare checkpoint (tail of ${plog}):" >&2
        tail -n 20 "$plog" >&2
        FAILED+=("$exp")
    fi
done

if [ "${#READY[@]}" -eq 0 ]; then
    echo "ERROR: no models could be prepared; nothing to evaluate." >&2
    exit 1
fi

# ---------- Table geometry (columns = ready models) ----------
DCOL=7   # len("dataset")/("AVERAGE")
for ds in "${DATASETS[@]}"; do n="${ds%%:*}/${ds#*:}"; [ "${#n}" -gt "$DCOL" ] && DCOL="${#n}"; done
MCOL=()
for m in "${READY[@]}"; do w="${#m}"; [ "$w" -lt 8 ] && w=8; MCOL+=("$w"); done
TOTAL="$DCOL"; for w in "${MCOL[@]}"; do TOTAL=$((TOTAL + w + 2)); done
sep() { printf '%*s\n' "$TOTAL" '' | tr ' ' '-'; }
print_row() {  # $1=label, $2.. = cells
    local label="$1"; shift
    printf "%-${DCOL}s" "$label"
    local i=0
    for c in "$@"; do printf "  %${MCOL[$i]}s" "$c"; i=$((i + 1)); done
    printf "\n"
}

# ---------- Evaluate DATASET-major and stream each CELL as it is computed -------
# The row label prints first, then each model's WER is appended (and flushed) the
# instant that eval finishes, so numbers appear one-by-one across the row instead
# of waiting for the whole row. Each model is reloaded per dataset. Numeric cells
# are accumulated (via results.tsv) for the AVERAGE row.
#
# NOTE: cells stream on the SAME line, so per-cell progress must NOT go to a
# terminal-shared STDERR (it would corrupt the row). Per-eval logs live in files;
# only post-row error notes go to STDERR.
RESULTS_TSV="${LOGDIR}/results.tsv"; : > "$RESULTS_TSV"

print_row "dataset" "${READY[@]}"
sep

for ds in "${DATASETS[@]}"; do
    d="${ds%%:*}"; s="${ds#*:}"; name="$d/$s"
    printf "%-${DCOL}s" "$name"        # start the row (label), no newline yet
    i=0; row_errs=""
    for exp in "${READY[@]}"; do
        elog="${LOGDIR}/eval_${exp}.log"
        echo "### ${name}" >> "$elog"
        if python "$RUN_EVAL_PY" --ckpt_path "${CKPT[$exp]}" \
            --dataset "$d" --split "$s" "${COMMON_ARGS[@]}" >>"$elog" 2>&1; then
            v="$(awk -v n="$name" '$1==n {print $2}' "$elog" | tail -1)"
            [ -n "$v" ] || v="n/a"
        else
            v="$(awk -v n="$name" '$1==n {print $2}' "$elog" | tail -1)"
            [ -n "$v" ] || v="ERR"
            row_errs+="    !! ${exp} @ ${name}: eval error (see ${elog})"$'\n'
        fi
        printf "  %${MCOL[$i]}s" "$v"   # append + flush this cell immediately
        printf '%s\t%s\t%s\n' "$exp" "$name" "$v" >> "$RESULTS_TSV"
        i=$((i + 1))
    done
    printf "\n"                          # terminate the row
    [ -n "$row_errs" ] && printf '%s' "$row_errs" >&2
done

# ---------- AVERAGE row (numeric cells only) ----------
sep
avgs=()
for exp in "${READY[@]}"; do
    avgs+=("$(awk -F'\t' -v m="$exp" \
        '$1==m && $3 ~ /^[0-9]+(\.[0-9]+)?$/ {s+=$3; c++} END{ if (c>0) printf "%.2f", s/c; else printf "--" }' \
        "$RESULTS_TSV")")
done
print_row "AVERAGE" "${avgs[@]}"

# ---------- Epilogue (STDERR) ----------
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "" >&2
    echo "WARNING: ${#FAILED[@]} model(s) failed to prepare: ${FAILED[*]}" >&2
    echo "Logs under ${LOGDIR}" >&2
    exit 1
fi
