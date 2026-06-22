#!/bin/bash
set -e
set -o pipefail

# ============================================================================
# Likelihood-selected time-warp CER eval for Mandarin ASR checkpoints over the
# local AISHELL-2 test manifests.
#
# For each test utterance, decode several time-warped copies and choose the
# hypothesis with the highest decoder score. References are used only to report
# CER, plus an oracle diagnostic for comparison.
#
# Usage:
#   MODEL=/abs/path/model.nemo ./eval_likelihood_timewarp_aishell.sh
#   ./eval_likelihood_timewarp_aishell.sh /abs/path/model.nemo [DEVICE_ID]
#
# Env overrides:
#   FACTORS        warp factors (default 0.9,1.0,1.1; 1.0 auto-added if absent)
#   METHOD         speed | time_stretch   (default speed)
#   SCORE_NORM     logprob_token | logprob_token_dur | none | token | char   (default logprob_token)
#   SCORE_EPSILON  switch from x1.0 only if gap >= epsilon  (default 0.01)
#   MANIFEST_DIR / AUDIO_SRC_PREFIX / AUDIO_DST_PREFIX   dataset paths
#   SETS / ONLY    test sets (default "test_android test_ios test_mic")
#   BATCH_SIZE     dataloader batch size (default 32)
#   MAX_EVAL_SAMPLES   cap samples per set (fast iteration)
#   KEEP_SPACES    1 -> do NOT collapse whitespace before CER
#   OUT_DIR        per-utterance reports + log (default likelihood_results_aishell/)
# ============================================================================

NEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
EVAL_PY="${NEMO_ROOT}/likelihood_timewarp_aishell.py"

export PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${NEMO_ROOT}/.cache/flashinfer}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -f "$EVAL_PY" ]; then
    echo "ERROR: cannot find likelihood driver at ${EVAL_PY}" >&2
    exit 1
fi

MANIFEST_DIR="${MANIFEST_DIR:-/home/hainanx/Workplace/data/aishell_eval/aishell/manifests}"
AUDIO_SRC_PREFIX="${AUDIO_SRC_PREFIX:-/data/mandarin/aishell2/evaluation/aishell}"
AUDIO_DST_PREFIX="${AUDIO_DST_PREFIX:-$(dirname "$MANIFEST_DIR")}"

if [ -n "${ONLY:-}" ]; then
    SETS="${ONLY//,/ }"
fi
SETS="${SETS:-test_android test_ios test_mic}"
SETS="${SETS//,/ }"

FACTORS="${FACTORS:-0.9,1.0,1.1}"
METHOD="${METHOD:-speed}"
SCORE_NORM="${SCORE_NORM:-logprob_token}"
# Only switch away from x1.0 if another factor's score exceeds x1.0 by >= SCORE_EPSILON.
# 0.01 is empirically optimal for logprob_token on AISHELL-2.  Set 0 to disable (pure argmax).
SCORE_EPSILON="${SCORE_EPSILON:-0.01}"
OUT_DIR="${OUT_DIR:-${NEMO_ROOT}/likelihood_results_aishell}"
BATCH_SIZE="${BATCH_SIZE:-32}"

if [ -n "${1:-}" ]; then
    MODEL="$1"
fi
DEVICE_ID="${2:-${DEVICE:-0}}"
LOCAL_CKPT_PATH="${MODEL:-}"

if [ -z "$LOCAL_CKPT_PATH" ]; then
    echo "Usage: MODEL=/abs/path/model.nemo $0   or   $0 /abs/path/model.nemo [DEVICE_ID]" >&2
    echo "ERROR: no local .nemo model path provided." >&2
    exit 1
fi
if [ ! -e "$LOCAL_CKPT_PATH" ]; then
    echo "ERROR: checkpoint not found at $LOCAL_CKPT_PATH" >&2
    exit 1
fi
case "$LOCAL_CKPT_PATH" in
    *.nemo) ;;
    *) echo "ERROR: checkpoint must be a local .nemo file: $LOCAL_CKPT_PATH" >&2; exit 1 ;;
esac

EXTRA_ARGS=()
[ -n "$AUDIO_SRC_PREFIX" ] && EXTRA_ARGS+=(--audio_src_prefix "$AUDIO_SRC_PREFIX")
[ -n "$AUDIO_DST_PREFIX" ] && EXTRA_ARGS+=(--audio_dst_prefix "$AUDIO_DST_PREFIX")
[ -n "${MAX_EVAL_SAMPLES:-}" ] && EXTRA_ARGS+=(--max_eval_samples "$MAX_EVAL_SAMPLES")
[ -n "${TOKENIZER_DIR:-}" ] && EXTRA_ARGS+=(--tokenizer_dir "$TOKENIZER_DIR")
[ -n "${MAX_SYMBOLS_PER_STEP:-}" ] && EXTRA_ARGS+=(--max_symbols_per_step "$MAX_SYMBOLS_PER_STEP")
[ "${KEEP_SPACES:-0}" = "1" ] && EXTRA_ARGS+=(--keep_spaces)

mkdir -p "$OUT_DIR"
EVAL_LOG="${OUT_DIR}/likelihood_timewarp_aishell_$(basename "${LOCAL_CKPT_PATH}").log"
: > "$EVAL_LOG"
echo "==> model=${LOCAL_CKPT_PATH}"
echo "==> factors=${FACTORS}  method=${METHOD}  score_norm=${SCORE_NORM}  epsilon=${SCORE_EPSILON}  sets='${SETS}'  device=${DEVICE_ID}"
echo "==> Logging to $EVAL_LOG"

IFS=',' read -r -a FACTOR_LIST <<< "$FACTORS"
_has_one=0
for f in "${FACTOR_LIST[@]}"; do [ "$f" = "1.0" ] && _has_one=1; done
if [ "$_has_one" -eq 0 ]; then
    FACTOR_LIST=("1.0" "${FACTOR_LIST[@]}")
fi

declare -A BASELINE BESTFIX SELECTED ORACLE SCORED PICKS ORACLE_PICKS AGREE FACTOR_CERS
for set_name in $SETS; do
    manifest="${MANIFEST_DIR}/${set_name}.json"
    if [ ! -f "$manifest" ]; then
        echo "  ${set_name}: manifest not found at ${manifest} -- skipping"
        BASELINE[$set_name]="NOFILE"; BESTFIX[$set_name]="NOFILE"; SELECTED[$set_name]="NOFILE"; ORACLE[$set_name]="NOFILE"; SCORED[$set_name]="?"
        continue
    fi

    echo "running likelihood time-warp on ${set_name}..."
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
        --score_norm "$SCORE_NORM" \
        --score_epsilon "$SCORE_EPSILON" \
        --device "$DEVICE_ID" \
        --batch_size "$BATCH_SIZE" \
        --output "${OUT_DIR}/likelihood_${set_name}.jsonl" \
        "${EXTRA_ARGS[@]}" \
        > "$run_log" 2>&1
    rc=$?
    set -e
    cat "$run_log" >> "$EVAL_LOG"

    if [ "$rc" -ne 0 ]; then
        echo "  ${set_name} FAILED (exit ${rc}) -- see ${EVAL_LOG}"
        BASELINE[$set_name]="FAIL"; BESTFIX[$set_name]="FAIL"; SELECTED[$set_name]="FAIL"; ORACLE[$set_name]="FAIL"
        SCORED[$set_name]="?"; PICKS[$set_name]="?"; ORACLE_PICKS[$set_name]="?"; AGREE[$set_name]="?"; FACTOR_CERS[$set_name]="?"
    else
        line=$(grep '^LIKELIHOOD_SUMMARY' "$run_log" | tail -1)
        SCORED[$set_name]=$(echo "$line" | grep -oE 'scored=[0-9]+' | cut -d= -f2)
        BASELINE[$set_name]=$(echo "$line" | grep -oE 'baseline=[0-9.]+' | cut -d= -f2)
        BESTFIX[$set_name]=$(echo "$line" | grep -oE 'best_fixed=[0-9.]+' | cut -d= -f2)
        SELECTED[$set_name]=$(echo "$line" | grep -oE 'selected=[0-9.]+' | cut -d= -f2)
        ORACLE[$set_name]=$(echo "$line" | grep -oE 'oracle=[0-9.]+' | cut -d= -f2)
        AGREE[$set_name]=$(echo "$line" | grep -oE 'agreement=[0-9.]+' | cut -d= -f2)
        FACTOR_CERS[$set_name]=$(echo "$line" | tr ' ' '\n' | awk -F= '$1=="factor_cers"{print $2}')
        PICKS[$set_name]=$(echo "$line" | tr ' ' '\n' | awk -F= '$1=="picks"{print $2}')
        ORACLE_PICKS[$set_name]=$(echo "$line" | tr ' ' '\n' | awk -F= '$1=="oracle_picks"{print $2}')
        echo "  ${set_name}: factor_CERs=${FACTOR_CERS[$set_name]} selected=${SELECTED[$set_name]} oracle=${ORACLE[$set_name]} (scored ${SCORED[$set_name]}; picks ${PICKS[$set_name]}; agree ${AGREE[$set_name]}%)"
    fi
    rm -f "$run_log"
done

echo ""
echo "================ AISHELL likelihood time-warp CER summary ================"
echo "ckpt   : $(basename "${LOCAL_CKPT_PATH}")"
echo "factors: ${FACTORS}   method: ${METHOD}   score_norm: ${SCORE_NORM}   epsilon: ${SCORE_EPSILON}   metric: CER (%)"
printf "  %-22s %8s" "set" "scored"
for f in "${FACTOR_LIST[@]}"; do printf " %8s" "x${f}"; done
printf " %8s %8s %8s  %s\n" "SELECT" "ORACLE" "agree%" "selected_pick_%"

declare -A FACTOR_SUMS
ss=0; so=0; cnt=0
for set_name in $SETS; do
    printf "  %-22s %8s" "$set_name" "${SCORED[$set_name]:-?}"
    for f in "${FACTOR_LIST[@]}"; do
        fc=$(echo "${FACTOR_CERS[$set_name]:-}" | tr ',' '\n' | awk -F: -v k="x${f}" '$1==k{print $2}')
        printf " %8s" "${fc:-NA}"
    done
    printf " %8s %8s %8s  %s\n" "${SELECTED[$set_name]:-NA}" "${ORACLE[$set_name]:-NA}" "${AGREE[$set_name]:-NA}" "${PICKS[$set_name]:-?}"
    if [[ "${SELECTED[$set_name]}" =~ ^[0-9.]+$ ]] && [[ "${ORACLE[$set_name]}" =~ ^[0-9.]+$ ]]; then
        ss=$(echo "$ss + ${SELECTED[$set_name]}" | bc -l)
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
    printf " %8s %8s %8s  %s\n" "----" "----" "" ""

    printf "  %-22s %8s" "AVERAGE" ""
    for f in "${FACTOR_LIST[@]}"; do
        if [ -n "${FACTOR_SUMS[$f]:-}" ]; then
            printf " %8.2f" "$(echo "${FACTOR_SUMS[$f]}/$cnt" | bc -l)"
        else
            printf " %8s" "NA"
        fi
    done
    printf " %8.2f %8.2f %8s  %s\n" "$(echo "$ss/$cnt" | bc -l)" "$(echo "$so/$cnt" | bc -l)" "" ""
fi
echo "========================================================================="
echo "Full log: ${EVAL_LOG}"
