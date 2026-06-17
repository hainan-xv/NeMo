#!/bin/bash
set -o pipefail

# ============================================================================
# Likelihood-selected time-warp eval over the Open-ASR-Leaderboard datasets.
#
# For each utterance, decode the requested warp factors and choose the hypothesis
# with the highest decoder score (default: score normalized by token count).
# References are used only for reporting WER.
#
# Usage:
#   ./eval_likelihood_timewarp.sh /abs/path/model.nemo
#   MODEL=/abs/path/model.nemo ./eval_likelihood_timewarp.sh
#
# Env overrides:
#   MODEL          local .nemo / .ckpt (or pass it as the first argument)
#   TOKENIZER_DIR  tokenizer override for .ckpt models
#   DATASET_PATH   HF dataset path (default hf-audio/esb-datasets-test-only-sorted)
#   FACTORS        warp factors (default 0.9,1.0,1.1; 1.0 auto-added if absent)
#   METHOD         time_stretch | speed   (default time_stretch)
#   SCORE_NORM     none | token | word | char   (default token)
#   DEVICE         GPU id (default 0)
#   BATCH_SIZE     (default 32)
#   MAX_EVAL_SAMPLES  cap wavs per dataset (default: none = full)
#   ONLY           comma-separated dataset tags to restrict to
#   OUT_DIR        where per-utterance reports + log go (default likelihood_results/)
# ============================================================================

NEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${NEMO_ROOT}/.cache/flashinfer}"
export HF_AUDIO_DECODER=soundfile
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -n "${1:-}" ]; then
    MODEL="$1"
fi
if [ -z "${MODEL:-}" ]; then
    echo "ERROR: set MODEL=/abs/path/model.nemo or run $0 /abs/path/model.nemo" >&2; exit 1
fi
if [ ! -e "$MODEL" ]; then
    echo "ERROR: model not found: $MODEL" >&2; exit 1
fi
case "$MODEL" in
    *.nemo|*.ckpt) ;;
    *) echo "ERROR: MODEL must be a .nemo or .ckpt checkpoint: $MODEL" >&2; exit 1 ;;
esac

FACTORS="${FACTORS:-0.9,1.0,1.1}"
METHOD="${METHOD:-time_stretch}"
SCORE_NORM="${SCORE_NORM:-token}"
DEVICE="${DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DATASET_PATH="${DATASET_PATH:-hf-audio/esb-datasets-test-only-sorted}"
OUT_DIR="${OUT_DIR:-${NEMO_ROOT}/likelihood_results}"
mkdir -p "$OUT_DIR"

DATASETS=(
    "ami_test"
    "earnings22_test"
    "gigaspeech_test"
    "librispeech_test.clean"
    "librispeech_test.other"
    "spgispeech_test"
    "voxpopuli_test"
)
if [ -n "${ONLY:-}" ]; then
    IFS=',' read -r -a DATASETS <<< "$ONLY"
fi

EXTRA=()
[ -n "${MAX_EVAL_SAMPLES:-}" ] && EXTRA+=(--max_samples "$MAX_EVAL_SAMPLES")
[ -n "${TOKENIZER_DIR:-}" ] && EXTRA+=(--tokenizer_dir "$TOKENIZER_DIR")

LOG="${OUT_DIR}/likelihood_timewarp_$(basename "$MODEL").log"
: > "$LOG"
echo "==> model=$MODEL"
echo "==> dataset_path=$DATASET_PATH"
echo "==> factors=$FACTORS  method=$METHOD  score_norm=$SCORE_NORM  device=$DEVICE"
echo "==> logging to $LOG"

declare -A BASELINE BESTFIX SELECTED ORACLE SCORED TOTAL PICKS ORACLE_PICKS AGREE
declare -A FACTOR_WERS FACTOR_WER_SUMS
IFS=',' read -r -a FACTOR_LIST <<< "$FACTORS"
for ds in "${DATASETS[@]}"; do
    echo "running likelihood time-warp on ${ds}..."
    run_log=$(mktemp)
    python "${NEMO_ROOT}/likelihood_timewarp_leaderboard.py" \
        --model "$MODEL" \
        --dataset "$ds" \
        --dataset_path "$DATASET_PATH" \
        --factors "$FACTORS" \
        --method "$METHOD" \
        --score_norm "$SCORE_NORM" \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --output "${OUT_DIR}/likelihood_${ds}.jsonl" \
        "${EXTRA[@]}" > "$run_log" 2>&1
    rc=$?
    { echo "==================== ${ds} ===================="; cat "$run_log"; } >> "$LOG"
    if [ "$rc" -ne 0 ]; then
        echo "  ${ds} FAILED (exit ${rc}) -- see ${LOG}"
        BASELINE[$ds]="FAIL"; BESTFIX[$ds]="FAIL"; SELECTED[$ds]="FAIL"; ORACLE[$ds]="FAIL"
        SCORED[$ds]="?"; TOTAL[$ds]="?"; PICKS[$ds]="?"; ORACLE_PICKS[$ds]="?"; AGREE[$ds]="?"
        FACTOR_WERS[$ds]="?"
    else
        line=$(grep '^LIKELIHOOD_SUMMARY' "$run_log" | tail -1)
        SCORED[$ds]=$(echo "$line" | grep -oE 'scored=[0-9]+' | cut -d= -f2)
        TOTAL[$ds]=$(echo "$line" | grep -oE 'total=[0-9]+' | cut -d= -f2)
        BASELINE[$ds]=$(echo "$line" | grep -oE 'baseline=[0-9.]+' | cut -d= -f2)
        BESTFIX[$ds]=$(echo "$line" | grep -oE 'best_fixed=[0-9.]+' | cut -d= -f2)
        SELECTED[$ds]=$(echo "$line" | grep -oE 'selected=[0-9.]+' | cut -d= -f2)
        ORACLE[$ds]=$(echo "$line" | grep -oE 'oracle=[0-9.]+' | cut -d= -f2)
        AGREE[$ds]=$(echo "$line" | grep -oE 'agreement=[0-9.]+' | cut -d= -f2)
        FACTOR_WERS[$ds]=$(echo "$line" | tr ' ' '\n' | awk -F= '$1=="factor_wers"{print $2}')
        PICKS[$ds]=$(echo "$line" | tr ' ' '\n' | awk -F= '$1=="picks"{print $2}')
        ORACLE_PICKS[$ds]=$(echo "$line" | tr ' ' '\n' | awk -F= '$1=="oracle_picks"{print $2}')
        echo "  ${ds}: factor_WERs=${FACTOR_WERS[$ds]} selected=${SELECTED[$ds]} oracle=${ORACLE[$ds]} (scored ${SCORED[$ds]}/${TOTAL[$ds]}; picks ${PICKS[$ds]}; agree ${AGREE[$ds]}%)"
    fi
    rm -f "$run_log"
done

echo ""
echo "================ Likelihood time-warp summary ================"
echo "model     : $(basename "$MODEL")"
echo "factors   : $FACTORS   method: $METHOD   score_norm: $SCORE_NORM"
printf "  %-24s %12s" "dataset" "scored"
for f in "${FACTOR_LIST[@]}"; do
    printf " %8s" "x${f}"
done
printf " %8s %8s %8s  %s\n" "SELECT" "ORACLE" "agree%" "selected_pick_%"
sb=0; ss=0; so=0; cnt=0
for ds in "${DATASETS[@]}"; do
    printf "  %-24s %12s" "$ds" "${SCORED[$ds]:-?}/${TOTAL[$ds]:-?}"
    for f in "${FACTOR_LIST[@]}"; do
        fw=$(echo "${FACTOR_WERS[$ds]:-}" | tr ',' '\n' | awk -F: -v k="x${f}" '$1==k{print $2}')
        printf " %8s" "${fw:-NA}"
    done
    printf " %8s %8s %8s  %s\n" "${SELECTED[$ds]:-NA}" "${ORACLE[$ds]:-NA}" "${AGREE[$ds]:-NA}" "${PICKS[$ds]:-?}"
    if [[ "${BASELINE[$ds]}" =~ ^[0-9.]+$ ]] && [[ "${SELECTED[$ds]}" =~ ^[0-9.]+$ ]] && [[ "${ORACLE[$ds]}" =~ ^[0-9.]+$ ]]; then
        sb=$(echo "$sb + ${BASELINE[$ds]}" | bc -l)
        ss=$(echo "$ss + ${SELECTED[$ds]}" | bc -l)
        so=$(echo "$so + ${ORACLE[$ds]}" | bc -l)
        for f in "${FACTOR_LIST[@]}"; do
            fw=$(echo "${FACTOR_WERS[$ds]:-}" | tr ',' '\n' | awk -F: -v k="x${f}" '$1==k{print $2}')
            if [[ "$fw" =~ ^[0-9.]+$ ]]; then
                FACTOR_WER_SUMS[$f]=$(echo "${FACTOR_WER_SUMS[$f]:-0} + $fw" | bc -l)
            fi
        done
        cnt=$((cnt+1))
    fi
done
if [ "$cnt" -gt 0 ]; then
    printf "  %-24s %12s" "----" ""
    for _f in "${FACTOR_LIST[@]}"; do
        printf " %8s" "----"
    done
    printf " %8s %8s %8s  %s\n" "----" "----" "" ""

    printf "  %-24s %12s" "AVERAGE" ""
    for f in "${FACTOR_LIST[@]}"; do
        if [ -n "${FACTOR_WER_SUMS[$f]:-}" ]; then
            printf " %8.2f" "$(echo "${FACTOR_WER_SUMS[$f]}/$cnt" | bc -l)"
        else
            printf " %8s" "NA"
        fi
    done
    printf " %8.2f %8.2f %8s  %s\n" "$(echo "$ss/$cnt" | bc -l)" "$(echo "$so/$cnt" | bc -l)" "" ""
fi
echo "==============================================================="
echo "Full log: ${LOG}"
