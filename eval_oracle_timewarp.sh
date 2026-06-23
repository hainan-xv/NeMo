#!/bin/bash
set -o pipefail

# ============================================================================
# Oracle time-warp ("speed cheat") eval over the Open-ASR-Leaderboard datasets,
# using the same HF dataset loading + local audio caching path as eval_asr_ord.sh.
#
# Mirrors eval_asr_ord.sh's dataset loop + summary table, but instead of a single
# WER it reports, per dataset: baseline (x1.0) WER, best single fixed warp, and
# the ORACLE best-of-N WER (cheating: per-utterance pick the lowest-WER warp).
#
# Usage:
#   ./eval_oracle_timewarp.sh /abs/path/model.nemo
#   MODEL=/abs/path/model.nemo ./eval_oracle_timewarp.sh
# Env overrides:
#   MODEL          local .nemo / .ckpt (or pass it as the first argument)
#   DATASET_PATH   HF dataset path (default hf-audio/esb-datasets-test-only-sorted)
#   FACTORS        warp factors (default 0.9,1.0,1.1; 1.0 auto-added if absent)
#   METHOD         time_stretch | speed   (default time_stretch)
#   DEVICE         GPU id (default 0)
#   BATCH_SIZE     (default 32)
#   MAX_EVAL_SAMPLES  cap wavs per dataset (default: none = full)
#   ONLY           comma-separated dataset tags to restrict to
#   OUT_DIR        where per-utterance reports + log go (default oracle_results/)
# ============================================================================

NEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba}"
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
DEVICE="${DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DATASET_PATH="${DATASET_PATH:-hf-audio/esb-datasets-test-only-sorted}"
OUT_DIR="${OUT_DIR:-${NEMO_ROOT}/oracle_results}"
# LM-based (non-cheating) warp selection. Set LM_MODEL="" to disable.
LM_MODEL="${LM_MODEL:-distilgpt2}"
LM_EPSILON="${LM_EPSILON:-0.0}"
LM_BATCH_SIZE="${LM_BATCH_SIZE:-16}"
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
# Needed when MODEL is a .ckpt (tokenizer lives outside the checkpoint); harmless for .nemo.
[ -n "${TOKENIZER_DIR:-}" ] && EXTRA+=(--tokenizer_dir "$TOKENIZER_DIR")
# LM-based warp selection (distilgpt2 by default); compares against oracle/baseline.
if [ -n "$LM_MODEL" ]; then
    EXTRA+=(--lm_model "$LM_MODEL" --lm_epsilon "$LM_EPSILON" --lm_batch_size "$LM_BATCH_SIZE")
fi

LOG="${OUT_DIR}/oracle_timewarp_$(basename "$MODEL").log"
: > "$LOG"
echo "==> model=$MODEL"
echo "==> dataset_path=$DATASET_PATH"
echo "==> factors=$FACTORS  method=$METHOD  device=$DEVICE"
echo "==> lm_select=${LM_MODEL:-<off>}  lm_epsilon=$LM_EPSILON"
echo "==> logging to $LOG"

declare -A BASELINE BESTFIX ORACLE SCORED TOTAL PICKS LMSEL LMPICKS
for ds in "${DATASETS[@]}"; do
    echo "running oracle time-warp on ${ds}..."
    run_log=$(mktemp)
    python "${NEMO_ROOT}/oracle_timewarp_leaderboard.py" \
        --model "$MODEL" \
        --dataset "$ds" \
        --dataset_path "$DATASET_PATH" \
        --factors "$FACTORS" \
        --method "$METHOD" \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --output "${OUT_DIR}/oracle_${ds}.jsonl" \
        "${EXTRA[@]}" > "$run_log" 2>&1
    rc=$?
    { echo "==================== ${ds} ===================="; cat "$run_log"; } >> "$LOG"
    if [ "$rc" -ne 0 ]; then
        echo "  ${ds} FAILED (exit ${rc}) -- see ${LOG}"
        BASELINE[$ds]="FAIL"; BESTFIX[$ds]="FAIL"; ORACLE[$ds]="FAIL"; SCORED[$ds]="?"; TOTAL[$ds]="?"; PICKS[$ds]="?"; LMSEL[$ds]="FAIL"; LMPICKS[$ds]="?"
    else
        line=$(grep '^ORACLE_SUMMARY' "$run_log" | tail -1)
        SCORED[$ds]=$(echo "$line" | grep -oE 'scored=[0-9]+' | cut -d= -f2)
        TOTAL[$ds]=$(echo "$line" | grep -oE 'total=[0-9]+' | cut -d= -f2)
        BASELINE[$ds]=$(echo "$line" | grep -oE 'baseline=[0-9.]+' | cut -d= -f2)
        BESTFIX[$ds]=$(echo "$line" | grep -oE 'best_fixed=[0-9.]+' | cut -d= -f2)
        ORACLE[$ds]=$(echo "$line" | grep -oE 'oracle=[0-9.]+' | cut -d= -f2)
        PICKS[$ds]=$(echo "$line" | grep -oE 'picks=[^[:space:]]+' | cut -d= -f2)
        lmline=$(grep '^LM_SUMMARY' "$run_log" | tail -1)
        if [ -n "$lmline" ]; then
            LMSEL[$ds]=$(echo "$lmline" | grep -oE 'lm_selected=[0-9.]+' | cut -d= -f2)
            LMPICKS[$ds]=$(echo "$lmline" | grep -oE 'picks=[^[:space:]]+' | cut -d= -f2)
        else
            LMSEL[$ds]="NA"; LMPICKS[$ds]="NA"
        fi
        echo "  ${ds}: baseline=${BASELINE[$ds]} best_fixed=${BESTFIX[$ds]} lm=${LMSEL[$ds]} oracle=${ORACLE[$ds]} (scored ${SCORED[$ds]}/${TOTAL[$ds]}; picks ${PICKS[$ds]})"
    fi
    rm -f "$run_log"
done

echo ""
echo "================== Oracle time-warp summary =================="
echo "model  : $(basename "$MODEL")"
echo "factors: $FACTORS   method: $METHOD   lm: ${LM_MODEL:-<off>} (eps=$LM_EPSILON)"
printf "  %-24s %12s %8s %8s %8s %8s  %s\n" "dataset" "scored" "base" "bestfix" "LM-SEL" "ORACLE" "lm_pick_%"
sb=0; so=0; sl=0; cnt=0; lcnt=0
for ds in "${DATASETS[@]}"; do
    printf "  %-24s %12s %8s %8s %8s %8s  %s\n" "$ds" "${SCORED[$ds]:-?}/${TOTAL[$ds]:-?}" "${BASELINE[$ds]:-NA}" "${BESTFIX[$ds]:-NA}" "${LMSEL[$ds]:-NA}" "${ORACLE[$ds]:-NA}" "${LMPICKS[$ds]:-?}"
    if [[ "${BASELINE[$ds]}" =~ ^[0-9.]+$ ]] && [[ "${ORACLE[$ds]}" =~ ^[0-9.]+$ ]]; then
        sb=$(echo "$sb + ${BASELINE[$ds]}" | bc -l); so=$(echo "$so + ${ORACLE[$ds]}" | bc -l); cnt=$((cnt+1))
    fi
    if [[ "${LMSEL[$ds]}" =~ ^[0-9.]+$ ]]; then
        sl=$(echo "$sl + ${LMSEL[$ds]}" | bc -l); lcnt=$((lcnt+1))
    fi
done
if [ "$cnt" -gt 0 ]; then
    printf "  %-24s %12s %8s %8s %8s %8s  %s\n" "----" "" "----" "" "----" "----" ""
    lm_avg="NA"
    [ "$lcnt" -gt 0 ] && lm_avg=$(printf "%.2f" "$(echo "$sl/$lcnt" | bc -l)")
    printf "  %-24s %12s %8.2f %8s %8s %8.2f  %s\n" "AVERAGE" "" "$(echo "$sb/$cnt" | bc -l)" "" "$lm_avg" "$(echo "$so/$cnt" | bc -l)" ""
fi
echo "============================================================="
echo "Full log: ${LOG}"
