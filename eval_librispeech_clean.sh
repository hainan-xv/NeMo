#!/bin/bash
set -e
set -o pipefail

# ============================================================================
# Evaluate an ASR checkpoint on LibriSpeech test-clean and store the RAW
# (un-normalized) reference / hypothesis text in a JSON file.
#
# Unlike eval_asr_ord.sh / run_eval_asr.py (which whisper-normalize the text
# before writing results), this keeps casing + punctuation verbatim.
#
# Usage:
#   ./eval_librispeech_clean.sh <modelfile> <output.json> [--quick_run]
#     --quick_run   only decode the first 10 utterances of test-clean
#
# Env overrides:
#   DEVICE_ID      CUDA device id (default 0)
#   BATCH_SIZE     dataloader batch size (default 32)
#   TOKENIZER_DIR  tokenizer dir for .ckpt loads (needed for some .ckpt files)
# ============================================================================

NEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH}"
RUN_PY="${NEMO_ROOT}/eval_librispeech_raw.py"

DEVICE_ID="${DEVICE_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TOKENIZER_DIR="${TOKENIZER_DIR:-}"
QUICK_RUN=0

# ---------- Parse args (flags + 2 positionals) ----------
POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --quick_run) QUICK_RUN=1; shift ;;
        --tokenizer_dir) TOKENIZER_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 <modelfile> <output.json> [--quick_run]"; exit 0 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]}"

MODEL="${1:-}"
OUTPUT="${2:-}"

if [ -z "$MODEL" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 <modelfile> <output.json> [--quick_run]" >&2
    exit 1
fi
if [ ! -e "$MODEL" ]; then
    echo "ERROR: model file not found: $MODEL" >&2
    exit 1
fi
if [ ! -f "$RUN_PY" ]; then
    echo "ERROR: cannot find driver at ${RUN_PY}" >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA_ARGS=()
[ -n "$TOKENIZER_DIR" ] && EXTRA_ARGS+=(--tokenizer_dir "$TOKENIZER_DIR")
if [ "$QUICK_RUN" = "1" ]; then
    echo "==> QUICK RUN: first 10 utterances of librispeech/test.clean"
    EXTRA_ARGS+=(--max_eval_samples 10)
fi

echo "==> Model : $MODEL"
echo "==> Output: $OUTPUT"

python "$RUN_PY" \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --dataset "librispeech" \
    --split "test.clean" \
    --device "$DEVICE_ID" \
    --batch_size "$BATCH_SIZE" \
    "${EXTRA_ARGS[@]}"
