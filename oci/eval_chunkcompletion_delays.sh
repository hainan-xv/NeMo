#!/bin/bash
# ============================================================================
# Evaluate the flexible-prompt (multi-delay) chunk-completion model at each of
# its three trained latency settings (delay 0 / 2 / 4) on the Open-ASR
# leaderboard, via eval_leaderboard_oci.sh.
#
# At inference the "delay constraint" is chosen purely by WHICH prompt is passed
# (the delay was a training-time alignment shift; the prompt selects behavior).
# So we run the eval once per prompt, each with a distinct RUN_TAG so their
# results/logs/HF dirs don't collide. The prompts here must match the recipe's
# data.dataset.delay_prompts verbatim (streaming_stt_granary2_lora_chunkcompletion_delayprompt.yaml).
#
# Usage:
#   ./oci/eval_chunkcompletion_delays.sh [EXP_NAME] [GPU]
#     EXP_NAME  grid experiment name (default: granary2_chunkcompletion_delayprompt)
#     GPU       device id (default: 0)
#
# Env overrides (all forwarded to eval_leaderboard_oci.sh):
#   DECODE_BACKEND  sslm (default; no HF conversion, has OOM backoff) | heh | vllm
#   CHUNK_SIZE      encoder frames/chunk to decode at (default 14)
#   DELAYS          space-separated subset of "0 2 4" to run (default: all three)
#   RUN_AVERAGING   1 (default) -> average top-k checkpoints; 0 -> best single
#   QUICK_TEST      1 -> 10 utts from ami only (fast smoke test per prompt)
#   ONLY            comma-separated dataset filter (e.g. "librispeech,gigaspeech")
#   CUSTOM_PROMPT   if set, run ONE eval with this exact prompt instead of the
#                   0/2/4 sweep (to test an arbitrary/experimental prompt).
#   CUSTOM_TAG      RUN_TAG label for the custom-prompt run (default: "custom").
#   plus any other eval_leaderboard_oci.sh knob (HEH_*, BATCH_SIZE, MODEL_CLASS, ...).
#
# Examples:
#   ./oci/eval_chunkcompletion_delays.sh                       # all 3 delays, sslm, chunk 14
#   DELAYS="0 4" CHUNK_SIZE=7 ./oci/eval_chunkcompletion_delays.sh
#   QUICK_TEST=1 ./oci/eval_chunkcompletion_delays.sh granary2_chunkcompletion_delayprompt 1
#   DECODE_BACKEND=heh HEH_BATCH_SIZE=16 HEH_MAX_NEW_TOKENS=64 ./oci/eval_chunkcompletion_delays.sh
#   CUSTOM_PROMPT="Transcribe the current chunk given the history." CUSTOM_TAG=myprompt \
#     ./oci/eval_chunkcompletion_delays.sh   # single run with a custom prompt
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL="$REPO_ROOT/eval_leaderboard_oci.sh"
if [ ! -x "$EVAL" ] && [ ! -f "$EVAL" ]; then
    echo "ERROR: eval driver not found at $EVAL" >&2
    exit 1
fi

EXP_NAME="${1:-granary2_chunkcompletion_delayprompt}"
GPU="${2:-0}"

# Decode config (overridable via env; sslm avoids per-prompt HF re-conversion).
DECODE_BACKEND="${DECODE_BACKEND:-sslm}"
CHUNK_SIZE="${CHUNK_SIZE:-14}"
RUN_AVERAGING="${RUN_AVERAGING:-1}"
DELAYS="${DELAYS:-0 2 4}"

MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.chunk_completion_model.ChunkCompletionSTTModel}"

# Shared instruction stem; the per-delay clause is appended below. These MUST
# match the recipe's delay_prompts exactly (the model keys behavior on the text).
CORE="You are doing streaming speech recognition. Given the transcript so far and the next audio chunk, output the words spoken in that chunk."
prompt_for_delay() {
    case "$1" in
        0) echo "$CORE Emit the words as soon as possible, minimizing latency." ;;
        2) echo "$CORE You may wait a little to gather more context before emitting, trading a small delay for better accuracy." ;;
        4) echo "$CORE Wait until you are confident before emitting, prioritizing accuracy over latency." ;;
        *) echo "ERROR: no prompt defined for delay=$1 (expected 0, 2, or 4)" >&2; return 1 ;;
    esac
}

# RUN_TAG suffix so a chunk-size sweep on top of the delay sweep stays isolated.
CHUNK_TAG="chunk${CHUNK_SIZE}"

# --- Custom-prompt mode: run ONE eval with an arbitrary prompt, skip the sweep ---
CUSTOM_PROMPT="${CUSTOM_PROMPT:-}"
CUSTOM_TAG="${CUSTOM_TAG:-custom}"
if [ -n "$CUSTOM_PROMPT" ]; then
    tag="${CUSTOM_TAG}_${CHUNK_TAG}"
    echo "==> Custom-prompt eval | exp=${EXP_NAME} | gpu=${GPU} | backend=${DECODE_BACKEND} | chunk=${CHUNK_SIZE} | RUN_TAG=${tag}"
    echo "    prompt: ${CUSTOM_PROMPT}"
    RUN_TAG="$tag" \
    SYSTEM_PROMPT="$CUSTOM_PROMPT" \
    MODEL_CLASS="$MODEL_CLASS" \
    DECODE_BACKEND="$DECODE_BACKEND" \
    RUN_AVERAGING="$RUN_AVERAGING" \
    CHUNK_SIZE="$CHUNK_SIZE" \
    bash "$EVAL" --gpu "$GPU" "$EXP_NAME"
    echo ""
    echo "==> Done. Results under \$LEADERBOARD_RUN/results/${EXP_NAME}_step*_${tag}/"
    exit 0
fi

echo "==> Flexible-prompt eval | exp=${EXP_NAME} | gpu=${GPU} | backend=${DECODE_BACKEND} | chunk=${CHUNK_SIZE} | delays='${DELAYS}'"
for d in $DELAYS; do
    prompt="$(prompt_for_delay "$d")"
    tag="delay${d}_${CHUNK_TAG}"
    echo ""
    echo "==================== delay=${d}  (RUN_TAG=${tag}) ===================="
    echo "    prompt: ${prompt}"
    RUN_TAG="$tag" \
    SYSTEM_PROMPT="$prompt" \
    MODEL_CLASS="$MODEL_CLASS" \
    DECODE_BACKEND="$DECODE_BACKEND" \
    RUN_AVERAGING="$RUN_AVERAGING" \
    CHUNK_SIZE="$CHUNK_SIZE" \
    bash "$EVAL" --gpu "$GPU" "$EXP_NAME"
done

echo ""
echo "==> Done. Results under \$LEADERBOARD_RUN/results/${EXP_NAME}_step*_delay{${DELAYS// /,}}_${CHUNK_TAG}/"
