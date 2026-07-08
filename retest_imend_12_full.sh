#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate the NEW multi-chunk model `imend_12` at a latency that is
# APPLES-TO-APPLES with the old ~7-WER maxChunks4 model's 1-chunk numbers.
#
# Why chunks/turn=2 here:
#   imend_12 was trained with chunk_size=7 (0.56 s) frames.
#   The old maxChunks4 model used chunk_size=14 (1.12 s) frames.
#   So ONE 14-frame turn of the old model == TWO 7-frame turns of imend_12.
#   => `imend_12 @ inference_audio_chunks_per_turn=2` consumes 2*7 = 14 audio
#      frames per LLM turn, i.e. the SAME effective chunk / latency as
#      `maxChunks4 @ inference_audio_chunks_per_turn=1`.
#   This is the fair comparison against the old model's chunks=1 column.
#
# NOTE: imend_12 also differs from the old model in delay (2 vs 1), lookahead
# (6 vs 13) and training data/alignment, so this is NOT a strict reproduction --
# it only equalizes the per-turn audio (latency) knob.
#
# Checkpoint: checkpoints/imend_12/imend_12-averaged.ckpt
#   = average of the 3 best-val_wer ckpts (step 16503/26004/30504, val_wer ~0.15).
#
# Usage:
#   ./retest_imend_12_full.sh                    # GPU 0, chunks/turn=2, full 8-set suite
#   CHUNKS=1 ./retest_imend_12_full.sh           # native single 7-frame chunk (lower latency)
#   GPU=1 BATCH=64 ./retest_imend_12_full.sh
#   ONLY=librispeech SPLIT=test.clean ./retest_imend_12_full.sh   # single set (quick)
# ---------------------------------------------------------------------------
set -euo pipefail

NEMO_ROOT="/home/hainanx/Workplace/NeMo_ord_sync_d146_current"
CKPT="${CKPT:-${NEMO_ROOT}/checkpoints/imend_12/imend_12-averaged.ckpt}"
GPU="${GPU:-0}"
BATCH="${BATCH:-64}"
CHUNKS="${CHUNKS:-2}"            # 2 -> 14 audio frames/turn == old maxChunks4 chunks=1
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
NO_REPEAT_NGRAM="${NO_REPEAT_NGRAM:-4}"
LOG="${LOG:-/tmp/imend_12_eval_chunks${CHUNKS}_$(date +%Y%m%d_%H%M%S).log}"

# nemo must import from THIS checkout (needed for the multi-chunk decode + knob).
export PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$NEMO_ROOT"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi

# Optional single-dataset quick check.
DS_ARGS=()
if [ -n "${ONLY:-}" ]; then
    DS_ARGS+=(--dataset "$ONLY" --split "${SPLIT:-test}")
fi
[ -n "${MAX_EVAL_SAMPLES:-}" ] && DS_ARGS+=(--max_eval_samples "$MAX_EVAL_SAMPLES")

echo "==> imend_12 eval of $(basename "$CKPT")"
echo "==> GPU=${GPU} BATCH=${BATCH} chunks/turn=${CHUNKS} (2 => 14 frames/turn == old maxChunks4 chunks=1)"
echo "==> max_new_tokens=${MAX_NEW_TOKENS} no_repeat_ngram=${NO_REPEAT_NGRAM}"
echo "==> LOG=${LOG}"

# NOTE: do NOT set HF_HUB_OFFLINE=1 (the esb dataset builder still does a Hub
# revision check even when cached, and offline mode makes it error out).
stdbuf -oL -eL python run_eval_sslm.py \
    --ckpt_path "$CKPT" \
    --device "$GPU" \
    --batch_size "$BATCH" \
    --inference_audio_chunks_per_turn "$CHUNKS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --no_repeat_ngram_size "$NO_REPEAT_NGRAM" \
    "${DS_ARGS[@]}" \
    2>&1 | tee "$LOG"

echo
echo "==> Done. Full log: $LOG"
echo "==> Compare vs old maxChunks4 chunks=1 (14-frame turn): ami 13.48 earnings22 14.02 gigaspeech 12.17 ls.clean 2.94 ls.other 6.09 spgi 3.65 voxpopuli 8.10 (step=40003 re-eval)"
