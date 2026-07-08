#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduce the "back then" MULTI-CHUNK SpeechLLM leaderboard numbers (~7 avg WER)
# with the CURRENT repo code, using a LOCAL checkpoint (already scp'd from the
# ORD grid).
#
# Model: variable/multi-chunk imend_loss, trained with max_audio_chunks_per_turn=4
#   EXP: ord_streaming_stt_granary1p1_lora_lr0.0004_warmup15000_n4_delay1_la13_chunk14_rnd_r1_imend_loss_maxChunks4_hainanCode_hainan
#   project Streaming_SLM_ord (cs-oci-ord grid), late May 2026.
#
# The ~7 numbers come from MULTI-CHUNK INFERENCE: the decoder consumes several
# audio chunks per LLM turn. On the original eval (step=8000, now evicted) the
# 8/8 average was:
#     chunks/turn=1 -> 7.92 | =2 -> 7.35 | =3 -> 7.17   (monotonic)
# Two things are REQUIRED to reproduce that:
#   1) --inference_audio_chunks_per_turn = 3   (grouping)
#   2) --max_new_tokens = 64                   (per-turn token budget; too small a
#      budget was the "more chunks = worse" cliff seen in debugging)
#   ( --no_repeat_ngram_size = 4 also matched the original eval_leaderboard_ord run )
#
# This checkpoint is step=40003 (the surviving top-k best; step=8000 was evicted
# by save_top_k), so absolute numbers may differ slightly from the 7.17 snapshot.
#
# Usage:
#   ./retest_varchunks_full.sh                 # GPU 0, chunks/turn=3, full 8-set suite
#   CHUNKS=2 ./retest_varchunks_full.sh        # sweep the chunks/turn knob
#   GPU=1 BATCH=64 ./retest_varchunks_full.sh
#   ONLY=librispeech SPLIT=test.clean ./retest_varchunks_full.sh   # single set (quick)
# ---------------------------------------------------------------------------
set -euo pipefail

NEMO_ROOT="/home/hainanx/Workplace/NeMo_ord_sync_d146_current"
CKPT="${CKPT:-${NEMO_ROOT}/checkpoints/varchunks_chunk14_maxChunks4/step=40003.ckpt}"
GPU="${GPU:-0}"
BATCH="${BATCH:-64}"
CHUNKS="${CHUNKS:-3}"            # inference audio chunks per LLM turn (>1 = multi-chunk)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
NO_REPEAT_NGRAM="${NO_REPEAT_NGRAM:-4}"
LOG="${LOG:-/tmp/varchunks_eval_chunks${CHUNKS}_$(date +%Y%m%d_%H%M%S).log}"

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

echo "==> Multi-chunk re-eval of $(basename "$CKPT")"
echo "==> GPU=${GPU} BATCH=${BATCH} chunks/turn=${CHUNKS} max_new_tokens=${MAX_NEW_TOKENS} no_repeat_ngram=${NO_REPEAT_NGRAM}"
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
echo "==> Reference (original step=8000): chunks=1 -> 7.92 | chunks=2 -> 7.35 | chunks=3 -> 7.17 (avg 8/8 WER)"
