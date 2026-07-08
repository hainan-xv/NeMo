#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Re-evaluate the best "back then" SpeechLLM (baseline_imend_loss_sub2,
# project Streaming_SLM_624, step=18002-last) with the CURRENT repo code,
# to check whether eval/decode has regressed.
#
# `nemo` already resolves to THIS repo (verified), so this runs the checkpoint
# through the current model/decode code. The checkpoint is local (10.4 GB) and
# the ESB datasets + base models (Qwen3-1.7B, nemotron-speech-streaming-en-0.6b)
# are already in the HF cache, so no OCI download is needed.
#
# Original (Jun 25) full-suite numbers for this ckpt, for comparison:
#   ami 13.91  earnings22 13.68  gigaspeech 12.07  ls.clean 2.93
#   ls.other 6.29  spgispeech 3.20  voxpopuli 8.49   -> avg ~8.65
#
# Usage:
#   ./retest_sub2_full.sh              # GPU 0, full 8-set suite
#   GPU=1 ./retest_sub2_full.sh        # pick a GPU
#   BATCH=32 ./retest_sub2_full.sh     # smaller batch (OOM backoff also halves automatically)
# ---------------------------------------------------------------------------
set -euo pipefail

NEMO_ROOT="/home/hainanx/Workplace/NeMo_ord_sync_d146_current"
CKPT="${CKPT:-${NEMO_ROOT}/checkpoints/baseline_imend_loss_sub2/step=18002-last.ckpt}"
GPU="${GPU:-0}"
BATCH="${BATCH:-64}"
LOG="${LOG:-/tmp/sub2_full_eval_$(date +%Y%m%d_%H%M%S).log}"

cd "$NEMO_ROOT"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi

echo "==> Re-eval ${CKPT}"
echo "==> GPU=${GPU}  BATCH=${BATCH} (auto-halves on OOM)  LOG=${LOG}"
echo "==> Full ESB suite (ami, earnings22, gigaspeech, librispeech clean/other, spgispeech, tedlium, voxpopuli)"

# NOTE: do NOT set HF_HUB_OFFLINE=1 -- the esb dataset builder still contacts the
# Hub for a revision check even when cached, and offline mode makes it error out.
stdbuf -oL -eL python run_eval_sslm.py \
    --ckpt_path "$CKPT" \
    --device "$GPU" \
    --batch_size "$BATCH" \
    2>&1 | tee "$LOG"

echo
echo "==> Done. Full log: $LOG"
