#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate `imend_sellora_mc4` on the Open-ASR leaderboard suite at
# 4 CHUNKS PER TURN -- the max chunk-grouping the model was trained with
# (max_audio_chunks_per_turn=4). With chunk_size=14, 4 chunks/turn feeds
# 4 * 14 = 56 audio frames (~4.48 s) into each LLM turn.
#
# Pipeline (runs LOCALLY on this box's GPU, pulls the ckpt FROM OCI):
#   1. PREPARE the checkpoint by reusing eval_leaderboard.sh's fetch+average
#      logic (RUN_AVERAGING=1 -> rsync every non '-last' top-k snapshot and
#      average them into checkpoints/<EXP>/<EXP>-averaged.ckpt). Set AVERAGE=0
#      to instead grab the single best (lowest-val_wer, non '-last') checkpoint.
#   2. EVALUATE with run_eval_sslm.py over the full 8-dataset leaderboard
#      (no --dataset => LEADERBOARD_DATASETS + built-in summary), passing
#      --inference_audio_chunks_per_turn=$CHUNKS. Batch auto-halves on CUDA OOM
#      down to MIN_BATCH.
#
# Usage:
#   ./eval_sellora_mc4.sh                       # avg top-k, GPU 0, 4 chunks/turn, full suite
#   CHUNKS=1 ./eval_sellora_mc4.sh              # single 14-frame chunk (lower latency)
#   AVERAGE=0 ./eval_sellora_mc4.sh             # best single ckpt instead of the average
#   GPU=1 BATCH=128 ./eval_sellora_mc4.sh
#   ONLY=librispeech SPLIT=test.clean ./eval_sellora_mc4.sh    # one dataset (quick)
#   QUICK_TEST=1 ./eval_sellora_mc4.sh          # 10 samples from ami/test (smoke)
#   EXP=imend_sellora_scratch ./eval_sellora_mc4.sh   # eval a different run the same way
#
# Env overrides: EXP, PROJECT, CHUNKS, AVERAGE, GPU, BATCH, MIN_BATCH,
#   MAX_NEW_TOKENS, NO_REPEAT_NGRAM, MAX_EVAL_SAMPLES, ONLY, SPLIT, QUICK_TEST,
#   FORCE_AVERAGE, FORCE_DOWNLOAD, LOCAL_ONLY_IF_EXIST (see eval_leaderboard.sh).
# ---------------------------------------------------------------------------
set -euo pipefail

NEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
EVAL_DRIVER="${NEMO_ROOT}/eval_leaderboard.sh"
RUN_EVAL_PY="${NEMO_ROOT}/run_eval_sslm.py"
[ -f "$EVAL_DRIVER" ] || { echo "ERROR: prepare driver not found at $EVAL_DRIVER" >&2; exit 1; }
[ -f "$RUN_EVAL_PY" ]  || { echo "ERROR: eval driver not found at $RUN_EVAL_PY" >&2; exit 1; }

EXP="${EXP:-imend_sellora_mc4}"
PROJECT="${PROJECT:-Streaming_SLM_Qwen1p7B}"
CHUNKS="${CHUNKS:-4}"                 # 4 => 4*14 = 56 audio frames/turn (trained max)
AVERAGE="${AVERAGE:-1}"               # 1 => average top-k; 0 => single best ckpt
GPU="${GPU:-0}"
BATCH="${BATCH:-256}"                 # auto-halves on CUDA OOM down to MIN_BATCH
MIN_BATCH="${MIN_BATCH:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
NO_REPEAT_NGRAM="${NO_REPEAT_NGRAM:-4}"
QUICK_TEST="${QUICK_TEST:-0}"

# nemo must import from THIS checkout (selective-LoRA + multi-chunk decode knob).
export PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH:-}"
export STREAMING_STT_MODEL_ROOT="${NEMO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Gated ESB datasets need the HF token (gitignored .hf_token, same as the sibling scripts).
if [ -z "${HF_TOKEN:-}" ] && [ -f "${NEMO_ROOT}/.hf_token" ]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${NEMO_ROOT}/.hf_token")"; export HF_TOKEN
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${NEMO_ROOT}/eval_results/${EXP}_chunks${CHUNKS}_${STAMP}"
mkdir -p "$LOGDIR"
PREP_LOG="${LOGDIR}/prepare.log"
EVAL_LOG="${LOGDIR}/eval.log"

# ---------- Phase 1: prepare (fetch + optionally average) the checkpoint ------
echo "==> [prepare] resolving ${EXP} under project ${PROJECT} (AVERAGE=${AVERAGE}) ..."
if ! PROJECT="$PROJECT" RUN_AVERAGING="$AVERAGE" PREPARE_ONLY=1 \
        LOCAL_ONLY_IF_EXIST="${LOCAL_ONLY_IF_EXIST:-0}" \
        FORCE_AVERAGE="${FORCE_AVERAGE:-0}" FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}" \
        "$EVAL_DRIVER" "$EXP" >"$PREP_LOG" 2>&1; then
    echo "ERROR: checkpoint prepare failed. Tail of ${PREP_LOG}:" >&2
    tail -n 25 "$PREP_LOG" >&2
    exit 1
fi
CKPT="$(grep '^PREPARED_CKPT=' "$PREP_LOG" | tail -1 | cut -d= -f2-)"
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: prepare produced no usable checkpoint (see ${PREP_LOG})." >&2
    tail -n 25 "$PREP_LOG" >&2
    exit 1
fi
echo "==> [prepare] checkpoint ready: ${CKPT} ($(du -h "$CKPT" | cut -f1))"

# ---------- Phase 2: evaluate at CHUNKS chunks/turn ---------------------------
EVAL_ARGS=(
    --ckpt_path "$CKPT"
    --dataset_path "hf-audio/esb-datasets-test-only-sorted"
    --device "$GPU"
    --batch_size "$BATCH"
    --min_batch_size "$MIN_BATCH"
    --inference_audio_chunks_per_turn "$CHUNKS"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --no_repeat_ngram_size "$NO_REPEAT_NGRAM"
)
# Optional single-dataset / smoke overrides. With no --dataset run_eval_sslm.py
# sweeps the full LEADERBOARD_DATASETS set and prints its own summary table.
if [ -n "${ONLY:-}" ]; then
    EVAL_ARGS+=(--dataset "$ONLY" --split "${SPLIT:-test}")
fi
if [ "$QUICK_TEST" = "1" ]; then
    EVAL_ARGS+=(--dataset ami --split test --max_eval_samples 10 --verbose)
fi
[ -n "${MAX_EVAL_SAMPLES:-}" ] && EVAL_ARGS+=(--max_eval_samples "$MAX_EVAL_SAMPLES")

echo "==> [eval] ${EXP} @ ${CHUNKS} chunk(s)/turn  (chunk_size=14 => $((CHUNKS*14)) audio frames/turn)"
echo "==> [eval] GPU=${GPU} BATCH=${BATCH} (min ${MIN_BATCH})  max_new_tokens=${MAX_NEW_TOKENS}  no_repeat_ngram=${NO_REPEAT_NGRAM}"
echo "==> [eval] log: ${EVAL_LOG}"
cd "$LOGDIR"
stdbuf -oL -eL python "$RUN_EVAL_PY" "${EVAL_ARGS[@]}" 2>&1 | tee "$EVAL_LOG"

echo
echo "==> Done. Full logs under ${LOGDIR}"
