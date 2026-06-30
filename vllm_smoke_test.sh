#!/bin/bash
set -e
set -o pipefail

# ============================================================================
# Minimal smoke test for the vLLM StreamingSTT inference path.
#
# Runs the colleagues' two-step Docker workflow on just a FEW utterances, to
# confirm the convert + decode containers actually run end-to-end:
#
#   1. convert: .ckpt  ->  vLLM model dir   (docker: /workspace/convert.sh)
#   2. infer:   vLLM model dir + tiny NeMo manifest
#               (docker: /workspace/b_streaming_infer.py)
#
# The tiny manifest (N utterances + 16k wavs + references) is produced with
# run_eval_sslm.py --dump_manifest, exactly like the VLLM path in
# eval_leaderboard.sh -- so this is a faithful, fast subset of that path.
#
# Usage:
#   ./vllm_smoke_test.sh /path/to/step=NNNN.ckpt        # 3 librispeech utts
#   N=5 ./vllm_smoke_test.sh <ckpt>                     # 5 utts
#   DEVICE=1 ./vllm_smoke_test.sh <ckpt>
#   DATASET=librispeech SPLIT=test.other ./vllm_smoke_test.sh <ckpt>
#   FORCE_CONVERT=1 ./vllm_smoke_test.sh <ckpt>         # rebuild vLLM dir
#
# Env:
#   N            number of utterances (default 3)
#   DEVICE       GPU id (default 0)
#   DATASET/SPLIT  HF ESB subset to sample from (default librispeech/test.clean)
#   VLLM_IMAGE   container image (default dongjig streaming-stt-eval:v2)
#   DOCKER_GPU_ARGS  override GPU passthrough (default: --gpus device=$DEVICE)
#   FORCE_CONVERT 1 -> re-run convert.sh even if the vLLM dir is cached
# ============================================================================

NEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH}"
export STREAMING_STT_MODEL_ROOT="${NEMO_ROOT}"

# HF token for the gated ESB dataset (mirrors the eval wrappers).
if [ -z "${HF_TOKEN:-}" ] && [ -f "${NEMO_ROOT}/.hf_token" ]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${NEMO_ROOT}/.hf_token")"
    export HF_TOKEN
fi

RUN_EVAL_PY="${NEMO_ROOT}/run_eval_sslm.py"
[ -f "$RUN_EVAL_PY" ] || { echo "ERROR: missing $RUN_EVAL_PY" >&2; exit 1; }

CKPT="${1:-${CKPT:-}}"
if [ -z "$CKPT" ]; then
    # Fall back to the most recent local checkpoint.
    CKPT="$(ls -t "${NEMO_ROOT}"/checkpoints/*/*.ckpt 2>/dev/null | head -1 || true)"
    [ -n "$CKPT" ] && echo "==> No ckpt given; using most recent local: $CKPT"
fi
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: '$CKPT' (pass one as arg 1)" >&2; exit 1; }

N="${N:-3}"
DEVICE_ID="${DEVICE:-0}"
DATASET="${DATASET:-librispeech}"
SPLIT="${SPLIT:-test.clean}"
VLLM_IMAGE="${VLLM_IMAGE:-gitlab-master.nvidia.com/dongjig/nemo_containers/streaming-stt-eval:v2}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:---gpus device=${DEVICE_ID}}"

WORK="${WORK:-${NEMO_ROOT}/eval_results/vllm_smoke}"
mkdir -p "$WORK"

echo "######################################################################"
echo "## vLLM smoke test"
echo "##   ckpt:    $CKPT"
echo "##   utts:    $N   ($DATASET/$SPLIT)"
echo "##   gpu:     cuda:${DEVICE_ID}"
echo "##   image:   $VLLM_IMAGE"
echo "##   workdir: $WORK"
echo "######################################################################"

# ---------- Step 1: convert ckpt -> vLLM model dir (cached) ----------
CKPT_DIR="$(cd "$(dirname "$CKPT")" && pwd)"
CKPT_BASE="$(basename "$CKPT")"
VLLM_NAME="vllm_${CKPT_BASE%.ckpt}"
VLLM_OUT="${CKPT_DIR}/${VLLM_NAME}"
if [ -d "$VLLM_OUT" ] && [ -n "$(ls -A "$VLLM_OUT" 2>/dev/null)" ] && [ "${FORCE_CONVERT:-0}" != "1" ]; then
    echo "==> [1/3] Reusing cached vLLM model (FORCE_CONVERT=1 to rebuild): $VLLM_OUT"
else
    echo "==> [1/3] Converting ckpt -> vLLM model dir via ${VLLM_IMAGE} ..."
    rm -rf "$VLLM_OUT"; mkdir -p "$VLLM_OUT"
    docker run --rm $DOCKER_GPU_ARGS -e NVIDIA_VISIBLE_DEVICES="${DEVICE_ID}" \
        -v "${CKPT_DIR}":/ckpt \
        -v "${CKPT_DIR}":/out \
        "$VLLM_IMAGE" \
        bash /workspace/convert.sh "/ckpt/${CKPT_BASE}" "/out/${VLLM_NAME}"
    echo "==> Convert complete: $VLLM_OUT"
fi

# ---------- Step 2: materialize N utterances + a NeMo manifest ----------
# Run from $WORK so run_eval_sslm.py caches wavs under $WORK/audio_cache and
# writes absolute audio_filepath paths the container can resolve.
MAN_PATH="${WORK}/manifest.json"
echo "==> [2/3] Building ${N}-utterance manifest ($DATASET/$SPLIT) ..."
( cd "$WORK" && python "$RUN_EVAL_PY" \
    --ckpt_path "$CKPT" \
    --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --dump_manifest "$MAN_PATH" \
    --max_eval_samples "$N" )
AUDIO_CACHE="${WORK}/audio_cache"
echo "    Manifest: $MAN_PATH"
echo "    --- manifest contents ---"
cat "$MAN_PATH"
echo "    -------------------------"

# ---------- Step 3: decode the tiny manifest in the vLLM container ----------
echo "==> [3/3] Running b_streaming_infer.py in ${VLLM_IMAGE} ..."
docker run --rm $DOCKER_GPU_ARGS -e NVIDIA_VISIBLE_DEVICES="${DEVICE_ID}" \
    -v "${VLLM_OUT}":/model \
    -v "${WORK}":/data \
    -v "${AUDIO_CACHE}":"${AUDIO_CACHE}" \
    -e B_MODEL=/model \
    -e B_MAN="/data/$(basename "$MAN_PATH")" \
    "$VLLM_IMAGE" \
    python /workspace/b_streaming_infer.py

echo ""
echo ">> vLLM smoke test finished. If you saw predictions/WER above, the path runs."
