#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:vllm-eval-test
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:30:00
#SBATCH --time-min 00:30:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# =============================================================================
# vLLM streaming-stt eval SMOKE TEST on OCI (pyxis/enroot).
#
#   [1/2] convert a Lightning .ckpt  -> vLLM model dir  (convert.sh)
#   [2/2] decode a manifest with     b_streaming_infer.py
#
# Uses dongjig's eval container. On OCI this works where the local desktop can't:
#   * host drivers are current  -> no CUDA error 803 (the :v2 image is CUDA 13.0);
#   * the ckpt config's /lustre .../heh/pretrained_models paths EXIST here, so
#     convert.sh instantiates the model without any config patching.
#
# Submit:
#   cd /lustre/fsw/portfolios/nemotron/users/hainanx/NeMo79
#   sbatch oci/test_vllm_eval_oci.sh
# Everything below is overridable via env (CONTAINER, CKPT, MANIFEST, ...).
#
# CONTAINER auth: pyxis imports the gitlab-master image on first use, which needs
# enroot registry credentials (~/.config/enroot/.credentials). If that fails,
# pre-import once to a .sqsh and point CONTAINER at it:
#   enroot import -o /lustre/fsw/portfolios/nemotron/users/hainanx/streaming-stt-eval_v2.sqsh \
#     'docker://gitlab-master.nvidia.com#dongjig/nemo_containers/streaming-stt-eval:v2'
#   CONTAINER=/lustre/.../streaming-stt-eval_v2.sqsh sbatch oci/test_vllm_eval_oci.sh
# =============================================================================
mkdir -p slurm_out

CONTAINER="${CONTAINER:-gitlab-master.nvidia.com#dongjig/nemo_containers/streaming-stt-eval:v2}"

# ---- checkpoint to convert (on lustre) ----
CKPT_DIR="${CKPT_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/results/Speechlm79/granary2_noblank_wer/granary2_noblank_wer/checkpoints}"
CKPT="${CKPT:-}"   # explicit path; if empty, newest non '-last' ckpt in CKPT_DIR

# ---- test data: a NeMo manifest (one json/line: id, audio_filepath, duration,
#      text) + its audio. DEFAULT is the recipe's validation manifest. If the
#      container errors on schema/audio, set MANIFEST to a plain manifest (e.g.
#      stage one from your desktop, or ask dongjig for his reference manifest)
#      and AUDIO_MOUNT to the dir whose subtree holds its audio_filepath targets.
MANIFEST="${MANIFEST:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/dongjig/aligned_amos/steve_val_mmlpc_mcv11_2k/mcv11_dev_clean_pcstrip_en_2k_qwen_aligned.json}"
AUDIO_MOUNT="${AUDIO_MOUNT:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm}"

# ---- outputs (writable lustre) ----
OUT_ROOT="${OUT_ROOT:-/lustre/fsw/portfolios/nemotron/users/hainanx/vllm_test}"
VLLM_NAME="${VLLM_NAME:-vllm_granary2_noblank_wer}"
mkdir -p "$OUT_ROOT"

# pretrained models referenced (by absolute path) in the ckpt config live here:
HEH_DIR=/lustre/fsw/portfolios/llmservice/users/heh

# ---- resolve ckpt + expose an '='-free name (convert.sh is a Hydra app; a
#      ckpt literally named 'step=NNNN-val_wer=...' breaks Hydra parsing) ----
if [ -z "$CKPT" ]; then
    CKPT=$(ls -t "$CKPT_DIR"/*.ckpt 2>/dev/null | grep -v -- '-last\.ckpt$' | head -1)
fi
[ -n "$CKPT" ] && [ -f "$CKPT" ] || {
    echo "ERROR: no checkpoint found. Set CKPT=/abs/path.ckpt or CKPT_DIR=<dir>. (CKPT_DIR=$CKPT_DIR)" >&2
    exit 1
}
CKPT_DIR_ABS="$(cd "$(dirname "$CKPT")" && pwd)"
CKPT_BASE="$(basename "$CKPT")"
if [[ "$CKPT_BASE" == *"="* ]]; then
    SAFE="${CKPT_BASE//=/_}"
    if [ ! -e "$CKPT_DIR_ABS/$SAFE" ]; then
        ln "$CKPT_DIR_ABS/$CKPT_BASE" "$CKPT_DIR_ABS/$SAFE" 2>/dev/null \
            || cp "$CKPT_DIR_ABS/$CKPT_BASE" "$CKPT_DIR_ABS/$SAFE"
    fi
    echo "==> ckpt name has '='; using '='-free copy: $SAFE"
    CKPT_BASE="$SAFE"
fi

[ -f "$MANIFEST" ] || { echo "ERROR: manifest not found: $MANIFEST (set MANIFEST=)" >&2; exit 1; }
MAN_DIR="$(cd "$(dirname "$MANIFEST")" && pwd)"
MAN_BASE="$(basename "$MANIFEST")"

echo "======================================================================"
echo "  CONTAINER : $CONTAINER"
echo "  CKPT      : $CKPT_DIR_ABS/$CKPT_BASE"
echo "  MANIFEST  : $MAN_DIR/$MAN_BASE"
echo "  AUDIO_MNT : $AUDIO_MOUNT"
echo "  OUT MODEL : $OUT_ROOT/$VLLM_NAME"
echo "======================================================================"

MOUNTS="--container-mounts=${CKPT_DIR_ABS}:/ckpt,${OUT_ROOT}:/out,${MAN_DIR}:/data,${AUDIO_MOUNT}:${AUDIO_MOUNT},${HEH_DIR}:${HEH_DIR}"

# Note: $-escapes are evaluated INSIDE the container, not at submit time.
read -r -d '' cmd <<EOF
set -euo pipefail
echo "===== container preflight ====="
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, '| cuda_avail', torch.cuda.is_available()); print('cuda_ok', float((torch.zeros(1, device='cuda')+1).item()))"
echo "----- /workspace contents -----"; ls -la /workspace

echo "===== [1/2] convert: /ckpt/${CKPT_BASE} -> /out/${VLLM_NAME} ====="
S=/workspace/convert_patched.sh; [ -f "\$S" ] || S=/workspace/convert.sh
echo "    using converter: \$S"
bash "\$S" "/ckpt/${CKPT_BASE}" "/out/${VLLM_NAME}"
echo "    convert OK; /out/${VLLM_NAME}:"; ls -la "/out/${VLLM_NAME}"

echo "===== [2/2] decode: ${MAN_BASE} ====="
export B_MODEL="/out/${VLLM_NAME}"
export B_MAN="/data/${MAN_BASE}"
echo "    B_MODEL=\$B_MODEL"
echo "    B_MAN=\$B_MAN"
python /workspace/b_streaming_infer.py
echo "===== DONE (vLLM smoke test) ====="
EOF

srun --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"

set +x
