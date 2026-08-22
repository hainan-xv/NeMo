#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:flex-spike
#SBATCH -p interactive,batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -t 00:40:00
#SBATCH --time-min 00:20:00
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# DIAGNOSTIC: does FlexAttention fix the win28 OOM?
#
# Runs scripts/flex_attention_spike.py on ONE GPU. It answers, on real hardware:
#   1. do flex logits AND gradients match the current dense-mask path?
#   2. peak memory for {dense, flex} x {no checkpointing, activation checkpointing}
#      at the exact shape that OOM'd (chunk_size=2, window=28, T~11k)
#   3. which arms survive which batch size
#
# This is a throwaway measurement, not part of training or eval.
#
# Usage:
#   ./oci_launch_interactive.sh launch/flex_spike.sh
#   ./oci_launch.sh launch/flex_spike.sh              # batch queue instead
#   BATCH_SIZES="1 2 4 8" ./oci_launch.sh launch/flex_spike.sh
# ============================================================================

read_optional_token() { [[ -r "$1" ]] && tr -d '\r\n' < "$1" || true; }
HF_TOKEN="$(read_optional_token "$HOME/.hf_token")"

mkdir -p slurm_out

# The LLM the recipe trains on; already staged under the heh pretrained dir.
LLM="${LLM:-/lustre/fsw/portfolios/llmservice/users/heh/pretrained_models/huggingface/Qwen/Qwen3-1.7B}"
CHUNK_FRAMES="${CHUNK_FRAMES:-2}"
WINDOW_FRAMES="${WINDOW_FRAMES:-28}"
N_CHUNKS="${N_CHUNKS:-338}"          # 54s (max_duration) at chunk_size=2
BATCH_SIZES="${BATCH_SIZES:-1 2 4}"
DTYPE="${DTYPE:-bf16}"

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_SCRIPT_cc}"
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh
HFCACHE="${OUTPUT_PREFIX}/hf_cache"

RESULTS_DIR="${OUTPUT_PREFIX}/results/flex_spike/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local$$}"
mkdir -p "$RESULTS_DIR" "$HFCACHE"

echo "==> flex attention spike"
echo "    llm:     ${LLM}"
echo "    shape:   chunk=${CHUNK_FRAMES} window=${WINDOW_FRAMES} n_chunks=${N_CHUNKS}"
echo "    batches: ${BATCH_SIZES}"
echo "    results: ${RESULTS_DIR}"

MOUNTS="--container-mounts=/lustre/fsw:/lustre/fsw,${CODE_DIR}:/code,${OUTPUT_PREFIX}:${OUTPUT_PREFIX},${H_DIR}:${H_DIR},${HFCACHE}:/hfcache/"

read -r -d '' cmd <<EOF
nvidia-smi \
&& cd /code \
&& echo "CODE COMMIT:" && git rev-parse HEAD \
&& export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& export HF_HOME="/hfcache/" \
&& export HF_TOKEN=${HF_TOKEN} \
&& export HF_HUB_OFFLINE=1 \
&& export TOKENIZERS_PARALLELISM=false \
&& export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
&& python /code/scripts/flex_attention_spike.py \
     --llm "${LLM}" \
     --chunk_frames ${CHUNK_FRAMES} \
     --window_frames ${WINDOW_FRAMES} \
     --n_chunks ${N_CHUNKS} \
     --batch_sizes ${BATCH_SIZES} \
     --dtype ${DTYPE} \
   2>&1 | tee ${RESULTS_DIR}/spike.log
EOF

CMD_FILE="${RESULTS_DIR}/container_cmd.sh"
printf '%s\n' "$cmd" > "$CMD_FILE"
chmod +x "$CMD_FILE"

srun -o "${RESULTS_DIR}/slurm-%j.out" -e "${RESULTS_DIR}/error-%j.out" \
     --container-image="$CONTAINER" $MOUNTS bash "$CMD_FILE"
