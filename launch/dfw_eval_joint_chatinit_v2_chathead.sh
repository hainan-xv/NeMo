#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-eval-joint-v2-chathead
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Leaderboard eval of the JOINT model's CHAT head.
#
# The joint model's CHAT side IS a CHAT transducer on the shared encoder, so we
# lift (encoder, decoder, joint) back out into a standalone EncDecCHATBPEModel
# .nemo and hand it to the EXISTING CHAT harness. Two reasons to do it this way
# rather than writing a bespoke driver:
#
#   1. The harness is already validated; a one-off driver would be the thing
#      most likely to be wrong, and its errors would look like model results.
#   2. The numbers land on the same axis as the standalone CHAT baselines
#      (dfw_chat_banded1_nodelay_v2 etc.), so they are directly comparable.
#
# The donor .nemo supplies config + tokenizer artifacts; only WEIGHTS come from
# the joint checkpoint. That pairs the weights with the feature normalisation
# and xscaling they were trained under -- which the Lightning checkpoint does
# not carry, because both are module attributes rather than parameters.
#
# eval_chat.sh reuses an existing averaged .nemo when nothing is newer, so the
# exported head is written to that path and the averaging step is a no-op.
# ============================================================================
set -uo pipefail
mkdir -p slurm_out

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-${DFW}/hainanx}"
export CODE_DIR="${CODE_DIR:-${OUTPUT_PREFIX}/NeMo_SCRIPT_cc}"
export CONTAINER="${CONTAINER:-${DFW}/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
export CACHE_DIR="${CACHE_DIR:-${OUTPUT_PREFIX}/leaderboard_cache}"
export H_DIR="${H_DIR:-${DFW}/users/heh}"
export PROJECT="${PROJECT:-SpeechlmDFW}"
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-${DFW}:${DFW}}"

ARM="${ARM:-dfw_joint_chatinit_p84_v2_chathead}"
JOINT_CKPT="${JOINT_CKPT:-${OUTPUT_PREFIX}/eval_snapshots/joint_v2_step48000.ckpt}"
DONOR="${DONOR:-${OUTPUT_PREFIX}/results/SpeechlmDFW/dfw_granary2_chat_banded1_both_nodelay_v2/averaged/top5-averaged.nemo}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${H_DIR}/pretrained_models/huggingface/Qwen/Qwen3-1.7B}"
TOPK="${TOPK:-5}"

RUN_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${ARM}"
AVG_NEMO="${RUN_DIR}/averaged/top${TOPK}-averaged.nemo"
mkdir -p "${RUN_DIR}/averaged" "${RUN_DIR}/${ARM}/checkpoints"

for f in "$JOINT_CKPT" "$DONOR"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

if [[ ! -f "$AVG_NEMO" ]]; then
    echo "==> exporting CHAT head: $(basename "$JOINT_CKPT") -> ${AVG_NEMO}"
    srun --container-image="$CONTAINER" \
         --container-mounts="${CODE_DIR}:/code,${DFW}:${DFW}" \
         bash -c "export PYTHONPATH=/code/.:\${PYTHONPATH} && cd /code && \
             python /code/scripts/export_joint_chat_head.py \
               --joint_ckpt '${JOINT_CKPT}' \
               --donor_nemo '${DONOR}' \
               --tokenizer_dir '${TOKENIZER_DIR}' \
               --out '${AVG_NEMO}'" || { echo "ERROR: export failed" >&2; exit 1; }
else
    echo "==> reusing existing ${AVG_NEMO}"
fi
# Must be newer than every checkpoint or eval_chat.sh rebuilds it by averaging.
touch "$AVG_NEMO"

resolve_launch_dir() {
    [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/launch/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
    [[ -f "${CODE_DIR}/launch/eval_chat.sh" ]] && { echo "${CODE_DIR}/launch"; return; }
    echo "ERROR: cannot locate eval_chat.sh" >&2; exit 1
}
LAUNCH_DIR="$(resolve_launch_dir)" || exit 1

ARM_EXP_NAME="${ARM}" ARM_MODEL_OVERRIDES="" \
ARM_CONFIG_NAME="nemotron_chat_transducer_granary2_qwen" \
ARM_TOKENIZER_DIR="${TOKENIZER_DIR}" \
TOPK="${TOPK}" FORCE_AVERAGE=0 \
EVAL_TAG="joint_v2_chathead" FRAME_TRIM=0 \
    bash "${LAUNCH_DIR}/eval_chat.sh"
