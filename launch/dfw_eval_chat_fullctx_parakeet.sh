#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-eval-chat-fullctx-parakeet
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
# pool0-01815 carries an old NVIDIA driver and fails torch's CUDA init outright;
# pool0-00407 is the reference recipe's known-bad node.
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# Open-ASR-Leaderboard eval of the FULL-CONTEXT (bidirectional) CHAT arm.
#
#   sbatch launch/dfw_eval_chat_fullctx_parakeet.sh
#
# Averages the arm's top-K checkpoints by val_wer and evaluates through the SAME
# shared backend (eval_chat.sh -> eval_nemotron.sh) every other arm uses, so the
# number is directly comparable to the streaming arms' 5.51 average / 8.99 AMI.
#
# THE OVERRIDES BELOW ARE MANDATORY, NOT COSMETIC. Averaging CONSTRUCTS the model
# before loading weights, so the config must describe the architecture the
# checkpoint was trained with. The averaging script cross-checks
# {joint.window_frames, joint.history_chunks, joint.chunk_size, loss_type}
# against the checkpoint's own hyper_parameters and REFUSES on a mismatch.
#
# They are copied from launch/dfw_chat_banded1_both_fullctx_parakeet.sh and must
# stay in step with it:
#   loss_type=banded, band_chunks=1, band_side=both    <- the objective
#   num_delay_frames=0                                 <- the "nodelay" arms use
#       UNMODIFIED alignments. The shared dfw_eval_chat.sh defaults this to 3,
#       which would decode at an operating point this arm never trained at.
#   frame_trim untouched (0)                           <- _banded_loss does not
#       set it; only _forced_alignment_loss does.
#
# chunk_size is stated in the config rather than inferred: a full-context encoder
# has att_context_size [-1,-1] and therefore no right context to infer it from.
#
# ENV
#   TOPK            checkpoints to average (default 5)
#   FORCE_AVERAGE   1 to rebuild an averaged .nemo that already exists
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-${DFW}/hainanx}"
export CODE_DIR="${CODE_DIR:-${OUTPUT_PREFIX}/NeMo_SCRIPT_cc}"
export CONTAINER="${CONTAINER:-${DFW}/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
export CACHE_DIR="${CACHE_DIR:-${OUTPUT_PREFIX}/leaderboard_cache}"
export H_DIR="${H_DIR:-${DFW}/users/heh}"
export PROJECT="${PROJECT:-SpeechlmDFW}"
# Mounted so the averaging step can reach the tokenizer; without it transformers
# treats the path as a hub repo id and fails with "Repo id must be in the form".
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-${DFW}:${DFW}}"

if [[ ! -d "${CACHE_DIR}" ]] || [[ -z "$(ls -A "${CACHE_DIR}" 2>/dev/null)" ]]; then
    echo "ERROR: leaderboard cache is empty or missing at ${CACHE_DIR}" >&2
    echo "       Stage it first:  sbatch launch/dfw_stage_leaderboard_cache.sh" >&2
    exit 1
fi

FA=model.forced_alignment
export ARM_EXP_NAME=dfw_granary2_chat_banded1_both_fullctx_parakeet
export ARM_CONFIG_NAME=nemotron_chat_transducer_granary2_qwen_fullctx
export ARM_TOKENIZER_DIR="${DFW}/users/heh/pretrained_models/huggingface/Qwen/Qwen3-1.7B"
export ARM_MODEL_OVERRIDES="model.loss_type=banded ${FA}.band_chunks=1 ++${FA}.band_side=both \
${FA}.recover_history_words=0 ${FA}.num_delay_frames=0 ${FA}.max_delay_frames=0 \
model.joint.history_chunks=0 ${FA}.target_construction=partition ${FA}.delay_word_final_punctuation=true"
export TOPK="${TOPK:-5}"
export FORCE_AVERAGE="${FORCE_AVERAGE:-0}"
export EVAL_TAG="avg${TOPK}"
export FRAME_TRIM=""
# The encoder is NON-CAUSAL: att_context_size [-1,-1], no look-ahead to choose.
# Without this the shared backend forces [left, 13] to decode at chunk 14 and the
# model refuses -- correctly, since that is a context it never trained with.
export FULL_CONTEXT=1

CKPTS="${OUTPUT_PREFIX}/results/${PROJECT}/${ARM_EXP_NAME}/${ARM_EXP_NAME}/checkpoints"
NCK="$(ls -1 "$CKPTS"/*.ckpt 2>/dev/null | grep -vc -- '-last' || true)"
BEST="$(ls -1 "$CKPTS"/*.ckpt 2>/dev/null | grep -oE 'val_wer=[0-9.]+' | sort -t= -k2 -g | head -1 || true)"
echo "==> full-context CHAT leaderboard eval"
echo "    arm:   ${ARM_EXP_NAME}"
echo "    ckpts: ${NCK:-0} (best ${BEST:-val_wer=?}), averaging top ${TOPK}"
if [[ "${NCK:-0}" -lt 1 ]]; then
    echo "ERROR: no checkpoints at ${CKPTS}" >&2
    exit 1
fi
if [[ "${NCK:-0}" -lt "${TOPK}" ]]; then
    echo "NOTE: only ${NCK} checkpoints exist; the average will use those."
fi

# Under sbatch $0 is a copy in Slurm's spool dir, and on a REQUEUE
# SLURM_SUBMIT_DIR comes back as the scratch root -- hence the absolute fallback.
resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_chat.sh" ]] && { echo "${here}"; return; }
    [[ -f "${CODE_DIR}/launch/eval_chat.sh" ]] && { echo "${CODE_DIR}/launch"; return; }
    echo "ERROR: cannot locate eval_chat.sh" >&2
    exit 1
}
exec bash "$(resolve_launch_dir)/eval_chat.sh"
