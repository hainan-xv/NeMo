#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:eval-chat-delay3
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Leaderboard eval -- CHAT with a FIXED 3-frame emission delay, no retraction.
#
#   sbatch launch/eval_chat_delay3.sh          <- no arguments, no environment to set
#
# Averages the top-5 checkpoints by val_wer with NeMo's own
# scripts/checkpoint_averaging/average_model_checkpoints.py, then runs the
# Open-ASR-Leaderboard eval on the result.
# ============================================================================

export ARM_EXP_NAME=granary2_chat_forced_asrvocab_delay3
export ARM_MODEL_OVERRIDES="model.loss_type=forced_alignment model.forced_alignment.num_delay_frames=3 model.forced_alignment.recover_history_words=0 model.joint.history_chunks=0"
export TOPK=5

find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_chat.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate eval_chat.sh" >&2; exit 1
}
exec bash "$(find_launch_dir)/eval_chat.sh"
