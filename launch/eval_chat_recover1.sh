#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:eval-chat-recover1
# Eval jobs go to the INTERACTIVE partition: they are single-node and short,
# and the batch blocks queue behind 8-node training for hours. The admin limit
# is ONE interactive job per user at a time, so do not launch two evals at once.
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Leaderboard eval -- CHAT with 1-word history recovery and no delay (win28).
#
#   sbatch launch/eval_chat_recover1.sh          <- no arguments, no environment to set
#
# Averages the top-5 checkpoints by val_wer with NeMo's own
# scripts/checkpoint_averaging/average_model_checkpoints.py, then runs the
# Open-ASR-Leaderboard eval on the result.
# ============================================================================

export ARM_EXP_NAME=granary2_chat_forced_asrvocab_win28_recover1_delay0
export ARM_MODEL_OVERRIDES="model.loss_type=forced_alignment model.forced_alignment.num_delay_frames=0 model.forced_alignment.recover_history_words=1 model.joint.history_chunks=1"
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
