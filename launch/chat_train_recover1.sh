#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:chat-recover1
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 8
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 04:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=8
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# CHAT, forced alignment, HISTORY RECOVERY of 1 word, no delay.
#
# Each chunk is additionally scored on the PREVIOUS chunk's last word, starting
# from the prefix before it. Nothing is removed from the output, so no chunk is
# trained to stop early; it teaches the model to restore a word when its history
# is one short, which is what makes retract-by-1 decoding legal.
#
# history_chunks = 1 is REQUIRED, not optional: the recovered word's audio lies
# in the previous chunk, so without the widened window the model would have to
# restore it from the prediction network blind.
#
#   sbatch launch/chat_train_recover1.sh          <- no arguments, no environment to set
#
# Every setting that defines THIS model is written below. chat_train.sh holds
# only the parts all CHAT runs share (data, mounts, container, schedule); it is
# never launched directly.
# ============================================================================

export LOSS_TYPE=forced_alignment
export DELAY_FRAMES=0
export RECOVER_WORDS=1
export HISTORY_CHUNKS=1
export MAX_DELAY_FRAMES=0
export EXP_NAME="${EXP_NAME:-granary2_chat_forced_asrvocab_win28_recover1_delay0}"

# Under sbatch $0 is a copy in Slurm's spool directory, so dirname "$0" has no
# sibling chat_train.sh; SLURM_SUBMIT_DIR is where the sbatch was issued.
find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/chat_train.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/chat_train.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/chat_train.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate chat_train.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}

exec bash "$(find_launch_dir)/chat_train.sh" "$@"
