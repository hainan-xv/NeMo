#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:chat-delay3
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
# CHAT, forced alignment, FIXED EMISSION DELAY of 3 frames.
#
# A word is emitted at the chunk holding its last frame + 3, and the joint sees
# the whole chunk. Every word therefore gets at least 3 frames of right context
# before it must be committed -- a word's final frames are often what
# disambiguate it. No retraction: this arm tests delay alone.
#
# history_chunks = 0: nothing is being taken away from the window, so the joint
# does not need to reach back a chunk.
#
#   sbatch launch/chat_train_delay3.sh          <- no arguments, no environment to set
#
# Every setting that defines THIS model is written below. chat_train.sh holds
# only the parts all CHAT runs share (data, mounts, container, schedule); it is
# never launched directly.
# ============================================================================

export LOSS_TYPE=forced_alignment
export DELAY_FRAMES=3
export RECOVER_WORDS=0
export HISTORY_CHUNKS=0
export MAX_DELAY_FRAMES=0
export EXP_NAME="${EXP_NAME:-granary2_chat_forced_asrvocab_delay3}"

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
