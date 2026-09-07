#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:chat-rnnt
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
# PLAIN CHAT: the marginalised RNN-T loss, and nothing else switched on.
#
#   sbatch launch/chat_train_rnnt.sh          <- no arguments, no environment to set
#
# The control for every forced-alignment arm. Same RNNTAttJoint, same encoder
# init, same 1,024-piece vocabulary, same 14-frame chunk grid, same data -- but
# the loss sums over EVERY alignment instead of conditioning on one, so nothing
# here depends on the word timings at all.
#
# Deliberately bare:
#   history_chunks 0   the joint sees only its own chunk (no win28)
#   no delay           there is no alignment to shift; the loss picks its own
#   no recovery        nothing to recover from
#   no flexible delay  the attention window is never trimmed
#
# So the gap between this and a forced arm is the objective plus whatever that
# arm switches on, and the gap between this and the donor RNN-T is the chunked
# cross-attention joint.
# ============================================================================

export LOSS_TYPE=rnnt
export DELAY_FRAMES=0
export RECOVER_WORDS=0
export HISTORY_CHUNKS=0
export MAX_DELAY_FRAMES=0
# One learning rate for every parameter, encoder included.
export LR=0.001
# NOT granary2_chat_standard_asrvocab: that directory holds the earlier
# standard-CHAT run, and resume_if_exists would silently continue it instead of
# starting the clean control this script describes.
export EXP_NAME="${EXP_NAME:-granary2_chat_rnnt_plain}"

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
