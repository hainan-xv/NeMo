#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:chat-rnnt-hc1-lr1e4
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
# PLAIN CHAT (marginalised RNN-T loss) with the WIDENED JOINT WINDOW.
#
#   sbatch launch/chat_train_rnnt_hc1_lr1e4.sh          <- no arguments
#
# Identical to chat_train_rnnt_lr1e4.sh except history_chunks 0 -> 1: the joint
# attends over the previous chunk as well as its own (28 frames read on a
# 14-frame emission grid), strictly backward-looking so latency is unchanged.
#
# WHY THIS RUN EXISTS. The flexible-delay arm beats the plain one by 15%
# relative on librispeech test.clean (0.0233 -> 0.0199), but it differs in TWO
# ways: it trims a random d frames off each chunk during training, AND it has
# history_chunks=1. At d=0 the two score identically (0.0233 both), which
# argues the window is not the cause -- but the two knobs have never been
# varied independently, so that is inference rather than measurement.
#
# This isolates the window. If it lands near 0.0233 the trim owns the gain; if
# near 0.0199, the history window does.
# ============================================================================

export LOSS_TYPE=rnnt
export DELAY_FRAMES=0
export RECOVER_WORDS=0
export HISTORY_CHUNKS=1
export MAX_DELAY_FRAMES=0

# --- the reference optimisation recipe ---
export LR=1e-4
export WARMUP_STEPS=5000
export MAX_STEPS=500000

export EXP_NAME="${EXP_NAME:-granary2_chat_rnnt_hc1_lr1e4}"

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
