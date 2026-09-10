#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:chat-flexdelay-lr1e4
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
# CHAT, forced alignment, FLEXIBLE (sampled) DELAY.
#
# Each batch draws d ~ U{0..4} and does two complementary things: shifts the
# alignment by d, and hides the last d frames of every chunk from the joint. A
# word emitted at chunk t then has its last frame in [t*C - d, (t+1)*C - d),
# which is exactly what survives the trim -- the model never has to emit a word
# it cannot hear, and never sees audio past the last word it owes.
#
# d is therefore a LATENCY knob, not a right-context knob: at d the model commits
# d frames BEFORE the chunk completes. Training across the range yields ONE model
# usable at any latency in it, instead of one model per latency. Validation and
# decoding are pinned to d = 2 (half the range).
#
# history_chunks = 1 is required: frames are being taken away, so the window must
# reach back a chunk to leave enough to attend to.
#
# One learning rate for every parameter (1e-3), encoder included.
#
#   sbatch launch/chat_train_flexdelay_lr1e4.sh          <- no arguments, no environment to set
#
# Every setting that defines THIS model is written below. chat_train.sh holds
# only the parts all CHAT runs share (data, mounts, container, schedule); it is
# never launched directly.
# ============================================================================

export LOSS_TYPE=forced_alignment
export DELAY_FRAMES=0
export RECOVER_WORDS=0
export HISTORY_CHUNKS=1
export MAX_DELAY_FRAMES=4
# --- reference optimisation recipe, from oci_chat/chat_fullctx.sh ---
# lr 1e-4 rather than the 1e-3 these arms first ran at. On the plain RNN-T
# control that change alone moved val_wer 0.1413 -> 0.1202; every CHAT recipe
# found on this machine peaks at or below ~2.5e-4 except one, and ours was ten
# times the only cosine reference.
export LR=1e-4
export WARMUP_STEPS=5000
export MAX_STEPS=500000

export EXP_NAME="${EXP_NAME:-granary2_chat_forced_asrvocab_flexdelay4_lr1e4}"

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
