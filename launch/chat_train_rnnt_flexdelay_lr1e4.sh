#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:chat-rnnt-flexd-lr1e4
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
# CHAT with the MARGINALISED RNN-T loss and randomly trimmed chunks.
#
#   sbatch launch/chat_train_rnnt_flexdelay_lr1e4.sh      <- no arguments
#
# Each batch draws d ~ U{0..4} and the joint's window ends d frames early, so
# the last d frames of every chunk are treated as not yet arrived. There is NO
# explicit delay and NO alignment shift: the RNN-T loss marginalises over every
# alignment, so it picks its own emission points given whatever audio it can
# see. The trim is the entire mechanism, and it is unsupervised in the
# alignment sense -- nothing here uses the word timings at all.
#
# d is a latency knob: at d the model commits d frames before the chunk
# completes. Training across the range gives one model usable at any latency in
# it. Validation and decoding are pinned to d = 2, half the range.
#
# history_chunks = 1 is required, not optional: frames are being taken away, so
# the window must reach back a chunk to leave enough to attend to.
#
# No flush chunk is needed here, unlike the forced-alignment arm. Mid-utterance
# the trimmed frames simply reappear in the next chunk's window; only the final
# chunk's trailing d frames are never seen, and with pad_extra_duration 0.5 s
# (~6 frames) those are silence padding for any d <= 4.
#
# Schedule is the reference recipe from oci_chat/chat_fullctx.sh, which moved
# the plain RNN-T control from val_wer 0.1413 to 0.1202.
# ============================================================================

export LOSS_TYPE=rnnt
export DELAY_FRAMES=0
export RECOVER_WORDS=0
export HISTORY_CHUNKS=1
export MAX_DELAY_FRAMES=4

export LR=1e-4
export WARMUP_STEPS=5000
export MAX_STEPS=500000

export EXP_NAME="${EXP_NAME:-granary2_chat_rnnt_flexdelay4_lr1e4}"

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
