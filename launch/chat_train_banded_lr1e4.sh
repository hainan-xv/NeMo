#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:chat-banded-lr1e4
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
# CHAT with the BANDED loss -- the middle ground between the two objectives.
#
#   sbatch launch/chat_train_banded_lr1e4.sh          <- no arguments
#
# Start from the frame-based chunk alignment, but do not trust it absolutely:
# sum over every valid RNN-T path that stays within ONE chunk of it. A word may
# be emitted a chunk early or a chunk late and the loss still credits it, so the
# aligner is a prior rather than a constraint.
#
# Cost sits between the two arms it interpolates. On Granary's ~1.5 tokens per
# chunk the band scores about 3U + T nodes against the forced path's U + T --
# roughly 2.2x -- while full marginalisation over T x U is about 33x. band 0
# reproduces the forced loss exactly and a very wide band reproduces full RNN-T;
# both are asserted in tests/collections/asr/test_banded_rnnt.py.
#
# Emission delay 3 frames, as in the forced delay3 arm, so the band is centred
# on an alignment that already gives each word some right context. No history
# chunk and no window trimming: this arm varies the LOSS only.
# ============================================================================

export LOSS_TYPE=banded
export DELAY_FRAMES=3
export BAND_CHUNKS=1
export RECOVER_WORDS=0
export HISTORY_CHUNKS=0
export MAX_DELAY_FRAMES=0
# --- reference optimisation recipe, from oci_chat/chat_fullctx.sh ---
# lr 1e-4 rather than the 1e-3 these arms first ran at. On the plain RNN-T
# control that change alone moved val_wer 0.1413 -> 0.1202; every CHAT recipe
# found on this machine peaks at or below ~2.5e-4 except one, and ours was ten
# times the only cosine reference.
export LR=1e-4
export WARMUP_STEPS=5000
export MAX_STEPS=500000

export EXP_NAME="${EXP_NAME:-granary2_chat_banded1_delay3_lr1e4}"

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
