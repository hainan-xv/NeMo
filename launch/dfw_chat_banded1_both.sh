#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-chat-banded1-both
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 04:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=8
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
#SBATCH --exclude=pool0-00407

# ============================================================================
# BIDIRECTIONAL BANDED CHAT on DFW -- the seventh arm.
#
#   sbatch launch/dfw_chat_banded1_both.sh     <- no arguments
#
# Identical to dfw_chat_banded1.sh in EVERY knob except band_side=both, so a
# delta between the two is the band's SECOND side and nothing else. That is the
# whole point of this arm: the one-sided choice was made on a memory argument
# for SCRIPT, not a quality measurement for CHAT, and CHAT does not pay SCRIPT's
# memory cost (see COST below). So it deserves its own measurement.
#
# WHAT "both" BUYS. A word may be emitted a chunk EARLIER as well as a chunk
# later than the aligner placed it -- u widens UPWARD as well as downward. The
# earlier half is the side that is hard to justify physically: a word cannot
# honestly be emitted before its audio has arrived, which is exactly what
# num_delay_frames=3 guards. If the aligner systematically places words LATE,
# though, the earlier half is the only half that can correct it.
#
# WHAT WE ALREADY KNOW, and why it does NOT settle this. On SCRIPT, two-sided
# lost to one-sided 6.28 vs 5.97 macro and was worse on all seven leaderboard
# splits. That is suggestive, NOT decisive here: the SCRIPT two-sided arm also
# had 16k fewer steps, and more importantly the two families pay for the second
# side very differently -- so a SCRIPT result cannot be assumed to transfer.
#
# COST, and why this arm is cheap where SCRIPT's was not. Widening the band
# inflates only the loss's NODE SET, not the sequence the model sees, so no
# batch-size change is needed and no OOM risk is introduced. Contrast SCRIPT,
# where the second side widens the packed branch spans and so drove the measured
# 0.212 -> 0.337 s/step and the K-padding memory work. Here the expected hit is
# roughly 2x the band nodes against T*U for the full lattice -- real, but small
# next to the encoder and the 151k-vocab joint.
#
# RESUME SAFETY. EXP_NAME differs from the one-sided arm, so this writes its own
# results tree and can never resume from -- or be contaminated by -- those
# weights. That contamination is not hypothetical: the SCRIPT banded arm silently
# resumed from two-sided weights after a band_side switch and had to be wiped.
# ============================================================================

export LOSS_TYPE=banded
export DELAY_FRAMES=3
export RECOVER_WORDS=0
export BAND_CHUNKS=1
export BAND_SIDE=both
export HISTORY_CHUNKS=0
export MAX_DELAY_FRAMES=0
export TARGET_CONSTRUCTION=partition
export DELAY_PUNCT=true
export LR=1e-4
export WARMUP_STEPS=5000
export MAX_STEPS=500000
export EPOCH_STEPS=2000

export CONFIG_NAME=nemotron_chat_transducer_granary2_qwen
export EXP_NAME="${EXP_NAME:-dfw_granary2_chat_banded1_both}"
export INIT_EXCLUDE='["prediction.embed","joint_net"]'

find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/dfw_chat_train.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/dfw_chat_train.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/dfw_chat_train.sh" ]] && { echo "${here}"; return; }
    # ABSOLUTE fallback, and it is not belt-and-braces -- it is the only thing that
    # works on a REQUEUE. Slurm hands the requeued job SLURM_SUBMIT_DIR pointing at
    # the scratch ROOT rather than the directory the job was submitted from, and $0
    # is the spool copy, so both of the lookups above miss. Both CHAT arms died this
    # way every 17 minutes for three hours (exit 127) after their first 4h wall,
    # while the self-contained SCRIPT arms requeued fine.
    local repo="${DFW_CODE_DIR:-/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/hainanx/NeMo_SCRIPT_cc}"
    [[ -f "${repo}/launch/dfw_chat_train.sh" ]] && { echo "${repo}/launch"; return; }
    echo "ERROR: cannot locate dfw_chat_train.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}, repo=${repo})" >&2
    exit 1
}
exec bash "$(find_launch_dir)/dfw_chat_train.sh" "$@"
