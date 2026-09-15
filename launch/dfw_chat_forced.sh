#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-chat-forced
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
# pool0-01815 carries an old NVIDIA driver (12020) and fails torch's CUDA init
# outright; it killed job 18686485 in 103s. pool0-00407 is the reference recipe's
# known-bad node.
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# BASELINE 2 of 3 on DFW: FORCED-ALIGNMENT CHAT.
#
#   sbatch launch/dfw_chat_forced.sh          <- no arguments
#
# Cross-entropy over the ONE word-to-chunk assignment the forced aligner chose,
# instead of marginalising over the lattice. The A/B partner of
# dfw_chat_rnnt.sh: identical in every respect except loss_type, so a delta
# between them is the objective and nothing else.
#
# delay=3 frames. A word's final frames are often what disambiguate it, so
# emitting at the chunk where it ENDS leaves the encoder no right context; 3 is
# what the speechlm2 recipe this was ported from used, and 0 measurably slows
# convergence. Note this also sets joint.frame_trim=3 at TRAINING time, so the
# model learns to emit from 11 of each chunk's 14 frames -- any eval of this arm
# must decode at frame_trim=3 or it is measuring a different operating point.
#
# Targets use the partition tokenization and word-final punctuation delay, the
# same fixes the OCI arms carry, so this is the fixed-target forced baseline
# rather than the legacy one.
# ============================================================================

export LOSS_TYPE=forced_alignment
export DELAY_FRAMES=3
export RECOVER_WORDS=0
export HISTORY_CHUNKS=0
export MAX_DELAY_FRAMES=0
export TARGET_CONSTRUCTION=partition
export DELAY_PUNCT=true
export LR=1e-4
export WARMUP_STEPS=5000
export MAX_STEPS=500000
export EPOCH_STEPS=2000

export CONFIG_NAME=nemotron_chat_transducer_granary2_qwen
export EXP_NAME="${EXP_NAME:-dfw_granary2_chat_forced}"
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
