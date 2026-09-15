#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-chat-banded1
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
# BANDED CHAT on DFW -- the fourth arm.
#
#   sbatch launch/dfw_chat_banded1.sh          <- no arguments
#
# Instead of conditioning on the ONE word-to-chunk assignment the aligner chose
# (dfw_chat_forced.sh) or ignoring it entirely and marginalising the whole lattice
# (dfw_chat_rnnt.sh), this keeps the alignment as a PRIOR: the RNN-T forward is
# restricted to a band around the forced path, so a word may drift a chunk from
# where the aligner put it and the loss sums every path in that band.
#
# The A/B partner of dfw_chat_forced.sh -- identical except band_chunks and
# loss_type, so a delta between them is the band. It is also the CHAT counterpart
# of dfw_script_banded1.sh, which lets the same question be asked of both model
# families on the same data.
#
# ONE-SIDED BAND (band_side=later). A word may be emitted a chunk LATER than the
# aligner placed it, never earlier. In lattice terms u -- the labels emitted by
# chunk t -- widens DOWNWARD only. That is the half aligner error can justify: a
# word whose audio ends just after a chunk boundary cannot honestly be emitted
# before that audio has arrived, which is what num_delay_frames=3 already guards.
# It also roughly halves the lattice against a two-sided band.
#
# COST. band_chunks=1 scores about 3x the nodes of the forced path against T*U for
# the full lattice, and one-sided roughly halves that again. Unlike the SCRIPT
# band, this does NOT inflate the sequence the model sees -- it only widens the
# loss's node set -- so it needs no batch-size change.
# ============================================================================

export LOSS_TYPE=banded
export DELAY_FRAMES=3
export RECOVER_WORDS=0
export BAND_CHUNKS=1
export HISTORY_CHUNKS=0
export MAX_DELAY_FRAMES=0
export TARGET_CONSTRUCTION=partition
export DELAY_PUNCT=true
export LR=1e-4
export WARMUP_STEPS=5000
export MAX_STEPS=500000
export EPOCH_STEPS=2000

export CONFIG_NAME=nemotron_chat_transducer_granary2_qwen
export EXP_NAME="${EXP_NAME:-dfw_granary2_chat_banded1}"
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
