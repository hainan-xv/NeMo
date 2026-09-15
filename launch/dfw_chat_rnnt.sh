#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-chat-rnnt
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
# BASELINE 1 of 3 on DFW: STANDARD CHAT.
#
#   sbatch launch/dfw_chat_rnnt.sh          <- no arguments
#
# The plain RNN-T objective: the loss marginalises over EVERY alignment path in
# the lattice, so the aligner's word timings are never used as targets. This is
# the reference the two alignment-based losses have to beat -- if forced or
# banded cannot improve on marginalising, the alignment is not carrying
# information worth the constraint.
#
# loss_type=rnnt makes every forced_alignment.* knob inert; they are left at the
# body's defaults rather than being set to misleading values here.
#
# Fixed chunk_size 14 (1.12 s), like every current CHAT arm. The encoder's
# att_context_size right context is derived from it in the config.
# ============================================================================

export LOSS_TYPE=rnnt
export HISTORY_CHUNKS=0
export LR=1e-4
export WARMUP_STEPS=5000
export MAX_STEPS=500000
export EPOCH_STEPS=2000

export CONFIG_NAME=nemotron_chat_transducer_granary2_qwen
export EXP_NAME="${EXP_NAME:-dfw_granary2_chat_rnnt}"
export INIT_EXCLUDE='["prediction.embed","joint_net"]'

find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/dfw_chat_train.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/dfw_chat_train.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/dfw_chat_train.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate dfw_chat_train.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
exec bash "$(find_launch_dir)/dfw_chat_train.sh" "$@"
