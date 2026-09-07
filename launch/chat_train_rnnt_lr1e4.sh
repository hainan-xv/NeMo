#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:chat-rnnt-lr1e4
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
# PLAIN CHAT (marginalised RNN-T loss) on the reference OPTIMISATION recipe.
#
#   sbatch launch/chat_train_rnnt_lr1e4.sh          <- no arguments
#
# Identical to chat_train_rnnt.sh except for the optimiser schedule, which is
# copied from ~/Workplace/NeMo_ord_sync_d146_current/oci_chat/chat_fullctx.sh --
# the only CHAT recipe found on this machine that uses CosineAnnealing, and the
# closest reference point for a warm-started RNNTAttJoint:
#
#       lr 1e-4     warmup 5000     CosineAnnealing     max_steps 500000
#
# Ours had been lr 1e-3 / warmup 2500 / max_steps 300000: TEN TIMES the
# reference peak, half the warmup, and -- because the runs only ever reach
# ~25-100k of those 300k steps -- a rate still at 76-99% of peak throughout,
# where every reference schedule has decayed hard. The gap widened with
# training (4.5x at 5k steps, 15x at 100k), which matches a model that trails
# the donor RNN-T by more the longer it trains.
#
# WHAT IS DELIBERATELY NOT COPIED. chat_fullctx.sh trains a FULL-CONTEXT
# encoder ([-1,-1], regular attention, non-causal subsampling) from parakeet.
# This keeps our chunked streaming encoder ([70,13], chunked_limited, causal
# downsampling) and our Granary 2.0 data, bucketing, vocabulary and warm start,
# so the only thing that changes against chat_train_rnnt.sh is the schedule.
# ============================================================================

export LOSS_TYPE=rnnt
export DELAY_FRAMES=0
export RECOVER_WORDS=0
export HISTORY_CHUNKS=0
export MAX_DELAY_FRAMES=0

# --- the reference optimisation recipe ---
export LR=1e-4
export WARMUP_STEPS=5000
export MAX_STEPS=500000

export EXP_NAME="${EXP_NAME:-granary2_chat_rnnt_lr1e4_wu5k}"

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
