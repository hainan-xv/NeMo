#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:chat-banded-qwen-lr1e4-guard
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
# BANDED loss on QWEN3's ~151.7k vocabulary.
#
#   sbatch launch/chat_train_banded_qwen.sh          <- no arguments
#
# Identical to chat_train_banded_lr1e4.sh except for the tokenizer, so the pair
# isolates the effect of vocabulary size. This is the case the band exists for:
# the full RNN-T loss needs a [B, T, U, V+1] tensor, which at V = 151,670 does
# not fit, while the band scores ~3U + T nodes and so grows with the alignment
# rather than with the lattice.
#
# The tokenizer does NOT come from the donor .nemo (that is the 1,024-piece
# SentencePiece model), so TOKENIZER_DIR is overridden to the Qwen directory and
# the extraction step is skipped.
#
# The warm start covers the encoder and joint.enc/pred only: the prediction
# network's embedding and the joint output layer are 148x larger here and have
# no counterpart in the donor, so they start random in both vocabulary arms --
# which is what keeps the comparison about the vocabulary.
#
# LOWER LR. The lr 1e-4 arm trained cleanly and then went NaN and stayed NaN:
# 8 nodes reached loss 12.8 by ~355 updates, 1 node reached 11.4 by ~360, and
# both collapsed there. Hitting it at the same STEP count under an 8x different
# global batch size is the tell -- a single bad batch would arrive 8x sooner in
# steps on 8 nodes, so this is step-driven (the optimiser), not data-driven.
#
# Both arms randomly initialise a 151.7k-row prediction embedding and a 151.7k
# output layer (INIT_EXCLUDE), which is 200M untrained parameters that the
# 1,024-piece arm warm-started instead -- so this is the arm where the step size
# has the most room to be wrong.
#
# lr 1e-4 RETRIED, now that the gradient guard is in place.
#
# The first lr 1e-4 attempt died permanently at step ~355. The cause was not the
# step size but gradient_clip_val: 1.0 -- clipping scales by max_norm/total_norm,
# which for an infinite total_norm is zero, and inf * 0 = NaN, so ONE overflowing
# gradient was converted into NaN weights. skip_nan_grad (now true in the config)
# runs in on_after_backward, before clipping, and drops that step instead.
#
# Dropping to lr 3e-5 only postponed the collapse (step ~355 -> ~1793), which is
# what showed the gradient rather than the step size was the problem. With the
# guard, 3e-5 then ran 3085 steps with exactly ONE gradient skip. This arm tests
# whether 1e-4 is now stable too, since it should converge faster if it is.
#
# EXP_NAME carries _gradguard deliberately. The dead lr 1e-4 run left a
# -last.ckpt holding NaN weights in granary2_chat_banded1_qwenvocab_lr1e4, and
# exp_manager resumes on a name match -- reusing the name would load the NaN
# weights and reproduce the failure instantly, looking like the guard had failed.
# ============================================================================

export LOSS_TYPE=banded
export DELAY_FRAMES=3
export BAND_CHUNKS=1
export RECOVER_WORDS=0
export HISTORY_CHUNKS=0
export MAX_DELAY_FRAMES=0
export LR=1e-4
export WARMUP_STEPS=5000
export MAX_STEPS=500000

export CONFIG_NAME=nemotron_chat_transducer_granary2_qwen
export TOKENIZER_DIR=/lustre/fsw/portfolios/llmservice/users/heh/pretrained_models/huggingface/Qwen/Qwen3-1.7B
export INIT_EXCLUDE='["prediction.embed","joint_net"]'
export EXP_NAME="${EXP_NAME:-granary2_chat_banded1_qwenvocab_lr1e4_gradguard}"

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
