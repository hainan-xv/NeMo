#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:chat-banded-qwen
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

export CONFIG_NAME=nemotron_chat_transducer_granary2_qwen
export TOKENIZER_DIR=/lustre/fsw/portfolios/llmservice/users/heh/pretrained_models/huggingface/Qwen/Qwen3-1.7B
export INIT_EXCLUDE='["prediction.embed","joint_net"]'
export EXP_NAME="${EXP_NAME:-granary2_chat_banded1_qwenvocab_lr1e4}"

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
