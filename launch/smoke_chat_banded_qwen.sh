#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:smoke-chat-banded-qwen
# INTERACTIVE partition: the batch blocks queue behind 8-node training for
# hours, and the point of this copy is to find out QUICKLY whether the real run
# works. The admin limit is ONE interactive job per user, so do not run this at
# the same time as an eval.
#SBATCH -p interactive
#SBATCH -N 1
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
# One-node copy of chat_train_banded_qwen.sh, for a quick first look.
#
#   sbatch launch/smoke_chat_banded_qwen.sh          <- no arguments
#
# ONLY THE ALLOCATION DIFFERS: 1 node on the interactive partition instead of 8
# on the batch blocks. The model, the loss, the tokenizer and the SCHEDULE are
# the real run's, line for line. A shortened epoch would have made this a test
# of a configuration nobody is going to train, which is worth very little --
# whatever this job hits, the 8-node job hits too.
#
# EXP_NAME is the real run's, deliberately. chat_train.sh appends _n<N> by
# itself whenever the allocation does not match the -N it declares, so this job
# lands in ..._n1 and cannot resume from or overwrite the 8-node run. That guard
# reads the -N of chat_train.sh (which is exec'd, so $0 is chat_train.sh, not
# this wrapper), so it fires here even though this script also says -N 1 --
# hardcoding a _n1 suffix would only get doubled into _n1_n1.
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
export EXP_NAME=granary2_chat_banded1_qwenvocab_lr1e4

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
