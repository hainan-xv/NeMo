#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:smoke-chat-banded-qwen
# INTERACTIVE partition: this is a single-node throwaway that only has to reach
# the first validation, and the batch blocks queue behind 8-node training for
# hours. The admin limit is ONE interactive job per user, so do not run this at
# the same time as an eval.
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 01:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# SMOKE TEST for chat_train_banded_qwen.sh.
#
#   sbatch launch/smoke_chat_banded_qwen.sh          <- no arguments
#
# Same model, same loss, same tokenizer as the real run -- only the schedule
# and the allocation differ. 60-step epochs, so it reaches VALIDATION in a few
# minutes instead of an hour.
#
# WHY VALIDATION IS THE POINT. Every failure this arm has had so far was in the
# decode path, not the training path: the 151.7k Qwen vocabulary reaches
# tokens_to_text as ids with an out-of-vocabulary blank, and a greedy hypothesis
# that begins with a blank used to crash there. Training steps alone prove
# nothing about that, so a useful smoke test is one that VALIDATES -- the short
# epoch exists to get there quickly.
#
# EXP_NAME is deliberately distinct from the real run's. exp_manager RESUMES
# from a matching directory, so a shared name would let this throwaway write
# 60-step checkpoints into an 8-node run's checkpoint directory.
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

# Short epochs -> validation runs early and often. This is the only line that
# makes this a smoke test rather than a one-node copy of the real run.
export EPOCH_STEPS=60
export EXP_NAME=granary2_chat_banded1_qwenvocab_lr1e4_smoke

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
