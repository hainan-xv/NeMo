#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:chat-forced
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
# CHAT with the FORCED-ALIGNMENT objective -- the win28 + history-recovery arm.
#
#   sbatch launch/chat_train_forced.sh
#
# A thin wrapper so each arm launches bare, with no parameters to remember and
# no chance of one arm's settings leaking into the other. Everything else --
# data, encoder init, tokenizer, schedule -- comes from chat_train.sh, which is
# the point: the two runs differ in these four lines and nothing more.
#
# HISTORY_CHUNKS=1 is "win28": the joint's keys and values also cover the
# previous chunk, strictly backward-looking so it costs no latency.
# RECOVER_WORDS=2 additionally scores each chunk on the previous chunk's last two
# words, which is what lets a decoder retract and have the model restore them.
# ============================================================================

export LOSS_TYPE=forced_alignment
export HISTORY_CHUNKS="${HISTORY_CHUNKS:-1}"
export RECOVER_WORDS="${RECOVER_WORDS:-2}"
export EXP_NAME="${EXP_NAME:-granary2_chat_forced_asrvocab_win28_recover}"

# Under sbatch, $0 is a COPY of this script in Slurm's spool directory, so
# dirname "$0" has no sibling chat_train.sh and the job dies with exit 127 five
# seconds in, before it creates a results directory to leave a log in.
# SLURM_SUBMIT_DIR is where the sbatch was issued, which is the repo root.
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
