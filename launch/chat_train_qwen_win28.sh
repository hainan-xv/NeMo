#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:streaming-stt-chat-qwen-win28
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 8
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00             # wall time
#SBATCH --time-min 04:00:00
#SBATCH --exclusive             # exclusive node access
#SBATCH --overcommit
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only email on failure
#SBATCH --ntasks-per-node=8     # one task per GPU !!! SET TO NUMBER OF GPUs PER NODE !!!
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# CHAT, Qwen vocabulary, with a 28-FRAME JOINT WINDOW -- the transducer
# analogue of SCRIPT's win28.
#
#   sbatch launch/chat_train_qwen_win28.sh        # no arguments, no env vars
#   ./oci_launch.sh launch/chat_train_qwen_win28.sh
#
# WHAT CHANGES. Only ``joint_history_chunks: 1``. The joint may attend to the
# previous chunk's 14 frames in addition to its own, so it reads 28 frames while
# still emitting on the 14-frame grid.
#
# WHY. Word-to-chunk assignment is by the word's LAST frame, so a word that
# straddles a boundary -- roughly a third of words at 1.12 s chunks -- has its
# onset in the previous chunk. The joint could previously only see the chunk it
# was emitting for, and had to recover that onset from whatever the encoder's
# left context had folded into those 14 frames. Now it can attend to it.
#
# WHAT DOES NOT CHANGE. Emission granularity, emission latency, and look-ahead.
# The window spans [max(0, t-1)*C, (t+1)*C) -- it ends at the current chunk's
# last frame, so the model still never sees audio past the chunk it emits for.
# That is what keeps this comparable to the plain Qwen arm and to SCRIPT.
#
# COMPARE AGAINST: launch/chat_train_qwen.sh (identical but
# joint_history_chunks: 0). The recipe body is shared with chat_train.sh, so the
# arms cannot drift apart.
# ============================================================================
set -uo pipefail

export CONFIG_NAME=streaming_stt_granary2_chat_qwenvocab_win28
export EXP_NAME=granary2_chat_qwenvocab_win28

# Under sbatch, Slurm COPIES the submitted script into a spool directory, so
# BASH_SOURCE points somewhere with no sibling chat_train.sh -- prefer
# SLURM_SUBMIT_DIR, exactly as the other launchers do.
resolve_launch_dir() {
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
LAUNCH_DIR="$(resolve_launch_dir)"

# Slurm honours the -N of the file actually submitted (this one), while
# chat_train.sh reads DESIGN_NODES from its OWN header to decide whether the
# allocation was scaled down and the _n<N> suffix is needed. If those disagreed,
# a 1-node debug run would skip the suffix and resume-then-overwrite the 8-node
# checkpoints -- silently, and only visible much later.
_this_n="$(grep -m1 -E '^#SBATCH[[:space:]]+-N[[:space:]]+[0-9]+' "$0" | grep -oE '[0-9]+$')"
_that_n="$(grep -m1 -E '^#SBATCH[[:space:]]+-N[[:space:]]+[0-9]+' "${LAUNCH_DIR}/chat_train.sh" | grep -oE '[0-9]+$')"
if [[ "$_this_n" != "$_that_n" ]]; then
    echo "ERROR: node counts disagree -- this script asks for ${_this_n}, chat_train.sh for ${_that_n}." >&2
    exit 1
fi

echo "==> CHAT, Qwen vocabulary, 28-frame joint window"
echo "    config:   ${CONFIG_NAME}"
echo "    exp_name: ${EXP_NAME}"
echo "    recipe:   ${LAUNCH_DIR}/chat_train.sh (shared with the other arms)"

exec bash "${LAUNCH_DIR}/chat_train.sh" "$@"
