#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:streaming-stt-chat-qwen
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
# CHAT transducer, arm 2 of 2: Qwen3's 151,936-piece vocabulary.
#
#   sbatch launch/chat_train_qwen.sh          # no arguments, no env vars
#   ./oci_launch.sh launch/chat_train_qwen.sh
#
# Arm 1 (the ~1,024-piece ASR vocabulary) is launch/chat_train.sh, likewise with
# no arguments.
#
# WHY THIS IS A WRAPPER, NOT A COPY. The two arms exist to answer one question --
# does a transducer gain from a large vocabulary the way the SpeechLM does? -- and
# that answer is only meaningful if NOTHING ELSE differs. Two 250-line scripts
# would drift within weeks (a tweak to the learning rate, the delay, the mount
# list) and the comparison would quietly stop being a controlled one, while still
# producing numbers that look publishable. So the entire recipe lives once, in
# chat_train.sh, and this file sets the two variables that define the arm.
#
# The tokenizer difference itself lives in the recipe pair, which is a 3-line
# diff: pretrained_llm, text_vocab_from_asr, vocab_size.
# ============================================================================
set -uo pipefail

export CONFIG_NAME=streaming_stt_granary2_chat_qwenvocab
# Overridable, so an LR sweep can be given its own results directory
# instead of resuming and overwriting the main run of this arm.
export EXP_NAME="${EXP_NAME:-granary2_chat_qwenvocab}"

# Under sbatch, Slurm COPIES the submitted script into a spool directory, so
# BASH_SOURCE points somewhere that has no sibling chat_train.sh -- prefer
# SLURM_SUBMIT_DIR, exactly as eval_asrvocab.sh does.
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

# chat_train.sh reads its DESIGN_NODES back from its OWN "#SBATCH -N" header to
# decide whether the allocation was scaled down (and so whether to append the
# _n<N> suffix that keeps a debug run from resuming and overwriting the real
# run's checkpoints). Slurm, meanwhile, honours the header of the file actually
# submitted -- this one. If the two headers disagreed, a full-scale run here
# would look scaled-down to chat_train.sh, or worse, a 1-node interactive run
# would NOT get the suffix and would clobber the 8-node checkpoints. Neither is
# visible in the logs, so check it.
_this_n="$(grep -m1 -E '^#SBATCH[[:space:]]+-N[[:space:]]+[0-9]+' "$0" | grep -oE '[0-9]+$')"
_that_n="$(grep -m1 -E '^#SBATCH[[:space:]]+-N[[:space:]]+[0-9]+' "${LAUNCH_DIR}/chat_train.sh" | grep -oE '[0-9]+$')"
if [[ "$_this_n" != "$_that_n" ]]; then
    echo "ERROR: node counts disagree -- this script asks for ${_this_n}, chat_train.sh for ${_that_n}." >&2
    echo "       chat_train.sh derives its scaled-down check from its own header, so a mismatch" >&2
    echo "       silently mis-tags EXP_NAME and can overwrite the other arm's checkpoints." >&2
    exit 1
fi

echo "==> CHAT arm 2: Qwen vocabulary"
echo "    config:   ${CONFIG_NAME}"
echo "    exp_name: ${EXP_NAME}"
echo "    recipe:   ${LAUNCH_DIR}/chat_train.sh (shared with arm 1)"

exec bash "${LAUNCH_DIR}/chat_train.sh" "$@"
