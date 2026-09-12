#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:probe-qwen-decode
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:30:00
# No --exclusive / --mem=0 here. This asks for ONE GPU, and the scheduler
# rejects a whole-node memory request against a single GPU as stranding the
# other seven. One GPU is plenty to decode eight utterances.
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Why does the Qwen banded arm drive its training loss from 15.5 to ~1.2 while
# val_wer never beats its epoch-0 value?
#
#   sbatch launch/probe_qwen_decode.sh          <- no arguments
#
# Decodes a handful of validation utterances from the newest checkpoint and
# prints the token IDS next to the text. The ids are the point: an empty
# hypothesis means the model collapsed to blanks, while sensible ids that
# detokenise badly would mean the HuggingFace shim. (The shim already
# round-trips correctly when tested directly, so the ids are the open question.)
#
# Runs on the GRID rather than locally because the checkpoints are 9.8 GB each
# -- copying one costs more than the whole probe.
# ============================================================================
set -uo pipefail

OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
PROJECT="${PROJECT:-SpeechlmScriptCC}"
CODE_DIR="${CODE_DIR:-${OUTPUT_PREFIX}/NeMo_SCRIPT_cc}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"

# The lr 1e-4 arm is the clearest case: its best val_wer (0.8683) is epoch 0, so
# every later epoch made the metric worse while the loss fell.
EXP="${EXP:-granary2_chat_banded1_qwenvocab_lr1e4_gradguard}"
CKPT_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP}/${EXP}/checkpoints"

# Newest, not best: the question is what the TRAINED model emits, and the "best"
# checkpoint here is the untrained epoch 0.
CKPT="$(ls -t "${CKPT_DIR}"/*.ckpt 2>/dev/null | head -1)"
if [[ -z "$CKPT" ]]; then
    echo "ERROR: no checkpoint under ${CKPT_DIR}" >&2
    exit 1
fi
echo "==> probing ${CKPT##*/}"

# Mount the two portfolio ROOTS rather than naming individual directories.
# The validation config reaches into several of them -- audio under /data/ASR,
# manifests under another user's aligned_amos tree, the tokenizer under a third
# -- and adding them one failure at a time cost two scheduling round trips.
# DATA_DIR is additionally aliased to /data because the manifests store audio
# paths as the training container sees them.
DATA_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data
MOUNTS="--container-mounts=${CODE_DIR}:/code,/lustre/fsw/portfolios/llmservice:/lustre/fsw/portfolios/llmservice,/lustre/fsw/portfolios/nemotron:/lustre/fsw/portfolios/nemotron,${DATA_DIR}:/data"

srun --ntasks=1 --nodes=1 --container-image="$CONTAINER" $MOUNTS bash -c "
    cd /code && export PYTHONPATH=/code:\$PYTHONPATH HYDRA_FULL_ERROR=1 &&
    python scripts/qwen_decode_probe.py --ckpt '${CKPT}' --n 8
"
