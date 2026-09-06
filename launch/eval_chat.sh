#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:chat-lb-eval
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Open-ASR-Leaderboard eval for a CHAT transducer.
#
#   ./oci_launch.sh launch/eval_chat.sh granary2_chat_asrvocab_win28_recover
#   RETRACT=1 ./oci_launch.sh launch/eval_chat.sh <exp>
#   RUN_AVERAGING=0 CKPT=<path> ./oci_launch.sh launch/eval_chat.sh <exp>
#
# A THIN WRAPPER over launch/eval_leaderboard.sh -- it sets the driver and model
# class and execs it. Everything that decides a NUMBER is therefore shared with
# the SCRIPT and nemotron evaluations: checkpoint averaging, the dataset list,
# the manifest reader, the seeded length-balanced shard partition, audio padding,
# the aggregation and the scoring.
#
# WHY THAT MATTERS. The first version of this eval was a standalone script, and
# it drifted in three ways that each silently changed the number: it scored with
# NeMo's WER class rather than the leaderboard's normaliser (AMI 11.51 vs
# 11.45); it sharded by DATASET, so spgispeech -- 53% of the corpus -- ran alone
# on one GPU while seven idled; and it evaluated "everything cached", which on
# the grid is TWELVE datasets rather than the leaderboard's seven. Reusing the
# shared launcher makes all three impossible rather than merely fixed.
#
# The CHAT-specific parts live in scripts/chat_leaderboard_eval.py, which mirrors
# speechlm_leaderboard_eval.py and differs only in building a ChatSTTModel and
# decoding through the chunk-synchronous transducer path.
#
# ENV
#   RETRACT    retract-by-k decoding (default: the checkpoint's own setting).
#              Also tags the results directory, so k=0 and k=1 cannot overwrite
#              each other.
#   plus every knob eval_leaderboard.sh accepts (RUN_AVERAGING, FORCE_AVERAGE,
#   CKPT, DATASETS, MAX_EVAL_SAMPLES, BATCH_SIZE, NGPU, PAD_EXTRA_SECONDS, ...)
# ============================================================================
set -uo pipefail

export EVAL_DRIVER="${EVAL_DRIVER:-chat_leaderboard_eval.py}"
export MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.chat_model.ChatSTTModel}"

# Retract is a CHAT-only decode knob, so it rides through EXTRA_EVAL_ARGS rather
# than becoming another flag on the shared launcher.
RETRACT="${RETRACT:-}"
if [[ -n "$RETRACT" ]]; then
    export EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-} --retract ${RETRACT}"
    # Distinct results dir per k, or the second run silently overwrites the first.
    export EVAL_TAG="${EVAL_TAG:-${1:-chat}}_retract${RETRACT}"
fi

# Under sbatch, Slurm COPIES the submitted script into a spool directory, so
# BASH_SOURCE points somewhere with no sibling launcher -- prefer SLURM_SUBMIT_DIR.
resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_leaderboard.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_leaderboard.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_leaderboard.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate eval_leaderboard.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
LAUNCH_DIR="$(resolve_launch_dir)"

echo "==> CHAT leaderboard eval (shared launcher)"
echo "    driver:   ${EVAL_DRIVER}"
echo "    class:    ${MODEL_CLASS}"
echo "    retract:  ${RETRACT:-<checkpoint default>}"

exec bash "${LAUNCH_DIR}/eval_leaderboard.sh" "$@"
