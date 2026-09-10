#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:eval-chat-all
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 03:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Leaderboard eval of all three models currently in training, in ONE job.
#
#   sbatch launch/eval_chat_all.sh          <- no arguments
#
# Each arm is averaged over its top-5 checkpoints by val_wer and then evaluated
# on the full Open-ASR-Leaderboard set. The arms run SEQUENTIALLY: each one
# already fans its shard out across all 8 GPUs, so running them concurrently
# would only make them contend for the same devices.
#
# The data split is the shared one from scripts/leaderboard_common.py --
# select_shard() pools every utterance across the seven datasets and partitions
# it seeded and length-balanced, so each GPU gets a roughly equal share of AUDIO
# rather than a whole dataset. That is what makes these numbers comparable to
# each other and to the nemotron baseline.
#
# Averaging uses NeMo's own scripts/checkpoint_averaging/average_model_checkpoints.py
# via launch/eval_chat.sh; nothing here reimplements it.
#
# Re-running is cheap: an arm whose averaged .nemo already exists is not
# re-averaged unless FORCE_AVERAGE=1.
#
# ENV
#   ARMS            override the list of "exp_name|overrides" entries
#   TOPK            checkpoints to average (default 5)
#   FORCE_AVERAGE   1 to rebuild an averaged .nemo that already exists
# ============================================================================
set -uo pipefail

TOPK="${TOPK:-5}"

# One entry per model: results directory, then the hydra overrides that rebuild
# ITS architecture. The overrides matter -- averaging constructs the model
# before loading weights, and history_chunks or the loss type differ per arm, so
# a wrong value here is a shape error at best and a silently different model at
# worst.
FA=model.forced_alignment
DEFAULT_ARMS=(
  "granary2_chat_rnnt_lr1e4_wu5k|model.loss_type=rnnt ${FA}.num_delay_frames=0 ${FA}.recover_history_words=0 ${FA}.max_delay_frames=0 model.joint.history_chunks=0"
  "granary2_chat_rnnt_flexdelay4_lr1e4|model.loss_type=rnnt ${FA}.num_delay_frames=0 ${FA}.recover_history_words=0 ${FA}.max_delay_frames=4 model.joint.history_chunks=1"
  "granary2_chat_banded1_delay3_lr1e4|model.loss_type=banded ${FA}.num_delay_frames=3 ${FA}.recover_history_words=0 ${FA}.band_chunks=1 ${FA}.max_delay_frames=0 model.joint.history_chunks=0"
)
if [[ -n "${ARMS:-}" ]]; then
    read -r -a ARM_LIST <<< "${ARMS}"
else
    ARM_LIST=("${DEFAULT_ARMS[@]}")
fi

OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
PROJECT="${PROJECT:-SpeechlmScriptCC}"

find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_chat.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate eval_chat.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
LAUNCH_DIR="$(find_launch_dir)"

echo "==> evaluating ${#ARM_LIST[@]} models, one after another"
for entry in "${ARM_LIST[@]}"; do
    echo "      ${entry%%|*}"
done
echo

declare -a STATUS=()
for entry in "${ARM_LIST[@]}"; do
    exp="${entry%%|*}"
    overrides="${entry#*|}"

    echo "############################################################"
    echo "### ${exp}"
    echo "############################################################"

    if [[ ! -d "${OUTPUT_PREFIX}/results/${PROJECT}/${exp}/${exp}/checkpoints" ]]; then
        echo "    SKIPPED: no checkpoints yet" >&2
        STATUS+=("${exp}|skipped (no checkpoints)")
        continue
    fi

    # A failure in one arm must not abandon the other two -- the whole point of
    # batching them is to come back to a full table.
    ARM_EXP_NAME="${exp}" ARM_MODEL_OVERRIDES="${overrides}" TOPK="${TOPK}" \
        FORCE_AVERAGE="${FORCE_AVERAGE:-0}" EVAL_TAG="avg${TOPK}" \
        bash "${LAUNCH_DIR}/eval_chat.sh"
    rc=$?
    if [[ $rc -eq 0 ]]; then
        STATUS+=("${exp}|ok")
    else
        echo "    FAILED (exit ${rc}); continuing with the next model" >&2
        STATUS+=("${exp}|FAILED (exit ${rc})")
    fi
    echo
done

echo "############################################################"
echo "### summary"
echo "############################################################"
for s in "${STATUS[@]}"; do
    printf '  %-46s %s\n' "${s%%|*}" "${s#*|}"
done
echo
echo "  WER tables:"
for entry in "${ARM_LIST[@]}"; do
    exp="${entry%%|*}"
    for agg in "${OUTPUT_PREFIX}/results/${PROJECT}/${exp}"/eval_*/chunk14_offline/aggregate.log; do
        [[ -f "$agg" ]] || continue
        printf '\n  === %s\n' "${exp}"
        grep -E '^RESULT' "$agg" | sed 's/^/    /'
    done
done
