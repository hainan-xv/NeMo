#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:eval-delay-sweep
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 03:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Sweep the INFERENCE latency of the flexible-delay model.
#
#   sbatch launch/eval_chat_delay_sweep.sh          <- no arguments
#
# The model was trained with d ~ U{0..4}: each batch hid the last d frames of
# every chunk, so it should be usable at any latency in that range. This decodes
# ONE averaged checkpoint at each d and reports the whole curve, which is the
# payoff of training that way -- five operating points from a single model.
#
# d is a latency knob, not a quality knob in the usual sense: at d the model
# commits a chunk's words d frames (d * 0.08 s) BEFORE the chunk completes. Low
# d sees more audio; high d answers sooner. The interesting question is how much
# accuracy the last 0.32 s of look-ahead is actually worth.
#
# Each d writes its own results directory (chunk14_offline_trim<d>), so the runs
# do not overwrite one another and the sweep can be resumed.
#
# The averaged .nemo is built once and reused across all five decodes.
# ============================================================================
set -uo pipefail

EXP="${EXP:-granary2_chat_rnnt_flexdelay4_lr1e4}"
DELAYS="${DELAYS:-0 1 2 3 4}"
TOPK="${TOPK:-5}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
PROJECT="${PROJECT:-SpeechlmScriptCC}"
OVERRIDES="model.loss_type=rnnt model.forced_alignment.num_delay_frames=0 \
model.forced_alignment.recover_history_words=0 model.forced_alignment.max_delay_frames=4 \
model.joint.history_chunks=1"

find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_chat.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate eval_chat.sh" >&2; exit 1
}
LAUNCH_DIR="$(find_launch_dir)"

echo "==> delay sweep for ${EXP} over d = ${DELAYS}"
for d in ${DELAYS}; do
    echo
    echo "############################################################"
    echo "### frame_trim = ${d}  (commits ${d} frames early)"
    echo "############################################################"
    ARM_EXP_NAME="${EXP}" ARM_MODEL_OVERRIDES="${OVERRIDES}" TOPK="${TOPK}" \
        FRAME_TRIM="${d}" EVAL_TAG="avg${TOPK}_trim${d}" FORCE_AVERAGE=0 \
        bash "${LAUNCH_DIR}/eval_chat.sh" || echo "    FAILED at d=${d}; continuing" >&2
done

echo
echo "############################################################"
echo "### macro WER by inference delay -- ${EXP}"
echo "############################################################"
printf '  %-6s %-10s %s\n' "d" "latency" "macro WER"
for d in ${DELAYS}; do
    agg="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP}"/eval_*/chunk14_offline_trim${d}/aggregate.log
    macro="$(grep -hE '^RESULT[[:space:]]+Average' $agg 2>/dev/null | awk '{print $3}' | tail -1)"
    printf '  %-6s %-10s %s\n' "${d}" "$(awk "BEGIN{printf \"-%.2fs\", ${d}*0.08}")" "${macro:-(missing)}"
done
