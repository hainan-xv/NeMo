#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-prof-model-samenode
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 01:00:00
#SBATCH --time-min 00:30:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
#SBATCH --exclude=pool0-00407,pool0-01815
# ---------------------------------------------------------------------------
# force_word_start A/B with the NODE held fixed.
#
# The previous pair (19769513/19769514) ran CONCURRENTLY ON DIFFERENT NODES --
# pool0-01296 vs pool0-01682 -- so the 11:49-vs-7:01 split confounds the flag
# with the hardware. Locally, on identical weights and audio, the two settings
# are indistinguishable (1.92s vs 1.93s) at every utterance length tested up to
# 52 s, which says the decode path is not the cost. This runs both settings
# BACK TO BACK in ONE allocation so node, contention and cache are shared.
# ---------------------------------------------------------------------------
set -euo pipefail
resolve_launch_dir() {
    [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/launch/eval_leaderboard.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    [[ -f "${CODE_DIR:-}/launch/eval_leaderboard.sh" ]] && { echo "${CODE_DIR}/launch"; return; }
    echo "ERROR: cannot locate eval_leaderboard.sh" >&2; exit 1
}
DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-${DFW}/hainanx}"
export CODE_DIR="${CODE_DIR:-${OUTPUT_PREFIX}/NeMo_SCRIPT_cc}"
export CONTAINER="${CONTAINER:-${DFW}/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
export CACHE_DIR="${CACHE_DIR:-${OUTPUT_PREFIX}/leaderboard_cache}"
export H_DIR="${H_DIR:-${DFW}/users/heh}"
export PROJECT="${PROJECT:-SpeechlmDFW}"
EXP_NAME="${EXP_NAME:-dfw_script_multilookahead_bandchunk113}"; export EXP_NAME
export MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"
export SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are doing streaming speech recognition. Given the transcript so far and the representation of the next audio chunk, output the words spoken in that chunk.}"
export CHUNK_SIZE="${CHUNK_SIZE:-14}"
export DATASETS="${DATASETS:-ami_cleaned:test}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-400}"
export RUN_AVERAGING="${RUN_AVERAGING:-1}"
export FORCE_AVERAGE="${FORCE_AVERAGE:-0}"
export USE_LAST="${USE_LAST:-0}"
BACKEND="$(resolve_launch_dir)/eval_leaderboard.sh"
# MODEL A/B with node, data, flags and inference code all held fixed.
# The inference path is byte-identical between these two models (verified by
# git diff: every change since the band work lands in the training packer, the
# loss, or argument plumbing -- never in generate() or the decode loop). So if
# one model is dramatically slower on the SAME audio, the cost is in what it
# GENERATES: the loop runs to <eot> and discards everything after a <read>
# gate, so a model that gates read without stopping burns max_new_tokens while
# emitting no text -- invisible in WER, expensive in wall time.
R=${OUTPUT_PREFIX}/results/${PROJECT}
OLD=$R/dfw_script_multilookahead_banded113_scratch/dfw_script_multilookahead_banded113_scratch/checkpoints/dfw_script_multilookahead_banded113_scratch-averaged.ckpt
NEW=$R/dfw_script_multilookahead_bandchunk113/dfw_script_multilookahead_bandchunk113/checkpoints/dfw_script_multilookahead_bandchunk113-averaged.ckpt
export RUN_AVERAGING=0
echo "### node: $(hostname)"
for m in "old:$OLD" "new:$NEW"; do
    tag=${m%%:*}; ck=${m##*:}
    echo ""
    echo "############ model=${tag} on $(hostname) ############"
    echo "     ckpt=${ck}"
    [[ -s "$ck" ]] || { echo "MISSING: $ck" >&2; continue; }
    t0=$(date +%s)
    CKPT="$ck" FORCE_WORD_START=0 RESULTS_SUFFIX="model_${tag}" bash "$BACKEND" || echo "WARN: ${tag} failed" >&2
    echo "#### model=${tag} wall=$(( $(date +%s) - t0 ))s"
done
