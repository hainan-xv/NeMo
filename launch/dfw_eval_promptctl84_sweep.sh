#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-eval-promptctl84-sweep
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
#SBATCH --exclude=pool0-00407,pool0-01815
# ---------------------------------------------------------------------------
# promptctl84: all five chunk sizes, prompt-controlled, in ONE allocation.
#
# Why one job and not two: the first eval in an allocation pays a large cold
# cost -- container start, a 5 GB checkpoint and the dataset cache read cold off
# lustre. Measured back-to-back on the SAME node, identical work ran 1503 s
# first and 160 s second; the force_word_start pair showed the same ordering
# effect (155 s then 103 s). Splitting chunk 7 and chunk 2 into separate jobs
# made each pay that cost again, and my batches/min samples -- taken inside the
# cold window -- projected 3.5 h and 18 h and got both jobs cancelled. Sharing
# one allocation amortises it, which is exactly why the latency sweep did all
# three chunk sizes in 46 min.
#
# The averaged checkpoint is the MODEL, not a decode setting, so it is built at
# most once and reused across chunk sizes (FORCE_AVERAGE=0; it already exists).
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
EXP_NAME="${EXP_NAME:-dfw_script_promptctl84_scratch}"; export EXP_NAME
export MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"
export SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are doing streaming speech recognition. Given the transcript so far and the representation of the next audio chunk, output the words spoken in that chunk.}"
# PROMPT CONTROL IS MANDATORY HERE. These checkpoints do not record the flag, so
# without it generate() skips render_control_prompt and decodes on a bare
# instruction the model never saw in training -- the control options are accepted and
# silently discarded, the run looks healthy, and the WER is meaningless.
export PROMPT_CONTROL=1
export NUM_DELAY_FRAMES="${NUM_DELAY_FRAMES:-3}"
export CAPITALIZATION="${CAPITALIZATION:-1}"
export PUNCTUATION="${PUNCTUATION:-1}"
# Forced-loss arm: words are emitted whole, so the word-start guard matches training.
export FORCE_WORD_START="${FORCE_WORD_START:-1}"
export RESULTS_SUFFIX="${RESULTS_SUFFIX:-d3c1p1}"
export RUN_AVERAGING="${RUN_AVERAGING:-1}"
export FORCE_AVERAGE="${FORCE_AVERAGE:-0}"
export USE_LAST="${USE_LAST:-0}"
BACKEND="$(resolve_launch_dir)/eval_leaderboard.sh"
echo "### node: $(hostname)"
# All five prompt-controlled sizes in ONE allocation so the cold start
# (container + 5 GB checkpoint + dataset cache off lustre) is paid once.
CHUNK_LIST="${CHUNK_LIST:-14 28 42 7 2}"
for cs in ${CHUNK_LIST}; do
    echo ""
    echo "############ ${EXP_NAME} chunk_size=${cs} fws=0 ############"
    t0=$(date +%s)
    CHUNK_SIZE="$cs" bash "$BACKEND" || echo "WARNING: chunk ${cs} failed; continuing" >&2
    echo "#### chunk=${cs} wall=$(( $(date +%s) - t0 ))s"
done
