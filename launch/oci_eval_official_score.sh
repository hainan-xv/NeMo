#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:oci-eval-official-score
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 00:30:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# THE OFFICIAL SCORE for every arm evaluated by oci_eval_official_*.sh.
#
#   sbatch launch/oci_eval_official_score.sh      <- final table, any time
#   bash   launch/oci_eval_official_score.sh --inline   <- from inside a job
#
# run_eval.py PRINTS evaluate/jiwer WER; the leaderboard PUBLISHES what
# normalizer/eval_utils.score_results() computes -- kaldialign batch_error_rate
# with merge_compounds=True, which counts a split compound ("white paper" vs
# "whitepaper") as zero errors. That is 0.3-0.5 WER more lenient, so the WER
# lines in the per-dataset logs are systematically PESSIMISTIC and are not
# comparable to any published figure. This is the number that is.
#
# MACRO-7 EXCLUDES plain earnings22, exactly as the DFW table does: it is
# superseded by earnings22_cleaned_aa_chunked, and counting both would weight
# the same corpus twice. Only arms with all 7 datasets present get a row, so a
# still-running arm is omitted rather than averaged over a subset -- a partial
# average would silently look like a real one.
# ============================================================================
set -uo pipefail
mkdir -p slurm_out

LUSTRE=/lustre/fsw/portfolios/nemotron
MY=${LUSTRE}/users/hainanx
HEH=/lustre/fsw/portfolios/llmservice/users/heh
CONTAINER="${CONTAINER:-${HEH}/containers/nemo-26.02-streaming-speechlm.sqsh}"
CODE_DIR="${CODE_DIR:-${MY}/NeMo_SCRIPT_cc}"
OASR="${MY}/open_asr_leaderboard"
PYLIBS="${MY}/pylibs"

srun --overlap -n1 -N1 --container-image="$CONTAINER" \
     --container-mounts="${LUSTRE}:${LUSTRE},${HEH}:${HEH},${CODE_DIR}:/code,${OASR}:/oasr" \
     bash -c "export PYTHONPATH=/code:/code/scripts:/oasr:${PYLIBS}:\${PYTHONPATH:-} && \
              python /code/scripts/score_leaderboard_snapshot.py ${OASR}/nemo_asr/results --oasr /oasr" \
  2>&1 | grep -vE "^srun:|CSV Summary|^\*{4,}|^model,"
