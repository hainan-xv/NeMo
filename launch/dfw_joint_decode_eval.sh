#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-joint-decode
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 01:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# Chunk-synchronous joint decoding of the CHAT and SCRIPT v2 arms.
#
#   sbatch launch/dfw_joint_decode_eval.sh
#   NUTTS=500 LAMS=0.3,0.5,0.7 sbatch launch/dfw_joint_decode_eval.sh
#
# A RESEARCH PROBE, not a leaderboard run. The joint decoder recomputes CHAT's
# prediction network from the full prefix every step and re-runs SCRIPT's
# per-chunk prompt for every token, so it is far slower per utterance than
# either model alone. Defaults to a few hundred utterances of one split.
#
# THE SWEEP ALWAYS INCLUDES BOTH ENDPOINTS, and they are the experiment's
# controls rather than padding: lam=1 is CHAT alone and lam=0 is SCRIPT alone,
# through the identical code path. On librispeech/test.clean the v2 arms scored
# 1.69 (CHAT) and 2.01 (SCRIPT) in the 09-16 eval, so lam=1 landing far from
# ~1.7 means the harness is broken -- wrong chunk count, wrong instruction
# string, mismatched padding -- and nothing in between is interpretable. Check
# the endpoints before reading the middle.
#
# ENV
#   NUTTS    utterances per split      (default 200)
#   LAMS     comma-separated CHAT weights (default 0.0,0.25,0.5,0.75,1.0)
#   SPLIT    dataset:split             (default librispeech:test.clean)
#   CHAT_ARM / SCRIPT_ARM   which v2 arms to fuse
# ============================================================================
set -uo pipefail
mkdir -p slurm_out

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
OUT="${OUTPUT_PREFIX:-${DFW}/hainanx}"
CODE_DIR="${CODE_DIR:-${OUT}/NeMo_SCRIPT_cc}"
CONTAINER="${CONTAINER:-${DFW}/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
CACHE_DIR="${CACHE_DIR:-${OUT}/leaderboard_cache}"
HEH="${DFW}/users/heh"
PROJECT="${PROJECT:-SpeechlmDFW}"

CHAT_ARM="${CHAT_ARM:-dfw_granary2_chat_banded1_nodelay_v2}"
SCRIPT_ARM="${SCRIPT_ARM:-dfw_granary2_script_banded1_nodelay_v2}"
NUTTS="${NUTTS:-200}"
LAMS="${LAMS:-0.0,0.25,0.5,0.75,1.0}"
SPLIT="${SPLIT:-librispeech:test.clean}"

CHAT_NEMO="${OUT}/results/${PROJECT}/${CHAT_ARM}/averaged/top5-averaged.nemo"
SCRIPT_CKPT="${OUT}/results/${PROJECT}/${SCRIPT_ARM}/${SCRIPT_ARM}/checkpoints/${SCRIPT_ARM}-averaged.ckpt"

for f in "$CHAT_NEMO" "$SCRIPT_CKPT"; do
    if [[ ! -s "$f" ]]; then
        echo "ERROR: missing model artifact: $f" >&2
        echo "       Both are produced by a leaderboard eval; run one first:" >&2
        echo "       sbatch launch/dfw_eval_v2_all.sh" >&2
        exit 1
    fi
done

echo "==> joint decode probe"
echo "    CHAT   ${CHAT_ARM}"
echo "    SCRIPT ${SCRIPT_ARM}"
echo "    ${SPLIT}, ${NUTTS} utts, lam=${LAMS}"
date

MOUNTS="--container-mounts=/lustre/fsw:/lustre/fsw,${CODE_DIR}:/code,${CACHE_DIR}:${CACHE_DIR},${HEH}:${HEH}"

read -r -d '' cmd <<CMD
export PYTHONPATH="/code:/code/scripts:\${PYTHONPATH}" \
&& export HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 \
&& cd /code \
&& python /code/scripts/joint_decode_eval.py \
    --chat_nemo "${CHAT_NEMO}" \
    --script_ckpt "${SCRIPT_CKPT}" \
    --cache_dir "${CACHE_DIR}" \
    --datasets "${SPLIT}" \
    --max_eval_samples ${NUTTS} \
    --lam "${LAMS}" \
    --verbose
CMD

srun --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
