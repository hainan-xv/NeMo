#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-chunk-err-probe
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH --time-min 00:30:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# How often is the frozen CHAT model wrong, on TRAIN vs HELD-OUT?
#
#   sbatch launch/dfw_chunk_error_probe.sh
#
# Sizes the corpus for a verifier/corrector before building it. Runs the SAME
# probe against both sources so the numbers are directly comparable; the GAP
# between them is the result that matters, because it says whether training-set
# hypotheses can be used at all or whether they must come from a held-out-fold
# or deliberately weakened decode.
# ============================================================================
set -uo pipefail
mkdir -p slurm_out

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
OUT="${OUTPUT_PREFIX:-${DFW}/hainanx}"
CODE_DIR="${CODE_DIR:-${OUT}/NeMo_SCRIPT_cc}"
CONTAINER="${CONTAINER:-${DFW}/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
HEH="${DFW}/users/heh"
ARM="${ARM:-dfw_granary2_chat_banded1_nodelay_v2}"
NEMO="${OUT}/results/SpeechlmDFW/${ARM}/averaged/top5-averaged.nemo"
BATCHES="${BATCHES:-40}"
VAL_MANIFEST="${VAL_MANIFEST:-${HEH}/data/mcv11_en_dev_aligned/mcv11_dev_clean_pcstrip_en_2k_qwen_aligned.json}"

if [[ ! -s "$NEMO" ]]; then
    echo "ERROR: no averaged .nemo at ${NEMO}" >&2; exit 1
fi

echo "==> chunk-error probe: ${ARM}"; date

MOUNTS="--container-mounts=/lustre/fsw:/lustre/fsw,${CODE_DIR}:/code,${HEH}:${HEH}"
read -r -d '' cmd <<CMD
export PYTHONPATH="/code:/code/scripts:\${PYTHONPATH}" HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 \
&& cd /code \
&& echo "################ TRAIN ################" \
&& python /code/scripts/chat_chunk_error_probe.py --nemo "${NEMO}" --source train --batches ${BATCHES} \
     --dump /code/slurm_out/chunk_err_train.jsonl ; \
echo "################ HELD-OUT ################" ; \
python /code/scripts/chat_chunk_error_probe.py --nemo "${NEMO}" --source manifest --manifest "${VAL_MANIFEST}" \
     --batches ${BATCHES} --dump /code/slurm_out/chunk_err_val.jsonl
CMD
srun --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
