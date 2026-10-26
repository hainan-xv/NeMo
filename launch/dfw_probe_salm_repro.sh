#!/bin/bash
# DFW: run the LOCAL NaN reproduction against the REAL granary data, on ONE GPU.
#
# Local LibriSpeech does not reproduce the NaN: 40 real training steps at matched
# sequence length (T up to 552) and batch 12 keep the loss finite and descending.
# The 8-node granary runs go NaN by batch 2. The remaining differences are the data,
# the token-bucketing sampler and DDP -- this isolates the first two by running the
# same single-process script with the cluster's own input_cfg and bucketing.
#
# One GPU, interactive, read-only apart from its own stdout.
#
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-salm-repro
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:45:00
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
#SBATCH --exclude=pool0-00407,pool0-01815,pool0-01764

set -uo pipefail
mkdir -p slurm_out

LUSTRE=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
HEH=${LUSTRE}/users/heh
MYDIR=${LUSTRE}/hainanx
CODE_DIR=${MYDIR}/NeMo_SCRIPT_cc
CONTAINER=${HEH}/containers/nemo-26.02-streaming-speechlm.sqsh
DATA_ROOT=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm
NEMO_USERS=/lustre/fsw/portfolios/nemotron/users
HFCACHE=${MYDIR}/hf_cache

PRETRAINED_LLM=${HEH}/pretrained_models/huggingface/Qwen/Qwen3-1.7B
PRETRAINED_ASR=${HEH}/pretrained_models/huggingface/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo
TRAIN_INPUT_CFG=${HEH}/data_configs/granary_v2_en_full_d0.5_b0.5_dfw_qwen_aligned.yaml
VAL_MANIFEST=${HEH}/data/mcv11_en_dev_aligned/mcv11_dev_clean_pcstrip_en_2k_qwen_aligned.json

STEPS="${STEPS:-60}"

MOUNTS="--container-mounts=${CODE_DIR}:/code,${HFCACHE}:/hfcache,${LUSTRE}:${LUSTRE},${DATA_ROOT}:${DATA_ROOT},${NEMO_USERS}:${NEMO_USERS}"

srun --container-image="$CONTAINER" $MOUNTS bash -c "
  cd /code && export PYTHONPATH=/code && export HF_HOME=/hfcache/ && export HF_HUB_OFFLINE=1 &&
  export HYDRA_FULL_ERROR=1 &&
  python /code/scripts/local_salm_repro.py \
      --config /code/examples/speechlm2/conf/salm_nocoder.yaml \
      --llm '${PRETRAINED_LLM}' \
      --asr '${PRETRAINED_ASR}' \
      --train_input_cfg '${TRAIN_INPUT_CFG}' \
      --val_manifest '${VAL_MANIFEST}' \
      --device cuda:0 \
      --steps ${STEPS}
"
