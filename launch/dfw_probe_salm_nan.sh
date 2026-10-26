#!/bin/bash
# DFW: localize where a NaN enters the offline SALM audio pathway.
#
# dfw_salm_nocoder_8node trains with loss=nan and the in-train diagnostic reported
# "finite embeds=False, num_frames=540" -- the NaN is present BEFORE the LLM, and it
# is not the zero-target 0/0 that was first suspected (that guard never fired).
#
# Rather than spend another 8-node run per hypothesis, this walks one GPU through
# the four candidate stages separately -- raw audio, log-mel, SpecAugment, encoder,
# projector -- and names the first one that goes non-finite, with the offending cut
# ids. Read-only: no optimizer, no DDP, no checkpoints.
#
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-probe-salm-nan
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

# Same seed the failing run used, so the shuffled stream is comparable. Note the
# config sets shard_seed=randomized, which is exactly why the NaN surfaced at step 4
# in one run and step 1514 in the next -- the offending example moves around.
SEED="${SEED:-42}"
BATCHES="${BATCHES:-400}"

MOUNTS="--container-mounts=${CODE_DIR}:/code,${HFCACHE}:/hfcache,${LUSTRE}:${LUSTRE},${DATA_ROOT}:${DATA_ROOT},${NEMO_USERS}:${NEMO_USERS}"

srun --container-image="$CONTAINER" $MOUNTS bash -c "
  cd /code && export PYTHONPATH=/code && export HF_HOME=/hfcache/ && export HF_HUB_OFFLINE=1 &&
  export HYDRA_FULL_ERROR=1 &&
  python /code/scripts/probe_salm_nan.py \
      --config /code/examples/speechlm2/conf/salm_nocoder.yaml \
      --train_input_cfg '${TRAIN_INPUT_CFG}' \
      --val_manifest '${VAL_MANIFEST}' \
      --pretrained_llm '${PRETRAINED_LLM}' \
      --pretrained_asr '${PRETRAINED_ASR}' \
      --seed ${SEED} --batches ${BATCHES}
"
