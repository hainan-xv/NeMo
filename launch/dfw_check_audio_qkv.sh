#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-check-audio-qkv
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:20:00
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
CKPT="${CKPT:?set CKPT}"
srun --container-image="$CONTAINER" \
     --container-mounts=${CODE_DIR}:/code,${LUSTRE}:${LUSTRE} \
     bash -c "cd /code && export PYTHONPATH=/code && python /code/scripts/check_audio_qkv_divergence.py --ckpt '${CKPT}'"
