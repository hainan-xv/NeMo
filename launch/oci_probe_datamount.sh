#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:probe-datamount
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 00:10:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
# ---------------------------------------------------------------------------
# Why every training resubmission since 15:11 fails:
#   enroot-mount: failed to mount: .../llmservice_nemo_speechlm/data  (ENOENT)
# The path IS visible from the login node, and the same nodes ran 4h training
# jobs this morning. The eval jobs run fine and differ in ONE way: they mount
# /lustre/fsw wholesale instead of bind-mounting the deep path.
#
# This probe separates the two possibilities:
#   (a) the deep path is genuinely gone on compute nodes -> no mount flag helps
#   (b) only the deep BIND MOUNT fails -> mounting /lustre/fsw is the fix
# Mounts only /lustre/fsw (the eval-style mount, known to work right now).
# ---------------------------------------------------------------------------
set -uo pipefail
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
D=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data

echo "### node: $(hostname)"
echo "### from the COMPUTE NODE, outside the container:"
ls -d "$D" 2>&1 | head -2

srun --container-image="${CONTAINER}" --container-mounts=/lustre/fsw:/lustre/fsw bash -c "
  echo '### INSIDE container (eval-style /lustre/fsw mount):'
  ls -d $D 2>&1 | head -2
  echo '--- can we actually read it? ---'
  ls $D 2>&1 | head -5
"
