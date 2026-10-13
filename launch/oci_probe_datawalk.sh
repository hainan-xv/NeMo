#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:probe-datawalk
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 00:10:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
# Walks the training-data path level by level, on the host and inside the
# container, to find exactly where container visibility stops. Probe 13973013
# showed the leaf is visible on the host but not in the container even with
# /lustre/fsw mounted wholesale -- this pins down which level diverges.
set -uo pipefail
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
echo "### node: $(hostname)"
echo "### HOST view:"
for p in /lustre/fsw /lustre/fsw/portfolios /lustre/fsw/portfolios/llmservice \
         /lustre/fsw/portfolios/llmservice/projects \
         /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm \
         /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data; do
  printf "  %-72s %s\n" "$p" "$(ls -d "$p" >/dev/null 2>&1 && echo OK || echo MISSING)"
done
echo "### is it a separate mount point on the host?"
mount | grep -iE "lustre|llmservice" | head -5
srun --container-image="${CONTAINER}" --container-mounts=/lustre/fsw:/lustre/fsw bash -c '
  echo "### CONTAINER view:"
  for p in /lustre/fsw /lustre/fsw/portfolios /lustre/fsw/portfolios/llmservice \
           /lustre/fsw/portfolios/llmservice/projects \
           /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm \
           /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data; do
    printf "  %-72s %s\n" "$p" "$(ls -d "$p" >/dev/null 2>&1 && echo OK || echo MISSING)"
  done
  echo "### what IS visible under projects/ in the container:"
  ls /lustre/fsw/portfolios/llmservice/projects/ 2>&1 | head -8
'
