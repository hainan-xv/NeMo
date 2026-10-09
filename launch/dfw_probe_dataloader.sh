#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-probe-dataloader
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:45:00
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
#SBATCH --exclude=pool0-00407,pool0-01815
# ---------------------------------------------------------------------------
# Time the Lhotse dataloader ALONE -- no model, no training.
#
# Four 4-node jobs were killed by the idle-GPU reaper, every one stuck at step
# 0/2000 with "Creating a Lhotse DynamicBucketingSampler" as its last log line.
# The same launcher trained fine on Oct 7 (1662/2000), and wandb is ruled out
# (a wandb-off job on Oct 9 stalled identically). The one thing that changed in
# between is the shared data config, edited Oct 7 18:27.
#
# ONE GPU, not 32: the probe needs none, and a 32-GPU job idling for 30 minutes
# is exactly what the reaper exists to kill. If this probe is itself reaped at
# 30 min, that is still a result -- it means startup exceeds the window.
# ---------------------------------------------------------------------------
set -euo pipefail
DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
MYDIR=${DFW}/hainanx
HEH=${DFW}/users/heh
LUSTRE=/lustre/fsw/portfolios/nemotron
CODE_DIR=${MYDIR}/NeMo_SCRIPT_cc
CONTAINER=${HEH}/containers/nemo-26.02-streaming-speechlm.sqsh
DATA_ROOT=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm
CFG="${CFG:-${HEH}/data_configs/granary_v2_en_full_d0.5_b0.5_dfw_qwen_aligned.yaml}"
MAX_DURATION="${MAX_DURATION:-20}"

echo "### node: $(hostname)"
echo "### input_cfg: ${CFG}"
echo "### cfg mtime: $(stat -c %y "${CFG}" 2>/dev/null | cut -d. -f1)"

MOUNTS="--container-mounts=${CODE_DIR}:/code,${LUSTRE}:${LUSTRE},${DATA_ROOT}:${DATA_ROOT}"
srun --container-image="$CONTAINER" $MOUNTS bash -c "
  cd /code && export PYTHONPATH=/code &&
  python /code/scripts/probe_dataloader.py \
      --input_cfg '${CFG}' --max_duration ${MAX_DURATION} --num_workers 8 --batches 3
"
