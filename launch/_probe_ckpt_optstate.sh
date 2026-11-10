#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:ckpt-optstate-probe
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:20:00
#SBATCH --exclusive
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
set -uo pipefail
LUSTRE=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
HEH=${LUSTRE}/users/heh
MYDIR=${LUSTRE}/hainanx
CONTAINER="${HEH}/containers/nemo-26.02-streaming-speechlm.sqsh"
CODE_DIR="${MYDIR}/NeMo_SCRIPT_cc"
CONV=${MYDIR}/results/SpeechlmDFW/dfw_script_convenc_causal_8node/checkpoints
STACK=${MYDIR}/results/SpeechlmDFW/dfw_joint_stack3_p84_int/checkpoints
srun --container-image="$CONTAINER" \
     --container-mounts=${CODE_DIR}:/code,${MYDIR}:${MYDIR},${LUSTRE}:${LUSTRE} \
     bash -c "export PYTHONPATH=/code:\${PYTHONPATH}; cd /code
C=\$(ls -t ${CONV}/*-last.ckpt 2>/dev/null | head -1)
S=\$(ls -t ${STACK}/*-last.ckpt 2>/dev/null | head -1)
echo '################ CONV (fails to resume):' \$C
python -u /code/_probe_ckpt_optstate.py \$C
echo
echo '################ STACK3 (resumes fine):' \$S
python -u /code/_probe_ckpt_optstate.py \$S"
