#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-band-layout
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:40:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# Compare banded-loss LAYOUTS on a real training batch.
#
#   sbatch launch/dfw_band_layout_check.sh
#
# The layouts cover the same node set, so loss and gradients must agree. The
# toy harness already shows that on synthetic counts; this checks it where the
# geometry is actually stressed -- pauses give empty chunks, dynamic bucketing
# gives very different T/U per utterance, and the alignments are real.
# ============================================================================
set -uo pipefail
mkdir -p slurm_out

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
MY=${DFW}/hainanx
CODE_DIR="${CODE_DIR:-${MY}/NeMo_SCRIPT_cc}"
CONTAINER="${CONTAINER:-${DFW}/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
NEMO="${NEMO:-${MY}/results/SpeechlmDFW/dfw_granary2_chat_spe1k_both/averaged/top5-averaged.nemo}"

[[ -f "$NEMO" ]] || { echo "### no model at $NEMO" >&2; exit 1; }
echo "==> band layout check on $NEMO"

srun --overlap -n1 -N1 --container-image="$CONTAINER" \
     --container-mounts="${DFW}:${DFW},${CODE_DIR}:/code" \
     bash -c "export PYTHONPATH=/code:/code/scripts:${MY}/pylibs:\${PYTHONPATH:-} && \
              cd /code && python scripts/chat_band_layout_check.py --nemo='${NEMO}' --batches=2"
