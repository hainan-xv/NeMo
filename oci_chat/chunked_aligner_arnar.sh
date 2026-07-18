#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:chunked-aligner-arnar-g2
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 4
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00            # wall time
#SBATCH --time-min 04:00:00
#SBATCH --exclusive             # exclusive node access
#SBATCH --overcommit
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --ntasks-per-node=8    # n tasks per machine (one task per GPU) !!!WARNING!!! - SET THIS TO NUMBER OF GPUs per Node
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Stochastically autoregressive / non-autoregressive Chunked-Aligner.
#
# Identical to oci_chat/chunked_aligner.sh, but enables the HAINAN decoder-masking
# regularizer (model.joint.masking_prob): during TRAINING the decoder
# (prediction-net) contribution to the joint is randomly zeroed, independently per
# output position u, with probability MASKING_PROB. Masked positions must be
# predicted from the encoder alone (NAR); unmasked positions use the AR history --
# so the model is trained to work both autoregressively and non-autoregressively.
# Inference is unchanged (standard AR chunked greedy decode).
#
# This is a thin wrapper: it sets MASKING_PROB (default 0.5) + a distinct exp name
# and then runs the main chunked_aligner.sh launcher body inside this Slurm job
# (the #SBATCH directives above define the allocation for this script).
#
# Submit from an OCI login node:
#   ./sync_to_oci.sh
#   sbatch oci_chat/chunked_aligner_arnar.sh
#   # tune the masking probability:
#   MASKING_PROB=0.3 sbatch oci_chat/chunked_aligner_arnar.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Probability of masking the decoder contribution to the joint (per output
# position, training only). 0.5 = the HAINAN default stochastic AR/NAR mix.
export MASKING_PROB="${MASKING_PROB:-0.5}"

# Run the main chunked-aligner launcher body inside this job (its #SBATCH lines are
# comments here; this wrapper's allocation is used, and the main script's srun
# fans the job out across the allocated nodes). EXP_NAME is derived inside the main
# script and automatically gets a _mask<MASKING_PROB> tag when masking is on.
bash "${SCRIPT_DIR}/chunked_aligner.sh"
