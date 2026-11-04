#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-donor-decode
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 00:30:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Does the SEEDED CHAT head decode at step 0, before any training?
#
# Identical to dfw_measure_grad_balance.sh except for the partition. Interactive
# QOS caps at 2 nodes per user and the summed training run holds both, so a
# 1-node measurement queued there waits on a job with hours left -- and freeing
# the slot would mean killing training to run a diagnostic.
#
# At init, ||dL_chat/d enc|| was ~3000x smaller than ||dL_script/d enc||, so the
# encoder was being shaped almost entirely by SCRIPT. val_wer_chat has since
# fallen 0.750 -> 0.665 while val_wer sits flat at ~0.109, which is consistent
# either with CHAT catching up or with it merely adapting to a representation it
# does not influence. This measures which.
#
# Reads a SNAPSHOT of the checkpoint: '-last.ckpt' is rewritten at every
# validation, and a 15GB file being overwritten mid-read is its own failure mode.
# ============================================================================
set -uo pipefail
mkdir -p slurm_out

LUSTRE=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
HEH=${LUSTRE}/users/heh
MYDIR=${LUSTRE}/hainanx
CONTAINER="${CONTAINER:-${HEH}/containers/nemo-26.02-streaming-speechlm.sqsh}"
CODE_DIR="${CODE_DIR:-${MYDIR}/NeMo_SCRIPT_cc}"
DONOR=${MYDIR}/results/SpeechlmDFW/dfw_granary2_chat_banded1_both_nodelay_v2/averaged/top5-averaged.nemo
CONFIG=/code/examples/speechlm2/conf/streaming_stt_granary2_full_joint_chat_script_p84.yaml
VAL_MANIFEST=${HEH}/data/mcv11_en_dev_aligned/mcv11_dev_clean_pcstrip_en_2k_qwen_aligned.json

DATA_ROOT=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm
NEMO_USERS=/lustre/fsw/portfolios/nemotron/users
MOUNTS="--container-mounts=${CODE_DIR}:/code,${LUSTRE}:${LUSTRE},${DATA_ROOT}:${DATA_ROOT},${NEMO_USERS}:${NEMO_USERS}"

read -r -d '' cmd <<EOF
export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& export HF_HUB_OFFLINE=1 \
&& export HYDRA_FULL_ERROR=1 \
&& cd /code \
&& python /code/scripts/probe_donor_decode.py \
    --config ${CONFIG} \
    --donor ${DONOR} \
    --manifest ${VAL_MANIFEST} \
    --n 4 --chunk 14 \
    --overrides model.attn_backend=dense model.prompt_include_chunk_size=false \
                data.dataset.prompt_include_chunk_size=false \
                model.pretrained_llm=${HEH}/pretrained_models/huggingface/Qwen/Qwen3-1.7B \
                model.pretrained_asr=${HEH}/pretrained_models/huggingface/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo \
                model.chat.tokenizer.dir=${HEH}/pretrained_models/huggingface/Qwen/Qwen3-1.7B
EOF
srun --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
