#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-corrector-smoke
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH --time-min 00:30:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# SMOKE TEST for the verifier/corrector: does it build, batch and step at all?
#
#   sbatch launch/dfw_script_corrector_smoke.sh
#
# ONE node, a few hundred steps, its own EXP_NAME. Deliberately not the real run:
# everything here is new code touching a frozen CHAT, a shared encoder and a
# warm start that is only PARTIAL, and each of those fails in a different way.
# What this is looking for, in order:
#
#   1. the model builds and reports ~340M trainable, not 948M -- if it reports
#      948M the encoder was not actually shared and nothing else is meaningful;
#   2. batches carry cuts (word timings), without which no labels exist;
#   3. accept_frac lands near the measured 0.93, not 0.0 or 1.0 -- either
#      extreme means the labeller and the decoder disagree about the data;
#   4. the loss falls.
# ============================================================================
set -uo pipefail
mkdir -p slurm_out

CLUSTER=dfw
GPUS_PER_NODE=8
LUSTRE=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
HEH=${LUSTRE}/users/heh
MYDIR=${LUSTRE}/hainanx

CONTAINER="${CONTAINER:-${HEH}/containers/nemo-26.02-streaming-speechlm.sqsh}"
CODE_DIR="${CODE_DIR:-${MYDIR}/NeMo_SCRIPT_cc}"
PROJECT_NAME="${PROJECT_NAME:-SpeechlmDFW}"
CONFIG_PATH=/code/examples/speechlm2/conf
CONFIG_NAME="${CONFIG_NAME:-streaming_stt_granary2_lora_script_corrector}"
EXP_NAME="${EXP_NAME:-dfw_corrector_smoke}"

# The model being verified, and the SCRIPT checkpoint we warm-start from.
CHAT_ARM="${CHAT_ARM:-dfw_granary2_chat_banded1_nodelay_v2}"
SCRIPT_ARM="${SCRIPT_ARM:-dfw_granary2_script_banded1_nodelay_v2}"
CHAT_NEMO="${MYDIR}/results/${PROJECT_NAME}/${CHAT_ARM}/averaged/top5-averaged.nemo"
INIT_CKPT="${MYDIR}/results/${PROJECT_NAME}/${SCRIPT_ARM}/${SCRIPT_ARM}/checkpoints/${SCRIPT_ARM}-averaged.ckpt"

for f in "$CHAT_NEMO" "$INIT_CKPT"; do
    [[ -s "$f" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done

PRETRAINED_LLM=${HEH}/pretrained_models/huggingface/Qwen/Qwen3-1.7B
PRETRAINED_ASR=${HEH}/pretrained_models/huggingface/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo
TRAIN_INPUT_CFG=${HEH}/data_configs/granary_v2_en_full_d0.5_b0.5_dfw_qwen_aligned.yaml
# The config inherits SCRIPT's validation manifest, an OCI path that does not
# exist on DFW. Without this override the run dies at the FIRST validation --
# which for this launcher is step 200, so it never reaches the metric it exists
# to measure.
VAL_MANIFEST=${HEH}/data/mcv11_en_dev_aligned/mcv11_dev_clean_pcstrip_en_2k_qwen_aligned.json
[[ -s "$VAL_MANIFEST" ]] || { echo "ERROR: validation manifest not found: ${VAL_MANIFEST}" >&2; exit 1; }

RESULTS_DIR=${MYDIR}/results/${PROJECT_NAME}/${EXP_NAME}
HFCACHE=${MYDIR}/hf_cache
mkdir -p "$RESULTS_DIR" "$HFCACHE"

# Hydra cannot parse "=" inside a VALUE, and every checkpoint here is named
# step=NNNN-val_wer=N.NNNN. Symlink under a clean name.
INIT_LINK=${MYDIR}/init_ckpts/${EXP_NAME}_init.ckpt
mkdir -p "$(dirname "$INIT_LINK")"; ln -sfn "$INIT_CKPT" "$INIT_LINK"

read_token() { [[ -r "$1" ]] || { echo "ERROR: missing token $1" >&2; exit 1; }; tr -d '\r\n' < "$1"; }
WANDB="$(read_token "$HOME/.wandb_token")"
HF_TOKEN="$(read_token "$HOME/.hf_token")"

DATA_ROOT=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm
MOUNTS="--container-mounts=${CODE_DIR}:/code,${RESULTS_DIR}:/results,${HFCACHE}:/hfcache,${LUSTRE}:${LUSTRE},${DATA_ROOT}:${DATA_ROOT}"

read -r -d '' cmd <<EOCMD
echo "*** CORRECTOR SMOKE: verifying ${CHAT_ARM}, warm start ${SCRIPT_ARM} ***" \
&& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 \
&& export WANDB_API_KEY=${WANDB} HF_HOME=/hfcache/ HF_TOKEN=${HF_TOKEN} HF_HUB_OFFLINE=1 \
&& export HYDRA_FULL_ERROR=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
&& export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NCCL_TIMEOUT_SEC=3600 \
&& export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& cd /code && git rev-parse --short HEAD \
&& python /code/examples/speechlm2/script_corrector_train.py \
    --config-path=${CONFIG_PATH} --config-name=${CONFIG_NAME} \
    ++model.chat_nemo=${CHAT_NEMO} \
    model.pretrained_llm=${PRETRAINED_LLM} \
    model.pretrained_asr=${PRETRAINED_ASR} \
    ++init_from_ckpt=${INIT_LINK} \
    model.optimizer.lr=5e-5 \
    data.train_ds.input_cfg=${TRAIN_INPUT_CFG} \
    data.train_ds.num_workers=4 \
    data.validation_ds.datasets.mcv_11_dev.manifest_filepath=${VAL_MANIFEST} \
    ++trainer.limit_train_batches=200 \
    ++trainer.val_check_interval=200 \
    trainer.max_steps=200 \
    trainer.devices=${GPUS_PER_NODE} \
    trainer.num_nodes=\${SLURM_JOB_NUM_NODES} \
    trainer.log_every_n_steps=5 \
    ++exp_manager.exp_dir=/results/ \
    ++exp_manager.create_wandb_logger=true \
    ++exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    ++exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    ++exp_manager.create_tensorboard_logger=false \
    ++exp_manager.name=${EXP_NAME}
EOCMD

srun -o "${RESULTS_DIR}/slurm-%j-%n.out" -e "${RESULTS_DIR}/error-%j-%n.out" \
     --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
