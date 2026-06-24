#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:streaming-stt-imend-loss
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 8
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
# Streaming SpeechLM (streaming_stt) finetune on OCI (draco-oci-iad).
#
# Adapted from NeMo/speechllm_oci/heh_new_with_im_end_loss.sh, with ONE key
# difference: this launches against MY code instead of the code baked into the
# container. We mount the git-synced repo (CODE_DIR, kept current via
# sync_to_oci.sh) at /code, prepend it to PYTHONPATH so `import nemo` resolves
# to /code, and run /code/examples/speechlm2/streaming_stt_train.py directly.
# The container is used only for the Python/CUDA/deps environment.
# ============================================================================

# Secrets are read from the environment -- never hardcode them (GitHub push
# protection blocks commits containing tokens). Export these before sbatch, e.g.
# add to ~/.bashrc on the login node:
#   export WANDB_API_KEY=...
#   export HF_TOKEN=...
#   export AIS_AUTHN_TOKEN=...   # OCI aistore data-store token (if needed)
WANDB="${WANDB_API_KEY:?set WANDB_API_KEY in your environment}"
HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your environment}"
AIS_AUTHN_TOKEN="${AIS_AUTHN_TOKEN:-}"

set -x
mkdir -p slurm_out
CLUSTER="oci"
GPUS_PER_NODE=8
SLURM_ACCOUNT='llmservice'
OLDUSERID='users/heh'
USERID='users/hainanx'
LUSTRE_ACCOUNT_PREFIX=/lustre/fsw/portfolios/${SLURM_ACCOUNT}

# Latest container with the current SpeechLM deps (we only use its environment;
# the actual NeMo code comes from /code below).
CONTAINER="gitlab-master.nvidia.com/hainanx/nemo_containers:speechlm_heh"

PROJECT_NAME=Streaming_SLM_chunk14_refactor_2

# Training parameters
MAX_STEPS=200000
VAL_CHECK_INTERVAL=2000
DELAY=1
LOOKAHEAD=13
CHUNK_SIZE=14 # 1.12s chunk

FREEZE_SPEECH_ENCODER=false

LR=0.0004
WARMUP_STEPS=15000

MODEL_SUFFIX=_n${SLURM_JOB_NUM_NODES}_FrzAE${FREEZE_SPEECH_ENCODER}_delay${DELAY}_la${LOOKAHEAD}_chunk${CHUNK_SIZE}_r8_t1

# Config file (heh's shared recipe dir; mounted read-only via H_DIR below).
CONFIG_PATH=/lustre/fsw/portfolios/llmservice/users/heh/scripts/streaming_speechlm/recipes/
CONFIG_NAME=streaming_stt_nss_granary_lora

EXP_NAME=${CLUSTER}_${CONFIG_NAME}_lr${LR}_warmup${WARMUP_STEPS}${MODEL_SUFFIX}
EXP_NAME=baseline_imend_loss

# Directories for manifests, data, etc.
RESULTS_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/results/$PROJECT_NAME/$EXP_NAME
PRETRAINED_MODEL_DIR=${LUSTRE_ACCOUNT_PREFIX}/${OLDUSERID}/pretrained_models
CHECKPOINT_DIR=${LUSTRE_ACCOUNT_PREFIX}/${OLDUSERID}/checkpoints/
QUESTIONS_DIR=${LUSTRE_ACCOUNT_PREFIX}/${OLDUSERID}/questions
HFCACHE=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/hf_cache
DATA_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh
HAINAN_DIR=/lustre/fsw/portfolios/llmservice/users/hainanx
# MY code (synced via oci/sync.sh to a UNIQUE per-branch dir) -> mounted as
# /code. Keep this in sync with OCI_REPO in oci/sync.sh.
CODE_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/NeMo_ord_sync_d146_current

# Make results dir
mkdir -p ${RESULTS_DIR}
OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

MOUNTS="--container-mounts=${DATA_DIR}:${DATA_DIR},${H_DIR}:${H_DIR},$HAINAN_DIR:$HAINAN_DIR,$CODE_DIR:/code,$RESULTS_DIR:/results,$DATA_DIR:/data,$PRETRAINED_MODEL_DIR:/pretrained,$CHECKPOINT_DIR:/checkpoints,${QUESTIONS_DIR}:/questions/,${HFCACHE}:/hfcache/"

# SLURM_JOB_NUM_NODES=1
# GPUS_PER_NODE=1
# export CUDA_VISIBLE_DEVICES=0

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& nvidia-smi \
&& export WANDB_API_KEY=${WANDB} \
&& cd /code \
&& git rev-parse HEAD \
&& export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& python -c "import nemo, nemo.collections.speechlm2; print('USING NeMo FROM:', nemo.__file__)" \
&& pip show torch \
&& export OMP_NUM_THREADS=1 \
&& export HF_HOME="/hfcache/" \
&& export HF_TOKEN=${HF_TOKEN} \
&& export HYDRA_FULL_ERROR=1 \
&& export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
&& export AIS_ENDPOINT=http://asr.iad.oci.aistore.nvidia.com:51080 \
&& export AIS_AUTHN_TOKEN="${AIS_AUTHN_TOKEN}" \
&& export NEMO_DATA_STORE_CACHE_DIR=/lustre/fsw/portfolios/llmservice/users/heh/nemo_cache \
&& echo "Starting training (running MY code at /code)" \
&& python /code/examples/speechlm2/streaming_stt_train.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    model.freeze_speech_encoder=${FREEZE_SPEECH_ENCODER} \
    model.att_context_size=[70,${LOOKAHEAD}] \
    model.chunk_size=${CHUNK_SIZE} \
    ++model.supervise_im_end_in_loss=true \
    model.optimizer.lr=$LR \
    model.lr_scheduler.warmup_steps=$WARMUP_STEPS \
    data.dataset.num_delay_frames=$DELAY \
    ++trainer.limit_train_batches=$VAL_CHECK_INTERVAL \
    ++trainer.val_check_interval=$VAL_CHECK_INTERVAL \
    trainer.max_steps=$MAX_STEPS \
    trainer.devices=$GPUS_PER_NODE \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES  \
    trainer.log_every_n_steps=10 \
    ++exp_manager.exp_dir=/results/ \
    ++exp_manager.create_wandb_logger=true \
    ++exp_manager.create_tensorboard_logger=false \
    ++exp_manager.max_time_per_run=00:03:55:00 \
    ++exp_manager.name=$EXP_NAME \
    ++exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    ++trainer.strategy.find_unused_parameters=false \
    ++model.perception.encoder.sync_max_audio_length=false \


EOF

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"

# bash -c "${cmd}"

set +x
