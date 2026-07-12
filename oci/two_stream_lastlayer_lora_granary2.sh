#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:streaming-stt-2sll-g2
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
# Streaming SpeechLM (streaming_stt) finetune on OCI (draco-oci, IAD).
#
# TWO-STREAM + LAST-LAYER-ONLY LoRA + FULL encoder training, on GRANARY 2.0 data.
# Based on oci/baseline_granary2.sh, plus:
#   ++model.two_stream_last_layer=true    (text runs layers[:-1] alone; audio is
#                                          fused only in the FINAL layer)
#   ++model.lora.last_layer_only=true     (LoRA adapters ONLY on the final
#                                          decoder layer -- where cross-modal
#                                          fusion happens)
# The speech encoder trains FULLY (freeze_speech_encoder=false). LoRA
# target_modules are widened for the single adapted layer.
#
# Runs MY code (git-synced repo mounted at /code), not the container's NeMo.
#
# Submit from an OCI login node (draco-oci-login-01.draco-oci-iad.nvidia.com):
#   cd /lustre/fsw/portfolios/llmservice/users/hainanx/NeMo79
#   sbatch oci/two_stream_lastlayer_lora_granary2.sh
# ============================================================================

# Secrets live only in token files on the OCI login node. Each file contains
# just the token value on one line and must be readable only by the owner:
#   chmod 600 ~/.wandb_token ~/.hf_token ~/.ais_authn_token
read_required_token() {
    local path="$1"
    if [[ ! -r "$path" ]]; then
        echo "ERROR: required token file is missing or unreadable: $path" >&2
        exit 1
    fi
    tr -d '\r\n' < "$path"
}

WANDB="$(read_required_token "$HOME/.wandb_token")"
HF_TOKEN="$(read_required_token "$HOME/.hf_token")"
AIS_AUTHN_TOKEN=""
if [[ -r "$HOME/.ais_authn_token" ]]; then
    AIS_AUTHN_TOKEN="$(tr -d '\r\n' < "$HOME/.ais_authn_token")"
fi

# Do not enable xtrace: the command below contains expanded token values.
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
CONTAINER="/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh"


PROJECT_NAME=Speechlm79

# Training parameters
MAX_STEPS=500000
VAL_CHECK_INTERVAL=2000
DELAY=1
LOOKAHEAD=13
CHUNK_SIZE=14 # 1.12s chunk

FREEZE_SPEECH_ENCODER=false

LR=0.0004
WARMUP_STEPS=15000

# LoRA target modules for the (single) final layer. Widened vs the default
# q/v-only set so the one adapted layer has enough capacity for cross-modal
# fusion. Override with LORA_TARGET_MODULES.
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}"

# Set DEBUG_CUDA=1 to run with CUDA_LAUNCH_BLOCKING=1 so a device-side assert
# reports the true failing op/line (default 0 = async, fast).
DEBUG_CUDA="${DEBUG_CUDA:-0}"
# Set DEBUG_VALIDATE_TOKENS=true to check input/target token ids are in range
# each step (turns an opaque CUDA assert into a clear Python error). Debug only.
DEBUG_VALIDATE_TOKENS="${DEBUG_VALIDATE_TOKENS:-false}"

# Granary 2.0 pre-aligned data (alignments embedded in the cuts, online aligner
# removed). This is the DEFAULT for this launcher. Set TRAIN_INPUT_CFG="" to
# fall back to the recipe's online-alignment data instead.
GRANARY2_CFG=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/dongjig/aligned_amos/granary_v2_en_pnc_qwen_aligned_filtered/granary_v2_en_pnc_qwen_aligned_filtered_safe_iad_s3_audio.yaml
TRAIN_INPUT_CFG="${TRAIN_INPUT_CFG-$GRANARY2_CFG}"
if [[ -n "$TRAIN_INPUT_CFG" ]]; then
  PREALIGN_OVERRIDES=("~forced_aligner" "data.train_ds.input_cfg=$TRAIN_INPUT_CFG")
else
  PREALIGN_OVERRIDES=()
fi

MODEL_SUFFIX=_n${SLURM_JOB_NUM_NODES}_FrzAE${FREEZE_SPEECH_ENCODER}_delay${DELAY}_la${LOOKAHEAD}_chunk${CHUNK_SIZE}_r8_t1

# Config file (heh's shared recipe dir; mounted read-only via H_DIR below).
CONFIG_PATH=/lustre/fsw/portfolios/llmservice/users/heh/scripts/streaming_speechlm/recipes/
CONFIG_NAME=streaming_stt_nss_granary_lora

EXP_NAME=${CLUSTER}_${CONFIG_NAME}_lr${LR}_warmup${WARMUP_STEPS}${MODEL_SUFFIX}
EXP_NAME=two_stream_lastlayer_lora_granary2

# Directories for manifests, data, etc.
# Write-heavy outputs (results/checkpoints, HF cache, checkpoint temp) go to the
# nemotron project, which has free quota. The llmservice project that holds the
# synced code is at its space limit, so writes there fail with EDQUOT. Override
# the output location with OUTPUT_PREFIX.
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
RESULTS_DIR=${OUTPUT_PREFIX}/results/$PROJECT_NAME/$EXP_NAME
PRETRAINED_MODEL_DIR=${LUSTRE_ACCOUNT_PREFIX}/${OLDUSERID}/pretrained_models
CHECKPOINT_DIR=${LUSTRE_ACCOUNT_PREFIX}/${OLDUSERID}/checkpoints/
QUESTIONS_DIR=${LUSTRE_ACCOUNT_PREFIX}/${OLDUSERID}/questions
HFCACHE=${OUTPUT_PREFIX}/hf_cache
DATA_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh
HAINAN_DIR=/lustre/fsw/portfolios/llmservice/users/hainanx
# MY code (synced via sync_to_oci.sh) -> mounted as /code.
CODE_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/NeMo79
CODE_DIR=/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo79/

# Stage checkpoint/restore temp files on the lustre results filesystem (same
# device as the checkpoint destination), not the container's small /tmp.
OCI_TMP_DIR="${OCI_TMP_DIR:-/results/tmp}"

# Make results + HF cache dirs (on the nemotron output filesystem)
mkdir -p ${RESULTS_DIR} ${HFCACHE}
OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

DONGJI_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/dongjig/aligned_amos/granary_v2_en_pnc_qwen_aligned_filtered

MOUNTS="--container-mounts=${DATA_DIR}:${DATA_DIR},${H_DIR}:${H_DIR},$HAINAN_DIR:$HAINAN_DIR,$CODE_DIR:/code,$RESULTS_DIR:/results,$DATA_DIR:/data,$PRETRAINED_MODEL_DIR:/pretrained,$CHECKPOINT_DIR:/checkpoints,${QUESTIONS_DIR}:/questions/,${HFCACHE}:/hfcache/,$DONGJI_DIR:$DONGJI_DIR"

# SLURM_JOB_NUM_NODES=1
# GPUS_PER_NODE=1
# export CUDA_VISIBLE_DEVICES=0

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& echo "*** DATA: Granary 2.0 pre-aligned -> data.train_ds.input_cfg=$TRAIN_INPUT_CFG ***" \
&& echo "*** EXPERIMENT: two_stream + last-layer-only LoRA + FULL encoder ***" \
&& echo "***   ++model.two_stream_last_layer=true ***" \
&& echo "***   ++model.lora.last_layer_only=true  target_modules=${LORA_TARGET_MODULES} ***" \
&& echo "***   (encoder trains fully; LoRA only on the final fusion layer) ***" \
&& nvidia-smi \
&& export WANDB_API_KEY=${WANDB} \
&& cd /code \
&& git rev-parse HEAD \
&& export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& python -c "import nemo, nemo.collections.speechlm2; print('USING NeMo FROM:', nemo.__file__)" \
&& python -c "import inspect, nemo.collections.speechlm2.models.streaming_stt_model as m; print('two-stream supported:', 'two_stream_llm_forward' in dir(m))" \
&& python -c "import inspect, nemo.collections.speechlm2.parts.lora as l; print('last-layer LoRA supported:', 'last_layer_only' in inspect.getsource(l.maybe_install_lora))" \
&& pip show torch \
&& export OMP_NUM_THREADS=1 \
&& export HF_HOME="/hfcache/" \
&& export HF_TOKEN=${HF_TOKEN} \
&& export HYDRA_FULL_ERROR=1 \
&& export CUDA_LAUNCH_BLOCKING=${DEBUG_CUDA} \
&& export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
&& export TMPDIR=${OCI_TMP_DIR} && mkdir -p ${OCI_TMP_DIR} && echo "staging TMPDIR=\$TMPDIR" \
&& export AIS_ENDPOINT=http://asr.iad.oci.aistore.nvidia.com:51080 \
&& export AIS_AUTHN_TOKEN="${AIS_AUTHN_TOKEN}" \
&& export NEMO_DATA_STORE_CACHE_DIR=/lustre/fsw/portfolios/llmservice/users/heh/nemo_cache \
&& echo "Starting training (running MY code at /code, GRANARY 2.0, TWO-STREAM + last-layer LoRA)" \
&& python /code/examples/speechlm2/streaming_stt_train.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    model.freeze_speech_encoder=${FREEZE_SPEECH_ENCODER} \
    model.att_context_size=[70,${LOOKAHEAD}] \
    model.chunk_size=${CHUNK_SIZE} \
    ++model.debug_validate_tokens=$DEBUG_VALIDATE_TOKENS \
    ++model.two_stream_last_layer=true \
    ++model.lora.last_layer_only=true \
    ++model.lora.target_modules=${LORA_TARGET_MODULES} \
    model.optimizer.lr=$LR \
    model.lr_scheduler.warmup_steps=$WARMUP_STEPS \
    data.dataset.num_delay_frames=$DELAY \
    ${PREALIGN_OVERRIDES[@]} \
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
    ++exp_manager.checkpoint_callback_params.monitor=val_wer \
    ++exp_manager.checkpoint_callback_params.mode=min \
    ++trainer.strategy.find_unused_parameters=false \
    ++model.perception.encoder.sync_max_audio_length=false \
    ++model.compact_template=true \
    ++data.dataset.compact_template=true \
    ++model.debug_validate_tokens=true \


EOF

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"

# bash -c "${cmd}"

set +x
