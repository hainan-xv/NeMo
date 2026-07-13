#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:chat-ce-nemotron06b-g2
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
# OCI launcher for the alignment-guided cross-entropy (CE) CHAT model.
#
# Same streaming CHAT model as oci_chat/chat.sh, but trained with an
# ALIGNMENT-GUIDED CROSS-ENTROPY loss instead of the alignment-free RNNT loss:
# each target token is assigned to the encoder chunk containing its word's ENDING
# timestamp (Granary v2 alignments; all subword tokens of a word share the word's
# ending chunk, + optional NUM_DELAY_FRAMES). The joint is evaluated only along the
# single forced path (token + blank steps), avoiding the [B,T,U,V] tensor and the
# RNNT forward-backward. Inference is unchanged (standard CHAT greedy decode).
#
# Uses the new entrypoint speech_to_text_chat_ce_bpe.py (EncDecRNNTBPEModelChatCE)
# with the shared CHAT config; training data flows through the alignment-carrying
# dataset (model.train_ds.use_chat_ce_dataset=true), validation is unchanged.
#
# Data: Granary v2.0 train + English mcv11 dev (same as the SpeechLM baseline).
# Encoder is warm-started from nemotron-speech-streaming-en-0.6b by default.
#
# Submit from an OCI login node:
#   ./sync_to_oci.sh
#   sbatch oci_chat/chat_ce.sh
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

# Latest container with the current SpeechLM/ASR deps (we only use its
# environment; the actual NeMo code comes from /code below).
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"

# Dedicated project for the CHAT models (separate wandb project + results space
# from the SpeechLM models in Speechlm79).
PROJECT_NAME="${PROJECT_NAME:-Chat79}"

# ---------------------------------------------------------------------------
# CHAT (alignment CE) model + training hyper-parameters
# ---------------------------------------------------------------------------
CONFIG_PATH="${CONFIG_PATH:-/code/examples/asr/conf/fastconformer/cache_aware_streaming}"
CONFIG_NAME="${CONFIG_NAME:-fastconformer_chat_transducer_bpe_streaming}"

# Encoder attention context [left, right]; streaming CHAT default (matches nemotron).
ATT_CONTEXT="${ATT_CONTEXT:-[70,13]}"
CTX_TAG="$(echo "${ATT_CONTEXT}" | tr -d '[] ' | tr ',' '_')"

# Fixed CHAT chunk size (encoder frames). Set explicitly so the alignment->chunk
# mapping (data) and the loss agree deterministically.
CHAT_CHUNK_SIZE="${CHAT_CHUNK_SIZE:-14}"
# Encoder frames to delay word emission after its end (word -> chunk mapping).
NUM_DELAY_FRAMES="${NUM_DELAY_FRAMES:-0}"

MAX_STEPS="${MAX_STEPS:-500000}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-6000}"
EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-1}"
VAL_CHECK_INTERVAL=$(( LIMIT_TRAIN_BATCHES / EVALS_PER_EPOCH ))
LR="${LR:-5.0}"                       # NoamAnnealing peak scale (config default)
WARMUP_STEPS="${WARMUP_STEPS:-15000}"
BATCH_DURATION="${BATCH_DURATION:-120}"
NUM_BUCKETS="${NUM_BUCKETS:-30}"
MAX_DURATION="${MAX_DURATION:-20.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SAVE_TOP_K="${SAVE_TOP_K:-5}"
PRECISION="${PRECISION:-bf16}"
# Alignment-CE loss reduction: mean_volume | mean_batch | sum | mean.
RNNT_REDUCTION="${RNNT_REDUCTION:-mean_volume}"

# Encoder warm-start from the nemotron streaming FastConformer encoder weights.
# INIT_ENCODER_MODE: local (default, .nemo on lustre) | hf (download) | scratch.
INIT_ENCODER_MODE="${INIT_ENCODER_MODE:-local}"
PRETRAINED_MODEL_DIR="${PRETRAINED_MODEL_DIR:-${LUSTRE_ACCOUNT_PREFIX}/${USERID}/pretrained_models}"
INIT_ENCODER_BASENAME="${INIT_ENCODER_BASENAME:-nemotron-speech-streaming-en-0.6b.nemo}"
INIT_ENCODER_HOST="${PRETRAINED_MODEL_DIR}/${INIT_ENCODER_BASENAME}"
INIT_ENCODER_CONTAINER="/pretrained/${INIT_ENCODER_BASENAME}"
INIT_ENCODER_NAME="${INIT_ENCODER_NAME:-nvidia/nemotron-speech-streaming-en-0.6b}"
case "${INIT_ENCODER_MODE}" in
  scratch|none|off)
    echo "INIT_ENCODER_MODE=scratch: training encoder from scratch (no checkpoint)."
    INIT_ENCODER_OVERRIDE=""
    INIT_ENCODER_DESC="scratch"
    ;;
  local|nemo)
    if [ ! -f "${INIT_ENCODER_HOST}" ]; then
      echo "ERROR: INIT_ENCODER_MODE=local but ${INIT_ENCODER_HOST} not found." >&2; exit 1
    fi
    INIT_ENCODER_OVERRIDE="+init_from_nemo_model.streaming_enc.path=${INIT_ENCODER_CONTAINER} +init_from_nemo_model.streaming_enc.include=[encoder]"
    INIT_ENCODER_DESC="${INIT_ENCODER_CONTAINER}"
    ;;
  hf|pretrained)
    INIT_ENCODER_OVERRIDE="+init_from_pretrained_model.streaming_enc.name=${INIT_ENCODER_NAME} +init_from_pretrained_model.streaming_enc.include=[encoder]"
    INIT_ENCODER_DESC="${INIT_ENCODER_NAME}"
    ;;
  *) echo "ERROR: unknown INIT_ENCODER_MODE='${INIT_ENCODER_MODE}'" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Data (Granary 2.0 train + English mcv11 dev, same as oci/baseline_granary2.sh)
# ---------------------------------------------------------------------------
GRANARY2_CFG=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/dongjig/aligned_amos/granary_v2_en_pnc_qwen_aligned_filtered/granary_v2_en_pnc_qwen_aligned_filtered_safe_iad_s3_audio.yaml
TRAIN_INPUT_CFG="${TRAIN_INPUT_CFG:-$GRANARY2_CFG}"

# English dev set: same as the SpeechLM baseline (streaming_stt_nss_granary_lora
# recipe's validation_ds -> mcv_11_dev). Matches the English Granary training.
# Path is under the /data mount (= $DATA_DIR), same as the recipe.
VAL_MANIFEST="${VAL_MANIFEST:-[/data/canary/canary_v0/manifests/data/ASR/MMLPC/en/val_test/mcv11/mcv11_dev_clean_pcstrip_en_2k.json]}"

# English 1024 BPE SentencePiece tokenizer (same vocab used by the other CHAT
# ASR launchers). Point TOKENIZER_DIR at the directory containing tokenizer.model.
ST_TOKENIZERS_ROOT="${ST_TOKENIZERS_ROOT:-${LUSTRE_ACCOUNT_PREFIX}/${USERID}/Workplace/multilingual/tokenizers/en}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${ST_TOKENIZERS_ROOT}/tokenizer_spe_bpe_v1024}"

EXP_NAME="${EXP_NAME:-${CLUSTER}_chat_ce_nemotron06b_g2_ctx${CTX_TAG}_chunk${CHAT_CHUNK_SIZE}_delay${NUM_DELAY_FRAMES}_lr${LR}_n${SLURM_JOB_NUM_NODES}}"

# Write-heavy outputs (results/checkpoints, HF cache, checkpoint temp) go to the
# nemotron project, which has free quota. Override with OUTPUT_PREFIX.
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
RESULTS_DIR=${OUTPUT_PREFIX}/results/$PROJECT_NAME/$EXP_NAME
CHECKPOINT_DIR=${LUSTRE_ACCOUNT_PREFIX}/${OLDUSERID}/checkpoints/
HFCACHE=${OUTPUT_PREFIX}/hf_cache
SPEECHLM_PROJECT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm
DATA_DIR=${SPEECHLM_PROJECT_DIR}/data
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh
HAINAN_DIR=/lustre/fsw/portfolios/llmservice/users/hainanx
# MY code (synced via sync_to_oci.sh) -> mounted as /code.
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo79/}"

# Stage checkpoint/restore temp files on the lustre results filesystem (same
# device as the checkpoint destination), not the container's small /tmp.
OCI_TMP_DIR="${OCI_TMP_DIR:-/results/tmp}"

# Make results + HF cache dirs (on the nemotron output filesystem)
mkdir -p ${RESULTS_DIR} ${HFCACHE}
OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

MOUNTS="--container-mounts=${SPEECHLM_PROJECT_DIR}:${SPEECHLM_PROJECT_DIR},${H_DIR}:${H_DIR},$HAINAN_DIR:$HAINAN_DIR,$CODE_DIR:/code,$RESULTS_DIR:/results,$DATA_DIR:/data,$PRETRAINED_MODEL_DIR:/pretrained,$CHECKPOINT_DIR:/checkpoints,${HFCACHE}:/hfcache/,${TOKENIZER_DIR}:/tokenizers,/lustre/fsw:/lustre/fsw,/lustre/fs12:/lustre/fs12"

# SLURM_JOB_NUM_NODES=1
# GPUS_PER_NODE=1
# export CUDA_VISIBLE_DEVICES=0

read -r -d '' cmd <<EOF
echo "*******STARTING CHAT ALIGNMENT-CE (FastConformer) - Granary 2.0********" \
&& echo "*** CONFIG: ${CONFIG_NAME} | ATT_CONTEXT=${ATT_CONTEXT} | chunk_size=${CHAT_CHUNK_SIZE} delay=${NUM_DELAY_FRAMES} ***" \
&& echo "*** LOSS: alignment-guided CE (forced path), reduction=${RNNT_REDUCTION} ***" \
&& echo "*** DATA: Granary 2.0 pre-aligned (lhotse) -> ${TRAIN_INPUT_CFG} ***" \
&& echo "*** ENCODER init: ${INIT_ENCODER_DESC} ***" \
&& nvidia-smi \
&& export WANDB_API_KEY=${WANDB} \
&& cd /code \
&& git rev-parse HEAD \
&& export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& python -c "import nemo, nemo.collections.asr; print('USING NeMo FROM:', nemo.__file__)" \
&& pip show torch \
&& export OMP_NUM_THREADS=1 \
&& export HF_HOME="/hfcache/" \
&& export HF_TOKEN=${HF_TOKEN} \
&& export HYDRA_FULL_ERROR=1 \
&& export TOKENIZERS_PARALLELISM=false \
&& export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
&& export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.3 \
&& export TMPDIR=${OCI_TMP_DIR} && mkdir -p ${OCI_TMP_DIR} && echo "staging TMPDIR=\$TMPDIR" \
&& export AIS_ENDPOINT=http://asr.iad.oci.aistore.nvidia.com:51080 \
&& export AIS_AUTHN_TOKEN="${AIS_AUTHN_TOKEN}" \
&& export NEMO_DATA_STORE_CACHE_DIR=/lustre/fsw/portfolios/llmservice/users/heh/nemo_cache \
&& echo "Starting training (running MY code at /code, CHAT alignment-CE model, GRANARY 2.0 data)" \
&& python /code/examples/asr/asr_transducer/speech_to_text_chat_ce_bpe.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    name=${EXP_NAME} \
    ${INIT_ENCODER_OVERRIDE} \
    model.encoder.att_context_size=${ATT_CONTEXT} \
    ++model.joint.chunk_size=${CHAT_CHUNK_SIZE} \
    model.skip_nan_grad=true \
    model.compute_eval_loss=false \
    ++model.rnnt_reduction=${RNNT_REDUCTION} \
    ++model.chat_ce.reduction=${RNNT_REDUCTION} \
    ++model.chat_ce.num_delay_frames=${NUM_DELAY_FRAMES} \
    ++model.train_ds.use_lhotse=true \
    ++model.train_ds.use_chat_ce_dataset=true \
    ++model.train_ds.input_cfg=${TRAIN_INPUT_CFG} \
    model.train_ds.manifest_filepath=null \
    ++model.train_ds.return_cuts=true \
    ++model.train_ds.skip_missing_manifest_entries=true \
    ++model.train_ds.batch_duration=${BATCH_DURATION} \
    ++model.train_ds.use_bucketing=true \
    ++model.train_ds.num_buckets=${NUM_BUCKETS} \
    ++model.train_ds.use_start_end_token=false \
    model.train_ds.max_duration=${MAX_DURATION} \
    model.train_ds.num_workers=4 \
    model.train_ds.shuffle=true \
    model.train_ds.pin_memory=true \
    model.validation_ds.manifest_filepath=${VAL_MANIFEST} \
    model.validation_ds.batch_size=${EVAL_BATCH_SIZE} \
    model.validation_ds.num_workers=4 \
    model.validation_ds.pin_memory=true \
    ++model.validation_ds.use_start_end_token=false \
    model.tokenizer.dir="/tokenizers" \
    model.tokenizer.type=bpe \
    model.optim.lr=${LR} \
    model.optim.sched.warmup_steps=${WARMUP_STEPS} \
    ++trainer.use_distributed_sampler=false \
    ++trainer.limit_train_batches=${LIMIT_TRAIN_BATCHES} \
    ++trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.devices=${GPUS_PER_NODE} \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    trainer.precision=${PRECISION} \
    trainer.sync_batchnorm=false \
    trainer.log_every_n_steps=20 \
    exp_manager.exp_dir=/results/${EXP_NAME} \
    exp_manager.name=${EXP_NAME} \
    exp_manager.create_wandb_logger=true \
    exp_manager.create_tensorboard_logger=false \
    exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    exp_manager.resume_if_exists=true \
    exp_manager.resume_ignore_no_checkpoint=true \
    exp_manager.checkpoint_callback_params.monitor=val_wer \
    exp_manager.checkpoint_callback_params.mode=min \
    exp_manager.checkpoint_callback_params.save_top_k=${SAVE_TOP_K} \
    ++exp_manager.checkpoint_callback_params.filename="'{step}-{val_wer:.4f}'" \
    ++exp_manager.max_time_per_run=00:03:55:00

EOF

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"

# bash -c "${cmd}"

set +x
