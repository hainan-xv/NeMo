#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:chat-fullctx-parakeet-rnnt-g2
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
# OCI launcher for the FULL-CONTEXT CHAT (Chunk-wise Attention Transducer) model,
# trained with the PURE RNNT (CHAT) loss -- no CTC.
#
# Default -N 4 (32 GPUs): same as oci_chat/tdt.sh; fewer nodes reduces AIS/lustre
# contention + idle-GPU risk. Override with: sbatch -N 8 oci_chat/chat_fullctx.sh
#
# Full-context (offline) variant of oci_chat/chat.sh. The encoder runs with
# unlimited attention and non-causal operations; the CHAT cross-attention joint
# still processes the encoder output in fixed-size chunks:
#   - att_context_size    = [-1,-1]   (unlimited / full context, regular style)
#   - causal_downsampling = false     (non-causal subsampling)
#   - conv_context_size   = null      (non-causal symmetric convolution)
#   - conv_norm_type      = batch_norm; preprocessor.normalize = per_feature
#   - joint.chunk_size    = ${CHAT_CHUNK_SIZE} (default 14; set explicitly since
#                           it can no longer be inferred from a chunked encoder)
#   - loss = pure CHAT RNNT (mean_volume), no CTC
#
# Training hyper-parameters match oci_chat/tdt.sh (LR, warmup, precision, lhotse
# bucketing, num_workers, AIS batch fetch, Granary 2.0 data, parakeet tokenizer).
#
# Warm-start: by default loads the FULL parakeet-tdt-0.6b-v2 .nemo (encoder +
# decoder + joint where shapes match; strict=False). Override with
# INIT_FROM=ctc_joint to resume from a prior CHAT CTC-joint PTL checkpoint instead.
#
# Model config:
#   examples/asr/conf/fastconformer/cache_aware_streaming/
#       fastconformer_chat_transducer_bpe_streaming.yaml
#
# Runs MY code (git-synced repo mounted at /code), not the container's NeMo.
#
# Submit from an OCI login node:
#   ./sync_to_oci.sh
#   sbatch oci_chat/chat_fullctx.sh
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

# Dedicated project for the CHAT models (separate wandb project + results space).
PROJECT_NAME="${PROJECT_NAME:-Chat79}"

# ---------------------------------------------------------------------------
# CHAT full-context model + training hyper-parameters (aligned with tdt.sh)
# ---------------------------------------------------------------------------
CONFIG_PATH="${CONFIG_PATH:-/code/examples/asr/conf/fastconformer/cache_aware_streaming}"
CONFIG_NAME="${CONFIG_NAME:-fastconformer_chat_transducer_bpe_streaming}"

# Full-context encoder knobs. The encoder is configured to match
# parakeet-tdt-0.6b-v2 (non-causal FastConformer, batch_norm conv, per-feature
# mel normalization, unlimited attention) so its encoder loads cleanly.
ATT_CONTEXT="${ATT_CONTEXT:-[-1,-1]}"              # [-1,-1] = unlimited (full) context
ATT_CONTEXT_STYLE="${ATT_CONTEXT_STYLE:-regular}"  # regular attention for full context
CAUSAL_DOWNSAMPLING="${CAUSAL_DOWNSAMPLING:-false}" # non-causal subsampling
CONV_CONTEXT_SIZE="${CONV_CONTEXT_SIZE:-null}"      # null -> symmetric (non-causal) convolution
CONV_NORM_TYPE="${CONV_NORM_TYPE:-batch_norm}"      # parakeet-tdt uses batch_norm
NORMALIZE="${NORMALIZE:-per_feature}"               # parakeet-tdt uses per-feature mel normalization
CTX_TAG="$(echo "${ATT_CONTEXT}" | tr -d '[] ' | tr ',' '_')"

# CHAT joint chunk size. In full-context mode the encoder is no longer chunked,
# so this can't be inferred from att_context_size and must be set explicitly.
CHAT_CHUNK_SIZE="${CHAT_CHUNK_SIZE:-14}"

# Prediction network RNN layers. Parakeet-tdt-0.6b-v2's decoder has 2 LSTM layers
# (config default here is 1). Match it to 2 so the whole decoder loads from the
# parakeet .nemo (with strict=False a 1-layer decoder would silently drop layer 1).
PRED_RNN_LAYERS="${PRED_RNN_LAYERS:-2}"

MAX_STEPS="${MAX_STEPS:-500000}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-6000}"
EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-1}"
VAL_CHECK_INTERVAL=$(( LIMIT_TRAIN_BATCHES / EVALS_PER_EPOCH ))
# Continuing from a strong checkpoint -> modest peak LR + warmup (same as tdt.sh).
# NOTE: the CHAT streaming config uses NoamAnnealing (where `lr` is only a
# multiplier: eff_lr = lr * d_model^-0.5 * min(step^-0.5, step*warmup^-1.5),
# so lr=1e-4 -> peak ~4e-8). We override the scheduler to CosineAnnealing below
# (matching tdt.sh) so LR here is the *actual* peak LR.
LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
BATCH_DURATION="${BATCH_DURATION:-120}"
NUM_BUCKETS="${NUM_BUCKETS:-30}"
MAX_DURATION="${MAX_DURATION:-20.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SAVE_TOP_K="${SAVE_TOP_K:-5}"
PRECISION="${PRECISION:-bf16-mixed}"
# DataLoader workers per rank. Granary audio is served over AIS; too few workers
# leaves GPUs waiting and triggers cluster "idle GPU" alerts even while W&B
# still shows loss moving (steps are just slow / bursty).
NUM_WORKERS="${NUM_WORKERS:-8}"
# Batch AIS audio fetch (Lhotse>=1.32). Big win vs per-cut HTTP GETs.
USE_AIS_GET_BATCH="${USE_AIS_GET_BATCH:-true}"
# RNNT (CHAT) loss reduction: mean_batch | mean | sum | mean_volume.
RNNT_REDUCTION="${RNNT_REDUCTION:-mean_volume}"
# IMPORTANT: parakeet-matched encoder uses batch_norm. Lightning `bf16`
# (bf16-true) casts BN to bf16 and commonly produces Inf loss at step 0.
# Use bf16-mixed (or 32) instead. Override with PRECISION=32 if needed.

# ---------------------------------------------------------------------------
# parakeet-tdt-0.6b-v2 checkpoint + tokenizer (same paths as tdt.sh)
# ---------------------------------------------------------------------------
PRETRAINED_MODEL_DIR="${PRETRAINED_MODEL_DIR:-/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/hainanx/pretrained_models}"
PARAKEET_BASENAME="${PARAKEET_BASENAME:-parakeet-tdt-0.6b-v2.nemo}"
PARAKEET_NEMO_HOST="${PRETRAINED_MODEL_DIR}/${PARAKEET_BASENAME}"
PARAKEET_NEMO_CONTAINER="/pretrained/${PARAKEET_BASENAME}"
if [ ! -f "${PARAKEET_NEMO_HOST}" ]; then
    echo "ERROR: parakeet checkpoint not found at ${PARAKEET_NEMO_HOST}" >&2
    echo "       Copy parakeet-tdt-0.6b-v2.nemo there or set PARAKEET_NEMO_HOST/PRETRAINED_MODEL_DIR." >&2
    exit 1
fi

# Tokenizer: EXACT tokenizer packaged inside parakeet-tdt-0.6b-v2.nemo.
TOKENIZER_DIR="${TOKENIZER_DIR:-${PRETRAINED_MODEL_DIR}/${PARAKEET_BASENAME%.nemo}_tokenizer}"
if [ ! -f "${TOKENIZER_DIR}/tokenizer.model" ]; then
    echo "==> Extracting parakeet tokenizer into ${TOKENIZER_DIR}"
    mkdir -p "${TOKENIZER_DIR}"
    _tok_tmp="$(mktemp -d)"
    tar --no-same-owner -xf "${PARAKEET_NEMO_HOST}" -C "${_tok_tmp}"
    cp "${_tok_tmp}"/*tokenizer.model "${TOKENIZER_DIR}/tokenizer.model"
    cp "${_tok_tmp}"/*vocab.txt        "${TOKENIZER_DIR}/vocab.txt"        2>/dev/null || true
    cp "${_tok_tmp}"/*tokenizer.vocab  "${TOKENIZER_DIR}/tokenizer.vocab"  2>/dev/null || true
    rm -rf "${_tok_tmp}"
fi
if [ ! -f "${TOKENIZER_DIR}/tokenizer.model" ]; then
    echo "ERROR: failed to extract tokenizer.model into ${TOKENIZER_DIR}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Warm-start: parakeet .nemo (default) or prior CHAT CTC-joint PTL checkpoint
# ---------------------------------------------------------------------------
INIT_FROM="${INIT_FROM:-parakeet}"
SRC_CTC_WEIGHT="${SRC_CTC_WEIGHT:-0.3}"

# Write-heavy outputs (results/checkpoints, HF cache, checkpoint temp) go to the
# nemotron project, which has free quota. Override with OUTPUT_PREFIX.
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"

EXP_NAME="${EXP_NAME:-${CLUSTER}_chat_fullctx_parakeet_rnnt_g2_ctx${CTX_TAG}_chunk${CHAT_CHUNK_SIZE}_lr${LR}_${PRECISION}_n${SLURM_JOB_NUM_NODES}}"
RESULTS_DIR=${OUTPUT_PREFIX}/results/$PROJECT_NAME/$EXP_NAME
mkdir -p "${RESULTS_DIR}"

case "${INIT_FROM}" in
  parakeet|nemo|pretrained)
    # Load ONLY encoder + decoder from parakeet. The CHAT RNNT joint has 1025
    # outputs (1024 vocab + blank) while parakeet's TDT joint has 1030
    # (+5 durations), so the joint cannot be loaded -- it is trained from scratch.
    # The dict form of init_from_nemo_model + include filter does a partial load
    # (load_part_of_state_dict) instead of a strict full load.
    INIT_OVERRIDE="+init_from_nemo_model.parakeet.path=${PARAKEET_NEMO_CONTAINER} +init_from_nemo_model.parakeet.include=[encoder,decoder]"
    INIT_DESC="${PARAKEET_NEMO_CONTAINER} (encoder+decoder only; joint trained from scratch)"
    ;;
  ctc_joint|ctc|ptl)
    SRC_EXP_NAME="${SRC_EXP_NAME:-${CLUSTER}_chat_fullctx_parakeet_hybctc${SRC_CTC_WEIGHT}_g2_ctx${CTX_TAG}_chunk${CHAT_CHUNK_SIZE}_lr${LR}_n${SLURM_JOB_NUM_NODES}}"
    SRC_CKPT_DIR="${OUTPUT_PREFIX}/results/${PROJECT_NAME}/${SRC_EXP_NAME}/${SRC_EXP_NAME}/${SRC_EXP_NAME}/checkpoints"
    INIT_CKPT="${INIT_CKPT:-$(ls -t ${SRC_CKPT_DIR}/*-last.ckpt 2>/dev/null | head -1)}"
    if [ -z "$INIT_CKPT" ] || [ ! -f "$INIT_CKPT" ]; then
        echo "ERROR: no *-last.ckpt found for the CTC-joint model under: ${SRC_CKPT_DIR}" >&2
        echo "       Set SRC_EXP_NAME=<ctc-joint exp name> or INIT_CKPT=<full path> explicitly." >&2
        exit 1
    fi
    # Checkpoint filenames contain '=' (e.g. step=NNNN-...), which Hydra cannot parse
    # in an override value. Symlink to an '='-free path and point the override there.
    SAFE_INIT_CKPT="${RESULTS_DIR}/init_from_ctc.ckpt"
    ln -sf "${INIT_CKPT}" "${SAFE_INIT_CKPT}"
    INIT_OVERRIDE="+init_from_ptl_ckpt=${SAFE_INIT_CKPT}"
    INIT_DESC="${INIT_CKPT} (from CTC-joint run; CTC head dropped)"
    ;;
  *)
    echo "ERROR: unknown INIT_FROM='${INIT_FROM}' (use parakeet or ctc_joint)" >&2
    exit 1
    ;;
esac
echo "==> INIT_FROM=${INIT_FROM}: ${INIT_DESC}"

# ---------------------------------------------------------------------------
# Data (Granary 2.0 pre-aligned train + English mcv11 dev, same as tdt.sh)
# ---------------------------------------------------------------------------
GRANARY2_CFG=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/dongjig/aligned_amos/granary_v2_en_pnc_qwen_aligned_filtered/granary_v2_en_pnc_qwen_aligned_filtered_safe_iad_s3_audio.yaml
TRAIN_INPUT_CFG="${TRAIN_INPUT_CFG:-$GRANARY2_CFG}"

# English dev set: same as the SpeechLM baseline / tdt.sh.
VAL_MANIFEST="${VAL_MANIFEST:-[/data/canary/canary_v0/manifests/data/ASR/MMLPC/en/val_test/mcv11/mcv11_dev_clean_pcstrip_en_2k.json]}"

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
echo "*******FULL-CONTEXT CHAT (FastConformer RNNT, pure - no CTC) - Granary 2.0********" \
&& echo "*** CONFIG: ${CONFIG_NAME} | ATT_CONTEXT=${ATT_CONTEXT} style=${ATT_CONTEXT_STYLE} ***" \
&& echo "*** FULL-CTX: causal_downsampling=${CAUSAL_DOWNSAMPLING} conv=${CONV_CONTEXT_SIZE}/${CONV_NORM_TYPE} normalize=${NORMALIZE} | CHAT chunk_size=${CHAT_CHUNK_SIZE} ***" \
&& echo "*** LOSS: pure CHAT(rnnt), reduction=${RNNT_REDUCTION} ***" \
&& echo "*** INIT: ${INIT_DESC} ***" \
&& echo "*** TOKENIZER: /tokenizers (extracted from the parakeet .nemo) ***" \
&& echo "*** PRECISION: ${PRECISION} (use bf16-mixed/32 with batch_norm; bf16-true often -> Inf) ***" \
&& echo "*** DATA: Granary 2.0 pre-aligned (lhotse) -> ${TRAIN_INPUT_CFG} ***" \
&& echo "*** RANK DIAG: SLURM_PROCID=\${SLURM_PROCID:-?} SLURM_LOCALID=\${SLURM_LOCALID:-?} SLURM_NTASKS=\${SLURM_NTASKS:-?} CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-?} ***" \
&& nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader \
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
&& export USE_AIS_GET_BATCH=${USE_AIS_GET_BATCH} \
&& export TMPDIR=${OCI_TMP_DIR} && mkdir -p ${OCI_TMP_DIR} && echo "staging TMPDIR=\$TMPDIR USE_AIS_GET_BATCH=\$USE_AIS_GET_BATCH" \
&& export AIS_ENDPOINT=http://asr.iad.oci.aistore.nvidia.com:51080 \
&& export AIS_AUTHN_TOKEN="${AIS_AUTHN_TOKEN}" \
&& export NEMO_DATA_STORE_CACHE_DIR=/lustre/fsw/portfolios/llmservice/users/heh/nemo_cache \
&& echo "Starting training (running MY code at /code, pure CHAT-RNNT full-context, GRANARY 2.0 data)" \
&& python /code/examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    name=${EXP_NAME} \
    ${INIT_OVERRIDE} \
    model.encoder.att_context_size=${ATT_CONTEXT} \
    model.encoder.att_context_style=${ATT_CONTEXT_STYLE} \
    model.encoder.causal_downsampling=${CAUSAL_DOWNSAMPLING} \
    model.encoder.conv_context_size=${CONV_CONTEXT_SIZE} \
    model.encoder.conv_norm_type=${CONV_NORM_TYPE} \
    model.preprocessor.normalize=${NORMALIZE} \
    model.decoder.prednet.pred_rnn_layers=${PRED_RNN_LAYERS} \
    ++model.joint.chunk_size=${CHAT_CHUNK_SIZE} \
    model.skip_nan_grad=true \
    model.compute_eval_loss=false \
    ++model.rnnt_reduction=${RNNT_REDUCTION} \
    ++model.train_ds.use_lhotse=true \
    ++model.train_ds.input_cfg=${TRAIN_INPUT_CFG} \
    model.train_ds.manifest_filepath=null \
    ++model.train_ds.return_cuts=true \
    ++model.train_ds.skip_missing_manifest_entries=true \
    ++model.train_ds.batch_duration=${BATCH_DURATION} \
    ++model.train_ds.use_bucketing=true \
    ++model.train_ds.num_buckets=${NUM_BUCKETS} \
    ++model.train_ds.use_start_end_token=false \
    ++model.train_ds.text_field=text \
    model.train_ds.max_duration=${MAX_DURATION} \
    model.train_ds.num_workers=${NUM_WORKERS} \
    model.train_ds.shuffle=true \
    model.train_ds.pin_memory=true \
    model.validation_ds.manifest_filepath=${VAL_MANIFEST} \
    model.validation_ds.batch_size=${EVAL_BATCH_SIZE} \
    model.validation_ds.num_workers=${NUM_WORKERS} \
    model.validation_ds.pin_memory=true \
    ++model.validation_ds.use_start_end_token=false \
    model.tokenizer.dir="/tokenizers" \
    model.tokenizer.type=bpe \
    model.optim.lr=${LR} \
    model.optim.sched.name=CosineAnnealing \
    ~model.optim.sched.d_model \
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
