#!/bin/bash
# ============================================================================
# Local (single-box, no SLURM / no container) from-scratch training of a
# FastConformer-CTC large model (8x dw-striding, conv_kernel 9) on LibriSpeech-960
# whose encoder ALTERNATES time-axis and CHANNEL-TOKEN-axis attention
# (ChunkAlternatingConformerEncoder).
#
# Idea (the "attention over the feature axis" formulation): instead of every
# layer attending over the time axis, every other Conformer layer is replaced by
# a ChannelAxisConformerLayer. It cuts the encoder output into chunks of
# CHUNK_SIZE frames, reshapes each [CHUNK_SIZE, d_model] block losslessly to
# [new_d_model, CHUNK_TOKENS] (new_d_model = CHUNK_SIZE*d_model/CHUNK_TOKENS),
# and runs a Conformer block whose self-attention mixes the CHUNK_TOKENS channel
# tokens (i.e. the attention weights live on the reshaped feature axis, per
# chunk), then reshapes back to [B, T, d_model]. No information is dropped.
#
# Usage:
#   ./train_ctc_chunkattn_librispeech.sh [USE_CHUNK_ATTN] [GPU_ID]
#     arg1 USE_CHUNK_ATTN : true (default) | false
#                           true  -> ChunkAlternatingConformerEncoder (this method)
#                           false -> plain ConformerEncoder (standard time-axis baseline)
#     arg2 GPU_ID         : CUDA device id to use (default 0)
#   e.g.  ./train_ctc_chunkattn_librispeech.sh true 0    # chunk/channel-attn run
#         ./train_ctc_chunkattn_librispeech.sh false 1   # standard baseline
#
# Architecture/chunk-geometry knobs are plain variables in the CONFIG block
# below. A SentencePiece BPE tokenizer is built once from the train manifest if
# it does not already exist (shared with the other librispeech scripts).
#
# Entrypoint:  examples/asr/asr_ctc/speech_to_text_ctc_bpe.py
# Base config: examples/asr/conf/fastconformer/fast-conformer_ctc_bpe.yaml (FastConformer-large)
# ============================================================================

set -eo pipefail

# Resolve repo root from this script's location so it can be run from anywhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# ============================ CONFIG (edit me) ==============================

# --- Data (local LibriSpeech manifests; audio_filepath entries are absolute) ---
DATA_DIR="/home/hainanx/Workplace/data/librispeech"
TRAIN_MANIFEST="${DATA_DIR}/train_960.json"
VAL_MANIFEST="${DATA_DIR}/dev_clean.json"     # add dev_other by making this a [a,b] list
TEST_MANIFEST="${DATA_DIR}/test_clean.json"   # set to null to skip the final test run

# --- Encoder architecture knobs ---
USE_CHUNK_ATTN="${1:-true}"   # arg1: true => channel-axis alternating encoder; false => standard
# FastConformer-large: 18 layers, d_model 512, 8 heads, conv kernel 9, 8x
# depthwise-striding subsampling.
N_LAYERS=18
D_MODEL=512
N_HEADS=8
FF_EXPANSION_FACTOR=4
CONV_KERNEL_SIZE=9
SUBSAMPLING_FACTOR=16

# --- Channel-token-axis (chunk) knobs (only used when USE_CHUNK_ATTN=true) ---
# Constraint: CHUNK_SIZE * D_MODEL must be divisible by CHUNK_TOKENS.
# new_d_model = CHUNK_SIZE * D_MODEL / CHUNK_TOKENS is the feature dim the
# channel layers operate at, and CHUNK_TOKENS is how many tokens they attend over.
CHUNK_SIZE=8                  # encoder frames per chunk (C)
CHUNK_TOKENS=8                # tokens per chunk attended over (M); new_d_model = C*D/M
CHANNEL_CONV_KERNEL_SIZE=3    # depthwise conv kernel over the M axis (must be ODD)
CHANNEL_LAYERS=odd            # 'odd' => layer 0 time, then alternate; 'even' => layer 0 channel

# --- Tokenizer (SentencePiece BPE; built from TRAIN_MANIFEST if missing) ---
VOCAB_SIZE=1024
TOKENIZER_ROOT="${REPO_ROOT}/tokenizers/librispeech"
TOKENIZER_DIR="${TOKENIZER_ROOT}/tokenizer_spe_bpe_v${VOCAB_SIZE}"  # produced by the builder

# --- Training knobs ---
DEVICES=1                   # number of local GPUs
CUDA_DEVICES="${2:-0}"      # arg2: which GPU id to expose (maps to CUDA_VISIBLE_DEVICES)
PRECISION=bf16              # bf16 | 16 | 32  (bf16 needs Ampere+)
BATCH_SIZE=16               # FastConformer-large (8x); bump if VRAM allows
NUM_WORKERS=8
ACCUM_GRAD_BATCHES=1
LR=0.5e-3                   # peak LR (config uses CosineAnnealing, so this is the literal peak)
WARMUP_STEPS=10000
GRAD_CLIP=1.0               # gradient_clip_val; 0.0 disables. Config default is 0.0 (off)
MAX_STEPS=100000
VAL_CHECK_INTERVAL=0.2     # fraction of an epoch between validations -> 5 evals/epoch
MAX_DURATION=16.7
MIN_DURATION=0.1

# --- Logging / output ---
EXP_DIR="${HOME}/nemo_experiments_librispeech"
USE_WANDB=true
# Dedicated project for the chunk/channel-axis-attention comparison (this run vs its
# FastConformer time-axis baseline launched with arg1=false).
WANDB_PROJECT="librispeech_chunk_channel_attn"
# WandB API key (reused from the ord3 launch scripts -> logs to the same account).
WANDB_API_KEY="e80635358f133b2c0e27fc1702cf9f416241705e"

# ============================ END CONFIG ====================================

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# --- Build the encoder-specific Hydra overrides ------------------------------
ENCODER_ARGS=()
if [ "${USE_CHUNK_ATTN}" = "true" ]; then
  if [ $(( (CHUNK_SIZE * D_MODEL) % CHUNK_TOKENS )) -ne 0 ]; then
    echo "ERROR: CHUNK_SIZE*D_MODEL (${CHUNK_SIZE}*${D_MODEL}) must be divisible by CHUNK_TOKENS (${CHUNK_TOKENS})."
    exit 1
  fi
  if [ $(( CHANNEL_CONV_KERNEL_SIZE % 2 )) -eq 0 ]; then
    echo "ERROR: CHANNEL_CONV_KERNEL_SIZE must be odd, got ${CHANNEL_CONV_KERNEL_SIZE}."
    exit 1
  fi
  NEW_D_MODEL=$(( CHUNK_SIZE * D_MODEL / CHUNK_TOKENS ))
  ENC_TAG="chunkAttn_C${CHUNK_SIZE}_M${CHUNK_TOKENS}"
  # Swap the encoder class and add the channel-axis-only arguments (++ creates keys).
  ENCODER_ARGS+=(
    "model.encoder._target_=nemo.collections.asr.modules.ChunkAlternatingConformerEncoder"
    "++model.encoder.chunk_size=${CHUNK_SIZE}"
    "++model.encoder.chunk_tokens=${CHUNK_TOKENS}"
    "++model.encoder.channel_conv_kernel_size=${CHANNEL_CONV_KERNEL_SIZE}"
    "++model.encoder.channel_layers=${CHANNEL_LAYERS}"
  )
else
  ENC_TAG="stdConformer"
fi

EXP_NAME="ls960_fastconformerLargeCtc_${ENC_TAG}_L${N_LAYERS}_d${D_MODEL}_h${N_HEADS}_v${VOCAB_SIZE}_subsampling${SUBSAMPLING_FACTOR}"

# Single-GPU -> let Lightning pick the strategy; multi-GPU -> DDP.
if [ "${DEVICES}" -gt 1 ]; then
  STRATEGY=ddp
else
  STRATEGY=auto
fi

# --- Build the BPE tokenizer once (idempotent; shared with the other scripts) ---
if [ -f "${TOKENIZER_DIR}/tokenizer.model" ]; then
  echo "[tokenizer] reusing existing tokenizer at ${TOKENIZER_DIR}"
else
  echo "[tokenizer] building SPE-BPE (vocab=${VOCAB_SIZE}) from ${TRAIN_MANIFEST}"
  mkdir -p "${TOKENIZER_ROOT}"
  python "${REPO_ROOT}/scripts/tokenizers/process_asr_text_tokenizer.py" \
    --manifest="${TRAIN_MANIFEST}" \
    --data_root="${TOKENIZER_ROOT}" \
    --vocab_size="${VOCAB_SIZE}" \
    --tokenizer=spe \
    --spe_type=bpe \
    --spe_character_coverage=1.0 \
    --no_lower_case \
    --log
fi

echo "*******STARTING (${EXP_NAME})********"
if [ "${USE_CHUNK_ATTN}" = "true" ]; then
  echo "chunk/channel-axis attn ON | chunk_size=${CHUNK_SIZE} chunk_tokens=${CHUNK_TOKENS} new_d_model=${NEW_D_MODEL} channel_layers=${CHANNEL_LAYERS}"
else
  echo "chunk/channel-axis attn OFF (standard time-axis ConformerEncoder baseline)"
fi
echo "layers=${N_LAYERS} d_model=${D_MODEL} heads=${N_HEADS} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
if [ "${USE_WANDB}" = "true" ]; then
  export WANDB_API_KEY="${WANDB_API_KEY}"
fi

python "${REPO_ROOT}/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py" \
    --config-path="${REPO_ROOT}/examples/asr/conf/fastconformer" \
    --config-name=fast-conformer_ctc_bpe \
    name="${EXP_NAME}" \
    "${ENCODER_ARGS[@]}" \
    model.encoder.n_layers="${N_LAYERS}" \
    model.encoder.d_model="${D_MODEL}" \
    model.encoder.n_heads="${N_HEADS}" \
    model.encoder.ff_expansion_factor="${FF_EXPANSION_FACTOR}" \
    model.encoder.conv_kernel_size="${CONV_KERNEL_SIZE}" \
    model.encoder.subsampling_factor="${SUBSAMPLING_FACTOR}" \
    model.tokenizer.dir="${TOKENIZER_DIR}" \
    model.tokenizer.type=bpe \
    model.train_ds.manifest_filepath="${TRAIN_MANIFEST}" \
    model.train_ds.batch_size="${BATCH_SIZE}" \
    model.train_ds.num_workers="${NUM_WORKERS}" \
    model.train_ds.max_duration="${MAX_DURATION}" \
    model.train_ds.min_duration="${MIN_DURATION}" \
    model.train_ds.shuffle=true \
    model.validation_ds.manifest_filepath="${VAL_MANIFEST}" \
    model.validation_ds.batch_size="${BATCH_SIZE}" \
    model.validation_ds.num_workers="${NUM_WORKERS}" \
    model.test_ds.manifest_filepath="${TEST_MANIFEST}" \
    model.test_ds.batch_size="${BATCH_SIZE}" \
    model.optim.lr="${LR}" \
    model.optim.sched.warmup_steps="${WARMUP_STEPS}" \
    trainer.devices="${DEVICES}" \
    trainer.accelerator=gpu \
    trainer.strategy="${STRATEGY}" \
    trainer.precision="${PRECISION}" \
    trainer.max_steps="${MAX_STEPS}" \
    ++trainer.max_epochs=-1 \
    trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
    trainer.accumulate_grad_batches="${ACCUM_GRAD_BATCHES}" \
    trainer.gradient_clip_val="${GRAD_CLIP}" \
    trainer.log_every_n_steps=50 \
    ++exp_manager.exp_dir="${EXP_DIR}" \
    ++exp_manager.resume_if_exists=true \
    ++exp_manager.resume_ignore_no_checkpoint=true \
    exp_manager.create_wandb_logger="${USE_WANDB}" \
    ++exp_manager.wandb_logger_kwargs.name="${EXP_NAME}" \
    ++exp_manager.wandb_logger_kwargs.project="${WANDB_PROJECT}"
