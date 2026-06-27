#!/bin/bash
# ============================================================================
# Local (single-box, no SLURM / no container) from-scratch training of an
# ALIGNER-ENCODER ASR model (Stooke et al. 2025, "Aligner-Encoders") on
# LibriSpeech-960, with a standard Conformer encoder.
#
# The Aligner-Encoder reuses a Conformer encoder + (for the AR variant) an RNN-T
# prediction net, but pairs encoder frame i with prediction step i one-to-one
# (the RNN-T lattice diagonal, no blank) and trains with a frame-wise
# cross-entropy loss instead of dynamic programming. An EOS token terminates
# decoding. See examples/asr/conf/aligner/aligner_encoder_bpe.yaml.
#
# Usage:
#   ./train_aligner_librispeech.sh [ALIGNER_TYPE] [GPU_ID]
#     arg1 ALIGNER_TYPE : ar (default) | nonar
#                         ar    -> autoregressive (prediction net + 1:1 joint)
#                         nonar -> per-frame head (CTC-like, no prediction net)
#     arg2 GPU_ID       : CUDA device id to use (default 0)
#   e.g.  ./train_aligner_librispeech.sh ar 0
#         ./train_aligner_librispeech.sh nonar 1
#
# Architecture / training knobs are plain variables in the CONFIG block below.
# A SentencePiece BPE tokenizer is built once from the train manifest if missing
# (shared with the other librispeech scripts).
#
# Entrypoint:  examples/asr/asr_aligner/speech_to_text_aligner_bpe.py
# Base config: examples/asr/conf/aligner/aligner_encoder_bpe.yaml (standard Conformer)
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

# --- Aligner knobs ---
ALIGNER_TYPE="${1:-ar}"     # arg1: ar (autoregressive) | nonar (per-frame head)
LABEL_SMOOTHING=0.1         # label-smoothing weight for the CE loss
AUX_NONAR_LOSS_WEIGHT=0.0   # >0 adds an auxiliary non-AR loss to the AR model

# --- Encoder architecture knobs (standard Conformer; FastConformer-large defaults) ---
N_LAYERS=17
D_MODEL=512
N_HEADS=8
FF_EXPANSION_FACTOR=4
CONV_KERNEL_SIZE=9
SUBSAMPLING_FACTOR=8

# --- Tokenizer (SentencePiece BPE; built from TRAIN_MANIFEST if missing) ---
VOCAB_SIZE=1024
TOKENIZER_ROOT="${REPO_ROOT}/tokenizers/librispeech"
TOKENIZER_DIR="${TOKENIZER_ROOT}/tokenizer_spe_bpe_v${VOCAB_SIZE}"  # produced by the builder

# --- Training knobs ---
DEVICES=1                   # number of local GPUs
CUDA_DEVICES="${2:-0}"      # arg2: which GPU id to expose (maps to CUDA_VISIBLE_DEVICES)
PRECISION=bf16              # bf16 | 16 | 32  (bf16 needs Ampere+)
BATCH_SIZE=16               # bump if VRAM allows
NUM_WORKERS=8
ACCUM_GRAD_BATCHES=4        # eff. batch = BATCH_SIZE*this
LR=1e-4                     # peak LR (config uses CosineAnnealing, so this is the literal peak)
WARMUP_STEPS=10000
MAX_STEPS=100000
VAL_CHECK_INTERVAL=0.2      # fraction of an epoch between validations -> 5 evals/epoch
MAX_DURATION=16.7
MIN_DURATION=0.1
GRAD_CLIP=1.0               # gradient_clip_val; 0.0 disables. Config default is 0.0 (off)
TIME_MASKS=5               # SpecAugment time masks (config default 10); fewer -> faster from-scratch convergence

# --- Logging / output ---
EXP_DIR="${HOME}/nemo_experiments_librispeech"
USE_WANDB=true
WANDB_PROJECT="librispeech_aligner"
# WandB API key (reused from the ord3 launch scripts -> logs to the same account).
WANDB_API_KEY="e80635358f133b2c0e27fc1702cf9f416241705e"

# ============================ END CONFIG ====================================

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

EXP_NAME="ls960_aligner_${ALIGNER_TYPE}_conformerLarge_L${N_LAYERS}_d${D_MODEL}_h${N_HEADS}_v${VOCAB_SIZE}"

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
echo "aligner_type=${ALIGNER_TYPE} | layers=${N_LAYERS} d_model=${D_MODEL} heads=${N_HEADS} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
if [ "${USE_WANDB}" = "true" ]; then
  export WANDB_API_KEY="${WANDB_API_KEY}"
fi

python "${REPO_ROOT}/examples/asr/asr_aligner/speech_to_text_aligner_bpe.py" \
    --config-path="${REPO_ROOT}/examples/asr/conf/aligner" \
    --config-name=aligner_encoder_bpe \
    name="${EXP_NAME}" \
    model.aligner_type="${ALIGNER_TYPE}" \
    model.label_smoothing="${LABEL_SMOOTHING}" \
    model.aux_nonar_loss_weight="${AUX_NONAR_LOSS_WEIGHT}" \
    model.encoder.n_layers="${N_LAYERS}" \
    model.encoder.d_model="${D_MODEL}" \
    model.encoder.n_heads="${N_HEADS}" \
    model.encoder.ff_expansion_factor="${FF_EXPANSION_FACTOR}" \
    model.encoder.conv_kernel_size="${CONV_KERNEL_SIZE}" \
    model.encoder.subsampling_factor="${SUBSAMPLING_FACTOR}" \
    ++model.spec_augment.time_masks="${TIME_MASKS}" \
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
