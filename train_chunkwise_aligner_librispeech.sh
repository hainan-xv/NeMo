#!/bin/bash
# ============================================================================
# Local (single-box, no SLURM / no container) from-scratch training of the
# CHUNKWISE-ALIGNER BASELINE on LibriSpeech-960, with a standard Conformer encoder.
#
# This is the published "Chunkwise Aligners for Streaming Speech Recognition"
# (arXiv:2605.11422) baseline that our alignment-free Chunked Aligner is compared
# against. The model is architecturally IDENTICAL to the Chunked Aligner (same
# encoder + RNN-T prediction net + joint + greedy chunked decoding); the ONLY
# difference is the training objective: a FROZEN external model force-aligns the
# transcript, the label->chunk assignment is fixed from that alignment, and the
# trainee maximizes the probability of that single path. The fraction of
# utterances skipped (alignment not left-packable into the lattice) is reported as
# `val_discard` alongside `val_wer`.
#
# Two external-alignment backends (set BACKEND below):
#   * qwen (default) -- WORD-level forced alignment via Qwen3-ForcedAligner. All
#     sub-words of a word share the word's chunk, so the external tokenizer need
#     NOT match this model's tokenizer. Requires `pip install -U qwen-asr`.
#   * ctc            -- TOKEN-level CTC forced alignment; the external CTC model
#     MUST share this model's tokenizer.
#
# See examples/asr/conf/aligner/chunkwise_aligner_encoder_bpe.yaml.
#
# Usage:
#   ./train_chunkwise_aligner_librispeech.sh [CHUNK_SIZE] [GPU_ID]
#     arg1 CHUNK_SIZE : encoder frames per chunk (default 12)
#     arg2 GPU_ID     : CUDA device id to use (default 0)
#   e.g.  ./train_chunkwise_aligner_librispeech.sh 12 0
#
# Entrypoint:  examples/asr/asr_aligner/speech_to_text_chunkwise_aligner_bpe.py
# Base config: examples/asr/conf/aligner/chunkwise_aligner_encoder_bpe.yaml
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

# --- Chunkwise-aligner knobs ---
CHUNK_SIZE="${1:-12}"       # arg1: encoder frames per chunk (each chunk emits <= chunk_size tokens)
REDUCTION=mean_volume       # 'mean_volume' (per-token NLL) | 'mean' (per-sequence NLL)

# --- Frozen external aligner backend ---
BACKEND="${BACKEND:-qwen}"   # 'qwen' (word-level, tokenizer-agnostic) | 'ctc' (token-level)

# qwen (word-level) backend: any forced aligner exposing word timestamps.
QWEN_ALIGNER_NAME="Qwen/Qwen3-ForcedAligner-0.6B"    # HF repo id or local dir
QWEN_ALIGNER_LANGUAGE="English"
QWEN_ALIGNER_DTYPE="bfloat16"

# ctc (token-level) backend: external CTC model MUST share the tokenizer below.
# Provide EITHER a local .nemo (preferred for reproducibility) OR a pretrained name.
EXTERNAL_ALIGNER_NEMO=""                              # e.g. /path/to/ls960_ctc_v1024.nemo
EXTERNAL_ALIGNER_NAME="stt_en_fastconformer_ctc_large"  # used only if EXTERNAL_ALIGNER_NEMO is empty

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

EXP_NAME="ls960_chunkwiseAligner_C${CHUNK_SIZE}_conformerLarge_L${N_LAYERS}_d${D_MODEL}_h${N_HEADS}_v${VOCAB_SIZE}"

# Single-GPU -> let Lightning pick the strategy; multi-GPU -> DDP.
if [ "${DEVICES}" -gt 1 ]; then
  STRATEGY=ddp
else
  STRATEGY=auto
fi

# External aligner overrides depend on the selected backend.
if [ "${BACKEND}" = "qwen" ]; then
  EXTERNAL_ALIGNER_OVERRIDE="model.external_aligner.backend=qwen \
    model.external_aligner.model_name=${QWEN_ALIGNER_NAME} \
    model.external_aligner.language=${QWEN_ALIGNER_LANGUAGE} \
    model.external_aligner.dtype=${QWEN_ALIGNER_DTYPE}"
  echo "[external-aligner] backend=qwen (word-level): ${QWEN_ALIGNER_NAME} lang=${QWEN_ALIGNER_LANGUAGE}"
elif [ -n "${EXTERNAL_ALIGNER_NEMO}" ]; then
  EXTERNAL_ALIGNER_OVERRIDE="model.external_aligner.backend=ctc model.external_aligner.model_path=${EXTERNAL_ALIGNER_NEMO}"
  echo "[external-aligner] backend=ctc, local CTC model: ${EXTERNAL_ALIGNER_NEMO}"
else
  EXTERNAL_ALIGNER_OVERRIDE="model.external_aligner.backend=ctc model.external_aligner.pretrained_name=${EXTERNAL_ALIGNER_NAME}"
  echo "[external-aligner] backend=ctc, pretrained CTC model: ${EXTERNAL_ALIGNER_NAME}"
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
echo "chunk_size=${CHUNK_SIZE} reduction=${REDUCTION} | layers=${N_LAYERS} d_model=${D_MODEL} heads=${N_HEADS} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
if [ "${USE_WANDB}" = "true" ]; then
  export WANDB_API_KEY="${WANDB_API_KEY}"
fi

python "${REPO_ROOT}/examples/asr/asr_aligner/speech_to_text_chunkwise_aligner_bpe.py" \
    --config-path="${REPO_ROOT}/examples/asr/conf/aligner" \
    --config-name=chunkwise_aligner_encoder_bpe \
    name="${EXP_NAME}" \
    model.chunked_aligner.chunk_size="${CHUNK_SIZE}" \
    model.chunked_aligner.reduction="${REDUCTION}" \
    ${EXTERNAL_ALIGNER_OVERRIDE} \
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
