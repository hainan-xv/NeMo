#!/bin/bash
# ============================================================================
# Local (single machine, multi-GPU) Aligner-Encoder training on LibriSpeech.
#
# Trains a small BPE Aligner-Encoder FROM SCRATCH on the local LibriSpeech JSON
# manifests under ~/Workplace/data/librispeech, using plain NeMo manifests
# (no lhotse / tarred data). Intended as a quick end-to-end smoke / overfit run
# and a clean place to verify the wandb logging path locally.
#
# It will (1) build a small SentencePiece BPE tokenizer from a manifest if one
# isn't present, then (2) launch training on the requested GPUs.
#
# Usage:
#   bash examples/asr/asr_aligner/local_librispeech_aligner.sh
#   ALIGNER_TYPE=nonar DEVICES=2 MAX_STEPS=500 bash .../local_librispeech_aligner.sh
#   WANDB_ENABLE=1 bash .../local_librispeech_aligner.sh   # also log to wandb
# ============================================================================
set -euo pipefail

# --- Repo root (this script lives in examples/asr/asr_aligner) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# --- Data (override via env) ---
DATA=${DATA:-${HOME}/Workplace/data/librispeech}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-${DATA}/train_clean_360.json}
VAL_MANIFEST=${VAL_MANIFEST:-${DATA}/dev_clean.json}
# Tokenizer is trained on this manifest (dev_clean has enough text for a small
# vocab; for a larger run point this + TRAIN_MANIFEST at train_960.json etc.).
TOKENIZER_MANIFEST=${TOKENIZER_MANIFEST:-${DATA}/dev_clean.json}
VOCAB_SIZE=${VOCAB_SIZE:-128}

# --- Run knobs (override via env) ---
ALIGNER_TYPE=${ALIGNER_TYPE:-ar}        # "ar" or "nonar"
DEVICES=${DEVICES:-2}                  # number of GPUs
PRECISION=${PRECISION:-32}             # 32 | 16 | bf16-mixed (bf16 needs Ampere+)
MAX_STEPS=${MAX_STEPS:-2000}
BATCH_SIZE=${BATCH_SIZE:-8}
EXP_DIR=${EXP_DIR:-${REPO_DIR}/nemo_experiments_aligner_local}
WANDB_ENABLE=${WANDB_ENABLE:-1}
# Default to a live online W&B run so panels can be inspected immediately.
# Override with WANDB_MODE=offline only when deliberately testing offline sync.
WANDB_MODE=${WANDB_MODE:-online}       # online | offline | disabled

echo "==============================================================="
echo " Aligner-Encoder LOCAL training"
echo "   repo:        ${REPO_DIR}"
echo "   train:       ${TRAIN_MANIFEST}"
echo "   val:         ${VAL_MANIFEST}"
echo "   tokenizer:   spe/bpe v${VOCAB_SIZE} (from ${TOKENIZER_MANIFEST})"
echo "   model:       aligner/${ALIGNER_TYPE} | devices=${DEVICES} | precision=${PRECISION} | max_steps=${MAX_STEPS}"
echo "   wandb:       enable=${WANDB_ENABLE} | mode=${WANDB_MODE}"
echo "==============================================================="

for f in "${TRAIN_MANIFEST}" "${VAL_MANIFEST}" "${TOKENIZER_MANIFEST}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: manifest not found: ${f}" >&2
    exit 1
  fi
done

# --- 1) Build a SentencePiece BPE tokenizer if not already present ---
TOK_ROOT="${REPO_DIR}/local_tokenizers"
TOK_DIR="${TOK_ROOT}/tokenizer_spe_bpe_v${VOCAB_SIZE}"
if [[ -f "${TOK_DIR}/tokenizer.model" ]]; then
  echo "Reusing existing tokenizer: ${TOK_DIR}"
else
  echo "Building tokenizer -> ${TOK_DIR}"
  mkdir -p "${TOK_ROOT}"
  python scripts/tokenizers/process_asr_text_tokenizer.py \
    --manifest="${TOKENIZER_MANIFEST}" \
    --data_root="${TOK_ROOT}" \
    --vocab_size="${VOCAB_SIZE}" \
    --tokenizer=spe \
    --spe_type=bpe \
    --spe_character_coverage=1.0 \
    --no_lower_case \
    --log
fi

if [[ ! -f "${TOK_DIR}/tokenizer.model" ]]; then
  echo "ERROR: tokenizer build did not produce ${TOK_DIR}/tokenizer.model" >&2
  exit 1
fi

# --- 2) Optional wandb wiring ---
WANDB_ARGS=()
if [[ "${WANDB_ENABLE}" == "1" ]]; then
  export WANDB_MODE="${WANDB_MODE}"
  WANDB_ARGS=(
    exp_manager.create_wandb_logger=true
    exp_manager.wandb_logger_kwargs.name=aligner_local_${ALIGNER_TYPE}
    exp_manager.wandb_logger_kwargs.project=aligner_local
  )
fi

# --- 3) Launch training ---
python examples/asr/asr_aligner/speech_to_text_aligner_bpe.py \
  --config-path=../conf/aligner \
  --config-name=aligner_encoder_bpe_local \
  name=aligner_local_${ALIGNER_TYPE} \
  model.aligner_type=${ALIGNER_TYPE} \
  model.tokenizer.dir=${TOK_DIR} \
  model.tokenizer.type=bpe \
  model.train_ds.manifest_filepath=${TRAIN_MANIFEST} \
  model.train_ds.batch_size=${BATCH_SIZE} \
  model.validation_ds.manifest_filepath=${VAL_MANIFEST} \
  model.validation_ds.batch_size=${BATCH_SIZE} \
  trainer.devices=${DEVICES} \
  trainer.precision=${PRECISION} \
  trainer.max_steps=${MAX_STEPS} \
  exp_manager.exp_dir=${EXP_DIR} \
  "${WANDB_ARGS[@]}"
