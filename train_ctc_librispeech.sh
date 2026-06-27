#!/bin/bash
# ============================================================================
# Local (single-box, no SLURM / no container) from-scratch training of a
# FastConformer-CTC large model (~120M, 8x dw-striding, conv_kernel 9) on
# LibriSpeech-960, with the trainable UNITARY RESIDUAL encoder feature
# (model.encoder.use_unitary_residual). The deep stack is a good stress test for
# whether gradients still flow with the orthogonal residuals.
#
# CTC counterpart of train_librispeech.sh (TDT). CTC trains substantially
# faster (no autoregressive joint / decoder loop), so it's the quicker way to
# A/B unitary-residual vs standard identity-residual encoders.
#
# Usage:
#   ./train_ctc_librispeech.sh [USE_UNITARY_RESIDUAL] [GPU_ID]
#     arg1 USE_UNITARY_RESIDUAL : true (default) | false  -> unitary vs standard baseline
#     arg2 GPU_ID               : CUDA device id to use (default 0)
#   e.g.  ./train_ctc_librispeech.sh true 0     # unitary run on GPU 0
#         ./train_ctc_librispeech.sh false 1    # standard baseline on GPU 1
#
# All the other knobs you'll typically want to sweep (n_layers, d_model, n_heads,
# vocab size, batch size, lr, ...) are plain variables in the CONFIG block
# below -- edit them and re-run. A SentencePiece BPE tokenizer is built once
# from the train manifest if it does not already exist.
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

# --- Encoder architecture knobs (the things you'll want to sweep) ---
USE_UNITARY_RESIDUAL="${1:-true}"   # arg1: true => trainable orthogonal residuals; false => standard
# FastConformer-large: 18 layers, d_model 512, 8 heads, conv kernel 9, 8x
# depthwise-striding subsampling. The depth is what makes gradient flow interesting.
N_LAYERS=18
D_MODEL=512
N_HEADS=8
FF_EXPANSION_FACTOR=4
CONV_KERNEL_SIZE=9
SUBSAMPLING_FACTOR=16

# --- Tokenizer (SentencePiece BPE; built from TRAIN_MANIFEST if missing) ---
VOCAB_SIZE=1024
TOKENIZER_ROOT="${REPO_ROOT}/tokenizers/librispeech"
TOKENIZER_DIR="${TOKENIZER_ROOT}/tokenizer_spe_bpe_v${VOCAB_SIZE}"  # produced by the builder

# --- Training knobs ---
DEVICES=1                   # number of local GPUs
CUDA_DEVICES="${2:-0}"      # arg2: which GPU id to expose (maps to CUDA_VISIBLE_DEVICES)
PRECISION=bf16              # bf16 | 16 | 32  (bf16 needs Ampere+)
BATCH_SIZE=64
NUM_WORKERS=8
ACCUM_GRAD_BATCHES=1
LR=0.1e-3                   # peak LR (config uses CosineAnnealing, so this is the literal peak)
WARMUP_STEPS=10000
MAX_STEPS=100000
VAL_CHECK_INTERVAL=0.2     # fraction of an epoch between validations -> 5 evals/epoch
MAX_DURATION=16.7
MIN_DURATION=0.1

# --- Logging / output ---
EXP_DIR="${HOME}/nemo_experiments_librispeech"
USE_WANDB=true
WANDB_PROJECT="librispeech_unitary_residual"
# WandB API key (reused from the ord3 launch scripts -> logs to the same account).
WANDB_API_KEY="e80635358f133b2c0e27fc1702cf9f416241705e"

# ============================ END CONFIG ====================================

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

if [ "${USE_UNITARY_RESIDUAL}" = "true" ]; then
  RESIDUAL_TAG=unitaryResidual
else
  RESIDUAL_TAG=stdResidual
fi

# --- Parameter-matched d_model ------------------------------------------------
# D_MODEL is the standard (identity-residual) model's width. The unitary residual
# adds 4 orthogonal d x d matrices per layer, so at the same d_model the unitary
# model would have more params. For a fair comparison, when unitary is ON we
# search for the d_model (a multiple of N_HEADS) whose TOTAL model parameter count
# best matches the standard model built at D_MODEL. We do this by actually
# instantiating the full model both ways and counting parameters (data loading is
# skipped). This requires a NeMo import + a few model builds, so it adds ~1 min.
if [ "${USE_UNITARY_RESIDUAL}" = "true" ]; then
  echo "[param-match] searching d_model so the unitary model matches the standard total params (building model a few times; ~1 min)..."
  MATCH_OUT=$(python - "${D_MODEL}" "${N_LAYERS}" "${N_HEADS}" "${FF_EXPANSION_FACTOR}" "${CONV_KERNEL_SIZE}" "${SUBSAMPLING_FACTOR}" "${TOKENIZER_DIR}" "${REPO_ROOT}/examples/asr/conf/fastconformer/fast-conformer_ctc_bpe.yaml" ctc <<'PY'
import sys
from omegaconf import OmegaConf, open_dict

d_std, n_layers, n_heads, ff, convk, sub = (int(sys.argv[i]) for i in range(1, 7))
tok_dir, cfg_path, model_type = sys.argv[7], sys.argv[8], sys.argv[9]

from nemo.collections.asr.models import EncDecCTCModelBPE, EncDecRNNTBPEModel

ModelCls = EncDecCTCModelBPE if model_type == 'ctc' else EncDecRNNTBPEModel
cfg = OmegaConf.load(cfg_path)
m = cfg.model
with open_dict(m):
    m.encoder.n_layers = n_layers
    m.encoder.n_heads = n_heads
    m.encoder.ff_expansion_factor = ff
    m.encoder.conv_kernel_size = convk
    m.encoder.subsampling_factor = sub
    m.tokenizer.dir = tok_dir
    m.tokenizer.type = 'bpe'
    m.train_ds.manifest_filepath = None
    m.validation_ds.manifest_filepath = None
    if 'test_ds' in m:
        m.test_ds.manifest_filepath = None

def count(d, unitary):
    with open_dict(m):
        m.encoder.d_model = int(d)
        m.encoder.use_unitary_residual = bool(unitary)
    model = ModelCls(cfg=m, trainer=None)
    n = sum(p.numel() for p in model.parameters())
    del model
    return n

p_std = count(d_std, False)
cands = list(range(n_heads, d_std + 1, n_heads))
# Total params increase monotonically with d, so binary-search the largest
# candidate whose unitary param count does not exceed the standard total.
lo, hi, res = 0, len(cands) - 1, 0
while lo <= hi:
    mid = (lo + hi) // 2
    if count(cands[mid], True) <= p_std:
        res = mid
        lo = mid + 1
    else:
        hi = mid - 1
choices = [cands[res]]
if res + 1 < len(cands):
    choices.append(cands[res + 1])
best = min(choices, key=lambda d: abs(count(d, True) - p_std))
sys.stderr.write("[param-match] standard(d=%d) total=%d ; unitary(d=%d) total=%d\n" % (d_std, p_std, best, count(best, True)))
print("__EFFECTIVE_D_MODEL__=%d" % best)
PY
)
  D_MODEL_EFFECTIVE=$(printf '%s\n' "${MATCH_OUT}" | sed -n 's/^__EFFECTIVE_D_MODEL__=//p' | tail -n1)
  if [ -z "${D_MODEL_EFFECTIVE}" ]; then
    echo "ERROR: parameter matching failed. Full output:"
    printf '%s\n' "${MATCH_OUT}"
    exit 1
  fi
  echo "[param-match] unitary d_model = ${D_MODEL_EFFECTIVE} (standard base d_model = ${D_MODEL})"
else
  D_MODEL_EFFECTIVE=${D_MODEL}
fi

EXP_NAME="ls960_fastconformerLargeCtc_${RESIDUAL_TAG}_L${N_LAYERS}_d${D_MODEL_EFFECTIVE}_h${N_HEADS}_v${VOCAB_SIZE}_subsampling${SUBSAMPLING_FACTOR}"

# Single-GPU -> let Lightning pick the strategy; multi-GPU -> DDP.
if [ "${DEVICES}" -gt 1 ]; then
  STRATEGY=ddp
else
  STRATEGY=auto
fi

# --- Build the BPE tokenizer once (idempotent; shared with the TDT script) ---
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
echo "use_unitary_residual=${USE_UNITARY_RESIDUAL} | layers=${N_LAYERS} d_model=${D_MODEL_EFFECTIVE} (base ${D_MODEL}) heads=${N_HEADS} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
if [ "${USE_WANDB}" = "true" ]; then
  export WANDB_API_KEY="${WANDB_API_KEY}"
fi

python "${REPO_ROOT}/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py" \
    --config-path="${REPO_ROOT}/examples/asr/conf/fastconformer" \
    --config-name=fast-conformer_ctc_bpe \
    name="${EXP_NAME}" \
    ++model.encoder.use_unitary_residual="${USE_UNITARY_RESIDUAL}" \
    model.encoder.n_layers="${N_LAYERS}" \
    model.encoder.d_model="${D_MODEL_EFFECTIVE}" \
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
    trainer.log_every_n_steps=50 \
    ++exp_manager.exp_dir="${EXP_DIR}" \
    ++exp_manager.resume_if_exists=true \
    ++exp_manager.resume_ignore_no_checkpoint=true \
    exp_manager.create_wandb_logger="${USE_WANDB}" \
    ++exp_manager.wandb_logger_kwargs.name="${EXP_NAME}" \
    ++exp_manager.wandb_logger_kwargs.project="${WANDB_PROJECT}"
