#!/usr/bin/env bash
# Local Streaming SpeechLM training on LibriSpeech.
#
# Defaults are sized as a development run: 1,000 train utterances, 100 dev
# utterances, one GPU, Qwen3-0.6B, and the ~115M streaming FastConformer-large.
# The script creates manifests under ./exp and launches StreamingSTTModel via
# torchrun. By default it reuses a pre-aligned train manifest (precomputed mode);
# validation is decode-only and needs no alignments. Set ALIGNMENT_MODE=online to
# generate word timestamps on the fly instead.
# Set TRAIN_LIMIT=0 / VAL_LIMIT=0 to use complete source manifests.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/hainanx/Workplace/data/librispeech}"
WORK_ROOT="${WORK_ROOT:-$HOME/local_runs/streaming_stt_librispeech}"
MANIFEST_DIR="$WORK_ROOT/manifests"
RUN_DIR="${RUN_DIR:-$WORK_ROOT/runs}"
RUN_NAME="${RUN_NAME:-streaming_stt_qwen0p6b_fc115m_librispeech}"
HF_HOME="${HF_HOME:-$WORK_ROOT/hf_cache}"
ENABLE_WANDB="${ENABLE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-streaming_stt_librispeech}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_NAME}"
# Stable, unique W&B run id per experiment. Without this, resume_if_exists makes
# exp_manager pass version=None as the W&B id, so runs in the same project can
# collide and a later launch overwrites an earlier run. Deriving the id from the
# run name keeps distinct experiments separate while still resuming on restart.
WANDB_RUN_ID="${WANDB_RUN_ID:-$WANDB_RUN_NAME}"

TRAIN_SOURCE="${TRAIN_SOURCE:-$DATA_ROOT/train_960-aligned.json}"
VAL_SOURCE="${VAL_SOURCE:-$DATA_ROOT/dev_clean.json}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
VAL_LIMIT="${VAL_LIMIT:-100}"
FORCE_REALIGN="${FORCE_REALIGN:-0}"
ALIGN_BATCH_SIZE="${ALIGN_BATCH_SIZE:-8}"
ALIGN_NUM_GPUS="${ALIGN_NUM_GPUS:-1}"
ALIGNMENT_MODE="${ALIGNMENT_MODE:-precomputed}"
FORCED_ALIGNER_MODEL="${FORCED_ALIGNER_MODEL:-Qwen/Qwen3-ForcedAligner-0.6B}"

PRETRAINED_LLM="${PRETRAINED_LLM:-Qwen/Qwen3-0.6B}"
PRETRAINED_ASR="${PRETRAINED_ASR:-stt_en_fastconformer_hybrid_large_streaming_80ms}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_STEPS="${MAX_STEPS:-2000}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-200}"
# Lightning interprets 1.0 as 100% of validation batches.
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-4e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-14}"
CHUNK_SIZE_STD="${CHUNK_SIZE_STD:-0}"
# Restrict LLM audio-frame queries to their own chunk's audio + text history
# (blocks attention to earlier chunks' audio). Fixed chunking only.
RESTRICT_AUDIO_ATTENTION="${RESTRICT_AUDIO_ATTENTION:-0}"
# Set BLANK_TOKEN="" to use Qwen's <|im_end|> directly for empty chunks.
# This is supported for fixed chunking only (CHUNK_SIZE > 0).
BLANK_TOKEN="${BLANK_TOKEN-<blank>}"
BLANK_TOKEN_OVERRIDE="model.blank_token=\"$BLANK_TOKEN\""
VAL_MAX_TOKENS_PER_CHUNK="${VAL_MAX_TOKENS_PER_CHUNK:-$CHUNK_SIZE}"
TRAIN_DECODE_EVERY_N_STEPS="${TRAIN_DECODE_EVERY_N_STEPS:-100}"
TRAIN_DECODE_MAX_TOKENS_PER_CHUNK="${TRAIN_DECODE_MAX_TOKENS_PER_CHUNK:-$CHUNK_SIZE}"
LEFT_CONTEXT="${LEFT_CONTEXT:-70}"
RIGHT_CONTEXT="${RIGHT_CONTEXT:-13}"
DELAY_FRAMES="${DELAY_FRAMES:-1}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
# Explicit modules are required by older PEFT releases that do not yet have
# Qwen3 in their built-in architecture mapping.
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}"
FREEZE_SPEECH_ENCODER="${FREEZE_SPEECH_ENCODER:-false}"
COMPACT_TEMPLATE="${COMPACT_TEMPLATE:-true}"
LOG_DETAILED_TRAIN_METRICS="${LOG_DETAILED_TRAIN_METRICS:-false}"

mkdir -p "$MANIFEST_DIR" "$RUN_DIR" "$HF_HOME" "$WORK_ROOT/matplotlib" "$WORK_ROOT/torch_cache"

for path in "$TRAIN_SOURCE" "$VAL_SOURCE"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: LibriSpeech manifest not found: $path" >&2
    exit 1
  fi
done

# Use the repository-local ignored token when the caller did not export one.
if [[ -z "${HF_TOKEN:-}" && -r "$REPO_ROOT/.hf_token" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '\r\n' < "$REPO_ROOT/.hf_token")"
fi
WANDB_HYDRA=false
if [[ "$ENABLE_WANDB" == "1" ]]; then
  WANDB_HYDRA=true
  if [[ -z "${WANDB_API_KEY:-}" && -r "$REPO_ROOT/.wandb_token" ]]; then
    export WANDB_API_KEY
    WANDB_API_KEY="$(tr -d '\r\n' < "$REPO_ROOT/.wandb_token")"
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: ENABLE_WANDB=1 but neither WANDB_API_KEY nor $REPO_ROOT/.wandb_token is available." >&2
    exit 1
  fi
elif [[ "$ENABLE_WANDB" != "0" ]]; then
  echo "ERROR: ENABLE_WANDB must be 0 or 1, got '$ENABLE_WANDB'." >&2
  exit 1
fi
export HF_HOME
export MPLCONFIGDIR="$WORK_ROOT/matplotlib"
export TORCH_HOME="$WORK_ROOT/torch_cache"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Prefer this checkout over any separately installed NeMo package.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Fail before alignment/model downloads if Python resolves the wrong checkout or
# this environment cannot import the required SpeechLM components.
python - "$REPO_ROOT" <<'PY'
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1]).resolve()
import nemo

nemo_path = pathlib.Path(nemo.__file__).resolve()
if repo_root not in nemo_path.parents:
    raise RuntimeError(f"Imported NeMo from {nemo_path}, expected checkout under {repo_root}")

from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel  # noqa: F401
from nemo.collections.speechlm2.modules.qwen_forced_aligner import QwenForcedAligner  # noqa: F401

print(f"NeMo preflight OK: {nemo_path}")
PY

subset_manifest() {
  local source="$1"
  local output="$2"
  local limit="$3"
  python - "$source" "$output" "$limit" <<'PY'
import filecmp
import os
import pathlib
import sys

source, output, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
temporary = output + ".tmp"
with open(source, encoding="utf-8") as src, open(temporary, "w", encoding="utf-8") as dst:
    for index, line in enumerate(src):
        if limit > 0 and index >= limit:
            break
        if line.strip():
            dst.write(line)
if os.path.exists(output) and filecmp.cmp(temporary, output, shallow=False):
    os.unlink(temporary)
else:
    os.replace(temporary, output)
PY
}

TRAIN_SUBSET="$MANIFEST_DIR/train.json"
VAL_SUBSET="$MANIFEST_DIR/dev.json"
TRAIN_ALIGNED="$MANIFEST_DIR/train-aligned.json"
VAL_ALIGNED="$MANIFEST_DIR/dev-aligned.json"

echo "==> Preparing LibriSpeech manifests"
subset_manifest "$TRAIN_SOURCE" "$TRAIN_SUBSET" "$TRAIN_LIMIT"
subset_manifest "$VAL_SOURCE" "$VAL_SUBSET" "$VAL_LIMIT"

has_alignments() {
  [[ -f "$1" ]] || return 1
  python - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as manifest:
    for line in manifest:
        if line.strip():
            item = json.loads(line)
            if item.get("alignments"):
                raise SystemExit(0)
raise SystemExit(1)
PY
}

ALIGNMENT_OVERRIDES=()
case "$ALIGNMENT_MODE" in
  online)
    TRAIN_MANIFEST="$TRAIN_SUBSET"
    VAL_MANIFEST="$VAL_SUBSET"
    ALIGNMENT_OVERRIDES=(
      "+forced_aligner={_target_:nemo.collections.speechlm2.modules.qwen_forced_aligner.QwenForcedAligner,pretrained_model:$FORCED_ALIGNER_MODEL,language:English}"
    )
    ;;
  precomputed)
    # Training needs word alignments. Prefer a source that already carries them
    # (e.g. train_960-aligned.json) so no re-alignment is needed; otherwise
    # generate them once for the train subset.
    if [[ "$FORCE_REALIGN" != "1" ]] && has_alignments "$TRAIN_SUBSET"; then
      echo "==> Using pre-aligned train manifest: $TRAIN_SUBSET"
      TRAIN_MANIFEST="$TRAIN_SUBSET"
    elif [[ "$FORCE_REALIGN" != "1" ]] && has_alignments "$TRAIN_ALIGNED" \
      && [[ ! "$TRAIN_SUBSET" -nt "$TRAIN_ALIGNED" ]]; then
      echo "==> Reusing existing aligned train manifest: $TRAIN_ALIGNED"
      TRAIN_MANIFEST="$TRAIN_ALIGNED"
    else
      echo "==> Generating word-level alignments for train (num_gpus=$ALIGN_NUM_GPUS)"
      python "$REPO_ROOT/scripts/speechlm2/align_manifest.py" \
        --input "$TRAIN_SUBSET" \
        --output "$TRAIN_ALIGNED" \
        --batch-size "$ALIGN_BATCH_SIZE" \
        --num-gpus "$ALIGN_NUM_GPUS"
      TRAIN_MANIFEST="$TRAIN_ALIGNED"
    fi

    if ! has_alignments "$TRAIN_MANIFEST"; then
      echo "ERROR: train manifest has no alignments; refusing to train an all-blank streaming model." >&2
      exit 1
    fi

    # Validation is decode-only (WER via autoregressive generation), so it needs
    # no alignments — use the raw dev subset directly.
    VAL_MANIFEST="$VAL_SUBSET"
    ;;
  *)
    echo "ERROR: ALIGNMENT_MODE must be 'online' or 'precomputed', got '$ALIGNMENT_MODE'." >&2
    exit 1
    ;;
esac

echo "==> Starting local Streaming SpeechLM training"
echo "    LLM:      $PRETRAINED_LLM"
echo "    ASR:      $PRETRAINED_ASR"
echo "    align:    $ALIGNMENT_MODE"
echo "    chunks:   N($CHUNK_SIZE, $CHUNK_SIZE_STD) frames (std=0 means fixed)"
echo "    audio-attn: restrict_to_own_chunk=$RESTRICT_AUDIO_ATTENTION"
if [[ -n "$BLANK_TOKEN" ]]; then
  echo "    blank:    $BLANK_TOKEN"
else
  echo "    blank:    disabled (empty chunks emit <|im_end|>)"
fi
echo "    train:    $TRAIN_MANIFEST"
echo "    val:      $VAL_MANIFEST"
echo "    run root: $RUN_DIR"
if [[ "$ENABLE_WANDB" == "1" ]]; then
  echo "    W&B:      $WANDB_PROJECT / $WANDB_RUN_NAME"
fi

cd "$REPO_ROOT"
torchrun --nproc-per-node="$NUM_GPUS" --master-port="$MASTER_PORT" \
  examples/speechlm2/streaming_stt_train.py \
  --config-name=streaming_stt_lora \
  "${ALIGNMENT_OVERRIDES[@]}" \
  "model.pretrained_llm=$PRETRAINED_LLM" \
  "model.pretrained_asr=$PRETRAINED_ASR" \
  "$BLANK_TOKEN_OVERRIDE" \
  "model.chunk_size=$CHUNK_SIZE" \
  "model.att_context_size=[$LEFT_CONTEXT,$RIGHT_CONTEXT]" \
  "++model.compact_template=$COMPACT_TEMPLATE" \
  "model.freeze_speech_encoder=$FREEZE_SPEECH_ENCODER" \
  "++model.log_detailed_train_metrics=$LOG_DETAILED_TRAIN_METRICS" \
  "++model.val_max_new_tokens_per_chunk=$VAL_MAX_TOKENS_PER_CHUNK" \
  "++model.train_decode_every_n_steps=$TRAIN_DECODE_EVERY_N_STEPS" \
  "++model.train_decode_max_new_tokens_per_chunk=$TRAIN_DECODE_MAX_TOKENS_PER_CHUNK" \
  "++model.restrict_audio_to_own_chunk=$RESTRICT_AUDIO_ATTENTION" \
  "model.lora.r=$LORA_RANK" \
  "model.lora.lora_alpha=$LORA_ALPHA" \
  "++model.lora.target_modules=$LORA_TARGET_MODULES" \
  "model.optimizer.lr=$LR" \
  "model.lr_scheduler.warmup_steps=$WARMUP_STEPS" \
  "data.dataset.num_delay_frames=$DELAY_FRAMES" \
  "++data.dataset.chunk_size_std=$CHUNK_SIZE_STD" \
  "++data.dataset.compact_template=$COMPACT_TEMPLATE" \
  "data.train_ds.manifest_filepath=$TRAIN_MANIFEST" \
  "data.train_ds.batch_size=$BATCH_SIZE" \
  "data.train_ds.num_workers=$NUM_WORKERS" \
  "data.validation_ds.datasets.val_set_0.manifest_filepath=$VAL_MANIFEST" \
  "data.validation_ds.batch_size=$VAL_BATCH_SIZE" \
  "data.validation_ds.num_workers=$NUM_WORKERS" \
  "trainer.devices=$NUM_GPUS" \
  "trainer.num_nodes=1" \
  "trainer.max_steps=$MAX_STEPS" \
  "trainer.limit_train_batches=$VAL_CHECK_INTERVAL" \
  "trainer.val_check_interval=$VAL_CHECK_INTERVAL" \
  "trainer.limit_val_batches=$LIMIT_VAL_BATCHES" \
  "trainer.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES" \
  "exp_manager.exp_dir=$RUN_DIR" \
  "exp_manager.name=$RUN_NAME" \
  "exp_manager.create_wandb_logger=$WANDB_HYDRA" \
  "exp_manager.wandb_logger_kwargs.project=$WANDB_PROJECT" \
  "exp_manager.wandb_logger_kwargs.name=$WANDB_RUN_NAME" \
  "++exp_manager.wandb_logger_kwargs.id=$WANDB_RUN_ID" \
  "exp_manager.wandb_logger_kwargs.resume=allow" \
  "exp_manager.max_time_per_run=null" \
  "++exp_manager.log_step_timing=false" \
  "++exp_manager.log_delta_step_timing=false" \
  "exp_manager.checkpoint_callback_params.monitor=val_wer" \
  "exp_manager.checkpoint_callback_params.mode=min" \
  "exp_manager.checkpoint_callback_params.every_n_train_steps=$VAL_CHECK_INTERVAL" \
  "exp_manager.checkpoint_callback_params.every_n_epochs=null"
