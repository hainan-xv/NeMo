#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:chat-fullctx-ce-qwen-g2
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
# OCI launcher: FULL-CONTEXT CHAT trained with ALIGNMENT-GUIDED CROSS-ENTROPY (CE),
# but REUSING THE QWEN TOKENIZER (same tokenizer as the SpeechLM scripts under
# oci/, which set model.pretrained_llm=Qwen/Qwen3-1.7B).
#
# This is the Qwen-tokenizer variant of oci/chat_fullctx_ce.sh. Everything except
# the TOKENIZER (and the consequences of the vocab change) is identical:
#   - full-context encoder matched to parakeet-tdt-0.6b-v2 (att_context=[-1,-1],
#     batch_norm conv, per-feature mel norm, non-causal subsampling)
#   - RNNTAttJoint CHAT joint, chunk_size=${CHAT_CHUNK_SIZE}
#   - alignment-guided CE objective (EncDecRNNTBPEModelChatCE, use_chat_ce_dataset)
#   - Granary v2 pre-aligned train + English mcv11 dev
#
# TOKENIZER: the HuggingFace Qwen tokenizer (default Qwen/Qwen3-1.7B), loaded via
# NeMo's AutoTokenizer (model.tokenizer.type=huggingface). No SentencePiece .model.
#
# !!! IMPORTANT prerequisites / consequences of using the Qwen tokenizer !!!
#   1) ASR HuggingFace-tokenizer support: enabled in this repo via a `huggingface`
#      branch in nemo/collections/asr/parts/mixins/mixins.py::_setup_monolingual_tokenizer,
#      which loads AutoTokenizer(pretrained_model_name=...). The label set is built
#      over the HF *base* vocab (added chat/control tokens excluded) so num_classes
#      matches the RNN-T blank id. NOTE: the tokenizer is NOT packed into the .nemo;
#      on restore it is re-created from pretrained_model_name (kept in the config),
#      so that HF id / local dir must stay resolvable in the run environment.
#   2) Vocabulary change: Qwen's vocab (~151k) != parakeet's 1024, so the decoder
#      and joint CANNOT warm-start from parakeet. We warm-start the ENCODER ONLY
#      (include=[encoder]); decoder + RNNTAttJoint are trained from scratch.
#   3) The RNNTAttJoint final projection becomes joint_hidden x ~151k. CHAT-CE only
#      evaluates the joint along the single forced path (not the [B,T,U,V] tensor),
#      so training is tractable, but greedy-decode validation + the output matrix +
#      softmax are heavier than with the 1024-vocab SentencePiece tokenizer.
#
# Submit from an OCI login node:
#   ./sync_to_oci.sh
#   sbatch oci/chat_fullctx_ce_qwen.sh
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
# Qwen tokenizer (same LLM tokenizer as the SpeechLM scripts under oci/).
# ---------------------------------------------------------------------------
# oci/imend.sh etc. use model.pretrained_llm=Qwen/Qwen3-1.7B; its tokenizer is
# what we reuse here. Override with any HF repo id or a local HF tokenizer dir.
LLM_TOKENIZER="${LLM_TOKENIZER:-Qwen/Qwen3-1.7B}"
TOKENIZER_TAG="$(echo "${LLM_TOKENIZER}" | tr '/:' '__')"

# ---------------------------------------------------------------------------
# CHAT full-context + alignment-CE model + training hyper-parameters
# ---------------------------------------------------------------------------
CONFIG_PATH="${CONFIG_PATH:-/code/examples/asr/conf/fastconformer/cache_aware_streaming}"
CONFIG_NAME="${CONFIG_NAME:-fastconformer_chat_transducer_bpe_streaming}"

# Full-context encoder knobs, matching parakeet-tdt-0.6b-v2 so the ENCODER loads
# cleanly from the parakeet .nemo.
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
# Encoder frames to delay word emission after its end (word -> chunk mapping, data).
NUM_DELAY_FRAMES="${NUM_DELAY_FRAMES:-0}"

# Prediction network RNN layers. The decoder is trained from scratch here (Qwen
# vocab != parakeet), so this is a free choice; keep 2 for capacity parity with
# oci/chat_fullctx_ce.sh.
PRED_RNN_LAYERS="${PRED_RNN_LAYERS:-2}"

MAX_STEPS="${MAX_STEPS:-500000}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-6000}"
EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-1}"
VAL_CHECK_INTERVAL=$(( LIMIT_TRAIN_BATCHES / EVALS_PER_EPOCH ))
# Encoder warm-started but decoder+joint from scratch -> a slightly higher peak LR
# than the all-warm-started variant is reasonable. Still overridable.
LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
BATCH_DURATION="${BATCH_DURATION:-120}"
NUM_BUCKETS="${NUM_BUCKETS:-30}"
MAX_DURATION="${MAX_DURATION:-20.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SAVE_TOP_K="${SAVE_TOP_K:-5}"
# batch_norm encoder -> use bf16-mixed (or 32); bf16-true often -> Inf at step 0.
PRECISION="${PRECISION:-bf16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-8}"
USE_AIS_GET_BATCH="${USE_AIS_GET_BATCH:-true}"
RNNT_REDUCTION="${RNNT_REDUCTION:-mean_volume}"

# ---------------------------------------------------------------------------
# parakeet-tdt-0.6b-v2 checkpoint (ENCODER warm-start only; no tokenizer reuse)
# ---------------------------------------------------------------------------
INIT_FROM="${INIT_FROM:-parakeet}"
PRETRAINED_MODEL_DIR="${PRETRAINED_MODEL_DIR:-/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/hainanx/pretrained_models}"
PARAKEET_BASENAME="${PARAKEET_BASENAME:-parakeet-tdt-0.6b-v2.nemo}"
PARAKEET_NEMO_HOST="${PRETRAINED_MODEL_DIR}/${PARAKEET_BASENAME}"
PARAKEET_NEMO_CONTAINER="/pretrained/${PARAKEET_BASENAME}"

case "${INIT_FROM}" in
  parakeet|nemo|pretrained)
    if [ ! -f "${PARAKEET_NEMO_HOST}" ]; then
        echo "ERROR: parakeet checkpoint not found at ${PARAKEET_NEMO_HOST}" >&2
        echo "       Copy parakeet-tdt-0.6b-v2.nemo there, set PRETRAINED_MODEL_DIR, or use INIT_FROM=scratch." >&2
        exit 1
    fi
    # ENCODER ONLY: the Qwen vocab differs from parakeet's, so decoder + joint are
    # trained from scratch. (include=[encoder] -> partial load, strict=False.)
    INIT_OVERRIDE="+init_from_nemo_model.parakeet.path=${PARAKEET_NEMO_CONTAINER} +init_from_nemo_model.parakeet.include=[encoder]"
    INIT_DESC="${PARAKEET_NEMO_CONTAINER} (encoder only; decoder+joint from scratch for the Qwen vocab)"
    ;;
  scratch|none|off)
    INIT_OVERRIDE=""
    INIT_DESC="scratch (no warm-start)"
    ;;
  *)
    echo "ERROR: unknown INIT_FROM='${INIT_FROM}' (use parakeet or scratch)" >&2
    exit 1
    ;;
esac
echo "==> INIT_FROM=${INIT_FROM}: ${INIT_DESC}"

# ---------------------------------------------------------------------------
# Data (Granary 2.0 pre-aligned train + English mcv11 dev, same as chat_ce.sh)
# ---------------------------------------------------------------------------
GRANARY2_CFG=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/dongjig/aligned_amos/granary_v2_en_pnc_qwen_aligned_filtered/granary_v2_en_pnc_qwen_aligned_filtered_safe_iad_s3_audio.yaml
TRAIN_INPUT_CFG="${TRAIN_INPUT_CFG:-$GRANARY2_CFG}"

VAL_MANIFEST="${VAL_MANIFEST:-[/data/canary/canary_v0/manifests/data/ASR/MMLPC/en/val_test/mcv11/mcv11_dev_clean_pcstrip_en_2k.json]}"

OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"

EXP_NAME="${EXP_NAME:-${CLUSTER}_chat_fullctx_ce_qwentok_${TOKENIZER_TAG}_g2_ctx${CTX_TAG}_chunk${CHAT_CHUNK_SIZE}_delay${NUM_DELAY_FRAMES}_lr${LR}_${PRECISION}_n${SLURM_JOB_NUM_NODES}}"
RESULTS_DIR=${OUTPUT_PREFIX}/results/$PROJECT_NAME/$EXP_NAME
mkdir -p "${RESULTS_DIR}"

CHECKPOINT_DIR=${LUSTRE_ACCOUNT_PREFIX}/${OLDUSERID}/checkpoints/
HFCACHE=${OUTPUT_PREFIX}/hf_cache
SPEECHLM_PROJECT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm
DATA_DIR=${SPEECHLM_PROJECT_DIR}/data
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh
HAINAN_DIR=/lustre/fsw/portfolios/llmservice/users/hainanx
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo79/}"

OCI_TMP_DIR="${OCI_TMP_DIR:-/results/tmp}"

mkdir -p ${RESULTS_DIR} ${HFCACHE}
OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

# Note: no /tokenizers mount -- the Qwen tokenizer is fetched by HF into /hfcache.
MOUNTS="--container-mounts=${SPEECHLM_PROJECT_DIR}:${SPEECHLM_PROJECT_DIR},${H_DIR}:${H_DIR},$HAINAN_DIR:$HAINAN_DIR,$CODE_DIR:/code,$RESULTS_DIR:/results,$DATA_DIR:/data,$PRETRAINED_MODEL_DIR:/pretrained,$CHECKPOINT_DIR:/checkpoints,${HFCACHE}:/hfcache/,/lustre/fsw:/lustre/fsw,/lustre/fs12:/lustre/fs12"

read -r -d '' cmd <<EOF
echo "*******FULL-CONTEXT CHAT ALIGNMENT-CE (QWEN tokenizer) - Granary 2.0********" \
&& echo "*** CONFIG: ${CONFIG_NAME} | ATT_CONTEXT=${ATT_CONTEXT} style=${ATT_CONTEXT_STYLE} ***" \
&& echo "*** FULL-CTX: causal_downsampling=${CAUSAL_DOWNSAMPLING} conv=${CONV_CONTEXT_SIZE}/${CONV_NORM_TYPE} normalize=${NORMALIZE} | CHAT chunk_size=${CHAT_CHUNK_SIZE} delay=${NUM_DELAY_FRAMES} ***" \
&& echo "*** TOKENIZER: huggingface ${LLM_TOKENIZER} (Qwen) ***" \
&& echo "*** LOSS: alignment-guided CE (forced path), reduction=${RNNT_REDUCTION} ***" \
&& echo "*** INIT: ${INIT_DESC} ***" \
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
&& echo "Starting training (running MY code at /code, full-context CHAT alignment-CE, QWEN tokenizer, GRANARY 2.0 data)" \
&& python /code/examples/asr/asr_transducer/speech_to_text_chat_ce_bpe.py \
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
    ++model.chat_ce.reduction=${RNNT_REDUCTION} \
    ++model.chat_ce.num_delay_frames=${NUM_DELAY_FRAMES} \
    model.tokenizer.type=huggingface \
    ~model.tokenizer.dir \
    +model.tokenizer.hf_kwargs.pretrained_model_name=${LLM_TOKENIZER} \
    +model.tokenizer.hf_kwargs.use_fast=true \
    +model.tokenizer.hf_kwargs.trust_remote_code=true \
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
    model.train_ds.num_workers=${NUM_WORKERS} \
    model.train_ds.shuffle=true \
    model.train_ds.pin_memory=true \
    model.validation_ds.manifest_filepath=${VAL_MANIFEST} \
    model.validation_ds.batch_size=${EVAL_BATCH_SIZE} \
    model.validation_ds.num_workers=${NUM_WORKERS} \
    model.validation_ds.pin_memory=true \
    ++model.validation_ds.use_start_end_token=false \
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
