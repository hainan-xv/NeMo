#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:tdt-parakeet06bv2-g2
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
# OCI launcher to REPRODUCE parakeet-tdt-0.6b-v2 and continue training on
# Granary v2.0.
#
# This is a plain offline (full-context) FastConformer-XL TDT model
# (EncDecRNNTBPEModel with the TDT loss), NOT a CHAT/streaming model. It is used
# as the reference/baseline: warm-start the WHOLE model (encoder + decoder +
# joint) from the released parakeet-tdt-0.6b-v2.nemo and keep training on the
# same Granary v2.0 data the CHAT launchers use.
#
# Key reproduction choices:
#   - Architecture: examples/asr/conf/fastconformer/
#       fastconformer_tdt_parakeet06b_bpe.yaml -- every architecture field is
#       copied from parakeet-tdt-0.6b-v2's own model_config.yaml (128 mel,
#       24-layer d_model=1024 FastConformer, dw_striding 8x, use_bias=false,
#       xscaling=false, conv_kernel=9 batch_norm; RNNTDecoder pred_rnn_layers=2;
#       TDT joint num_extra_outputs=5; loss_name=tdt sigma=0.02 omega=0.1).
#   - Weights: +init_from_nemo_model=<parakeet .nemo> (str form -> loads the FULL
#       state dict: encoder + decoder + joint, strict=False).
#   - Tokenizer: the EXACT 1024-BPE SentencePiece tokenizer that ships inside
#       parakeet-tdt-0.6b-v2.nemo. It is extracted from the .nemo on the login
#       node (below) and mounted at /tokenizers, so we don't depend on any
#       external tokenizer directory.
#
# Runs MY code (git-synced repo mounted at /code), not the container's NeMo.
#
# Prereq: parakeet-tdt-0.6b-v2.nemo must live at ${PARAKEET_NEMO_HOST} on OCI
# lustre (see PRETRAINED_MODEL_DIR below):
#   /lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/hainanx/pretrained_models/parakeet-tdt-0.6b-v2.nemo
#
# Submit from an OCI login node:
#   ./sync_to_oci.sh
#   sbatch oci_chat/tdt.sh
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
# parakeet-tdt-0.6b-v2 (offline FastConformer TDT) model + training hyper-params
# ---------------------------------------------------------------------------
CONFIG_PATH="${CONFIG_PATH:-/code/examples/asr/conf/fastconformer}"
CONFIG_NAME="${CONFIG_NAME:-fastconformer_tdt_parakeet06b_bpe}"

MAX_STEPS="${MAX_STEPS:-500000}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-6000}"
EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-1}"
VAL_CHECK_INTERVAL=$(( LIMIT_TRAIN_BATCHES / EVALS_PER_EPOCH ))
# Continuing from a converged checkpoint -> modest peak LR + warmup.
LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
BATCH_DURATION="${BATCH_DURATION:-120}"
NUM_BUCKETS="${NUM_BUCKETS:-30}"
MAX_DURATION="${MAX_DURATION:-20.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SAVE_TOP_K="${SAVE_TOP_K:-5}"
PRECISION="${PRECISION:-bf16}"
# TDT loss reduction: mean_volume | mean_batch | sum | mean.
RNNT_REDUCTION="${RNNT_REDUCTION:-mean_volume}"

# ---------------------------------------------------------------------------
# Warm-start: the FULL parakeet-tdt-0.6b-v2 model (encoder + decoder + joint)
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
# Load the entire state dict (encoder + decoder + joint) from the .nemo.
INIT_OVERRIDE="+init_from_nemo_model=${PARAKEET_NEMO_CONTAINER}"

# ---------------------------------------------------------------------------
# Tokenizer: EXACT tokenizer packaged inside parakeet-tdt-0.6b-v2.nemo.
# Extract it once (on the login node) into a stable dir and mount at /tokenizers.
# ---------------------------------------------------------------------------
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
# Data (Granary 2.0 pre-aligned train + English mcv11 dev, same as the CHAT runs)
# ---------------------------------------------------------------------------
GRANARY2_CFG=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/dongjig/aligned_amos/granary_v2_en_pnc_qwen_aligned_filtered/granary_v2_en_pnc_qwen_aligned_filtered_safe_iad_s3_audio.yaml
TRAIN_INPUT_CFG="${TRAIN_INPUT_CFG:-$GRANARY2_CFG}"

# English dev set (same as the SpeechLM baseline / CHAT launchers).
VAL_MANIFEST="${VAL_MANIFEST:-[/data/canary/canary_v0/manifests/data/ASR/MMLPC/en/val_test/mcv11/mcv11_dev_clean_pcstrip_en_2k.json]}"

EXP_NAME="${EXP_NAME:-${CLUSTER}_tdt_parakeet06bv2_g2_lr${LR}_n${SLURM_JOB_NUM_NODES}}"

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
echo "*******REPRODUCE parakeet-tdt-0.6b-v2 (offline FastConformer TDT) - Granary 2.0********" \
&& echo "*** CONFIG: ${CONFIG_NAME} (arch mirrors parakeet-tdt-0.6b-v2) ***" \
&& echo "*** INIT: ${PARAKEET_NEMO_CONTAINER} (FULL model: encoder+decoder+joint) ***" \
&& echo "*** TOKENIZER: /tokenizers (extracted from the parakeet .nemo) ***" \
&& echo "*** DATA: Granary 2.0 pre-aligned (lhotse) -> ${TRAIN_INPUT_CFG} ***" \
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
&& echo "Starting training (running MY code at /code, parakeet-tdt-0.6b-v2 reproduction, GRANARY 2.0 data)" \
&& python /code/examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    name=${EXP_NAME} \
    ${INIT_OVERRIDE} \
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
