#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:streaming-stt-g2-nb-wlenstoch
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
# Streaming SpeechLM (streaming_stt) finetune on OCI (draco-oci, IAD).
#
# Same setup as streaming_stt_finetune_granary2_noblank_wer_wlendelay.sh, but the
# per-word emission delay is drawn STOCHASTICALLY instead of from the fixed
# word-length table. Each word's delay ~ Binomial(word_delay_max=3, p), with
#   p = sigmoid((midpoint - n_letters) / slope)   (midpoint=4.5, slope=1.0)
# so shorter words get a higher expected delay (2L~2.8, 3L~2.5, 4L~1.9, 5L~1.1,
# 6L~0.6, 7L~0.2 frames), while the full range {0..3} keeps non-zero mass for
# augmentation diversity. delay >= 0 always, so token monotonicity is preserved.
#
# All other knobs match the fixed-delay launcher (recipe owns modeling; this
# launcher passes heh's training-loop overrides + val_wer monitoring).
#
# Usage (optional random seed as $1; defaults to 42):
#   cd /lustre/fsw/portfolios/llmservice/users/hainanx/NeMo79
#   sbatch oci/streaming_stt_finetune_granary2_noblank_wer_wlendelay_stochastic.sh          # seed 42
#   sbatch oci/streaming_stt_finetune_granary2_noblank_wer_wlendelay_stochastic.sh 123      # seed 123
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

# Random seed for the Lhotse dataloader (heh takes this as $1). Optional here.
LHOTSE_RND_SEED="${1:-42}"

# Do not enable xtrace: the command below contains expanded token values.
mkdir -p slurm_out
CLUSTER="oci"
GPUS_PER_NODE=8
SLURM_ACCOUNT='llmservice'
OLDUSERID='users/heh'
USERID='users/hainanx'
LUSTRE_ACCOUNT_PREFIX=/lustre/fsw/portfolios/${SLURM_ACCOUNT}

# Latest container with the current SpeechLM deps (we only use its environment;
# the actual NeMo code comes from /code below).
CONTAINER="gitlab-master.nvidia.com/hainanx/nemo_containers:speechlm_heh"
CONTAINER="/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh"


PROJECT_NAME=Speechlm79

# Training parameters (match heh's granary2 launcher overrides).
MAX_STEPS=200000
VAL_CHECK_INTERVAL=2000
DELAY=3
LR=0.0001
WARMUP_STEPS=10000
COMPACT_TEMPLATE=true

# Stochastic word-length delay hyperparameters (see recipe defaults). Override
# from the environment to sweep the schedule without editing the recipe.
WORD_DELAY_MAX="${WORD_DELAY_MAX:-3}"
WORD_DELAY_MIDPOINT="${WORD_DELAY_MIDPOINT:-4.5}"
WORD_DELAY_SLOPE="${WORD_DELAY_SLOPE:-1.0}"

# heh forces HF offline (models load from the recipe's absolute local paths,
# mounted via HEH_DIR). Set HF_HUB_OFFLINE=0 to allow hub downloads if needed.
HF_HUB_OFFLINE_FLAG="${HF_HUB_OFFLINE:-1}"

# Set DEBUG_CUDA=1 to run with CUDA_LAUNCH_BLOCKING=1 so a device-side assert
# reports the true failing op/line (default 0 = async, fast).
DEBUG_CUDA="${DEBUG_CUDA:-0}"
# Set DEBUG_VALIDATE_TOKENS=true to check input/target token ids are in range
# each step (turns an opaque CUDA assert into a clear Python error). Debug only.
DEBUG_VALIDATE_TOKENS="${DEBUG_VALIDATE_TOKENS:-false}"

# Config: OUR recipe, shipped in the synced repo at /code.
CONFIG_PATH=/code/examples/speechlm2/conf/
CONFIG_NAME=streaming_stt_granary2_lora_noblank_wer

EXP_NAME=granary2_noblank_wer_wlendelay_stochastic

# Warm start: full-state resume from the most recent checkpoint of the FIXED
# word-length-delay run (same recipe/architecture). This restores weights +
# optimizer + LR scheduler + global step, so the stochastic run CONTINUES from
# where the fixed-delay run left off (no fresh LR warmup). Because the recipe
# sets resume_if_exists=true + resume_ignore_no_checkpoint=true, this init
# checkpoint is only used on the FIRST launch; after any preemption the run
# resumes from its OWN -last.ckpt instead. Override the source run with SRC_EXP,
# or point at an exact file with INIT_CKPT_HOST.
SRC_EXP="${SRC_EXP:-granary2_noblank_wer_wlendelay}"

# Directories for manifests, data, etc.
# Write-heavy outputs (results/checkpoints, HF cache, checkpoint temp) go to the
# nemotron project, which has free quota. The llmservice project that holds the
# synced code is at its space limit, so writes there fail with EDQUOT. Override
# the output location with OUTPUT_PREFIX.
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
RESULTS_DIR=${OUTPUT_PREFIX}/results/$PROJECT_NAME/$EXP_NAME
PRETRAINED_MODEL_DIR=${LUSTRE_ACCOUNT_PREFIX}/${OLDUSERID}/pretrained_models
CHECKPOINT_DIR=${LUSTRE_ACCOUNT_PREFIX}/${OLDUSERID}/checkpoints/
QUESTIONS_DIR=${LUSTRE_ACCOUNT_PREFIX}/${OLDUSERID}/questions
HFCACHE=${OUTPUT_PREFIX}/hf_cache
DATA_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh
HAINAN_DIR=/lustre/fsw/portfolios/llmservice/users/hainanx
# MY code (synced via sync_to_oci.sh) -> mounted as /code.
CODE_DIR=${LUSTRE_ACCOUNT_PREFIX}/${USERID}/NeMo79
CODE_DIR=/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo79/

# Stage checkpoint/restore temp files on the lustre results filesystem (same
# device as the checkpoint destination), not the container's small /tmp.
OCI_TMP_DIR="${OCI_TMP_DIR:-/results/tmp}"

# Make results + HF cache dirs (on the nemotron output filesystem)
mkdir -p ${RESULTS_DIR} ${HFCACHE}
OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

DONGJI_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/dongjig/aligned_amos/granary_v2_en_pnc_qwen_aligned_filtered
# The recipe's validation manifest lives under dongjig's steve_val dir.
DONGJI_ROOT=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/dongjig

# Resolve the fixed-delay run's most-recent checkpoint (host path) at submit
# time. Prefer the rolling *-last.ckpt (latest full training state); fall back
# to the newest *.ckpt. Its directory is mounted read-only into /init_ckpt.
SRC_CKPT_DIR_HOST="${OUTPUT_PREFIX}/results/${PROJECT_NAME}/${SRC_EXP}/${SRC_EXP}/checkpoints"
if [[ -z "${INIT_CKPT_HOST:-}" ]]; then
    INIT_CKPT_HOST=$(ls -t "${SRC_CKPT_DIR_HOST}"/*-last.ckpt 2>/dev/null | head -1)
    if [[ -z "${INIT_CKPT_HOST}" ]]; then
        INIT_CKPT_HOST=$(ls -t "${SRC_CKPT_DIR_HOST}"/*.ckpt 2>/dev/null | head -1)
    fi
fi
if [[ -z "${INIT_CKPT_HOST}" || ! -f "${INIT_CKPT_HOST}" ]]; then
    echo "ERROR: no init checkpoint found under ${SRC_CKPT_DIR_HOST}" >&2
    echo "       Train the fixed-delay run first, set SRC_EXP=<run>, or set" >&2
    echo "       INIT_CKPT_HOST=/abs/path/to/checkpoint.ckpt to override." >&2
    exit 1
fi
INIT_CKPT_DIR_HOST=$(dirname "${INIT_CKPT_HOST}")
INIT_CKPT_BASENAME=$(basename "${INIT_CKPT_HOST}")
INIT_CKPT_CONTAINER="/init_ckpt/${INIT_CKPT_BASENAME}"
echo "==> Warm start (full resume) from: ${INIT_CKPT_HOST}"

MOUNTS="--container-mounts=${DATA_DIR}:${DATA_DIR},${H_DIR}:${H_DIR},$HAINAN_DIR:$HAINAN_DIR,$CODE_DIR:/code,$RESULTS_DIR:/results,$DATA_DIR:/data,$PRETRAINED_MODEL_DIR:/pretrained,$CHECKPOINT_DIR:/checkpoints,${QUESTIONS_DIR}:/questions/,${HFCACHE}:/hfcache/,$DONGJI_ROOT:$DONGJI_ROOT,${INIT_CKPT_DIR_HOST}:/init_ckpt:ro"

# SLURM_JOB_NUM_NODES=1
# GPUS_PER_NODE=1
# export CUDA_VISIBLE_DEVICES=0

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& echo "*** RECIPE: ${CONFIG_NAME} (granary2, no-blank, multi chunk-size [2,4,7,10,14,28]) ***" \
&& echo "*** ALIGNMENT: STOCHASTIC word-length delay ~ Binomial(${WORD_DELAY_MAX}, sigmoid((${WORD_DELAY_MIDPOINT}-n_letters)/${WORD_DELAY_SLOPE})) -- fixed num_delay_frames ignored ***" \
&& echo "*** WARM START (full resume): ${INIT_CKPT_CONTAINER} (from ${SRC_EXP}) -- only on first launch; own -last.ckpt after preemption ***" \
&& echo "*** MONITOR: val_wer (min) -- our code replaces val_acc ***" \
&& echo "*** SEED: ${LHOTSE_RND_SEED} ***" \
&& nvidia-smi \
&& export WANDB_API_KEY=${WANDB} \
&& cd /code \
&& echo "CODE COMMIT:" \
&& git rev-parse HEAD \
&& export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& python -c "import nemo, nemo.collections.speechlm2; print('USING NeMo FROM:', nemo.__file__)" \
&& python -c "import nemo.collections.speechlm2.data.streaming_stt_dataset as d; print('stochastic word-length delay supported:', hasattr(d, 'sample_word_length_delay_frames'))" \
&& pip show torch \
&& export OMP_NUM_THREADS=1 \
&& export HF_HOME="/hfcache/" \
&& export HF_TOKEN=${HF_TOKEN} \
&& export HF_HUB_OFFLINE=${HF_HUB_OFFLINE_FLAG} \
&& export HYDRA_FULL_ERROR=1 \
&& export CUDA_LAUNCH_BLOCKING=${DEBUG_CUDA} \
&& export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
&& export TMPDIR=${OCI_TMP_DIR} && mkdir -p ${OCI_TMP_DIR} && echo "staging TMPDIR=\$TMPDIR" \
&& export AIS_ENDPOINT=http://asr.iad.oci.aistore.nvidia.com:51080 \
&& export AIS_AUTHN_TOKEN="${AIS_AUTHN_TOKEN}" \
&& export NEMO_DATA_STORE_CACHE_DIR=/lustre/fsw/portfolios/llmservice/users/heh/nemo_cache \
&& echo "Starting training (running MY code at /code, GRANARY 2.0 no-blank multi-chunk, val_wer, STOCHASTIC word-length delay)" \
&& python /code/examples/speechlm2/streaming_stt_train.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    ++model.compact_template=${COMPACT_TEMPLATE} \
    ++data.dataset.compact_template=${COMPACT_TEMPLATE} \
    ++data.dataset.use_word_length_delay=true \
    ++data.dataset.word_length_delay_stochastic=true \
    ++data.dataset.word_delay_max=${WORD_DELAY_MAX} \
    ++data.dataset.word_delay_midpoint=${WORD_DELAY_MIDPOINT} \
    ++data.dataset.word_delay_slope=${WORD_DELAY_SLOPE} \
    ++model.debug_validate_tokens=$DEBUG_VALIDATE_TOKENS \
    model.optimizer.lr=$LR \
    model.lr_scheduler.warmup_steps=$WARMUP_STEPS \
    data.dataset.num_delay_frames=$DELAY \
    data.train_ds.seed=$LHOTSE_RND_SEED \
    ++trainer.limit_train_batches=$VAL_CHECK_INTERVAL \
    ++trainer.val_check_interval=$VAL_CHECK_INTERVAL \
    trainer.max_steps=$MAX_STEPS \
    trainer.devices=$GPUS_PER_NODE \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES  \
    trainer.log_every_n_steps=10 \
    ++exp_manager.exp_dir=/results/ \
    ++exp_manager.create_wandb_logger=true \
    ++exp_manager.create_tensorboard_logger=false \
    ++exp_manager.max_time_per_run=00:03:55:00 \
    ++exp_manager.name=$EXP_NAME \
    ++exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
    ++exp_manager.checkpoint_callback_params.monitor=val_wer \
    ++exp_manager.checkpoint_callback_params.mode=min \
    ++exp_manager.checkpoint_callback_params.save_top_k=5 \
    ++exp_manager.resume_from_checkpoint=${INIT_CKPT_CONTAINER} \
    ++model.perception.encoder.sync_max_audio_length=false \


EOF

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"

# bash -c "${cmd}"

set +x
