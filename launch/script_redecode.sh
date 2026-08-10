#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:streaming-stt-script-redecode
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
# SCRIPT streaming SpeechLM finetune on OCI (draco-oci, IAD).
#
# Same data / infra as oci/streaming_stt_finetune_granary2_noblank_wer.sh, but
# trained with the SCRIPT objective via a different entrypoint + recipe:
#   entrypoint: examples/speechlm2/script_train.py
#               (ScriptSTTModel + ScriptSTTDataset)
#   recipe:     /code/examples/speechlm2/conf/streaming_stt_granary2_lora_script_redecode.yaml
#
# WINDOWED RE-DECODING self-correction variant. Replaces the <del> backspace
# token with re-decoding: each chunk is transcribed several times, each time with
# one more chunk of LOOKAHEAD audio, always conditioning on clean (believed-
# correct) history. Training adds, per chunk c, one branch per lookahead level
# j=0..R over an (M+1)-chunk audio window (j=0 is the base branch); inference
# emits each chunk immediately as a low-latency preview and LOCKS it R chunks
# later at maximal lookahead. No delete token, no synthetic-error manufacturing.
#   window N = model.audio_history_chunks + 1 (recipe: M=2 -> N=3 chunks)
#   depth  R = model.redecode_depth           (recipe: R=2 -> lock lag 2 chunks)
# See nemo/collections/speechlm2/docs/script_windowed_redecoding.md
#
# Keeps the BASELINE fixed-delay=3 operating point and chunk-size set (narrowed to
# [2,7,10,14]) so results stay comparable to older baseline numbers. Uses the newer
# prompt design (defined in the recipe): the prompt states the per-batch chunk size
# and the target text format (cap x punct, varied via vary_text_repr) and ends with
# the "The text history is:" connector. INITIALIZED from the BASELINE SCRIPT
# checkpoint (set INIT_CKPT below), i.e. redecode = baseline + windowed
# re-decoding; the j=0 branch is close to the baseline objective, so it warm-starts
# well.
#
# Runs MY code (git-synced repo mounted at /code), not the container's NeMo.
#
# Usage (optional random seed as $1; defaults to 42):
#   cd /lustre/fsw/portfolios/llmservice/users/hainanx/NeMo79
#   sbatch launch/script_redecode.sh          # seed 42
#   sbatch launch/script_redecode.sh 123      # seed 123
#
# Weights are AUTO-INITIALIZED from the best checkpoint of INIT_EXP
# (default granary2_script_baseline) -- no need to set anything. Override:
#   INIT_CKPT=/path/to.ckpt   (pin a checkpoint)
#   INIT_EXP=<other_exp>      (init from a different run)
#   INIT_CKPT=none            (train from the base pretrained LLM+ASR)
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


PROJECT_NAME=SpeechlmRefactored

# Training parameters (match the granary2 no-blank launcher overrides).
MAX_STEPS=300000
VAL_CHECK_INTERVAL=2000
# Fixed delay of 3 frames -- matches the older BASELINE results this is compared
# against (the redecode recipe uses no prompt-controlled / exact delay).
DELAY=3
LR=0.0001
WARMUP_STEPS=10000
COMPACT_TEMPLATE=true

# heh forces HF offline (models load from the recipe's absolute local paths,
# mounted via HEH_DIR). Set HF_HUB_OFFLINE=0 to allow hub downloads if needed.
HF_HUB_OFFLINE_FLAG="${HF_HUB_OFFLINE:-1}"

# Set DEBUG_CUDA=1 to run with CUDA_LAUNCH_BLOCKING=1 so a device-side assert
# reports the true failing op/line (default 0 = async, fast).
DEBUG_CUDA="${DEBUG_CUDA:-0}"
# Set DEBUG_VALIDATE_TOKENS=true to check input/target token ids are in range
# each step (turns an opaque CUDA assert into a clear Python error). Debug only.
DEBUG_VALIDATE_TOKENS="${DEBUG_VALIDATE_TOKENS:-false}"

# Config: OUR SCRIPT recipe, shipped in the synced repo at /code.
CONFIG_PATH=/code/examples/speechlm2/conf/
CONFIG_NAME=streaming_stt_granary2_lora_script_redecode

EXP_NAME=granary2_script_redecode

# Baseline operating point (CLI overrides on top of the recipe), matched to
# launch/script_baseline.sh so redecode is directly comparable to older results.
# One chunk size is drawn per batch from this set (encoder frames).
# 28 is dropped: with the M=2 re-decode window (3 chunks of audio per branch) a
# 14-frame chunk already gives the model up to ~42 frames of context, so the
# 28-frame regime is effectively covered by the window. 10 is added for a finer
# mid-range operating point.
CHUNK_SIZES="[2,7,10,14]"

# The prompt design lives entirely in the recipe (train_system_prompt + chunk-size
# clause + vary_text_repr format clause + "The text history is:" connector, with a
# byte-identical fully-rendered validation system_prompt). We deliberately do NOT
# override data.dataset.system_prompt here -- doing so would clobber the recipe's
# carefully-rendered validation prompt.

# Initialize weights from a good checkpoint of INIT_EXP (default: the BASELINE
# SCRIPT model, so redecode = baseline + windowed re-decoding). If INIT_CKPT is
# empty it is AUTO-RESOLVED below to that exp's best (lowest val_wer) checkpoint,
# so you normally don't set anything. Set INIT_CKPT=/path/to.ckpt to pin one,
# INIT_EXP=<other_exp> to init from a different run, or INIT_CKPT=none to train
# from the base pretrained LLM+ASR.
INIT_EXP="${INIT_EXP:-granary2_script_baseline}"
INIT_CKPT="${INIT_CKPT:-}"

# Directories for manifests, data, etc.
# Write-heavy outputs (results/checkpoints, HF cache, checkpoint temp) go to the
# nemotron project, which has free quota. The llmservice project that holds the
# synced code is at its space limit, so writes there fail with EDQUOT. Override
# the output location with OUTPUT_PREFIX.
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
RESULTS_DIR=${OUTPUT_PREFIX}/results/$PROJECT_NAME/$EXP_NAME

# Auto-resolve the init checkpoint from INIT_EXP when not given explicitly. Picks the
# best (lowest val_wer, from the "...val_wer=X.ckpt" name), else the newest, else the
# rolling -last. resume_if_exists=true means this only seeds the FIRST launch; later
# relaunches auto-resume this run's own checkpoints. INIT_CKPT=none => base pretrained.
if [[ -z "$INIT_CKPT" ]]; then
    _INIT_DIR="${OUTPUT_PREFIX}/results/${PROJECT_NAME}/${INIT_EXP}/${INIT_EXP}/checkpoints"
    INIT_CKPT="$(ls "${_INIT_DIR}"/*.ckpt 2>/dev/null | grep -v -- '-last\.ckpt$' | grep -v -- '-averaged\.ckpt$' | sort -t= -k3 -g | head -1)"
    [[ -z "$INIT_CKPT" ]] && INIT_CKPT="$(ls -t "${_INIT_DIR}"/*-last.ckpt 2>/dev/null | head -1)"
    if [[ -n "$INIT_CKPT" ]]; then
        echo "==> Auto-resolved INIT_CKPT from ${INIT_EXP}: ${INIT_CKPT}"
    else
        echo "WARNING: no checkpoint under ${_INIT_DIR}; training from base pretrained (set INIT_CKPT= or INIT_EXP=)."
    fi
fi
# The ckpt filename contains '=' (step=..-val_wer=..), which breaks Hydra override
# parsing, and it lives outside the container's mounted dirs. Expose it via a
# clean-named symlink under the (mounted) results dir, mount its source dir, and
# point resume_from_checkpoint at the symlink.
INIT_CKPT_ARG=""
INIT_MOUNT=""
if [[ -n "$INIT_CKPT" && "$INIT_CKPT" != "none" ]]; then
    mkdir -p "$RESULTS_DIR"
    ln -sfn "$INIT_CKPT" "${RESULTS_DIR}/init_from.ckpt"
    INIT_MOUNT="$(dirname "$INIT_CKPT")"
    INIT_CKPT_ARG="++exp_manager.resume_from_checkpoint=/results/init_from.ckpt"
fi
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

MOUNTS="--container-mounts=${DATA_DIR}:${DATA_DIR},${H_DIR}:${H_DIR},$HAINAN_DIR:$HAINAN_DIR,$CODE_DIR:/code,$RESULTS_DIR:/results,$DATA_DIR:/data,$PRETRAINED_MODEL_DIR:/pretrained,$CHECKPOINT_DIR:/checkpoints,${QUESTIONS_DIR}:/questions/,${HFCACHE}:/hfcache/,$DONGJI_ROOT:$DONGJI_ROOT${INIT_MOUNT:+,${INIT_MOUNT}:${INIT_MOUNT}}"

# SLURM_JOB_NUM_NODES=1
# GPUS_PER_NODE=1
# export CUDA_VISIBLE_DEVICES=0

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& echo "*** RECIPE: ${CONFIG_NAME} (SCRIPT, WINDOWED RE-DECODING, init=${INIT_CKPT:-none}, granary2, no-blank | delay=${DELAY} | vary_text_repr + chunk-size prompt | chunk-size ${CHUNK_SIZES}) ***" \
&& echo "*** OBJECTIVE: p(words_k | text_history_<k, audio_{k-M..k+j}); re-decode each chunk at lookahead 0..R on clean history ***" \
&& echo "*** MONITOR: val_wer (min) -- windowed re-decoding streaming decode (previews + R-chunk lock lag) ***" \
&& echo "*** SEED: ${LHOTSE_RND_SEED} ***" \
&& nvidia-smi \
&& export WANDB_API_KEY=${WANDB} \
&& cd /code \
&& echo "CODE COMMIT:" \
&& git rev-parse HEAD \
&& export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& python -c "import nemo, nemo.collections.speechlm2; print('USING NeMo FROM:', nemo.__file__)" \
&& python -c "from nemo.collections.speechlm2.models.script_model import ScriptSTTModel; print('SCRIPT model available')" \
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
&& echo "Starting training (running MY code at /code, SCRIPT streaming SpeechLM, GRANARY 2.0)" \
&& python /code/examples/speechlm2/script_train.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    ++model.compact_template=${COMPACT_TEMPLATE} \
    ++data.dataset.compact_template=${COMPACT_TEMPLATE} \
    ++model.debug_validate_tokens=$DEBUG_VALIDATE_TOKENS \
    model.optimizer.lr=$LR \
    model.lr_scheduler.warmup_steps=$WARMUP_STEPS \
    data.dataset.num_delay_frames=$DELAY \
    model.chunk_size="${CHUNK_SIZES}" \
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
    ++model.perception.encoder.sync_max_audio_length=false \
    ${INIT_CKPT_ARG} \


EOF

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"

# bash -c "${cmd}"

set +x
