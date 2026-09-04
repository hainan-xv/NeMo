#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:streaming-stt-chat
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 8
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00             # wall time
#SBATCH --time-min 04:00:00
#SBATCH --exclusive             # exclusive node access
#SBATCH --overcommit
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only email on failure
#SBATCH --ntasks-per-node=8     # one task per GPU !!! SET TO NUMBER OF GPUs PER NODE !!!
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# CHAT transducer trained on a FORCED ALIGNMENT -- two vocabulary arms.
#
# THIS script is arm 1, the ~1,024-piece ASR vocabulary. Arm 2 (151,936-piece
# Qwen) is launch/chat_train_qwen.sh, a thin wrapper that sets CONFIG_NAME and
# EXP_NAME and execs this file, so the recipe below is shared verbatim by both
# and the arms cannot drift apart. Neither takes arguments:
#
#   sbatch launch/chat_train.sh          # arm 1: ~1k ASR vocabulary
#   sbatch launch/chat_train_qwen.sh     # arm 2: 151,936 Qwen vocabulary
#
# (or the same paths via ./oci_launch.sh from a local checkout.)
#
# Anything changed here therefore applies to BOTH arms -- which is the point.
# Anything that should apply to only one belongs in the recipe pair
# examples/speechlm2/conf/streaming_stt_granary2_chat_{asrvocab,qwenvocab}.yaml,
# currently a 3-line diff.
#
# The arms differ ONLY in the tokenizer, which is the point: the SpeechLM lost
# just ~0.14 WER going from 151,936 pieces to 1,024, and whether a transducer
# behaves the same way is the open question.
#
# WHY FORCED ALIGNMENT. The RNN-T loss marginalises over alignments and needs a
# [B, T, U, V] tensor; at V=151,936 that does not fit, so the large-vocabulary
# arm could not be trained at all. Each word is instead fixed to the chunk
# holding its last token -- the same assignment, delay and tokenisation the
# SpeechLM uses, from the same dataset class -- reducing the loss to
# cross-entropy over U + T steps.
#
# The fixed alignment is not a handicap: the SpeechLM trains the same way and is
# competitive with the RNN-T baseline, and CHAT's greedy decode walks chunks
# emitting until a blank -- the same procedure as training, so no train/test
# mismatch. What is fixed is emission latency (aligned chunk + delay).
#
# Trains from the pretrained ASR encoder (INIT_CKPT=none): no SCRIPT checkpoint
# is shape-compatible with a transducer.
# ============================================================================

# Secrets live only in token files on the OCI login node, each holding just the
# token on one line and readable only by the owner:
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

# Lhotse dataloader seed (optional positional arg).
LHOTSE_RND_SEED="${1:-42}"

# Do NOT enable xtrace: the command below contains expanded token values.
mkdir -p slurm_out

GPUS_PER_NODE=8
SLURM_ACCOUNT='llmservice'
OLDUSERID='users/heh'
LUSTRE_ACCOUNT_PREFIX=/lustre/fsw/portfolios/${SLURM_ACCOUNT}

# We use the container only for its environment; the NeMo code comes from /code.
CONTAINER="/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh"

PROJECT_NAME=SpeechlmScriptCC

# --- Training parameters ---
MAX_STEPS="${MAX_STEPS:-300000}"
# 16000, 4x the SCRIPT recipe's 4000. This drives BOTH limit_train_batches (the
# "epoch" length) and val_check_interval below, so it is the whole train/validate
# cycle. CHAT steps ~5 it/s against the SpeechLM's much slower step -- it has no
# 1.7B LLM in the loop -- so a 4000-step epoch meant validating and
# checkpointing every ~14 minutes, which is mostly overhead. Per-step decode
# visibility is now LOG_TRAIN_DECODE_EVERY's job instead, so the long cycle
# costs no insight.
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-16000}"

# Print ref / forced-alignment target / greedy hypothesis on TRAINING data every
# N steps (0 disables). Cheap: one greedy decode of LOG_TRAIN_DECODE_N
# utterances on rank 0.
LOG_TRAIN_DECODE_EVERY="${LOG_TRAIN_DECODE_EVERY:-500}"
LOG_TRAIN_DECODE_N="${LOG_TRAIN_DECODE_N:-2}"
LR="${LR:-0.0001}"
# 2500, the recipe's own value. The previous 10000 was inherited from the SCRIPT
# launcher, where a 1.7B LLM genuinely needs a long warmup. CHAT has no LLM, and
# with a 16000-step epoch a 10000-step warmup meant most of "epoch 0" ran at a
# fraction of the target learning rate -- which on its own makes convergence look
# far slower than a normal RNN-T fine-tune.
WARMUP_STEPS="${WARMUP_STEPS:-2500}"

# --- SCRIPT operating point ---
DELAY="${DELAY:-3}"
# dense | flex | script -- all mathematically identical; flex is fastest.
ATTN_BACKEND="${ATTN_BACKEND:-flex}"
ACT_CKPT="${ACT_CKPT:-true}"
AUDIO_HISTORY_CHUNKS="${AUDIO_HISTORY_CHUNKS:-0}"
CHUNK_SIZES="${CHUNK_SIZES:-[2,4,7,10,14,28]}"
# Apostrophe-free by construction: the Hydra override wraps it in single quotes.
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are doing streaming speech recognition. Given the transcript so far and the representation of the next audio chunk, output the words spoken in that chunk.}"

CONFIG_PATH=/code/examples/speechlm2/conf/
CONFIG_NAME="${CONFIG_NAME:-streaming_stt_granary2_chat_asrvocab}"
EXP_NAME="${EXP_NAME:-granary2_chat_asrvocab}"

# --- Tag runs that use a non-default node count ---
# RESULTS_DIR is derived from EXP_NAME and the recipe sets resume_if_exists=true,
# so a scaled-down run (e.g. the 1-node interactive debug run submitted by
# oci_launch_interactive.sh, which overrides --nodes) sharing an EXP_NAME with the
# full-scale run would RESUME FROM and then OVERWRITE that run's checkpoints, and
# collide with its wandb run. Append _n<N> whenever the allocation differs from
# what this script's own "#SBATCH -N" asks for, so the two never touch.
# Read back from the header rather than hardcoding, so the two cannot drift.
# Escape hatch: SKIP_NODE_SUFFIX=1 (e.g. to deliberately resume a run at a new scale).
DESIGN_NODES="$(grep -m1 -E '^#SBATCH[[:space:]]+-N[[:space:]]+[0-9]+' "$0" 2>/dev/null | grep -oE '[0-9]+$' || true)"
DESIGN_NODES="${DESIGN_NODES:-8}"
ACTUAL_NODES="${SLURM_JOB_NUM_NODES:-$DESIGN_NODES}"
if [[ "${SKIP_NODE_SUFFIX:-0}" != "1" && "$ACTUAL_NODES" -ne "$DESIGN_NODES" ]]; then
    EXP_NAME="${EXP_NAME}_n${ACTUAL_NODES}"
    echo "==> Allocation is ${ACTUAL_NODES} node(s), not the designed ${DESIGN_NODES}; EXP_NAME -> ${EXP_NAME}"
fi

# Set HF_HUB_OFFLINE=0 to allow hub downloads (models otherwise load from the
# absolute local paths in the recipe).
HF_HUB_OFFLINE_FLAG="${HF_HUB_OFFLINE:-1}"
# DEBUG_CUDA=1 runs with CUDA_LAUNCH_BLOCKING so a device-side assert reports the
# true failing op/line (default 0 = async, fast).
DEBUG_CUDA="${DEBUG_CUDA:-0}"

# --- Paths ---
# Write-heavy outputs go to the nemotron portfolio: the llmservice portfolio that
# holds the synced code is at its quota limit and writes there fail with EDQUOT.
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
RESULTS_DIR=${OUTPUT_PREFIX}/results/$PROJECT_NAME/$EXP_NAME

PRETRAINED_MODEL_DIR=${LUSTRE_ACCOUNT_PREFIX}/${OLDUSERID}/pretrained_models
CHECKPOINT_DIR=${LUSTRE_ACCOUNT_PREFIX}/${OLDUSERID}/checkpoints/
QUESTIONS_DIR=${LUSTRE_ACCOUNT_PREFIX}/${OLDUSERID}/questions
HFCACHE=${OUTPUT_PREFIX}/hf_cache
DATA_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh
HAINAN_DIR=/lustre/fsw/portfolios/llmservice/users/hainanx
DONGJI_ROOT=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/dongjig
# The synced repo (git-synced via sync_to_oci.sh) -> mounted as /code.
# Keep in sync with OCI_REPO in sync_to_oci.sh.
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_SCRIPT_cc}"
# Stage checkpoint temp files on the lustre results filesystem (same device as
# the checkpoint destination), not the container's small /tmp.
OCI_TMP_DIR="${OCI_TMP_DIR:-/results/tmp}"

mkdir -p ${RESULTS_DIR} ${HFCACHE}
OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

# --- Optional warm start ---
# Empty INIT_CKPT with a set INIT_EXP auto-resolves that run's latest checkpoint.
# INIT_CKPT=none trains from the base pretrained LLM + ASR. resume_if_exists=true
# means this only seeds the FIRST launch; relaunches resume this run's own ckpts.
INIT_EXP="${INIT_EXP:-}"
INIT_CKPT="${INIT_CKPT:-none}"
if [[ -z "$INIT_CKPT" && -n "$INIT_EXP" ]]; then
    _INIT_DIR="${OUTPUT_PREFIX}/results/${PROJECT_NAME}/${INIT_EXP}/${INIT_EXP}/checkpoints"
    INIT_CKPT="$(ls -t "${_INIT_DIR}"/*-last.ckpt 2>/dev/null | head -1)"
    [[ -z "$INIT_CKPT" ]] && INIT_CKPT="$(ls "${_INIT_DIR}"/*.ckpt 2>/dev/null | grep -v -- '-averaged\.ckpt$' | sort -t= -k2 -g | tail -1)"
    [[ -z "$INIT_CKPT" ]] && INIT_CKPT="$(ls -t "${_INIT_DIR}"/*.ckpt 2>/dev/null | head -1)"
    if [[ -n "$INIT_CKPT" ]]; then
        echo "==> Auto-resolved INIT_CKPT from ${INIT_EXP}: ${INIT_CKPT}"
    else
        echo "WARNING: no checkpoint under ${_INIT_DIR}; training from base pretrained."
    fi
fi

# A warm start restores the FULL training state, global_step included. If the
# parent is already at or past MAX_STEPS this run has nothing to do: Lightning
# prints "max_steps reached", exits 0 after ~3 minutes, and Slurm reports
# COMPLETED with no checkpoints -- a failure that looks like a success. Job
# 12865606 died exactly this way. Fail loudly instead.
if [[ -n "$INIT_CKPT" && "$INIT_CKPT" != "none" ]]; then
    _INIT_STEP="$(basename "$INIT_CKPT" | grep -oE 'step=[0-9]+' | head -1 | cut -d= -f2)"
    if [[ -n "$_INIT_STEP" && "$_INIT_STEP" -ge "$MAX_STEPS" ]]; then
        echo "ERROR: warm-start checkpoint is at step ${_INIT_STEP}, but MAX_STEPS=${MAX_STEPS}." >&2
        echo "       resume_from_checkpoint restores global_step, so training would stop" >&2
        echo "       immediately and the job would exit 0 having trained nothing." >&2
        echo "       Raise it:  MAX_STEPS=$((_INIT_STEP + 100000)) ./oci_launch.sh <this script>" >&2
        exit 1
    fi
fi
# Checkpoint filenames contain '=' (step=..-val_wer=..), which breaks Hydra
# override parsing, and they live outside the mounted dirs. Expose one through a
# clean-named symlink under the (mounted) results dir and mount its source dir.
INIT_CKPT_ARG=""
INIT_MOUNT=""
if [[ -n "$INIT_CKPT" && "$INIT_CKPT" != "none" ]]; then
    ln -sfn "$INIT_CKPT" "${RESULTS_DIR}/init_from.ckpt"
    INIT_MOUNT="$(dirname "$INIT_CKPT")"
    INIT_CKPT_ARG="++exp_manager.resume_from_checkpoint=/results/init_from.ckpt"
fi

MOUNTS="--container-mounts=${DATA_DIR}:${DATA_DIR},${H_DIR}:${H_DIR},$HAINAN_DIR:$HAINAN_DIR,$CODE_DIR:/code,$RESULTS_DIR:/results,$DATA_DIR:/data,$PRETRAINED_MODEL_DIR:/pretrained,$CHECKPOINT_DIR:/checkpoints,${QUESTIONS_DIR}:/questions/,${HFCACHE}:/hfcache/,$DONGJI_ROOT:$DONGJI_ROOT${INIT_MOUNT:+,${INIT_MOUNT}:${INIT_MOUNT}}"

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "*** RECIPE: ${CONFIG_NAME} (SCRIPT flex-attention baseline | delay=${DELAY} | attn=${ATTN_BACKEND} ckpt=${ACT_CKPT} | chunk sizes ${CHUNK_SIZES}) ***" \
&& echo "*** OBJECTIVE: p(words_k | text_history_<k, audio_k); packed spine+branch, single O(L) forward ***" \
&& echo "*** MONITOR: val_wer (min) -- chunk-synchronous streaming decode ***" \
&& echo "*** WARM START: init=${INIT_CKPT:-none} ***" \
&& echo "*** SEED: ${LHOTSE_RND_SEED} ***" \
&& nvidia-smi \
&& export WANDB_API_KEY=${WANDB} \
&& cd /code \
&& echo "CODE COMMIT:" \
&& git rev-parse HEAD \
&& export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& python -c "import nemo, nemo.collections.speechlm2; print('USING NeMo FROM:', nemo.__file__)" \
&& python -c "from nemo.collections.speechlm2 import ScriptSTTModel; print('SCRIPT model available')" \
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
&& echo "Starting CHAT forced-alignment training (repo at /code, GRANARY 2.0)" \
&& python /code/examples/speechlm2/chat_train.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    model.optimizer.lr=$LR \
    model.lr_scheduler.warmup_steps=$WARMUP_STEPS \
    data.dataset.num_delay_frames=${DELAY} \
    ++model.log_train_decode_every_n_steps=${LOG_TRAIN_DECODE_EVERY} \
    ++model.log_train_decode_examples=${LOG_TRAIN_DECODE_N} \
    data.train_ds.seed=$LHOTSE_RND_SEED \
    ++trainer.limit_train_batches=$VAL_CHECK_INTERVAL \
    ++trainer.val_check_interval=$VAL_CHECK_INTERVAL \
    trainer.max_steps=$MAX_STEPS \
    trainer.devices=$GPUS_PER_NODE \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
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
    ${INIT_CKPT_ARG}
EOF

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
