#!/bin/bash
# DFW: OFFLINE encoder-free ASR (SALM), 8 nodes.
#
# The simpler task. SCRIPT has to learn acoustics from scratch AND do it
# under chunked streaming with a re-prefilled text history; SALM drops the
# chunking, the history and the latency constraint, so this isolates the one
# question that matters: can the LLM read mel features at all?
#
# FAIR BY CONSTRUCTION. Same front-end as the best streaming arm (val_wer
# 0.193): StackingEncoder, normalize per_feature, 1024-8192-2048 MLP,
# SpecAugment, nemotron as pretrained_asr. SALM resolves perception through
# setup_speech_encoder, which now SHARES resolve_perception_sources() with
# the streaming path -- before that refactor these keys silently did nothing
# here. Same granary2 data, same buckets, same max_duration 54, same nodes.
#
# Data note: read_lhotse_as_conversation delegates to read_cutset_from_config,
# which honours a nested `input_cfg` -- so the granary2 config loads normally
# and is then wrapped as conversations.
#
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-salm-nocoder
# DFW's default GPU partition. Unlike OCI there is ONE pool of 1850 nodes rather
# than batch_block1/3/4, so no comma-list is needed.
#SBATCH -p batch
#SBATCH -N 8
# Slurm requeues on NODE_FAIL by default here. That turned one retry into
# three simultaneous jobs writing the same checkpoint dir. Retry explicitly.
#SBATCH --no-requeue
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 04:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=8
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
# Known-bad node, excluded by the reference DFW recipe.
# pool0-01815 carries an old NVIDIA driver (12020) and fails torch's CUDA init
# outright; it killed job 18686485 in 103s. pool0-00407 is the reference recipe's
# known-bad node.
#SBATCH --exclude=pool0-00407,pool0-01815,pool0-01764


# ============================================================================
# OFFLINE SALM -- NO chunking, NO delay, NO prompt control.
#
# This launcher was forked from the streaming SCRIPT recipe and originally
# carried that recipe's whole knob set (DELAY, CHUNK_SIZES, DELAY_CANDIDATES,
# CAP_PROB/PUNCT_PROB, band/branch grouping, VAL_CHUNK_SIZE). None of them were
# ever passed to salm_train.py -- salm.yaml has no such keys -- so they were
# dead text that only made the RECIPE banner lie. They are gone. What remains
# below is exactly what this job overrides.
#
# Everything that makes the comparison fair lives in salm_nocoder.yaml:
# StackingEncoder, normalize per_feature, the 1024-8192-2048 MLP, SpecAugment,
# nemotron as pretrained_asr. Here we only match the DATA and the scale to the
# streaming arms: same granary2 input_cfg, same buckets, same max_duration 54,
# same 8 nodes.
# ============================================================================

set -uo pipefail
mkdir -p slurm_out

CLUSTER=dfw
GPUS_PER_NODE=8
LUSTRE=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
HEH=${LUSTRE}/users/heh
MYDIR=${LUSTRE}/hainanx
CONTAINER="${CONTAINER:-${HEH}/containers/nemo-26.02-streaming-speechlm.sqsh}"
CODE_DIR="${CODE_DIR:-${MYDIR}/NeMo_SCRIPT_cc}"
PROJECT_NAME="${PROJECT_NAME:-SpeechlmDFW}"
CONFIG_PATH=/code/examples/speechlm2/conf
CONFIG_NAME="${CONFIG_NAME:-salm_nocoder}"
EXP_NAME="${EXP_NAME:-dfw_salm_nocoder_8node}"

# From-scratch schedule: this arm has no warm start, so 1e-4 with a long warmup
# is the right shape, not the 5e-5 a warm start would want.
LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-10000}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-2000}"
MAX_STEPS="${MAX_STEPS:-500000}"
CREATE_WANDB="${CREATE_WANDB:-true}"

# Matched to the streaming arms so the offline/streaming comparison is about the
# task, not the data distribution.
MAX_DURATION="${MAX_DURATION:-54}"
BUCKET_BINS="${BUCKET_BINS:-[4.32,6.0,7.04,7.92,8.8,9.6,10.4,11.12,11.89,12.66,13.47,14.8,16.92,20.0,22.4,29.44,35.0,39.6]}"
BUCKET_BATCH_SIZE="${BUCKET_BATCH_SIZE:-[38,29,25,22,20,18,17,15,14,13,12,11,10,8,7,6,5,4]}"

# 4, not the YAML's 8. DFW nodes have 128 CPUs against OCI's 248, so 8 ranks x 8
# workers = 64 readers starved the loader. Lowering this is what made DFW train.
NUM_WORKERS="${NUM_WORKERS:-4}"

# No warm start: the point of this arm is whether an LLM can read mel features
# at all, starting from the base LLM and an untrained projector.
INIT_CKPT="${INIT_CKPT:-none}"

# Safe to resubmit: auto-resubmission re-runs this script verbatim, and INIT_CKPT
# is consumed as exp_manager.resume_from_checkpoint, which restores the PARENT's
# global_step. Warm-start only until this run has a -last.ckpt of its own.
_OWN="${OUTPUT_PREFIX:-$MYDIR}/results/${PROJECT_NAME}/${EXP_NAME}/${EXP_NAME}/checkpoints"
if compgen -G "${_OWN}/*-last.ckpt" >/dev/null 2>&1; then
    echo "==> ${EXP_NAME} has its own checkpoints; ignoring the warm start"
    INIT_CKPT="none"
fi

# DFW-side data and model paths. These are the ONLY substantive config
# differences from the OCI arm, so they are overrides rather than a forked YAML
# -- a second config would drift, and the model/dataset mirror asserts in
# script_train.py exist precisely because that drift is hard to see.
PRETRAINED_LLM=${HEH}/pretrained_models/huggingface/Qwen/Qwen3-1.7B
PRETRAINED_ASR=${HEH}/pretrained_models/huggingface/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo
TRAIN_INPUT_CFG=${HEH}/data_configs/granary_v2_en_full_d0.5_b0.5_dfw_qwen_aligned.yaml
VAL_MANIFEST=${HEH}/data/mcv11_en_dev_aligned/mcv11_dev_clean_pcstrip_en_2k_qwen_aligned.json

# A 1-node allocation writes to a DIFFERENT EXP_NAME so an interactive probe can
# never resume from, or overwrite, the real 8-node run's checkpoints.
DESIGN_NODES="$(grep -m1 -E '^#SBATCH[[:space:]]+-N[[:space:]]+[0-9]+' "$0" 2>/dev/null | grep -oE '[0-9]+$' || true)"
DESIGN_NODES="${DESIGN_NODES:-2}"
ACTUAL_NODES="${SLURM_JOB_NUM_NODES:-$DESIGN_NODES}"
if [[ "${SKIP_NODE_SUFFIX:-0}" != "1" && "$ACTUAL_NODES" -ne "$DESIGN_NODES" ]]; then
    EXP_NAME="${EXP_NAME}_n${ACTUAL_NODES}"
    echo "==> Allocation is ${ACTUAL_NODES} node(s), not the designed ${DESIGN_NODES}; EXP_NAME -> ${EXP_NAME}"
fi

# Hydra's override grammar cannot parse a VALUE containing "=", and every
# checkpoint this project writes is named "step=NNNN-val_wer=N.NNNN-last.ckpt".
# Passing the path directly dies with "mismatched input '=' expecting <EOF>"
# before the model is even built. A symlink under a clean name is the simplest
# thing that cannot be broken by a future filename scheme.
INIT_ARG=""
if [[ "${INIT_CKPT}" == "none" || -z "${INIT_CKPT}" ]]; then
    echo "==> NO warm start: SCRIPT state trains from scratch (base LLM + ASR encoder still pretrained)"
else
    INIT_LINK=${MYDIR}/init_ckpts/${EXP_NAME}_init.ckpt
    mkdir -p "$(dirname "$INIT_LINK")"
    if [[ ! -e "$INIT_CKPT" ]]; then
        echo "ERROR: warm-start checkpoint not found: ${INIT_CKPT}" >&2
        exit 1
    fi
    ln -sfn "$INIT_CKPT" "$INIT_LINK"
    echo "==> warm start: ${INIT_LINK} -> ${INIT_CKPT}"
    INIT_ARG="++init_from_ckpt=${INIT_LINK}"
fi

RESULTS_DIR=${MYDIR}/results/${PROJECT_NAME}/${EXP_NAME}
HFCACHE=${MYDIR}/hf_cache
mkdir -p "$RESULTS_DIR" "$HFCACHE"
OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

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

LHOTSE_RND_SEED="${1:-42}"

# IDENTITY mounts of both project roots the data config references, because the
# manifests and tar paths inside it are ABSOLUTE lustre paths -- they have to
# resolve to the same string inside the container as outside.
#
# There are exactly two roots, confirmed by scanning the input_cfg rather than
# assumed: the manifests live under nemotron_speechprod_asr/users/heh, and the
# AUDIO TARS live under llmservice_nemo_speechlm. Mounting only the first is what
# failed job 18615569 -- it sailed through the sanity check (which uses the mcv11
# val manifest, a plain .json under the mounted root) and then died on the first
# TRAINING batch with FileNotFoundError on .../ASR/YTC/en12/audio_52.tar.
DATA_ROOT=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm

# THIRD root, and it is NOT optional. The inherited comment claimed the
# input_cfg needs exactly two roots; scanning the file says otherwise:
#   457 paths under /lustre/fsw/portfolios/llmservice   (DATA_ROOT)
#    54 paths under /lustre/fsw/portfolios/nemotron/users/heh/manifests
# ${LUSTRE} is .../nemotron/PROJECTS/nemotron_speechprod_asr, which does NOT
# cover .../nemotron/USERS. The sharded manifests live there, exist on the
# host, and were simply invisible inside the container -- the job reached the
# training loop and then died on the first shard read with
#   FileNotFoundError: .../nemotron/users/heh/manifests/.../manifest_639.jsonl
# This is the sharded/tarred read-path failure DFW has shown since ~2026-10-07.
NEMO_USERS=/lustre/fsw/portfolios/nemotron/users
MOUNTS="--container-mounts=${CODE_DIR}:/code,${RESULTS_DIR}:/results,${HFCACHE}:/hfcache,${LUSTRE}:${LUSTRE},${DATA_ROOT}:${DATA_ROOT},${NEMO_USERS}:${NEMO_USERS}"

# Do NOT enable xtrace: the command below contains expanded token values.
read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& nvidia-smi \
&& echo "*** RECIPE: ${CONFIG_NAME} (DFW, OFFLINE SALM encoder-free | lr ${LR} | max_dur ${MAX_DURATION} | 8 nodes) ***" \
&& export WANDB_API_KEY=${WANDB} \
&& export HF_HOME="/hfcache/" \
&& export HF_TOKEN=${HF_TOKEN} \
&& export HF_HUB_OFFLINE=1 \
&& export HYDRA_FULL_ERROR=1 \
&& export PYTORCH_ALLOC_CONF=expandable_segments:True \
&& export TORCH_NCCL_TIMEOUT_SEC=3600 \
&& export OMP_NUM_THREADS=1 \
&& export MKL_NUM_THREADS=1 \
&& export TORCH_NCCL_USE_COMM_NONBLOCKING=0 \
&& export TORCH_FR_BUFFER_SIZE=0 \
&& export TORCH_NCCL_ENABLE_MONITORING=0 \
&& export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=240 \
&& export NCCL_IB_TIMEOUT=22 \
&& export NCCL_IB_RETRY_CNT=10 \
&& export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& cd /code \
&& git rev-parse HEAD \
&& python /code/examples/speechlm2/salm_train.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    model.pretrained_llm=${PRETRAINED_LLM} \
    model.pretrained_asr=${PRETRAINED_ASR} \
    model.optimizer.lr=${LR} \
    model.lr_scheduler.warmup_steps=${WARMUP_STEPS} \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    trainer.devices=${GPUS_PER_NODE} \
    trainer.max_steps=${MAX_STEPS} \
    data.train_ds.input_cfg.0.input_cfg=${TRAIN_INPUT_CFG} \
    data.train_ds.num_workers=${NUM_WORKERS} \
    ++data.train_ds.max_duration=${MAX_DURATION} \
    data.train_ds.bucket_duration_bins="${BUCKET_BINS}" \
    data.train_ds.bucket_batch_size="${BUCKET_BATCH_SIZE}" \
    data.train_ds.seed=${LHOTSE_RND_SEED} \
    data.validation_ds.datasets.mcv_11_dev.input_cfg.0.manifest_filepath=${VAL_MANIFEST} \
    ++trainer.limit_train_batches=${VAL_CHECK_INTERVAL} \
    ++trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
    ++exp_manager.exp_dir=/results/ \
    ++exp_manager.create_wandb_logger=${CREATE_WANDB} \
    ++exp_manager.create_tensorboard_logger=false \
    ++exp_manager.max_time_per_run=00:03:55:00 \
    ++exp_manager.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME}
EOF

srun -o "$OUTFILE" -e "$ERRFILE" --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
