#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-script-banded1-both
# DFW's default GPU partition. Unlike OCI there is ONE pool of 1850 nodes rather
# than batch_block1/3/4, so no comma-list is needed.
#SBATCH -p batch
#SBATCH -N 4
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
#SBATCH --exclude=pool0-00407

# ============================================================================
# SCRIPT with a TWO-SIDED banded loss (band_side=both).
#
#   sbatch launch/dfw_script_banded1_both.sh          <- no arguments
#
# The A/B partner of dfw_script_banded1.sh, which uses band_side=later. Identical
# in every other respect, so a delta between them is the band's DIRECTION and
# nothing else.
#
# WHAT THE TWO SIDES MEAN. The band moves the CUT between chunks, and cut index
# runs opposite to emission time:
#   later  -- a word may be emitted a chunk LATER than the aligner placed it.
#             The half aligner error can justify: a word whose audio ends just
#             after a boundary cannot honestly be emitted before that audio has
#             arrived, which is what num_delay_frames=3 already guards.
#   both   -- also allows the reverse, emitting a word EARLIER than placed, which
#             asks the model to commit from audio that may not have arrived.
#
# COST. `both` offers 3 candidate cuts per chunk against 2, so roughly 50% more
# branch segments and a correspondingly longer packed sequence -- which is the
# term that dominates memory here. The one-sided arm measured 0.203 s/step against
# the baseline's 0.164; expect this one to land nearer the 0.28-0.47 the two-sided
# band cost before the per-chunk span fix.
#
# MEMORY. This is the configuration that OOMed repeatedly before that fix, so it
# is the real test of whether per-chunk span sizing removed the tail or merely
# made it rarer. bucket_batch_size is left at the one-sided arm's conservative
# values deliberately -- changing two things at once would make a failure
# unattributable.
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
# New wandb project for the DFW era, separate from OCI's SpeechlmScriptCC. It
# also separates the results tree, so a DFW run can never be confused with --
# or resumed from -- an OCI one of the same arm name.
PROJECT_NAME="${PROJECT_NAME:-SpeechlmDFW}"

# --- the banded recipe, identical to the OCI arm ---
CONFIG_PATH=/code/examples/speechlm2/conf
CONFIG_NAME="${CONFIG_NAME:-streaming_stt_granary2_lora_script_banded1}"
EXP_NAME="${EXP_NAME:-dfw_granary2_script_banded1_both}"

MAX_STEPS="${MAX_STEPS:-500000}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-2000}"
DELAY="${DELAY:-3}"
LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
CHUNK_SIZES="${CHUNK_SIZES:-14}"
BAND_WORDS="${BAND_WORDS:-1}"
BAND_SIDE="${BAND_SIDE:-both}"
ACT_CKPT="${ACT_CKPT:-true}"
ATTN_BACKEND="${ATTN_BACKEND:-dense}"
NUM_WORKERS="${NUM_WORKERS:-4}"

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
DESIGN_NODES="${DESIGN_NODES:-8}"
ACTUAL_NODES="${SLURM_JOB_NUM_NODES:-$DESIGN_NODES}"
if [[ "${SKIP_NODE_SUFFIX:-0}" != "1" && "$ACTUAL_NODES" -ne "$DESIGN_NODES" ]]; then
    EXP_NAME="${EXP_NAME}_n${ACTUAL_NODES}"
    echo "==> Allocation is ${ACTUAL_NODES} node(s), not the designed ${DESIGN_NODES}; EXP_NAME -> ${EXP_NAME}"
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

MOUNTS="--container-mounts=${CODE_DIR}:/code,${RESULTS_DIR}:/results,${HFCACHE}:/hfcache,${LUSTRE}:${LUSTRE},${DATA_ROOT}:${DATA_ROOT}"

# Do NOT enable xtrace: the command below contains expanded token values.
read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& nvidia-smi \
&& echo "*** RECIPE: ${CONFIG_NAME} (DFW, SCRIPT banded | band_words=${BAND_WORDS} side=${BAND_SIDE} | chunk ${CHUNK_SIZES} | delay ${DELAY}) ***" \
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
&& python /code/examples/speechlm2/script_train.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    model.pretrained_llm=${PRETRAINED_LLM} \
    model.pretrained_asr=${PRETRAINED_ASR} \
    model.optimizer.lr=${LR} \
    model.lr_scheduler.warmup_steps=${WARMUP_STEPS} \
    model.chunk_size="${CHUNK_SIZES}" \
    ++model.band_words=${BAND_WORDS} \
    ++model.band_side=${BAND_SIDE} \
    ++model.activation_checkpointing=${ACT_CKPT} \
    ++model.attn_backend=${ATTN_BACKEND} \
    data.dataset.num_delay_frames=${DELAY} \
    data.train_ds.input_cfg=${TRAIN_INPUT_CFG} \
    data.train_ds.num_workers=${NUM_WORKERS} \
    data.train_ds.seed=${LHOTSE_RND_SEED} \
    data.validation_ds.datasets.mcv_11_dev.manifest_filepath=${VAL_MANIFEST} \
    ++trainer.limit_train_batches=${VAL_CHECK_INTERVAL} \
    ++trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.devices=${GPUS_PER_NODE} \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    trainer.log_every_n_steps=10 \
    ++exp_manager.exp_dir=/results/ \
    ++exp_manager.create_wandb_logger=true \
    ++exp_manager.create_tensorboard_logger=false \
    ++exp_manager.max_time_per_run=00:03:55:00 \
    ++exp_manager.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME}
EOF

srun -o "$OUTFILE" -e "$ERRFILE" --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
