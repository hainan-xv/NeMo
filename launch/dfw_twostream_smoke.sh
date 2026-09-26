#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-twostream-smoke
# DFW's default GPU partition. Unlike OCI there is ONE pool of 1850 nodes rather
# than batch_block1/3/4, so no comma-list is needed.
#SBATCH -p interactive
#SBATCH -N 1
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
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# TWO-STREAM SCRIPT -- first real-data smoke test. 1 node, interactive.
#
#   sbatch launch/dfw_twostream_smoke.sh
#
# WHAT IS BEING TESTED, in order of what would break first:
#   1. the model builds and the adapter pulls a real batch apart correctly
#   2. the joint layer runs over [text | audio blocks] and produces logits
#   3. the lattice accepts the new sigma and returns a FINITE loss
#   4. the loss goes down
#
# band_words=0 ON PURPOSE. It makes the lattice degenerate to the aligner's
# single assignment, i.e. exactly the forced loss -- so any discrepancy is in the
# new sigma pipeline rather than in the band. Widening the band is the next test,
# not this one.
#
# chunk_size=14 for the same reason: the large-chunk case is the one packed
# SCRIPT trains happily, so it isolates the architecture change from the
# small-chunk memory problem this whole line of work exists to fix.
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
EXP_NAME="${EXP_NAME:-dfw_twostream_smoke}"

MAX_STEPS="${MAX_STEPS:-500000}"
# --- v2 CHANGES: rebalanced buckets, smaller LR ------------------------------
#
# NOT a flat multiplier, unlike the CHAT v2 arms. SCRIPT cannot take one.
# Measured on the v1 arms: 49.2-65.3 GiB of 81.5 GiB per GPU, i.e. 80%
# utilisation, against CHAT's 37%. SCRIPT is ACTIVATION-dominated (fixed cost is
# only ~18-20 GiB for 948M trainable of 2.36B) where CHAT is fixed-cost
# dominated, so a 2x applied here would OOM outright.
#
# WHAT IS ACTUALLY FREE. Per-step audio load is wildly unbalanced: the long-tail
# buckets carry 39.6 audio-seconds per step while the short buckets carry 11-21.
# Peak memory is set by the LONG buckets, which sit at batch 1 and cannot be
# reduced. So the short buckets are leaving memory idle that the peak already
# reserves -- raising them toward the same audio-per-step costs nothing in PEAK,
# it only stops wasting the reservation. The long tail is left at 1 throughout.
#
# The target is deliberately under 39.6: 28 audio-s, leaving margin because
# SCRIPT's cost tracks token count (packed spine + per-chunk branches), which
# correlates with duration but is not determined by it. Cross-rank spread on the
# v1 arms was 33% at identical nominal batch, which is that effect.
#
# HISTORY, because this list has been wrong in both directions. It was cut four
# times chasing OOMs -- [38,29,25,...] -> [12,9,8,...] -> [13,9,8,...] ->
# [7,5,4,...] -> [4,3,2,...] -- and those OOMs were later traced to global-K
# padding, fixed by per-chunk K sizing. The cuts were never walked back, so the
# current list is roughly a tenth of the original for a reason that no longer
# applies.
# MEMORY, RESIZED FOR CHUNK 2. The inherited DFW schedule is sized for chunk 14:
# 18 bins, batches up to 6, and no max_duration cap (so 40s). The packed sequence
# scales with chunks x candidates-per-chunk, and at 80ms frames chunk 2 gives 500
# chunks for a 40s clip against chunk 14's 36, while a two-sided band_words=2
# gives 5 candidates against 3. That is ~2500 packed units where 375 already OOMs.
#
# These are the OCI chunk-2 settings: cap at 20s and one utterance per bucket
# (two in the shortest). 125 chunks x 5 = 625 units, ~1.7x the OOMing config --
# still tight, which is why OOM skips are expected and why the backward-pass
# guard in script_model.py matters here.
# CHUNK 7, NOT 2 -- and the full 20s range restored.
#
# The chunk-2 attempt does not fit. At 12s it ran with 504 MiB free of 79.11 GiB,
# took 4 caught forward OOMs plus a FATAL backward one, and step time spiked to
# 102s as the allocator thrashed. Batch was already at the floor, so the only
# remaining levers were cutting max_duration below 12s or giving up the wider
# band -- either of which changes the experiment rather than running it.
#
# Packed sequence scales with chunks x candidates. At 80ms frames a 20s clip is
# 250 encoder frames: 36 chunks at chunk 7 against 125 at chunk 2. With a
# two-sided band_words=2 giving 5 candidates that is 36 x 5 = 180 units, against
# the 375 that OOMs -- so chunk 7 needs no duration cut at all.
#
# KEEPING 20s ALSO PROTECTS THE COMPARISON: script_multi trains at
# max_duration=20, so cutting it here would confound the band change with a
# training-distribution change.
#
# The gap at cs7 is real but tractable: on the official board script_multi scores
# 6.119 against nemotron's 5.446. At cs2 the gap is 3.05 but unrunnable at this
# band width.
#
# Batch stays at the floor for this first run even though 180 units leaves ~2x
# headroom. Raising it is the obvious throughput win once the arm is confirmed
# stable; a failed launch costs more than a slow one, and this arm has burned
# three already.
MAX_DURATION="${MAX_DURATION:-20}"
BUCKET_BINS="${BUCKET_BINS:-[4.32,6.0,7.04,7.92,8.8,9.6,10.4,11.12,11.89,12.66,13.47,14.8,16.92,20.0]}"
BUCKET_BATCH_SIZE="${BUCKET_BATCH_SIZE:-[2,1,1,1,1,1,1,1,1,1,1,1,1,1]}"
#
# LR 1e-4 -> 5e-5, matching the CHAT v2 arms. These are WARM STARTS from a
# converged model, but 1e-4 with a 5000-step warmup is a from-scratch schedule;
# on trained weights that is large enough to walk the initialisation back.
LR="${LR:-5e-5}"
# ---------------------------------------------------------------------------

VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-2000}"
DELAY="${DELAY:-0}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
CHUNK_SIZES="${CHUNK_SIZES:-14}"
BAND_WORDS="${BAND_WORDS:-0}"
# How many trailing LLM layers see audio. 1 is the design point; a knob because
# last-layer-only fusion is a bet, not a known quantity.
JOINT_LAYERS="${JOINT_LAYERS:-1}"
BAND_SIDE="${BAND_SIDE:-both}"
ACT_CKPT="${ACT_CKPT:-true}"
ATTN_BACKEND="${ATTN_BACKEND:-dense}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# WARM START from the standard SCRIPT arm rather than from the donor ASR + fresh
# LoRA. Weights only -- step counter, LR schedule and optimizer moments all start
# fresh, so this arm runs its own schedule from step 0 with trained parameters.
# The seed arm is loss_type=forced on the SAME Qwen LLM and perception stack, so
# every tensor matches; script_train.py REFUSES if none do, which is the failure
# mode that otherwise masquerades as a successful warm start.
# WARM START FROM DFW'S OWN BANDED BOTH-SIDE ARM (val_wer 0.0882), hard-linked
# so save_top_k cannot rotate it away mid-run. Same banded objective and same
# band_side, differing only in chunk size and band width -- the closest start
# available on this grid. band_words is a LOSS-side knob, so 1 -> 2 transfers.
INIT_CKPT="${INIT_CKPT:-/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/hainanx/results/SpeechlmDFW/pinned_init/step=150000-val_wer=0.0882.ckpt}"

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
INIT_LINK=${MYDIR}/init_ckpts/${EXP_NAME}_init.ckpt
mkdir -p "$(dirname "$INIT_LINK")"
if [[ ! -e "$INIT_CKPT" ]]; then
    echo "ERROR: warm-start checkpoint not found: ${INIT_CKPT}" >&2
    exit 1
fi
ln -sfn "$INIT_CKPT" "$INIT_LINK"
echo "==> warm start: ${INIT_LINK} -> ${INIT_CKPT}"

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
    ++model.two_stream=true \
    ++model.joint_layers=${JOINT_LAYERS} \
    ++model.band_words=${BAND_WORDS} \
    ++model.band_side=${BAND_SIDE} \
    ++model.activation_checkpointing=${ACT_CKPT} \
    ++model.attn_backend=${ATTN_BACKEND} \
    data.dataset.num_delay_frames=${DELAY} \
    ++init_from_ckpt=${INIT_LINK} \
    data.train_ds.input_cfg=${TRAIN_INPUT_CFG} \
    data.train_ds.num_workers=${NUM_WORKERS} \
    ++data.train_ds.max_duration=${MAX_DURATION} \
    ++data.train_ds.bucket_duration_bins="${BUCKET_BINS}" \
    data.train_ds.bucket_batch_size="${BUCKET_BATCH_SIZE}" \
    ++model.val_chunk_size=7 \
    ++model.val_max_new_tokens_per_chunk=12 \
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
