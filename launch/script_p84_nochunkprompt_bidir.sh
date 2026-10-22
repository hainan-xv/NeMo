#!/bin/bash
# Prompt-controlled SCRIPT, 84-context chunk set, with BOTH:
#   * the chunk size REMOVED from the instruction (prompt_include_chunk_size=false)
#   * BIDIRECTIONAL attention within each branch's audio block
#     (model.bidirectional_audio=true)
#
# The LLM is otherwise causal everywhere, including across the frames of a
# single chunk -- but those frames all arrive together, so there is no
# streaming reason to hide the later ones from the earlier ones. Requires the
# flex backend, which this recipe already uses (ATTN_BACKEND=flex).
#
# NOTE: 2 nodes, not 8, so the effective batch is 4x smaller than the sibling
# runs. Compare its val_wer trend against its own warm start, not against the
# 8-node arm step-for-step.
#
# This arm does NOT pin encoder_chunk_size: the encoder keeps its normal
# chunk-sized look-ahead. Only the LLM-side audio masking changes.
#
# WHY: render_control_prompt normally writes "The audio is chunked every N
# frames ..." into the instruction, so two fronts running at different chunk
# sizes get DIFFERENT instructions. Their prompts diverge at the chunk-size
# digit -- which sits before the text history -- so the [instruction][history]
# prefix, the part that grows and is worth caching, cannot be shared between
# fronts. Measured on the demo's 2/28 pair: 37 common tokens out of 54/55.
#
# The chunk size does not need to be stated: the audio window the model is shown
# is exactly chunk_size frames wide, so it stays observable. The emission delay
# is NOT observable and is still stated, which is why the demo must drive both
# fronts at the SAME delay for their prompts to match.
#
# Warm-started from the webpage model's own lineage (promptctl84_full_lr1e4,
# step=52000, val_wer 0.0859) rather than the ragged arm, which carries a
# plain-grid regression. The no-chunk wording differs from the trained wording
# only by dropping " every N frames", keeping the warm start close.
#
# This is NOT the ragged-grid experiment: the uniform grid is used throughout.
#
# OCI copy of the prompt-controlled SCRIPT recipe with the 84-context chunk set.
# Written because DFW is unusable: since ~2026-10-07 18:40 every 4-node job there
# hangs after 'Creating a Lhotse DynamicBucketingSampler' without reaching step 1,
# including UNMODIFIED dfw_script_banded1 and dfw_chat_banded1, and the idle reaper
# kills them at 31 minutes. A control reading a plain JSON manifest trains fine, so
# the fault is the sharded/tarred read path, not any recipe. This arm continues on
# OCI until that is resolved.
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:p84-nochunk-bidir
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 2
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
# PROMPT-CONTROLLED SCRIPT -- one checkpoint, every operating point.
#
# Same model, data and attention backend as launch/script_flex.sh. The change is
# that four decoding settings are drawn during training and STATED in each
# example's instruction, so at inference you select an operating point by asking
# for it in words instead of training a model per setting:
#
#   chunk size      drawn per BATCH   (the encoder's right context is chunk-1,
#                                      so it cannot vary within a batch)
#   delay           drawn per EXAMPLE from DELAY_CANDIDATES
#   capitalization  drawn per EXAMPLE with probability CAP_PROB
#   punctuation     drawn per EXAMPLE with probability PUNCT_PROB
#
# Targets are restyled to match each draw, so instruction and supervision always
# agree. Training and inference render that sentence with the same function, so
# they cannot drift apart.
#
# Because the model now has to serve 6 chunk sizes x 7 delays x 4 styles, expect
# it to need more steps than a single-operating-point run to reach the same WER
# at any one of them. Compare against granary2_script_flex at matched settings.
#
# CAVEAT: the leaderboard normalizer lowercases and strips punctuation before
# scoring, so the cap/punct controls do NOT show up in leaderboard WER. They need
# a case- and punctuation-preserving scorer to be measured.
#
# Runs the synced repo mounted at /code, NOT the container's bundled NeMo.
#
# Usage (from the repo root on the OCI login node):
#   sbatch launch/script_promptctl.sh          # seed 42, from scratch
#   sbatch launch/script_promptctl.sh 123      # seed 123
#   DELAY_CANDIDATES="[0,3,6]" sbatch launch/script_promptctl.sh   # coarser delays
#
# Or from your laptop:
#   ./oci_launch.sh launch/script_promptctl.sh
#   ./oci_launch_interactive.sh MAX_STEPS=50 VAL_CHECK_INTERVAL=25 launch/script_promptctl.sh
#
# Knobs (env overrides):
#   DELAY                -- emission delay in encoder frames (default 3)
#   ATTN_BACKEND         -- dense | flex | script (default dense);
#                           all identical mathematically, flex is fastest
#   ACT_CKPT             -- recompute LLM activations in backward (default false)
#   AUDIO_HISTORY_CHUNKS -- previous chunks of audio per branch (default 0)
#   CHUNK_SIZES          -- multi chunk-size list (default [2,4,7,10,14,28])
#   MAX_STEPS / LR / WARMUP_STEPS / VAL_CHECK_INTERVAL
#   EXP_NAME / CONFIG_NAME / OUTPUT_PREFIX / CODE_DIR
#   INIT_EXP / INIT_CKPT -- warm start (INIT_CKPT=none => base pretrained)
#
# NOTE on DELAY vs AUDIO_HISTORY_CHUNKS: a positive delay makes a word be emitted
# from a LATER chunk than the one its audio ended in. With the default
# AUDIO_HISTORY_CHUNKS=0 each branch sees only its own chunk, so a delayed word's
# acoustics are no longer in the window when the model must predict it. If you
# raise DELAY much, raise AUDIO_HISTORY_CHUNKS to >= 1 as well.
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

# New project for this arm: separates the results tree and the wandb project
# from SpeechlmScriptCC, which holds the older OCI SCRIPT/CHAT runs.
PROJECT_NAME="${PROJECT_NAME:-SpeechlmPromptCtl84}"

# --- Training parameters ---
MAX_STEPS="${MAX_STEPS:-300000}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-4000}"
# 2e-5, NOT the 1e-4 the LoRA arms use. 1e-4 is a LoRA learning rate; applied to
# every weight of a pretrained Qwen3 it damages the model fast. Full-finetune LRs
# for a 1.7B LLM are normally 1e-5 to 2e-5.
# 1e-4, NOT 2e-5. Measured on 1 node with LR as the only variable: at step 4000
# 2e-5 gave val_wer 1.4091 and 1e-4 gave 0.1809 -- better at 4k than 2e-5 managed
# at 12k (0.2428). The 2e-5 8-node arms reached 0.1612/0.1753/0.1967 at 12k.
LR="${LR:-0.0001}"
WARMUP_STEPS="${WARMUP_STEPS:-10000}"

# --- SCRIPT operating point ---
DELAY="${DELAY:-3}"
# dense | flex | script -- all mathematically identical; flex is fastest, and is
# what the LoRA arms trained under, so keep it here for a like-for-like comparison.
ATTN_BACKEND="${ATTN_BACKEND:-flex}"
# Prompt control. DELAY_CANDIDATES is a Hydra list literal (no spaces).
# Max delay 4: the chunk grid now reaches 42 frames (3.36 s), so a large
# frame delay on top of a long chunk pushes emission past anything useful.
DELAY_CANDIDATES="${DELAY_CANDIDATES:-[0,1,2,3,4]}"
CAP_PROB="${CAP_PROB:-0.5}"
PUNCT_PROB="${PUNCT_PROB:-0.5}"
# false, NOT true. Activation checkpointing is BROKEN under full finetuning here:
# the recompute packs its saved tensors in a different order than the forward and
# torch aborts with CheckpointError at step 0. The same recipe with LoRA (most
# weights frozen, so a much shorter save list) checkpoints fine. Verified without
# it on 1 node: 1.16 it/s, zero OOM, val_wer 1.4091 -> 0.6560 -> 0.2428 over
# steps 4k/8k/12k from scratch. Per-GPU memory is identical at 8 nodes.
ACT_CKPT="${ACT_CKPT:-false}"
AUDIO_HISTORY_CHUNKS="${AUDIO_HISTORY_CHUNKS:-0}"
# 28 and 42 are new. They are only legal because the config raises the
# encoder left context to 84: the Conformer requires
#   att_context_size[0] % (att_context_size[1] + 1) == 0
# and the right context is set per batch to chunk_size - 1, so the left
# context must DIVIDE every chunk size. 70 fails for 28 and 42
# (70 % 28 = 14, 70 % 42 = 28); 84 divides 2/7/14/28/42 exactly.
CHUNK_SIZES="${CHUNK_SIZES:-[2,7,14,28,42]}"
# Apostrophe-free by construction: the Hydra override wraps it in single quotes.
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are doing streaming speech recognition. Given the transcript so far and the representation of the next audio chunk, output the words spoken in that chunk.}"

CONFIG_PATH=/code/examples/speechlm2/conf/
CONFIG_NAME="${CONFIG_NAME:-streaming_stt_granary2_full_script_promptctl84}"
# _lr1e4: a NEW name, because resume_if_exists=true would otherwise resume from
# (and overwrite) the 2e-5 run's checkpoints sitting in the old directory.
# Its own name: a different model (bidirectional audio) AND a different node
# count. The automatic _n{nodes} suffix compares the allocation against
# DESIGN_NODES parsed from this script's own '#SBATCH -N' -- 2 here -- so
# design == actual and the suffix never fires. Pin it explicitly or this
# would share a checkpoint directory with the 8-node causal-masked run.
EXP_NAME="${EXP_NAME:-granary2_script_p84_nochunkprompt_bidir}"

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
# No SCRIPT warm start on purpose: the existing checkpoints carry LoRA adapter
# weights, which have no counterpart in a full-finetune model. Starts from the
# pretrained Qwen3 + nemotron encoder.
# Warm start from the best full-finetune checkpoint (step 52k, val_wer 0.0859):
# this run only teaches the model that the chunk grid is not fixed to
# multiples of chunk_size, so there is nothing to re-learn from scratch.
#
# NOT the -averaged.ckpt. INIT_CKPT is consumed as
# exp_manager.resume_from_checkpoint, i.e. Lightning's RESUME path, which
# restores optimizer and scheduler state. The averaged checkpoint is
# weights-only (5.3 GB vs 14 GB) and dies with "Trying to restore optimizer
# state but checkpoint contains only the model". The auto-resolve branch
# below already filters -averaged for exactly this reason.
# CONTINUATION: no warm start. INIT_CKPT must stay "none" so the recipe does
# NOT pass ++exp_manager.resume_from_checkpoint -- that flag points at the
# step-52k parent and restores ITS global_step, which would silently throw
# away everything this experiment has trained since. With it unset,
# resume_if_exists=true picks up this run's own -last.ckpt instead.
INIT_CKPT="${INIT_CKPT:-/lustre/fsw/portfolios/nemotron/users/hainanx/results/SpeechlmPromptCtl84/granary2_script_promptctl84_full_lr1e4/granary2_script_promptctl84_full_lr1e4/checkpoints/step=52000-val_wer=0.0859.ckpt}"
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

# DATA_DIR is a SHARED symlink -> /lustre/fs11/portfolios/coreai/.../data, and
# that target vanished on 2026-10-12 between 09:15 and 15:11 (the symlink itself
# is unchanged since 2026-09-29). Every job since then died at container start:
#   enroot-mount: failed to mount: .../data: No such file or directory
# because enroot cannot create a destination under a dangling symlink.
# This recipe does NOT need it -- the train yaml pulls audio from s3:// (29 refs,
# 0 /data refs) and its manifests live under users/dongjig on fs12. So mount it
# only when it actually resolves: a dead path in someone else's portfolio must
# not be able to stop our training.
DATA_MOUNT=""
if [[ -d "${DATA_DIR}" ]]; then
    DATA_MOUNT="${DATA_DIR}:${DATA_DIR},${DATA_DIR}:/data,"
else
    echo "==> WARNING: ${DATA_DIR} does not resolve -- omitting it from container mounts."
    echo "==>          (train audio comes from s3://, manifests from users/dongjig, so this is survivable)"
fi
MOUNTS="--container-mounts=${DATA_MOUNT}${H_DIR}:${H_DIR},$HAINAN_DIR:$HAINAN_DIR,$CODE_DIR:/code,$RESULTS_DIR:/results,$PRETRAINED_MODEL_DIR:/pretrained,$CHECKPOINT_DIR:/checkpoints,${QUESTIONS_DIR}:/questions/,${HFCACHE}:/hfcache/,$DONGJI_ROOT:$DONGJI_ROOT${INIT_MOUNT:+,${INIT_MOUNT}:${INIT_MOUNT}}"

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "*** RECIPE: ${CONFIG_NAME} (PROMPT-CONTROLLED SCRIPT | delays=${DELAY_CANDIDATES} P(cap)=${CAP_PROB} P(punct)=${PUNCT_PROB} | attn=${ATTN_BACKEND} ckpt=${ACT_CKPT} | chunk sizes ${CHUNK_SIZES}) ***" \
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
&& echo "Starting SCRIPT training (repo at /code, GRANARY 2.0)" \
&& python /code/examples/speechlm2/script_train.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    model.optimizer.lr=$LR \
    model.lr_scheduler.warmup_steps=$WARMUP_STEPS \
    model.chunk_size="${CHUNK_SIZES}" \
    model.audio_history_chunks=${AUDIO_HISTORY_CHUNKS} \
    data.dataset.num_delay_frames=${DELAY} \
    data.dataset.delay_candidates=${DELAY_CANDIDATES} \
    data.dataset.capitalization_prob=${CAP_PROB} \
    model.prompt_include_chunk_size=false \
    model.bidirectional_audio=true \
    data.dataset.punctuation_prob=${PUNCT_PROB} \
    model.attn_backend=${ATTN_BACKEND} \
    model.activation_checkpointing=${ACT_CKPT} \
    data.dataset.system_prompt="'${SYSTEM_PROMPT}'" \
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
    ++model.perception.encoder.sync_max_audio_length=false \
    ${INIT_CKPT_ARG}
EOF

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
