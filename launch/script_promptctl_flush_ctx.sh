#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:streaming-stt-script-promptctl-flush-ctx
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
# PROMPT-CONTROLLED SCRIPT streaming SpeechLM finetune, WITH <flush> AND a
# DELAY-SIZED LEFT AUDIO CONTEXT (draco-oci).
#
# Same recipe/knobs as launch/script_promptctl_flush.sh (chunk-size / emission
# delay / cap+punct via prompt, plus the <flush> end-of-audio drain), with ONE
# architecture change: every branch's audio window is extended LEFT by
# AUDIO_LEFT_CONTEXT_FRAMES (= EXACT_MAX_DELAY) encoder frames, so a branch sees
# (chunk_size + max_delay) frames = its own chunk PLUS a fixed slab of pre-chunk
# history.
#
# WHY: with the default per-chunk window (audio_history_chunks=0,
# audio_window_frames=0) each branch only sees its OWN chunk's audio. A word held
# back by a positive emission delay is emitted from a LATER chunk whose window no
# longer contains that word's acoustics -- only the encoder's bounded left-context
# reaches back -- so accuracy degrades as the requested delay grows (e.g. d6 worse
# than d3 even after <flush> fixes the tail-drop). Prepending max_delay frames of
# left context guarantees a delay-held word's OWN audio is inside the window of the
# chunk that finally emits it, so higher delay should stop hurting (and can help,
# via extra right context). The window is sized identically at train and inference
# (see nemo/collections/speechlm2/parts/script.py::_audio_window_start), and the
# per-branch audio is fed via the explicit frame-index gather path (overlapping
# windows reuse encoder frames).
#
# AUDIO_LEFT_CONTEXT_FRAMES defaults to EXACT_MAX_DELAY; the chunk size still
# varies per batch (sampled in training / specified at inference) while the left
# slab is a fixed frame count. No new parameters, so this WARM-STARTS from the
# trained flush model (granary2_script_promptctl_flush) and only has to learn to
# use the newly-visible pre-chunk audio.
#
# Runs the clean SCRIPT repo (git-synced via sync_to_oci.sh, mounted at /code),
# NOT the container's bundled NeMo.
#
# Usage (optional random seed as $1; defaults to 42):
#   cd /lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_script_clean
#   sbatch launch/script_promptctl_flush_ctx.sh          # seed 42
#   sbatch launch/script_promptctl_flush_ctx.sh 123       # seed 123
#
# Knobs (env overrides):
#   AUDIO_LEFT_CONTEXT_FRAMES -- pre-chunk left audio context in frames (default = EXACT_MAX_DELAY)
#   FLUSH_PROB   -- non-final-chunk flush probability (default 0.1)
#   FLUSH_TOKEN  -- unused special token used as <flush> (default <|object_ref_end|>)
#   INIT_EXP     -- experiment to warm-start from (default granary2_script_promptctl_flush)
#   INIT_CKPT    -- pin a checkpoint, or "none" to train from base pretrained
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

# Random seed for the Lhotse dataloader. Optional here.
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


PROJECT_NAME=SpeechlmScriptClean

# Training parameters (match the granary2 no-blank launcher overrides).
MAX_STEPS=300000
VAL_CHECK_INTERVAL=2000
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
# Set DEBUG_PROMPT_EVERY=N to periodically print the EXACT rendered system prompt
# (with the batch's sampled delay/chunk/cap/punct) from the dataloader, so you can
# confirm {delay}/{chunk_size}/{format_clause} are really substituted at train
# time. Prints on batch #1 then every N batches, per dataloader worker. 0 = off.
DEBUG_PROMPT_EVERY="${DEBUG_PROMPT_EVERY:-0}"

# Config: OUR SCRIPT recipe, shipped in the synced repo at /code. Same BASE recipe
# as the baseline -- the prompt-control knobs are added below as ++ CLI overrides
# (no dedicated prompt-control yaml).
CONFIG_PATH=/code/examples/speechlm2/conf/
CONFIG_NAME=streaming_stt_granary2_lora_script

EXP_NAME=granary2_script_promptctl_flush_ctx

# --- FLUSH token (end-of-audio + on-demand drain of delay-held words) ---
# FLUSH_PROB: each non-final chunk becomes a flush chunk with this probability (the
# final chunk always flushes). FLUSH_TOKEN: an UNUSED special token repurposed as
# <flush> (must be distinct from the self-correction delete_token). Both model.* and
# data.dataset.* keys are set below in the training command: model.* is baked into
# the checkpoint cfg so eval auto-enables the final-chunk flush; data.dataset.*
# drives training supervision.
# NOTE: FLUSH_TOKEN contains Hydra-grammar-special chars ("<|...|>"), so it MUST be
# passed with the ="'${VAR}'" double-then-single quoting (like the prompt templates
# below): bash strips the outer double quotes and Hydra receives a single-quoted
# string. Passing it bare (++model.flush_token=<|object_ref_end|>) makes Hydra throw
# LexerNoViableAltException.
FLUSH_PROB="${FLUSH_PROB:-0.1}"
FLUSH_TOKEN="${FLUSH_TOKEN:-<|object_ref_end|>}"

# --- Prompt-control operating point (CLI overrides on top of the base recipe) ---
# (1) CHUNK SIZE: one drawn per batch from this set (encoder frames), stated in
#     the prompt via CHUNK_SIZE_PROMPT_TEMPLATE below.
CHUNK_SIZES="[2,7,14,28]"
# (2) LATENCY: emission delay ~ Uniform[0, EXACT_MAX_DELAY] frames per batch.
EXACT_MAX_DELAY=6
# (3) LEFT AUDIO CONTEXT: prepend this many encoder frames of pre-chunk history to
#     EVERY branch window, so each branch sees (chunk_size + AUDIO_LEFT_CONTEXT_FRAMES)
#     frames. Defaults to EXACT_MAX_DELAY so a word held back by the max delay still
#     has its own audio inside the chunk that emits it. Same at train and inference.
AUDIO_LEFT_CONTEXT_FRAMES="${AUDIO_LEFT_CONTEXT_FRAMES:-${EXACT_MAX_DELAY}}"
# Chunk-size clause appended to the prompt ({chunk_size} filled with the per-batch size).
CHUNK_SIZE_PROMPT_TEMPLATE="Process the audio in chunks of {chunk_size} frames."
# Training prompt template. {delay} = the per-batch exact delay; {format_clause} =
# the per-batch capitalization/punctuation instruction. Apostrophe-free so the CLI
# single-quoting stays safe (the default template in code says "each chunk's words").
# NOTE: the prompt wording is UNCHANGED from the non-flush run -- <flush> and the
# left audio context are architecture-level, not prompt clauses.
PROMPT_TEMPLATE="You are doing streaming speech recognition. Given the transcript so far and the next audio chunk, output the words spoken in that chunk. Emit the words of each chunk with a fixed delay of {delay} frames. {format_clause}"
# Validation prompt (read verbatim by model.val_system_prompt). Must equal the
# training render for a fixed operating point so val is in-distribution:
# PROMPT_TEMPLATE with (delay=3, caps+punct clause) + the chunk clause for 14
# frames (matches data.val_dataset_overrides.chunk_size=14 in the base recipe).
# The val references are plain caps+punct transcripts.
SYSTEM_PROMPT="You are doing streaming speech recognition. Given the transcript so far and the next audio chunk, output the words spoken in that chunk. Emit the words of each chunk with a fixed delay of 3 frames. Write the text with normal capitalization and punctuation. Process the audio in chunks of 14 frames."

# --- Warm-start from the trained flush PROMPT-CONTROL SCRIPT checkpoint ---
# The left audio context adds NO new parameters (it only feeds more encoder frames
# into each branch), so warm-start from granary2_script_promptctl_flush by default:
# it already knows delay conditioning + <flush> and only has to learn to use the
# newly-visible pre-chunk audio. If INIT_CKPT is empty it is AUTO-RESOLVED below to
# that exp's most recent checkpoint. Override: INIT_CKPT=/path/to.ckpt to pin one,
# INIT_EXP=<other_exp> to init from a different run, or INIT_CKPT=none to train from
# the base pretrained LLM+ASR. The init run MUST share this model's architecture
# (same base recipe / chunk-size set) or the loaded weights will not line up.
INIT_EXP="${INIT_EXP:-granary2_script_promptctl_flush}"
INIT_CKPT="${INIT_CKPT:-}"

# Directories for manifests, data, etc.
# Write-heavy outputs (results/checkpoints, HF cache, checkpoint temp) go to the
# nemotron project, which has free quota. The llmservice project that holds the
# synced code is at its space limit, so writes there fail with EDQUOT. Override
# the output location with OUTPUT_PREFIX.
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
RESULTS_DIR=${OUTPUT_PREFIX}/results/$PROJECT_NAME/$EXP_NAME

# Auto-resolve the init checkpoint from INIT_EXP when not given explicitly. Prefer
# the rolling "-last.ckpt", else the newest "step=..-val_wer=..ckpt" by step, else
# newest by mtime. resume_if_exists=true means this only seeds the FIRST launch;
# later relaunches auto-resume THIS run's own checkpoints. INIT_CKPT=none => base
# pretrained.
if [[ -z "$INIT_CKPT" ]]; then
    _INIT_DIR="${OUTPUT_PREFIX}/results/${PROJECT_NAME}/${INIT_EXP}/${INIT_EXP}/checkpoints"
    INIT_CKPT="$(ls -t "${_INIT_DIR}"/*-last.ckpt 2>/dev/null | head -1)"
    [[ -z "$INIT_CKPT" ]] && INIT_CKPT="$(ls "${_INIT_DIR}"/*.ckpt 2>/dev/null | grep -v -- '-averaged\.ckpt$' | sort -t= -k2 -g | tail -1)"
    [[ -z "$INIT_CKPT" ]] && INIT_CKPT="$(ls -t "${_INIT_DIR}"/*.ckpt 2>/dev/null | head -1)"
    if [[ -n "$INIT_CKPT" ]]; then
        echo "==> Auto-resolved LATEST INIT_CKPT from ${INIT_EXP}: ${INIT_CKPT}"
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
# The clean SCRIPT repo, git-synced via sync_to_oci.sh -> mounted as /code.
# Keep this in sync with OCI_REPO in sync_to_oci.sh.
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_script_clean}"

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
&& echo "*** RECIPE: ${CONFIG_NAME} (PROMPT-CONTROLLED SCRIPT + <flush> + left-audio-context, granary2, no-blank | chunk-size ${CHUNK_SIZES} via prompt | delay ~ U[0,${EXACT_MAX_DELAY}] via prompt | left_ctx=${AUDIO_LEFT_CONTEXT_FRAMES} frames | cap/punct on/off via prompt | flush_token=${FLUSH_TOKEN} flush_prob=${FLUSH_PROB}) ***" \
&& echo "*** WARM START: init=${INIT_CKPT:-none} (from ${INIT_EXP}) ***" \
&& echo "*** OBJECTIVE: p(words_k | text_history_<k, audio_{k window = chunk + ${AUDIO_LEFT_CONTEXT_FRAMES} left frames}) with end-of-audio + on-demand <flush> drain; packed spine+branch, single O(L) forward ***" \
&& echo "*** MONITOR: val_wer (min) -- streaming spine-KV decode ***" \
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
&& echo "Starting training (running the clean SCRIPT repo at /code, PROMPT-CONTROLLED SCRIPT + <flush> + left-audio-context streaming SpeechLM, GRANARY 2.0)" \
&& python /code/examples/speechlm2/script_train.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    ++model.compact_template=${COMPACT_TEMPLATE} \
    ++data.dataset.compact_template=${COMPACT_TEMPLATE} \
    ++model.debug_validate_tokens=$DEBUG_VALIDATE_TOKENS \
    model.optimizer.lr=$LR \
    model.lr_scheduler.warmup_steps=$WARMUP_STEPS \
    model.chunk_size="${CHUNK_SIZES}" \
    data.dataset.num_delay_frames=-1 \
    ++data.dataset.exact_delay=true \
    ++data.dataset.exact_max_delay=${EXACT_MAX_DELAY} \
    ++data.dataset.vary_text_repr=true \
    ++data.dataset.debug_print_prompt_every=${DEBUG_PROMPT_EVERY} \
    ++model.audio_left_context_frames=${AUDIO_LEFT_CONTEXT_FRAMES} \
    ++data.dataset.audio_left_context_frames=${AUDIO_LEFT_CONTEXT_FRAMES} \
    ++model.flush=true \
    ++model.flush_token="'${FLUSH_TOKEN}'" \
    ++data.dataset.flush=true \
    ++data.dataset.flush_token="'${FLUSH_TOKEN}'" \
    ++data.dataset.flush_prob=${FLUSH_PROB} \
    ++data.dataset.chunk_size_prompt_template="'${CHUNK_SIZE_PROMPT_TEMPLATE}'" \
    ++data.dataset.prompt_template="'${PROMPT_TEMPLATE}'" \
    data.dataset.system_prompt="'${SYSTEM_PROMPT}'" \
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
