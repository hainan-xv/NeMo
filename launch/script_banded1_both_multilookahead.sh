#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:streaming-stt-script-banded1-both-multi
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
# SCRIPT with the BANDED loss -- the A/B partner of granary2_script_tgtfix.
#
#   sbatch launch/script_banded1.sh          <- no arguments
#
# This is launch/script_tgtfix.sh with CONFIG_NAME and EXP_NAME changed. The
# recipe is streaming_stt_granary2_lora_script_tgtfix.yaml plus loss_type=banded,
# band_words=1, target_construction=partition and twod_layout=true. Same
# Qwen3-1.7B vocabulary, same chunk_size [2,4,7,10,14,28], same num_delay_frames 3,
# same audio_history_chunks 0, same optimiser, same seed, same respell_targets --
# so any delta is the loss.
#
# WHAT THE BAND DOES. SCRIPT trains as conditional text completion: given the
# transcript so far and one chunk of audio, emit the words that chunk reveals.
# Which words those are is ONE assignment chosen by the forced aligner, and the
# loss is plain cross-entropy over it -- the exact analogue of CHAT's
# forced_alignment. The aligner is not exact, so a word whose audio has not
# finished arriving is still charged to the earlier chunk, and the model is
# trained to guess it. The band marginalises over every assignment whose chunk
# boundaries sit within band_words WORDS of the aligner's, so the model may place
# a boundary word on either side without being penalised for it.
#
# WHY IT IS AFFORDABLE. The SCRIPT attention rule gives each branch a PREFIX of
# the spine and no access to any other branch, and the spine carries no audio. So
# the state "chunk t, spine cut u" fully determines a branch's input, and the
# candidate cuts of a chunk differ only in branch_prefix -- which enters the model
# solely through the branch mask. One spine forward therefore serves every
# candidate, and one branch forward per candidate scores EVERY span length at once
# (the score of emitting spine[u:u+k] is a running prefix sum). A band of width C
# costs C branch forwards, not C times the number of span lengths.
#
# WHY target_construction=partition IS MANDATORY. The band moves the cut between
# chunks. Only under the partition tokenization -- one tokenization of the whole
# transcript, split at the chunk bounds -- is the spine id sequence independent of
# where that cut falls. Under the legacy per-chunk tokenization moving a word
# re-tokenizes both neighbours, two paths reaching the same (chunk, cut) no longer
# share a history prefix, and the dynamic program is simply wrong. The model
# refuses to build in that combination rather than train it.
#
# THE CONTROL. band_words=0 reproduces the forced loss EXACTLY -- asserted
# end-to-end through the real model in
# tests/collections/speechlm2/test_script.py::test_banded_band_zero_equals_the_forced_loss_end_to_end,
# not merely "both are finite" the way CHAT's same-named test does. So
# granary2_script_tgtfix is a valid baseline for this arm.
#
# THE EVIDENCE THIS IS WORTH RUNNING. On the CHAT side the band beat the forced
# objective while being LESS trained: leaderboard macro 6.41 at mean step 16k
# versus 6.50 at 20k, over the same Qwen vocabulary and window.
#
# ORIGINAL TARGET-FIX NOTES, still in force (respell_targets is on here too):
#
# WHAT THE FIX DOES. The forced aligner ran on NORMALISED text, so punctuation
# inside a word is gone: 'forty-eight' arrives as 'fortyeight', 'U.S.' as 'US',
# '4,723,000,000' as '4723000000'. compute_word_spans' literal find() then fails
# two ways, both harmful:
#
#   * it returns -1 and the word gets no span, so script_messages holds its
#     cursor and the chunk is supervised as SILENCE while the word is actually
#     being spoken; the text is dumped into a later chunk. Measured at
#     chunk_size 14: 41.5% of such words land >=1 chunk late, up to 7.84 s.
#   * or it matches the respelled letters INSIDE a later word -- 'US' finds the
#     'us' in 'business' -- so the cursor jumps FORWARD over audio not yet
#     heard, every word in between loses its span, and the chunk boundary tears
#     mid-word, giving targets that decode as 'HMO bus iness'.
#
# With respell_targets on, both searches are whole-word anchored and the EARLIER
# hit wins: 100.000% of 454,819 aligner words located on the eval manifests
# (0.369% unlocated with it off), every mid-word tear gone, LibriSpeech spans
# byte-identical (it has no PnC, so nothing to respell -- which also makes it the
# built-in control dataset inside the leaderboard).
#
# NOT WARM-STARTED. The _ft arms in this family fine-tune from
# granary2_script_baseline, but that baseline is already fitted to the OLD
# targets, which would confound "better targets" with "more steps". This trains
# from the same starting point the baseline did.
#
# The change is gated default-off at every layer because compute_word_spans is
# SHARED with the interleaved StreamingSTTModel, whose targets would otherwise
# change on 7.71% of utterances -- including its teacher-forced validation.
# SCRIPT streaming SpeechLM finetune on the OCI grid (Granary 2.0, LoRA, no-blank).
#
# Each utterance is packed as a pure-text SPINE (the running transcript) plus one
# BRANCH per audio chunk (that chunk's audio + the words it reveals), trained in a
# single O(L) forward under a custom 4D mask:
#
#     p(words_k | text_history_<k, audio_k)
#
# Runs the synced repo mounted at /code, NOT the container's bundled NeMo.
#
# Usage (from the repo root on the OCI login node):
#   sbatch launch/script_baseline.sh          # seed 42
#   sbatch launch/script_baseline.sh 123      # seed 123
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

# Same wandb/results namespace as the OCI CHAT arms (oci_chat_train.sh), so the
# SCRIPT and CHAT runs of this era sit together rather than split across the
# older SpeechlmScriptCC tree.
PROJECT_NAME=SpeechlmOCI

# --- Training parameters ---
MAX_STEPS="${MAX_STEPS:-300000}"
# 2000, not the 4000 the forced arms use. Benchmarked locally on Qwen3-1.7B, the
# band costs ~3.4x the forced step averaged over the chunk_size draw (1.5x at
# chunk_size 14, 5.4x at chunk_size 2). The forced arm runs ~1.48 s/step on 8
# nodes, so banded lands near 5 s/step: 4000 steps would be ~5.6 h against a
# 3h55m max_time_per_run, and the run would be killed before it ever wrote a
# checkpoint -- restarting from scratch on every requeue, forever. 2000 steps is
# ~2.8 h and fits. Checkpoints stay step-indexed, so they remain directly
# comparable to the forced arm's, just twice as dense.
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-2000}"
LR="${LR:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-10000}"

# --- SCRIPT operating point ---
DELAY="${DELAY:-0}"
# dense | flex | script -- all mathematically identical; flex is fastest.
ATTN_BACKEND="${ATTN_BACKEND:-dense}"
# 0 reproduces the forced loss EXACTLY and is the control for isolating what the
# band itself costs, since it runs the identical 2-D banded machinery at C=1.
BAND_WORDS="${BAND_WORDS:-1}"
# THE POINT OF THIS ARM. band_side=later only lets a word slide LATER than
# the aligner put it; "both" also allows earlier. On DFW that difference
# was worth it only once the delay was removed -- with delay>0 the two
# shifts compound and the band ends up centred off the true alignment,
# which is why DELAY is 0 here and not the 3 that script_banded1.sh uses.
BAND_SIDE="${BAND_SIDE:-both}"
# ON for this arm, unlike the forced arms. The band triples the packed sequence,
# and job 13398712 died with a CUDA OOM of 1.71 GiB *inside loss.backward()* at
# 77.99/79.33 GiB used. That OOM is NOT recoverable by the training_step guard:
# the guard wraps the forward, and Lightning calls backward() afterwards, so a
# backward OOM escapes it and takes the rank -- and with it the DDP job -- down.
# Activation checkpointing roughly halves activation memory for ~30% more compute,
# which is the difference between "rare skipped batch" and "dead job".
# (The local Qwen3-1.7B benchmark that reported 8-11 GiB peak had this ON; the
# grid default was off, which is why those figures did not transfer.)
ACT_CKPT="${ACT_CKPT:-true}"
# The band makes the packed length vary widely across buckets (332 -> 2791 tokens
# at chunk 14), so the caching allocator sees a wide spread of block sizes and
# fragments. The failing OOM reported 358 MiB reserved-but-unallocated while
# asking for 1.30 GiB. expandable_segments lets the allocator grow a segment
# instead of needing one contiguous free block, which is exactly this shape of
# failure; PyTorch's own OOM message recommends it.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
AUDIO_HISTORY_CHUNKS="${AUDIO_HISTORY_CHUNKS:-0}"
# SINGLE chunk size, unlike every other SCRIPT arm. Benchmarked on Qwen3-1.7B,
# the band's cost tracks the SEGMENT count, which is the chunk count times the
# candidates per chunk -- so it is worst exactly where chunks are smallest:
# 5.4x forced at chunk_size 2 (20 s audio -> 375 segments, 3120 tokens) against
# 1.5-2.7x at chunk_size 14. Drawing uniformly from the six sizes would spend a
# sixth of every epoch in the 5.4x case for a latency setting we are not yet
# trying to answer questions about. Pinning 14 buys a fast first read; widen it
# once the band is known to help.
# MULTIPLE LOOK-AHEADS. chunk_size is sampled PER BATCH from this list, so one
# model is trained across 2, 7 and 14 frames of look-ahead (~0.16 s / 0.56 s /
# 1.12 s) instead of a separate arm per latency.
#
# The band needs no adjustment for this: band_candidate_cuts counts the band
# in WORDS, off the aligner word starts, and takes no chunk_size at all. So
# band_words=1 means the same thing at every chunk size -- the band structure
# is held fixed across look-aheads by construction, not by configuration.
#
# Validation needs no pinning either: val_chunk_size defaults to 14 when 14 is
# among the candidates, so val_wer stays comparable with the single-chunk arm.
CHUNK_SIZES="${CHUNK_SIZES:-[2,7,14]}"

# MEMORY AT chunk_size=2. Job 13605298 died with 25 consecutive OOM batches, and
# every one of them was chunk_size=2 -- 7 and 14 were fine. At chunk 2 an
# utterance has 7x the chunks of chunk 14, and both the SCRIPT forward and the
# banded lattice scale with chunk count, so the recipe's batch sizes (already
# mostly 1, because the banded loss is heavy) do not fit.
#
# Two levers, applied together:
#   1. CAP DURATION. The failures were long utterances at small chunk size -- a
#      40 s cut is ~250 chunks at chunk 2. Capping at 20 s removes the tail that
#      cannot fit rather than shrinking batches that are already 1. The bins are
#      truncated to match; lhotse needs len(bins) == len(batch_size).
#   2. HALVE THE BATCH SIZES, floored at 1.
#
# NOTE this makes the arm's DATA DISTRIBUTION differ from the chunk-14 arm, which
# sees the full 40 s tail. WER between the two is therefore not a clean
# like-for-like comparison of look-ahead alone.
MAX_DURATION="${MAX_DURATION:-20}"
BUCKET_BINS="${BUCKET_BINS:-[4.32,6.0,7.04,7.92,8.8,9.6,10.4,11.12,11.89,12.66,13.47,14.8,16.92,20.0]}"
BUCKET_BATCH_SIZE="${BUCKET_BATCH_SIZE:-[2,1,1,1,1,1,1,1,1,1,1,1,1,1]}"
# Apostrophe-free by construction: the Hydra override wraps it in single quotes.
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are doing streaming speech recognition. Given the transcript so far and the representation of the next audio chunk, output the words spoken in that chunk.}"

CONFIG_PATH=/code/examples/speechlm2/conf/
CONFIG_NAME="${CONFIG_NAME:-streaming_stt_granary2_lora_script_banded1}"
EXP_NAME="${EXP_NAME:-granary2_script_banded1_both_multilookahead}"

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
# Warm start from the standard SCRIPT arm, weights only -- the same design as
# the DFW both arm (which pinned that arm's -last checkpoint).
#
# PINNED ABSOLUTE, not via INIT_EXP. INIT_EXP resolves the run directory
# relative to PROJECT_NAME, and this arm reports into SpeechlmOCI while the
# baseline lives under the older SpeechlmScriptCC tree. With INIT_EXP the
# lookup missed, the launcher fell back to "training from base pretrained"
# and the run trained with NO warm start at all -- visible only as a missing
# init_from_ckpt log line, since the epoch counter looks identical either way.
INIT_CKPT="${INIT_CKPT:-/lustre/fsw/portfolios/nemotron/users/hainanx/results/SpeechlmScriptCC/granary2_script_baseline/granary2_script_baseline/checkpoints/step=191306-val_wer=0.0854-last.ckpt}"
INIT_EXP="${INIT_EXP:-}"
INIT_CKPT="${INIT_CKPT:-}"
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

# NOTE: with ++init_from_ckpt (weights only) global_step starts at 0, so this
# guard no longer applies -- it is kept only for the resume_from_checkpoint
# path and is skipped below.
# A warm start restores the FULL training state, global_step included. If the
# parent is already at or past MAX_STEPS this run has nothing to do: Lightning
# prints "max_steps reached", exits 0 after ~3 minutes, and Slurm reports
# COMPLETED with no checkpoints -- a failure that looks like a success. Job
# 12865606 died exactly this way. Fail loudly instead.
if [[ "${INIT_WEIGHTS_ONLY:-1}" != "1" && -n "$INIT_CKPT" && "$INIT_CKPT" != "none" ]]; then
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
    # WEIGHTS ONLY, not resume_from_checkpoint. This arm changes the LOSS, so it
    # needs a fresh LR schedule and fresh optimizer moments; resuming the parent
    # trainer state starts it at the baseline's epoch 47 / step ~194k on an
    # already-decayed LR, which is a continuation, not a warm start. Observed on
    # job 13602919, which came up at "Epoch 47". Matches dfw_script_banded1_both_nodelay_v2.
    INIT_CKPT_ARG="++init_from_ckpt=/results/init_from.ckpt"
fi

MOUNTS="--container-mounts=${DATA_DIR}:${DATA_DIR},${H_DIR}:${H_DIR},$HAINAN_DIR:$HAINAN_DIR,$CODE_DIR:/code,$RESULTS_DIR:/results,$DATA_DIR:/data,$PRETRAINED_MODEL_DIR:/pretrained,$CHECKPOINT_DIR:/checkpoints,${QUESTIONS_DIR}:/questions/,${HFCACHE}:/hfcache/,$DONGJI_ROOT:$DONGJI_ROOT${INIT_MOUNT:+,${INIT_MOUNT}:${INIT_MOUNT}}"

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "*** RECIPE: ${CONFIG_NAME} (SCRIPT, granary2, no-blank | band_words=${BAND_WORDS} side=${BAND_SIDE} | delay=${DELAY} | audio_history_chunks=${AUDIO_HISTORY_CHUNKS} | chunk sizes ${CHUNK_SIZES} | max_dur ${MAX_DURATION} | bbs ${BUCKET_BATCH_SIZE}) ***" \
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
    ++data.train_ds.max_duration=${MAX_DURATION} \
    ++data.train_ds.bucket_duration_bins="${BUCKET_BINS}" \
    ++data.train_ds.bucket_batch_size="${BUCKET_BATCH_SIZE}" \
    model.audio_history_chunks=${AUDIO_HISTORY_CHUNKS} \
    data.dataset.num_delay_frames=${DELAY} \
    ++model.attn_backend=${ATTN_BACKEND} \
    ++model.band_words=${BAND_WORDS} \
    ++model.band_side=${BAND_SIDE} \
    ++model.activation_checkpointing=${ACT_CKPT} \
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
