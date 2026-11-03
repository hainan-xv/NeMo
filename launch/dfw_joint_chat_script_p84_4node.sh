#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-joint-chat-script-p84
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
# pool0-01815 carries an old NVIDIA driver (12020) and fails torch's CUDA init
# outright; it killed job 18686485 in 103s. pool0-00407 is the reference recipe's
# known-bad node.
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# JOINT CHAT + SCRIPT ON ONE SHARED ENCODER. alpha = 1, 4 nodes.
#
# One ConformerEncoder, two heads, both losses on every step:
#
#     L = L_script + CHAT_LOSS_WEIGHT * L_chat
#
# The two families were already closer than they look -- same nemotron-0.6b
# encoder donor, same Qwen3 vocabulary, same banded objective, and crucially the
# same word alignments, both read from cut.custom["alignments"]. What differed
# was only the head: SCRIPT projects into Qwen3 and scores text completion;
# CHAT runs an RNN-T decoder + attention joint. Mounting both on one encoder
# asks whether the encoder learns a better representation when it has to serve
# both, and gives a like-for-like comparison no pair of separate runs can
# (identical data order, identical alignment, identical latency draw).
#
# LATENCY IS SHARED, NOT MERELY MATCHED. SCRIPT draws a chunk per batch from
# CHUNK_SIZES; that same draw is pushed into the CHAT joint's chunk_size and
# into the encoder look-ahead (right = chunk - 1). Left context is 84 at every
# operating point, which every chunk in {2,7,14,28} divides -- the condition
# `chunked_limited` masking requires (left % (right+1) == 0). 42 is dropped from
# the p84 recipe's list so both heads train on the same four points.
#
# DELAY IS PINNED TO 3. SCRIPT can draw a delay per EXAMPLE; the CHAT banded
# loss carries one delay for the whole batch. A multi-valued delay list would
# therefore give the two heads DIFFERENT alignments for the same audio, which
# is the one thing a joint model must not do.
#
# The CHAT head is ~203M parameters, dominated by the 151669-way joint output.
# Its own encoder/preprocessor are built from config and then dropped in
# JointChatScriptModel.__init__ -- they are a second copy of the shared
# front-end, and left in place they would be checkpointed, counted by the
# optimizer, and flagged by DDP as unused.
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
CONFIG_NAME="${CONFIG_NAME:-streaming_stt_granary2_full_joint_chat_script_p84}"
# v2: FIRST run with a correct read-out. Everything before this trained with
# lm_head applied WITHOUT model.norm, which made the logits garbage -- measured
# on Qwen3-1.7B predicting its own next token, 175.37 nats/token without the norm
# against 4.43 with it. A fresh EXP_NAME so those curves do not sit in the same
# wandb series as this one; they are not comparable.
# v3: FIRST run whose WER is meaningful. Everything earlier decoded validation
# through the INHERITED packed-SCRIPT generate (audio at layer 0, all N layers)
# at val_chunk_size=7 while training at 14 -- so dev_wer described a different
# architecture at a look-ahead the model never trained on. Fresh EXP_NAME because
# the metric changes MEANING here, not just value; the old points must not sit in
# the same series.
#
# Weights carry over via init_from_ckpt (weights only, optimiser reset): the
# earlier runs' LOSS was valid -- they already had the normalised read-out -- so
# the ~1h of training they did is worth keeping even though their WER was not.
EXP_NAME="${EXP_NAME:-dfw_joint_chat_script_p84}"

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
#
# LR 1e-4 -> 5e-5, matching the CHAT v2 arms. These are WARM STARTS from a
# converged model, but 1e-4 with a 5000-step warmup is a from-scratch schedule;
# on trained weights that is large enough to walk the initialisation back.
LR="${LR:-1e-4}"
# ---------------------------------------------------------------------------

VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-2000}"
DELAY="${DELAY:-3}"
WARMUP_STEPS="${WARMUP_STEPS:-10000}"
# wandb logging. Training does NOT need it: val_wer lives in checkpoint
# FILENAMES, not in wandb, so a run with this off loses live curves only.
CREATE_WANDB="${CREATE_WANDB:-true}"

# --- Data sizing: the YAML's BATCHES, the proven DURATION RANGE -------------
# Two failures bracket this setting, and the fix has to avoid both.
#
#  v1  max_duration 20 + batch [2,1,1,...]  (inherited from the banded recipe,
#      where chunk-2 is memory-bound) -> 32 GPUs doing batch-of-1 work. The idle
#      reaper killed it at 100% and 75% waste.
#  v2  dropped all overrides, taking the YAML's max_duration 54 + batch [38..4]
#      -> DynamicBucketingSampler stalled 29 of 31 minutes and never produced a
#      training step. Reaper again, 31% waste.
#
# Why v2 stalled: the YAML's bins run to 39.6s, and its last four buckets
# (22.4/29.44/35.0/39.6) hold the rarest clips. The sampler fills buckets before
# yielding a first batch, so those four gated startup -- on half the intended
# workers. The first 14 bins are byte-identical to the ones that filled quickly
# for every arm on this data, so the duration range is kept at 20 and only the
# BATCH SIZES are taken from the YAML: 10-19x v1, with a bucket profile already
# proven on this input_cfg.
MAX_DURATION="${MAX_DURATION:-20}"
BUCKET_BINS="${BUCKET_BINS:-[4.32,6.0,7.04,7.92,8.8,9.6,10.4,11.12,11.89,12.66,13.47,14.8,16.92,20.0]}"
BUCKET_BATCH_SIZE="${BUCKET_BATCH_SIZE:-[38,29,25,22,20,18,17,15,14,13,12,11,10,8]}"
CHUNK_SIZES="${CHUNK_SIZES:-[2,7,14,28]}"
# SCRIPT trains only at chunks >= this; smaller draws are CHAT-only steps.
SCRIPT_MIN_CHUNK="${SCRIPT_MIN_CHUNK:-7}"
# One head per batch (sampled per step) instead of summing both losses.
LOSS_SAMPLING="${LOSS_SAMPLING:-true}"
CHAT_LOSS_PROB="${CHAT_LOSS_PROB:-0.5}"
# val_chunk_size must stay a single int and match the train chunk here.
VAL_CHUNK_SIZE="${VAL_CHUNK_SIZE:-14}"
# Scalar fallback only; BAND_BY_CHUNK below is what actually applies.
# PER-CHUNK bands, parallel to CHUNK_SIZES: 14->1, 7->1, 2->4 (per side).
# Bands per side, parallel to CHUNK_SIZES: 14->1, 7->1, 2->3.
# Narrower at chunk 2 than the [1,1,4] arm: C = 2B+1 drops 9 -> 7, so chunk-2
# batches carry ~22% fewer branch segments. Those are the tail that dominates
# step time (p90 2.80s, max 10.10s while p50 is 0.227s), so this trades band
# width for throughput on exactly the batches that cost the most.

# --- exact memory reduction for the banded loss (see script_model.py) ---
# Branch SEGMENTS per group. Branches are conditionally independent given the
# spine (the mask permits spine keys below prefix_len and own-segment only),
# so [spine | subset of segments] is the SAME computation on fewer segments.
# Each group is checkpointed, so only the tiny span-logprob tensor stays live.
# Verified exact against the fused path: deltas <= 1e-4 at batch 1 and 3.
# 0: the POSITION budget governs. A segment count cannot know how wide its segments
# are (K_t varies per chunk), so it needs re-tuning per (band, chunk) pair.
BRANCH_GROUP_SIZE="${BRANCH_GROUP_SIZE:-0}"
# Packed positions per group. Batches whose whole utterance fits this budget take the
# ORIGINAL fused forward -- no per-group gather, no checkpoint recompute, no spine
# re-run. That matters most here: at chunk 14 and 7 the band is 1 (C=3) and the
# utterance fits easily, so two thirds of this mix now skips the group machinery.
# Measured locally, 13.7s utterance, Qwen3-1.7B:
#     chunk 14 band 1:  grouped 1.57s  ->  fused 0.54s   (2.90x)
#     chunk  2 band 7:  59 groups 9.35s -> ~4 groups 4.41s (1.59x)
BRANCH_GROUP_TOKENS="${BRANCH_GROUP_TOKENS:-2000}"
# Rows of the vocab projection materialised at once. The (B, L, 151936)
# logits and their float32 copy measured 48-63% of peak; this caps them.
LOGIT_CHUNK="${LOGIT_CHUNK:-1024}"
# How many trailing LLM layers see audio. 1 is the design point; a knob because
# last-layer-only fusion is a bet, not a known quantity.
JOINT_LAYERS="${JOINT_LAYERS:-28}"
# Validation MUST decode at the chunk size this arm trains at. It was pinned to 7
# here, inherited from the chunk-7 launcher this was copied from, while training
# ran at 14 -- so val_wer described a look-ahead the model never saw. A
# CONFIGURED val_chunk_size beats the auto-default, so copying a launcher and
# changing CHUNK_SIZES alone is not enough.
VAL_MAX_NEW_TOKENS="${VAL_MAX_NEW_TOKENS:-24}"
# THE POINT OF THIS ARM: a candidate cut may land inside a word, anywhere within
# the same word-defined band. The word-atomic rule makes a 2-frame chunk emit an
# entire multi-token word in one 0.16 s step; letting the cut fall mid-word
# spreads those tokens over the chunks whose audio actually carries them.
# Costs ~1.2-1.35x candidates per chunk, so ~25% more packed positions.
# EVAL MUST pass FORCE_WORD_START=0 or decoding re-imposes what this relaxes.
# Band counted in CHUNKS, applied PER TOKEN: a token may be emitted at most B
# chunks from the aligner's placement, so chunk t's cut ranges over
# [aligner_cut_{t-B}, aligner_cut_{t+B}] and EVERY token index between is legal --
# a word may therefore be split across chunks. BAND_BY_CHUNK below is in CHUNKS.
# band_token_cuts is ignored when BAND_UNIT=chunk.

# --- Prompt control -------------------------------------------------------
# Max delay is 4 (was 8): the chunk grid now reaches 42 frames (3.36 s), so a
# large frame delay on top of a long chunk would push emission far past what any
# streaming use wants.
# A SINGLE delay: the prompt states cap, punct and one delay, and never the
# chunk size. Naming the chunk size makes two operating points diverge at that
# digit and destroys the shared KV prefix self-correction depends on.
DELAY_CANDIDATES="${DELAY_CANDIDATES:-[3]}"
CAP_PROB="${CAP_PROB:-0.5}"
PUNCT_PROB="${PUNCT_PROB:-0.5}"
CHAT_LOSS_WEIGHT="${CHAT_LOSS_WEIGHT:-1.0}"
ACT_CKPT="${ACT_CKPT:-true}"
ATTN_BACKEND="${ATTN_BACKEND:-dense}"
# 8, matching the YAML. The 4 here was inherited from the banded recipe and
# halved the prefetch available to fill buckets.
# 4, not the inherited 8. DFW nodes have 128 CPUs against OCI's 248, so 8 ranks x 8
# workers = 64 readers starves the host and the loader dies with
#   RuntimeError: DataLoader worker (pid ...) is killed by signal: Killed
# which is how job 21138375 failed after 13 minutes. Lowering this is what made
# every other DFW arm in this project train.
NUM_WORKERS="${NUM_WORKERS:-4}"

# Tokenise in the DATALOADER WORKERS, not the main process.
#
# v2 and v3 both stalled ~24-29 minutes at "Creating a Lhotse
# DynamicBucketingSampler" and produced ZERO training steps before the idle
# reaper killed them. v3 ruled out max_duration (it used the proven 20s/14-bucket
# range), leaving batch size as the only difference from the arm that trained.
#
# The mechanism: with pretokenize=True the sampler tokenises every cut it
# inspects while filling buckets, serially in the main process. v1 needed ~2
# cuts per bucket to emit a batch; v3 needs 38 across 14 buckets -- ~20x the
# serial tokeniser work before the FIRST batch exists, with Qwen3's large
# tokenizer. Hence minutes of idle GPUs.
#
# Cost: disables token-per-second filtering and 2D bucketing. This recipe uses
# 1D bucketing (bucket_duration_bins only), so only the t/s filter is lost.
PRETOKENIZE="${PRETOKENIZE:-false}"

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
# Warm start from DFW's own promptctl84 arm (best val_wer 0.0901). Both arms of
# this pair use the SAME checkpoint so n is the only variable between them.
# NO WARM START. Only the pretrained ENCODER donor (PRETRAINED_ASR) and the
# base Qwen3 are loaded; nothing from any model we have already trained. A
# SCRIPT warm start would both bias the result toward our existing models and
# make the two heads asymmetric -- SCRIPT resuming while the 203M CHAT head
# starts random, dumping large RNN-T gradients into an encoder SCRIPT already
# depends on. Starting both heads untrained keeps the comparison honest.
INIT_CKPT="${INIT_CKPT:-none}"

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

# THIRD root, and not optional. ${LUSTRE} is nemotron/PROJECTS/...; the granary
# manifests live under nemotron/USERS/heh/manifests and are a different mount.
# Without this the job reaches the training loop and dies on the first shard with
#   [Errno 2] No such file or directory: .../nemotron/users/heh/manifests/granary_...
# which is exactly how jobs 21132207/21132208 failed after 30 minutes.
NEMO_USERS=/lustre/fsw/portfolios/nemotron/users
MOUNTS="--container-mounts=${CODE_DIR}:/code,${RESULTS_DIR}:/results,${HFCACHE}:/hfcache,${LUSTRE}:${LUSTRE},${DATA_ROOT}:${DATA_ROOT},${NEMO_USERS}:${NEMO_USERS}"

# Do NOT enable xtrace: the command below contains expanded token values.
read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& nvidia-smi \
&& echo "*** RECIPE: ${CONFIG_NAME} (DFW, JOINT CHAT+SCRIPT on a shared encoder | alpha=${CHAT_LOSS_WEIGHT} | chunk ${CHUNK_SIZES} | left-ctx 84 | delays=${DELAY_CANDIDATES} cap=${CAP_PROB} punct=${PUNCT_PROB}) ***" \
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
    model.chat.tokenizer.dir=${PRETRAINED_LLM} \
    model.optimizer.lr=${LR} \
    model.lr_scheduler.warmup_steps=${WARMUP_STEPS} \
    model.chunk_size="${CHUNK_SIZES}" \
    model.prompt_include_chunk_size=false \
    ++model.modality_local_layers=0 \
    ++model.two_stream=false \
    ++model.branch_group_size=${BRANCH_GROUP_SIZE} \
    ++model.branch_group_tokens=${BRANCH_GROUP_TOKENS} \
    ++model.logit_chunk=${LOGIT_CHUNK} \
    data.dataset.delay_candidates=${DELAY_CANDIDATES} \
    data.dataset.capitalization_prob=${CAP_PROB} \
    data.dataset.punctuation_prob=${PUNCT_PROB} \
    ++model.chat_loss_weight=${CHAT_LOSS_WEIGHT} \
    ++model.script_min_chunk_size=${SCRIPT_MIN_CHUNK} \
    ++model.loss_sampling=${LOSS_SAMPLING} \
    ++model.chat_loss_prob=${CHAT_LOSS_PROB} \
    ++model.activation_checkpointing=${ACT_CKPT} \
    ++model.attn_backend=${ATTN_BACKEND} \
    data.dataset.num_delay_frames=${DELAY} \
    ${INIT_ARG} \
    data.train_ds.input_cfg=${TRAIN_INPUT_CFG} \
    data.train_ds.num_workers=${NUM_WORKERS} \
    ++data.train_ds.pretokenize=${PRETOKENIZE} \
    ++data.train_ds.max_duration=${MAX_DURATION} \
    ++data.train_ds.bucket_duration_bins="${BUCKET_BINS}" \
    data.train_ds.bucket_batch_size="${BUCKET_BATCH_SIZE}" \
    ++model.val_chunk_size=${VAL_CHUNK_SIZE} \
    ++model.val_max_new_tokens_per_chunk=${VAL_MAX_NEW_TOKENS} \
    data.train_ds.seed=${LHOTSE_RND_SEED} \
    data.validation_ds.datasets.mcv_11_dev.manifest_filepath=${VAL_MANIFEST} \
    ++trainer.limit_train_batches=${VAL_CHECK_INTERVAL} \
    ++trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.devices=${GPUS_PER_NODE} \
    trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
    trainer.log_every_n_steps=10 \
    ++exp_manager.exp_dir=/results/ \
    ++exp_manager.create_wandb_logger=${CREATE_WANDB} \
    ++exp_manager.create_tensorboard_logger=false \
    ++exp_manager.max_time_per_run=00:03:55:00 \
    ++exp_manager.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME}
EOF

srun -o "$OUTFILE" -e "$ERRFILE" --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
