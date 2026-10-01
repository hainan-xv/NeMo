#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-twostream-band0-lasttok-lr8e4
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
# TWO-STREAM SCRIPT, band_words=0, 4 nodes. RNN-T-STYLE JOINER.
#
# LR 8e-4: 4X the previous large LR (2e-4), 16x the original 5e-5 and 8x dfw_granary2_script_baseline.
#
# Part of an LR sweep: faster LRs have trained better so far, so this pushes
# until it breaks. warmup_steps stays 5000, so the peak is reached on the same
# step schedule -- at this peak the warmup is doing real work and an early
# blowup, if it comes, will land right after it ends.
#
# NOTE: no arm remains at the baseline's 1e-4, so this sweep has no LR-matched
# control running alongside it.
#
# (superseded) LR 2e-4: DOUBLE dfw_granary2_script_baseline.
# Past the matched-baseline point on purpose, to see whether the joint keeps
# benefiting from a larger step or starts to come apart. The 50.34M fully
# trainable joint layer is the exposed part -- a local probe already spiked
# grad_norm to ~1300 at 5e-5, so divergence here is a live possibility rather
# than a theoretical one. Watch the first few hundred steps.
#
# (superseded note) LR MATCHED TO dfw_granary2_script_baseline: 1e-4, not the 5e-5 this arm's
# ancestor defaulted to. That baseline is the fair reference -- same 4 nodes x 8
# devices, near-identical step time (0.166s vs 0.21s), same warmup_steps=5000,
# same CosineAnnealing/min_lr/max_steps -- so steps are comparable units and the
# 2x LR gap was NOT masked by batch size or parallelism.
#
# Deliberately NOT matched to that baseline, and why:
#   max_duration      20 vs its 40   -- 20 was chosen for two-stream's memory
#                                       profile (n_cells*w audio duplication);
#                                       raising it risks the OOMs we already hit.
#   bucket_batch_size [2,1,...] vs its [4,3,2,...] -- same reason.
#   num_delay_frames  0  vs its 3    -- a task change, not a hyperparameter; the
#                                       two-stream arms are designed at delay 0.
# Those three remain open confounds against that baseline.
#
# Exactly ONE change from dfw_twostream_band0_leftpos: joint_text_context=last.
# The joint attends to a SINGLE text key -- the most recent history token -- instead
# of the whole prefix before the cut. The text stream is causal, so that one vector
# already summarises everything before the cut; this makes the summary the only
# channel rather than letting the joint re-read the prefix token by token. The joint
# becomes f(audio frames, text state), i.e. an RNN-T joiner.
#
# Deliberately one change from leftpos so the pair IS attributable, unlike leftpos
# vs v3 which moved three things at once.
#
# Inherited from leftpos (see that file for the full rationale):
#
# 1. audio_position_mode=left. Audio sits in a FIXED attention slot before all
#    text instead of at base = m + p. Implemented by shifting text right by w, so
#    every index stays >= 0 -- RoPE sees only relative offsets, so this is exactly
#    "audio at -w..-1" without negative position_ids. Cells of one chunk now share
#    identical positions, which is what later allows one block per chunk instead of
#    one per cell (n_cells*w -> T*w + n_cells).
#    Trade-off accepted knowingly: audio->text offsets become negative in relative
#    terms and the most RECENT text token becomes the most distant key.
#
# 2. The within-block mask is now causal. It was anti-causal (a stray .T), so the
#    read-out position saw only ITSELF -- 1 audio frame per chunk instead of w=14,
#    with no indirect path through a single joint layer.
#
# 3. The dedicated joint layer actually takes effect now. In the previous arm
#    (19353886) the config never reached core_cfg -- TwoStreamSTTModelConfig was
#    missing @dataclass AND ScriptSTTModel hardcodes to_dataclass(ScriptSTTModelConfig),
#    so extra_joint_layer silently stayed False and that run was a duplicate of v3.
#    Confirmed by trainable=948096000, byte-identical to the baseline.
#
# These are THREE simultaneous changes; a win is not attributable without follow-ups.
#
# Differs from dfw_twostream_band0.sh in exactly one respect: the joint gets its
# own fully trainable decoder layer (50.34M params, warm-started from the LLM's
# last layer) instead of borrowing that layer while frozen.
#
# Why: with the borrowed layer, ALL audio-text interaction was carried by r=128
# LoRA on q_proj/v_proj of a single frozen layer -- 917,504 trainable params,
# under 0.1% of the 948M trainable total. The dedicated layer is 55x that.
#
# A second, independent gain: the text stream previously read hidden_states[-2],
# so the borrowed layer's output never reached it and the text representation was
# a 27-layer Qwen rather than 28. With a dedicated joint, text uses all 28 layers
# and the joint reads the last layer's PRE-norm residual stream via a hook
# (hidden_states[-1] is post-norm; feeding that in would double-normalise).
#
# Warm start comes from the SAME donor checkpoint as band0_v3, so the two arms
# are comparable from step 0 apart from the joint.
#
# NOTE: these are TWO changes at once (joint capacity and text depth). If this
# arm wins, which one did it is not yet separable.
#
#   sbatch launch/dfw_twostream_band0.sh
#
# Promotion of the 1-node smoke test, which reached ~900 steps with the loss
# falling 46.9 -> 26.5 and no errors. Still band 0 (the lattice degenerates to
# the aligner's single assignment, i.e. exactly the forced loss) and chunk 14,
# so this measures the ARCHITECTURE, not the band and not the small-chunk memory
# problem. Widening the band is the next experiment.
#
# loss_reduction=mean_volume: SUM of per-utterance NLL over SUM of target tokens,
# matching packed SCRIPT and NeMo's RNN-T convention, so every token carries
# equal weight regardless of which utterance it came from.
#
# EXPECTED LOSS AT INIT. Locally, with the fix and a stand-in encoder, the loss
# starts near 2x log(V) (~24 against log V = 11.93) and falls below log(V) within
# a few steps. A start in the hundreds means the read-out is broken again.
#
# NOTE ON THE REPORTED LOSS. The lattice scores tokens AND one <eot> per chunk,
# but mean_volume divides by tokens only. num_emissions is logged alongside
# num_targets so that gap is visible rather than silently inflating the
# per-token figure.
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
EXP_NAME="${EXP_NAME:-dfw_twostream_band0_lasttok_lr8e4}"

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
LR="${LR:-8e-4}"
# ---------------------------------------------------------------------------

VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-2000}"
DELAY="${DELAY:-0}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
CHUNK_SIZES="${CHUNK_SIZES:-14}"
BAND_WORDS="${BAND_WORDS:-0}"
# How many trailing LLM layers see audio. 1 is the design point; a knob because
# last-layer-only fusion is a bet, not a known quantity.
JOINT_LAYERS="${JOINT_LAYERS:-1}"
# Validation MUST decode at the chunk size this arm trains at. It was pinned to 7
# here, inherited from the chunk-7 launcher this was copied from, while training
# ran at 14 -- so val_wer described a look-ahead the model never saw. A
# CONFIGURED val_chunk_size beats the auto-default, so copying a launcher and
# changing CHUNK_SIZES alone is not enough.
VAL_MAX_NEW_TOKENS="${VAL_MAX_NEW_TOKENS:-24}"
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
INIT_CKPT="${INIT_CKPT:-/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/hainanx/results/SpeechlmDFW/pinned_init/dfw_twostream_band0_v2_carry.ckpt}"

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
    ++model.loss_reduction=mean_volume \
    ++model.joint_layers=${JOINT_LAYERS} \
    ++model.extra_joint_layer=true \
    ++model.audio_position_mode=left \
    ++model.joint_text_context=last \
    ++model.extra_joint_init_from_last=true \
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
    ++model.val_chunk_size=${CHUNK_SIZES} \
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
    ++exp_manager.create_wandb_logger=true \
    ++exp_manager.create_tensorboard_logger=false \
    ++exp_manager.max_time_per_run=00:03:55:00 \
    ++exp_manager.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME}
EOF

srun -o "$OUTFILE" -e "$ERRFILE" --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
