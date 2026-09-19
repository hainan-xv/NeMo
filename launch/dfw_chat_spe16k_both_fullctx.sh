#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-chat-spe16k-both-fullctx
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 04:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=8
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
# pool0-01815 carries an old NVIDIA driver (12020) and fails torch's CUDA init
# outright; it killed job 18686485 in 103s. pool0-00407 is the reference recipe's
# known-bad node.
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# CHAT band-1 both-side, FULL-CONTEXT encoder, 16k SentencePiece vocabulary,
# at 10x the streaming arms' batch size.
#
#   sbatch launch/dfw_chat_spe16k_both_fullctx.sh      <- no arguments
#
# This is the intersection of the two arms that already exist:
#
#   dfw_granary2_chat_spe16k_both          streaming encoder, 16k SPE vocab
#   dfw_granary2_chat_banded1_both_fullctx_parakeet   full-context, Qwen 151k
#
# so it completes a 2x2 over {streaming, full-context} x {Qwen 151k, SPE 16k}.
# Both existing arms score on the same 7-dataset macro (4.98 streaming/Qwen,
# 4.42 fullctx/Qwen, 5.14 streaming/SPE16k at max_symbols=15), which makes the
# fourth cell the one number needed to say whether the full-context encoder's
# -0.56 carries over to a purpose-built English vocabulary or is specific to
# Qwen's.
#
# NOT STREAMABLE, same as its Qwen sibling: full attention plus non-causal
# downsampling and convolution means no output exists until the audio ends. It
# is a reference point, not a deployable configuration.
#
# Encoder donor is parakeet-tdt-0.6b-v2, NOT the streaming nemotron. The config's
# encoder block is parakeet's field for field (batch_norm convs, xscaling false,
# att_context_size [-1,-1]); seeding a full-context encoder from a cache-aware
# streaming one would mismatch on exactly those fields.
# ============================================================================

export LOSS_TYPE=banded
export DELAY_FRAMES=0
export RECOVER_WORDS=0
export BAND_CHUNKS=1
export BAND_SIDE=both
export HISTORY_CHUNKS=0
export MAX_DELAY_FRAMES=0
export TARGET_CONSTRUCTION=partition
export DELAY_PUNCT=true
export WARMUP_STEPS=5000
export MAX_STEPS=500000
export EPOCH_STEPS=2000

# --- BATCH x10 vs the streaming arms ----------------------------------------
#
# REQUESTED EXPLICITLY, and it is the one knob that differs from a plain
# fullctx+SPE16k arm. The streaming arms run
#   [76,58,50,44,40,36,34,30,28,26,24,22,20,16,14,12,10,8]
# (itself 2x the YAML), and this is that list x10, written out absolutely so a
# YAML retune cannot silently change it.
export BUCKET_BATCH_SIZE='[760,580,500,440,400,360,340,300,280,260,240,220,200,160,140,120,100,80]'
#
# READ THIS BEFORE THE FIRST LAUNCH. Two independent reasons to expect trouble,
# neither of which is a reason not to try it -- only a reason to watch step 1
# rather than discover it at hour 3:
#
#   1. OOM RISK IS HIGH, and higher here than the multiplier suggests. Those
#      bucket sizes were measured against a [70,13] streaming window at ~30 GiB
#      of 81. Full attention is QUADRATIC in utterance length, so the per-sample
#      activation cost in this arm is already above where any of those numbers
#      were taken -- the Qwen fullctx sibling runs the 1x list and carries a
#      "WATCH MEMORY" warning at that. 10x on top is well outside measured
#      territory. This project has OOM'd five times from batch sizes reasoned
#      about rather than measured.
#
#      If it OOMs, step DOWN this ladder rather than re-deriving a number:
#        x5  [380,290,250,220,200,180,170,150,140,130,120,110,100,80,70,60,50,40]
#        x2  [152,116,100,88,80,72,68,60,56,52,48,44,40,32,28,24,20,16]
#        x1  [76,58,50,44,40,36,34,30,28,26,24,22,20,16,14,12,10,8]
#
#   2. STEP COUNTS ARE NOT COMPARABLE to any other arm. 10x batch means 10x
#      samples per step, so this arm's step 10,000 has seen what a streaming arm
#      sees at step 100,000. Compare samples-seen or wall-clock, never the step
#      counter, and do not read an early val_wer curve against the other arms'
#      without rescaling the x-axis.
#
# LR IS DELIBERATELY UNCHANGED at the streaming arms' 5e-5, so this arm differs
# from dfw_granary2_chat_spe16k_both in encoder and batch only. That is the
# minimal-change choice, and it is also the DEBATABLE one: at 10x the batch,
# linear scaling would argue for 5e-4 and sqrt scaling for ~1.6e-4, and an
# unscaled LR at 10x batch is effectively a much smaller step per sample -- the
# most likely way for this arm to come out looking flat rather than wrong. If it
# trains stably but slowly, raise LR here first.
export LR=5e-5
# ---------------------------------------------------------------------------

export CONFIG_NAME=nemotron_chat_transducer_granary2_spe_fullctx
export EXP_NAME="${EXP_NAME:-dfw_granary2_chat_spe16k_both_fullctx}"
# dfw_chat_train.sh defaults DESIGN_NODES to 4 and cannot grep it from $0 (it is
# exec'd from this wrapper). Left unset, a 2-node allocation would be treated as
# an undersized smoke test and silently renamed to ..._n2.
export DESIGN_NODES=2

# ENCODER ONLY, for two compounding reasons: parakeet-tdt-0.6b-v2 is a TDT model
# whose prediction network and joint carry duration outputs this model does not
# have, AND its vocabulary is 1,024 SentencePiece pieces against this arm's
# 16,384. Widening this list would not warm-start more -- init_from_nemo_model
# skips shape mismatches SILENTLY, so the tensors would sit at init while the
# load reported success.
export INIT_INCLUDE='["encoder."]'
export INIT_EXCLUDE='[]'

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr

# The purpose-built 16k vocabulary, from launch/dfw_build_spe_vocabs.sh. The
# trainer only extracts a donor SentencePiece when this directory holds no
# tokenizer.model, so pointing at a built one is enough to override it. Same
# directory as the streaming spe16k arm, deliberately: an identical vocabulary
# is what makes the two comparable.
export TOKENIZER_DIR="${DFW}/hainanx/tokenizers/granary2_en_spe/v16384"
if [[ ! -f "${TOKENIZER_DIR}/tokenizer.model" ]]; then
    echo "ERROR: no tokenizer at ${TOKENIZER_DIR}" >&2
    echo "       Build it first: sbatch launch/dfw_build_spe_vocabs.sh" >&2
    exit 1
fi

# parakeet-tdt-0.6b-v2, staged from HuggingFace. A .nemo because
# init_from_nemo_model is what the CHAT trainer exposes.
export INIT_NEMO="${INIT_NEMO:-${DFW}/hainanx/pretrained_models/nvidia/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo}"
if [[ ! -f "${INIT_NEMO}" ]]; then
    echo "ERROR: parakeet donor not found: ${INIT_NEMO}" >&2
    echo "       Stage it with:" >&2
    echo "       curl -L -H \"Authorization: Bearer \$(cat ~/.hf_token)\" \\" >&2
    echo "         -o ${INIT_NEMO} \\" >&2
    echo "         https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/resolve/main/parakeet-tdt-0.6b-v2.nemo" >&2
    exit 1
fi

find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/dfw_chat_train.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/dfw_chat_train.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/dfw_chat_train.sh" ]] && { echo "${here}"; return; }
    # ABSOLUTE fallback, and it is not belt-and-braces -- it is the only thing that
    # works on a REQUEUE. Slurm hands the requeued job SLURM_SUBMIT_DIR pointing at
    # the scratch ROOT rather than the directory the job was submitted from, and $0
    # is the spool copy, so both of the lookups above miss.
    local repo="${DFW_CODE_DIR:-${DFW}/hainanx/NeMo_SCRIPT_cc}"
    [[ -f "${repo}/launch/dfw_chat_train.sh" ]] && { echo "${repo}/launch"; return; }
    echo "ERROR: cannot locate dfw_chat_train.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}, repo=${repo})" >&2
    exit 1
}
exec bash "$(find_launch_dir)/dfw_chat_train.sh" "$@"
