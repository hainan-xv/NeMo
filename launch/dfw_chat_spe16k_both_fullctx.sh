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
# at 2x the streaming arms' batch size (10x OOM'd -- see the batch block).
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

# --- BATCH x2 vs the streaming arms -----------------------------------------
#
# x10 WAS TRIED FIRST AND OOM'd, job 18956472, before a single logged step.
# It was not marginal: two ranks died with ~280 MiB free of a 79.11 GiB card
# while asking for ~585 MiB, i.e. the card was saturated, not just over.
#
#   GPU 4: 79.11 GiB total, 302.94 MiB free -- tried to allocate 586.00 MiB
#   GPU 5: 79.11 GiB total, 266.94 MiB free -- tried to allocate 582.00 MiB
#
# That is the ladder's x2 rung, reached by SKIPPING x5. The skip is deliberate:
# full attention is quadratic in utterance length, the bucket sizes were all
# measured against a [70,13] streaming window, and the Qwen full-context sibling
# carries a "WATCH MEMORY" warning while running the 1x list. x5 would have been
# another guess at a number this project has now OOM'd six times by guessing.
#
# This is the streaming arms' list x2 -- which is also the list the Qwen
# full-context arm runs, so memory behaviour here has a direct precedent rather
# than an extrapolation. Absolute, not a multiplier, so a YAML retune cannot
# silently change it.
export BUCKET_BATCH_SIZE='[152,116,100,88,80,72,68,60,56,52,48,44,40,32,28,24,20,16]'
#
# If this OOMs too, the only rung left is x1:
#   [76,58,50,44,40,36,34,30,28,26,24,22,20,16,14,12,10,8]
#
# STEP COUNTS ARE STILL NOT COMPARABLE to the streaming arms -- 2x batch means
# 2x samples per step. Compare samples-seen or wall-clock, not the step counter.
#
# LR UNCHANGED at the streaming arms' 5e-5. At x2 this is far less of a concern
# than it was at x10: it is the same batch/LR pairing the other v2 arms and the
# Qwen full-context arm already run, so this arm now differs from
# dfw_granary2_chat_spe16k_both in the ENCODER ALONE -- which is the comparison
# the 2x2 actually wants.
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
