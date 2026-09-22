#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-chat-spe-multivocab
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

# CHAT band-1 both-side with a PURPOSE-BUILT 1k SentencePiece vocabulary.
#
# The LOW end of the sweep. 8k/16k/32k measured 4.92/4.95/4.95 macro-7 -- flat
# across a 4x size range, with 8k both the best AND the smallest tested, so the
# trend is unbounded below. 1k/2k/4k probe whether it turns down.
#
# NOTE FOR EVAL: a smaller vocabulary means MORE tokens per word (1024-piece
# needs ~23 tokens on a probe sentence vs 13 at 16k), hence more inner
# decoder iterations per CHUNK. max_symbols is a per-chunk cap, so evaluate these
# arms at max_symbols>=15 or they risk being penalised by truncation rather than
# by vocabulary.
#
# One of six arms that replace Qwen3's 151.7k multilingual LLM
# vocabulary with one trained on the Granary v2 transcripts -- cased and
# punctuated. Everything else is held at dfw_chat_banded1_both_nodelay_v2's
# settings, so the three arms plus that one isolate vocabulary size.
#
# The warm start is ENCODER ONLY: decoder and joint are vocabulary-shaped.

# ============================================================================
# MULTI-VOCAB CHAT: one encoder, three vocabulary heads (1k / 2k / 4k).
#
#   sbatch launch/dfw_chat_spe_multivocab.sh      <- no arguments
#
# Each training batch SAMPLES one head uniformly and is scored entirely with it.
# Validation is pinned to head 0 (1k), so val_wer is one comparable curve rather
# than a mixture whose value depends on which head was drawn.
#
# WHY IT MIGHT WORK. The vocabulary sweep found 1k/4k/8k statistically
# indistinguishable (4.81/4.83/4.82 macro-7), and the encoder is ~0.6B of ~0.65B
# parameters -- the vocabulary-shaped tensors are a rounding error. If the
# encoder representation is genuinely vocabulary-agnostic, one encoder should
# serve all three for free. That is the hypothesis.
#
# NOT JUST AN EXTRA OUTPUT LAYER. The banded loss builds its lattice from
# PER-CHUNK TOKEN COUNTS, so switching head changes the targets, the band
# geometry and the lattice node count -- all rebuilt per batch. The three heads
# therefore see genuinely different loss surfaces over a shared encoder.
#
# HEAD 0 IS LOAD-BEARING. setup_training_data runs inside ModelPT.__init__ and
# builds the Lhotse dataset with self.tokenizer, so TOKENIZER_DIR below fixes
# what every batch's `transcript` tensor is encoded with for the whole run. It
# must equal multivocab.tokenizer_dirs[0] in the config; the model raises if not.
# On heads 1 and 2 the training-WER reference is re-encoded from the cut text,
# because scoring head 2's output against head 0's ids would be meaningless
# rather than merely noisy.
#
# WATCH: train_head_frac_0/1/2 in wandb. Uniform sampling should hold them near
# 1/3 each; a starved head would make the comparison between heads unfair and is
# otherwise invisible.
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

# --- v2 CHANGES: bigger batch, smaller LR -----------------------------------
#
# BATCH x2, MEASURED not guessed. nvidia-smi on the running v1 arms showed
# ~30 GiB used of 81 GiB per H100 -- 37%. Doubling every bucket puts the
# activation share at roughly 2x while the fixed weight/optimizer share is
# unchanged, which lands near 50 GiB and leaves real headroom for the long-
# utterance tail. This project has OOM'd four times from batch sizes reasoned
# about rather than measured, hence the reading above rather than a ratio.
#
# It also RESTORES THE GLOBAL BATCH the node cut took away: 2 nodes x 2x batch
# equals the 4-node arms' global batch exactly. So these runs are once again
# comparable in samples-per-step to everything measured before the cut, and the
# variance introduced by halving the nodes is removed at the source.
#
# The list is ABSOLUTE and doubles the YAML's
#   [38, 29, 25, 22, 20, 18, 17, 15, 14, 13, 12, 11, 10, 8, 7, 6, 5, 4]
# which is duplicated here deliberately: an absolute override cannot silently
# become wrong if the YAML is retuned, whereas a multiplier could.
#
# BATCH x2 AGAIN on top of that, matching dfw_chat_spe16k_both_fullctx.sh, i.e.
# 4x the YAML base. Requested explicitly.
#
# COMPARABILITY WARNING. The FINISHED 8k/16k/32k arms ran at the x2 list above
# (4.92/4.95/4.95 macro-7). These three run at x4, so a 1k-vs-8k difference now
# confounds vocabulary size with batch size. To read the six-point sweep as a
# vocabulary result, one arm must be re-run at x4 as a bridge -- spe8k is the
# natural choice, being the best of the finished three.
#
# MEMORY IS FINE, measured not assumed: the full-context arm runs this exact
# list at ~47 GiB of 81.5 per rank, and full attention costs MORE per sample
# than this arm's [70,13] window, so streaming at the same batch sits below that.
export BUCKET_BATCH_SIZE='[152,116,100,88,80,72,68,60,56,52,48,44,40,32,28,24,20,16]'
#
# LR 1e-4 -> 5e-5. Two reasons, and the second is the stronger one:
#   1. it is the conservative direction while the batch change beds in;
#   2. these arms are WARM STARTS from a converged model, but 1e-4 with a
#      5000-step warmup is a FROM-SCRATCH schedule. Applied to trained weights
#      that is large enough to walk the initialisation back before it helps,
#      which is a plausible reading of the flat trajectory we are reacting to.
# NOTE this is deliberately on the conservative side: with the global batch
# restored above, strict linear scaling would justify keeping 1e-4. If these
# arms now look SLOW rather than noisy, raising LR is the first thing to try.
export LR=5e-5
# ---------------------------------------------------------------------------

export CONFIG_NAME=nemotron_chat_transducer_granary2_spe_multivocab
export EXP_NAME="${EXP_NAME:-dfw_granary2_chat_spe_multivocab}"
# dfw_chat_train.sh defaults DESIGN_NODES to 4 and cannot grep it from $0 (it is
# exec'd from this wrapper). Left unset, a 2-node allocation would be treated as
# an undersized smoke test and silently renamed to ..._n2.
export DESIGN_NODES=2
# FULL transfer, unlike the donor-seeded arms. The seed is dfw_granary2_chat_forced
# -- the same nemotron_chat_transducer_granary2_qwen config, so the same 151k
# vocabulary and the same tensor shapes throughout. The usual exclusions exist
# only to survive a vocabulary change and would here discard the 151k embedding
# and the 151k joint output projection, which is most of what we are warm
# starting FOR.
# ENCODER ONLY. decoder.* and joint.* are VOCABULARY-SHAPED -- the prediction
# network's embedding and the joint's output projection are both sized by the
# vocabulary -- so with a 1k SentencePiece vocabulary they cannot transfer
# from a 151.7k-piece Qwen arm. Including them would not warm-start more; the
# shapes would simply fail to match and the tensors would stay at init while
# the load reported success.
# FULL init of head 0, not encoder-only. The donor is the TRAINED single-vocab
# 1k arm (macro-7 4.81), whose vocabulary is head 0's exactly, so its decoder and
# joint transfer tensor for tensor -- there is no reason to throw them away and
# relearn them. Heads 1 (2k) and 2 (4k) have different vocabularies, so their
# decoder/joint start from scratch, which is the intended asymmetry.
#
# INIT_EXCLUDE is empty on purpose: the launcher's default excludes
# prediction.embed, which is right when the vocabulary CHANGES and wrong here,
# where donor and head 0 are both 1,024 pieces.
export INIT_INCLUDE='["encoder.","decoder.","joint."]'
export INIT_EXCLUDE='[]'

# The standard CHAT arm's top-5 average, produced by its own leaderboard eval.
# A .nemo (not a .ckpt) because init_from_nemo_model is what the CHAT trainer
# exposes; the eval pipeline already builds exactly this artifact.
DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
# The purpose-built vocabulary, from launch/dfw_build_spe_vocabs.sh. The
# trainer only extracts a donor SentencePiece when this directory holds no
# tokenizer.model, so pointing at a built one is enough to override it.
export TOKENIZER_DIR="${DFW}/hainanx/tokenizers/granary2_en_spe/v1024"
if [[ ! -f "${TOKENIZER_DIR}/tokenizer.model" ]]; then
    echo "ERROR: no tokenizer at ${TOKENIZER_DIR}" >&2
    echo "       Build it first: sbatch launch/dfw_build_spe_vocabs.sh" >&2
    exit 1
fi
export INIT_NEMO="${INIT_NEMO:-${DFW}/hainanx/results/SpeechlmDFW/dfw_granary2_chat_spe1k_both/averaged/top5-averaged.nemo}"

if [[ ! -f "${INIT_NEMO}" ]]; then
    echo "ERROR: warm-start model not found: ${INIT_NEMO}" >&2
    echo "       Build it by evaluating the standard arm:" >&2
    echo "       ARMS=dfw_granary2_chat_forced sbatch launch/dfw_eval_chat.sh" >&2
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
