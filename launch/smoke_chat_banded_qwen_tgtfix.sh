#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:smoke-chat-banded-qwen-tgtfix
# INTERACTIVE partition: the batch blocks queue behind 8-node training for
# hours, and the point of this copy is to find out QUICKLY whether the real run
# works. The admin limit is ONE interactive job per user, so do not run this at
# the same time as an eval or another smoke.
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

# ============================================================================
# BANDED loss on QWEN3's ~151.7k vocabulary.
#
#   sbatch launch/smoke_chat_banded_qwen_tgtfix.sh          <- no arguments
#
# One-node copy of chat_train_banded_qwen_tgtfix_lr1e4.sh (job 13369044), which
# is queued behind the 8-node ASR-vocab arms. ONLY THE ALLOCATION DIFFERS: 1 node
# on the interactive partition instead of 8 on the batch blocks. The model, the
# loss, the tokenizer, the target construction and the SCHEDULE are the real
# run's, line for line -- a shortened epoch would have made this a test of a
# configuration nobody is going to train.
#
# EXP_NAME is the real run's, deliberately: chat_train.sh appends _n<N> itself
# when the allocation does not match the -N it declares (it is exec'd, so $0 is
# chat_train.sh), so this lands in ..._n1 and cannot resume from or overwrite the
# 8-node run. Setting an explicit _n1 here would produce _n1_n1.
#
# WHAT TO LOOK FOR FIRST: `offset-fallback N/M utts` in the step-500 line. It
# should be near 0% (measured 0.29% offline with this tokenizer). If it is high,
# the partition path is not running and this arm is training the legacy
# objective -- which is exactly how the SentencePiece version failed at first.
#
# THE TARGET FIX, on the LLM vocabulary. Paired with
# chat_train_banded_asrvocab_tgtfix_lr1e4.sh: same loss, same band, same delay,
# same lr and the same target construction, differing only in the vocabulary --
# which is the 1,024-vs-151,669 comparison the banded loss exists to make
# possible at all.
#
#   1. word_spans retries a word that is not a literal substring of the
#      transcript with punctuation ignored. The aligner ran on normalised text,
#      so 'forward-looking' reaches us as 'forwardlooking' and previously got no
#      span, which dropped it from the target. Also catches abbreviations:
#      'am' -> 'a.m', 'US' -> 'U.S'.
#   2. Chunk targets are a PARTITION of one tokenization of the whole
#      transcript, split at verified boundaries, so the concatenated per-chunk
#      ids equal the whole-transcript ids. This matters MORE here than at 1k:
#      byte-level BPE has no dummy prefix, and tokenizing chunks separately is
#      exactly what produced 'the bestselling singleby a Germanartist'.
#   3. Word-final punctuation is emitted at the following word's chunk.
#
# The split is verified per utterance and falls back to the text path when a
# token straddles a boundary. Measured on the aligned manifests with THIS
# tokenizer: 0.29% of utterances fall back (earnings22 1.62%, spgispeech 0.38%,
# the rest 0.00%), so the partition path runs for 99.7% of the data. The run
# logs its own rate every 500 steps as `offset-fallback N/M utts`; if that
# climbs, this arm is quietly training the legacy objective and the comparison
# is void.
#
# Identical to chat_train_banded_lr1e4.sh except for the tokenizer, so the pair
# isolates the effect of vocabulary size. This is the case the band exists for:
# the full RNN-T loss needs a [B, T, U, V+1] tensor, which at V = 151,670 does
# not fit, while the band scores ~3U + T nodes and so grows with the alignment
# rather than with the lattice.
#
# The tokenizer does NOT come from the donor .nemo (that is the 1,024-piece
# SentencePiece model), so TOKENIZER_DIR is overridden to the Qwen directory and
# the extraction step is skipped.
#
# The warm start covers the encoder and joint.enc/pred only: the prediction
# network's embedding and the joint output layer are 148x larger here and have
# no counterpart in the donor, so they start random in both vocabulary arms --
# which is what keeps the comparison about the vocabulary.
# ============================================================================

export LOSS_TYPE=banded
export DELAY_FRAMES=3
export BAND_CHUNKS=1
export RECOVER_WORDS=0
export HISTORY_CHUNKS=0
export MAX_DELAY_FRAMES=0
# Targets are a split of ONE whole-transcript tokenization, and word-final
# punctuation is emitted at the FOLLOWING word's chunk. See the config.
export TARGET_CONSTRUCTION=partition
export DELAY_PUNCT=true
# --- reference optimisation recipe, from oci_chat/chat_fullctx.sh ---
# lr 1e-4 rather than the 1e-3 these arms first ran at. On the plain RNN-T
# control that change alone moved val_wer 0.1413 -> 0.1202; every CHAT recipe
# found on this machine peaks at or below ~2.5e-4 except one, and ours was ten
# times the only cosine reference.
export LR=1e-4
export WARMUP_STEPS=5000
export MAX_STEPS=500000

export CONFIG_NAME=nemotron_chat_transducer_granary2_qwen
export TOKENIZER_DIR=/lustre/fsw/portfolios/llmservice/users/heh/pretrained_models/huggingface/Qwen/Qwen3-1.7B
export INIT_EXCLUDE='["prediction.embed","joint_net"]'
export EXP_NAME=granary2_chat_banded1_qwenvocab_tgtfix_lr1e4

# Under sbatch $0 is a copy in Slurm's spool directory, so dirname "$0" has no
# sibling chat_train.sh; SLURM_SUBMIT_DIR is where the sbatch was issued.
find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/chat_train.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/chat_train.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/chat_train.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate chat_train.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}

exec bash "$(find_launch_dir)/chat_train.sh" "$@"
