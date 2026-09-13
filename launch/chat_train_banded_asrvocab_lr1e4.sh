#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:chat-banded-asrvocab-lr1e4
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 8
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
# BANDED loss on the 1,024-piece ASR (donor SentencePiece) vocabulary.
#
#   sbatch launch/chat_train_banded_asrvocab_lr1e4.sh      <- no arguments
#
# WHY THIS EXISTS SEPARATELY FROM chat_train_banded_lr1e4.sh
# ---------------------------------------------------------
# Same recipe, re-run on current code. The earlier ASR-vocabulary banded arm
# (EXP_NAME granary2_chat_banded1_delay3_lr1e4, now at epoch 74, best
# val_wer 0.1139) was trained BEFORE the Qwen-vocabulary bring-up, which
# changed three things in the shared model class:
#
#   1. _setup_tokenizer accepts a HuggingFace AutoTokenizer (chat_bpe_models.py)
#   2. _chunk_texts_for_tokenizer prepends a word-boundary space for tokenizers
#      that do not supply one themselves
#   3. the RNN-T decoding blank is derived from the full vocabulary size rather
#      than the underlying tokenizer's, so it matches the joint's blank
#
# All three are designed to be no-ops for SentencePiece, and were verified as
# such before this launch: encode("word") -> ['▁wor','d'] so the property
# _tokenizer_supplies_word_prefix is True and fix (2) is skipped; for the 1k
# vocabulary the joint blank and the decoder blank index are both 128, so fix
# (3) changes nothing. The per-chunk ids still concatenate to the whole-
# transcript ids exactly (22 vs 22 on the check case). This run confirms that
# end to end on real data rather than by argument.
#
# A NEW EXP_NAME IS DELIBERATE. exp_manager runs with resume_if_exists, so
# reusing granary2_chat_banded1_delay3_lr1e4 would resume that run from
# epoch 74 under the new code -- a continuation, not the clean re-test this is
# meant to be. The name also matches the granary2_chat_banded1_qwenvocab_*
# arms, so the ASR/Qwen vocabulary pair reads as a pair.
#
# THE OBJECTIVE. Start from the frame-based chunk alignment but do not trust it
# absolutely: sum over every valid RNN-T path that stays within ONE chunk of it,
# so a word may be emitted a chunk early or a chunk late and the loss still
# credits it. band 0 reproduces the forced loss exactly and a very wide band
# reproduces full RNN-T -- both verified end to end against NeMo's own
# warprnnt_numba RNNTLoss, which agrees to 0.0 at band >= T.
#
# Cost is set by the alignment, not the lattice: the band scores ~3U + T nodes
# against the forced path's U + T, while full marginalisation is T x U. (Note
# the "~1.5 tokens per chunk" figure in the older comments and in the yaml is
# too low for Granary at 1.12 s chunks -- the real load is several times that,
# so the node count is correspondingly larger. It only affects the cost
# estimate, not correctness.)
#
# Emission delay 3 frames, as in the forced delay3 arm, so the band is centred
# on an alignment that already gives each word some right context. No history
# chunk and no window trimming: this arm varies the LOSS only, which makes
# granary2_chat_rnnt_lr1e4_wu5k the matched control.
# ============================================================================

export LOSS_TYPE=banded
export DELAY_FRAMES=3
export BAND_CHUNKS=1
export RECOVER_WORDS=0
export HISTORY_CHUNKS=0
export MAX_DELAY_FRAMES=0
# --- reference optimisation recipe, from oci_chat/chat_fullctx.sh ---
# Identical to chat_train_banded_lr1e4.sh so the two are comparable: lr 1e-4
# rather than the 1e-3 these arms first ran at. On the plain RNN-T control that
# change alone moved val_wer 0.1413 -> 0.1202.
export LR=1e-4
export WARMUP_STEPS=5000
export MAX_STEPS=500000

export EXP_NAME="${EXP_NAME:-granary2_chat_banded1_asrvocab_lr1e4}"

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
