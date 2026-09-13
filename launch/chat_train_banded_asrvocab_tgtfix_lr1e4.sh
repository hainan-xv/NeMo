#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:chat-banded-asrvocab-tgtfix-lr1e4
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
# WHY THIS EXISTS SEPARATELY
# --------------------------
# The target-construction pair for granary2_chat_banded1_asrvocab_lr1e4. Same
# recipe, same code, differing ONLY in how the per-chunk TARGETS are built:
#
#   1. word_spans retries a word that is not a literal substring of the
#      transcript with punctuation ignored on both sides. The aligner ran on
#      normalised text, so the transcript's "forward-looking" reaches us as
#      "forwardlooking" and previously got no span at all -- and a word with no
#      span cannot be sliced into its chunk, so it vanished from the target.
#      Measured over the aligned manifests: 0.462% of all aligner words, 0.528%
#      on spgispeech, 0.000% on LibriSpeech (no punctuation to strip). The retry
#      recovers 93.8% of them.
#
#   2. Chunk targets are now a PARTITION of one whole-transcript tokenization,
#      split by character offset, rather than a separate tokenization of each
#      chunk's text. The concatenated per-chunk ids equal the whole-transcript
#      ids by construction, for any tokenizer family.
#
#   3. DELAY_PUNCT=true: a token that is pure punctuation is emitted at the
#      chunk of the FOLLOWING word. Whether a comma or period belongs after a
#      word usually cannot be decided until the next word is heard, so charging
#      it to the chunk that emits the word it trails asks the model to predict
#      it from audio that does not determine it. Utterance-final punctuation has
#      no following word and does not move.
#
# The run logs its own mismatch rate every 500 steps
# (forced_alignment.log_target_mismatch_every_n_steps), so the figure above is
# confirmed on the real training manifest rather than inferred from the eval
# manifests it was measured on.
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
# Word-final punctuation is emitted at the FOLLOWING word's chunk.
export TARGET_CONSTRUCTION=partition
export DELAY_PUNCT=true
# --- reference optimisation recipe, from oci_chat/chat_fullctx.sh ---
# Identical to chat_train_banded_lr1e4.sh so the two are comparable: lr 1e-4
# rather than the 1e-3 these arms first ran at. On the plain RNN-T control that
# change alone moved val_wer 0.1413 -> 0.1202.
export LR=1e-4
export WARMUP_STEPS=5000
export MAX_STEPS=500000

export EXP_NAME="${EXP_NAME:-granary2_chat_banded1_asrvocab_tgtfix_lr1e4}"

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
