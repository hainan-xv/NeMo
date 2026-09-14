#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:chat-forced-qwen-win28-recover1
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
# BANDED loss on QWEN3's ~151.7k vocabulary.
#
#   sbatch launch/chat_train_forced_qwen_win28_recover1_lr1e4.sh          <- no arguments
#
# FORCED-ALIGNMENT loss on the win28 Qwen arm -- the base for history recovery.
#
# Deliberately NOT the banded loss. Recovery lives in build_forced_path, which
# only _forced_alignment_loss reaches; the banded loss goes through band_nodes,
# which sees per-chunk token COUNTS only. And recovery re-emits the previous
# chunk's last k words -- the same u twice -- which a monotone RNN-T lattice path
# cannot represent at any band width. So the fixed-alignment loss is the right
# base, and band_chunks is irrelevant here.
#
# NOTE the forced path differs from the banded one at delay=3 in a second way:
# _forced_alignment_loss sets joint.frame_trim=delay and appends a flush chunk,
# so the joint sees 11 of each chunk's 14 frames and the +3-frame alignment shift
# is compensated. That is the original forced design, not an accident.
#
# THIS ARM ADDS HISTORY RECOVERY (recover_history_words=1).
#
# Each chunk t is additionally scored on the previous chunk's last word, starting
# from the prefix that precedes it:
#
#     chunk t-1 :  ... W1 BLANK            <- unchanged, blank stays put
#     chunk t   :  W1 [own words] BLANK    <- begins one word earlier in u
#
# Nothing is removed from any output, so no chunk is trained to stop early -- the
# earlier stochastic word-delay version did remove, and biased toward premature
# blanks. It teaches the model to resume from a history that is one word short,
# which is what makes RETRACT-K DECODING in-distribution.
#
# With window_frames=28 the joint's window at chunk t spans chunk t-1 then chunk
# t, so the t-1/t boundary is the MIDPOINT of the window and the retracted word's
# acoustics are still inside it. At window_frames=0 (own chunk only) there would
# be nothing to recover from -- which is why this is built on win28.
#
# Retract-k is a DECODE knob, so this one checkpoint is evaluable at k=0,1,2 with
# no retraining. Measured previously on the 1k arm: retract-1 improved 14/14
# dataset-arm combinations (full AMI 0.1151 -> 0.1122), retract-2 was flat to
# worse, and k>=3 is out of distribution because training only supports k<=2.
#
# ============================================================================

export LOSS_TYPE=forced_alignment
export DELAY_FRAMES=3
export BAND_CHUNKS=1
export RECOVER_WORDS=1
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

export CONFIG_NAME=nemotron_chat_transducer_granary2_qwen_win28
export TOKENIZER_DIR=/lustre/fsw/portfolios/llmservice/users/heh/pretrained_models/huggingface/Qwen/Qwen3-1.7B
export INIT_EXCLUDE='["prediction.embed","joint_net"]'
export EXP_NAME="${EXP_NAME:-granary2_chat_forced_qwenvocab_win28_recover1_lr1e4}"

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
