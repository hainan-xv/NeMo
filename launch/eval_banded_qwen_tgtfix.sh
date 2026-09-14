#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:eval-banded-qwen-tgtfix
# Eval jobs go to the INTERACTIVE partition: single-node and short, while the
# batch blocks queue behind 8-node training for hours. The admin limit is ONE
# interactive job per user, so do not launch this alongside another eval or a
# smoke test.
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# The Qwen banded (band_chunks=1) target-fix arm, on its own.
#
#   sbatch launch/eval_banded_qwen_tgtfix.sh          <- no arguments
#
# Averages the TOP-5 checkpoints by val_wer with NeMo's own
# scripts/checkpoint_averaging/average_model_checkpoints.py, then evaluates on the
# full 74,838-utterance Open-ASR-Leaderboard set through the shared select_shard
# partition -- so the number is comparable to every other arm in this project.
# The averaged .nemo is reused ONLY when newer than every checkpoint, so an arm
# that has trained further is re-averaged automatically.
#
# TOPK=1 evaluates the single best checkpoint instead of the average, if what you
# want is literally the latest state rather than the comparable number.
#
# The arm carries target_construction=partition and delay_word_final_punctuation,
# and uses the Qwen config + vocabulary at averaging time -- averaging builds the
# model before loading weights, so a 151,669-piece checkpoint against the
# 1,024-piece config is a shape error at best.
#
# NOTE val_wer in the banner is the ASR collection's VERBATIM metric (raw .split()
# + edit distance, no normalisation). It is NOT comparable to speechlm2's val_wer,
# which Whisper-normalises both sides. Only the leaderboard macro is cross-family.
#
# ENV
#   TOPK            checkpoints to average (default 5)
#   FORCE_AVERAGE   1 to rebuild an averaged .nemo that already exists
# ============================================================================
set -uo pipefail

TOPK="${TOPK:-5}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
PROJECT="${PROJECT:-SpeechlmScriptCC}"
QWEN_TOK="${QWEN_TOK:-/lustre/fsw/portfolios/llmservice/users/heh/pretrained_models/huggingface/Qwen/Qwen3-1.7B}"

FA=model.forced_alignment
# Every arm is the same architecture; only the vocabulary differs. The overrides
# rebuild it for averaging, which happens before any weights are loaded.
BANDED_OVERRIDES="model.loss_type=banded ${FA}.num_delay_frames=3 ${FA}.recover_history_words=0 ${FA}.band_chunks=1 ${FA}.max_delay_frames=0 model.joint.history_chunks=0"

# exp_name | config_name | tokenizer_dir ("" = extract the donor SentencePiece)
ARM_LIST=(
  "granary2_chat_banded1_qwenvocab_tgtfix_lr1e4|nemotron_chat_transducer_granary2_qwen|${QWEN_TOK}"
)

# Under sbatch $0 is a copy in Slurm's spool directory, so dirname "$0" has no
# sibling eval_chat.sh; SLURM_SUBMIT_DIR is where the sbatch was issued.
find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_chat.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate eval_chat.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
LAUNCH_DIR="$(find_launch_dir)"

echo "==> evaluating ${#ARM_LIST[@]} models, one after another"
declare -a STATUS=()

for entry in "${ARM_LIST[@]}"; do
    exp="${entry%%|*}"
    rest="${entry#*|}"
    cfg="${rest%%|*}"
    tok="${rest#*|}"

    CKPTS="${OUTPUT_PREFIX}/results/${PROJECT}/${exp}/${exp}/checkpoints"
    echo
    echo "############################################################"
    echo "### ${exp}"
    echo "###   config=${cfg}  tokenizer=${tok:-<donor SentencePiece>}"
    if [[ -d "$CKPTS" ]]; then
        # Print the training level next to the arm: the four are NOT epoch-matched
        # and a table without this invites reading a training difference as a
        # target-construction difference.
        EPOCH="$(ls -1 "$CKPTS"/*.ckpt 2>/dev/null | grep -oE 'epoch=[0-9]+' | sort -t= -k2 -n | tail -1)"
        BEST="$(ls -1 "$CKPTS"/*.ckpt 2>/dev/null | grep -oE 'val_wer=[0-9.]+' | sort -t= -k2 -g | head -1)"
        echo "###   max ${EPOCH:-epoch=?}  best ${BEST:-val_wer=?}"
    fi
    echo "############################################################"

    if [[ ! -d "$CKPTS" ]]; then
        echo "    SKIPPED: no checkpoints at ${CKPTS}" >&2
        STATUS+=("${exp}|skipped (no checkpoints)")
        continue
    fi

    ARM_EXP_NAME="${exp}" \
    ARM_MODEL_OVERRIDES="${BANDED_OVERRIDES}" \
    ARM_CONFIG_NAME="${cfg}" \
    ARM_TOKENIZER_DIR="${tok}" \
    TOPK="${TOPK}" FORCE_AVERAGE="${FORCE_AVERAGE:-0}" \
    EVAL_TAG="avg${TOPK}" FRAME_TRIM="" \
        bash "${LAUNCH_DIR}/eval_chat.sh"
    rc=$?
    if [[ $rc -eq 0 ]]; then
        STATUS+=("${exp}|ok")
    else
        echo "    FAILED (exit ${rc}); continuing with the next model" >&2
        STATUS+=("${exp}|FAILED (exit ${rc})")
    fi
done

echo
echo "############################################################"
echo "### summary"
echo "############################################################"
for s in "${STATUS[@]}"; do
    printf '  %-48s %s\n' "${s%%|*}" "${s#*|}"
done
