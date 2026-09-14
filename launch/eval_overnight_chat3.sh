#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:eval-overnight-chat3
# INTERACTIVE partition. The admin limit is ONE interactive job per user, which
# is used here deliberately: this job and the SCRIPT eval are both submitted with
# --begin, and the limit serialises them -- the SCRIPT one starts when this ends.
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
# 2h30 for three arms. Measured: one Qwen arm (average top-5 + the full 74,838
# utterances) took 12m30s, the 1k-vocab arms ~8m each. 50m expected, so this is
# ~3x margin -- and a shorter wall backfills onto interactive sooner than 4h.
#SBATCH -t 02:30:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Overnight leaderboard eval of the three CHAT arms currently training.
#
#   sbatch --begin=now+6hours launch/eval_overnight_chat3.sh
#
#   granary2_chat_banded1_qwenvocab_win28_lr1e4           banded,  win28
#   granary2_chat_forced_qwenvocab_win28_lr1e4            forced,  win28, recover 0
#   granary2_chat_forced_qwenvocab_win28_recover1_lr1e4   forced,  win28, recover 1
#
# All three share the Qwen 151,669 vocabulary, window_frames=28, the target fix,
# and lr 1e-4. Each is averaged over its top-5 checkpoints by val_wer with NeMo's
# own scripts/checkpoint_averaging/average_model_checkpoints.py and evaluated on
# the full 74,838-utterance set through the shared select_shard partition.
#
# PER-ARM frame_trim IS NOT COSMETIC. _forced_alignment_loss sets
# joint.frame_trim = num_delay_frames (3) at TRAINING time, so the forced arms
# learn to emit from 11 of each chunk's 14 frames. _pin_inference_delay only acts
# when max_delay_frames > 0 -- it is 0 here -- so a restored .nemo decodes at the
# constructor default of 0 unless told otherwise, which is a DIFFERENT operating
# point from the one the model was trained and validated at. This project has
# already been burned by exactly that (a delay sweep reported at d=0 when it
# meant d=2). The banded arm never sets frame_trim, so it stays 0.
#
# loss_type must also be passed per arm: the win28 YAML says `banded`, and the
# forced arms override it via the launcher's env at train time. The averaging
# step now cross-checks {joint.window_frames, joint.history_chunks,
# joint.chunk_size, loss_type} against the checkpoint's own hyper_parameters and
# REFUSES on a mismatch, so a wrong value here fails loudly instead of silently
# loading the weights into a differently-shaped objective.
#
# Arms run SEQUENTIALLY; a failure in one does not abandon the rest. An arm with
# no checkpoints yet is skipped with a message rather than failing the job.
#
# ENV
#   TOPK            checkpoints to average per arm (default 5)
#   FORCE_AVERAGE   1 to rebuild an averaged .nemo that already exists
# ============================================================================
set -uo pipefail

TOPK="${TOPK:-5}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
PROJECT="${PROJECT:-SpeechlmScriptCC}"
QWEN_TOK="${QWEN_TOK:-/lustre/fsw/portfolios/llmservice/users/heh/pretrained_models/huggingface/Qwen/Qwen3-1.7B}"
CFG=nemotron_chat_transducer_granary2_qwen_win28

FA=model.forced_alignment
# Reproduce the training config faithfully. target_construction and
# delay_word_final_punctuation are training-only knobs -- transcribe() never
# builds a target -- but they are recorded in the averaged .nemo, so passing
# them keeps it self-describing rather than claiming the `legacy` defaults
# these arms did not train with. band_chunks is likewise inert on the forced
# path; it is passed uniformly because all three arms set it.
COMMON="${FA}.num_delay_frames=3 ${FA}.max_delay_frames=0 model.joint.history_chunks=0 ${FA}.band_chunks=1 ${FA}.target_construction=partition ${FA}.delay_word_final_punctuation=true"

# exp_name | model overrides | frame_trim
ARM_LIST=(
  "granary2_chat_banded1_qwenvocab_win28_lr1e4|model.loss_type=banded ${FA}.recover_history_words=0 ${COMMON}|"
  "granary2_chat_forced_qwenvocab_win28_lr1e4|model.loss_type=forced_alignment ${FA}.recover_history_words=0 ${COMMON}|3"
  "granary2_chat_forced_qwenvocab_win28_recover1_lr1e4|model.loss_type=forced_alignment ${FA}.recover_history_words=1 ${COMMON}|3"
)

# Under sbatch $0 is a copy in Slurm's spool directory, so dirname "$0" has no
# sibling eval_chat.sh; SLURM_SUBMIT_DIR is where the sbatch was issued.
find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_chat.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate eval_chat.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
LAUNCH_DIR="$(find_launch_dir)"

echo "==> overnight eval of ${#ARM_LIST[@]} CHAT arms, one after another"
date
declare -a STATUS=()

for entry in "${ARM_LIST[@]}"; do
    exp="${entry%%|*}"; rest="${entry#*|}"
    overrides="${rest%|*}"; trim="${rest##*|}"
    CKPTS="${OUTPUT_PREFIX}/results/${PROJECT}/${exp}/${exp}/checkpoints"

    echo
    echo "############################################################"
    echo "### ${exp}"
    echo "###   frame_trim=${trim:-0}  overrides=${overrides}"
    if [[ -d "$CKPTS" ]]; then
        EPOCH="$(ls -1 "$CKPTS"/*.ckpt 2>/dev/null | grep -oE 'epoch=[0-9]+' | sort -t= -k2 -n | tail -1)"
        BEST="$(ls -1 "$CKPTS"/*.ckpt 2>/dev/null | grep -oE 'val_wer=[0-9.]+' | sort -t= -k2 -g | head -1)"
        echo "###   max ${EPOCH:-epoch=?}  best ${BEST:-val_wer=?}"
    fi
    echo "############################################################"

    if [[ ! -d "$CKPTS" ]] || [[ -z "$(ls -A "$CKPTS"/*.ckpt 2>/dev/null)" ]]; then
        echo "    SKIPPED: no checkpoints at ${CKPTS}" >&2
        STATUS+=("${exp}|skipped (no checkpoints)")
        continue
    fi

    ARM_EXP_NAME="${exp}" ARM_MODEL_OVERRIDES="${overrides}" \
    ARM_CONFIG_NAME="${CFG}" ARM_TOKENIZER_DIR="${QWEN_TOK}" \
    TOPK="${TOPK}" FORCE_AVERAGE="${FORCE_AVERAGE:-0}" \
    EVAL_TAG="avg${TOPK}" FRAME_TRIM="${trim}" \
        bash "${LAUNCH_DIR}/eval_chat.sh"
    rc=$?
    if [[ $rc -eq 0 ]]; then STATUS+=("${exp}|ok"); else
        echo "    FAILED (exit ${rc}); continuing with the next arm" >&2
        STATUS+=("${exp}|FAILED (exit ${rc})")
    fi
done

echo
echo "############################################################"
echo "### summary"
echo "############################################################"
date
for s in "${STATUS[@]}"; do printf '  %-56s %s\n' "${s%%|*}" "${s#*|}"; done
