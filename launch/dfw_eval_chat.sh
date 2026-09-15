#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-chat-lb-eval
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
#SBATCH --exclude=pool0-00407

# ============================================================================
# Open-ASR-Leaderboard eval of the CHAT arms on the CW DFW cluster.
#
#   sbatch launch/dfw_eval_chat.sh                 <- every arm with checkpoints
#   ARMS=dfw_granary2_chat_rnnt_1k sbatch launch/dfw_eval_chat.sh   <- just one
#
# Averages each arm's top-K checkpoints by val_wer and evaluates the result on the
# full 74,838-utterance suite, through the SAME shared backend (eval_chat.sh ->
# eval_nemotron.sh) the OCI evals use -- so DFW numbers are produced by identical
# code and remain comparable to what we collected there.
#
# PER-ARM OVERRIDES ARE MANDATORY, NOT COSMETIC. Averaging CONSTRUCTS the model
# before loading weights, so the config must describe the same architecture the
# checkpoint was trained with. The averaging script cross-checks
# {joint.window_frames, joint.history_chunks, joint.chunk_size, loss_type} against
# the checkpoint's own hyper_parameters and REFUSES on a mismatch -- a wrong value
# here fails loudly rather than silently loading weights into a different
# objective.
#
# THE THREE ARMS DIFFER IN WAYS THAT MATTER HERE:
#   rnnt_1k   1,024-piece vocabulary, NOT Qwen. The full RNN-T loss is intractable
#             at 151k (job 18617613 sat at step 0 for ten minutes), so this arm
#             alone uses the small vocab and the donor-extracted tokenizer.
#   forced    Qwen vocab, and _forced_alignment_loss sets joint.frame_trim=3 at
#             TRAINING time -- so it must DECODE at trim 3 or it is measured at an
#             operating point it never trained at. This project has already been
#             burned by exactly that.
#   banded1   Qwen vocab, band_chunks=1, band_side=later. frame_trim is untouched
#             by _banded_loss, so it decodes at 0.
#
# Arms run SEQUENTIALLY; one failure does not abandon the rest. An arm with no
# checkpoints is skipped with a message.
#
# ENV
#   ARMS            space-separated subset of arm names (default: all three)
#   TOPK            checkpoints to average per arm (default 5)
#   FORCE_AVERAGE   1 to rebuild an averaged .nemo that already exists
# ============================================================================
set -uo pipefail

mkdir -p slurm_out

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-${DFW}/hainanx}"
export CODE_DIR="${CODE_DIR:-${OUTPUT_PREFIX}/NeMo_SCRIPT_cc}"
export CONTAINER="${CONTAINER:-${DFW}/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
export CACHE_DIR="${CACHE_DIR:-${OUTPUT_PREFIX}/leaderboard_cache}"
export H_DIR="${H_DIR:-${DFW}/users/heh}"
export PROJECT="${PROJECT:-SpeechlmDFW}"

TOPK="${TOPK:-5}"
QWEN_TOK="${DFW}/users/heh/pretrained_models/huggingface/Qwen/Qwen3-1.7B"
FA=model.forced_alignment
COMMON="${FA}.num_delay_frames=3 ${FA}.max_delay_frames=0 model.joint.history_chunks=0 ${FA}.target_construction=partition ${FA}.delay_word_final_punctuation=true"

if [[ ! -d "${CACHE_DIR}" ]] || [[ -z "$(ls -A "${CACHE_DIR}" 2>/dev/null)" ]]; then
    echo "ERROR: leaderboard cache is empty or missing at ${CACHE_DIR}" >&2
    echo "       Stage it first:  sbatch launch/dfw_stage_leaderboard_cache.sh" >&2
    exit 1
fi

# exp_name | config | tokenizer ("" = extract the donor SentencePiece) | overrides | frame_trim
ALL_ARMS=(
  "dfw_granary2_chat_rnnt_1k|nemotron_chat_transducer_granary2||model.loss_type=rnnt ${FA}.band_chunks=1 ${FA}.recover_history_words=0 ${COMMON}|"
  "dfw_granary2_chat_forced|nemotron_chat_transducer_granary2_qwen|${QWEN_TOK}|model.loss_type=forced_alignment ${FA}.band_chunks=1 ${FA}.recover_history_words=0 ${COMMON}|3"
  "dfw_granary2_chat_banded1|nemotron_chat_transducer_granary2_qwen|${QWEN_TOK}|model.loss_type=banded ${FA}.band_chunks=1 ${FA}.band_side=later ${FA}.recover_history_words=0 ${COMMON}|"
)

WANTED="${ARMS:-}"
find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_chat.sh" ]] && { echo "${here}"; return; }
    # Absolute fallback: SLURM_SUBMIT_DIR is unreliable on a requeue.
    [[ -f "${CODE_DIR}/launch/eval_chat.sh" ]] && { echo "${CODE_DIR}/launch"; return; }
    echo "ERROR: cannot locate eval_chat.sh" >&2
    exit 1
}
LAUNCH_DIR="$(find_launch_dir)"

echo "==> DFW CHAT leaderboard eval"
date
declare -a STATUS=()

for entry in "${ALL_ARMS[@]}"; do
    IFS='|' read -r exp cfg tok overrides trim <<< "$entry"
    if [[ -n "$WANTED" ]] && [[ " $WANTED " != *" $exp "* ]]; then
        continue
    fi

    CKPTS="${OUTPUT_PREFIX}/results/${PROJECT}/${exp}/${exp}/checkpoints"
    echo
    echo "############################################################"
    echo "### ${exp}"
    echo "###   config=${cfg}  tokenizer=${tok:-<donor SentencePiece>}  frame_trim=${trim:-0}"
    if [[ -d "$CKPTS" ]]; then
        BEST="$(ls -1 "$CKPTS"/*.ckpt 2>/dev/null | grep -oE 'val_wer=[0-9.]+' | sort -t= -k2 -g | head -1)"
        NCK="$(ls -1 "$CKPTS"/*.ckpt 2>/dev/null | grep -vc -- '-last')"
        echo "###   ${NCK:-0} checkpoints, best ${BEST:-val_wer=?}"
    fi
    echo "############################################################"

    if [[ ! -d "$CKPTS" ]] || [[ -z "$(ls -A "$CKPTS"/*.ckpt 2>/dev/null)" ]]; then
        echo "    SKIPPED: no checkpoints at ${CKPTS}" >&2
        STATUS+=("${exp}|skipped (no checkpoints)")
        continue
    fi

    ARM_EXP_NAME="${exp}" ARM_MODEL_OVERRIDES="${overrides}" \
    ARM_CONFIG_NAME="${cfg}" ARM_TOKENIZER_DIR="${tok}" \
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
for s in "${STATUS[@]}"; do printf '  %-44s %s\n' "${s%%|*}" "${s#*|}"; done
