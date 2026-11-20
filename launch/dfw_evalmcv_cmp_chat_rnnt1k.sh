#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:mcv-cmp-chat-rnnt1k
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
# pool0-01815 carries an old NVIDIA driver and fails torch's CUDA init outright;
# pool0-00407 is the reference recipe's known-bad node.
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# Open-ASR-Leaderboard eval of the 1k purpose-built-vocabulary CHAT arm.
#
#   sbatch launch/dfw_eval_chat_spe1k.sh
#
# STREAMING (att_context_size [70,13]), unlike the full-context parakeet arm
# this was cloned from -- so NO --full_context, and eval pads 0.5s to flush the
# tail, matching data.dataset.pad_extra_duration at training.
#
# The tokenizer override is MANDATORY: averaging CONSTRUCTS the model, and with
# the Qwen path it would build a 151.7k-piece output layer for a 1k checkpoint.
set -euo pipefail

mkdir -p slurm_out

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-${DFW}/hainanx}"
export CODE_DIR="${CODE_DIR:-${OUTPUT_PREFIX}/NeMo_SCRIPT_cc}"
export CONTAINER="${CONTAINER:-${DFW}/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
export CACHE_DIR="${CACHE_DIR:-${OUTPUT_PREFIX}/leaderboard_cache}"
export H_DIR="${H_DIR:-${DFW}/users/heh}"
export PROJECT="${PROJECT:-SpeechlmCmp}"
# Mounted so the averaging step can reach the tokenizer; without it transformers
# treats the path as a hub repo id and fails with "Repo id must be in the form".
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-${DFW}:${DFW}}"

if [[ ! -d "${CACHE_DIR}" ]] || [[ -z "$(ls -A "${CACHE_DIR}" 2>/dev/null)" ]]; then
    echo "ERROR: leaderboard cache is empty or missing at ${CACHE_DIR}" >&2
    echo "       Stage it first:  sbatch launch/dfw_stage_leaderboard_cache.sh" >&2
    exit 1
fi

FA=model.forced_alignment
# A hard-linked SNAPSHOT of cmp_chat_rnnt1k, not the live training directory:
# the arm is still training with save_top_k=5, so its checkpoint set rotates,
# and averaging a moving set is not reproducible.
export ARM_EXP_NAME=cmp_chat_rnnt1k_lb
export ARM_CONFIG_NAME=cmp_chat_rnnt1k
# The 1k SentencePiece model this arm trained with, written into its own
# results dir at train time (the launcher mounts it as /results/tokenizer).
export ARM_TOKENIZER_DIR="${DFW}/hainanx/results/SpeechlmCmp/cmp_chat_rnnt1k/tokenizer"
# loss_type=rnnt, NOT banded. Averaging CONSTRUCTS the model and cross-checks
# loss_type against the checkpoint's own hyper_parameters, refusing on a
# mismatch -- so the banded overrides this file was cloned from would abort.
export ARM_MODEL_OVERRIDES="model.loss_type=rnnt"
export TOPK="${TOPK:-5}"
export FORCE_AVERAGE="${FORCE_AVERAGE:-0}"
export EVAL_TAG="${EVAL_TAG:-cmp_chat_rnnt1k_mcv}"
export FRAME_TRIM=""

CKPTS="${OUTPUT_PREFIX}/results/${PROJECT}/${ARM_EXP_NAME}/${ARM_EXP_NAME}/checkpoints"
NCK="$(ls -1 "$CKPTS"/*.ckpt 2>/dev/null | grep -vc -- '-last' || true)"
BEST="$(ls -1 "$CKPTS"/*.ckpt 2>/dev/null | grep -oE 'val_wer=[0-9.]+' | sort -t= -k2 -g | head -1 || true)"
echo "==> full-context CHAT leaderboard eval"
echo "    arm:   ${ARM_EXP_NAME}"
echo "    ckpts: ${NCK:-0} (best ${BEST:-val_wer=?}), averaging top ${TOPK}"
if [[ "${NCK:-0}" -lt 1 ]]; then
    echo "ERROR: no checkpoints at ${CKPTS}" >&2
    exit 1
fi
if [[ "${NCK:-0}" -lt "${TOPK}" ]]; then
    echo "NOTE: only ${NCK} checkpoints exist; the average will use those."
fi

# Under sbatch $0 is a copy in Slurm's spool dir, and on a REQUEUE
# SLURM_SUBMIT_DIR comes back as the scratch root -- hence the absolute fallback.
# ============================================================================
# THE VALIDATION SET, DECODED PROPERLY.
#
# Same 2000 MCV11-dev utterances that val_wer is computed on during training,
# staged into the leaderboard cache as mcv11_dev:dev from the very manifest the
# training runs point at. Run through the SAME decode path and the SAME
# normalizer as the 7-dataset leaderboard eval, so the number here is directly
# comparable BOTH to val_wer and to the leaderboard macro.
#
# The point: val_wer ranks cmp_chat_rnnt1k ahead of both SCRIPT arms while the
# leaderboard ranks it well behind. This isolates which half of that is the
# DATA (MCV11 is clean read speech) and which half, if any, is the measurement.
# ============================================================================
# KEEP_HISTORY: give this run its OWN results directory.
#
# eval_nemotron.sh builds its path as eval_<ckpt_ts>/<DECODE_LABEL>, and
# DECODE_LABEL is "chunk<N>_<mode>[_pad<x>]" -- it does NOT mention the dataset.
# So the leaderboard run and this MCV run of the SAME checkpoint resolve to the
# SAME directory, and whichever finishes last silently overwrites the other.
# That is exactly what happened on 2026-11-20: all five CHAT leaderboard results
# were replaced by their MCV counterparts, and the jobs still reported COMPLETED.
# (eval_leaderboard.sh, which the SCRIPT arms use, prefixes the dataset into the
# leaf and so was never affected.) KEEP_HISTORY=1 appends the job id, so the MCV
# run lands beside the leaderboard result instead of on top of it.
export KEEP_HISTORY="${KEEP_HISTORY:-1}"

export DATASETS="${DATASETS:-mcv11_dev:dev}"

resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_chat.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_chat.sh" ]] && { echo "${here}"; return; }
    [[ -f "${CODE_DIR}/launch/eval_chat.sh" ]] && { echo "${CODE_DIR}/launch"; return; }
    echo "ERROR: cannot locate eval_chat.sh" >&2
    exit 1
}
exec bash "$(resolve_launch_dir)/eval_chat.sh"
