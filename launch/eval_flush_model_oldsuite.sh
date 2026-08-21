#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:sslm-lb-script-flush-oldsuite
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1   # ONE task; fans 8 python procs across the 8 GPUs
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Leaderboard eval for a <flush>-trained PROMPT-CONTROLLED SCRIPT model, scored
# on the OLD/EARLIER 8-set leaderboard suite (UNCLEANED ami/gigaspeech/voxpopuli
# + tedlium) instead of the current cleaned 7-set. This is a thin wrapper around
# eval_flush_model.sh that only PINS the dataset suite, so you can compare the
# clean-repo flush model against the PRIOR model's 6.93 on the SAME datasets.
#
# It runs the CLEAN repo's code + the flush checkpoint (exactly like
# eval_flush_model.sh); ONLY the DATASETS differ. Everything else -- offline
# encode (USE_STATE_MACHINE=0), 0.5s trailing-silence pad, prompt build from
# chunk/delay/cap/punct, pooled 8-GPU shards, per-dataset + macro WER -- is
# identical.
#
# PREREQ: the OLD-suite datasets must be staged in CACHE_DIR in the standard
# layout <CACHE_DIR>/<name>/<split>/_cache_manifest.jsonl for names:
#   librispeech (test.clean/test.other), ami, earnings22, gigaspeech,
#   spgispeech, tedlium, voxpopuli   (test split).
# These are the SAME cache dir + names the legacy 8-set run used, so if
# eval_promptctl_all_legacy.sh reproduced 6.93 they're already present.
#
# POSITIONAL ARGS (identical to eval_flush_model.sh):
#   $1  MODEL_NAME  exp/model name        (default: granary2_script_promptctl_flush)
#   $2  CHUNK_SIZE  frames/chunk          (default: 14)
#   $3  DELAY       emission delay frames (default: 3)
#   $4  CAP         capitalization 1|0    (default: 1)
#   $5  PUNCT       punctuation 1|0       (default: 1)
#
# Usage (from the clean repo root on OCI):
#   sbatch launch/eval_flush_model_oldsuite.sh                                 # defaults, old 8-set
#   sbatch launch/eval_flush_model_oldsuite.sh granary2_script_promptctl_flush 14 3 0 0
#   for d in 1 3 6; do sbatch launch/eval_flush_model_oldsuite.sh granary2_script_promptctl_flush 14 $d 0 0; done
#
# Override the suite/cache if your staging differs: DATASETS="..." CACHE_DIR=... sbatch ...
# All other knobs (RUN_AVERAGING, CKPT/STEP/USE_LAST, BATCH_SIZE, USE_STATE_MACHINE,
# MAX_EVAL_SAMPLES, wandb, ...) are env vars handled downstream -- see
# eval_flush_model.sh / eval_promptctl.sh / eval_leaderboard.sh headers.
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

# --- Map positional args (same contract as eval_flush_model.sh) ---
MODEL_NAME="${1:-${EXP_NAME:-granary2_script_promptctl_flush}}"
CHUNK_SIZE="${2:-${CHUNK_SIZE:-14}}"
DELAY="${3:-${DELAY:-3}}"
CAP="${4:-${CAP:-1}}"
PUNCT="${5:-${PUNCT:-1}}"
export CHUNK_SIZE DELAY CAP PUNCT

# --- Offline encode (match training); same rationale as eval_flush_model.sh ---
export USE_STATE_MACHINE="${USE_STATE_MACHINE:-0}"

# --- Pin the OLD/EARLIER 8-set suite (uncleaned ami/gigaspeech/voxpopuli + tedlium).
# Verbatim from the previous backend's default DATASETS. Overridable via env.
export DATASETS="${DATASETS:-librispeech:test.clean librispeech:test.other ami:test earnings22:test gigaspeech:test spgispeech:test tedlium:test voxpopuli:test}"

# --- Tag results so the old-suite flush runs don't collide with the cleaned-suite ones.
export EVAL_TAG="${EVAL_TAG:-flush_oldsuite_c${CHUNK_SIZE}_d${DELAY}_cap${CAP}_punct${PUNCT}}"

echo "==> flush-model leaderboard eval on OLD 8-set suite (delegating to eval_promptctl.sh)"
echo "    model:      ${MODEL_NAME}"
echo "    chunk_size: ${CHUNK_SIZE}   delay: ${DELAY}   cap: ${CAP}   punct: ${PUNCT}"
echo "    datasets:   ${DATASETS}"
echo "    eval_tag:   ${EVAL_TAG}"

# Locate eval_promptctl.sh (same resolution as eval_flush_model.sh). Under sbatch,
# Slurm COPIES this script into a spool dir, so prefer SLURM_SUBMIT_DIR.
resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_promptctl.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_promptctl.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_promptctl.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate eval_promptctl.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
LAUNCH_DIR="$(resolve_launch_dir)"
cd "${SLURM_SUBMIT_DIR:-$LAUNCH_DIR}"

# eval_promptctl.sh builds the prompt from the exported knobs and execs the shared
# pooled-shard backend; DATASETS/CACHE_DIR/USE_STATE_MACHINE flow through as env.
exec bash "${LAUNCH_DIR}/eval_promptctl.sh" "${MODEL_NAME}"
