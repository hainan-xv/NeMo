#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:sslm-lb-promptctl-all-legacy-cleaned
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# LEGACY eval for the BEST prior prompt-controlled model, on the CURRENT CLEANED
# 7-set leaderboard suite -- runs the OLD code + OLD model, but scored on the
# SAME datasets as this repo's clean eval so the macro WER is directly
# COMPARABLE to the clean-repo best (6.06).
#
# It is a THIN WRAPPER around launch/eval_promptctl_all_legacy.sh: it pins the
# dataset suite + cache to the cleaned 7-set (ami_cleaned / gigaspeech_cleaned /
# voxpopuli_cleaned_aa, no tedlium) and tags results '_cleaned', then delegates.
# Everything else (OLD code mounted as /code, OLD checkpoint, prompt build) is
# identical to eval_promptctl_all_legacy.sh.
#
#   Original 8-set suite  -> launch/eval_promptctl_all_legacy.sh          (reproduces 6.93)
#   Cleaned  7-set suite  -> launch/eval_promptctl_all_legacy_cleaned.sh  (this; ~6.06-comparable)
#
# PREREQ: the CLEANED cache must be staged at CACHE_DIR with the clean-repo layout
#   <CACHE_DIR>/<name>/<split>/_cache_manifest.jsonl  (+ 16 kHz wavs)
# for names ami_cleaned, gigaspeech_cleaned, voxpopuli_cleaned_aa, earnings22,
# spgispeech, librispeech (test.clean/test.other). This is the same cache this
# repo's eval_leaderboard.sh uses, so it's already staged if clean-repo evals run.
#
# Usage (identical args to eval_promptctl_all_legacy.sh):
#   sbatch launch/eval_promptctl_all_legacy_cleaned.sh                    # BEST: cap nopunct 4 14 nocorrect
#   sbatch launch/eval_promptctl_all_legacy_cleaned.sh <cap> <punct> <delay> <chunk> <correct>
#   for d in 3 4; do sbatch launch/eval_promptctl_all_legacy_cleaned.sh cap nopunct $d 14 nocorrect; done
#
# Env overrides (same as the base legacy script): OLD_CODE_DIR, EXP_NAME, PROJECT,
# OUTPUT_PREFIX, LEGACY_BACKEND, BACKEND, RUN_AVERAGING, CKPT/STEP/USE_LAST, ...
# You may override DATASETS / CACHE_DIR here too if your cleaned cache differs.
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

# CLEANED 7-set suite (must match this repo's eval_leaderboard.sh DATASETS) +
# the shared staged cache. Override via env if your staging differs.
export DATASETS="${DATASETS:-librispeech:test.clean librispeech:test.other ami_cleaned:test earnings22:test gigaspeech_cleaned:test spgispeech:test voxpopuli_cleaned_aa:test}"
export CACHE_DIR="${CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache}"
# Distinguish these results from the original-suite runs.
export SUITE_TAG="${SUITE_TAG:-cleaned}"

echo "==> CLEANED-suite legacy promptctl_all eval (comparable to clean-repo 6.06)"
echo "    datasets:  ${DATASETS}"
echo "    cache_dir: ${CACHE_DIR}"

# Resolve the base legacy launcher (same dir under sbatch spool or launch/).
resolve_here() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_promptctl_all_legacy.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_promptctl_all_legacy.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_promptctl_all_legacy.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate eval_promptctl_all_legacy.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
BASE_DIR="$(resolve_here)"
cd "${SLURM_SUBMIT_DIR:-$BASE_DIR}"

# Delegate: DATASETS/CACHE_DIR/SUITE_TAG are exported so they survive the exec and
# flow through the base script to the OLD backend.
exec bash "${BASE_DIR}/eval_promptctl_all_legacy.sh" "$@"
