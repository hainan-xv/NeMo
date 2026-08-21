#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:nemotron-rnnt-oldsuite
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1   # ONE task; the backend fans 8 python procs across 8 GPUs
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Leaderboard eval for the NEMOTRON 3.5 streaming RNNT model
# (nvidia/nemotron-speech-streaming-en-0.6b, an EncDecRNNTBPEModel -- the SAME
# base ASR the script/SpeechLM models fine-tune from), scored on the OLD/EARLIER
# 8-set leaderboard suite (UNCLEANED ami/gigaspeech/voxpopuli + tedlium).
#
# DECODE PATH -- read this, it's the whole point of the file:
#   The ONLY blessed way to run this model is `model.transcribe()` (see the model
#   -support test tests/e2e_nightly/test_model_support_*nemotron_speech_streaming*,
#   which does exactly that). Both NeMo ASR *example* entrypoints crash on it with
#   a "Floating point exception" (SIGFPE, integer div-by-zero in the streaming
#   setup):
#     - examples/asr/speech_to_text_eval.py                 (offline hydra path)
#     - examples/asr/asr_cache_aware_streaming/...infer.py  (cache-aware buffer)
#   `model.transcribe()` is exactly what launch/eval_parakeet_leaderboard.sh's
#   DEFAULT `transcribe` backend uses -- and that launcher is model-agnostic (any
#   .nemo via $1), so it IS the script that already runs the nemotron model on the
#   NEW datasets. This wrapper just points that same launcher at the OLD suite.
#
# THE COMBINATION (per "just combine the two scripts we already have"):
#   * launch/eval_parakeet_leaderboard.sh (transcribe backend) = run the nemotron
#     model on the leaderboard (the "run nemotron on the new datasets" half), and
#   * DATASETS = the old 8-set list the SCRIPT/flush old-suite runs use (the "score
#     on the old datasets" half; verbatim from eval_flush_model_oldsuite.sh).
# Same pre-staged cache, same pooled-shard sharding, same leaderboard-faithful
# scorer -> the number is directly comparable to eval_flush_model_oldsuite.sh's
# aggregate.log and to eval_promptctl_all_legacy.sh's 6.93.
#
# PREREQ: the OLD-suite datasets must be staged in CACHE_DIR in the standard
# layout <CACHE_DIR>/<name>/<split>/_cache_manifest.jsonl for names:
#   librispeech (test.clean/test.other), ami, earnings22, gigaspeech,
#   spgispeech, tedlium, voxpopuli   (test split).
#
# Usage (from the clean repo root on OCI):
#   sbatch launch/eval_nemotron_nemo_oldsuite.sh                 # nemotron .nemo, old 8-set
#   sbatch launch/eval_nemotron_nemo_oldsuite.sh /path/to/other.nemo   # $1 = .nemo path
#   MAX_EVAL_SAMPLES=10 sbatch launch/eval_nemotron_nemo_oldsuite.sh    # quick smoke test
#   BATCH_SIZE=32 sbatch launch/eval_nemotron_nemo_oldsuite.sh          # bigger transcribe batch
#   DATASETS="tedlium:test" sbatch launch/eval_nemotron_nemo_oldsuite.sh  # single dataset
#
# All other knobs (NEMO_MODEL, CACHE_DIR, BATCH_SIZE, MAX_EVAL_SAMPLES, SHUFFLE_SEED,
# CONTAINER, CODE_DIR, OUTPUT_PREFIX, REPORT_WANDB, ...) are env vars handled by the
# backend -- see eval_parakeet_leaderboard.sh's header.
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

# --- Pin the Nemotron 3.5 streaming RNNT .nemo (EncDecRNNTBPEModel). $1 overrides. ---
# Same path used as `pretrained_asr` across examples/speechlm2/conf/*.yaml.
NEMO_MODEL="${1:-${NEMO_MODEL:-/lustre/fsw/portfolios/llmservice/users/heh/pretrained_models/huggingface/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo}}"
export NEMO_MODEL

# --- Force the blessed decode backend: model.transcribe() (NOT the SIGFPE-ing
# NeMo-native speech_to_text_eval.py path). This is the default anyway; pin it so a
# stray BACKEND=nemo in the environment can't reintroduce the crash.
export BACKEND=transcribe

# --- Pin the OLD/EARLIER 8-set suite (uncleaned ami/gigaspeech/voxpopuli + tedlium).
# Verbatim from eval_flush_model_oldsuite.sh so the datasets match exactly. Overridable.
export DATASETS="${DATASETS:-librispeech:test.clean librispeech:test.other ami:test earnings22:test gigaspeech:test spgispeech:test tedlium:test voxpopuli:test}"

# --- Tag results so the nemotron old-suite runs don't collide with cleaned ones.
export EXP_NAME="${EXP_NAME:-nemotron-speech-streaming-en-0.6b_oldsuite}"
export EVAL_TAG="${EVAL_TAG:-oldsuite}"

echo "==> Nemotron 3.5 RNNT leaderboard eval on OLD 8-set suite (delegating to eval_parakeet_leaderboard.sh, transcribe backend)"
echo "    nemo_model: ${NEMO_MODEL}"
echo "    datasets:   ${DATASETS}"
echo "    exp_name:   ${EXP_NAME}"

# Locate eval_parakeet_leaderboard.sh. Under sbatch, Slurm COPIES this script into a
# spool dir, so BASH_SOURCE is unreliable -- prefer SLURM_SUBMIT_DIR (cwd at submit).
resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_parakeet_leaderboard.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_parakeet_leaderboard.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_parakeet_leaderboard.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate eval_parakeet_leaderboard.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
LAUNCH_DIR="$(resolve_launch_dir)"
cd "${SLURM_SUBMIT_DIR:-$LAUNCH_DIR}"

# Already inside the 8-GPU allocation: run the pooled-shard backend as a normal
# bash script (its own #SBATCH headers are ignored). $1 = the .nemo path.
exec bash "${LAUNCH_DIR}/eval_parakeet_leaderboard.sh" "${NEMO_MODEL}"
