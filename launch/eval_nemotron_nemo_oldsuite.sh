#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:nemotron-rnnt-oldsuite
#SBATCH -p interactive,batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH --time-min 01:00:00
# 1 GPU caps RAM (scheduler rejects --mem=0 for a single-GPU job); this eval is
# GPU-bound and needs little host RAM.
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# NeMo-native leaderboard eval for the NEMOTRON 3.5 streaming RNNT model
# (nvidia/nemotron-speech-streaming-en-0.6b, an EncDecRNNTBPEModel -- the SAME
# base ASR the script/SpeechLM models fine-tune from), scored on the OLD/EARLIER
# 8-set leaderboard suite (UNCLEANED ami/gigaspeech/voxpopuli + tedlium).
#
# This is a thin wrapper around eval_parakeet_nemo.sh (the maximally-canonical
# single-GPU NeMo-native reference: decode with examples/asr/speech_to_text_eval.py,
# re-score with scripts/rescore_nemo_eval.py for a leaderboard-faithful WER). It
# ONLY pins two things -- the .nemo model and the DATASETS suite -- so you get the
# baseline offline RNNT number on the SAME datasets the script/flush oldsuite runs
# are scored on (see eval_flush_model_oldsuite.sh's aggregate.log), for an
# apples-to-apples comparison against those streaming SCRIPT results.
#
# PREREQ: the OLD-suite datasets must be staged in CACHE_DIR in the standard
# layout <CACHE_DIR>/<name>/<split>/_cache_manifest.jsonl for names:
#   librispeech (test.clean/test.other), ami, earnings22, gigaspeech,
#   spgispeech, tedlium, voxpopuli   (test split).
# These are the SAME cache dir + names the flush oldsuite / legacy 8-set runs use.
#
# Usage (from the clean repo root on OCI):
#   sbatch launch/eval_nemotron_nemo_oldsuite.sh                 # nemotron .nemo, old 8-set
#   sbatch launch/eval_nemotron_nemo_oldsuite.sh /path/to/other.nemo   # $1 = .nemo path
#   MAX_SAMPLES=50 sbatch launch/eval_nemotron_nemo_oldsuite.sh  # quick smoke test
#   BATCH_SIZE=32 sbatch launch/eval_nemotron_nemo_oldsuite.sh   # bigger transcribe batch
#   DATASETS="tedlium:test" sbatch launch/eval_nemotron_nemo_oldsuite.sh  # single dataset
#
# All other knobs (NEMO_MODEL, CACHE_DIR, BATCH_SIZE, MAX_SAMPLES, CONTAINER,
# CODE_DIR, OUTPUT_PREFIX, ...) are env vars handled by the backend -- see
# eval_parakeet_nemo.sh's header.
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

# --- Pin the Nemotron 3.5 streaming RNNT .nemo (EncDecRNNTBPEModel). $1 overrides. ---
# Same path used as `pretrained_asr` across examples/speechlm2/conf/*.yaml.
NEMO_MODEL="${1:-${NEMO_MODEL:-/lustre/fsw/portfolios/llmservice/users/heh/pretrained_models/huggingface/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo}}"
export NEMO_MODEL

# --- Pin the OLD/EARLIER 8-set suite (uncleaned ami/gigaspeech/voxpopuli + tedlium).
# Verbatim from eval_flush_model_oldsuite.sh so the datasets match exactly. Overridable.
export DATASETS="${DATASETS:-librispeech:test.clean librispeech:test.other ami:test earnings22:test gigaspeech:test spgispeech:test tedlium:test voxpopuli:test}"

# --- Tag results so the nemotron old-suite runs don't collide with parakeet/cleaned ones.
export EXP_NAME="${EXP_NAME:-nemotron-speech-streaming-en-0.6b_oldsuite}"

echo "==> Nemotron 3.5 RNNT NeMo-native eval on OLD 8-set suite (delegating to eval_parakeet_nemo.sh)"
echo "    nemo_model: ${NEMO_MODEL}"
echo "    datasets:   ${DATASETS}"
echo "    exp_name:   ${EXP_NAME}"

# Locate eval_parakeet_nemo.sh. Under sbatch, Slurm COPIES this script into a spool
# dir, so BASH_SOURCE is unreliable -- prefer SLURM_SUBMIT_DIR (cwd at submit time).
resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_parakeet_nemo.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_parakeet_nemo.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_parakeet_nemo.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate eval_parakeet_nemo.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
LAUNCH_DIR="$(resolve_launch_dir)"
cd "${SLURM_SUBMIT_DIR:-$LAUNCH_DIR}"

# eval_parakeet_nemo.sh takes the .nemo path as $1; DATASETS/EXP_NAME flow as env.
exec bash "${LAUNCH_DIR}/eval_parakeet_nemo.sh" "${NEMO_MODEL}"
