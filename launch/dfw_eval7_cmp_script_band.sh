#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:lb7-cmp-script-band
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
# pool0-01815 carries an old NVIDIA driver (12020) and fails torch's CUDA init
# outright; it killed job 18686485 in 103s. pool0-00407 is the reference recipe's
# known-bad node.
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# Open-ASR-Leaderboard eval for a SCRIPT model on the CW DFW cluster.
#
#   sbatch launch/dfw_eval_script.sh dfw_granary2_script_baseline
#   sbatch launch/dfw_eval_script.sh dfw_granary2_script_banded1 14
#
# The DFW port of eval_script.sh. It sets DFW paths and then execs the SAME
# shared backend (eval_leaderboard.sh) the OCI evals use, so a DFW number is
# produced by exactly the same code path -- averaging, sharding, normalisation
# and scoring -- as every number already collected on OCI. Only the paths differ,
# which is what keeps the two clusters' results comparable.
#
# The only change the backend needed was making H_DIR overridable: it defaulted
# to an OCI-only directory, and container mounts fail outright on a path that
# does not exist.
#
# WHY interactive: DFW's batch partition is 1850 nodes and rarely contended, but
# eval is single-node and short, and interactive has the same 4h limit here. The
# one-job-per-user rule that made this awkward on OCI does not appear to apply.
#
# POSITIONAL ARGS
#   $1  EXP_NAME    experiment to evaluate  (REQUIRED)
#   $2  CHUNK_SIZE  decode chunk size in encoder frames (default 14)
#
# NOTE ON CHUNK SIZE: use one the model actually trained on. The banded arms are
# pinned to 14; the baseline draws from [2,7,10,14], so it can be evaluated at
# any of those -- but compare arms at the SAME size or the number means nothing.
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-${DFW}/hainanx}"
export CODE_DIR="${CODE_DIR:-${OUTPUT_PREFIX}/NeMo_SCRIPT_cc}"
export CONTAINER="${CONTAINER:-${DFW}/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
export CACHE_DIR="${CACHE_DIR:-${OUTPUT_PREFIX}/leaderboard_cache}"
# DFW's stand-in for the OCI llmservice/users/heh tree: it holds the pretrained
# models and data configs, and must be mounted for a restored .nemo to resolve
# anything it recorded at train time.
export H_DIR="${H_DIR:-${DFW}/users/heh}"
# The DFW-era wandb project, matching the training runs.
export PROJECT="${PROJECT:-SpeechlmCmp}"

# ============================================================================
# The WORD band (band_words=1, band_side=later). Trains on [14,28,42]; evaluated at 14.
#
# Evaluated from a hard-linked SNAPSHOT (cmp_script_band_lb), not the live training
# directory. The arm is still training with save_top_k=5, so its checkpoint set
# rotates -- and this harness also WRITES <exp>-averaged.ckpt back into the
# directory it reads. Pointing it at the live dir would both average a moving
# set and drop files into a running job's checkpoint folder.
# ============================================================================
export EXP_NAME="cmp_script_band_lb"

# Prompt-controlled model: the decode knobs must match what validation used, or
# the leaderboard number describes a different operating point than val_wer.
# cmp_script.yaml: val_num_delay_frames=3, val_capitalization/punctuation=true.
export PROMPT_CONTROL="${PROMPT_CONTROL:-1}"
export NUM_DELAY_FRAMES="${NUM_DELAY_FRAMES:-3}"
export CAPITALIZATION="${CAPITALIZATION:-1}"
export PUNCTUATION="${PUNCTUATION:-1}"
# band_token_cuts=false in both arms, so cuts are word-anchored at train time
# and decoding must impose the same rule.
export FORCE_WORD_START="${FORCE_WORD_START:-1}"
# Average the top-5, the project default; measured worth ~2% relative earlier
# (5.82 -> 5.69). Identical setting on all three arms so they stay comparable.
export RUN_AVERAGING="${RUN_AVERAGING:-1}"
export EVAL_TAG="${EVAL_TAG:-cmp_script_band_chunk7}"
# 14: the one chunk every arm shares and the size val_wer is computed at.
# CHUNK 7, not 14 -- a lower-latency operating point (0.56s vs 1.12s).
# OUT OF DISTRIBUTION: this arm trained on [14,28,42] only. 84 %% 7 == 0 so the
# geometry is legal and it will decode, but it never saw a 7-frame chunk. A gap
# here measures chunk-size generalisation, NOT the banded objective.
# The results leaf is chunk7_d3_c1_p1, so this cannot collide with the chunk-14
# run; and the cached top-5 average is reused, so both operating points are
# measured on byte-identical weights.
export CHUNK_SIZE="${CHUNK_SIZE:-7}"
export MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"

# Decode prompt -- MUST be byte-for-byte the training instruction. Drift here is
# silently out of distribution.
export SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are doing streaming speech recognition. Given the transcript so far and the representation of the next audio chunk, output the words spoken in that chunk.}"

if [[ ! -d "${CACHE_DIR}" ]] || [[ -z "$(ls -A "${CACHE_DIR}" 2>/dev/null)" ]]; then
    echo "ERROR: leaderboard cache is empty or missing at ${CACHE_DIR}" >&2
    echo "       Stage it first:  sbatch launch/dfw_stage_leaderboard_cache.sh" >&2
    exit 1
fi

CKPT_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/${EXP_NAME}/checkpoints"
echo "==> DFW SCRIPT leaderboard eval"
echo "    exp:        ${EXP_NAME}"
echo "    chunk_size: ${CHUNK_SIZE}"
echo "    ckpt dir:   ${CKPT_DIR}"
echo "    cache:      ${CACHE_DIR}"
[[ -d "$CKPT_DIR" ]] || echo "WARNING: no checkpoints directory at ${CKPT_DIR}" >&2

# Under sbatch $0 is a copy in Slurm's spool directory, and on a REQUEUE
# SLURM_SUBMIT_DIR comes back as the scratch root rather than the submit dir --
# which silently broke the CHAT arms for three hours. Hence the absolute fallback.
resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_leaderboard.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_leaderboard.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_leaderboard.sh" ]] && { echo "${here}"; return; }
    [[ -f "${CODE_DIR}/launch/eval_leaderboard.sh" ]] && { echo "${CODE_DIR}/launch"; return; }
    echo "ERROR: cannot locate eval_leaderboard.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}, CODE_DIR=${CODE_DIR})" >&2
    exit 1
}
exec bash "$(resolve_launch_dir)/eval_leaderboard.sh"
