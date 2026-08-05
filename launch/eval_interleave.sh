#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:sslm-lb-interleave
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
# Open-ASR-Leaderboard eval for the ORIGINAL interleaved SpeechLM baseline
# (heh's StreamingSTTModel), so we can compare interleaving VS our SCRIPT model.
#
# These checkpoints were trained by a colleague and live OUTSIDE our
# results/<PROJECT>/<EXP> layout, so unlike eval_script_*.sh this wrapper pins an
# EXTERNAL checkpoint via CKPT= (which disables averaging) and points EXP_CFG at
# that checkpoint's own exp_config.yaml (sibling of its checkpoints/ dir). It then
# execs the SAME pooled-shard, work-balanced backend (eval_leaderboard_slurm.sh):
# pool all utterances, global shuffle (fixed seed), 8 duration-sorted shards, one
# decode proc/GPU -> wall time ~= sum(all)/8.
#
# The interleaved model is a plain ASR model: MODEL_CLASS=StreamingSTTModel and
# the classic prompt "Transcribe the audio into text." (its training/inference
# default). Its 3-frame delay is baked into the weights; the one inference knob is
# CHUNK_SIZE (encoder frames/chunk). This is a MULTI chunk-size ("mcs") model, so
# with no override the backend uses the model's default (the LONGEST it saw, 28).
# We therefore default CHUNK_SIZE=14 and let you change it (env or 2nd arg).
#
# By default it logs into the SAME wandb eval project as the SCRIPT evals
# (PROJECT=SpeechlmRefactored -> <PROJECT>_leaderboard_eval), grouped by EXP_NAME
# (interleave_<variant>), so the stepped per-dataset WER lines overlay the SCRIPT
# runs directly.
#
# Usage (from the NeMo79 repo root on OCI):
#   sbatch launch/eval_interleave.sh                       # noblank_v2 (r1), chunk 14
#   sbatch launch/eval_interleave.sh readwrite             # read-write variant (r2), chunk 14
#   sbatch launch/eval_interleave.sh noblank 7             # chunk size 7
#   CHUNK_SIZE=28 sbatch launch/eval_interleave.sh         # chunk size 28 (env form)
#   sbatch launch/eval_interleave.sh --quick_run           # noblank, chunk 14, 10 utts/ds
#   for c in 7 14 28; do sbatch launch/eval_interleave.sh noblank $c; done  # chunk sweep
#
# Positional:
#   variant : noblank (default) | readwrite   -- which colleague checkpoint to eval
#   chunk   : encoder frames/chunk (default 14) -- also settable via CHUNK_SIZE=
#
# Flags (may appear anywhere among the args):
#   --quick_run[=N]   decode only the first N (default 10) utts of EACH dataset for
#                     a fast debug run; tags RESULTS_DIR + wandb run with _quick.
#
# The colleague's exp_config references the LLM/ASR by bare Hub id
# (e.g. Qwen/Qwen3-1.7B), which the offline (HF_HUB_OFFLINE=1) container can't
# fetch during HF conversion. We therefore hand the backend the SAME local,
# mounted mirrors our own SCRIPT configs use (heh's pretrained_models dir);
# patch_exp_config.py only swaps them in when the config value isn't already a
# valid local path. Override with PRETRAINED_LLM_OVERRIDE / PRETRAINED_ASR_OVERRIDE
# if this baseline used a different LLM/encoder.
#
# Optional env (forwarded to eval_leaderboard_slurm.sh):
#   CKPT=<path>       eval an arbitrary interleaved checkpoint (overrides variant)
#   EXP_CFG=<path>    its exp_config.yaml (default: <ckpt>/../../exp_config.yaml)
#   EXP_NAME=<name>   wandb group + results subdir (default: interleave_<variant>)
#   PROJECT=SpeechlmRefactored (default; shares the SCRIPT eval wandb project)
#   PRETRAINED_LLM_OVERRIDE / PRETRAINED_ASR_OVERRIDE   local LLM/ASR mirrors
#   BACKEND=heh|sslm  BATCH_SIZE=...  CHUNK_SIZE=...  DATASETS="..."  SHUFFLE_SEED=1234
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

# --- Flags (--quick_run), stripped out before reading the positional variant ---
QUICK_RUN=0
QUICK_N=10
POSITIONAL=()
for _arg in "$@"; do
    case "$_arg" in
        --quick_run|--quick-run) QUICK_RUN=1 ;;
        --quick_run=*|--quick-run=*) QUICK_RUN=1; QUICK_N="${_arg#*=}" ;;
        *) POSITIONAL+=("$_arg") ;;
    esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"
QUICK_SUFFIX=""
if (( QUICK_RUN )); then
    export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-$QUICK_N}"
    QUICK_SUFFIX="_quick"
fi

# --- Variant -> colleague checkpoint (full lustre paths from heh) ---
VARIANT="${1:-noblank}"
_ROOT="/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/heh/results/Streaming_SLM_debug"
case "$VARIANT" in
    noblank|noblank_v2)
        _RUN="oci_streaming_stt_granary2_lora_mcs_noblank_v2_lr0.0001_warmup10000_n8_delay3_rnd_compacttrue_r1_t1"
        CKPT_DEFAULT="${_ROOT}/${_RUN}/${_RUN}/checkpoints/step=108000.ckpt"
        EXP_NAME_DEFAULT="interleave_noblank"
        ;;
    readwrite|rw)
        _RUN="oci_streaming_stt_granary2_lora_mcs_readwrite_lr0.0001_warmup10000_n8_delay3_rnd_compacttrue_r2_t1"
        CKPT_DEFAULT="${_ROOT}/${_RUN}/${_RUN}/checkpoints/step=114000.ckpt"
        EXP_NAME_DEFAULT="interleave_readwrite"
        ;;
    *)
        echo "ERROR: unknown variant '${VARIANT}' (expected 'noblank' or 'readwrite')." >&2
        exit 1
        ;;
esac

# --- Chunk size: 2nd positional wins, else CHUNK_SIZE env, else 14 ---
CHUNK_ARG="${2:-}"
if [[ -n "$CHUNK_ARG" ]]; then
    if ! [[ "$CHUNK_ARG" =~ ^[0-9]+$ ]] || (( CHUNK_ARG < 1 )); then
        echo "ERROR: chunk must be a positive integer (got '$CHUNK_ARG')" >&2
        exit 1
    fi
    CHUNK_SIZE=$((10#$CHUNK_ARG))
else
    CHUNK_SIZE="${CHUNK_SIZE:-14}"
fi

# --- Model + run identity ---
CKPT="${CKPT:-$CKPT_DEFAULT}"
# The exp_config.yaml sits next to the checkpoints/ dir: <...>/<run>/<run>/exp_config.yaml.
EXP_CFG="${EXP_CFG:-$(dirname "$(dirname "$CKPT")")/exp_config.yaml}"
EXP_NAME="${EXP_NAME:-$EXP_NAME_DEFAULT}"
# Share the SCRIPT eval project so interleave and SCRIPT runs sit side by side.
PROJECT="${PROJECT:-SpeechlmRefactored}"
MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.streaming_stt_model.StreamingSTTModel}"
# Interleaved model's default ASR prompt (its training/inference framing).
SYSTEM_PROMPT="${SYSTEM_PROMPT:-Transcribe the audio into text.}"
# Local, mounted mirrors of the LLM + ASR (identical to our SCRIPT configs, since
# this is the same granary2 / Qwen3-1.7B / nemotron-streaming family). The backend
# only applies these if the checkpoint's exp_config value isn't a valid local path
# (i.e. it's a bare Hub id -> unreachable offline). Under H_DIR mount.
PRETRAINED_LLM_OVERRIDE="${PRETRAINED_LLM_OVERRIDE:-/lustre/fsw/portfolios/llmservice/users/heh/pretrained_models/huggingface/Qwen/Qwen3-1.7B}"
PRETRAINED_ASR_OVERRIDE="${PRETRAINED_ASR_OVERRIDE:-/lustre/fsw/portfolios/llmservice/users/heh/pretrained_models/huggingface/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo}"
# quick-run marker only; the backend also appends _chunk<n> + the launch timestamp.
EVAL_TAG=""
(( QUICK_RUN )) && EVAL_TAG="quick"

export SYSTEM_PROMPT MODEL_CLASS EXP_NAME PROJECT EVAL_TAG CKPT EXP_CFG CHUNK_SIZE
export PRETRAINED_LLM_OVERRIDE PRETRAINED_ASR_OVERRIDE

echo "==> interleaved-baseline leaderboard eval"
echo "    variant:       ${VARIANT}"
echo "    exp_name:      ${EXP_NAME}"
echo "    project:       ${PROJECT}"
echo "    model_class:   ${MODEL_CLASS}"
echo "    chunk_size:    ${CHUNK_SIZE}"
echo "    ckpt:          ${CKPT}"
echo "    exp_cfg:       ${EXP_CFG}"
echo "    llm_mirror:    ${PRETRAINED_LLM_OVERRIDE}"
echo "    asr_mirror:    ${PRETRAINED_ASR_OVERRIDE}"
(( QUICK_RUN )) && echo "    quick_run:     first ${QUICK_N} utts/dataset (MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES})"
echo "    system_prompt: ${SYSTEM_PROMPT}"
echo "    wandb_run:     ${EXP_NAME}${QUICK_SUFFIX} (+_chunk/_<launch-time> appended by backend)"

# Locate eval_leaderboard_slurm.sh (see note in eval_script_baseline.sh).
resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        if [[ -f "${SLURM_SUBMIT_DIR}/eval_leaderboard_slurm.sh" ]]; then
            echo "${SLURM_SUBMIT_DIR}"; return
        fi
        if [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_leaderboard_slurm.sh" ]]; then
            echo "${SLURM_SUBMIT_DIR}/launch"; return
        fi
    fi
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "${here}/eval_leaderboard_slurm.sh" ]]; then
        echo "${here}"; return
    fi
    echo "ERROR: cannot locate eval_leaderboard_slurm.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
LAUNCH_DIR="$(resolve_launch_dir)"
cd "${SLURM_SUBMIT_DIR:-$LAUNCH_DIR}"

# Already inside the sbatch allocation: run the shared pooled-shard body as a
# normal bash script (its own #SBATCH headers are ignored). $1 = EXP_NAME.
exec bash "${LAUNCH_DIR}/eval_leaderboard_slurm.sh" "${EXP_NAME}"
