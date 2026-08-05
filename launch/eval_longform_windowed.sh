#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:sslm-longform-win
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 04:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1   # ONE task; fans 8 python procs across the 8 GPUs
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# WINDOWED long-form eval of a SINGLE checkpoint (SCRIPT *or* interleaving).
#
# A SIMPLER long-form method than eval_longform.sh's whole-recording streaming:
# chop each recording into fixed ~WINDOW_SEC-second windows (snapped to a whole
# number of chunks, since one chunk = chunk_size * 0.08s; e.g. 60s -> 54 x 14
# frames = 60.48s), decode each window INDEPENDENTLY (a fresh decode with no
# cross-window state / no growing context), then concatenate a recording's
# per-window hypotheses in order to form its full transcript and score WER
# against the full reference. This bounds per-decode context (fast, no ~1h
# streaming state) at the cost of losing context across window boundaries.
#
# Windows are referenced as offset/duration into the original audio (NO audio is
# rewritten), tagged with utt_id + window_index; the window aggregator stitches
# them back. Same backend as the other evals (launch/eval_leaderboard_slurm.sh,
# heh engine + convert-once + wandb); only the shard build + reduce differ, keyed
# by LONGFORM_WINDOW_SEC.
#
# Takes the ckpt PATH directly (works for our SCRIPT runs AND a colleague's
# external interleaving runs) and derives EXP_CFG as the exp_config sibling of the
# checkpoint's checkpoints/ dir. The prompt is auto-set per model type.
#
# !!! SCALE: windowing does NOT reduce total audio (~420h for the full 3-set
# suite), it only bounds per-decode length; a full run is still huge. Start with
# --quick_run, scope via LONGFORM_DIR=.../<one_dataset>, and/or MAX_EVAL_SAMPLES=.
#
# Usage (from the NeMo79 repo root on OCI):
#   sbatch launch/eval_longform_windowed.sh script     <ckpt.ckpt>            # 60s windows
#   sbatch launch/eval_longform_windowed.sh interleave <ckpt.ckpt>
#   WINDOW_SEC=30 sbatch launch/eval_longform_windowed.sh script <ckpt.ckpt>  # 30s windows
#   sbatch launch/eval_longform_windowed.sh script <ckpt.ckpt> 14 --quick_run
#
# Positional (order matters):
#   type   : script | interleave   -- selects MODEL_CLASS + default prompt
#   ckpt   : full path to the .ckpt to evaluate (REQUIRED)
#   chunk  : encoder frames/chunk (optional, default 14) -- windows snap to a multiple
#
# Flags (may appear anywhere):
#   --quick_run[=N]   fast smoke test: window only the N (default 8) globally
#                     SHORTEST recordings; tags RESULTS_DIR + wandb run _quick.
#   --stratified[=K]  the SMALL representative set: shortest K (default 8) utts per
#                     minute bucket (2-5,...,40-60, capped at MAX_RANGE_MIN=60),
#                     which are THEN windowed. Tags runs _strat<max>m; takes
#                     precedence over --quick_run.
#
# Key env:
#   WINDOW_SEC     target window length in seconds (default 60; snapped to a whole
#                  number of chunk_size-frame chunks by the builder).
#   LONGFORM_DIR   long-form manifests root on lustre (default
#                  /lustre/fsw/portfolios/nemotron/users/hainanx/longform); point at
#                  a single dataset subdir to scope down.
#   SYSTEM_PROMPT  decode prompt (defaults per type; OVERRIDE for a prompt-controlled
#                  SCRIPT model to match its training prompt).
#   EXP_NAME       wandb group + results subdir (default longform_win_<type>).
#   PROJECT        SpeechlmRefactored (default) -> wandb project <PROJECT>_longform_eval.
#   EXP_CFG        exp_config.yaml (default: sibling of the ckpt's checkpoints/ dir).
#   PRETRAINED_LLM_OVERRIDE / PRETRAINED_ASR_OVERRIDE   local LLM/ASR mirrors for
#                  HF conversion of an external (bare-Hub-id) exp_config.
#   CHUNK_SIZE / BATCH_SIZE(=8) / MAX_EVAL_SAMPLES / REPORT_WANDB=auto|1|0
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

# --- Flags (--quick_run / --stratified), stripped before reading positionals -
QUICK_RUN=0
QUICK_N=8   # window only the 8 globally shortest recordings for a fast smoke test
STRATIFIED=0
STRAT_K=8   # per minute-bucket count for the small range-stratified set
POSITIONAL=()
for _arg in "$@"; do
    case "$_arg" in
        --quick_run|--quick-run) QUICK_RUN=1 ;;
        --quick_run=*|--quick-run=*) QUICK_RUN=1; QUICK_N="${_arg#*=}" ;;
        --stratified|--small) STRATIFIED=1 ;;
        --stratified=*|--small=*) STRATIFIED=1; STRAT_K="${_arg#*=}" ;;
        *) POSITIONAL+=("$_arg") ;;
    esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

# --stratified selects the small range-stratified set (shortest STRAT_K per minute
# bucket up to MAX_RANGE_MIN); windows are then duration-balanced across GPUs.
RUN_SUFFIX=""
if (( STRATIFIED )); then
    export LONGFORM_STRATIFIED=1
    export LONGFORM_PER_RANGE="${STRAT_K}"
    export LONGFORM_MAX_RANGE_MIN="${MAX_RANGE_MIN:-60}"
    RUN_SUFFIX="_strat${LONGFORM_MAX_RANGE_MIN%.*}m"
    (( QUICK_RUN )) && echo "NOTE: --stratified set; ignoring --quick_run." >&2
elif (( QUICK_RUN )); then
    export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-$QUICK_N}"
    RUN_SUFFIX="_quick"
fi

# --- Positional: type, ckpt, chunk ------------------------------------------
TYPE="${1:-}"
CKPT_ARG="${2:-}"
CHUNK_ARG="${3:-}"

if [[ -z "$TYPE" || -z "$CKPT_ARG" ]]; then
    echo "usage: sbatch launch/eval_longform_windowed.sh <script|interleave> <ckpt.ckpt> [chunk] [--quick_run]" >&2
    exit 1
fi
if [[ ! -f "$CKPT_ARG" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT_ARG}" >&2
    exit 1
fi

# Chunk size: 3rd positional wins, else CHUNK_SIZE env, else 14.
if [[ -n "$CHUNK_ARG" ]]; then
    if ! [[ "$CHUNK_ARG" =~ ^[0-9]+$ ]] || (( CHUNK_ARG < 1 )); then
        echo "ERROR: chunk must be a positive integer (got '$CHUNK_ARG')" >&2
        exit 1
    fi
    CHUNK_SIZE=$((10#$CHUNK_ARG))
else
    CHUNK_SIZE="${CHUNK_SIZE:-14}"
fi

# Window length (seconds); the builder snaps it to a whole number of chunks.
WINDOW_SEC="${WINDOW_SEC:-60}"

# --- Per-type model class + default prompt + pretrained mirrors -------------
case "$TYPE" in
    script)
        MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"
        # Baseline SCRIPT decode prompt (byte-for-byte the training instruction in
        # launch/script_baseline.sh). OVERRIDE via SYSTEM_PROMPT= for a
        # prompt-controlled model, matching how THAT model was trained.
        SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are doing streaming speech recognition. You are given the text history so far, followed by the audio representation of the next chunk; output the words spoken in that chunk. The text history is:}"
        PRETRAINED_LLM_OVERRIDE="${PRETRAINED_LLM_OVERRIDE:-}"
        PRETRAINED_ASR_OVERRIDE="${PRETRAINED_ASR_OVERRIDE:-}"
        ;;
    interleave|interleaving)
        MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.streaming_stt_model.StreamingSTTModel}"
        SYSTEM_PROMPT="${SYSTEM_PROMPT:-Transcribe the audio into text.}"
        PRETRAINED_LLM_OVERRIDE="${PRETRAINED_LLM_OVERRIDE:-/lustre/fsw/portfolios/llmservice/users/heh/pretrained_models/huggingface/Qwen/Qwen3-1.7B}"
        PRETRAINED_ASR_OVERRIDE="${PRETRAINED_ASR_OVERRIDE:-/lustre/fsw/portfolios/llmservice/users/heh/pretrained_models/huggingface/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo}"
        ;;
    *)
        echo "ERROR: unknown model type '${TYPE}' (expected 'script' or 'interleave')." >&2
        exit 1
        ;;
esac

# --- Model + run identity ---------------------------------------------------
CKPT="$CKPT_ARG"
EXP_CFG="${EXP_CFG:-$(dirname "$(dirname "$CKPT")")/exp_config.yaml}"
EXP_NAME="${EXP_NAME:-longform_win_${TYPE}}"
PROJECT="${PROJECT:-SpeechlmRefactored}"
# Shares the long-form wandb project with eval_longform.sh; the win<sec> tag in the
# run name keeps windowed runs distinct from the whole-recording streaming ones.
WANDB_EVAL_PROJECT="${WANDB_EVAL_PROJECT:-${PROJECT}_longform_eval}"

# Windows are short (~WINDOW_SEC), so batch several per GPU (heh OOM backoff guards).
BATCH_SIZE="${BATCH_SIZE:-8}"
BACKEND="heh"
export LONGFORM_WINDOW_SEC="${WINDOW_SEC}"

LONGFORM_DIR="${LONGFORM_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/longform}"

# RESULTS_DIR + wandb run name (backend appends _chunk<N> + launch timestamp).
EVAL_TAG="lfwin${WINDOW_SEC}s${RUN_SUFFIX}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXP_NAME}_win${WINDOW_SEC}s_chunk${CHUNK_SIZE}${RUN_SUFFIX}}"

export TYPE MODEL_CLASS SYSTEM_PROMPT CKPT EXP_CFG EXP_NAME PROJECT
export WANDB_EVAL_PROJECT WANDB_RUN_NAME EVAL_TAG CHUNK_SIZE BATCH_SIZE BACKEND LONGFORM_DIR
export PRETRAINED_LLM_OVERRIDE PRETRAINED_ASR_OVERRIDE

echo "==> windowed long-form eval (${TYPE})"
echo "    exp_name:      ${EXP_NAME}"
echo "    project:       ${PROJECT}  (wandb: ${WANDB_EVAL_PROJECT})"
echo "    model_class:   ${MODEL_CLASS}"
echo "    window_sec:    ${WINDOW_SEC} (snapped to a multiple of ${CHUNK_SIZE}-frame chunks by the builder)"
echo "    chunk_size:    ${CHUNK_SIZE}   batch_size: ${BATCH_SIZE}"
echo "    ckpt:          ${CKPT}"
echo "    exp_cfg:       ${EXP_CFG}"
echo "    longform_dir:  ${LONGFORM_DIR}"
[[ -n "$PRETRAINED_LLM_OVERRIDE" ]] && echo "    llm_mirror:    ${PRETRAINED_LLM_OVERRIDE}"
[[ -n "$PRETRAINED_ASR_OVERRIDE" ]] && echo "    asr_mirror:    ${PRETRAINED_ASR_OVERRIDE}"
if (( STRATIFIED )); then
    echo "    stratified:    shortest ${STRAT_K}/bucket, buckets up to ${LONGFORM_MAX_RANGE_MIN%.*} min (then windowed)"
elif (( QUICK_RUN )); then
    echo "    quick_run:     window the ${QUICK_N} globally shortest recordings (MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES})"
fi
echo "    wandb_run:     ${WANDB_RUN_NAME} (+_<launch-time> appended by backend)"
echo "    system_prompt: ${SYSTEM_PROMPT}"

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

exec bash "${LAUNCH_DIR}/eval_leaderboard_slurm.sh" "${EXP_NAME}"
