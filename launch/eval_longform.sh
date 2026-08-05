#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:sslm-longform-eval
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
# LONG-FORM eval of a SINGLE checkpoint (SCRIPT *or* interleaving), so we can
# compare long-form robustness across the two designs on the SAME footing.
#
# Long-form sets are a handful of VERY long recordings (minutes to ~1h) shipped
# as NeMo manifests under LONGFORM_DIR (default 3 sets: tedlium_longform,
# earnings22_longform, apptek_callcenter_dialogues). Their audio_filepath is
# RELATIVE to each manifest's own dir; the shard builder
# (leaderboard_heh_shards.py build_longform) resolves them to absolute paths,
# tags each utt with its dataset (parent-dir name, so apptek's 14 per-locale
# manifests pool into ONE key), and DURATION-balances the few huge clips across
# the 8 GPUs (greedy longest-processing-time) so wall time ~= total_dur / 8.
#
# It then execs the SAME pooled-shard backend as every other eval
# (launch/eval_leaderboard_slurm.sh, heh engine): convert the ckpt to HF once,
# decode each shard with streaming_stt_generate.py (state-machine, batch_size=1
# since one clip can be ~1h), aggregate per-dataset WER, and log to wandb.
#
# Unlike eval_script_baseline.sh (which resolves a ckpt from our results tree),
# this takes the ckpt PATH directly (works for both our SCRIPT runs and a
# colleague's external interleaving runs) and derives EXP_CFG as the exp_config
# sibling of the checkpoint's checkpoints/ dir.
#
# !!! SCALE WARNING: the full 3-set suite is ~420h of audio (~53h/GPU) -- days
# of streaming decode. Start with --quick_run, and/or point LONGFORM_DIR at a
# single dataset subdir (e.g. .../longform/tedlium_longform), and/or cap with
# MAX_EVAL_SAMPLES=. Bump #SBATCH -t for real runs.
#
# Usage (from the NeMo79 repo root on OCI):
#   sbatch launch/eval_longform.sh script    <ckpt.ckpt>            # SCRIPT model
#   sbatch launch/eval_longform.sh interleave <ckpt.ckpt>           # interleaving model
#   sbatch launch/eval_longform.sh script    <ckpt.ckpt> 7          # chunk size 7
#   sbatch launch/eval_longform.sh script    <ckpt.ckpt> --quick_run
#   LONGFORM_DIR=/lustre/.../tedlium_longform sbatch launch/eval_longform.sh script <ckpt.ckpt> --quick_run
#
# Positional (order matters):
#   type   : script | interleave   -- selects MODEL_CLASS + default prompt
#   ckpt   : full path to the .ckpt to evaluate (REQUIRED)
#   chunk  : encoder frames/chunk (optional, default 14) -- also via CHUNK_SIZE=
#
# Flags (may appear anywhere):
#   --quick_run[=N]   decode only the first N (default 10) utts of EACH dataset
#                     for a fast debug run; tags RESULTS_DIR + wandb run _quick.
#
# Key env:
#   LONGFORM_DIR   root holding the long-form manifests on lustre
#                  (default /lustre/fsw/portfolios/nemotron/users/hainanx/longform).
#                  Point at a single dataset subdir to eval just that set.
#   SYSTEM_PROMPT  decode prompt. Defaults per type (see below); OVERRIDE this for
#                  a prompt-controlled SCRIPT model so it matches how it was trained.
#   EXP_NAME       wandb group + results subdir (default longform_<type>). Set a
#                  distinct name per checkpoint to keep runs from grouping together.
#   PROJECT        SpeechlmRefactored (default) -> wandb project <PROJECT>_longform_eval.
#   EXP_CFG        exp_config.yaml (default: sibling of the ckpt's checkpoints/ dir).
#   PRETRAINED_LLM_OVERRIDE / PRETRAINED_ASR_OVERRIDE   local LLM/ASR mirrors used
#                  during HF conversion when the exp_config references a bare Hub id
#                  (default: heh's Qwen3-1.7B + nemotron-streaming, for interleave).
#   CHUNK_SIZE / BATCH_SIZE(=1) / MAX_EVAL_SAMPLES / REPORT_WANDB=auto|1|0
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

# --- Flags (--quick_run), stripped before reading positionals ---------------
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

# --- Positional: type, ckpt, chunk ------------------------------------------
TYPE="${1:-}"
CKPT_ARG="${2:-}"
CHUNK_ARG="${3:-}"

if [[ -z "$TYPE" || -z "$CKPT_ARG" ]]; then
    echo "usage: sbatch launch/eval_longform.sh <script|interleave> <ckpt.ckpt> [chunk] [--quick_run]" >&2
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

# --- Per-type model class + default prompt + pretrained mirrors -------------
case "$TYPE" in
    script)
        MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"
        # Baseline SCRIPT decode prompt (byte-for-byte the training instruction in
        # launch/script_baseline.sh). OVERRIDE via SYSTEM_PROMPT= for a
        # prompt-controlled model, matching how THAT model was trained.
        SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are doing streaming speech recognition. You are given the text history so far, followed by the audio representation of the next chunk; output the words spoken in that chunk. The text history is:}"
        # Our SCRIPT configs already point at local pretrained paths -> no override.
        PRETRAINED_LLM_OVERRIDE="${PRETRAINED_LLM_OVERRIDE:-}"
        PRETRAINED_ASR_OVERRIDE="${PRETRAINED_ASR_OVERRIDE:-}"
        ;;
    interleave|interleaving)
        MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.streaming_stt_model.StreamingSTTModel}"
        SYSTEM_PROMPT="${SYSTEM_PROMPT:-Transcribe the audio into text.}"
        # External interleaving exp_configs reference the LLM/ASR by bare Hub id,
        # unreachable in the offline container; hand the backend local mirrors
        # (patch_exp_config.py only swaps them in if the config path isn't local).
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
# The exp_config.yaml sits next to the checkpoints/ dir: <run>/checkpoints/x.ckpt
# -> <run>/exp_config.yaml.
EXP_CFG="${EXP_CFG:-$(dirname "$(dirname "$CKPT")")/exp_config.yaml}"
EXP_NAME="${EXP_NAME:-longform_${TYPE}}"
PROJECT="${PROJECT:-SpeechlmRefactored}"
# Long-form gets its OWN wandb project so it never mixes with the short-form
# leaderboard runs; group by EXP_NAME so the 3 stepped per-dataset WER lines
# (apptek/earnings22/tedlium + avg) of different models overlay for comparison.
WANDB_EVAL_PROJECT="${WANDB_EVAL_PROJECT:-${PROJECT}_longform_eval}"

# Long-form recordings are enormous -> one utt per GPU at a time.
BATCH_SIZE="${BATCH_SIZE:-1}"
BACKEND="heh"

# Where the long-form manifests live on lustre (must sit under a mounted root:
# /lustre/fsw or /lustre/fs12). Override to a single dataset subdir to scope down.
LONGFORM_DIR="${LONGFORM_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/longform}"

# RESULTS_DIR + wandb run name (backend appends _chunk<N> + launch timestamp).
EVAL_TAG="longform${QUICK_SUFFIX}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXP_NAME}_chunk${CHUNK_SIZE}${QUICK_SUFFIX}}"

export TYPE MODEL_CLASS SYSTEM_PROMPT CKPT EXP_CFG EXP_NAME PROJECT
export WANDB_EVAL_PROJECT WANDB_RUN_NAME EVAL_TAG CHUNK_SIZE BATCH_SIZE BACKEND LONGFORM_DIR
export PRETRAINED_LLM_OVERRIDE PRETRAINED_ASR_OVERRIDE

echo "==> long-form eval (${TYPE})"
echo "    exp_name:      ${EXP_NAME}"
echo "    project:       ${PROJECT}  (wandb: ${WANDB_EVAL_PROJECT})"
echo "    model_class:   ${MODEL_CLASS}"
echo "    chunk_size:    ${CHUNK_SIZE}   batch_size: ${BATCH_SIZE}"
echo "    ckpt:          ${CKPT}"
echo "    exp_cfg:       ${EXP_CFG}"
echo "    longform_dir:  ${LONGFORM_DIR}"
[[ -n "$PRETRAINED_LLM_OVERRIDE" ]] && echo "    llm_mirror:    ${PRETRAINED_LLM_OVERRIDE}"
[[ -n "$PRETRAINED_ASR_OVERRIDE" ]] && echo "    asr_mirror:    ${PRETRAINED_ASR_OVERRIDE}"
(( QUICK_RUN )) && echo "    quick_run:     first ${QUICK_N} utts/dataset (MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES})"
echo "    wandb_run:     ${WANDB_RUN_NAME} (+_<launch-time> appended by backend)"
echo "    system_prompt: ${SYSTEM_PROMPT}"

# Locate eval_leaderboard_slurm.sh (see note in eval_script_baseline.sh): under
# sbatch, Slurm copies this file to a spool dir, so prefer SLURM_SUBMIT_DIR.
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
