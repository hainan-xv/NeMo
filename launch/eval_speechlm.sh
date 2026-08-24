#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:speechlm-lb-eval
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
# Open-ASR-Leaderboard eval for the ORIGINAL streaming SpeechLM -- the third
# system in the comparison, alongside SCRIPT and the cache-aware nemotron RNNT.
#
#   sbatch launch/eval_speechlm.sh            # chunk 14
#   sbatch launch/eval_speechlm.sh 7          # chunk 7
#   for c in 2 7 14; do ./oci_launch.sh launch/eval_speechlm.sh $c; done
#
# WHOSE MODEL, AND WHERE IT LIVES
#   Trained by heh with
#     .../heh/scripts/streaming_speechlm/oci_sslm_granary2_ft_lora_r1.sh
#   which resolves to
#     EXP  = oci_streaming_stt_granary2_lora_mcs_noblank_v2_lr0.0001_warmup10000
#            _n8_delay3_rnd_compacttrue_r1_t1
#     path = /lustre/fsw/portfolios/llmservice/users/heh/results/Streaming_SLM_debug/<EXP>/<EXP>/checkpoints
#   Two checkpoints exist: step=108000.ckpt and step=200000-last.ckpt. Training
#   reached its max_steps of 200000, so the -last checkpoint IS the final model
#   and is the default here.
#
#   NOTE we do NOT average. Averaging needs several non-last checkpoints and this
#   run kept only one, so there is nothing to average over.
#
# READS HIS TREE, WRITES OURS
#   The checkpoint is read from heh's portfolio (already visible: the container
#   mounts /lustre/fsw wholesale) but results are written under OUR OUTPUT_PREFIX
#   and PROJECT, so nothing is written into someone else's directory and the
#   numbers sit beside the SCRIPT runs for comparison.
#
# COMPARABILITY
#   Same staged dataset cache, same shard seed, same scorer as the SCRIPT and
#   nemotron drivers -- they share scripts/leaderboard_common.py, so a given
#   utterance lands in the same shard and is scored identically for all three.
#   The system prompt below is the one this model trained with
#   ("Transcribe the audio into text."), which is NOT the SCRIPT prompt; each
#   model must be given its own instruction or the decode is out of distribution.
#
# POSITIONAL ARGS
#   $1  CHUNK_SIZE   decode chunk size in encoder frames (default: 14)
#
# ENV
#   CKPT             exact checkpoint path (default: the step=200000-last above)
#   EXP_NAME         label for OUR results dir (default speechlm_heh_noblank_v2)
#   STREAMING_EMBS=1 use true cache-aware streaming perception instead of the
#                    batched offline embeddings (slower; the deployment number)
#   EMIT_DELAY_FRAMES  inference-time emission delay (default 0)
#   Everything else (DATASETS, BATCH_SIZE, MAX_NEW_TOKENS, MAX_EVAL_SAMPLES,
#   wandb, ...) is handled by eval_leaderboard.sh -- see its header.
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

CHUNK_SIZE="${1:-${CHUNK_SIZE:-14}}"

# --- where his model lives ---
HEH_RESULTS="${HEH_RESULTS:-/lustre/fsw/portfolios/llmservice/users/heh/results/Streaming_SLM_debug}"
HEH_EXP="${HEH_EXP:-oci_streaming_stt_granary2_lora_mcs_noblank_v2_lr0.0001_warmup10000_n8_delay3_rnd_compacttrue_r1_t1}"
CKPT="${CKPT:-${HEH_RESULTS}/${HEH_EXP}/${HEH_EXP}/checkpoints/step=200000-last.ckpt}"

# --- where OUR results go ---
PROJECT="${PROJECT:-SpeechlmScriptCC}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
EXP_NAME="${EXP_NAME:-speechlm_heh_noblank_v2}"

EVAL_DRIVER="${EVAL_DRIVER:-speechlm_leaderboard_eval.py}"
MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.streaming_stt_model.StreamingSTTModel}"

# This model's OWN training instruction -- deliberately not the SCRIPT prompt.
SYSTEM_PROMPT="${SYSTEM_PROMPT:-Transcribe the audio into text.}"

# CKPT is set explicitly, so eval_leaderboard.sh disables averaging by itself.
EVAL_TAG="${EVAL_TAG:-heh_v2}"

STREAMING_EMBS="${STREAMING_EMBS:-0}"
EMIT_DELAY_FRAMES="${EMIT_DELAY_FRAMES:-0}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-} --emit_delay_frames ${EMIT_DELAY_FRAMES}"
[[ "$STREAMING_EMBS" == "1" || "$STREAMING_EMBS" == "true" ]] && EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS} --streaming_embs"

export EXP_NAME PROJECT OUTPUT_PREFIX MODEL_CLASS EVAL_DRIVER SYSTEM_PROMPT CHUNK_SIZE EVAL_TAG CKPT EXTRA_EVAL_ARGS

echo "==> streaming SpeechLM leaderboard eval"
echo "    checkpoint: ${CKPT}"
echo "    chunk_size: ${CHUNK_SIZE} frames ($(python3 -c "print(f'{${CHUNK_SIZE}*0.08:.2f}')" 2>/dev/null || echo '?')s)"
echo "    embs:       $([[ "$STREAMING_EMBS" == "1" ]] && echo 'true cache-aware streaming' || echo 'offline (chunk-limited attention)')"
echo "    results ->  ${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}"
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    echo "       Set CKPT= explicitly, or check HEH_EXP." >&2
    exit 1
fi

# Locate the shared backend. Under sbatch, Slurm COPIES this script into a spool
# dir, so BASH_SOURCE is unreliable -- prefer SLURM_SUBMIT_DIR.
resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_leaderboard.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_leaderboard.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_leaderboard.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate eval_leaderboard.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>})" >&2
    exit 1
}
LAUNCH_DIR="$(resolve_launch_dir)"
cd "${SLURM_SUBMIT_DIR:-$LAUNCH_DIR}"

exec bash "${LAUNCH_DIR}/eval_leaderboard.sh" "${EXP_NAME}"
