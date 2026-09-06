#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:chat-lb-eval
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
# Open-ASR-Leaderboard eval for a CHAT transducer (ChatSTTModel).
#
#   ./oci_launch.sh launch/eval_chat.sh granary2_chat_asrvocab_win28_recover
#   RETRACT=1 ./oci_launch.sh launch/eval_chat.sh granary2_chat_asrvocab_win28_recover
#   RUN_AVERAGING=0 CKPT=<path> ./oci_launch.sh launch/eval_chat.sh <exp>
#
# WHY A SEPARATE SCRIPT FROM eval_leaderboard.sh. That driver calls generate()
# on a SpeechLM. CHAT decodes through the chunk-synchronous greedy transducer
# path (transcribe_ids) and has its own decode-time knob (retract_words), so it
# needs its own driver. Averaging, dataset list and scoring are deliberately
# identical, so a number here is comparable to the SCRIPT and nemotron columns.
#
# CHECKPOINT AVERAGING. Averaging the top-k checkpoints is usually worth a few
# tenths of WER and costs nothing at inference. ChatSTTModel is a plain
# LightningModule, not a NeMo ModelPT, so NeMo's checkpoint_averaging/ scripts
# do not apply -- they construct an EncDec* model and emit a .nemo. What they do
# mathematically is a mean over state_dict float tensors, which is exactly what
# scripts/average_script_ckpts.py does, model-agnostically (integer buffers such
# as step counters are taken from the first checkpoint rather than averaged,
# which would corrupt them).
#
# The averaged file is CACHED next to the checkpoints and reused unless
# FORCE_AVERAGE=1 or a newer input appeared.
#
# POSITIONAL
#   $1  EXP_NAME   experiment to evaluate (REQUIRED)
#
# ENV
#   RETRACT         retract-by-k decoding (default: the checkpoint's own setting)
#   RUN_AVERAGING   1 (default) average top-k non-last checkpoints
#   AVG_TOP_K       how many of the best checkpoints to average (default 5)
#   FORCE_AVERAGE   1 to recompute a cached average
#   CKPT            explicit checkpoint; disables averaging
#   DATASETS        comma-separated dataset:split (default: everything cached)
#   MAX_SAMPLES     cap utterances per dataset (smoke test)
#   BATCH_SIZE      default 8
# ============================================================================
set -euo pipefail
mkdir -p slurm_out

EXP_NAME="${1:-${EXP_NAME:-}}"
PROJECT="${PROJECT:-SpeechlmScriptCC}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
CACHE_DIR="${CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache}"
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_SCRIPT_cc}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"

if [[ -z "$EXP_NAME" ]]; then
    echo "ERROR: no experiment name given." >&2
    echo "usage: sbatch launch/eval_chat.sh <exp_name>" >&2
    ls -1 "${OUTPUT_PREFIX}/results/${PROJECT}" 2>/dev/null | grep -i chat | sed 's/^/  /' >&2 || true
    exit 1
fi

RETRACT="${RETRACT:-}"
RUN_AVERAGING="${RUN_AVERAGING:-1}"
AVG_TOP_K="${AVG_TOP_K:-5}"
FORCE_AVERAGE="${FORCE_AVERAGE:-0}"
CKPT="${CKPT:-}"
DATASETS="${DATASETS:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-8}"

CKPT_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/${EXP_NAME}/checkpoints"
RESULTS_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/leaderboard${RETRACT:+_retract${RETRACT}}"
mkdir -p "${RESULTS_DIR}"

# An explicit checkpoint always wins over (and disables) averaging.
if [[ -n "$CKPT" ]]; then
    RUN_AVERAGING=0
elif [[ "$RUN_AVERAGING" == "1" ]]; then
    CKPT="${CKPT_DIR}/${EXP_NAME}-averaged.ckpt"
    # Best-by-val_wer, not newest: with save_top_k the newest is often not the
    # best, and -last carries whatever the metric was when training stopped --
    # which for a job killed before its first validation is 0.0000.
    mapfile -t _AVG_IN < <(ls -1 "${CKPT_DIR}"/*.ckpt 2>/dev/null \
        | grep -v -- '-last\.ckpt$' | grep -v -- '-averaged\.ckpt$' \
        | grep -E 'val_wer=[0-9]+\.[0-9]+' \
        | sed -E 's/.*val_wer=([0-9.]+)\.ckpt/\1 &/' | sort -g | head -n "${AVG_TOP_K}" | cut -d' ' -f2-)
    if [[ ${#_AVG_IN[@]} -eq 0 ]]; then
        echo "ERROR: RUN_AVERAGING=1 but no scored checkpoints under ${CKPT_DIR}" >&2
        exit 1
    fi
    echo "==> Averaging ${#_AVG_IN[@]} best-val_wer checkpoint(s):"
    printf '      %s\n' "${_AVG_IN[@]##*/}"
else
    CKPT="$(ls -1 "${CKPT_DIR}"/*.ckpt 2>/dev/null | grep -E 'val_wer=[0-9]+\.[0-9]+' \
            | sed -E 's/.*val_wer=([0-9.]+)\.ckpt/\1 &/' | sort -g | head -1 | cut -d' ' -f2-)"
    [[ -z "$CKPT" ]] && { echo "ERROR: no checkpoint under ${CKPT_DIR}" >&2; exit 1; }
fi

echo "==> CHAT leaderboard eval"
echo "    exp:      ${EXP_NAME}"
echo "    ckpt:     ${CKPT}"
echo "    retract:  ${RETRACT:-<checkpoint default>}"
echo "    results:  ${RESULTS_DIR}"

AVG_CMD=""
if [[ "$RUN_AVERAGING" == "1" ]]; then
    AVG_TMP="${CKPT}.tmp.$$"
    printf '%s\n' "${_AVG_IN[@]}" > "${RESULTS_DIR}/avg_inputs.txt"
    # Write to a temp file and mv into place: mv is atomic within a filesystem,
    # so a concurrent eval either sees the old average or the new one, never a
    # half-written file.
    AVG_CMD="if [[ '${FORCE_AVERAGE}' == '1' ]] || [[ ! -s '${CKPT}' ]]; then \
        echo '==> averaging'; python /code/scripts/average_script_ckpts.py --output '${AVG_TMP}' \$(cat '${RESULTS_DIR}/avg_inputs.txt') \
        && mv -f '${AVG_TMP}' '${CKPT}'; else echo '==> reusing cached ${CKPT}'; fi && "
fi

# The cache lives on the llmservice portfolio; results on nemotron. Mount both.
MOUNTS="--container-mounts=${CODE_DIR}:/code,${OUTPUT_PREFIX}:${OUTPUT_PREFIX},${CACHE_DIR}:${CACHE_DIR}"

read -r -d '' CMD <<EOF || true
cd /code && export PYTHONPATH=/code:\${PYTHONPATH:-} && export HF_HOME=${OUTPUT_PREFIX}/hf_cache \
&& ${AVG_CMD} \
for i in \$(seq 0 \$((${NUM_GPUS} - 1))); do \
    CUDA_VISIBLE_DEVICES=\$i python /code/scripts/speechlm2/chat_leaderboard_eval.py \
        --ckpt '${CKPT}' --cache-dir '${CACHE_DIR}' --output-dir '${RESULTS_DIR}' \
        --shard \$i --num-shards ${NUM_GPUS} --batch-size ${BATCH_SIZE} \
        ${RETRACT:+--retract ${RETRACT}} ${DATASETS:+--datasets ${DATASETS}} ${MAX_SAMPLES:+--max-samples ${MAX_SAMPLES}} \
        > '${RESULTS_DIR}/shard'\$i.log 2>&1 & \
done; wait \
&& python /code/scripts/speechlm2/chat_leaderboard_eval.py --aggregate --output-dir '${RESULTS_DIR}' \
   2>&1 | tee '${RESULTS_DIR}/summary.txt'
EOF

srun --container-image="$CONTAINER" $MOUNTS bash -c "${CMD}"
