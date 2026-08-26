#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:nemotron-leaderboard-eval
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1   # ONE task; it fans 8 python procs across the 8 GPUs
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Open-ASR-Leaderboard eval for the cache-aware streaming NEMOTRON ASR model --
# the streaming baseline to compare SCRIPT against.
#
# Uses the SAME staged dataset cache, the SAME shard partition (same seed) and
# the SAME leaderboard normalizer as launch/eval_leaderboard.sh, because both
# drivers import scripts/leaderboard_common.py. A given utterance therefore lands
# in the same shard and is scored identically for both systems, so the two macro
# WERs are directly comparable.
#
# COMPARABLE LATENCY POINTS
#   Nemotron was trained for exactly four look-aheads:
#       chunk 14 (1.12s), chunk 7 (0.56s), chunk 2 (0.16s), chunk 1 (0.08s)
#   SCRIPT was trained on chunk sizes [2, 4, 7, 10, 14, 28]. The overlap -- and
#   therefore the only honest head-to-head points -- is {2, 7, 14}. The driver
#   REFUSES an unsupported chunk size rather than silently degrading, because
#   NeMo's set_default_att_context_size only warns.
#
# DECODE MODE
#   MODE=offline (default) encodes each utterance in one pass with attention
#     restricted to [left, chunk-1]. Since att_context_style is chunked_limited,
#     the look-ahead does not compound across layers, so no frame depends on
#     audio past its own chunk boundary. This matches how the SCRIPT eval
#     encodes, and is the apples-to-apples number.
#   MODE=streaming runs true cache-aware chunk-by-chunk decoding
#     (conformer_stream_step with carried KV/conv caches). Slower, and fp32-only,
#     but it is the real deployment number. Running both is a good sanity check:
#     they should agree closely.
#
# USAGE (from the repo root on the OCI login node)
#   sbatch launch/eval_nemotron.sh                 # chunk 14, offline
#   sbatch launch/eval_nemotron.sh 7               # chunk 7
#   MODE=streaming sbatch launch/eval_nemotron.sh 14
#   for c in 2 7 14; do sbatch launch/eval_nemotron.sh $c; done
#
# Or from your laptop:
#   ./oci_launch.sh launch/eval_nemotron.sh 14
#   ./oci_launch_interactive.sh MAX_EVAL_SAMPLES=32 launch/eval_nemotron.sh 14
#
# WANDB LAYOUT   (project: <PROJECT>_eval_v2 -- deliberately separate from the
#                 older <PROJECT>_leaderboard_eval, whose runs use the previous
#                 naming and would clutter the same page)
#   group    = <exp_name>_<checkpoint timestamp>   e.g. granary2_script_baseline_20260824_2117
#   run name = the decode configuration           e.g. chunk7_d6_c1_p0, chunk14_sm_se
#   So one group holds one MODEL VERSION, and each decode setting is a separate
#   line inside it -- directly comparable. Retraining produces a new group rather
#   than mixing old and new numbers on the same axis. Override with WANDB_GROUP /
#   WANDB_RUN_NAME.
#   Re-running the SAME checkpoint at the SAME setting reuses the line name on
#   purpose, so repeats overlay and their spread is visible.
# ============================================================================

# No `set -euo pipefail`: the heredoc read and the ls|grep pipelines below
# legitimately return non-zero.

read_optional_token() {
    [[ -r "$1" ]] && tr -d '\r\n' < "$1" || true
}
HF_TOKEN="$(read_optional_token "$HOME/.hf_token")"
WANDB_TOKEN="$(read_optional_token "$HOME/.wandb_token")"

mkdir -p slurm_out

CHUNK_SIZE="${1:-${CHUNK_SIZE:-14}}"
MODE="${MODE:-offline}"                     # offline | streaming
EXP_NAME="${EXP_NAME:-nemotron_streaming_0.6b}"
PROJECT="${PROJECT:-SpeechlmScriptCC}"

# The .nemo lives on fs12, NOT fsw -- it needs its own mount (see MOUNTS below).
MODEL_PATH="${MODEL_PATH:-/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/hainanx/pretrained_models/nemotron-speech-streaming-en-0.6b.nemo}"

BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
# MUST match the SCRIPT eval's seed or the two systems see different partitions.
SHUFFLE_SEED="${SHUFFLE_SEED:-1234}"
DTYPE="${DTYPE:-bf16}"                      # ignored in streaming mode (fp32-only)
NGPU="${NGPU:-8}"

DATASETS="${DATASETS:-librispeech:test.clean librispeech:test.other ami_cleaned:test earnings22:test gigaspeech_cleaned:test spgispeech:test voxpopuli_cleaned_aa:test}"
DATASETS_CSV="$(echo "$DATASETS" | tr -s ' ' ',')"

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_SCRIPT_cc}"
CACHE_DIR="${CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache}"
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh
HFCACHE="${OUTPUT_PREFIX}/hf_cache"
OCI_TMP_DIR="${OCI_TMP_DIR:-/tmp/nemotron_eval_${SLURM_JOB_ID:-$$}}"

if [[ ! -d "$CACHE_DIR" ]]; then
    echo "ERROR: leaderboard cache not found at ${CACHE_DIR}." >&2
    echo "       Stage it first:  sbatch launch/stage_leaderboard_cache.sh" >&2
    exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: model not found at ${MODEL_PATH} (set MODEL_PATH=)." >&2
    exit 1
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
JOB_TAG="${SLURM_JOB_ID:-local$$}"
EVAL_TAG="${EVAL_TAG:-${MODE}}"
RESULTS_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/leaderboard_eval_chunk${CHUNK_SIZE}_${EVAL_TAG}_${RUN_TS}_${JOB_TAG}"
SHARD_DIR="${RESULTS_DIR}/shards"
mkdir -p "$SHARD_DIR" "$HFCACHE"
OUTFILE="${RESULTS_DIR}/slurm-%j-%n.out"
ERRFILE="${RESULTS_DIR}/error-%j-%n.out"

REPORT_WANDB="${REPORT_WANDB:-auto}"
case "${REPORT_WANDB,,}" in
    auto)            [[ -n "$WANDB_TOKEN" ]] && REPORT_WANDB=1 || REPORT_WANDB=0 ;;
    1|true|yes|on)   REPORT_WANDB=1 ;;
    0|false|no|off)  REPORT_WANDB=0 ;;
    *) echo "WARNING: unrecognised REPORT_WANDB='${REPORT_WANDB}'; disabling." >&2; REPORT_WANDB=0 ;;
esac

cat > "${RESULTS_DIR}/run_config.yaml" <<YAML
timestamp: "${RUN_TS}"
job_id: "${JOB_TAG}"
exp_name: "${EXP_NAME}"
project: "${PROJECT}"
backend: "nemotron"
eval_tag: "${EVAL_TAG}"
model_path: "${MODEL_PATH}"
mode: "${MODE}"
chunk_size: ${CHUNK_SIZE}
chunk_seconds: $(python3 -c "print(f'{${CHUNK_SIZE} * 0.08:.2f}')" 2>/dev/null || echo "null")
dtype: "${DTYPE}"
batch_size: ${BATCH_SIZE}
num_gpus: ${NGPU}
shuffle_seed: ${SHUFFLE_SEED}
max_eval_samples: ${MAX_EVAL_SAMPLES}
datasets: "${DATASETS_CSV}"
cache_dir: "${CACHE_DIR}"
results_dir: "${RESULTS_DIR}"
YAML

echo "==> nemotron leaderboard eval"
echo "    model:      ${MODEL_PATH}"
echo "    mode:       ${MODE}   chunk_size: ${CHUNK_SIZE} frames"
echo "    results ->  ${RESULTS_DIR}"

# /lustre/fsw and /lustre/fs12 are separate autofs roots; the model lives on
# fs12 while the code/cache/results live on fsw, so BOTH must be bound. The
# broad catch-alls come first so ancestor binds do not shadow the leaves.
MOUNTS="--container-mounts=/lustre/fsw:/lustre/fsw,/lustre/fs12:/lustre/fs12,${CODE_DIR}:/code,${OUTPUT_PREFIX}:${OUTPUT_PREFIX},${CACHE_DIR}:${CACHE_DIR},${H_DIR}:${H_DIR},${HFCACHE}:/hfcache/"

# wandb identity: GROUP = <model>_<checkpoint timestamp>, RUN NAME = decode config.
# Same convention as launch/eval_leaderboard.sh so all three systems line up in
# one project. The .nemo file's mtime stands in for a checkpoint timestamp.
if [[ -e "$MODEL_PATH" ]]; then
    CKPT_TS="$(date -r "$MODEL_PATH" +%Y%m%d_%H%M 2>/dev/null || echo unknown)"
else
    CKPT_TS="unknown"
fi
DECODE_LABEL="chunk${CHUNK_SIZE:-default}_${MODE}"

WANDB_CLAUSE=""
if [[ "$REPORT_WANDB" == "1" ]]; then
    WANDB_GROUP="${WANDB_GROUP:-${EXP_NAME}_${CKPT_TS}}"
    WANDB_RUN_NAME="${WANDB_RUN_NAME:-${DECODE_LABEL}}"
    WANDB_EVAL_PROJECT="${WANDB_EVAL_PROJECT:-${PROJECT}_eval_v2}"
    WANDB_CLAUSE="&& { export WANDB_API_KEY='${WANDB_TOKEN}'; python /code/scripts/eval_wandb_report.py --project '${WANDB_EVAL_PROJECT}' --run_name '${WANDB_RUN_NAME}' --results_dir '${RESULTS_DIR}' --group '${WANDB_GROUP}' --job_type nemotron 2>&1 | tee '${RESULTS_DIR}/wandb_report.log' || true; }"
fi

read -r -d '' cmd <<EOF
echo "*******STARTING NEMOTRON LEADERBOARD EVAL********" \
&& nvidia-smi \
&& cd /code \
&& echo "CODE COMMIT:" && git rev-parse HEAD \
&& export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& export OMP_NUM_THREADS=1 \
&& export HF_HOME="/hfcache/" \
&& export HF_TOKEN=${HF_TOKEN} \
&& export HF_HUB_OFFLINE=1 \
&& export TOKENIZERS_PARALLELISM=false \
&& export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
&& export TMPDIR=${OCI_TMP_DIR} && mkdir -p ${OCI_TMP_DIR} \
&& python -c "import kaldialign; assert hasattr(kaldialign,'batch_error_rate')" 2>/dev/null || pip install -U --quiet kaldialign \
&& echo "Pooled datasets: ${DATASETS_CSV}" \
&& echo "Fanning ${NGPU} balanced shards across ${NGPU} GPUs (seed=${SHUFFLE_SEED})..." \
&& rm -f ${SHARD_DIR}/shard*_of*.generations.jsonl \
&& pids=() \
&& for gpu in \$(seq 0 \$(( ${NGPU} - 1 ))); do \
      log="${RESULTS_DIR}/shard_\${gpu}.log"; \
      echo "  [gpu \$gpu] shard \$gpu/${NGPU} -> \${log}"; \
      CUDA_VISIBLE_DEVICES=\$gpu python /code/scripts/nemotron_leaderboard_eval.py \
        --model_path "${MODEL_PATH}" \
        --datasets "${DATASETS_CSV}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${SHARD_DIR}" \
        --num_shards ${NGPU} \
        --shard_index \$gpu \
        --shuffle_seed ${SHUFFLE_SEED} \
        --device 0 \
        --batch_size ${BATCH_SIZE} \
        --max_eval_samples ${MAX_EVAL_SAMPLES} \
        --mode ${MODE} \
        --chunk_size ${CHUNK_SIZE} \
        --dtype ${DTYPE} \
        > "\${log}" 2>&1 & \
      pids+=(\$!); \
   done \
&& fail=0 && for p in "\${pids[@]}"; do wait "\$p" || fail=1; done \
&& echo "" \
&& echo "================ Nemotron Leaderboard WER (chunk=${CHUNK_SIZE}, ${MODE}) ================" \
&& python /code/scripts/nemotron_leaderboard_eval.py --aggregate --output_dir "${SHARD_DIR}" 2>&1 | tee "${RESULTS_DIR}/aggregate.log" \
${WANDB_CLAUSE} \
&& echo "" \
&& echo "Per-shard logs + generations under: ${RESULTS_DIR}" \
&& exit \$fail
EOF

CMD_FILE="${RESULTS_DIR}/container_cmd.sh"
printf '%s\n' "$cmd" > "$CMD_FILE"
chmod +x "$CMD_FILE"

srun -o "$OUTFILE" -e "$ERRFILE" --container-image="$CONTAINER" $MOUNTS bash "$CMD_FILE"
