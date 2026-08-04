#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:chat-leaderboard-eval
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
# Parallel Open-ASR-Leaderboard eval for a CHAT / RNNT ASR model, ON OCI: one
# node, work BALANCED across 8 GPUs using the SAME pooled-shard machinery as the
# SpeechLM eval (oci/eval_leaderboard_slurm.sh) -- NOT one-dataset-per-GPU.
#
# We POOL every utterance across all datasets, shuffle with a fixed seed, and
# deal them round-robin into 8 duration-sorted shards (so each GPU gets an even
# 1/8 slice regardless of how differently sized the datasets are). Total wall
# time ~= sum(all)/8 instead of the single largest dataset.
#
# It REUSES the tracked pooled-shard helper scripts/leaderboard_heh_shards.py:
#   build      pool + shuffle + shard the pre-staged cache into shard{k}_of{N}.json
#   aggregate  reduce shard{k}_of{N}.generations.jsonl -> per-dataset WER + avg
# The ONLY ASR-specific piece is the per-shard DECODE: instead of heh's
# streaming_stt_generate.py we run scripts/asr_leaderboard_shard_decode.py, which
# loads the NeMo ASR model and transcribes the shard, writing the identical
# {dataset_key, text, pred_text} generations format. Both ref and hyp are
# normalized with the SAME whisper EnglishTextNormalizer heh uses, so the WER is
# directly comparable to the SpeechLM leaderboard runs in the same project.
#
# Reads a PRE-STAGED wav/manifest cache on lustre (no HF audio download on the
# compute node), the SAME layout the SpeechLM eval uses:
#   <CACHE_DIR>/<dataset>/<split>/_cache_manifest.jsonl  + 16kHz wavs
# The checkpoint is already on lustre (the training run wrote it) -> no rsync.
#
# Usage:
#   cd /lustre/fsw/portfolios/llmservice/users/hainanx/NeMo79
#   sbatch oci/eval_leaderboard_chat_slurm.sh <EXP_NAME>
#
# Key env:
#   PROJECT           results project dir (default Chat79)
#   CACHE_DIR         pre-staged leaderboard cache root on lustre
#                     (default /lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache)
#   BATCH_SIZE        per-shard transcribe batch size (default 32)
#   USE_NORMALIZER    english (default) | basic | none  (match heh for comparability)
#   DATASETS          space/comma 'name:split' list (default: full 8-set suite)
#   SHUFFLE_SEED      seed for the pooled global shuffle (default 1234)
#   MAX_EVAL_SAMPLES  cap samples per dataset (fast smoke test; 0 = all)
#   CKPT=<path>       eval this exact .nemo/.ckpt (overrides EXP resolution)
#   STEP=<n>          eval step=<n>.ckpt explicitly
#   USE_LAST=1        eval the rolling -last.ckpt
#   PREFER_CKPT=1     use the best .ckpt even if a .nemo exists
#   CKPT_DIR=<path>   override the checkpoints dir (skip the layout autodetect)
#   CHUNK_SIZE        override CHAT joint chunk_size (full-context models)
#   MAX_SYMBOLS       override greedy max symbols per (chunk) step
#   ATT_CONTEXT_SIZE  override encoder att context, e.g. "[70,13]"
#   OUTPUT_PREFIX     results root (default nemotron users/hainanx)
#   NGPU              GPUs to fan across (default 8)
# ============================================================================

read_optional_token() { [[ -r "$1" ]] && tr -d '\r\n' < "$1" || true; }
HF_TOKEN="$(read_optional_token "$HOME/.hf_token")"

mkdir -p slurm_out
SLURM_ACCOUNT='llmservice'
LUSTRE_ACCOUNT_PREFIX=/lustre/fsw/portfolios/${SLURM_ACCOUNT}

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"

PROJECT="${PROJECT:-Chat79}"
EXP_NAME="${1:-${EXP_NAME:-}}"
[[ -n "$EXP_NAME" ]] || { echo "Usage: sbatch $0 <EXP_NAME>  (or set CKPT=/EXP_NAME=)" >&2; exit 1; }

BATCH_SIZE="${BATCH_SIZE:-32}"
USE_NORMALIZER="${USE_NORMALIZER:-english}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
SHUFFLE_SEED="${SHUFFLE_SEED:-1234}"
NGPU="${NGPU:-8}"
DATASETS="${DATASETS:-librispeech:test.clean librispeech:test.other ami:test earnings22:test gigaspeech:test spgispeech:test tedlium:test voxpopuli:test}"
DATASETS_CSV="$(echo "$DATASETS" | tr -s ' ' ',')"

# Optional decoding overrides (passed through to the ASR decode when set).
CHUNK_SIZE="${CHUNK_SIZE:-}"
MAX_SYMBOLS="${MAX_SYMBOLS:-}"
ATT_CONTEXT_SIZE="${ATT_CONTEXT_SIZE:-}"

# Pre-staged leaderboard cache on lustre (same layout as the SpeechLM eval).
CACHE_DIR="${CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache}"
if [[ ! -d "$CACHE_DIR" ]]; then
    echo "ERROR: CACHE_DIR not found: $CACHE_DIR (stage the leaderboard cache there, or set CACHE_DIR=)." >&2
    exit 1
fi

OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"

# ---- Resolve the checkpoints dir on lustre (CHAT ASR exp layout) ----
# chat_fullctx_ce*.sh set exp_dir=/results/<EXP> and name=<EXP> under a
# RESULTS_DIR of <OUTPUT_PREFIX>/results/<PROJECT>/<EXP>, i.e. checkpoints live at
# <OUTPUT_PREFIX>/results/<PROJECT>/<EXP>/<EXP>/<EXP>/checkpoints (triple <EXP>).
# We autodetect across a few layouts so this also works for other exp managers.
if [[ -z "${CKPT_DIR:-}" ]]; then
    for cand in \
        "${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/${EXP_NAME}/${EXP_NAME}/checkpoints" \
        "${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/${EXP_NAME}/checkpoints" \
        "${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/checkpoints"; do
        if [[ -d "$cand" ]]; then CKPT_DIR="$cand"; break; fi
    done
fi
if [[ -z "${CKPT_DIR:-}" || ! -d "$CKPT_DIR" ]]; then
    echo "ERROR: could not locate a checkpoints dir for EXP='${EXP_NAME}' under ${OUTPUT_PREFIX}/results/${PROJECT}/ ." >&2
    echo "       Set CKPT_DIR=<dir> or CKPT=<file> explicitly." >&2
    exit 1
fi

# ---- Resolve the model file (.nemo preferred; else best/last/step .ckpt) ----
if [[ -z "${CKPT:-}" ]]; then
    if [[ -n "${STEP:-}" ]]; then
        CKPT="${CKPT_DIR}/step=${STEP}.ckpt"
    elif [[ "${USE_LAST:-0}" == "1" ]]; then
        CKPT="$(ls -t "${CKPT_DIR}"/*-last.ckpt 2>/dev/null | head -1)"
    else
        if [[ "${PREFER_CKPT:-0}" != "1" ]]; then
            CKPT="$(ls -t "${CKPT_DIR}"/*.nemo 2>/dev/null | head -1)"
        fi
        if [[ -z "${CKPT:-}" ]]; then
            CKPT="$(ls -t "${CKPT_DIR}"/*.ckpt 2>/dev/null | grep -v -- '-last\.ckpt$' | head -1)"
            [[ -z "${CKPT:-}" ]] && CKPT="$(ls -t "${CKPT_DIR}"/*-last.ckpt 2>/dev/null | head -1)"
        fi
    fi
fi
if [[ -z "${CKPT:-}" || ! -f "$CKPT" ]]; then
    echo "ERROR: could not resolve a model file under ${CKPT_DIR} (set CKPT=, STEP=, USE_LAST=1, or PREFER_CKPT=1)." >&2
    exit 1
fi
echo "==> Model: ${CKPT}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
JOB_TAG="${SLURM_JOB_ID:-local$$}"
RESULTS_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/leaderboard_eval_chat_${RUN_TS}_${JOB_TAG}"
SHARD_DIR="${RESULTS_DIR}/shards"
mkdir -p "$SHARD_DIR"

HFCACHE="${OUTPUT_PREFIX}/hf_cache"; mkdir -p "$HFCACHE"
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo79/}"
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh
# Short node-local TMPDIR: Python multiprocessing AF_UNIX socket paths are capped
# at ~108 bytes and the deep timestamped RESULTS_DIR would overflow them.
OCI_TMP_DIR="${OCI_TMP_DIR:-/tmp/chat_eval_${SLURM_JOB_ID:-$$}}"

OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out
# Broad lustre mounts cover the ckpt (nemotron), the cache (llmservice) and /code.
MOUNTS="--container-mounts=${CODE_DIR}:/code,${H_DIR}:${H_DIR},${HFCACHE}:/hfcache/,/lustre/fsw:/lustre/fsw,/lustre/fs12:/lustre/fs12"

# ---- Record run config ----
{
    echo "# CHAT/RNNT leaderboard eval run config"
    echo "timestamp:         ${RUN_TS}"
    echo "slurm_job_id:      ${JOB_TAG}"
    echo "exp_name:          ${EXP_NAME}"
    echo "project:           ${PROJECT}"
    echo "checkpoint:        ${CKPT}"
    echo "batch_size:        ${BATCH_SIZE}"
    echo "use_normalizer:    ${USE_NORMALIZER}"
    echo "num_gpus:          ${NGPU}"
    echo "shuffle_seed:      ${SHUFFLE_SEED}"
    echo "max_eval_samples:  ${MAX_EVAL_SAMPLES}"
    echo "datasets:          ${DATASETS_CSV}"
    echo "cache_dir:         ${CACHE_DIR}"
    echo "chunk_size:        ${CHUNK_SIZE:-<model default>}"
    echo "max_symbols:       ${MAX_SYMBOLS:-<model default>}"
    echo "att_context_size:  ${ATT_CONTEXT_SIZE:-<model default>}"
    echo "results_dir:       ${RESULTS_DIR}"
} > "${RESULTS_DIR}/run_config.yaml"
echo "==> Wrote run config: ${RESULTS_DIR}/run_config.yaml"

read -r -d '' cmd <<EOF
echo "*******CHAT/RNNT leaderboard eval (pooled shards over ${NGPU} GPUs)********" \
&& echo "*** EXP=${EXP_NAME} | PROJECT=${PROJECT} ***" \
&& echo "*** CKPT=${CKPT} ***" \
&& echo "*** CACHE_DIR=${CACHE_DIR} | BATCH_SIZE=${BATCH_SIZE} | SEED=${SHUFFLE_SEED} | NORM=${USE_NORMALIZER} ***" \
&& nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
&& cd /code \
&& git rev-parse HEAD \
&& export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& export OMP_NUM_THREADS=1 \
&& export HF_HOME="/hfcache/" \
&& export HF_TOKEN=${HF_TOKEN} \
&& export TOKENIZERS_PARALLELISM=false \
&& export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
&& export HYDRA_FULL_ERROR=1 \
&& export TMPDIR=${OCI_TMP_DIR} && mkdir -p ${OCI_TMP_DIR} \
&& python -c "import nemo, nemo.collections.asr; print('NeMo:', nemo.__file__)" \
&& echo "==> Building ${NGPU} pooled shard manifests (seed=${SHUFFLE_SEED}) ..." \
&& python /code/scripts/leaderboard_heh_shards.py build \
      --cache_dir "${CACHE_DIR}" \
      --datasets "${DATASETS_CSV}" \
      --out_dir "${SHARD_DIR}" \
      --num_shards ${NGPU} \
      --shuffle_seed ${SHUFFLE_SEED} \
      --max_eval_samples ${MAX_EVAL_SAMPLES} \
&& rm -f ${SHARD_DIR}/shard*_of*.generations.jsonl \
&& echo "Fanning ${NGPU} shards across ${NGPU} GPUs (ASR transcribe)..." \
&& pids=() \
&& for gpu in \$(seq 0 \$(( ${NGPU} - 1 ))); do \
      log="${RESULTS_DIR}/shard_\${gpu}.log"; \
      man="${SHARD_DIR}/shard\${gpu}_of${NGPU}.json"; \
      gen="${SHARD_DIR}/shard\${gpu}_of${NGPU}.generations.jsonl"; \
      echo "  [gpu \$gpu] \${man} -> \${log}"; \
      CUDA_VISIBLE_DEVICES=\$gpu python /code/scripts/asr_leaderboard_shard_decode.py \
        --shard_manifest "\${man}" \
        --model_path "${CKPT}" \
        --output "\${gen}" \
        --device 0 \
        --batch_size ${BATCH_SIZE} \
        --use_normalizer ${USE_NORMALIZER} \
        ${CHUNK_SIZE:+--chunk_size ${CHUNK_SIZE}} \
        ${MAX_SYMBOLS:+--max_symbols ${MAX_SYMBOLS}} \
        ${ATT_CONTEXT_SIZE:+--att_context_size "${ATT_CONTEXT_SIZE}"} \
        > "\${log}" 2>&1 & \
      pids+=(\$!); \
   done \
&& fail=0 && for p in "\${pids[@]}"; do wait "\$p" || fail=1; done \
&& echo "" \
&& echo "==================== Leaderboard WER (CHAT/RNNT) ====================" \
&& python /code/scripts/leaderboard_heh_shards.py aggregate --out_dir "${SHARD_DIR}" 2>&1 | tee "${RESULTS_DIR}/aggregate.log" \
&& echo "" \
&& echo "Per-shard logs: ${RESULTS_DIR} | manifests+generations: ${SHARD_DIR}" \
&& exit \$fail
EOF

CMD_FILE="${RESULTS_DIR}/container_cmd.sh"
printf '%s\n' "$cmd" > "$CMD_FILE"
chmod +x "$CMD_FILE"
echo "==> Container command: ${CMD_FILE}"

srun -o "$OUTFILE" -e "$ERRFILE" --container-image="$CONTAINER" $MOUNTS bash "$CMD_FILE"

set +x
