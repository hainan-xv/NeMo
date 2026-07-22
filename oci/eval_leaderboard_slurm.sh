#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:sslm-leaderboard-eval
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
# Parallel Open-ASR-Leaderboard eval for a SpeechLM model, ON OCI: one node,
# one dataset per GPU (8 datasets over 8 GPUs), so the full suite finishes in
# ~one dataset's time instead of eight.
#
# Uses the TRACKED, offline driver scripts/speechlm_leaderboard_eval.py (synced
# to /code), which reads a PRE-STAGED wav/manifest cache on lustre (no HF
# download on compute nodes). Stage the cache once (the run_eval_sslm layout:
#   <CACHE_DIR>/<dataset>/<split>/_cache_manifest.jsonl  + 16kHz wavs
# e.g. rsync your local ~/leaderboard_run/cache to lustre).
#
# The checkpoint is already on lustre (the training run wrote it) -> no rsync.
#
# Usage:
#   cd /lustre/fsw/portfolios/llmservice/users/hainanx/NeMo79
#   sbatch oci/eval_leaderboard_slurm.sh <EXP_NAME>
#   # (defaults CACHE_DIR to /lustre/.../users/hainanx/leaderboard_cache; override with CACHE_DIR=)
#
# Key env:
#   CACHE_DIR         pre-staged leaderboard cache root on lustre
#                     (default /lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache)
#   PROJECT           results project dir (default Speechlm79)
#   MODEL_CLASS       model class (default ChunkCompletionSTTModel)
#   SYSTEM_PROMPT     decode prompt (match the model's training prompt!)
#   CHUNK_SIZE        decode chunk size override (encoder frames)
#   BATCH_SIZE        per-dataset batch size (default 32; OOM auto-halves)
#   MAX_NEW_TOKENS    per-chunk decode cap (default 64)
#   USE_LAST=1        eval the rolling -last.ckpt (default: best non-last by mtime)
#   STEP=<n>          eval step=<n>.ckpt explicitly
#   CKPT=<path>       eval this exact .ckpt (overrides EXP resolution)
#   DATASETS          space-separated 'name:split' list (default: full 8-set suite)
#   MAX_EVAL_SAMPLES  cap samples per dataset (fast smoke test)
#   OUTPUT_PREFIX     results root (default nemotron users/hainanx)
# ============================================================================

read_optional_token() { [[ -r "$1" ]] && tr -d '\r\n' < "$1" || true; }
HF_TOKEN="$(read_optional_token "$HOME/.hf_token")"

mkdir -p slurm_out
SLURM_ACCOUNT='llmservice'
USERID='users/hainanx'
OLDUSERID='users/heh'
LUSTRE_ACCOUNT_PREFIX=/lustre/fsw/portfolios/${SLURM_ACCOUNT}

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"

PROJECT="${PROJECT:-Speechlm79}"
EXP_NAME="${1:-${EXP_NAME:-granary2_chunkcompletion}}"
MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.chunk_completion_model.ChunkCompletionSTTModel}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-Transcribe the audio into text.}"
CHUNK_SIZE="${CHUNK_SIZE:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
USE_LAST="${USE_LAST:-0}"
STEP="${STEP:-}"
CKPT="${CKPT:-}"
DATASETS="${DATASETS:-librispeech:test.clean librispeech:test.other ami:test earnings22:test gigaspeech:test spgispeech:test tedlium:test voxpopuli:test}"

# Pre-staged leaderboard cache on lustre (rsync of ~/leaderboard_run/cache).
CACHE_DIR="${CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache}"
if [[ ! -d "$CACHE_DIR" ]]; then
    echo "ERROR: CACHE_DIR not found: $CACHE_DIR (stage the leaderboard cache there, or set CACHE_DIR=)." >&2
    exit 1
fi

OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
CKPT_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/${EXP_NAME}/checkpoints"

# ---- Resolve the checkpoint on lustre (no rsync; it's already here) ----
if [[ -z "$CKPT" ]]; then
    if [[ -n "$STEP" ]]; then
        CKPT="${CKPT_DIR}/step=${STEP}.ckpt"
    elif [[ "$USE_LAST" == "1" ]]; then
        CKPT="$(ls -t "${CKPT_DIR}"/*-last.ckpt 2>/dev/null | head -1)"
    else
        CKPT="$(ls -t "${CKPT_DIR}"/*.ckpt 2>/dev/null | grep -v -- '-last\.ckpt$' | head -1)"
        [[ -z "$CKPT" ]] && CKPT="$(ls -t "${CKPT_DIR}"/*-last.ckpt 2>/dev/null | head -1)"
    fi
fi
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "ERROR: could not resolve a checkpoint under ${CKPT_DIR} (set CKPT=, STEP=, or USE_LAST=1)." >&2
    exit 1
fi

RESULTS_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/leaderboard_eval$( [[ -n "$CHUNK_SIZE" ]] && echo "_chunk${CHUNK_SIZE}" )"
mkdir -p "$RESULTS_DIR"
HFCACHE="${OUTPUT_PREFIX}/hf_cache"; mkdir -p "$HFCACHE"
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo79/}"
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh
OCI_TMP_DIR="${OCI_TMP_DIR:-${RESULTS_DIR}/tmp}"

OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out
# Broad lustre mounts cover ckpt (nemotron), pretrained (llmservice/heh) and the
# staged cache wherever it lives; /code is our synced checkout.
MOUNTS="--container-mounts=${CODE_DIR}:/code,${H_DIR}:${H_DIR},${HFCACHE}:/hfcache/,/lustre/fsw:/lustre/fsw,/lustre/fs12:/lustre/fs12"

NGPU=8

read -r -d '' cmd <<EOF
echo "*******SpeechLM leaderboard eval (1 node, one dataset per GPU)********" \
&& echo "*** EXP=${EXP_NAME} | MODEL_CLASS=${MODEL_CLASS} ***" \
&& echo "*** CKPT=${CKPT} ***" \
&& echo "*** CACHE_DIR=${CACHE_DIR} | CHUNK_SIZE=${CHUNK_SIZE:-<default>} | BATCH_SIZE=${BATCH_SIZE} ***" \
&& nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
&& cd /code \
&& git rev-parse HEAD \
&& export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& export OMP_NUM_THREADS=1 \
&& export HF_HOME="/hfcache/" \
&& export HF_TOKEN=${HF_TOKEN} \
&& export HF_HUB_OFFLINE=1 \
&& export TOKENIZERS_PARALLELISM=false \
&& export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
&& export HYDRA_FULL_ERROR=1 \
&& export TMPDIR=${OCI_TMP_DIR} && mkdir -p ${OCI_TMP_DIR} \
&& python -c "import nemo, nemo.collections.speechlm2; print('NeMo:', nemo.__file__)" \
&& DATASETS_ARR=(${DATASETS}) \
&& echo "Launching \${#DATASETS_ARR[@]} datasets across ${NGPU} GPUs..." \
&& pids=() \
&& for i in "\${!DATASETS_ARR[@]}"; do \
      ds="\${DATASETS_ARR[\$i]}"; gpu=\$(( i % ${NGPU} )); \
      safe=\$(echo "\$ds" | tr ':/' '__'); \
      echo "  [gpu \$gpu] \$ds -> ${RESULTS_DIR}/\${safe}.log"; \
      python /code/scripts/speechlm_leaderboard_eval.py \
        --ckpt_path "${CKPT}" \
        --model_class "${MODEL_CLASS}" \
        --datasets "\$ds" \
        --device "\$gpu" \
        --cache_dir "${CACHE_DIR}" \
        --batch_size ${BATCH_SIZE} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --max_eval_samples ${MAX_EVAL_SAMPLES} \
        --system_prompt "${SYSTEM_PROMPT}" \
        --output_dir "${RESULTS_DIR}" \
        ${CHUNK_SIZE:+--chunk_size ${CHUNK_SIZE}} \
        > "${RESULTS_DIR}/\${safe}.log" 2>&1 & \
      pids+=(\$!); \
   done \
&& fail=0 && for p in "\${pids[@]}"; do wait "\$p" || fail=1; done \
&& echo "" \
&& echo "==================== Leaderboard WER ====================" \
&& printf "  %-28s %8s %10s\n" "Dataset" "WER(%)" "Time(s)" \
&& printf "  %-28s %8s %10s\n" "----------------------------" "--------" "----------" \
&& grep -h -P "^RESULT\t" ${RESULTS_DIR}/*.log 2>/dev/null | awk -F"\t" '{printf "  %-28s %8s %10s\n", \$2, \$3, \$4; if(\$3+0==\$3){s+=\$3; n++}} END{if(n>0){printf "  %-28s %8s\n", "----------------------------", "--------"; printf "  %-28s %8.2f\n", "Average", s/n}}' \
&& echo "" \
&& echo "Per-dataset logs + generations under: ${RESULTS_DIR}" \
&& exit \$fail
EOF

srun -o "$OUTFILE" -e "$ERRFILE" --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"

set +x
