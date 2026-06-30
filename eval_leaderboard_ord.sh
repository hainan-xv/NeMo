#!/bin/bash
set -e
set -o pipefail

# ============================================================================
# Download a StreamingSTTModel checkpoint from the ORD grid and evaluate it on
# the Open ASR Leaderboard datasets LOCALLY (this box's GPU).
#
# This is the ORD counterpart of eval_leaderboard.sh (which targets OCI). It
# scp's the checkpoint from the ORD cluster (cs-oci-ord), then runs THIS repo's
# run_eval_sslm.py over the HF ESB / Open-ASR-Leaderboard suite. Use this for
# the ord3/ StreamingSTT models (e.g. the new_imend_qwen0p6b*.sh trio).
#
# IMPORTANT: it deliberately uses THIS repo (NeMo_ord_sync_d146_current) for
# both the eval driver and PYTHONPATH, because models trained with
# `model.encoder_subsampling_factor>1` (the trainable encoder subsampler) can
# ONLY be constructed by this checkout -- the older ~/Workplace/NeMo checkout
# lacks that config key and module.
#
# Usage:
#   ./eval_leaderboard.sh [--last] [--gpu N] <EXP_NAME> [STEP]
#     (no STEP)      -> best (non '-last') checkpoint by val-WER
#     --last         -> use the rolling -last.ckpt (same as USE_LAST=1)
#     --gpu N        -> evaluate on GPU N (same as the 3rd positional arg)
#     <STEP>         -> step=<STEP>.ckpt explicitly
#
# Examples:
#   ./eval_leaderboard_ord.sh ord_streaming_stt_granary1p1_lora_lr0.0001_warmup10000_n8_delay2_la1_chunk2_rnd_compact_imend_qwen0p6b_r4_t1
#   ./eval_leaderboard_ord.sh --last --gpu 1 <EXP_NAME>
#   QUICK_TEST=1 ./eval_leaderboard_ord.sh <EXP_NAME>
#   ./eval_leaderboard.sh baseline_imend_loss_sub2 12000 0
#
# Env overrides:
#   PROJECT          force the project dir on ORD (else try the defaults below)
#   BATCH_SIZE       eval batch size (default 128)
#   MAX_NEW_TOKENS   per-chunk decode cap (default 64)
#   MAX_EVAL_SAMPLES cap samples per dataset (fast iteration)
#   QUICK_TEST       1 -> 10 samples from ami/test only
#   ONLY             comma-separated dataset filter, e.g. "librispeech,ami"
#   FORCE_DOWNLOAD   1 -> re-scp even if a local copy exists (default: reuse cache)
#   SKIP_DOWNLOAD    deprecated no-op (cache reuse is now the default)
#   USE_LAST         1 -> use -last.ckpt instead of the best snapshot
#   RUN_AVERAGING    1 -> rsync ALL non '-last' .ckpt snapshots and average their
#                         state_dicts into <EXP>-averaged.ckpt, then eval that
#   FORCE_AVERAGE    1 -> recompute the averaged ckpt even if cached
#   VLLM             1 -> fast path: convert the ckpt + decode in the
#                         streaming-stt-eval container (vLLM) instead of the
#                         in-process run_eval_sslm.py. Requires docker + GPU.
#   VLLM_IMAGE       container image for the vLLM path (default: dongjig v2)
#   FORCE_CONVERT    1 -> re-run convert.sh even if the vLLM dir is cached
#   SYSTEM_PROMPT    override the system prompt
# ============================================================================

# ---------- ORD connection (cs-oci-ord) ----------
REMOTE_HOST="${REMOTE_HOST:-cs-oci-ord-login-01.nvidia.com}"
REMOTE_USER="${REMOTE_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/lustre/fsw/portfolios/llmservice/users/hainanx/results}"
# Try these project dirs in order (override with PROJECT=foo). Streaming_SLM_ORD3_new
# is the project used by the new_imend_qwen0p6b*.sh trio under ord3/.
PROJECT_CANDIDATES_DEFAULT=("Streaming_SLM_ORD3_new" "Streaming_SLM_ORD3")
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no"

# ---------- Local paths (THIS repo) ----------
NEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOCAL_CKPT_DIR="${NEMO_ROOT}/checkpoints"
RUN_EVAL_PY="${NEMO_ROOT}/run_eval_sslm.py"
if [ ! -f "$RUN_EVAL_PY" ]; then
    echo "ERROR: cannot find eval driver at ${RUN_EVAL_PY}" >&2
    exit 1
fi
# Force the eval to import nemo from THIS checkout (needed for subsampled models).
export PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH}"
export STREAMING_STT_MODEL_ROOT="${NEMO_ROOT}"

# HF token for the gated ESB datasets (hf-audio/esb-datasets-test-only-sorted).
# Hardcoded in the gitignored ${NEMO_ROOT}/.hf_token (NEVER committed) so the
# eval works without a `huggingface-cli login`. An existing $HF_TOKEN in the
# environment takes precedence; this only fills it in when unset.
if [ -z "${HF_TOKEN:-}" ] && [ -f "${NEMO_ROOT}/.hf_token" ]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${NEMO_ROOT}/.hf_token")"
    export HF_TOKEN
fi

# ---------- Arguments ----------
# Flags (--last, --gpu/--device N) may appear anywhere; positional args are
# still supported: <EXP_NAME> [STEP] [DEVICE_ID].
USE_LAST="${USE_LAST:-0}"
DEVICE_ID="${DEVICE_ID:-0}"
_POS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --last)               USE_LAST=1; shift ;;
        --gpu|--device)       DEVICE_ID="$2"; shift 2 ;;
        --gpu=*|--device=*)   DEVICE_ID="${1#*=}"; shift ;;
        -h|--help)            echo "Usage: $0 [--last] [--gpu N] <EXP_NAME> [STEP]"; exit 0 ;;
        --)                   shift; while [ $# -gt 0 ]; do _POS+=("$1"); shift; done ;;
        -*)                   echo "ERROR: unknown option: $1" >&2; exit 1 ;;
        *)                    _POS+=("$1"); shift ;;
    esac
done
EXP_NAME="${_POS[0]:-}"
STEP="${_POS[1]:-}"
[ -n "${_POS[2]:-}" ] && DEVICE_ID="${_POS[2]}"
[ -n "$EXP_NAME" ] || { echo "Usage: $0 [--last] [--gpu N] <EXP_NAME> [STEP]" >&2; exit 1; }
BATCH_SIZE="${BATCH_SIZE:-128}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-4}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
QUICK_TEST="${QUICK_TEST:-0}"
ONLY="${ONLY:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
# vLLM fast path: VLLM=1 converts the ckpt and decodes inside the colleague's
# streaming-stt-eval container instead of running run_eval_sslm.py in-process.
VLLM="${VLLM:-0}"
VLLM_IMAGE="${VLLM_IMAGE:-gitlab-master.nvidia.com/dongjig/nemo_containers/streaming-stt-eval:v2}"

# ---------- Resolve project dir on ORD ----------
if [ -n "${PROJECT:-}" ]; then
    PROJECT_CANDIDATES=("$PROJECT")
else
    PROJECT_CANDIDATES=("${PROJECT_CANDIDATES_DEFAULT[@]}")
fi
REMOTE_CKPT_DIR=""
for proj in "${PROJECT_CANDIDATES[@]}"; do
    candidate="${REMOTE_RESULTS_ROOT}/${proj}/${EXP_NAME}/${EXP_NAME}/checkpoints"
    if ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "[ -d '${candidate}' ]" 2>/dev/null; then
        REMOTE_CKPT_DIR="$candidate"
        echo "==> Resolved project: ${proj}"
        break
    fi
done
if [ -z "$REMOTE_CKPT_DIR" ]; then
    echo "ERROR: experiment '${EXP_NAME}' not found under any of:" >&2
    for proj in "${PROJECT_CANDIDATES[@]}"; do
        echo "       ${REMOTE_RESULTS_ROOT}/${proj}/${EXP_NAME}/${EXP_NAME}/checkpoints" >&2
    done
    echo "       Override with PROJECT=<name>." >&2
    exit 1
fi

# ---------- Step 1: pick + download (or average) the checkpoint ----------
if [ "${RUN_AVERAGING:-0}" = "1" ]; then
    # ---- Weight averaging over all non '-last' .ckpt snapshots ----
    # StreamingSTT writes plain Lightning .ckpt (no .nemo), so we rsync every
    # non '-last' checkpoint and average their state_dicts with
    # average_sslm_ckpts.py into a single <EXP>-averaged.ckpt.
    [ -n "$STEP" ] && echo "WARNING: RUN_AVERAGING=1 ignores explicit STEP=$STEP" >&2 && STEP=""
    AVG_SCRIPT="${NEMO_ROOT}/average_sslm_ckpts.py"
    [ -f "$AVG_SCRIPT" ] || { echo "ERROR: averager not found at $AVG_SCRIPT" >&2; exit 1; }
    AVG_DIR="${LOCAL_CKPT_DIR}/${EXP_NAME}/avg_inputs"
    LOCAL_CKPT_PATH="${LOCAL_CKPT_DIR}/${EXP_NAME}/${EXP_NAME}-averaged.ckpt"
    STEP="averaged"

    if [ -f "$LOCAL_CKPT_PATH" ] && [ -s "$LOCAL_CKPT_PATH" ] && [ "${FORCE_AVERAGE:-0}" != "1" ] && [ "${FORCE_DOWNLOAD:-0}" != "1" ]; then
        echo "==> Reusing cached averaged checkpoint (FORCE_AVERAGE=1 to rebuild): $LOCAL_CKPT_PATH"
    else
        echo "==> Listing non '-last' checkpoints on ORD..."
        REMOTE_CKPT_FILES=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
            "ls ${REMOTE_CKPT_DIR}/*.ckpt 2>/dev/null | grep -v -- '-last\.ckpt$' | xargs -r -n1 basename")
        [ -n "$REMOTE_CKPT_FILES" ] || { echo "ERROR: no non '-last' checkpoints in ${REMOTE_CKPT_DIR}" >&2; exit 1; }
        NUM_CKPTS=$(echo "$REMOTE_CKPT_FILES" | wc -l)
        echo "    Found ${NUM_CKPTS} checkpoint(s) to average:"
        echo "$REMOTE_CKPT_FILES" | sed 's/^/      - /'

        # Sync EXACTLY the current non '-last' set, then prune stale local ckpts so
        # we average ONLY these ${NUM_CKPTS}.
        mkdir -p "$AVG_DIR"
        CKPT_LIST_FILE=$(mktemp)
        printf '%s\n' "$REMOTE_CKPT_FILES" > "$CKPT_LIST_FILE"
        RSYNC_OPTS=(-vh --times --partial --files-from="$CKPT_LIST_FILE" -e "ssh $SSH_OPTS")
        [ "${FORCE_DOWNLOAD:-0}" = "1" ] && RSYNC_OPTS+=(--ignore-times)
        echo "==> Syncing ${NUM_CKPTS} checkpoint(s) from ORD via rsync..."
        if ! rsync "${RSYNC_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_DIR}/" "$AVG_DIR/"; then
            rm -f "$CKPT_LIST_FILE"; echo "ERROR: rsync of averaging checkpoints failed." >&2; exit 1
        fi
        rm -f "$CKPT_LIST_FILE"

        AVG_INPUTS=()
        shopt -s nullglob
        for f in "$AVG_DIR"/*.ckpt; do
            base=$(basename "$f")
            case "$base" in *-last.ckpt) continue ;; esac
            if printf '%s\n' "$REMOTE_CKPT_FILES" | grep -qxF "$base"; then
                AVG_INPUTS+=("$f")
            else
                echo "==> Pruning stale checkpoint (not in current set): $base"; rm -f "$f"
            fi
        done
        shopt -u nullglob
        [ "${#AVG_INPUTS[@]}" -eq "$NUM_CKPTS" ] || { echo "ERROR: expected ${NUM_CKPTS} ckpts, have ${#AVG_INPUTS[@]} in ${AVG_DIR}." >&2; exit 1; }

        echo "==> Averaging ${NUM_CKPTS} checkpoint(s) via average_sslm_ckpts.py ..."
        rm -f "$LOCAL_CKPT_PATH"
        python "$AVG_SCRIPT" --output "$LOCAL_CKPT_PATH" "${AVG_INPUTS[@]}"
        [ -f "$LOCAL_CKPT_PATH" ] || { echo "ERROR: averaging did not produce $LOCAL_CKPT_PATH" >&2; exit 1; }
        echo "==> Average complete ($(du -h "$LOCAL_CKPT_PATH" | cut -f1))"
    fi
else
    if [ -z "$STEP" ]; then
        if [ "${USE_LAST:-0}" = "1" ]; then
            echo "==> USE_LAST=1: finding most recent -last.ckpt on ORD..."
            REMOTE_LIST_CMD="ls -t ${REMOTE_CKPT_DIR}/*-last.ckpt"
        else
            echo "==> Finding best (non '-last') checkpoint on ORD..."
            REMOTE_LIST_CMD="ls -t ${REMOTE_CKPT_DIR}/*.ckpt 2>/dev/null | grep -v -- '-last\.ckpt$'"
        fi
        CKPT_FILENAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "${REMOTE_LIST_CMD} | head -1 | xargs -r basename")
        if [ -z "$CKPT_FILENAME" ] && [ "${USE_LAST:-0}" != "1" ]; then
            echo "    No best-WER checkpoint found; falling back to -last.ckpt..."
            CKPT_FILENAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
                "ls -t ${REMOTE_CKPT_DIR}/*-last.ckpt 2>/dev/null | head -1 | xargs -r basename")
        fi
        if [ -z "$CKPT_FILENAME" ]; then
            echo "ERROR: No checkpoints found in ${REMOTE_CKPT_DIR}" >&2
            exit 1
        fi
        STEP="${CKPT_FILENAME%.ckpt}"; STEP="${STEP#step=}"
        echo "    Found: ${CKPT_FILENAME} (step=${STEP})"
    else
        CKPT_FILENAME="step=${STEP}.ckpt"
    fi

    REMOTE_CKPT_PATH="${REMOTE_CKPT_DIR}/${CKPT_FILENAME}"
    LOCAL_CKPT_PATH="${LOCAL_CKPT_DIR}/${EXP_NAME}/${CKPT_FILENAME}"

    # Cache-first: if a non-empty local copy already exists, reuse it (so a full
    # run right after a QUICK_TEST does NOT re-pull the same checkpoint). Set
    # FORCE_DOWNLOAD=1 to re-fetch. SKIP_DOWNLOAD=1 is kept as a no-op alias.
    if [ -f "$LOCAL_CKPT_PATH" ] && [ -s "$LOCAL_CKPT_PATH" ] && [ "${FORCE_DOWNLOAD:-0}" != "1" ]; then
        echo "==> Reusing cached local checkpoint (FORCE_DOWNLOAD=1 to re-pull): $LOCAL_CKPT_PATH"
        echo "    ($(du -h "$LOCAL_CKPT_PATH" | cut -f1))"
    else
        echo "==> Downloading checkpoint from ORD..."
        echo "    Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_PATH}"
        echo "    Local:  ${LOCAL_CKPT_PATH}"
        mkdir -p "$(dirname "$LOCAL_CKPT_PATH")"
        if ! scp $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_PATH}" "$LOCAL_CKPT_PATH"; then
            # Don't leave a truncated file behind to be "reused" next time.
            [ -f "$LOCAL_CKPT_PATH" ] && [ ! -s "$LOCAL_CKPT_PATH" ] && rm -f "$LOCAL_CKPT_PATH"
            echo "ERROR: scp failed for ${REMOTE_CKPT_PATH}" >&2
            exit 1
        fi
        echo "==> Download complete ($(du -h "$LOCAL_CKPT_PATH" | cut -f1))"
    fi
fi

# ---------- Step 2: evaluate on the leaderboard datasets ----------
DATASETS=(
    "ami:test"
    "earnings22:test"
    "gigaspeech:test"
    "librispeech:test.clean"
    "librispeech:test.other"
    "spgispeech:test"
    "tedlium:test"
    "voxpopuli:test"
)

# Optional ONLY filter (dataset names, comma-separated).
if [ -n "$ONLY" ]; then
    IFS=',' read -r -a _only <<< "$ONLY"
    _filtered=()
    for entry in "${DATASETS[@]}"; do
        ename="${entry%%:*}"
        for want in "${_only[@]}"; do
            [ "$ename" = "$want" ] && _filtered+=("$entry")
        done
    done
    DATASETS=("${_filtered[@]}")
fi

EXTRA_ARGS=()
if [ "$QUICK_TEST" = "1" ]; then
    DATASETS=("ami:test")
    EXTRA_ARGS+=(--max_eval_samples 10 --verbose)
    echo "==> QUICK TEST: 10 samples from ami/test only"
fi
[ -n "$MAX_EVAL_SAMPLES" ] && EXTRA_ARGS+=(--max_eval_samples "$MAX_EVAL_SAMPLES")
[ -n "$SYSTEM_PROMPT" ] && EXTRA_ARGS+=(--system_prompt "$SYSTEM_PROMPT")

RESULTS_DIR="${NEMO_ROOT}/eval_results/${EXP_NAME}_step${STEP}"
mkdir -p "$RESULTS_DIR"
> "${RESULTS_DIR}/eval_log.txt"

# ---------- (VLLM) convert the checkpoint to a vLLM model dir, once ----------
# Cached next to the ckpt as vllm_<ckptname>/. The container's convert.sh
# mounts the ckpt dir as both /ckpt and /out.
if [ "$VLLM" = "1" ]; then
    CKPT_DIR="$(cd "$(dirname "$LOCAL_CKPT_PATH")" && pwd)"
    CKPT_BASE="$(basename "$LOCAL_CKPT_PATH")"
    # GPU passthrough for docker. Default uses --gpus (NVIDIA Container Toolkit).
    # If your daemon only has the legacy runtime, override e.g.:
    #   DOCKER_GPU_ARGS="--runtime=nvidia" ...
    # NVIDIA_VISIBLE_DEVICES is also forwarded so the runtime path works too.
    DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:---gpus device=${DEVICE_ID}}"
    VLLM_NAME="vllm_${CKPT_BASE%.ckpt}"
    VLLM_OUT="${CKPT_DIR}/${VLLM_NAME}"
    if [ -d "$VLLM_OUT" ] && [ -n "$(ls -A "$VLLM_OUT" 2>/dev/null)" ] && [ "${FORCE_CONVERT:-0}" != "1" ]; then
        echo "==> Reusing converted vLLM model (FORCE_CONVERT=1 to rebuild): $VLLM_OUT"
    else
        echo "==> Converting checkpoint -> vLLM model dir via ${VLLM_IMAGE}"
        rm -rf "$VLLM_OUT"; mkdir -p "$VLLM_OUT"
        docker run --rm $DOCKER_GPU_ARGS -e NVIDIA_VISIBLE_DEVICES="${DEVICE_ID}" \
            -v "${CKPT_DIR}":/ckpt \
            -v "${CKPT_DIR}":/out \
            "$VLLM_IMAGE" \
            bash /workspace/convert.sh "/ckpt/${CKPT_BASE}" "/out/${VLLM_NAME}"
        echo "==> Convert complete: $VLLM_OUT"
    fi
    MANIFEST_DIR="${RESULTS_DIR}/manifests"
    mkdir -p "$MANIFEST_DIR"
fi

echo ""
echo "==> Running ASR leaderboard evaluation $([ "$VLLM" = "1" ] && echo '(vLLM)')"
echo "    Checkpoint: $LOCAL_CKPT_PATH"
echo "    Device:     cuda:${DEVICE_ID}   Batch: ${BATCH_SIZE}"
echo "    Results:    ${RESULTS_DIR}"
echo ""
cd "$RESULTS_DIR"

for ds_entry in "${DATASETS[@]}"; do
    IFS=':' read -r DATASET SPLIT <<< "$ds_entry"
    echo "----------------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')] Evaluating: ${DATASET}/${SPLIT}"
    echo "----------------------------------------------------------------------"
    if [ "$VLLM" = "1" ]; then
        # 1) Materialize this split to 16k wavs (under ./audio_cache) + a NeMo
        #    manifest, reusing run_eval_sslm.py's exact loading/normalization.
        MAN_PATH="${MANIFEST_DIR}/${DATASET}_${SPLIT}.json"
        python "$RUN_EVAL_PY" \
            --ckpt_path "$LOCAL_CKPT_PATH" \
            --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --dump_manifest "$MAN_PATH" \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee -a "${RESULTS_DIR}/eval_log.txt"
        # 2) Decode in the vLLM container. The manifest holds absolute host paths
        #    under ./audio_cache, so mount that tree at the same path so they
        #    resolve inside the container; mount the manifest dir as /data.
        AUDIO_CACHE="${RESULTS_DIR}/audio_cache"
        echo "Dataset: ${DATASET}/${SPLIT}" | tee -a "${RESULTS_DIR}/eval_log.txt"
        docker run --rm $DOCKER_GPU_ARGS -e NVIDIA_VISIBLE_DEVICES="${DEVICE_ID}" \
            -v "${VLLM_OUT}":/model \
            -v "${MANIFEST_DIR}":/data \
            -v "${AUDIO_CACHE}":"${AUDIO_CACHE}" \
            -e B_MODEL=/model \
            -e B_MAN="/data/$(basename "$MAN_PATH")" \
            "$VLLM_IMAGE" \
            python /workspace/b_streaming_infer.py \
            2>&1 | tee -a "${RESULTS_DIR}/eval_log.txt"
    else
        python "$RUN_EVAL_PY" \
            --ckpt_path "$LOCAL_CKPT_PATH" \
            --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --device "$DEVICE_ID" \
            --batch_size "$BATCH_SIZE" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --no_repeat_ngram_size "$NO_REPEAT_NGRAM_SIZE" \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee -a "${RESULTS_DIR}/eval_log.txt"
    fi
    echo ""
done

# ---------- Step 3: summary table ----------
echo "======================================================================"
echo "Evaluation complete. Results in: ${RESULTS_DIR}"
echo "======================================================================"
python3 -c "
import re
datasets = [
    ('ami', 'test'), ('earnings22', 'test'), ('gigaspeech', 'test'),
    ('librispeech', 'test.clean'), ('librispeech', 'test.other'),
    ('spgispeech', 'test'), ('tedlium', 'test'), ('voxpopuli', 'test'),
]
log = open('${RESULTS_DIR}/eval_log.txt').read()
entries = re.findall(r'Dataset:\s*(\S+/\S+)[\s\S]*?WER:\s*([\d.]+)\s*%', log)
wer_map = {ds: float(w) for ds, w in entries}
print(f'  {\"Dataset\":<25} {\"WER (%)\":>8}')
print(f'  {\"-\" * 25} {\"-\" * 8}')
total = 0.0; n = 0
for ds, split in datasets:
    key = f'{ds}/{split}'
    if key in wer_map:
        print(f'  {key:<25} {wer_map[key]:>8.2f}'); total += wer_map[key]; n += 1
    else:
        print(f'  {key:<25} {\"--\":>8}')
if n > 0:
    print(f'  {\"-\" * 25} {\"-\" * 8}')
    print(f'  {\"Average\":<25} {total / n:>8.2f}')
"
