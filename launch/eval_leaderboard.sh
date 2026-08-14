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
# Parallel Open-ASR-Leaderboard eval for the clean SCRIPT repo, ON OCI: one node,
# work BALANCED across 8 GPUs. Instead of one dataset per GPU (imbalanced, since
# datasets differ hugely in size), we POOL every utterance across all datasets,
# shuffle with a fixed seed, and give each GPU an even 1/8 slice. Total wall time
# ~= sum(all)/8 instead of the single largest dataset.
#
# BACKEND: the self-contained in-process driver scripts/speechlm_leaderboard_eval.py
# loads ONE Lightning .ckpt directly (no HF conversion) and decodes each shard via
# the model's own generate() -- the SAME code path as training-time val_wer, so the
# numbers are directly comparable. Each GPU process shards internally
# (--num_shards 8 --shard_index <gpu>, seeded global shuffle + duration sort) and
# writes a dataset-tagged generations JSONL; a final --aggregate reduces them into
# per-dataset + macro WER.
#
# Reads a PRE-STAGED wav/manifest cache on lustre (no HF download on compute nodes):
#   <CACHE_DIR>/<dataset>/<split>/_cache_manifest.jsonl  + 16 kHz mono wavs
# The checkpoint is already on lustre (the training run wrote it) -> no rsync.
#
# Usage (from the clean repo root on OCI):
#   cd /lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_script_clean
#   sbatch launch/eval_leaderboard.sh                      # EXP_NAME=granary2_script_baseline
#   sbatch launch/eval_leaderboard.sh granary2_script_baseline
#   CHUNK_SIZE=7 sbatch launch/eval_leaderboard.sh
#   MAX_EVAL_SAMPLES=10 sbatch launch/eval_leaderboard.sh  # smoke test (first 10 utts/ds)
#
# Key env:
#   CACHE_DIR         pre-staged leaderboard cache root on lustre
#                     (default /lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache)
#   PROJECT           results project dir (default SpeechlmScriptClean; MUST match training)
#   MODEL_CLASS       model class (default ScriptSTTModel)
#   SYSTEM_PROMPT     decode prompt (MUST match the model's training prompt!)
#   CHUNK_SIZE        decode chunk size override (encoder frames; default = model default)
#   BATCH_SIZE        per-shard batch size (default 32; sslm OOM auto-halves)
#   MAX_NEW_TOKENS    per-chunk decode cap (default 64)
#   RUN_AVERAGING     1 (default) -> average the top-k non-last ckpts into
#                     <CKPT_DIR>/<EXP>-averaged.ckpt (cached + reused); 0 -> single ckpt
#   FORCE_AVERAGE=1   recompute the averaged ckpt even if it's cached
#   USE_LAST=1        eval the rolling -last.ckpt (disables averaging)
#   STEP=<n>          eval step=<n>.ckpt explicitly (disables averaging)
#   CKPT=<path>       eval this exact .ckpt (overrides EXP resolution; disables averaging)
#   DATASETS          space-separated 'name:split' list (default: full 8-set suite)
#   SHUFFLE_SEED      seed for the pooled global shuffle (default 1234)
#   MAX_EVAL_SAMPLES  cap samples per dataset (fast smoke test)
#   SELF_CORRECT=1    SCRIPT redecode models: emit the self-corrected LOCKED stream
#   OUTPUT_PREFIX     results root (default nemotron users/hainanx)
#   EVAL_TAG          optional label spliced into RESULTS_DIR + wandb run name
#   REPORT_WANDB      auto (report iff ~/.wandb_token exists) | 1 (force) | 0 (off)
# ============================================================================
# NOTE: intentionally NOT `set -euo pipefail`. This body uses `read -r -d '' <<EOF`
# (returns non-zero at EOF) and `ls | grep | head` checkpoint-resolution pipelines
# (grep exits 1 when a pattern legitimately has no match), both of which would
# abort early under `set -e`. Hard failures are handled with explicit `exit 1`
# guards below (missing cache dir / unresolved checkpoint).

read_optional_token() { [[ -r "$1" ]] && tr -d '\r\n' < "$1" || true; }
HF_TOKEN="$(read_optional_token "$HOME/.hf_token")"
WANDB_TOKEN="$(read_optional_token "$HOME/.wandb_token")"

mkdir -p slurm_out

PROJECT="${PROJECT:-SpeechlmScriptClean}"
EXP_NAME="${1:-${EXP_NAME:-granary2_script_baseline}}"
MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-Transcribe the audio into text.}"
CHUNK_SIZE="${CHUNK_SIZE:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
SHUFFLE_SEED="${SHUFFLE_SEED:-1234}"
# Current PUBLIC leaderboard suite (github.com/huggingface/open_asr_leaderboard,
# 2026-08): CLEANED AMI/GigaSpeech/VoxPopuli variants, TED-LIUM dropped. Stage the
# cache from the hub dataset `hf-audio/open-asr-leaderboard` with configs
# ami_cleaned / gigaspeech_cleaned / voxpopuli_cleaned_aa / earnings22 / spgispeech
# and librispeech (splits test.clean, test.other), under <CACHE_DIR>/<name>/<split>/.
DATASETS="${DATASETS:-librispeech:test.clean librispeech:test.other ami_cleaned:test earnings22:test gigaspeech_cleaned:test spgispeech:test voxpopuli_cleaned_aa:test}"
# Comma-joined form for the driver's --datasets (which splits on commas).
DATASETS_CSV="$(echo "$DATASETS" | tr -s ' ' ',')"

# Self-correction (SCRIPT redecode models only): decode the self-corrected LOCKED
# stream instead of the default non-corrective j=0 stream. Ignored by other models.
SELF_CORRECT="${SELF_CORRECT:-0}"
SELF_CORRECT_FLAG=""
if [[ "$SELF_CORRECT" == 1 || "$SELF_CORRECT" == true ]]; then
    SELF_CORRECT_FLAG="--self_correct"
fi

# Checkpoint averaging (DEFAULT ON): average the top-k (non '-last') checkpoints --
# the ones exp_manager keeps by val_wer -- into <CKPT_DIR>/<EXP>-averaged.ckpt,
# cached in the model folder and REUSED on later runs. FORCE_AVERAGE=1 recomputes
# it. Setting CKPT=/STEP=/USE_LAST=1 selects a specific ckpt and disables averaging.
RUN_AVERAGING="${RUN_AVERAGING:-1}"
FORCE_AVERAGE="${FORCE_AVERAGE:-0}"
USE_LAST="${USE_LAST:-0}"
STEP="${STEP:-}"
CKPT="${CKPT:-}"

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
# The clean SCRIPT repo, git-synced via sync_to_oci.sh -> mounted as /code.
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_script_clean}"
# Pre-staged leaderboard cache on lustre (rsync of ~/leaderboard_run/cache).
CACHE_DIR="${CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache}"
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh

if [[ ! -d "$CACHE_DIR" ]]; then
    echo "ERROR: CACHE_DIR not found: $CACHE_DIR (stage the leaderboard cache there, or set CACHE_DIR=)." >&2
    exit 1
fi

# exp_manager writes checkpoints to results/<PROJECT>/<EXP>/<EXP>/checkpoints/
# (the nested <EXP> comes from exp_manager.name; the outer one is our RESULTS_DIR).
CKPT_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/${EXP_NAME}/checkpoints"

# A specific-checkpoint request wins over (and disables) averaging.
if [[ -n "$CKPT" || -n "$STEP" || "$USE_LAST" == "1" ]]; then
    RUN_AVERAGING=0
fi

# ---- Resolve the checkpoint on lustre (no rsync; it's already here) ----
AVG_INPUTS_FILE=""
DO_AVG=0
if [[ "$RUN_AVERAGING" == "1" ]]; then
    AVG_CKPT="${CKPT_DIR}/${EXP_NAME}-averaged.ckpt"
    mapfile -t _AVG_IN < <(ls -t "${CKPT_DIR}"/*.ckpt 2>/dev/null | grep -v -- '-last\.ckpt$' | grep -v -- '-averaged\.ckpt$')
    if [[ ${#_AVG_IN[@]} -eq 0 ]]; then
        echo "ERROR: RUN_AVERAGING=1 but no non-last checkpoints under ${CKPT_DIR}." >&2
        echo "       (use USE_LAST=1 to eval the rolling -last.ckpt, or set CKPT=/STEP=.)" >&2
        exit 1
    fi
    CKPT="$AVG_CKPT"
    DO_AVG=1
    echo "==> Will average ${#_AVG_IN[@]} checkpoint(s) -> ${AVG_CKPT} (cached; reused unless FORCE_AVERAGE=1)."
else
    if [[ -z "$CKPT" ]]; then
        if [[ -n "$STEP" ]]; then
            CKPT="${CKPT_DIR}/step=${STEP}.ckpt"
        elif [[ "$USE_LAST" == "1" ]]; then
            CKPT="$(ls -t "${CKPT_DIR}"/*-last.ckpt 2>/dev/null | head -1)"
        else
            CKPT="$(ls -t "${CKPT_DIR}"/*.ckpt 2>/dev/null | grep -v -- '-last\.ckpt$' | grep -v -- '-averaged\.ckpt$' | head -1)"
            [[ -z "$CKPT" ]] && CKPT="$(ls -t "${CKPT_DIR}"/*-last.ckpt 2>/dev/null | head -1)"
        fi
    fi
    if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
        echo "ERROR: could not resolve a checkpoint under ${CKPT_DIR} (set CKPT=, STEP=, USE_LAST=1, or RUN_AVERAGING=1)." >&2
        exit 1
    fi
fi

# Timestamp + job id (+ optional EVAL_TAG) so concurrent evals never share a dir.
RUN_TS="$(date +%Y%m%d_%H%M%S)"
JOB_TAG="${SLURM_JOB_ID:-local$$}"
CHUNK_TAG=""; [[ -n "$CHUNK_SIZE" ]] && CHUNK_TAG="_chunk${CHUNK_SIZE}"
EVAL_TAG_SUFFIX=""; [[ -n "${EVAL_TAG:-}" ]] && EVAL_TAG_SUFFIX="_${EVAL_TAG}"
RESULTS_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/leaderboard_eval${CHUNK_TAG}${EVAL_TAG_SUFFIX}_${RUN_TS}_${JOB_TAG}"
mkdir -p "$RESULTS_DIR"
SHARD_DIR="${RESULTS_DIR}/shards"; mkdir -p "$SHARD_DIR"

# The list of checkpoints to average, one per line (read inside the container).
if [[ "$DO_AVG" == "1" ]]; then
    AVG_INPUTS_FILE="${SHARD_DIR}/avg_inputs.txt"
    printf '%s\n' "${_AVG_IN[@]}" > "$AVG_INPUTS_FILE"
fi

HFCACHE="${OUTPUT_PREFIX}/hf_cache"; mkdir -p "$HFCACHE"
# TMPDIR must stay SHORT: Python multiprocessing creates AF_UNIX sockets at
# ${TMPDIR}/pymp-*/listener-*, and Linux caps the socket path at ~108 bytes. The
# deep, timestamped RESULTS_DIR blows past that, so default to a short node-local path.
OCI_TMP_DIR="${OCI_TMP_DIR:-/tmp/sslm_eval_${SLURM_JOB_ID:-$$}}"

OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out
# Broad lustre mounts cover the ckpt (nemotron) and the staged cache wherever it
# lives; /code is our synced checkout.
MOUNTS="--container-mounts=${CODE_DIR}:/code,${H_DIR}:${H_DIR},${HFCACHE}:/hfcache/,/lustre/fsw:/lustre/fsw"

NGPU=8

# ---- Weights & Biases reporting of the final per-dataset WER (default: auto) ----
REPORT_WANDB="${REPORT_WANDB:-auto}"
WANDB_EVAL_PROJECT="${WANDB_EVAL_PROJECT:-${PROJECT}_leaderboard_eval}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXP_NAME}${EVAL_TAG_SUFFIX}${CHUNK_TAG}}_${RUN_TS}"
DO_WANDB=0
case "$REPORT_WANDB" in
    auto)          [[ -n "${WANDB_TOKEN:-}" ]] && DO_WANDB=1 ;;
    1|true|yes|on)  DO_WANDB=1 ;;
    0|false|no|off) DO_WANDB=0 ;;
    *) echo "WARNING: unknown REPORT_WANDB='${REPORT_WANDB}' (expected auto|1|0); treating as off." >&2 ;;
esac
WANDB_MODE_EXPORT=""
[[ -n "${WANDB_MODE:-}" ]] && WANDB_MODE_EXPORT="export WANDB_MODE='${WANDB_MODE}'; "
WANDB_CLAUSE=""
if [[ "$DO_WANDB" == "1" ]]; then
    echo "==> wandb reporting ON -> project='${WANDB_EVAL_PROJECT}' run='${WANDB_RUN_NAME}'"
    WANDB_CLAUSE="&& { ${WANDB_MODE_EXPORT}export WANDB_API_KEY='${WANDB_TOKEN}'; python /code/scripts/eval_wandb_report.py --project '${WANDB_EVAL_PROJECT}' --run_name '${WANDB_RUN_NAME}' --results_dir '${RESULTS_DIR}' --group '${EXP_NAME}' --job_type sslm 2>&1 | tee '${RESULTS_DIR}/wandb_report.log' || true; }"
fi

# ---- Record this run's configuration under the (timestamped) eval folder ----
{
    echo "# SpeechLM leaderboard eval run config"
    echo "timestamp:            ${RUN_TS}"
    echo "slurm_job_id:         ${JOB_TAG}"
    echo "eval_tag:             ${EVAL_TAG:-}"
    echo "exp_name:             ${EXP_NAME}"
    echo "project:              ${PROJECT}"
    echo "backend:              sslm"
    echo "scoring:              open_asr_leaderboard (vendored normalizer + kaldialign merge_compounds)"
    echo "model_class:          ${MODEL_CLASS}"
    echo "checkpoint:           ${CKPT}"
    echo "run_averaging:        ${RUN_AVERAGING}"
    echo "force_average:        ${FORCE_AVERAGE}"
    if [[ "$DO_AVG" == "1" ]]; then
        echo "averaged_num_inputs:  ${#_AVG_IN[@]}"
        echo "averaged_inputs:"
        for f in "${_AVG_IN[@]}"; do echo "  - ${f}"; done
    fi
    echo "system_prompt:        \"${SYSTEM_PROMPT}\""
    echo "chunk_size:           ${CHUNK_SIZE:-<model default>}"
    echo "self_correct:         $( [[ "$SELF_CORRECT" == 1 || "$SELF_CORRECT" == true ]] && echo 'true (locked/corrected stream)' || echo 'false (non-corrective j=0 stream)')"
    echo "batch_size:           ${BATCH_SIZE}"
    echo "max_new_tokens:       ${MAX_NEW_TOKENS}"
    echo "num_gpus:             ${NGPU}"
    echo "shuffle_seed:         ${SHUFFLE_SEED}"
    echo "max_eval_samples:     ${MAX_EVAL_SAMPLES}"
    echo "datasets:             ${DATASETS_CSV}"
    echo "cache_dir:            ${CACHE_DIR}"
    echo "results_dir:          ${RESULTS_DIR}"
    echo "report_wandb:         ${DO_WANDB}"
    echo "wandb_eval_project:   ${WANDB_EVAL_PROJECT}"
    echo "wandb_run_name:       ${WANDB_RUN_NAME}"
} > "${RESULTS_DIR}/run_config.yaml"
echo "==> Wrote run config: ${RESULTS_DIR}/run_config.yaml"

# System prompt may contain spaces / apostrophes (e.g. "chunk's"). Pass it via a
# file the container reads (SP_TEXT=$(cat ...)) so nested quotes can't break the
# outer shell quoting.
printf '%s' "$SYSTEM_PROMPT" > "${SHARD_DIR}/system_prompt.txt"

# Averaging step (in-container; torch). Runs once before the GPU fan-out. Skipped
# when the cached averaged ckpt already exists (unless FORCE_AVERAGE=1).
AVG_CLAUSE=""
if [[ "$DO_AVG" == "1" ]]; then
    AVG_CLAUSE="&& if [[ '${FORCE_AVERAGE}' == '1' || ! -f '${CKPT}' ]]; then echo '==> Averaging ${#_AVG_IN[@]} checkpoints -> ${CKPT}'; python /code/scripts/average_sslm_ckpts.py --output '${CKPT}' \$(cat '${AVG_INPUTS_FILE}'); else echo '==> Reusing cached averaged checkpoint: ${CKPT}'; fi "
fi

read -r -d '' cmd <<EOF
echo "*******SpeechLM leaderboard eval (sslm in-process, pooled shards over ${NGPU} GPUs)********" \
&& echo "*** EXP=${EXP_NAME} | MODEL_CLASS=${MODEL_CLASS} ***" \
&& echo "*** CKPT=${CKPT} ***" \
&& echo "*** CACHE_DIR=${CACHE_DIR} | CHUNK_SIZE=${CHUNK_SIZE:-<default>} | BATCH_SIZE=${BATCH_SIZE} | SEED=${SHUFFLE_SEED} ***" \
&& echo "*** system_prompt: [\$(cat '${SHARD_DIR}/system_prompt.txt')] ***" \
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
&& { python -c "import kaldialign; kaldialign.batch_error_rate" 2>/dev/null || { echo '==> installing/upgrading kaldialign (leaderboard-faithful WER: batch_error_rate + merge_compounds)'; pip install -U --no-input --quiet kaldialign; }; } \
${AVG_CLAUSE} \
&& [ -f "${CKPT}" ] || { echo "ERROR: checkpoint missing at ${CKPT}"; exit 1; } \
&& echo "Pooled datasets: ${DATASETS_CSV}" \
&& echo "Fanning ${NGPU} balanced shards across ${NGPU} GPUs (seed=${SHUFFLE_SEED})..." \
&& rm -f ${SHARD_DIR}/shard*_of*.generations.jsonl \
&& SP_TEXT=\$(cat '${SHARD_DIR}/system_prompt.txt') \
&& pids=() \
&& for gpu in \$(seq 0 \$(( ${NGPU} - 1 ))); do \
      log="${RESULTS_DIR}/shard_\${gpu}.log"; \
      echo "  [gpu \$gpu] shard \$gpu/${NGPU} -> \${log}"; \
      CUDA_VISIBLE_DEVICES=\$gpu python /code/scripts/speechlm_leaderboard_eval.py \
        --ckpt_path "${CKPT}" \
        --model_class "${MODEL_CLASS}" \
        --datasets "${DATASETS_CSV}" \
        --num_shards ${NGPU} \
        --shard_index \$gpu \
        --shuffle_seed ${SHUFFLE_SEED} \
        --device 0 \
        --cache_dir "${CACHE_DIR}" \
        --batch_size ${BATCH_SIZE} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --max_eval_samples ${MAX_EVAL_SAMPLES} \
        --system_prompt "\$SP_TEXT" \
        --output_dir "${SHARD_DIR}" \
        ${CHUNK_SIZE:+--chunk_size ${CHUNK_SIZE}} \
        ${SELF_CORRECT_FLAG} \
        > "\${log}" 2>&1 & \
      pids+=(\$!); \
   done \
&& fail=0 && for p in "\${pids[@]}"; do wait "\$p" || fail=1; done \
&& echo "" \
&& echo "==================== Leaderboard WER ====================" \
&& python /code/scripts/speechlm_leaderboard_eval.py --aggregate --output_dir "${SHARD_DIR}" 2>&1 | tee "${RESULTS_DIR}/aggregate.log" \
${WANDB_CLAUSE} \
&& echo "" \
&& echo "Per-shard logs + generations under: ${RESULTS_DIR}" \
&& exit \$fail
EOF

# Run via a script file (not bash -c "...") so nested quotes in the prompt cannot
# break the outer shell quoting.
CMD_FILE="${RESULTS_DIR}/container_cmd.sh"
printf '%s\n' "$cmd" > "$CMD_FILE"
chmod +x "$CMD_FILE"
echo "==> Container command: ${CMD_FILE}"

srun -o "$OUTFILE" -e "$ERRFILE" --container-image="$CONTAINER" $MOUNTS bash "$CMD_FILE"
