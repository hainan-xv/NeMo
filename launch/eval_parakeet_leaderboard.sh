#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:parakeet-lb-eval
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
# REFERENCE/sanity Open-ASR-Leaderboard eval for a STANDARD NeMo ASR model
# (default: Parakeet TDT 0.6B v2), on OCI: one node, work BALANCED across 8 GPUs.
#
# WHY: this reuses the EXACT same staged cache, the same pooled-shard sharding,
# the same generations JSONL schema, and the same leaderboard-faithful scorer
# (vendored normalizer + kaldialign merge_compounds) as launch/eval_leaderboard.sh
# -- only the decode call differs (ASRModel.transcribe() vs the SpeechLM
# generate()). So running a KNOWN public model here should reproduce its published
# leaderboard WER, confirming our cache + scorer match the board. If Parakeet lines
# up but the SCRIPT model doesn't, the gap is the model, not our pipeline.
#
# Reads the PRE-STAGED wav/manifest cache on lustre (no HF download on compute):
#   <CACHE_DIR>/<dataset>/<split>/_cache_manifest.jsonl + 16 kHz mono wavs
# (stage it once with launch/stage_leaderboard_cache.sh).
#
# Usage (from the clean repo root on OCI):
#   cd /lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_script_clean
#   sbatch launch/eval_parakeet_leaderboard.sh                       # default .nemo
#   sbatch launch/eval_parakeet_leaderboard.sh /path/to/model.nemo   # $1 = .nemo path
#   MAX_EVAL_SAMPLES=10 sbatch launch/eval_parakeet_leaderboard.sh    # smoke test (10 utts/ds)
#
# Key env:
#   NEMO_MODEL        .nemo model to eval (default parakeet.nemo; $1 overrides)
#   CACHE_DIR         pre-staged leaderboard cache root on lustre
#   BATCH_SIZE        transcribe() batch size (default 32; OOM auto-halves)
#   DATASETS          space/comma 'name:split' list (default = current public suite)
#   SHUFFLE_SEED      seed for the pooled global shuffle (default 1234, matches sslm)
#   MAX_EVAL_SAMPLES  cap samples per dataset (fast smoke test)
#   EVAL_TAG          optional label spliced into RESULTS_DIR + wandb run name
#   REPORT_WANDB      auto (report iff ~/.wandb_token exists) | 1 (force) | 0 (off)
# ============================================================================
# NOTE: intentionally NOT `set -euo pipefail` -- same reasons as eval_leaderboard.sh
# (the `read -r -d '' <<EOF` heredoc returns non-zero at EOF). Hard failures use
# explicit `exit 1` guards below.

read_optional_token() { [[ -r "$1" ]] && tr -d '\r\n' < "$1" || true; }
WANDB_TOKEN="$(read_optional_token "$HOME/.wandb_token")"

mkdir -p slurm_out

# --- Model identity ---
PROJECT="${PROJECT:-SpeechlmScriptClean}"
# $1 (or NEMO_MODEL) selects the .nemo; default is the downloaded Parakeet TDT v2.
NEMO_MODEL="${1:-${NEMO_MODEL:-/lustre/fsw/portfolios/nemotron/users/hainanx/parakeet.nemo}}"
# EXP_NAME is just a results-folder label here (no checkpoint dir to resolve).
EXP_NAME="${EXP_NAME:-$(basename "${NEMO_MODEL%.nemo}")}"

BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
SHUFFLE_SEED="${SHUFFLE_SEED:-1234}"
# Current PUBLIC leaderboard suite (must match the staged cache + eval_leaderboard.sh).
DATASETS="${DATASETS:-librispeech:test.clean librispeech:test.other ami_cleaned:test earnings22:test gigaspeech_cleaned:test spgispeech:test voxpopuli_cleaned_aa:test}"
DATASETS_CSV="$(echo "$DATASETS" | tr -s ' ' ',')"

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
# The clean SCRIPT repo, git-synced via sync_to_oci.sh -> mounted as /code.
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_script_clean}"
# Pre-staged leaderboard cache on lustre (populated by stage_leaderboard_cache.sh).
CACHE_DIR="${CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache}"
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh

if [[ ! -f "$NEMO_MODEL" ]]; then
    echo "ERROR: .nemo model not found: $NEMO_MODEL (download it, or pass the path as \$1 / NEMO_MODEL=)." >&2
    exit 1
fi
if [[ ! -d "$CACHE_DIR" ]]; then
    echo "ERROR: CACHE_DIR not found: $CACHE_DIR (stage the leaderboard cache there first)." >&2
    exit 1
fi

# Timestamp + job id (+ optional EVAL_TAG) so concurrent evals never share a dir.
RUN_TS="$(date +%Y%m%d_%H%M%S)"
JOB_TAG="${SLURM_JOB_ID:-local$$}"
EVAL_TAG_SUFFIX=""; [[ -n "${EVAL_TAG:-}" ]] && EVAL_TAG_SUFFIX="_${EVAL_TAG}"
RESULTS_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/parakeet_leaderboard_eval${EVAL_TAG_SUFFIX}_${RUN_TS}_${JOB_TAG}"
mkdir -p "$RESULTS_DIR"
SHARD_DIR="${RESULTS_DIR}/shards"; mkdir -p "$SHARD_DIR"

HFCACHE="${OUTPUT_PREFIX}/hf_cache"; mkdir -p "$HFCACHE"
# TMPDIR must stay SHORT (AF_UNIX socket path cap ~108 bytes); the deep RESULTS_DIR
# would overflow it, so default to a short node-local path.
OCI_TMP_DIR="${OCI_TMP_DIR:-/tmp/parakeet_eval_${SLURM_JOB_ID:-$$}}"

OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out
# The .nemo may live on a DIFFERENT lustre filesystem than fsw (e.g. /lustre/fs12).
# Bind its directory directly so it's visible regardless of which lustre it's on.
MODEL_DIR="$(dirname "$NEMO_MODEL")"
# Bind each needed lustre leaf DIRECTLY (source==target). /lustre/fsw (and fs12,
# etc.) is an autofs tree whose lazily-mounted sub-paths do NOT propagate into the
# container's private mount namespace under a broad bind, so a broad
# /lustre/fsw:/lustre/fsw alone can leave RESULTS_DIR/CACHE_DIR (and a model on
# another lustre) invisible in the container ("No such file"). Direct binds force
# autofs to resolve at mount time (same reason /code and /hfcache work).
# OUTPUT_PREFIX covers results/shards + container_cmd.sh; CACHE_DIR is the staged
# cache; MODEL_DIR is wherever the .nemo lives. The broad bind is FIRST as a
# catch-all -- it must precede the direct binds or mounting an ancestor last would
# shadow the children.
MOUNTS="--container-mounts=/lustre/fsw:/lustre/fsw,${CODE_DIR}:/code,${OUTPUT_PREFIX}:${OUTPUT_PREFIX},${CACHE_DIR}:${CACHE_DIR},${H_DIR}:${H_DIR},${HFCACHE}:/hfcache/"
# Add the model dir only if it isn't already covered by a direct bind above
# (enroot errors on a duplicate source==target mount).
case ":${OUTPUT_PREFIX}:${CACHE_DIR}:${H_DIR}:${CODE_DIR}:" in
    *":${MODEL_DIR}:"*) : ;;                              # already directly bound
    *) MOUNTS="${MOUNTS},${MODEL_DIR}:${MODEL_DIR}" ;;
esac

NGPU=8

# ---- Weights & Biases reporting of the final per-dataset WER (default: auto) ----
REPORT_WANDB="${REPORT_WANDB:-auto}"
WANDB_EVAL_PROJECT="${WANDB_EVAL_PROJECT:-${PROJECT}_leaderboard_eval}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-parakeet_${EXP_NAME}${EVAL_TAG_SUFFIX}}_${RUN_TS}"
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
    WANDB_CLAUSE="&& { ${WANDB_MODE_EXPORT}export WANDB_API_KEY='${WANDB_TOKEN}'; python /code/scripts/eval_wandb_report.py --project '${WANDB_EVAL_PROJECT}' --run_name '${WANDB_RUN_NAME}' --results_dir '${RESULTS_DIR}' --group '${EXP_NAME}' --job_type parakeet 2>&1 | tee '${RESULTS_DIR}/wandb_report.log' || true; }"
fi

# ---- Record this run's configuration under the (timestamped) eval folder ----
{
    echo "# Parakeet (standard NeMo ASR) leaderboard eval run config"
    echo "timestamp:            ${RUN_TS}"
    echo "slurm_job_id:         ${JOB_TAG}"
    echo "eval_tag:             ${EVAL_TAG:-}"
    echo "exp_name:             ${EXP_NAME}"
    echo "project:              ${PROJECT}"
    echo "backend:              parakeet"
    echo "scoring:              open_asr_leaderboard (vendored normalizer + kaldialign merge_compounds)"
    echo "nemo_model:           ${NEMO_MODEL}"
    echo "batch_size:           ${BATCH_SIZE}"
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

echo "==> Parakeet leaderboard eval"
echo "    nemo_model:   ${NEMO_MODEL}"
echo "    exp_name:     ${EXP_NAME}"
echo "    datasets:     ${DATASETS_CSV}"
echo "    cache_dir:    ${CACHE_DIR}"
echo "    batch_size:   ${BATCH_SIZE}  seed:${SHUFFLE_SEED}  max_eval_samples:${MAX_EVAL_SAMPLES}"

read -r -d '' cmd <<EOF
echo "*******Parakeet leaderboard eval (pooled shards over ${NGPU} GPUs)********" \
&& echo "*** MODEL=${NEMO_MODEL} ***" \
&& nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
&& cd /code \
&& git rev-parse HEAD \
&& export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& export OMP_NUM_THREADS=1 \
&& export HF_HOME="/hfcache/" \
&& export HF_HUB_OFFLINE=1 \
&& export TOKENIZERS_PARALLELISM=false \
&& export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
&& export HYDRA_FULL_ERROR=1 \
&& export TMPDIR=${OCI_TMP_DIR} && mkdir -p ${OCI_TMP_DIR} \
&& python -c "import nemo, nemo.collections.asr; print('NeMo:', nemo.__file__)" \
&& { python -c "import kaldialign; kaldialign.batch_error_rate" 2>/dev/null || { echo '==> installing/upgrading kaldialign (leaderboard-faithful WER: batch_error_rate + merge_compounds)'; pip install -U --no-input --quiet kaldialign; }; } \
&& echo "Pooled datasets: ${DATASETS_CSV}" \
&& echo "Fanning ${NGPU} balanced shards across ${NGPU} GPUs (seed=${SHUFFLE_SEED})..." \
&& rm -f ${SHARD_DIR}/shard*_of*.generations.jsonl \
&& pids=() \
&& for gpu in \$(seq 0 \$(( ${NGPU} - 1 ))); do \
      log="${RESULTS_DIR}/shard_\${gpu}.log"; \
      echo "  [gpu \$gpu] shard \$gpu/${NGPU} -> \${log}"; \
      CUDA_VISIBLE_DEVICES=\$gpu python /code/scripts/parakeet_leaderboard_eval.py \
        --nemo_model "${NEMO_MODEL}" \
        --datasets "${DATASETS_CSV}" \
        --num_shards ${NGPU} \
        --shard_index \$gpu \
        --shuffle_seed ${SHUFFLE_SEED} \
        --device 0 \
        --cache_dir "${CACHE_DIR}" \
        --batch_size ${BATCH_SIZE} \
        --max_eval_samples ${MAX_EVAL_SAMPLES} \
        --output_dir "${SHARD_DIR}" \
        > "\${log}" 2>&1 & \
      pids+=(\$!); \
   done \
&& fail=0 && for p in "\${pids[@]}"; do wait "\$p" || fail=1; done \
&& echo "" \
&& echo "==================== Leaderboard WER ====================" \
&& python /code/scripts/parakeet_leaderboard_eval.py --aggregate --output_dir "${SHARD_DIR}" 2>&1 | tee "${RESULTS_DIR}/aggregate.log" \
${WANDB_CLAUSE} \
&& echo "" \
&& echo "Per-shard logs + generations under: ${RESULTS_DIR}" \
&& exit \$fail
EOF

# Run via a script file (not bash -c "...") so nested quotes cannot break the outer
# shell. Write it INSIDE the checkout (bind-mounted directly at /code) so the
# container can always open it (a path under the broad /lustre/fsw tree may be
# invisible in the container -- autofs; see MOUNTS note above).
mkdir -p "${CODE_DIR}/slurm_out"
CMD_BASENAME="parakeet_eval_cmd_${SLURM_JOB_ID:-local$$}.sh"
printf '%s\n' "$cmd" > "${CODE_DIR}/slurm_out/${CMD_BASENAME}"
chmod +x "${CODE_DIR}/slurm_out/${CMD_BASENAME}"
echo "==> Container command: /code/slurm_out/${CMD_BASENAME}"

srun -o "$OUTFILE" -e "$ERRFILE" --container-image="$CONTAINER" $MOUNTS bash "/code/slurm_out/${CMD_BASENAME}"
