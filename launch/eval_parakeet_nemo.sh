#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:parakeet-nemo-eval
#SBATCH -p interactive,batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH --time-min 01:00:00
# 1 GPU caps RAM (scheduler rejects --mem=0 for a single-GPU job); this eval is
# GPU-bound and needs little host RAM.
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# NeMo-native leaderboard eval for a standard ASR model (default: Parakeet TDT
# 0.6B v2), SINGLE GPU, NO sharding/pooling -- the maximally-canonical reference.
#
# Decode: NeMo's OWN maintained examples/asr/speech_to_text_eval.py (which wraps
# transcribe_speech.py), run once PER dataset against that dataset's pre-staged
# _cache_manifest.jsonl. This removes every bit of our custom decode/harness from
# the loop, so it isolates whether an off-vs-board number comes from our code.
#
# Score: NeMo's script prints its own plain word_error_rate (no leaderboard
# normalization) per dataset; then scripts/rescore_nemo_eval.py re-scores the SAME
# predictions with the leaderboard-faithful WER (vendored normalizer + kaldialign
# merge_compounds), so you get BOTH a NeMo-native number and a board-comparable one.
#
# Reads the PRE-STAGED cache on lustre (stage it with launch/stage_leaderboard_cache.sh):
#   <CACHE_DIR>/<dataset>/<split>/_cache_manifest.jsonl  ({audio_filepath,duration,reference})
#
# Usage (from the clean repo root on OCI):
#   sbatch launch/eval_parakeet_nemo.sh                        # default .nemo
#   sbatch launch/eval_parakeet_nemo.sh /path/to/model.nemo    # $1 = .nemo path
#   MAX_SAMPLES=50 sbatch launch/eval_parakeet_nemo.sh         # quick check (see note)
#
# Key env:
#   NEMO_MODEL   .nemo to eval (default parakeet-tdt-0.6b-v2 on fs12; $1 overrides)
#   CACHE_DIR    pre-staged leaderboard cache root on lustre
#   BATCH_SIZE   transcribe batch size (default 16)
#   DATASETS     space/comma 'name:split' list (default = current public suite)
#   MAX_SAMPLES  cap utts per dataset via a trimmed temp manifest (0 = all)
# ============================================================================
# NOTE: not `set -euo pipefail` (heredoc `read -d ''` returns non-zero at EOF).

mkdir -p slurm_out

PROJECT="${PROJECT:-SpeechlmScriptClean}"
NEMO_MODEL="${1:-${NEMO_MODEL:-/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/hainanx/pretrained_models/parakeet-tdt-0.6b-v2.nemo}}"
EXP_NAME="${EXP_NAME:-$(basename "${NEMO_MODEL%.nemo}")}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DATASETS="${DATASETS:-librispeech:test.clean librispeech:test.other ami_cleaned:test earnings22:test gigaspeech_cleaned:test spgispeech:test voxpopuli_cleaned_aa:test}"

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_script_clean}"
CACHE_DIR="${CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache}"
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh

if [[ ! -f "$NEMO_MODEL" ]]; then
    echo "ERROR: .nemo model not found: $NEMO_MODEL (pass the path as \$1 or NEMO_MODEL=)." >&2
    exit 1
fi
if [[ ! -d "$CACHE_DIR" ]]; then
    echo "ERROR: CACHE_DIR not found: $CACHE_DIR (stage the leaderboard cache first)." >&2
    exit 1
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
JOB_TAG="${SLURM_JOB_ID:-local$$}"
RESULTS_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/parakeet_nemo_eval_${RUN_TS}_${JOB_TAG}"
PRED_DIR="${RESULTS_DIR}/preds"
mkdir -p "$PRED_DIR"

HFCACHE="${OUTPUT_PREFIX}/hf_cache"; mkdir -p "$HFCACHE"
OCI_TMP_DIR="${OCI_TMP_DIR:-/tmp/parakeet_nemo_${SLURM_JOB_ID:-$$}}"

OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out
# Direct leaf binds (autofs sub-paths don't propagate into the container under a
# broad bind; see launch/eval_leaderboard.sh for the full rationale). Broad bind
# FIRST as a catch-all so it can't shadow the specific child mounts.
MODEL_DIR="$(dirname "$NEMO_MODEL")"
MOUNTS="--container-mounts=/lustre/fsw:/lustre/fsw,${CODE_DIR}:/code,${OUTPUT_PREFIX}:${OUTPUT_PREFIX},${CACHE_DIR}:${CACHE_DIR},${H_DIR}:${H_DIR},${HFCACHE}:/hfcache/"
case ":${OUTPUT_PREFIX}:${CACHE_DIR}:${H_DIR}:${CODE_DIR}:" in
    *":${MODEL_DIR}:"*) : ;;
    *) MOUNTS="${MOUNTS},${MODEL_DIR}:${MODEL_DIR}" ;;
esac

# Comma-normalize the dataset list for the in-container loop.
DATASETS_SP="$(echo "$DATASETS" | tr ',' ' ')"

echo "==> Parakeet NeMo-native eval (single GPU, no sharding)"
echo "    nemo_model:  ${NEMO_MODEL}"
echo "    datasets:    ${DATASETS_SP}"
echo "    cache_dir:   ${CACHE_DIR}"
echo "    results_dir: ${RESULTS_DIR}"

# ---- Record run config ----
{
    echo "# Parakeet NeMo-native (speech_to_text_eval.py) leaderboard eval"
    echo "timestamp:     ${RUN_TS}"
    echo "slurm_job_id:  ${JOB_TAG}"
    echo "exp_name:      ${EXP_NAME}"
    echo "project:       ${PROJECT}"
    echo "backend:       parakeet_nemo_native"
    echo "decode:        examples/asr/speech_to_text_eval.py (single GPU)"
    echo "scoring:       open_asr_leaderboard rescore (vendored normalizer + kaldialign merge_compounds)"
    echo "nemo_model:    ${NEMO_MODEL}"
    echo "batch_size:    ${BATCH_SIZE}"
    echo "max_samples:   ${MAX_SAMPLES}"
    echo "datasets:      ${DATASETS_SP}"
    echo "cache_dir:     ${CACHE_DIR}"
    echo "results_dir:   ${RESULTS_DIR}"
} > "${RESULTS_DIR}/run_config.yaml"

read -r -d '' cmd <<EOF
echo "*******Parakeet NeMo-native leaderboard eval (single GPU)********" \
&& echo "*** MODEL=${NEMO_MODEL} ***" \
&& nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
&& cd /code/examples/asr \
&& export PYTHONPATH="/code:\${PYTHONPATH}" \
&& export HF_HOME="/hfcache/" \
&& export HF_HUB_OFFLINE=1 \
&& export TOKENIZERS_PARALLELISM=false \
&& export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
&& export HYDRA_FULL_ERROR=1 \
&& export TMPDIR=${OCI_TMP_DIR} && mkdir -p ${OCI_TMP_DIR} \
&& python -c "import nemo, nemo.collections.asr; print('NeMo:', nemo.__version__)" \
&& { python -c "import kaldialign; kaldialign.batch_error_rate" 2>/dev/null || { echo '==> installing/upgrading kaldialign'; pip install -U --no-input --quiet kaldialign; }; } \
&& for entry in ${DATASETS_SP}; do \
      name="\${entry%%:*}"; split="\${entry#*:}"; [ "\$split" = "\$entry" ] && split=test; \
      man="${CACHE_DIR}/\${name}/\${split}/_cache_manifest.jsonl"; \
      out="${PRED_DIR}/\${name}__\${split}.json"; \
      if [ ! -f "\$man" ]; then echo "  [skip] missing manifest: \$man"; continue; fi; \
      use_man="\$man"; \
      if [ "${MAX_SAMPLES}" -gt 0 ]; then use_man="${OCI_TMP_DIR}/\${name}__\${split}.head.jsonl"; head -n ${MAX_SAMPLES} "\$man" > "\$use_man"; fi; \
      echo "==> [\${name}/\${split}] decoding -> \${out}"; \
      python speech_to_text_eval.py \
        model_path="${NEMO_MODEL}" \
        dataset_manifest="\$use_man" \
        output_filename="\$out" \
        gt_text_attr_name=reference \
        batch_size=${BATCH_SIZE} \
        amp=True \
        use_cer=False \
        text_processing.do_lowercase=true \
        text_processing.rm_punctuation=true \
      || echo "  WARN: speech_to_text_eval failed for \${name}/\${split}"; \
   done \
&& echo "" \
&& echo "==================== Leaderboard WER (rescored) ====================" \
&& python /code/scripts/rescore_nemo_eval.py --pred_dir "${PRED_DIR}" 2>&1 | tee "${RESULTS_DIR}/leaderboard_rescore.log" \
&& echo "" \
&& echo "Predictions + logs under: ${RESULTS_DIR}"
EOF

# Write the container command inside the checkout (bind-mounted at /code) so it's
# always visible in the container (a path under the broad /lustre/fsw tree may not
# be -- autofs; see MOUNTS note).
mkdir -p "${CODE_DIR}/slurm_out"
CMD_BASENAME="parakeet_nemo_cmd_${SLURM_JOB_ID:-local$$}.sh"
printf '%s\n' "$cmd" > "${CODE_DIR}/slurm_out/${CMD_BASENAME}"
chmod +x "${CODE_DIR}/slurm_out/${CMD_BASENAME}"
echo "==> Container command: /code/slurm_out/${CMD_BASENAME}"

srun -o "$OUTFILE" -e "$ERRFILE" --container-image="$CONTAINER" $MOUNTS bash "/code/slurm_out/${CMD_BASENAME}"
