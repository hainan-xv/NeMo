#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:nemotron-streaming-eval
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
# CACHE-AWARE STREAMING leaderboard eval for the NEMOTRON 3.5 streaming RNNT
# model (nvidia/nemotron-speech-streaming-en-0.6b -- the SAME base ASR the
# script/SpeechLM models fine-tune from), SINGLE GPU.
#
# WHY THIS SCRIPT EXISTS: the nemotron model is a *cache-aware streaming* encoder
# (att_context_size = [left, right] with a small right lookahead). Decoding it
# through NeMo's OFFLINE full-context path (examples/asr/speech_to_text_eval.py,
# as launch/eval_parakeet_nemo.sh does) crashes with a "Floating point exception"
# (SIGFPE): the offline path assumes a full-context model and ends up dividing by
# a zero streaming chunk/step. The CORRECT way to decode this model is NeMo's own
# cache-aware streaming entrypoint, which simulates true chunked streaming using
# the model's configured att_context_size:
#     examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py
#
# HOW IT COMBINES OUR TWO EXISTING PIECES (per the design ask "just combine them"):
#   * DECODE  = NeMo's canonical streaming-infer example  (the "run the nemotron
#               streaming model" half)                                         and
#   * HARNESS = the SAME pre-staged leaderboard cache + leaderboard-faithful
#               re-scorer (scripts/rescore_nemo_eval.py: vendored normalizer +
#               kaldialign merge_compounds) used by launch/eval_parakeet_nemo.sh
#               and the SCRIPT/flush old-suite runs (the "score on the leaderboard
#               datasets" half).
# So the ONLY thing that differs from eval_parakeet_nemo.sh is the decode call;
# the cache, dataset loop, output layout, and scorer are identical -> numbers are
# directly comparable to the SCRIPT/flush aggregate.log on the same DATASETS.
#
# The streaming-infer example reads its reference from the manifest field `text`
# and only writes an output json when refs are present, so per dataset we first
# materialize a temp manifest that copies our staged `reference` field into `text`
# (the audio/duration are untouched). Its output file is renamed to the
# `<name>__<split>.json` layout rescore_nemo_eval.py expects, then rescored.
#
# Reads the PRE-STAGED cache on lustre (stage it with launch/stage_leaderboard_cache.sh):
#   <CACHE_DIR>/<dataset>/<split>/_cache_manifest.jsonl  ({audio_filepath,duration,reference})
#
# Usage (from the clean repo root on OCI):
#   sbatch launch/eval_nemotron_streaming.sh                       # nemotron, NEW cleaned 7-set
#   sbatch launch/eval_nemotron_streaming.sh /path/to/other.nemo   # $1 = .nemo path
#   DATASETS="tedlium:test" sbatch launch/eval_nemotron_streaming.sh   # single dataset
#   MAX_SAMPLES=50 sbatch launch/eval_nemotron_streaming.sh        # quick smoke test
#   ATT_CONTEXT_SIZE="[70,1]" sbatch launch/eval_nemotron_streaming.sh  # pick a lookahead
#   CHUNK_SIZE=16 LEFT_CHUNKS=2 sbatch launch/eval_nemotron_streaming.sh # full-ctx model -> simulate
#
# Key env:
#   NEMO_MODEL        .nemo to eval (default nemotron-speech-streaming-en-0.6b; $1 overrides)
#   CACHE_DIR         pre-staged leaderboard cache root on lustre
#   DATASETS          space/comma 'name:split' list (default = current NEW cleaned suite)
#   BATCH_SIZE        streaming batch size (default 32)
#   MAX_SAMPLES       cap utts per dataset via a trimmed temp manifest (0 = all)
#   ATT_CONTEXT_SIZE  optional "[left,right]" to pick a lookahead (default: model's own)
#   CHUNK_SIZE / SHIFT_SIZE / LEFT_CHUNKS  optional; only for FULL-context models
#                     (a streaming model sets its chunking automatically -> leave unset)
#   EXP_NAME          results-folder label (default: <nemo basename>_streaming)
# ============================================================================
# NOTE: not `set -euo pipefail` (heredoc `read -d ''` returns non-zero at EOF).

mkdir -p slurm_out

PROJECT="${PROJECT:-SpeechlmScriptClean}"
# Pin the Nemotron 3.5 streaming RNNT .nemo (EncDecRNNTBPEModel). $1 overrides.
# Same path used as `pretrained_asr` across examples/speechlm2/conf/*.yaml.
NEMO_MODEL="${1:-${NEMO_MODEL:-/lustre/fsw/portfolios/llmservice/users/heh/pretrained_models/huggingface/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo}}"
EXP_NAME="${EXP_NAME:-$(basename "${NEMO_MODEL%.nemo}")_streaming}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
# Default = current NEW cleaned public suite (matches the staged cache + the
# SCRIPT/flush cleaned aggregate.log). Override DATASETS for the old 8-set suite
# (see launch/eval_nemotron_nemo_oldsuite.sh, which just pins that list).
DATASETS="${DATASETS:-librispeech:test.clean librispeech:test.other ami_cleaned:test earnings22:test gigaspeech_cleaned:test spgispeech:test voxpopuli_cleaned_aa:test}"

# --- Optional streaming-decode knobs (empty => use the model's own defaults) ---
ATT_CONTEXT_SIZE="${ATT_CONTEXT_SIZE:-}"
CHUNK_SIZE="${CHUNK_SIZE:-}"
SHIFT_SIZE="${SHIFT_SIZE:-}"
LEFT_CHUNKS="${LEFT_CHUNKS:-}"
DECODE_CLAUSE=""
[[ -n "$ATT_CONTEXT_SIZE" ]] && DECODE_CLAUSE="${DECODE_CLAUSE} att_context_size=${ATT_CONTEXT_SIZE}"
[[ -n "$CHUNK_SIZE"        ]] && DECODE_CLAUSE="${DECODE_CLAUSE} chunk_size=${CHUNK_SIZE}"
[[ -n "$SHIFT_SIZE"        ]] && DECODE_CLAUSE="${DECODE_CLAUSE} shift_size=${SHIFT_SIZE}"
[[ -n "$LEFT_CHUNKS"       ]] && DECODE_CLAUSE="${DECODE_CLAUSE} left_chunks=${LEFT_CHUNKS}"

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
RESULTS_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/nemotron_streaming_eval_${RUN_TS}_${JOB_TAG}"
PRED_DIR="${RESULTS_DIR}/preds"
mkdir -p "$PRED_DIR"

HFCACHE="${OUTPUT_PREFIX}/hf_cache"; mkdir -p "$HFCACHE"
OCI_TMP_DIR="${OCI_TMP_DIR:-/tmp/nemotron_stream_${SLURM_JOB_ID:-$$}}"

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

echo "==> Nemotron cache-aware STREAMING eval (single GPU)"
echo "    nemo_model:  ${NEMO_MODEL}"
echo "    datasets:    ${DATASETS_SP}"
echo "    cache_dir:   ${CACHE_DIR}"
echo "    decode:      speech_to_text_cache_aware_streaming_infer.py ${DECODE_CLAUSE:-(model defaults)}"
echo "    results_dir: ${RESULTS_DIR}"

# ---- Record run config ----
{
    echo "# Nemotron cache-aware streaming leaderboard eval"
    echo "timestamp:     ${RUN_TS}"
    echo "slurm_job_id:  ${JOB_TAG}"
    echo "exp_name:      ${EXP_NAME}"
    echo "project:       ${PROJECT}"
    echo "backend:       nemotron_cache_aware_streaming"
    echo "decode:        examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py (single GPU)"
    echo "decode_knobs:  ${DECODE_CLAUSE:-<model defaults>}"
    echo "scoring:       open_asr_leaderboard rescore (vendored normalizer + kaldialign merge_compounds)"
    echo "nemo_model:    ${NEMO_MODEL}"
    echo "batch_size:    ${BATCH_SIZE}"
    echo "max_samples:   ${MAX_SAMPLES}"
    echo "datasets:      ${DATASETS_SP}"
    echo "cache_dir:     ${CACHE_DIR}"
    echo "results_dir:   ${RESULTS_DIR}"
} > "${RESULTS_DIR}/run_config.yaml"

read -r -d '' cmd <<EOF
echo "*******Nemotron cache-aware STREAMING leaderboard eval (single GPU)********" \
&& echo "*** MODEL=${NEMO_MODEL} ***" \
&& nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
&& cd /code \
&& git rev-parse HEAD \
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
      if [ ! -f "\$man" ]; then echo "  [skip] missing manifest: \$man"; continue; fi; \
      src="\$man"; \
      if [ "${MAX_SAMPLES}" -gt 0 ]; then src="${OCI_TMP_DIR}/\${name}__\${split}.head.jsonl"; head -n ${MAX_SAMPLES} "\$man" > "\$src"; fi; \
      tmpman="${OCI_TMP_DIR}/\${name}__\${split}.jsonl"; \
      python -c "import json,sys; f=open(sys.argv[1]); g=open(sys.argv[2],'w'); [g.write(json.dumps({**json.loads(x),'text':json.loads(x).get('reference','')})+chr(10)) for x in f if x.strip()]; g.close()" "\$src" "\$tmpman"; \
      sout="${PRED_DIR}/_raw_\${name}__\${split}"; mkdir -p "\$sout"; \
      echo "==> [\${name}/\${split}] streaming-decoding -> \${sout}"; \
      python examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
        model_path="${NEMO_MODEL}" \
        dataset_manifest="\$tmpman" \
        output_path="\$sout" \
        batch_size=${BATCH_SIZE} \
        cuda=0 \
        amp=True${DECODE_CLAUSE} \
      || echo "  WARN: streaming infer failed for \${name}/\${split}"; \
      out="\$(ls "\$sout"/streaming_out_*.json 2>/dev/null | head -n 1)"; \
      if [ -n "\$out" ]; then mv "\$out" "${PRED_DIR}/\${name}__\${split}.json"; else echo "  WARN: no streaming output json for \${name}/\${split}"; fi; \
   done \
&& echo "" \
&& echo "==================== Leaderboard WER (rescored) ====================" \
&& python /code/scripts/rescore_nemo_eval.py --pred_dir "${PRED_DIR}" --gt_field text --pred_field pred_text 2>&1 | tee "${RESULTS_DIR}/leaderboard_rescore.log" \
&& echo "" \
&& echo "Predictions + logs under: ${RESULTS_DIR}"
EOF

# Write the container command inside the checkout (bind-mounted at /code) so it's
# always visible in the container (a path under the broad /lustre/fsw tree may not
# be -- autofs; see MOUNTS note).
mkdir -p "${CODE_DIR}/slurm_out"
CMD_BASENAME="nemotron_stream_cmd_${SLURM_JOB_ID:-local$$}.sh"
printf '%s\n' "$cmd" > "${CODE_DIR}/slurm_out/${CMD_BASENAME}"
chmod +x "${CODE_DIR}/slurm_out/${CMD_BASENAME}"
echo "==> Container command: /code/slurm_out/${CMD_BASENAME}"

srun -o "$OUTFILE" -e "$ERRFILE" --container-image="$CONTAINER" $MOUNTS bash "/code/slurm_out/${CMD_BASENAME}"
