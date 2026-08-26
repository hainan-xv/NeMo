#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:script-leaderboard-eval
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
# Open-ASR-Leaderboard eval backend for SCRIPT models.
#
# Shared backend: thin per-model wrappers (launch/eval_script_baseline.sh) set a
# few env vars and `exec` this script INSIDE the same allocation, so its #SBATCH
# headers are inert comments and there is exactly one job / one node / 8 GPUs.
# You can also submit it directly:  sbatch launch/eval_leaderboard.sh <exp_name>
#
# It fans out 8 independent python processes, one per GPU. There is no
# torch.distributed: all utterances from all datasets are POOLED, globally
# shuffled with a shared seed, and split `pos % 8`, so every GPU gets a
# length-balanced mix instead of whole datasets (which differ hugely in size and
# clip length). Each shard writes a JSONL; a final reduce pass computes
# per-dataset WER plus a macro average.
#
# PREREQUISITE: the dataset cache must already be staged on lustre (CACHE_DIR).
# Compute nodes run HF_HUB_OFFLINE=1 and download nothing. Stage once with
#   sbatch launch/stage_leaderboard_cache.sh
#
# KEY KNOBS (all env vars; see the table in the header comments below)
#   EXP_NAME / PROJECT      which run's checkpoints to evaluate
#   SYSTEM_PROMPT           MUST match the model's training instruction
#   CHUNK_SIZE              decode chunk size in encoder frames
#   EVAL_DRIVER             driver under scripts/ (default script_leaderboard_eval.py;
#                           use speechlm_leaderboard_eval.py for StreamingSTTModel)
#   RUN_AVERAGING / CKPT / STEP / USE_LAST     which checkpoint
#   DATASETS, BATCH_SIZE, MAX_NEW_TOKENS, MAX_EVAL_SAMPLES, FORCE_WORD_START
#   NUM_DELAY_FRAMES        [prompt-controlled models] delay to request, in frames
#   CAPITALIZATION          [prompt-controlled models] 1/0
#   PUNCTUATION             [prompt-controlled models] 1/0
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

# NOTE: deliberately no `set -euo pipefail` -- the `read -r -d '' <<EOF` heredoc
# and the `ls | grep | head` pipelines below legitimately return non-zero.

read_optional_token() {
    [[ -r "$1" ]] && tr -d '\r\n' < "$1" || true
}
HF_TOKEN="$(read_optional_token "$HOME/.hf_token")"
WANDB_TOKEN="$(read_optional_token "$HOME/.wandb_token")"

mkdir -p slurm_out

# --- What to evaluate ---
EXP_NAME="${1:-${EXP_NAME:-granary2_script_baseline}}"
PROJECT="${PROJECT:-SpeechlmScriptCC}"
EVAL_DRIVER="${EVAL_DRIVER:-script_leaderboard_eval.py}"   # which driver under scripts/
MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"

# --- Decode configuration ---
# The model keys its behaviour on the exact training instruction; any drift here
# is out-of-distribution and will silently cost WER.
SYSTEM_PROMPT="${SYSTEM_PROMPT:-Transcribe the audio into text.}"
CHUNK_SIZE="${CHUNK_SIZE:-}"          # empty => model default (--chunk_size omitted)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_HISTORY_TOKENS="${MAX_HISTORY_TOKENS:-0}"
FORCE_WORD_START="${FORCE_WORD_START:-1}"
# Prompt-controlled models only. Empty => not requested => flag omitted entirely,
# because a model trained without prompt control rejects these outright.
NUM_DELAY_FRAMES="${NUM_DELAY_FRAMES:-}"
CAPITALIZATION="${CAPITALIZATION:-}"      # 1/0, empty = model default
PUNCTUATION="${PUNCTUATION:-}"            # 1/0, empty = model default
# Match training's data.dataset.pad_extra_duration: the trailing silence is real
# audio the encoder consumes, and it is where delay-held tail words land.
PAD_EXTRA_SECONDS="${PAD_EXTRA_SECONDS:-0.5}"

BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"   # 0 = all
SHUFFLE_SEED="${SHUFFLE_SEED:-1234}"        # must be identical across shards
NGPU="${NGPU:-8}"

DATASETS="${DATASETS:-librispeech:test.clean librispeech:test.other ami_cleaned:test earnings22:test gigaspeech_cleaned:test spgispeech:test voxpopuli_cleaned_aa:test}"
DATASETS_CSV="$(echo "$DATASETS" | tr -s ' ' ',')"

# --- Checkpoint selection ---
RUN_AVERAGING="${RUN_AVERAGING:-1}"   # average top-k non-last ckpts (cached)
FORCE_AVERAGE="${FORCE_AVERAGE:-0}"   # recompute even if the cached avg exists
USE_LAST="${USE_LAST:-0}"             # eval the rolling *-last.ckpt
STEP="${STEP:-}"                      # eval step=<n>.ckpt
CKPT="${CKPT:-}"                      # exact path

# --- Paths ---
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_SCRIPT_cc}"
CACHE_DIR="${CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache}"
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh
HFCACHE="${OUTPUT_PREFIX}/hf_cache"
# Deliberately SHORT: multiprocessing's AF_UNIX socket paths are capped at 108
# bytes, and a lustre TMPDIR blows past that.
OCI_TMP_DIR="${OCI_TMP_DIR:-/tmp/script_eval_${SLURM_JOB_ID:-$$}}"

if [[ ! -d "$CACHE_DIR" ]]; then
    echo "ERROR: leaderboard cache not found at ${CACHE_DIR}." >&2
    echo "       Stage it first:  sbatch launch/stage_leaderboard_cache.sh" >&2
    exit 1
fi

# --- Result location ---
RUN_TS="$(date +%Y%m%d_%H%M%S)"
JOB_TAG="${SLURM_JOB_ID:-local$$}"
CHUNK_TAG=""; [[ -n "$CHUNK_SIZE" ]] && CHUNK_TAG="_chunk${CHUNK_SIZE}"
EVAL_TAG_SUFFIX=""; [[ -n "${EVAL_TAG:-}" ]] && EVAL_TAG_SUFFIX="_${EVAL_TAG}"
RESULTS_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/leaderboard_eval${CHUNK_TAG}${EVAL_TAG_SUFFIX}_${RUN_TS}_${JOB_TAG}"
SHARD_DIR="${RESULTS_DIR}/shards"
mkdir -p "$SHARD_DIR" "$HFCACHE"
OUTFILE="${RESULTS_DIR}/slurm-%j-%n.out"
ERRFILE="${RESULTS_DIR}/error-%j-%n.out"

# ---------------------------------------------------------------------------
# Resolve the checkpoint. exp_manager nests the run name twice.
# ---------------------------------------------------------------------------
CKPT_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/${EXP_NAME}/checkpoints"

# An explicit checkpoint request always wins over (and disables) averaging.
if [[ -n "$CKPT" || -n "$STEP" || "$USE_LAST" == "1" ]]; then
    RUN_AVERAGING=0
fi

AVG_INPUTS_FILE=""
DO_AVG=0
if [[ "$RUN_AVERAGING" == "1" ]]; then
    AVG_CKPT="${CKPT_DIR}/${EXP_NAME}-averaged.ckpt"
    mapfile -t _AVG_IN < <(ls -t "${CKPT_DIR}"/*.ckpt 2>/dev/null | grep -v -- '-last\.ckpt$' | grep -v -- '-averaged\.ckpt$')
    if [[ ${#_AVG_IN[@]} -eq 0 ]]; then
        echo "ERROR: RUN_AVERAGING=1 but no non-last checkpoints under ${CKPT_DIR}." >&2
        echo "       Use USE_LAST=1 to eval the rolling -last.ckpt, or set CKPT=/STEP=." >&2
        exit 1
    fi
    CKPT="$AVG_CKPT"
    DO_AVG=1
    AVG_INPUTS_FILE="${SHARD_DIR}/avg_inputs.txt"
    printf '%s\n' "${_AVG_IN[@]}" > "$AVG_INPUTS_FILE"
    echo "==> Will average ${#_AVG_IN[@]} checkpoint(s) -> ${AVG_CKPT} (cached; reused unless FORCE_AVERAGE=1)"
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
        echo "ERROR: could not resolve a checkpoint under ${CKPT_DIR}." >&2
        echo "       Set CKPT=, STEP=, USE_LAST=1, or RUN_AVERAGING=1." >&2
        exit 1
    fi
fi

# --- Optional flags ---
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"   # verbatim extra flags for the chosen driver

FORCE_WORD_START_FLAG=""
[[ "$FORCE_WORD_START" == "0" || "$FORCE_WORD_START" == "false" ]] && FORCE_WORD_START_FLAG="--no_force_word_start"

# Prompt-control flags, emitted only when explicitly set.
CAP_FLAG=""
[[ "$CAPITALIZATION" == "1" || "$CAPITALIZATION" == "true"  ]] && CAP_FLAG="--capitalization"
[[ "$CAPITALIZATION" == "0" || "$CAPITALIZATION" == "false" ]] && CAP_FLAG="--no_capitalization"
PUNCT_FLAG=""
[[ "$PUNCTUATION" == "1" || "$PUNCTUATION" == "true"  ]] && PUNCT_FLAG="--punctuation"
[[ "$PUNCTUATION" == "0" || "$PUNCTUATION" == "false" ]] && PUNCT_FLAG="--no_punctuation"

# The system prompt can contain apostrophes, semicolons and quotes; passing it
# through the command string would be a quoting minefield. Write it to a file and
# read it back inside the container instead.
printf '%s' "$SYSTEM_PROMPT" > "${SHARD_DIR}/system_prompt.txt"

AVG_CLAUSE=""
if [[ "$DO_AVG" == "1" ]]; then
    AVG_CLAUSE="&& if [[ '${FORCE_AVERAGE}' == '1' || ! -f '${CKPT}' ]]; then echo '==> Averaging ${#_AVG_IN[@]} checkpoints -> ${CKPT}'; python /code/scripts/average_script_ckpts.py --output '${CKPT}' \$(cat '${AVG_INPUTS_FILE}'); else echo '==> Reusing cached averaged checkpoint: ${CKPT}'; fi "
fi

# --- wandb (optional) ---
REPORT_WANDB="${REPORT_WANDB:-auto}"
case "${REPORT_WANDB,,}" in
    auto)            [[ -n "$WANDB_TOKEN" ]] && REPORT_WANDB=1 || REPORT_WANDB=0 ;;
    1|true|yes|on)   REPORT_WANDB=1 ;;
    0|false|no|off)  REPORT_WANDB=0 ;;
    *) echo "WARNING: unrecognised REPORT_WANDB='${REPORT_WANDB}'; disabling." >&2; REPORT_WANDB=0 ;;
esac

# --- Run manifest, so a results dir is self-describing months later ---
# ---------------------------------------------------------------------------
# wandb identity
#
#   GROUP    = <model>_<checkpoint timestamp>   -- one group per model VERSION
#   RUN NAME = the decode configuration          -- one line per operating point
#
# So every decode setting for a given checkpoint lands as a separate line inside
# one group, and re-training the model starts a fresh group rather than mixing
# old and new numbers on the same axis.
#
# The timestamp comes from the newest checkpoint that FEEDS this eval, not from
# the file named in $CKPT: under averaging that file is produced inside the
# container and does not exist yet at this point.
CKPT_STAMP_SRC="$CKPT"
[[ "$DO_AVG" == "1" && ${#_AVG_IN[@]} -gt 0 ]] && CKPT_STAMP_SRC="${_AVG_IN[0]}"
if [[ -n "$CKPT_STAMP_SRC" && -e "$CKPT_STAMP_SRC" ]]; then
    CKPT_TS="$(date -r "$CKPT_STAMP_SRC" +%Y%m%d_%H%M 2>/dev/null || echo unknown)"
else
    CKPT_TS="unknown"
fi
# Step number, when the filename carries one -- more meaningful than a date when
# skimming a wandb group, so it rides along in the config.
CKPT_STEP="$(basename "${CKPT_STAMP_SRC:-}" 2>/dev/null | grep -oE 'step=[0-9]+' | head -1 | cut -d= -f2)"

# The line label: every knob that changes what is decoded, and nothing else.
# Deliberately excludes the run timestamp, so re-evaluating the same checkpoint
# at the same setting overlays instead of adding a near-duplicate line.
DECODE_LABEL="chunk${CHUNK_SIZE:-default}"
[[ -n "${NUM_DELAY_FRAMES}" ]] && DECODE_LABEL="${DECODE_LABEL}_d${NUM_DELAY_FRAMES}"
[[ -n "${CAPITALIZATION}"   ]] && DECODE_LABEL="${DECODE_LABEL}_c${CAPITALIZATION}"
[[ -n "${PUNCTUATION}"      ]] && DECODE_LABEL="${DECODE_LABEL}_p${PUNCTUATION}"
[[ "${STATE_MACHINE:-}"    == "1" ]] && DECODE_LABEL="${DECODE_LABEL}_sm"
[[ "${STREAMING_ENCODE:-}" == "1" ]] && DECODE_LABEL="${DECODE_LABEL}_se"
[[ "${USE_LAST}" == "1" ]] && DECODE_LABEL="${DECODE_LABEL}_last"
[[ "${RUN_AVERAGING}" != "1" && "${USE_LAST}" != "1" ]] && DECODE_LABEL="${DECODE_LABEL}_single"


cat > "${RESULTS_DIR}/run_config.yaml" <<YAML
timestamp: "${RUN_TS}"
job_id: "${JOB_TAG}"
exp_name: "${EXP_NAME}"
project: "${PROJECT}"
backend: "script"
eval_tag: "${EVAL_TAG:-}"
model_class: "${MODEL_CLASS}"
checkpoint: "${CKPT}"
run_averaging: ${RUN_AVERAGING}
ckpt_timestamp: "${CKPT_TS}"
ckpt_step: "${CKPT_STEP}"
decode_label: "${DECODE_LABEL}"
state_machine: "${STATE_MACHINE:-}"
streaming_encode: "${STREAMING_ENCODE:-}"
eval_driver: "${EVAL_DRIVER}"
num_averaged_inputs: ${#_AVG_IN[@]}
system_prompt: |
  ${SYSTEM_PROMPT}
chunk_size: "${CHUNK_SIZE}"
max_new_tokens: ${MAX_NEW_TOKENS}
max_history_tokens: ${MAX_HISTORY_TOKENS}
force_word_start: ${FORCE_WORD_START}
num_delay_frames: "${NUM_DELAY_FRAMES}"
capitalization: "${CAPITALIZATION}"
punctuation: "${PUNCTUATION}"
pad_extra_seconds: ${PAD_EXTRA_SECONDS}
batch_size: ${BATCH_SIZE}
num_gpus: ${NGPU}
shuffle_seed: ${SHUFFLE_SEED}
max_eval_samples: ${MAX_EVAL_SAMPLES}
datasets: "${DATASETS_CSV}"
cache_dir: "${CACHE_DIR}"
results_dir: "${RESULTS_DIR}"
YAML

echo "==> exp=${EXP_NAME} project=${PROJECT}"
echo "==> ckpt=${CKPT}"
echo "==> chunk_size=${CHUNK_SIZE:-<model default>} force_word_start=${FORCE_WORD_START}"
[[ -n "${NUM_DELAY_FRAMES}${CAPITALIZATION}${PUNCTUATION}" ]] && echo "==> prompt control: delay=${NUM_DELAY_FRAMES:-<default>} cap=${CAPITALIZATION:-<default>} punct=${PUNCTUATION:-<default>}"
echo "==> results -> ${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# In-container command
# ---------------------------------------------------------------------------
# /lustre/fsw is autofs; its lazily-mounted sub-paths do not propagate into the
# container's private namespace, so each leaf is bound directly with
# source==target. The broad catch-all must come FIRST so ancestor binds do not
# shadow the specific children that follow.
MOUNTS="--container-mounts=/lustre/fsw:/lustre/fsw,${CODE_DIR}:/code,${OUTPUT_PREFIX}:${OUTPUT_PREFIX},${CACHE_DIR}:${CACHE_DIR},${H_DIR}:${H_DIR},${HFCACHE}:/hfcache/"

WANDB_CLAUSE=""
if [[ "$REPORT_WANDB" == "1" ]]; then
    WANDB_GROUP="${WANDB_GROUP:-${EXP_NAME}_${CKPT_TS}}"
    WANDB_RUN_NAME="${WANDB_RUN_NAME:-${DECODE_LABEL}}"
    WANDB_EVAL_PROJECT="${WANDB_EVAL_PROJECT:-${PROJECT}_eval_v2}"
    WANDB_CLAUSE="&& { export WANDB_API_KEY='${WANDB_TOKEN}'; python /code/scripts/eval_wandb_report.py --project '${WANDB_EVAL_PROJECT}' --run_name '${WANDB_RUN_NAME}' --results_dir '${RESULTS_DIR}' --group '${WANDB_GROUP}' --job_type script 2>&1 | tee '${RESULTS_DIR}/wandb_report.log' || true; }"
fi

# Driver-specific flags. --max_history_tokens, --force_word_start and the
# prompt-control knobs exist only in script_leaderboard_eval.py; handing them to
# another driver is an argparse failure, not a no-op.
if [[ "$EVAL_DRIVER" == "script_leaderboard_eval.py" ]]; then
    DRIVER_ARGS="--max_history_tokens ${MAX_HISTORY_TOKENS} ${FORCE_WORD_START_FLAG}"
    DRIVER_ARGS="${DRIVER_ARGS} ${NUM_DELAY_FRAMES:+--num_delay_frames ${NUM_DELAY_FRAMES}} ${CAP_FLAG} ${PUNCT_FLAG}"
    [[ "${STATE_MACHINE:-}" == "1" ]] && DRIVER_ARGS="${DRIVER_ARGS} --state_machine"
    [[ "${STREAMING_ENCODE:-}" == "1" ]] && DRIVER_ARGS="${DRIVER_ARGS} --streaming_encode"
else
    DRIVER_ARGS=""
    for v in MAX_HISTORY_TOKENS:0 NUM_DELAY_FRAMES: CAPITALIZATION: PUNCTUATION:; do
        name="${v%%:*}"; default="${v#*:}"
        if [[ -n "${!name:-}" && "${!name}" != "$default" ]]; then
            echo "WARNING: ${name}=${!name} is a SCRIPT-only knob; ignored for EVAL_DRIVER=${EVAL_DRIVER}" >&2
        fi
    done
fi

read -r -d '' cmd <<EOF
echo "*******STARTING LEADERBOARD EVAL********" \
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
&& python -c "from nemo.utils.model_utils import import_class_by_path as I; I('${MODEL_CLASS}'); print('model class OK: ${MODEL_CLASS}')" \
&& python -c "import kaldialign; assert hasattr(kaldialign,'batch_error_rate')" 2>/dev/null || pip install -U --quiet kaldialign \
${AVG_CLAUSE} \
&& if [[ ! -f "${CKPT}" ]]; then echo "ERROR: checkpoint missing: ${CKPT}"; exit 1; fi \
&& echo "Pooled datasets: ${DATASETS_CSV}" \
&& echo "Fanning ${NGPU} balanced shards across ${NGPU} GPUs (seed=${SHUFFLE_SEED})..." \
&& rm -f ${SHARD_DIR}/shard*_of*.generations.jsonl \
&& SP_TEXT=\$(cat '${SHARD_DIR}/system_prompt.txt') \
&& pids=() \
&& for gpu in \$(seq 0 \$(( ${NGPU} - 1 ))); do \
      log="${RESULTS_DIR}/shard_\${gpu}.log"; \
      echo "  [gpu \$gpu] shard \$gpu/${NGPU} -> \${log}"; \
      CUDA_VISIBLE_DEVICES=\$gpu python /code/scripts/${EVAL_DRIVER} \
        --ckpt_path "${CKPT}" \
        --model_class "${MODEL_CLASS}" \
        --datasets "${DATASETS_CSV}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${SHARD_DIR}" \
        --num_shards ${NGPU} \
        --shard_index \$gpu \
        --shuffle_seed ${SHUFFLE_SEED} \
        --device 0 \
        --batch_size ${BATCH_SIZE} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --max_eval_samples ${MAX_EVAL_SAMPLES} \
        --system_prompt "\$SP_TEXT" \
        --pad_extra_seconds ${PAD_EXTRA_SECONDS} \
        ${CHUNK_SIZE:+--chunk_size ${CHUNK_SIZE}} \
        ${DRIVER_ARGS} ${EXTRA_EVAL_ARGS} \
        > "\${log}" 2>&1 & \
      pids+=(\$!); \
   done \
&& fail=0 && for p in "\${pids[@]}"; do wait "\$p" || fail=1; done \
&& echo "" \
&& echo "==================== Leaderboard WER ====================" \
&& python /code/scripts/${EVAL_DRIVER} --aggregate --output_dir "${SHARD_DIR}" 2>&1 | tee "${RESULTS_DIR}/aggregate.log" \
${WANDB_CLAUSE} \
&& echo "" \
&& echo "Per-shard logs + generations under: ${RESULTS_DIR}" \
&& exit \$fail
EOF

# Write the command to a file rather than passing `bash -c "$cmd"`: the system
# prompt may contain apostrophes that would otherwise break the quoting.

CMD_FILE="${RESULTS_DIR}/container_cmd.sh"
printf '%s\n' "$cmd" > "$CMD_FILE"
chmod +x "$CMD_FILE"

srun -o "$OUTFILE" -e "$ERRFILE" --container-image="$CONTAINER" $MOUNTS bash "$CMD_FILE"
