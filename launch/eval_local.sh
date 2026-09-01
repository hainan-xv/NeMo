#!/bin/bash
# ============================================================================
# Run a leaderboard-style eval on THIS machine -- no Slurm, no container.
#
# WHY. Speech Commands is 11,005 one-second clips: the cluster ran it in 2m31s
# on 8 GPUs, so the queue wait dominates the work by two orders of magnitude.
# Two local A6000s finish it in minutes with no queue at all.
#
# IDENTICAL SCORING BY CONSTRUCTION. This calls the SAME driver, the SAME
# run_eval_shards.sh fan-out (with its dead-GPU recovery), and the SAME
# `--aggregate` pass as launch/eval_leaderboard.sh. The only differences are
# mechanical: no srun/container, N local GPUs instead of 8, and paths under
# $HOME instead of lustre. Nothing about normalisation, alignment or WER
# differs, so a number from here is comparable to a cluster number.
#
# USAGE
#   launch/eval_local.sh <model-key> [chunk_size]
#
#   launch/eval_local.sh script       14
#   launch/eval_local.sh fullctx      14
#   launch/eval_local.sh nemotron     14
#   for m in script fullctx nemotron; do launch/eval_local.sh $m 14; done
#
# KNOBS (env)
#   DATASETS           default "speech_commands:test"
#   NGPU               default = all visible GPUs
#   BATCH_SIZE         default 32
#   MAX_NEW_TOKENS     default 256 -- deliberately high, see below
#   CACHE_DIR          default ~/Workplace/leaderboard_cache
#   RESULTS_ROOT       default ~/Workplace/eval_results_local
#   MAX_EVAL_SAMPLES   default 0 (all); set small for a smoke test
#
# MAX_NEW_TOKENS is 256 rather than the usual 64 because the question this
# dataset answers is whether a model OVER-generates on a 1-second clip. A tight
# cap would truncate exactly the behaviour being measured and understate it.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

MODEL_KEY="${1:-}"
CHUNK_SIZE="${2:-${CHUNK_SIZE:-14}}"
if [[ -z "$MODEL_KEY" ]]; then
    echo "usage: launch/eval_local.sh <script|fullctx|nemotron> [chunk_size]" >&2
    exit 1
fi

CKPT_DIR="${CKPT_DIR:-$HOME/Workplace/ckpts}"
CACHE_DIR="${CACHE_DIR:-$HOME/Workplace/leaderboard_cache}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/Workplace/eval_results_local}"
DATASETS="${DATASETS:-speech_commands:test}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
# Same defaults as the cluster path, so the two are comparable.
PAD_EXTRA_SECONDS="${PAD_EXTRA_SECONDS:-0.5}"
SHUFFLE_SEED="${SHUFFLE_SEED:-1234}"
MAX_HISTORY_TOKENS="${MAX_HISTORY_TOKENS:-0}"

SCRIPT_PROMPT="You are doing streaming speech recognition. Given the transcript so far and the representation of the next audio chunk, output the words spoken in that chunk."

case "$MODEL_KEY" in
    script)
        EXP="granary2_script_baseline"
        CKPT="${CKPT_DIR}/granary2_script_baseline-averaged.ckpt"
        DRIVER="scripts/script_leaderboard_eval.py"
        MODEL_CLASS="nemo.collections.speechlm2.models.script_model.ScriptSTTModel"
        SYSTEM_PROMPT="$SCRIPT_PROMPT"
        DRIVER_ARGS="--max_history_tokens ${MAX_HISTORY_TOKENS}"
        ;;
    fullctx)
        EXP="granary2_script_fullctx_ft"
        CKPT="${CKPT_DIR}/granary2_script_fullctx_ft-averaged.ckpt"
        DRIVER="scripts/script_leaderboard_eval.py"
        MODEL_CLASS="nemo.collections.speechlm2.models.script_model.ScriptSTTModel"
        # Same instruction as every other SCRIPT recipe -- full_context changes
        # the LAYOUT, not the prompt, and it rides in the checkpoint config.
        SYSTEM_PROMPT="$SCRIPT_PROMPT"
        DRIVER_ARGS="--max_history_tokens ${MAX_HISTORY_TOKENS}"
        ;;
    nemotron)
        # The streaming RNNT/cache-aware ASR baseline -- no LLM, so it has no
        # generative prior to over-generate WITH. That is exactly why it belongs
        # in this comparison: it is the control for "is over-generation an LLM
        # artefact or an ASR one?"
        EXP="nemotron_streaming_0.6b"
        CKPT="${NEMOTRON_NEMO:-$(find "$HOME/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b" -name '*.nemo' 2>/dev/null | head -1)}"
        DRIVER="scripts/nemotron_leaderboard_eval.py"
        MODEL_CLASS=""
        SYSTEM_PROMPT=""
        # "streaming" = true cache-aware stepping, which is what this model is
        # for and what makes it the right control here. "offline" restricts the
        # encoder identically but encodes in one pass; equivalent in dependency
        # structure, not in cache-boundary behaviour.
        NEMOTRON_MODE="${NEMOTRON_MODE:-streaming}"
        DRIVER_ARGS="--mode ${NEMOTRON_MODE}"
        ;;
    *)
        echo "ERROR: unknown model key '${MODEL_KEY}' (want script|fullctx|nemotron)" >&2
        exit 1 ;;
esac

if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint for '${MODEL_KEY}' not found: ${CKPT:-<unset>}" >&2
    exit 1
fi
if [[ ! -d "${CACHE_DIR}/${DATASETS%%:*}" ]]; then
    echo "ERROR: no staged data at ${CACHE_DIR}/${DATASETS%%:*}" >&2
    echo "       stage it with scripts/stage_speech_commands.py" >&2
    exit 1
fi

NGPU="${NGPU:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}"
[[ "$NGPU" -lt 1 ]] && { echo "ERROR: no GPUs visible" >&2; exit 1; }

DS_SLUG="$(echo "$DATASETS" | tr ' ,' '__' | tr -cd '[:alnum:]_-' | cut -c1-40)"
LABEL="${DS_SLUG}_chunk${CHUNK_SIZE}"
RESULTS_DIR="${RESULTS_ROOT}/${EXP}/${LABEL}"
SHARD_DIR="${RESULTS_DIR}/shards"
mkdir -p "$SHARD_DIR"

echo "==> local eval"
echo "    model:    ${MODEL_KEY}  (${EXP})"
echo "    ckpt:     ${CKPT}"
echo "    driver:   ${DRIVER}"
echo "    data:     ${DATASETS}   cache=${CACHE_DIR}"
echo "    chunk:    ${CHUNK_SIZE}   gpus=${NGPU}   batch=${BATCH_SIZE}   max_new_tokens=${MAX_NEW_TOKENS}"
echo "    results:  ${RESULTS_DIR}"

export PYTHONPATH="${HERE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASETS_CSV="$(echo "$DATASETS" | tr ' ' ',')"

# Stale generations from a previous run would be aggregated alongside the new
# ones -- the fan-out only overwrites the shards it actually produces.
rm -f "${SHARD_DIR}"/shard*_of*.generations.jsonl

COMMON=(
    --datasets "$DATASETS_CSV"
    --cache_dir "$CACHE_DIR"
    --output_dir "$SHARD_DIR"
    --shuffle_seed "$SHUFFLE_SEED"
    --batch_size "$BATCH_SIZE"
    --max_eval_samples "$MAX_EVAL_SAMPLES"
    --chunk_size "$CHUNK_SIZE"
)
# The two drivers have genuinely different interfaces, so build the rest per
# driver rather than passing flags one of them would reject:
#   nemotron: --model_path, no --pad_extra_seconds, no LLM args
#   script:   --ckpt_path, --model_class, --system_prompt, --max_new_tokens
if [[ "$MODEL_KEY" == "nemotron" ]]; then
    COMMON+=(--model_path "$CKPT")
    # shellcheck disable=SC2206
    COMMON+=(${DRIVER_ARGS})
else
    COMMON+=(
        --ckpt_path "$CKPT"
        --model_class "$MODEL_CLASS"
        --system_prompt "$SYSTEM_PROMPT"
        --max_new_tokens "$MAX_NEW_TOKENS"
        --pad_extra_seconds "$PAD_EXTRA_SECONDS"
    )
    # shellcheck disable=SC2206
    COMMON+=(${DRIVER_ARGS})
fi

# NOTE: run_eval_shards.sh recovers a dead GPU by re-running its slice with
# --subshard_count/--subshard_index. The nemotron driver does not define those,
# so recovery would fail for it -- it would report the bad shard rather than
# silently aggregating a partial set, which is the safe direction.
bash scripts/run_eval_shards.sh \
    --ngpu "$NGPU" --shard-dir "$SHARD_DIR" --log-dir "$RESULTS_DIR" \
    --driver "${HERE}/${DRIVER}" -- "${COMMON[@]}"

echo "==================== WER ===================="
python "${HERE}/${DRIVER}" --aggregate --output_dir "$SHARD_DIR" 2>&1 | tee "${RESULTS_DIR}/aggregate.log"
echo "==> ${RESULTS_DIR}/aggregate.log"
