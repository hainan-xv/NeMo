#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-build-spe-vocabs
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 04:00:00
#SBATCH --time-min 01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# Build the CHAT SentencePiece vocabularies: 8k, 16k, 32k.
#
#   sbatch launch/dfw_build_spe_vocabs.sh
#
# WHY. The CHAT arms currently borrow Qwen3's ~151.7k vocabulary, which is a
# MULTILINGUAL LLM vocabulary: most of its pieces can never appear in English
# ASR output, and the ones that do are shaped by an LM's text distribution
# rather than by transcribed speech. That was the right choice for testing
# whether the banded loss makes a large vocabulary affordable -- it does -- but
# it confounds "does the method work" with "is this the right vocabulary".
# A vocabulary trained on the actual transcripts isolates that.
#
# CAPITALIZATION AND PUNCTUATION ARE KEPT (--no_lower_case, and no stripping in
# the corpus builder): these models emit punctuated, cased text, and a
# lowercased vocabulary would force every capital through a separate piece.
#
# The corpus is sampled in proportion to each corpus's TRAINING WEIGHT -- see
# scripts/build_chat_spe_corpus.py for why uniform sampling would fit the
# vocabulary to a distribution the model never trains on.
#
# ENV
#   MAX_LINES   corpus size (default 3,000,000)
#   VOCABS      sizes to build (default "8192 16384 32768")
#   SPE_TYPE    bpe | unigram (default bpe, matching Qwen's algorithm)
# ============================================================================
set -uo pipefail
mkdir -p slurm_out

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
MY=${DFW}/hainanx
HEH=${DFW}/users/heh
CONTAINER="${CONTAINER:-${HEH}/containers/nemo-26.02-streaming-speechlm.sqsh}"
CODE_DIR="${CODE_DIR:-${MY}/NeMo_SCRIPT_cc}"
INPUT_CFG="${INPUT_CFG:-${HEH}/data_configs/granary_v2_en_full_d0.5_b0.5_dfw_qwen_aligned.yaml}"

OUT_ROOT="${MY}/tokenizers/granary2_en_spe"
CORPUS="${OUT_ROOT}/corpus.txt"
MAX_LINES="${MAX_LINES:-3000000}"
VOCABS="${VOCABS:-8192 16384 32768}"
SPE_TYPE="${SPE_TYPE:-bpe}"
mkdir -p "$OUT_ROOT"

DATA_ROOT=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm
MOUNTS="--container-mounts=${CODE_DIR}:/code,${DFW}:${DFW},${DATA_ROOT}:${DATA_ROOT}"

echo "==> building CHAT SentencePiece vocabularies"
echo "    corpus:  ${CORPUS}  (max ${MAX_LINES} lines, weighted by training mixture)"
echo "    vocabs:  ${VOCABS}  type=${SPE_TYPE}  cased, punctuated"

# ---- 1. corpus -------------------------------------------------------------
if [[ -s "$CORPUS" ]]; then
    echo "==> reusing existing corpus ($(wc -l < "$CORPUS") lines); delete it to rebuild"
else
    srun --container-image="$CONTAINER" $MOUNTS bash -c "
        cd /code && export PYTHONPATH=/code:\${PYTHONPATH:-} &&
        python scripts/build_chat_spe_corpus.py \
            --input_cfg ${INPUT_CFG} --out ${CORPUS} --max_lines ${MAX_LINES}
    " || { echo "ERROR: corpus extraction failed" >&2; exit 1; }
fi
[[ -s "$CORPUS" ]] || { echo "ERROR: empty corpus ${CORPUS}" >&2; exit 1; }

# ---- 2. one vocabulary per size -------------------------------------------
# Built SEQUENTIALLY on purpose: SentencePiece training is memory-hungry and a
# 32k build over millions of lines can peak high enough to take the others down
# with it if they share a node.
for V in ${VOCABS}; do
    D="${OUT_ROOT}/v${V}"
    if [[ -f "${D}/tokenizer.model" ]]; then
        echo "==> v${V}: reusing ${D}/tokenizer.model"
        continue
    fi
    echo; echo "############ vocab ${V}"
    mkdir -p "$D"
    srun --container-image="$CONTAINER" $MOUNTS bash -c "
        cd /code && export PYTHONPATH=/code:\${PYTHONPATH:-} &&
        python scripts/tokenizers/process_asr_text_tokenizer.py \
            --data_file=${CORPUS} \
            --data_root=${D} \
            --vocab_size=${V} \
            --tokenizer=spe \
            --spe_type=${SPE_TYPE} \
            --spe_character_coverage=1.0 \
            --no_lower_case \
            --log
    " || { echo "  [FAIL] vocab ${V}" >&2; continue; }
    # process_asr_text_tokenizer nests its output; hoist it so the path a
    # training config points at contains tokenizer.model directly.
    found="$(find "$D" -name tokenizer.model | head -1)"
    if [[ -n "$found" && "$(dirname "$found")" != "$D" ]]; then
        cp "$(dirname "$found")"/* "$D"/ 2>/dev/null
        echo "  hoisted $(dirname "$found") -> ${D}"
    fi
done

echo; echo "############ summary"
for V in ${VOCABS}; do
    D="${OUT_ROOT}/v${V}"
    if [[ -f "${D}/tokenizer.model" ]]; then
        echo "  v${V}: OK  ${D}/tokenizer.model  ($(stat -c%s "${D}/tokenizer.model") bytes)"
    else
        echo "  v${V}: MISSING" >&2
    fi
done
