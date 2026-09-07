#!/bin/bash
# ============================================================================
# Shared body for the CHAT leaderboard evals. Never launched directly -- each
# arm has its own script (eval_chat_delay3.sh, ...) that names its settings.
#
# Two steps in one allocation:
#   1. average the TOP-K checkpoints by val_wer into a .nemo, using NeMo's own
#      scripts/checkpoint_averaging/average_model_checkpoints.py
#   2. run the ordinary Open-ASR-Leaderboard eval on that .nemo
#
# Averaging needs the model's architecture, which differs per arm (history
# chunks, flexible delay), so the arm's own overrides are passed through to the
# same training config the run used. The averaged .nemo records `target =
# EncDecCHATBPEModel`, so eval_nemotron.sh restores the right class with no
# special-casing and every CHAT number stays comparable to the nemotron one.
# ============================================================================
set -euo pipefail

: "${ARM_EXP_NAME:?each eval wrapper must set ARM_EXP_NAME}"
TOPK="${TOPK:-5}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
PROJECT="${PROJECT:-SpeechlmScriptCC}"
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_SCRIPT_cc}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"

RUN_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${ARM_EXP_NAME}"
CKPT_DIR="${RUN_DIR}/${ARM_EXP_NAME}/checkpoints"
TOKENIZER_DIR="${RUN_DIR}/tokenizer"
# The donor .nemo IS the tokenizer. Runs before the /results fix left no
# tokenizer on lustre, so extract it here when it is missing rather than failing
# -- it is the same 1,024-piece vocabulary either way.
INIT_NEMO="${INIT_NEMO:-/lustre/fsw/portfolios/llmservice/users/heh/pretrained_models/huggingface/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo}"
AVG_DIR="${RUN_DIR}/averaged"
AVG_NAME="${AVG_DIR}/top${TOPK}"
AVG_NEMO="${AVG_NAME}-averaged.nemo"

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "ERROR: no checkpoints at ${CKPT_DIR}" >&2
    exit 1
fi
mkdir -p "$AVG_DIR"

if [[ ! -f "${TOKENIZER_DIR}/tokenizer.model" ]]; then
    echo "==> no tokenizer at ${TOKENIZER_DIR}; extracting from the donor"
    mkdir -p "$TOKENIZER_DIR"
    python3 - "$INIT_NEMO" "$TOKENIZER_DIR" <<'PYEOF'
import os, sys, tarfile
src, dst = sys.argv[1], sys.argv[2]
with tarfile.open(src, "r:") as tf:
    for m in tf.getmembers():
        for want in ("tokenizer.model", "tokenizer.vocab", "vocab.txt"):
            if m.name.endswith(want):
                m.name = want
                tf.extract(m, dst)
assert os.path.isfile(os.path.join(dst, "tokenizer.model")), "no tokenizer.model in " + src
print("    tokenizer ->", dst)
PYEOF
fi

# Top-K by val_wer, read off the filenames the checkpoint callback writes.
# `-last` and `-unfinished` are excluded: `-last` duplicates a scored checkpoint
# (averaging it would silently double its weight) and `-unfinished` may be a
# partial write.
mapfile -t BEST < <(ls "${CKPT_DIR}"/*.ckpt 2>/dev/null \
    | grep -v -- '-last' | grep -v -- 'unfinished' \
    | sed -E 's/.*val_wer=([0-9.]+).*/\1 &/' | sort -g -k1,1 | head -n "$TOPK" | cut -d' ' -f2-)

if [[ ${#BEST[@]} -eq 0 ]]; then
    echo "ERROR: no scored checkpoints in ${CKPT_DIR}" >&2
    exit 1
fi
echo "==> averaging ${#BEST[@]} checkpoints for ${ARM_EXP_NAME}:"
printf '      %s\n' "${BEST[@]##*/}"
CKPT_CSV="$(IFS=,; echo "${BEST[*]}")"

MOUNTS="--container-mounts=${CODE_DIR}:/code,${OUTPUT_PREFIX}:${OUTPUT_PREFIX},/lustre/fsw/portfolios/llmservice:/lustre/fsw/portfolios/llmservice"

if [[ -f "$AVG_NEMO" && "${FORCE_AVERAGE:-0}" != "1" ]]; then
    echo "==> reusing existing ${AVG_NEMO} (FORCE_AVERAGE=1 to rebuild)"
else
    srun --ntasks=1 --nodes=1 --container-image="$CONTAINER" $MOUNTS bash -c "
        cd /code && export PYTHONPATH=/code:\$PYTHONPATH HYDRA_FULL_ERROR=1 &&
        python scripts/checkpoint_averaging/average_model_checkpoints.py \
            --config-path=/code/examples/asr/conf/fastconformer/cache_aware_streaming \
            --config-name=nemotron_chat_transducer_granary2 \
            name=${AVG_NAME} \
            +model_class=nemo.collections.asr.models.EncDecCHATBPEModel \
            +checkpoint_paths=\\\"[${CKPT_CSV}]\\\" \
            model.tokenizer.dir=${TOKENIZER_DIR} \
            ${ARM_MODEL_OVERRIDES:-} \
            ~model.train_ds ~model.validation_ds ~model.test_ds \
            ~trainer.strategy \
            trainer.devices=1 trainer.accelerator=cpu trainer.precision=32 \
            trainer.num_nodes=1 trainer.logger=false trainer.enable_checkpointing=false
    "
fi

[[ -f "$AVG_NEMO" ]] || { echo "ERROR: averaging produced no ${AVG_NEMO}" >&2; exit 1; }
echo "==> averaged model: ${AVG_NEMO}"

export MODEL_PATH="$AVG_NEMO"
export EXP_NAME="$ARM_EXP_NAME"
export EVAL_TAG="${EVAL_TAG:-avg${TOPK}}"

find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_nemotron.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_nemotron.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_nemotron.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate eval_nemotron.sh" >&2; exit 1
}
exec bash "$(find_launch_dir)/eval_nemotron.sh"
