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
# same training config the run used. Only train_ds is dropped -- it would build
# the full Granary loader for nothing. validation_ds MUST be kept: transcribe()
# uses it as the template for its temporary dataloader, and without it every
# batch fails with "Key 'validation_ds' is not in struct" and the eval reports a
# clean, complete-looking 100% WER. The averaged .nemo records `target =
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

# The training config and vocabulary differ per arm. The 1,024-piece arms use
# the donor SentencePiece model extracted into the run dir; the Qwen arms use a
# HuggingFace directory and their own config. Defaulting to the 1k case keeps
# every existing wrapper working unchanged.
ARM_CONFIG_NAME="${ARM_CONFIG_NAME:-nemotron_chat_transducer_granary2}"
RUN_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${ARM_EXP_NAME}"
CKPT_DIR="${RUN_DIR}/${ARM_EXP_NAME}/checkpoints"
TOKENIZER_DIR="${ARM_TOKENIZER_DIR:-${RUN_DIR}/tokenizer}"
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
if [[ -n "${ARM_TOKENIZER_DIR:-}" ]]; then
    if [[ ! -f "${TOKENIZER_DIR}/tokenizer.model" && ! -f "${TOKENIZER_DIR}/tokenizer.json" ]]; then
        echo "ERROR: ARM_TOKENIZER_DIR=${TOKENIZER_DIR} holds no tokenizer.model or tokenizer.json" >&2
        exit 1
    fi
    echo "==> using the arm's own tokenizer: ${TOKENIZER_DIR}"
fi

if [[ -z "${ARM_TOKENIZER_DIR:-}" && ! -f "${TOKENIZER_DIR}/tokenizer.model" ]]; then
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
mapfile -t SCORED < <(ls "${CKPT_DIR}"/*.ckpt 2>/dev/null \
    | grep -v -- '-last' | grep -v -- 'unfinished' \
    | sed -E 's/.*val_wer=([0-9.]+).*/\1 &/' | sort -g -k1,1)

if [[ ${#SCORED[@]} -eq 0 ]]; then
    echo "ERROR: no scored checkpoints in ${CKPT_DIR}" >&2
    exit 1
fi

# DROP OUTLIERS BEFORE TAKING THE TOP K. Early in a run there are fewer than K
# scored checkpoints, so top-K silently reaches back to epoch 0 -- which on these
# arms sits at val_wer 0.63-0.82 against 0.15-0.18 for a trained epoch. Averaging
# that in produces a model that never existed, and the output looks completely
# normal: the banner lists K checkpoints and the leaderboard prints a number.
# Anything worse than MAX_WER_RATIO x the best is excluded and SAID SO, so a
# short arm reports an honest average over 2 checkpoints instead of a quiet
# fiction over 5. Set MAX_WER_RATIO=0 to disable.
MAX_WER_RATIO="${MAX_WER_RATIO:-2.0}"
BEST_WER="${SCORED[0]%% *}"
declare -a KEPT=() DROPPED=()
for row in "${SCORED[@]}"; do
    wer="${row%% *}"; path="${row#* }"
    if [[ "$MAX_WER_RATIO" != "0" ]] \
       && awk -v w="$wer" -v b="$BEST_WER" -v r="$MAX_WER_RATIO" 'BEGIN{exit !(b>0 && w>b*r)}'; then
        DROPPED+=("${path##*/}")
    else
        KEPT+=("$path")
    fi
done
if [[ ${#DROPPED[@]} -gt 0 ]]; then
    echo "==> excluding ${#DROPPED[@]} checkpoint(s) worse than ${MAX_WER_RATIO}x the best val_wer (${BEST_WER}):"
    printf '      %s\n' "${DROPPED[@]}"
fi

# Drop any checkpoint that has VANISHED since the listing above. A live training
# run with save_top_k deletes a checkpoint the moment a better one lands, so the
# averaging step can select a file and then fail on it with FileNotFoundError --
# observed, and it aborts the whole arm rather than degrading.
declare -a ALIVE=()
for path in "${KEPT[@]}"; do
    [[ -f "$path" ]] && ALIVE+=("$path")
done
if [[ ${#ALIVE[@]} -lt ${#KEPT[@]} ]]; then
    echo "==> $(( ${#KEPT[@]} - ${#ALIVE[@]} )) checkpoint(s) disappeared mid-listing (live training); using ${#ALIVE[@]}"
fi
KEPT=("${ALIVE[@]}")

BEST=("${KEPT[@]:0:$TOPK}")
if [[ ${#BEST[@]} -eq 0 ]]; then
    echo "ERROR: every checkpoint in ${CKPT_DIR} was excluded as an outlier" >&2
    exit 1
fi
echo "==> averaging ${#BEST[@]} checkpoints for ${ARM_EXP_NAME}:"
printf '      %s\n' "${BEST[@]##*/}"
CKPT_CSV="$(IFS=,; echo "${BEST[*]}")"

# The third mount carries the TOKENIZER and the donor .nemo, and it is
# cluster-specific: OCI keeps them under llmservice, DFW under the
# nemotron_speechprod_asr project. Averaging CONSTRUCTS the model, so an
# unmounted tokenizer directory is not a missing-file error -- transformers falls
# back to treating the path as a hub repo id and fails with "Repo id must be in
# the form 'repo_name'", which reads nothing like a mount problem. That is
# exactly how the first DFW CHAT eval failed.
EXTRA_MOUNTS="${EXTRA_MOUNTS:-/lustre/fsw/portfolios/llmservice:/lustre/fsw/portfolios/llmservice}"
MOUNTS="--container-mounts=${CODE_DIR}:/code,${OUTPUT_PREFIX}:${OUTPUT_PREFIX},${EXTRA_MOUNTS}"

# Reuse only if the averaged model is NEWER than every checkpoint it could be
# built from. Reusing on existence alone silently evaluates stale weights after
# a run has trained further, and the result looks completely normal -- there is
# nothing in the output to say the .nemo predates the checkpoints.
STALE=0
if [[ -f "$AVG_NEMO" ]]; then
    NEWEST_CKPT="$(ls -t "${CKPT_DIR}"/*.ckpt 2>/dev/null | head -1)"
    [[ -n "$NEWEST_CKPT" && "$NEWEST_CKPT" -nt "$AVG_NEMO" ]] && STALE=1
fi
if [[ -f "$AVG_NEMO" && "$STALE" == "0" && "${FORCE_AVERAGE:-0}" != "1" ]]; then
    echo "==> reusing ${AVG_NEMO} (newer than every checkpoint; FORCE_AVERAGE=1 to rebuild)"
else
    [[ "$STALE" == "1" ]] && echo "==> checkpoints are newer than the averaged model; rebuilding"
    srun --ntasks=1 --nodes=1 --container-image="$CONTAINER" $MOUNTS bash -c "
        cd /code && export PYTHONPATH=/code:\$PYTHONPATH HYDRA_FULL_ERROR=1 &&
        python scripts/checkpoint_averaging/average_model_checkpoints.py \
            --config-path=/code/examples/asr/conf/fastconformer/cache_aware_streaming \
            --config-name=${ARM_CONFIG_NAME} \
            name=${AVG_NAME} \
            +model_class=${ARM_MODEL_CLASS:-nemo.collections.asr.models.EncDecCHATBPEModel} \
            +checkpoint_paths=\\\"[${CKPT_CSV}]\\\" \
            model.tokenizer.dir=${TOKENIZER_DIR} \
            ${ARM_MODEL_OVERRIDES:-} \
            ~model.train_ds \
            ~trainer.strategy \
            trainer.devices=1 trainer.accelerator=cpu trainer.precision=32 \
            trainer.num_nodes=1 trainer.logger=false trainer.enable_checkpointing=false
    "
fi

[[ -f "$AVG_NEMO" ]] || { echo "ERROR: averaging produced no ${AVG_NEMO}" >&2; exit 1; }
echo "==> averaged model: ${AVG_NEMO}"

# AVERAGE_ONLY: produce the .nemo and stop. Lets another launcher refresh an
# average WITHOUT also paying for this script's own (non-official) eval, so
# "re-average then score officially" is one job instead of two.
if [[ "${AVERAGE_ONLY:-0}" == "1" ]]; then
    echo "==> AVERAGE_ONLY=1: stopping before the eval"
    exit 0
fi

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
