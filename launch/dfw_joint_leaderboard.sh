#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-joint-lb
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# Full-leaderboard chunk-synchronous joint decoding, at three mixing weights.
#
#   sbatch launch/dfw_joint_leaderboard.sh
#   LAM_LIST="0.0 0.5 1.0" sbatch launch/dfw_joint_leaderboard.sh
#
# Runs the SAME production backend (eval_leaderboard.sh -> script_leaderboard_eval.py)
# the SCRIPT arms are scored with, only with --chat_nemo set, so the fusion runs
# sharded across 8 GPUs with the normal batching, OOM halving, aggregation and
# scorer. The numbers are therefore directly comparable to every other row in
# the results table.
#
# WHY THREE WEIGHTS IN ONE JOB, sequentially: lam=0 and lam=1 are the CONTROLS.
#   lam=0.0  SCRIPT alone -- must land near its own leaderboard number (5.94)
#   lam=1.0  CHAT alone   -- must land near its own (5.52)
#   lam=0.5  the actual experiment
# If either endpoint misses, the harness is wrong and the middle value means
# nothing. Running all three in ONE allocation means they share a node, a cache
# and a scorer, so a discrepancy cannot be blamed on environment drift between
# submissions.
#
# NOTE lam=1.0 is CHAT's distribution driving SCRIPT's decode loop, NOT the CHAT
# model's own RNN-T decoder. It should land CLOSE to 5.52 but need not match it
# exactly: the emission grid, the per-chunk token cap and the end-of-chunk
# handling are SCRIPT's. Treat a small gap as expected and a large one as a bug.
#
# ENV
#   LAM_LIST     space-separated weights (default "0.0 0.5 1.0")
#   CHAT_ARM / SCRIPT_ARM   which v2 arms to fuse
#   DATASETS     override the split list (default: all seven)
# ============================================================================
set -uo pipefail
mkdir -p slurm_out

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-${DFW}/hainanx}"
export CODE_DIR="${CODE_DIR:-${OUTPUT_PREFIX}/NeMo_SCRIPT_cc}"
export CONTAINER="${CONTAINER:-${DFW}/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
export CACHE_DIR="${CACHE_DIR:-${OUTPUT_PREFIX}/leaderboard_cache}"
export H_DIR="${H_DIR:-${DFW}/users/heh}"
export PROJECT="${PROJECT:-SpeechlmDFW}"

CHAT_ARM="${CHAT_ARM:-dfw_granary2_chat_banded1_nodelay_v2}"
SCRIPT_ARM="${SCRIPT_ARM:-dfw_granary2_script_banded1_nodelay_v2}"
LAM_LIST="${LAM_LIST:-0.0 0.5 1.0}"
# ON-DEMAND fusion: a chunk whose weakest CHAT margin clears this never invokes
# SCRIPT at all. inf = always fuse. Measured on 200 utts of test.other: 2.0
# skipped 46.5% of chunk-decodes for a 0.01 WER cost.
SKIP_THRESHOLD="${SKIP_THRESHOLD:-inf}"

CHAT_NEMO="${OUTPUT_PREFIX}/results/${PROJECT}/${CHAT_ARM}/averaged/top5-averaged.nemo"
if [[ ! -s "$CHAT_NEMO" ]]; then
    echo "ERROR: CHAT .nemo not found: ${CHAT_NEMO}" >&2
    echo "       It is produced by a CHAT leaderboard eval; run one first." >&2
    exit 1
fi
if [[ ! -d "${CACHE_DIR}" ]] || [[ -z "$(ls -A "${CACHE_DIR}" 2>/dev/null)" ]]; then
    echo "ERROR: leaderboard cache is empty or missing at ${CACHE_DIR}" >&2
    exit 1
fi

export EXP_NAME="${SCRIPT_ARM}"
export CHUNK_SIZE="${CHUNK_SIZE:-14}"
export MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"
export SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are doing streaming speech recognition. Given the transcript so far and the representation of the next audio chunk, output the words spoken in that chunk.}"

resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_leaderboard.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_leaderboard.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_leaderboard.sh" ]] && { echo "${here}"; return; }
    [[ -f "${CODE_DIR}/launch/eval_leaderboard.sh" ]] && { echo "${CODE_DIR}/launch"; return; }
    echo "ERROR: cannot locate eval_leaderboard.sh" >&2; exit 1
}
LAUNCH_DIR="$(resolve_launch_dir)"

echo "############################################################"
echo "### joint leaderboard: CHAT=${CHAT_ARM}"
echo "###                    SCRIPT=${SCRIPT_ARM}"
echo "###                    weights: ${LAM_LIST}"
date
echo "############################################################"

declare -a STATUS=()
for lam in ${LAM_LIST}; do
    echo
    echo "============================================================"
    echo "=== lam=${lam}$( [[ "$lam" == "0.0" ]] && echo '   (control: SCRIPT alone)'; [[ "$lam" == "1.0" ]] && echo '   (control: CHAT alone)' )"
    echo "============================================================"
    # lam=0 must NOT load the CHAT model at all: that is the true SCRIPT-alone
    # control, and fusing with weight zero would still route through the fusion
    # code, so a bug there could not be detected by it.
    if [[ "$lam" == "0.0" ]]; then
        EXTRA=""
    else
        EXTRA="--chat_nemo ${CHAT_NEMO} --fusion_lam ${lam} --fusion_skip_threshold ${SKIP_THRESHOLD:-inf}"
    fi
    # RESULTS_SUFFIX, not EVAL_TAG: the results path is keyed on the checkpoint
    # mtime and the decode label, neither of which varies across weights, so
    # without a suffix all three runs land in one directory and overwrite each
    # other's logs.
    _sfx="lam${lam}"; [[ "${SKIP_THRESHOLD}" != "inf" ]] && _sfx="${_sfx}_skip${SKIP_THRESHOLD}"
    EVAL_TAG="joint_${_sfx}" RESULTS_SUFFIX="${_sfx}" EXTRA_EVAL_ARGS="${EXTRA}" \
        bash "${LAUNCH_DIR}/eval_leaderboard.sh"
    rc=$?
    if [[ $rc -eq 0 ]]; then STATUS+=("lam=${lam}|ok"); else
        echo "    FAILED (exit ${rc}); continuing" >&2
        STATUS+=("lam=${lam}|FAILED (exit ${rc})")
    fi
done

echo
echo "############################################################"
echo "### summary"
date
echo "############################################################"
for s in "${STATUS[@]}"; do printf '  %-14s %s\n' "${s%%|*}" "${s#*|}"; done
echo
echo "### macro WER per weight"
for lam in ${LAM_LIST}; do
    _sfx="lam${lam}"; [[ "${SKIP_THRESHOLD}" != "inf" ]] && _sfx="${_sfx}_skip${SKIP_THRESHOLD}"
    L="$(ls -t "${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}"/eval_*/*_${_sfx}/aggregate.log 2>/dev/null | head -1)"
    if [[ -n "$L" ]]; then
        printf '  lam=%-6s %s\n' "$lam" "$(awk -F'\t' '$1=="RESULT" && $2=="Average"{v=$3} END{print v}' "$L")"
    else
        printf '  lam=%-6s %s\n' "$lam" "(no aggregate.log)"
    fi
done
