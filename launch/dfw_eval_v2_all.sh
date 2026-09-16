#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-eval-v2-all
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
# pool0-01815 carries an old NVIDIA driver (12020) and fails torch's CUDA init.
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# Open-ASR-Leaderboard eval of ALL FOUR v2 arms, in one interactive job.
#
#   sbatch launch/dfw_eval_v2_all.sh                      <- all four
#   ARMS="chat_later script_both" sbatch launch/dfw_eval_v2_all.sh   <- a subset
#
# The four arms span BOTH model families and BOTH band directions, and the two
# families need different backends -- CHAT averages checkpoints into a .nemo and
# goes through eval_chat.sh, SCRIPT averages in-place and goes through
# eval_leaderboard.sh. This script is the single entry point over both, so one
# submission produces the whole comparison table on one node, one cache and one
# scorer. That shared-pipeline property is the point: it is what makes the
# numbers comparable to each other and to everything already collected.
#
# THE v2 ARMS ALL TRAIN AT delay=0, WHICH CHANGES HOW THEY MUST DECODE.
# EncDecCHATBPEModel derives joint.frame_trim from num_delay_frames, so these
# arms decode at trim 0 -- NOT the trim 3 that dfw_eval_chat.sh uses for the
# delay=3 forced arm. Evaluating them at trim 3 would measure them at an
# operating point they never trained at, which this project has already been
# burned by once.
#
# Arms run SEQUENTIALLY and INDEPENDENTLY: one failure is reported and the rest
# still run, and an arm with no checkpoints is skipped with a message rather
# than aborting the job. Each arm runs in a SUBSHELL so the env a backend needs
# cannot leak into the next arm.
#
# RUNTIME ~1h for all four (measured: SCRIPT ~12 min, CHAT ~9-16 min each),
# comfortably inside the 4h interactive limit.
#
# ENV
#   ARMS           subset of {chat_later, chat_both, script_later, script_both}
#   TOPK           checkpoints to average per arm (default 5)
#   CHUNK_SIZE     SCRIPT decode chunk size (default 14 -- what they trained on)
#   FORCE_AVERAGE  1 to rebuild an averaged model that already exists
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
# eval_chat.sh defaults this to an OCI-only path; on DFW that hides the Qwen
# tokenizer inside the container and transformers then reads the directory as a
# hub repo id ("Repo id must be in the form 'repo_name'").
export EXTRA_MOUNTS="${EXTRA_MOUNTS:-${DFW}:${DFW}}"

TOPK="${TOPK:-5}"
CHUNK_SIZE="${CHUNK_SIZE:-14}"
QWEN_TOK="${DFW}/users/heh/pretrained_models/huggingface/Qwen/Qwen3-1.7B"

if [[ ! -d "${CACHE_DIR}" ]] || [[ -z "$(ls -A "${CACHE_DIR}" 2>/dev/null)" ]]; then
    echo "ERROR: leaderboard cache is empty or missing at ${CACHE_DIR}" >&2
    echo "       Stage it first:  sbatch launch/dfw_stage_leaderboard_cache.sh" >&2
    exit 1
fi

# Shared CHAT overrides. num_delay_frames=0 is the v2 change; everything else
# matches how these arms were trained.
FA=model.forced_alignment
CHAT_COMMON="${FA}.num_delay_frames=0 ${FA}.max_delay_frames=0 model.joint.history_chunks=0"
CHAT_COMMON="${CHAT_COMMON} ${FA}.target_construction=partition ${FA}.delay_word_final_punctuation=true"
CHAT_COMMON="${CHAT_COMMON} ${FA}.band_chunks=1 ${FA}.recover_history_words=0 model.loss_type=banded"

# key | family | exp_name | band_side
ALL_ARMS=(
  "chat_later|chat|dfw_granary2_chat_banded1_nodelay_v2|later"
  "chat_both|chat|dfw_granary2_chat_banded1_both_nodelay_v2|both"
  "script_later|script|dfw_granary2_script_banded1_nodelay_v2|later"
  "script_both|script|dfw_granary2_script_banded1_both_nodelay_v2|both"
)

WANTED="${ARMS:-}"

resolve_launch_dir() {
    local probe="$1"
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/${probe}" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/${probe}" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/${probe}" ]] && { echo "${here}"; return; }
    # Absolute fallback: SLURM_SUBMIT_DIR comes back as the scratch root on a
    # requeue, which silently broke the CHAT arms for three hours.
    [[ -f "${CODE_DIR}/launch/${probe}" ]] && { echo "${CODE_DIR}/launch"; return; }
    echo "ERROR: cannot locate ${probe}" >&2
    return 1
}
LAUNCH_DIR="$(resolve_launch_dir eval_chat.sh)" || exit 1

echo "############################################################"
echo "### DFW v2 leaderboard eval -- all four arms"
echo "###   topk=${TOPK}  chunk_size=${CHUNK_SIZE}  cache=${CACHE_DIR}"
date
echo "############################################################"

# Marker file stamped at job start. The summary below only accepts an
# aggregate.log NEWER than this, because eval_*/ directories accumulate: an arm
# that FAILS this run still has last run's aggregate.log on disk, and reading
# "the newest aggregate.log" then reports STALE numbers under a fresh timestamp.
# That happened -- a failed arm was reported with a full result table identical
# to the previous run's.
JOB_START_MARKER="$(mktemp)"

declare -a STATUS=()

for entry in "${ALL_ARMS[@]}"; do
    IFS='|' read -r key family exp side <<< "$entry"
    if [[ -n "$WANTED" ]] && [[ " $WANTED " != *" $key "* ]]; then
        continue
    fi

    CKPTS="${OUTPUT_PREFIX}/results/${PROJECT}/${exp}/${exp}/checkpoints"
    echo
    echo "============================================================"
    echo "=== ${key}   (${family}, band_side=${side}, delay=0)"
    echo "===   ${exp}"
    if [[ -d "$CKPTS" ]]; then
        NCK="$(ls -1 "$CKPTS"/*.ckpt 2>/dev/null | grep -vc -- '-last')"
        BEST="$(ls -1 "$CKPTS"/*.ckpt 2>/dev/null | grep -oE 'val_wer=[0-9.]+' | sort -t= -k2 -g | head -1)"
        echo "===   ${NCK:-0} checkpoints, best ${BEST:-val_wer=?}"
    fi
    echo "============================================================"

    if [[ ! -d "$CKPTS" ]] || [[ -z "$(ls -A "$CKPTS"/*.ckpt 2>/dev/null)" ]]; then
        echo "    SKIPPED: no checkpoints at ${CKPTS}" >&2
        STATUS+=("${key}|skipped (no checkpoints)")
        continue
    fi

    # Subshell per arm: the two backends want different variables, and a value
    # left over from one arm would silently mis-decode the next.
    (
        if [[ "$family" == "chat" ]]; then
            ARM_EXP_NAME="${exp}" \
            ARM_CONFIG_NAME="nemotron_chat_transducer_granary2_qwen" \
            ARM_TOKENIZER_DIR="${QWEN_TOK}" \
            ARM_MODEL_OVERRIDES="${CHAT_COMMON} ${FA}.band_side=${side}" \
            TOPK="${TOPK}" FORCE_AVERAGE="${FORCE_AVERAGE:-0}" \
            EVAL_TAG="avg${TOPK}" \
            FRAME_TRIM="" \
                bash "${LAUNCH_DIR}/eval_chat.sh"
        else
            EXP_NAME="${exp}" \
            CHUNK_SIZE="${CHUNK_SIZE}" \
            MODEL_CLASS="nemo.collections.speechlm2.models.script_model.ScriptSTTModel" \
            SYSTEM_PROMPT="You are doing streaming speech recognition. Given the transcript so far and the representation of the next audio chunk, output the words spoken in that chunk." \
            EVAL_TAG="${exp}" \
                bash "${LAUNCH_DIR}/eval_leaderboard.sh"
        fi
    )
    rc=$?
    if [[ $rc -eq 0 ]]; then
        STATUS+=("${key}|ok")
    else
        echo "    FAILED (exit ${rc}); continuing with the next arm" >&2
        STATUS+=("${key}|FAILED (exit ${rc})")
    fi
done

echo
echo "############################################################"
echo "### summary"
date
echo "############################################################"
for s in "${STATUS[@]}"; do printf '  %-16s %s\n' "${s%%|*}" "${s#*|}"; done
echo
echo "### macro WER per arm"
for entry in "${ALL_ARMS[@]}"; do
    IFS='|' read -r key family exp side <<< "$entry"
    [[ -n "$WANTED" ]] && [[ " $WANTED " != *" $key "* ]] && continue
    L="$(ls -t "${OUTPUT_PREFIX}/results/${PROJECT}/${exp}"/eval_*/*/aggregate.log 2>/dev/null | head -1)"
    if [[ -z "$L" ]]; then
        printf '  %-16s %s\n' "$key" "(never evaluated)"
    elif [[ ! "$L" -nt "$JOB_START_MARKER" ]]; then
        # Older than this job => it is the PREVIOUS run's result, not ours.
        printf '  %-16s %s\n' "$key" "NO RESULT THIS RUN (stale log from $(date -r "$L" '+%m-%d %H:%M') ignored)"
    else
        # awk, not grep -E: \t is not portable in an ERE pattern.
        printf '  %-16s %s\n' "$key" "$(awk -F'\t' '$1=="RESULT" && $2=="Average"{v=$3} END{print v}' "$L")"
    fi
done
rm -f "$JOB_START_MARKER"
