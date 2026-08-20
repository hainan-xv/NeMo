#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:sslm-lb-promptctl-legacy
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1   # ONE task; fans 8 python procs across the 8 GPUs
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# LEGACY prompt-controlled leaderboard eval -- runs the OLD code + OLD model.
#
# This launcher LIVES in the clean repo (launch/) so it syncs to OCI with it,
# but it deliberately runs NOTHING from the clean repo: it builds the prompt the
# way the *previous* implementation did and then execs the *previous* repo's
# pooled-shard backend (launch/eval_leaderboard_slurm.sh), mounting the previous
# repo's checkout as /code and pointing at the previous model's checkpoints. It
# is a faithful copy of
#   ~/Workplace/NeMo_ord_sync_d146_current/launch/eval_leaderboard_promptctl.sh
# (the recipe that got the BEST high-delay WER with the old code), with only the
# code/backend/model paths redirected at the OLD tree. Use it to A/B the old
# stack against the clean repo on the SAME grid with the SAME cache.
#
# PREREQ: sync the OLD repo to OCI first (its own sync_to_oci.sh), so the old
# code lands at OLD_CODE_DIR and its backend at
# ${OLD_CODE_DIR}/launch/eval_leaderboard_slurm.sh. The old model checkpoints
# must already be on lustre under ${OUTPUT_PREFIX}/results/${PROJECT}/<EXP>/.
#
# Usage (from the clean repo root on OCI):
#   sbatch launch/eval_promptctl_legacy.sh <model> <cap> <punct> <delay>
#
#   <model>  : promptctl | promptctl_d8
#              (also accepts the full exp names granary2_script_promptctl[_d8])
#   <cap>    : cap | nocap     (aliases: true/1/yes , false/0/no)
#   <punct>  : punct | nopunct (aliases: true/1/yes , false/0/no)
#   <delay>  : integer frames; promptctl in [0,4], promptctl_d8 in [0,8]
#
# Examples:
#   sbatch launch/eval_promptctl_legacy.sh promptctl_d8 nocap nopunct 8
#   sbatch launch/eval_promptctl_legacy.sh promptctl    cap   punct   2
#   for d in 1 3 6 8; do sbatch launch/eval_promptctl_legacy.sh promptctl_d8 nocap nopunct $d; done
#
# Where the OLD code / model live (all overridable via env):
#   OLD_CODE_DIR    OCI checkout of the previous repo, mounted as /code
#                   (default matches NeMo_ord_sync_d146_current/sync_to_oci.sh)
#   PROJECT         previous results project dir (default Speechlm79)
#   OUTPUT_PREFIX   results root (default = old backend's own default: nemotron)
#   LEGACY_BACKEND  explicit path to the old eval_leaderboard_slurm.sh
#                   (default ${OLD_CODE_DIR}/launch/eval_leaderboard_slurm.sh)
#
# Optional env (forwarded to the OLD backend -- see its header):
#   CHUNK_SIZE=14 (default)   BACKEND=heh|sslm   BATCH_SIZE=...
#   USE_LAST=1 / STEP=n / CKPT=path / RUN_AVERAGING=0 / MAX_EVAL_SAMPLES=...
#   HEH_USE_STATE_MACHINE / HEH_PAD_DURATION / ...
# ============================================================================
set -euo pipefail

mkdir -p slurm_out

usage() {
    echo "usage: sbatch launch/eval_promptctl_legacy.sh <model> <cap> <punct> <delay>" >&2
    echo "  model : promptctl | promptctl_d8" >&2
    echo "  cap   : cap | nocap" >&2
    echo "  punct : punct | nopunct" >&2
    echo "  delay : integer frames (promptctl: 0-4, promptctl_d8: 0-8)" >&2
    exit 1
}

[[ $# -ge 4 ]] || usage

MODEL_ARG="$1"
CAP_ARG="$2"
PUNCT_ARG="$3"
DELAY_ARG="$4"

# ---- model -> exp name + max delay (matches the OLD promptctl launcher) ----
case "$MODEL_ARG" in
    promptctl|granary2_script_promptctl)
        EXP_NAME=granary2_script_promptctl
        MAX_DELAY=4
        ;;
    promptctl_d8|d8|granary2_script_promptctl_d8)
        EXP_NAME=granary2_script_promptctl_d8
        MAX_DELAY=8
        ;;
    *)
        echo "ERROR: unknown model '$MODEL_ARG' (expected promptctl | promptctl_d8)" >&2
        usage
        ;;
esac

# ---- cap / punct -> bool ----
parse_bool() {
    # $1 = value, $2 = positive name (cap|punct), $3 = negative name (nocap|nopunct)
    local v
    v="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    case "$v" in
        "$2"|true|1|yes|on)  echo 1 ;;
        "$3"|false|0|no|off) echo 0 ;;
        *)
            echo "ERROR: expected '$2' or '$3' (got '$1')" >&2
            exit 1
            ;;
    esac
}
CAP="$(parse_bool "$CAP_ARG" cap nocap)"
PUNCT="$(parse_bool "$PUNCT_ARG" punct nopunct)"

# ---- delay ----
if ! [[ "$DELAY_ARG" =~ ^[0-9]+$ ]]; then
    echo "ERROR: delay must be a non-negative integer (got '$DELAY_ARG')" >&2
    exit 1
fi
DELAY=$((10#$DELAY_ARG))
if (( DELAY < 0 || DELAY > MAX_DELAY )); then
    echo "ERROR: delay=$DELAY out of range for ${EXP_NAME} (allowed 0..${MAX_DELAY})" >&2
    exit 1
fi

# ---- format clause (verbatim from the OLD ScriptSTTDataset._DEFAULT_FORMAT_CLAUSES) ----
if (( CAP )) && (( PUNCT )); then
    FORMAT_CLAUSE="Write the text with normal capitalization and punctuation."
    REPR_TAG=cap_punct
elif (( CAP )) && ! (( PUNCT )); then
    FORMAT_CLAUSE="Write the text with normal capitalization but no punctuation."
    REPR_TAG=cap_nopunct
elif ! (( CAP )) && (( PUNCT )); then
    FORMAT_CLAUSE="Write the text in all lowercase, keeping punctuation."
    REPR_TAG=nocap_punct
else
    FORMAT_CLAUSE="Write the text in all lowercase with no punctuation."
    REPR_TAG=nocap_nopunct
fi

# Prompt template VERBATIM from the OLD promptctl recipe (exact_delay +
# vary_text_repr). NOTE: this is intentionally the OLD wording ("Emit each
# chunk's words with a fixed delay of N frames.") and has NO "...chunks of N
# frames." clause -- it must byte-match what the OLD model saw in training.
SYSTEM_PROMPT="You are doing streaming speech recognition. Given the transcript so far and the next audio chunk, output the words spoken in that chunk. Emit each chunk's words with a fixed delay of ${DELAY} frames. ${FORMAT_CLAUSE}"

MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"
CHUNK_SIZE="${CHUNK_SIZE:-14}"
# Distinguishes concurrent promptctl evals in RESULTS_DIR (shared avg/HF are locked).
EVAL_TAG="d${DELAY}_${REPR_TAG}"

# ---- point /code + results at the OLD tree/model ----
# Where the previous repo is checked out on OCI (mounted as /code by the old
# backend). Default = the OCI_REPO in NeMo_ord_sync_d146_current/sync_to_oci.sh.
OLD_CODE_DIR="${OLD_CODE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/NeMo79}"
# The old backend reads CODE_DIR and mounts ${CODE_DIR}:/code -> old code runs.
export CODE_DIR="$OLD_CODE_DIR"
# The previous model's results project (old backend default is Speechlm79 too;
# set explicitly so it's obvious and overridable).
PROJECT="${PROJECT:-Speechlm79}"

export SYSTEM_PROMPT MODEL_CLASS CHUNK_SIZE EXP_NAME EVAL_TAG PROJECT

echo "==> LEGACY promptctl leaderboard eval (OLD code + OLD model)"
echo "    old_code_dir:  ${OLD_CODE_DIR}  (mounted as /code)"
echo "    project:       ${PROJECT}"
echo "    exp_name:      ${EXP_NAME}"
echo "    eval_tag:      ${EVAL_TAG}"
echo "    cap/punct:     ${REPR_TAG}"
echo "    delay:         ${DELAY} frames (max trained ${MAX_DELAY})"
echo "    chunk_size:    ${CHUNK_SIZE}"
echo "    system_prompt: ${SYSTEM_PROMPT}"

# Locate the OLD backend. Unlike the clean-repo launchers (which resolve a
# sibling backend relative to the submit dir), this one MUST run the previous
# repo's backend, so we resolve it under OLD_CODE_DIR. LEGACY_BACKEND overrides.
LEGACY_BACKEND="${LEGACY_BACKEND:-}"
if [[ -z "$LEGACY_BACKEND" ]]; then
    if [[ -f "${OLD_CODE_DIR}/launch/eval_leaderboard_slurm.sh" ]]; then
        LEGACY_BACKEND="${OLD_CODE_DIR}/launch/eval_leaderboard_slurm.sh"
    elif [[ -f "${OLD_CODE_DIR}/oci/eval_leaderboard_slurm.sh" ]]; then
        LEGACY_BACKEND="${OLD_CODE_DIR}/oci/eval_leaderboard_slurm.sh"
    fi
fi
if [[ -z "$LEGACY_BACKEND" || ! -f "$LEGACY_BACKEND" ]]; then
    echo "ERROR: cannot find the OLD backend eval_leaderboard_slurm.sh under ${OLD_CODE_DIR}." >&2
    echo "       Sync the previous repo to OCI first (its sync_to_oci.sh), or set" >&2
    echo "       OLD_CODE_DIR=/path/to/old/checkout (or LEGACY_BACKEND=/full/path)." >&2
    exit 1
fi
echo "    legacy_backend: ${LEGACY_BACKEND}"

# Keep cwd at the submit dir so relative paths (slurm_out/) land where expected.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Already inside the sbatch allocation: run the OLD pooled-shard backend as a
# normal bash script (its own #SBATCH headers are ignored). $1 = EXP_NAME.
exec bash "$LEGACY_BACKEND" "${EXP_NAME}"
