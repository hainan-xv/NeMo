#!/bin/bash
# ============================================================================
# Re-evaluate every model in the project that has trained since its last eval.
#
# Scans the results tree, and for each experiment compares the newest TRAINING
# checkpoint against the newest COMPLETED eval. If training has moved on, it
# regenerates the averaged checkpoint (FORCE_AVERAGE=1, since the average is
# cached and would otherwise be stale) and submits a fresh eval. If not, it does
# nothing. One independent Slurm job per experiment, so they all run in parallel.
#
# Safe to run on a cron or after every training day: experiments that have not
# progressed cost nothing.
#
# USAGE
#   ./eval_all.sh                       # evaluate whatever is stale, chunk 14
#   ./eval_all.sh --dry-run             # show the decisions, submit nothing
#   ./eval_all.sh --chunks "2 7 14"     # one job per (model, chunk size)
#   ./eval_all.sh --force               # re-evaluate everything regardless
#   ./eval_all.sh --only 'win28'        # only experiments matching this regex
#   ./eval_all.sh --skip 'promptctl'    # additionally skip these
#   ./eval_all.sh MAX_EVAL_SAMPLES=32   # env passthrough to every job
#
# WHAT COUNTS AS "NEWER"
#   Newest mtime among the experiment's *.ckpt files, EXCLUDING -last.ckpt and
#   -averaged.ckpt -- exactly the set eval_leaderboard.sh averages over. Compared
#   against the mtime of the newest aggregate.log that actually finished (one
#   containing a RESULT/Average row). A crashed or half-written eval therefore
#   does not count as up to date, and the rolling -last.ckpt (rewritten every few
#   minutes during training) does not make a model look perpetually stale.
#
# WHAT IS SKIPPED, AND WHY
#   * Directories with no checkpoints/ dir -- e.g. nemotron_streaming_0.6b, which
#     is a restored .nemo baseline with its own driver (launch/eval_nemotron.sh),
#     not a SCRIPT training run.
#   * Experiments matching SKIP_PATTERN, default '_n1$'. The interactive
#     launcher appends _n1 to mark short single-node debug runs; evaluating a
#     50-step smoke test burns a GPU-hour for nothing. Pass --skip '' to include
#     them.
#   * Experiments whose previous eval is still running -- detected as an eval
#     directory with no finished aggregate.log, created less than
#     INFLIGHT_MAX_AGE ago (6h). This makes the script safe to run repeatedly
#     without piling up duplicate jobs. Past that age an unfinished eval is
#     assumed to have crashed, so it never blocks an experiment permanently.
#   Every skip is printed with its reason -- nothing is dropped silently.
# ============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ./oci_env.sh

PROJECT="${PROJECT:-SpeechlmScriptCC}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
RESULTS_ROOT="${OUTPUT_PREFIX}/results/${PROJECT}"

CHUNKS="${CHUNKS:-14}"
ONLY_PATTERN="${ONLY_PATTERN:-}"
SKIP_PATTERN="${SKIP_PATTERN-_n1$}"
FORCE=0
DRY_RUN=0
# An unfinished eval dir older than this is assumed dead rather than running, so
# one crashed job cannot block an experiment from ever being re-evaluated.
INFLIGHT_MAX_AGE="${INFLIGHT_MAX_AGE:-21600}"   # 6h

ENV_ASSIGNMENTS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --chunks)     CHUNKS="$2"; shift 2 ;;
        --only)       ONLY_PATTERN="$2"; shift 2 ;;
        --skip)       SKIP_PATTERN="$2"; shift 2 ;;
        -f|--force)   FORCE=1; shift ;;
        -n|--dry-run) DRY_RUN=1; shift ;;
        -h|--help)    sed -n '2,/^# ===*$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        [A-Za-z_]*=*) ENV_ASSIGNMENTS+=("$1"); shift ;;
        *)            echo "ERROR: unexpected argument: $1" >&2; exit 1 ;;
    esac
done

echo "==> eval-all scan"
echo "    project:     ${PROJECT}"
echo "    results:     ${RESULTS_ROOT}"
echo "    chunk sizes: ${CHUNKS}"
[[ -n "$ONLY_PATTERN" ]] && echo "    only:        /${ONLY_PATTERN}/"
[[ -n "$SKIP_PATTERN" ]] && echo "    skip:        /${SKIP_PATTERN}/"
[[ "$FORCE" -eq 1 ]] && echo "    force:       re-evaluating everything"
echo ""

# ---------------------------------------------------------------------------
# One round trip: gather the state of every experiment, PER CHUNK SIZE, as TSV.
#   exp <TAB> n_ckpts <TAB> ckpt_epoch <TAB> ckpt_name <TAB> chunk <TAB> eval_epoch <TAB> eval_dir <TAB> pending_epoch
#
# Per chunk size, not per experiment: each chunk size is its own result. An
# experiment freshly evaluated at chunk 14 has NOT been evaluated at chunk 2, and
# comparing against "the newest eval of any chunk size" would call it up to date
# and silently skip the chunk sizes you asked for.
#
# Rows with chunk "-" carry an experiment-level skip (no checkpoints at all).
# ---------------------------------------------------------------------------
SCAN=$(oci_ssh bash -s -- "$RESULTS_ROOT" $CHUNKS <<'REMOTE'
set -u
ROOT="$1"; shift
CHUNK_LIST="$*"
[ -d "$ROOT" ] || { echo "__NOROOT__"; exit 0; }
for d in "$ROOT"/*/; do
    exp=$(basename "$d")
    ck="${d}${exp}/checkpoints"
    if [ ! -d "$ck" ]; then
        printf '%s\t-1\t0\t-\t-\t0\t-\t0\n' "$exp"
        continue
    fi
    # The set eval_leaderboard.sh averages over: no rolling -last, no prior average.
    mapfile -t cks < <(ls -t "$ck"/*.ckpt 2>/dev/null | grep -v -- '-last\.ckpt$' | grep -v -- '-averaged\.ckpt$')
    n=${#cks[@]}
    if [ "$n" -eq 0 ]; then
        printf '%s\t0\t0\t-\t-\t0\t-\t0\n' "$exp"
        continue
    fi
    cepoch=$(stat -c %Y "${cks[0]}" 2>/dev/null || echo 0)
    cname=$(basename "${cks[0]}")

    for c in $CHUNK_LIST; do
        # Only this chunk size's eval dirs. The trailing underscore in the glob
        # keeps chunk 2 from matching chunk 28's directories.
        eepoch=0; edir="-"; pepoch=0
        for ed in "$d"leaderboard_eval_chunk"${c}"_*/; do
            [ -d "$ed" ] || continue
            t=$(stat -c %Y "$ed" 2>/dev/null || echo 0)
            if [ -f "${ed}aggregate.log" ] && grep -q "^RESULT.Average" "${ed}aggregate.log" 2>/dev/null; then
                at=$(stat -c %Y "${ed}aggregate.log" 2>/dev/null || echo 0)
                if [ "$at" -gt "$eepoch" ]; then eepoch=$at; edir=$(basename "$ed"); fi
            else
                if [ "$t" -gt "$pepoch" ]; then pepoch=$t; fi
            fi
        done
        printf '%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n' "$exp" "$n" "$cepoch" "$cname" "$c" "$eepoch" "$edir" "$pepoch"
    done
done
REMOTE
)

if [[ "$SCAN" == *__NOROOT__* ]]; then
    echo "ERROR: results root not found on the grid: ${RESULTS_ROOT}" >&2
    exit 1
fi
if [[ -z "${SCAN//[[:space:]]/}" ]]; then
    echo "No experiments found under ${RESULTS_ROOT}." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Decide
# ---------------------------------------------------------------------------
ago() {  # epoch -> "3h ago" / "never"
    local t="$1"
    [[ "$t" == "0" ]] && { echo "never"; return; }
    local s=$(( $(date +%s) - t ))
    if   [[ $s -lt 3600   ]]; then echo "$((s/60))m ago"
    elif [[ $s -lt 86400  ]]; then echo "$((s/3600))h ago"
    else                           echo "$((s/86400))d ago"; fi
}

TO_EVAL=()   # "exp<TAB>chunk" pairs
printf "  %-34s %5s %6s %13s %13s  %s\n" "experiment" "chunk" "ckpts" "newest ckpt" "last eval" "decision"
printf "  %s\n" "$(printf '%.0s-' {1..108})"

now=$(date +%s)
while IFS=$'\t' read -r exp n cepoch cname chunk eepoch edir pepoch; do
    [[ -z "$exp" ]] && continue
    # An eval directory with no finished aggregate.log, created recently, means a
    # job is probably still running. Bounded by INFLIGHT_MAX_AGE so a CRASHED eval
    # cannot block this experiment forever -- after that it is treated as dead.
    inflight=0
    if [[ "${pepoch:-0}" != "0" ]] && (( now - pepoch < INFLIGHT_MAX_AGE )); then inflight=1; fi

    decision=""
    if [[ "$n" == "-1" ]]; then
        decision="skip: no checkpoints/ dir (not a SCRIPT training run)"
    elif [[ "$n" == "0" ]]; then
        decision="skip: no averageable checkpoints yet"
    elif [[ -n "$ONLY_PATTERN" ]] && ! [[ "$exp" =~ $ONLY_PATTERN ]]; then
        decision="skip: does not match --only"
    elif [[ -n "$SKIP_PATTERN" ]] && [[ "$exp" =~ $SKIP_PATTERN ]]; then
        decision="skip: matches --skip"
    elif [[ "$inflight" -eq 1 && "$FORCE" -eq 0 ]]; then
        decision="skip: a chunk-${chunk} eval started $(ago "$pepoch") has not finished (--force)"
    elif [[ "$FORCE" -eq 1 ]]; then
        decision="EVAL (forced)"
        TO_EVAL+=("${exp}"$'\t'"${chunk}")
    elif [[ "$eepoch" == "0" ]]; then
        decision="EVAL: never evaluated at chunk ${chunk}"
        TO_EVAL+=("${exp}"$'\t'"${chunk}")
    elif [[ "$cepoch" -gt "$eepoch" ]]; then
        decision="EVAL: new checkpoints since the chunk-${chunk} eval"
        TO_EVAL+=("${exp}"$'\t'"${chunk}")
    else
        decision="up to date"
    fi
    printf "  %-34s %5s %6s %13s %13s  %s\n" "$exp" "$chunk" \
        "$([[ "$n" == "-1" ]] && echo "-" || echo "$n")" "$(ago "$cepoch")" "$(ago "$eepoch")" "$decision"
done <<< "$SCAN"

echo ""
if [[ ${#TO_EVAL[@]} -eq 0 ]]; then
    echo "Nothing to do — every experiment is up to date at every requested chunk size."
    exit 0
fi

echo "==> ${#TO_EVAL[@]} job(s) to submit"
echo ""

# ---------------------------------------------------------------------------
# Submit — one job per (experiment, chunk size)
# ---------------------------------------------------------------------------
DRY_FLAG=""; [[ "$DRY_RUN" -eq 1 ]] && DRY_FLAG="--dry-run"

submitted=()
first=1
for pair in "${TO_EVAL[@]}"; do
    IFS=$'\t' read -r exp c <<< "$pair"
    sync_flag="--no-sync"
    if [[ "$first" -eq 1 ]]; then sync_flag=""; first=0; fi   # sync-check once

    label="${exp} chunk=${c}"
    echo "--- ${label} ---"
    # FORCE_AVERAGE=1: the averaged checkpoint is cached, so without this the
    # eval would silently score the OLD average and the whole run would be
    # pointless. This is the "regenerate the averaged checkpoint" step.
    if ! out="$(./oci_launch.sh $sync_flag $DRY_FLAG \
                    FORCE_AVERAGE=1 ${ENV_ASSIGNMENTS[@]+"${ENV_ASSIGNMENTS[@]}"} \
                    launch/eval_script.sh "$exp" "$c" 2>&1)"; then
        echo "$out"
        echo "    !! FAILED to submit" >&2
        submitted+=("${label}: FAILED")
        continue
    fi
    echo "$out" | grep -E '^(==>|    )' || true
    jid="$(sed -n 's/^==> Job \([0-9]*\) submitted.*/\1/p' <<< "$out" | tail -1)"
    submitted+=("${label}: ${jid:-dry-run}")
    echo ""
done

echo "===================== submitted ====================="
printf '  %s\n' "${submitted[@]}"
echo ""
if [[ "$DRY_RUN" -eq 0 ]]; then
    echo "Watch:   ssh ${OCI_USER}@${OCI_HOST} squeue -u \$USER"
    echo "Results: ${RESULTS_ROOT}/<exp>/leaderboard_eval_*/aggregate.log"
fi
