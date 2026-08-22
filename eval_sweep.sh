#!/bin/bash
# ============================================================================
# Submit the leaderboard eval for SCRIPT and nemotron across several chunk sizes.
#
# One job per (system, chunk_size). All jobs share the staged dataset cache, the
# shard partition seed and the scorer, so the resulting WERs are directly
# comparable across both systems and all latencies.
#
# USAGE
#   ./eval_sweep.sh                          # both systems, chunks 2 7 14
#   ./eval_sweep.sh --chunks "2 7"           # subset of chunk sizes
#   ./eval_sweep.sh --only script            # one system
#   ./eval_sweep.sh --only nemotron
#   ./eval_sweep.sh --dry-run                # print submissions, send nothing
#   ./eval_sweep.sh MAX_EVAL_SAMPLES=32      # quick smoke sweep (env passthrough)
#   ./eval_sweep.sh --exp granary2_script_baseline_n1   # a 1-node training run
#
# CHUNK SIZES
#   Defaults to "2 7 14" -- the sizes BOTH models were trained for, so every
#   point is a like-for-like comparison. SCRIPT also supports 4, 10 and 28, but
#   nemotron does not (its trained look-aheads are chunk 14/7/2/1), and its eval
#   driver refuses anything else rather than silently degrading. Chunk size is in
#   encoder frames of 0.08s, so 2 -> 0.16s, 7 -> 0.56s, 14 -> 1.12s.
#
# SYNC
#   The first submission runs the usual sync check (and syncs if the grid is
#   stale); the rest are launched with --no-sync so one sweep does not re-check
#   the grid once per job.
# ============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

CHUNKS="${CHUNKS:-2 7 14}"
ONLY="${ONLY:-both}"                 # script | nemotron | both
SCRIPT_EXP="${SCRIPT_EXP:-granary2_script_baseline}"
NEMOTRON_MODE="${MODE:-offline}"     # offline | streaming
DRY_RUN=0

ENV_ASSIGNMENTS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --chunks)     CHUNKS="$2"; shift 2 ;;
        --only)       ONLY="$2"; shift 2 ;;
        --exp)        SCRIPT_EXP="$2"; shift 2 ;;
        --mode)       NEMOTRON_MODE="$2"; shift 2 ;;
        -n|--dry-run) DRY_RUN=1; shift ;;
        -h|--help)    sed -n '2,/^# ===*$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        # Anything of the form VAR=VALUE is forwarded to every job.
        [A-Za-z_]*=*) ENV_ASSIGNMENTS+=("$1"); shift ;;
        *)            echo "ERROR: unexpected argument: $1" >&2; exit 1 ;;
    esac
done

case "$ONLY" in
    script|nemotron|both) ;;
    *) echo "ERROR: --only must be script, nemotron or both (got '$ONLY')" >&2; exit 1 ;;
esac

DRY_FLAG=""
[[ "$DRY_RUN" -eq 1 ]] && DRY_FLAG="--dry-run"

echo "==> eval sweep"
echo "    chunk sizes: ${CHUNKS}"
echo "    systems:     ${ONLY}"
echo "    script exp:  ${SCRIPT_EXP}"
echo "    nemotron:    mode=${NEMOTRON_MODE}"
[[ ${#ENV_ASSIGNMENTS[@]} -gt 0 ]] && echo "    env:         ${ENV_ASSIGNMENTS[*]}"
echo ""

submitted=()
first=1

launch() {
    # $1 = human label, rest = args for oci_launch.sh
    local label="$1"; shift
    local sync_flag="--no-sync"
    if [[ "$first" -eq 1 ]]; then
        sync_flag=""       # let the first submission do the sync check
        first=0
    fi

    echo "--- ${label} ---"
    local out
    if ! out="$(./oci_launch.sh $sync_flag $DRY_FLAG "$@" 2>&1)"; then
        echo "$out"
        echo "    !! ${label} FAILED to submit" >&2
        submitted+=("${label}: FAILED")
        return
    fi
    echo "$out" | grep -E '^(==>|    )' || true
    local jid
    jid="$(echo "$out" | sed -n 's/^==> Job \([0-9]*\) submitted.*/\1/p' | tail -1)"
    submitted+=("${label}: ${jid:-dry-run}")
    echo ""
}

for c in $CHUNKS; do
    if [[ "$ONLY" == "script" || "$ONLY" == "both" ]]; then
        launch "SCRIPT   chunk=${c}" ${ENV_ASSIGNMENTS[@]+"${ENV_ASSIGNMENTS[@]}"} \
            launch/eval_script_baseline.sh "$SCRIPT_EXP" "$c"
    fi
    if [[ "$ONLY" == "nemotron" || "$ONLY" == "both" ]]; then
        launch "nemotron chunk=${c}" "MODE=${NEMOTRON_MODE}" ${ENV_ASSIGNMENTS[@]+"${ENV_ASSIGNMENTS[@]}"} \
            launch/eval_nemotron.sh "$c"
    fi
done

echo "===================== submitted ====================="
printf '  %s\n' "${submitted[@]}"
echo ""
if [[ "$DRY_RUN" -eq 0 ]]; then
    echo "Watch:   ssh ${OCI_USER:-hainanx}@\${OCI_HOST} squeue -u \$USER"
    echo "Results: <OUTPUT_PREFIX>/results/<PROJECT>/<exp>/leaderboard_eval_*/aggregate.log"
fi
