#!/bin/bash
# ============================================================================
# Evaluate the read/write SCRIPT variants against the baseline they came from.
#
# The two warm-started runs differ in exactly one thing, so this is a clean A/B:
#
#   granary2_script_readwrite_ft    read/write gate, BRANCH ONLY
#   granary2_script_rw_history_ft   read/write gate, ALSO IN THE HISTORY
#
# Both were initialized from granary2_script_baseline, which is therefore the
# reference point: it is included by default so "did the gate help?" is a
# same-table comparison rather than a cross-run recollection.
#
# USAGE
#   ./eval_readwrite.sh                    # submit evals, chunks 2 7 14
#   ./eval_readwrite.sh --chunks 14        # one chunk size
#   ./eval_readwrite.sh --dry-run          # show what would be submitted
#   ./eval_readwrite.sh --no-baseline      # just the two variants
#   ./eval_readwrite.sh --report           # collect finished results into a table
#
# SUBMISSION is delegated to ./eval_all.sh, which already owns the parts that are
# easy to get wrong: per-chunk staleness (an experiment evaluated at chunk 14 is
# NOT evaluated at chunk 2), regenerating the cached averaged checkpoint, and
# refusing to pile up duplicates while an eval is still in flight. This script
# only decides WHICH experiments to look at.
#
# --report reads the newest COMPLETED aggregate.log per (experiment, chunk) and
# prints a comparison table. Runs still training will show fewer chunk sizes;
# that is reported, never silently filled in.
# ============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ./oci_env.sh

PROJECT="${PROJECT:-SpeechlmScriptCC}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
RESULTS_ROOT="${OUTPUT_PREFIX}/results/${PROJECT}"

VARIANTS="${VARIANTS:-granary2_script_readwrite_ft granary2_script_rw_history_ft}"
BASELINE="${BASELINE:-granary2_script_baseline}"
CHUNKS="${CHUNKS:-2 7 14}"
WITH_BASELINE=1
REPORT=0
PASSTHRU=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chunks)      CHUNKS="$2"; shift 2 ;;
        --report)      REPORT=1; shift ;;
        --no-baseline) WITH_BASELINE=0; shift ;;
        -h|--help)     sed -n '2,/^# ===*$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)             PASSTHRU+=("$1"); shift ;;   # --dry-run, --force, VAR=VALUE ...
    esac
done

EXPS="$VARIANTS"
[[ "$WITH_BASELINE" -eq 1 ]] && EXPS="$BASELINE $EXPS"

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
if [[ "$REPORT" -eq 1 ]]; then
    echo "==> collecting results from ${RESULTS_ROOT}"
    echo ""
    # exp <TAB> chunk <TAB> macro-WER, taken from the newest FINISHED aggregate.log.
    ROWS=$(oci_ssh bash -s -- "$RESULTS_ROOT" $EXPS <<'REMOTE'
set -u
ROOT="$1"; shift
for exp in "$@"; do
    [ -d "$ROOT/$exp" ] || { echo -e "$exp\t-\tMISSING"; continue; }
    found=0
    for d in "$ROOT/$exp"/leaderboard_eval_chunk*/; do
        [ -d "$d" ] || continue
        a="${d}aggregate.log"
        [ -f "$a" ] || continue
        wer=$(grep -P "^RESULT\tAverage\t" "$a" 2>/dev/null | cut -f3 | tail -1)
        [ -n "$wer" ] || continue
        c=$(basename "$d"); c="${c#leaderboard_eval_chunk}"; c="${c%%_*}"
        t=$(stat -c %Y "$a")
        echo -e "$exp\t$c\t$wer\t$t"
        found=1
    done
    if [ "$found" -eq 0 ]; then echo -e "$exp\t-\tNONE\t0"; fi
done
# Explicit: a trailing `[ ... ] && ...` that evaluates false would make this
# remote script exit 1, which under `set -e` locally kills the report before it
# prints anything.
exit 0
REMOTE
)
    if [[ -z "${ROWS//[[:space:]]/}" ]]; then
        echo "  ERROR: the scan returned nothing -- could not read ${RESULTS_ROOT} on the grid." >&2
        exit 1
    fi
    python3 - "$CHUNKS" <<PY
import sys, collections
rows = """$ROWS""".strip().split("\n")
chunks = sys.argv[1].split()
exps, best = [], {}
status = {}
for line in rows:
    p = line.split("\t")
    if len(p) < 3: continue
    exp, c, wer = p[0], p[1], p[2]
    if exp not in exps: exps.append(exp)
    if wer in ("MISSING", "NONE"):
        status[exp] = wer; continue
    t = int(p[3]) if len(p) > 3 else 0
    # newest completed eval wins for a given (exp, chunk)
    if best.get((exp, c), (-1, None))[0] < t:
        best[(exp, c)] = (t, float(wer))

seen_chunks = sorted({c for (_, c) in best}, key=int)
show = [c for c in chunks if c in seen_chunks] or seen_chunks
if not show:
    print("  No completed evals yet for any of:", ", ".join(exps))
else:
    w = max(len(e) for e in exps) + 2
    print(f"  {'experiment':<{w}}" + "".join(f"{'chunk '+c:>12}" for c in show))
    print("  " + "-" * (w + 12 * len(show)))
    for e in exps:
        cells = ""
        for c in show:
            v = best.get((e, c))
            cells += f"{v[1]:>12.2f}" if v else f"{'-':>12}"
        print(f"  {e:<{w}}{cells}")
    print()
    print("  macro WER% (mean of per-dataset kaldialign WERs), newest completed eval per cell.")
    print("  '-' means that (experiment, chunk) has no finished eval yet -- not a zero.")
for e, s in status.items():
    print(f"  [note] {e}: {'no results directory' if s=='MISSING' else 'no completed eval yet'}")
PY
    exit 0
fi

# ---------------------------------------------------------------------------
# Submit — eval_all.sh owns staleness / averaging / in-flight handling
# ---------------------------------------------------------------------------
# Anchor both ends so e.g. granary2_script_readwrite_ft does not also match a
# future granary2_script_readwrite_ft_v2.
PATTERN="^($(echo "$EXPS" | tr ' ' '|'))\$"

echo "==> read/write eval"
echo "    experiments: ${EXPS}"
echo "    chunk sizes: ${CHUNKS}"
echo "    delegating submission to ./eval_all.sh"
echo ""
exec ./eval_all.sh --only "$PATTERN" --skip '' --chunks "$CHUNKS" ${PASSTHRU[@]+"${PASSTHRU[@]}"}
