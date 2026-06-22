#!/bin/bash
set -o pipefail

# ============================================================================
# Run eval_aishell.sh over EVERY model in an RNO grid (project) folder and
# print a single clean, aligned results table to stdout.
#
# It enumerates the experiments under
#   <REMOTE_RESULTS_ROOT>/<PROJECT>/<EXP>/<EXP>/checkpoints
# (default PROJECT=Mandarin_202606_enc256l18_enginit), then for each one calls
# the existing ./eval_aishell.sh with the most vanilla config -- except
# RUN_AVERAGING is forced ON for every model (averages the top-k non '-last'
# ckpts into a *-averaged.nemo and evaluates that).
#
# Output streams:
#   * stdout = ONLY the results table. A header row is printed first, then the
#     moment a model finishes its row is printed right away, column-aligned
#     under the header. Nothing else touches stdout, so it stays clean:
#         ./eval_aishell_grid.sh > table.txt        # table only
#         ./eval_aishell_grid.sh 2>/dev/null         # table only, on screen
#   * stderr = all the per-model eval logs (download / averaging / inference),
#     streamed live so you can still watch progress.
#
# Usage:
#   ./eval_aishell_grid.sh [EXP1 EXP2 ...]
#     (no args) -> every experiment in the project that has a checkpoints dir
#     (args)    -> only the named experiments
#
# Env overrides:
#   PROJECT              grid/project folder (default Mandarin_202606_enc256l18_enginit)
#   DEVICE_ID            GPU id passed to eval_aishell.sh (default 0)
#   SETS / ONLY          test sets to score (default "test_android test_ios test_mic")
#   plus anything eval_aishell.sh honours (REMOTE_*, MANIFEST_DIR, BATCH_SIZE,
#   FORCE_AVERAGE, REUSE_AVG, ...). RUN_AVERAGING is forced to 1 here.
# ============================================================================

NEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${NEMO_ROOT}/eval_aishell.sh"
if [ ! -x "$EVAL_SCRIPT" ] && [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: cannot find eval_aishell.sh at ${EVAL_SCRIPT}" >&2
    exit 1
fi

# ---------- Grid / remote config (mirrors eval_aishell.sh defaults) ----------
PROJECT="${PROJECT:-Mandarin_202606_enc256l18_enginit}"
REMOTE_HOST="${REMOTE_HOST:-draco-rno-login}"
REMOTE_USER="${REMOTE_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/gpfs/fs1/projects/ent_aiapps/users/hainanx/results}"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no"

DEVICE_ID="${DEVICE_ID:-0}"

# Which test sets / columns to score (vanilla default).
if [ -n "${ONLY:-}" ]; then
    SETS="${ONLY//,/ }"
fi
SETS="${SETS:-test_android test_ios test_mic}"
SETS="${SETS//,/ }"

# RUN_AVERAGING is forced ON for every model, per request.
export RUN_AVERAGING=1
# Stream child python output without buffering so logs appear live.
export PYTHONUNBUFFERED=1

# ---------- Resolve the list of experiments to evaluate ----------
if [ "$#" -gt 0 ]; then
    MODELS=("$@")
    echo "==> Evaluating ${#MODELS[@]} explicitly-named experiment(s) in project '${PROJECT}'." >&2
else
    echo "==> Listing experiments with checkpoints under ${REMOTE_RESULTS_ROOT}/${PROJECT}/ ..." >&2
    LIST_CMD="cd '${REMOTE_RESULTS_ROOT}/${PROJECT}' 2>/dev/null && for d in */; do d=\${d%/}; [ -d \"\$d/\$d/checkpoints\" ] && echo \"\$d\"; done"
    mapfile -t MODELS < <(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "$LIST_CMD" 2>/dev/null)
    if [ "${#MODELS[@]}" -eq 0 ]; then
        echo "ERROR: no experiments with a checkpoints/ dir found under ${REMOTE_RESULTS_ROOT}/${PROJECT}/" >&2
        echo "       (check the RNO connection / PROJECT name, or pass experiment names as args.)" >&2
        exit 1
    fi
    echo "    Found ${#MODELS[@]} model(s):" >&2
    printf '      - %s\n' "${MODELS[@]}" >&2
fi

# ---------- Table geometry ----------
# Model column wide enough for the longest exp name (and the "model" header).
MODEL_W=5
for m in "${MODELS[@]}"; do
    [ "${#m}" -gt "$MODEL_W" ] && MODEL_W="${#m}"
done
# Each metric column: wide enough for the widest set name (and "average").
COL_W=7
for s in $SETS average; do
    [ "${#s}" -gt "$COL_W" ] && COL_W="${#s}"
done

print_table_row() {  # $1=model, $2..=cell values (one per set, then average)
    # Cells (set values, then average) -> a single clean table row on stdout.
    local model="$1"; shift
    local line
    printf -v line "%-${MODEL_W}s" "$model"
    local v
    for v in "$@"; do
        printf -v line "%s  %${COL_W}s" "$line" "$v"
    done
    printf '%s\n' "$line"
}

print_table_header() {
    local hdr_cols=($SETS average)
    print_table_row "model" "${hdr_cols[@]}"
}

# ---------- Run the grid ----------
echo "==> RUN_AVERAGING=1 | sets: ${SETS} | device: ${DEVICE_ID}" >&2
echo "==> stdout = clean results table; all eval logs stream on stderr below." >&2

# stdout: header first, then one row per model as it finishes.
print_table_header

TMP_LOG="$(mktemp)"
trap 'rm -f "$TMP_LOG"' EXIT

for exp in "${MODELS[@]}"; do
    [ -z "$exp" ] && continue
    {
        echo ""
        echo "############################################################################"
        echo "## MODEL: ${exp}"
        echo "############################################################################"
    } >&2

    # Run the existing per-model eval; logs -> stderr (live) and captured to file.
    PROJECT="$PROJECT" RUN_AVERAGING=1 SETS="$SETS" DEVICE_ID="$DEVICE_ID" \
        bash "$EVAL_SCRIPT" "$exp" "" "$DEVICE_ID" 2>&1 | tee "$TMP_LOG" >&2
    rc="${PIPESTATUS[0]}"

    # Parse per-set CER + average straight from the captured summary block.
    cells=()
    for s in $SETS; do
        val=$(grep -E "^  ${s}[[:space:]]" "$TMP_LOG" | tail -1 | awk '{print $2}')
        if [ "$rc" -ne 0 ] && [ -z "$val" ]; then
            val="ERR"
        fi
        cells+=("${val:-N/A}")
    done
    avg=$(grep -E "^  AVERAGE[[:space:]]" "$TMP_LOG" | tail -1 | awk '{print $2}')
    if [ -z "$avg" ] && [ "$rc" -ne 0 ]; then
        avg="ERR"
    fi
    cells+=("${avg:-N/A}")

    # Print this model's row right away on stdout, aligned under the header.
    print_table_row "$exp" "${cells[@]}"
done
