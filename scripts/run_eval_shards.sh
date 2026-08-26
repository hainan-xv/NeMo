#!/bin/bash
# ============================================================================
# Fan an eval across the node's GPUs, then RECOVER any GPU that died.
#
# One GPU failing used to poison a whole eval quietly: the driver catches a bad
# batch and writes empty hypotheses so one bad batch cannot kill a shard, but a
# CUDA "unspecified launch failure" poisons the context, so EVERY later batch on
# that GPU fails too. The result was a complete-looking run in which 1/8 of the
# corpus was blank and scored as 100% deletions -- 18.16 macro instead of ~6.5,
# with nothing in aggregate.log to say why.
#
# So: after the fan-out, work out which shards are trustworthy. A shard is BAD if
# its process exited non-zero, its output file is missing, or its log recorded any
# failed batch. Each bad shard's utterances are then split evenly across the GOOD
# GPUs and decoded again (--subshard_count/--subshard_index), and the bad shard's
# original output -- the file full of empty hypotheses -- is deleted so the
# aggregate cannot see it.
#
# Exit status is non-zero if any shard is still unrecovered, so the caller can
# refuse to publish a partial number.
#
# USAGE
#   run_eval_shards.sh --ngpu 8 --shard-dir DIR --log-dir DIR \
#       --driver /code/scripts/script_leaderboard_eval.py -- <common driver args>
#
# The driver args are passed verbatim; this script appends --num_shards,
# --shard_index, --device and (on recovery) the sub-shard flags.
# ============================================================================
set -uo pipefail

NGPU=8; SHARD_DIR=""; LOG_DIR=""; DRIVER=""; MAX_RECOVERY_ROUNDS="${MAX_RECOVERY_ROUNDS:-1}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ngpu)       NGPU="$2"; shift 2 ;;
        --shard-dir)  SHARD_DIR="$2"; shift 2 ;;
        --log-dir)    LOG_DIR="$2"; shift 2 ;;
        --driver)     DRIVER="$2"; shift 2 ;;
        --)           shift; break ;;
        *)            echo "run_eval_shards: unexpected argument: $1" >&2; exit 2 ;;
    esac
done
COMMON=("$@")
[[ -n "$SHARD_DIR" && -n "$LOG_DIR" && -n "$DRIVER" ]] || {
    echo "run_eval_shards: --shard-dir, --log-dir and --driver are required" >&2; exit 2; }

shard_file() { echo "${SHARD_DIR}/shard${1}_of${NGPU}.generations.jsonl"; }

# A shard is only trustworthy if it exited cleanly AND never swallowed a batch.
shard_ok() {  # $1 = shard index, $2 = exit status, $3 = log
    [[ "$2" == "0" ]] || { echo "    shard $1: process exited $2"; return 1; }
    [[ -s "$(shard_file "$1")" ]] || { echo "    shard $1: no output file"; return 1; }
    local nf
    nf="$(grep -c 'batch at .* failed' "$3" 2>/dev/null || true)"
    [[ "${nf:-0}" -eq 0 ]] || { echo "    shard $1: ${nf} failed batch(es)"; return 1; }
    return 0
}

echo "==> decoding ${NGPU} shards"
pids=(); logs=()
for gpu in $(seq 0 $((NGPU - 1))); do
    log="${LOG_DIR}/shard_${gpu}.log"
    logs+=("$log")
    echo "  [gpu ${gpu}] shard ${gpu}/${NGPU} -> ${log}"
    CUDA_VISIBLE_DEVICES="$gpu" python "$DRIVER" "${COMMON[@]}" \
        --num_shards "$NGPU" --shard_index "$gpu" --device 0 > "$log" 2>&1 &
    pids+=($!)
done

BAD=(); GOOD=()
for gpu in $(seq 0 $((NGPU - 1))); do
    st=0; wait "${pids[$gpu]}" || st=$?
    if shard_ok "$gpu" "$st" "${logs[$gpu]}"; then GOOD+=("$gpu"); else BAD+=("$gpu"); fi
done
echo "==> ${#GOOD[@]} shard(s) OK, ${#BAD[@]} bad"

# --- recovery ---
round=0
while [[ ${#BAD[@]} -gt 0 && ${#GOOD[@]} -gt 0 && $round -lt $MAX_RECOVERY_ROUNDS ]]; do
    round=$((round + 1))
    echo "==> recovery round ${round}: redistributing ${#BAD[@]} shard(s) over ${#GOOD[@]} healthy GPU(s)"
    STILL_BAD=()
    for b in "${BAD[@]}"; do
        # Drop the failed shard's output first: it is full of empty hypotheses
        # and would otherwise be scored alongside the recovered slices.
        rm -f "$(shard_file "$b")"
        G=${#GOOD[@]}
        rpids=(); rlogs=()
        for j in $(seq 0 $((G - 1))); do
            g="${GOOD[$j]}"
            rlog="${LOG_DIR}/shard_${b}_recover_${j}of${G}.log"
            rlogs+=("$rlog")
            echo "  [gpu ${g}] shard ${b} slice ${j}/${G} -> ${rlog}"
            CUDA_VISIBLE_DEVICES="$g" python "$DRIVER" "${COMMON[@]}" \
                --num_shards "$NGPU" --shard_index "$b" \
                --subshard_count "$G" --subshard_index "$j" --device 0 > "$rlog" 2>&1 &
            rpids+=($!)
        done
        ok=1
        for j in $(seq 0 $((G - 1))); do
            st=0; wait "${rpids[$j]}" || st=$?
            f="${SHARD_DIR}/shard${b}_of${NGPU}_sub${j}of${G}.generations.jsonl"
            if [[ "$st" != "0" ]] || [[ ! -s "$f" ]] || grep -q 'batch at .* failed' "${rlogs[$j]}" 2>/dev/null; then
                echo "    slice ${j}/${G} of shard ${b} FAILED"
                ok=0
            fi
        done
        if [[ "$ok" == "1" ]]; then echo "    shard ${b}: recovered"; else STILL_BAD+=("$b"); fi
    done
    BAD=("${STILL_BAD[@]+"${STILL_BAD[@]}"}")
done

if [[ ${#BAD[@]} -gt 0 ]]; then
    echo "ERROR: shard(s) ${BAD[*]} could not be recovered." >&2
    echo "       Refusing to aggregate: those utterances would be scored as empty" >&2
    echo "       hypotheses and the WER would look plausible but be wrong." >&2
    exit 1
fi
echo "==> all shards accounted for"
