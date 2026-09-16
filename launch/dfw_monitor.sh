#!/bin/bash
# ============================================================================
# DFW-side training monitor + periodic evaluator.
#
#   nohup bash launch/dfw_monitor.sh > ~/dfw_monitor.log 2>&1 &
#   tail -f ~/dfw_monitor.log
#
# Runs ON THE DFW LOGIN NODE, independent of any Claude session or laptop. It
# does two things on a timer:
#
#   every HEALTH_INTERVAL (1h)  health-check the arms and log one line each
#   every EVAL_INTERVAL   (7h)  submit the full four-arm leaderboard eval
#
# WHAT IT DELIBERATELY DOES NOT DO: cancel, relaunch or otherwise touch training
# jobs. Resubmission is auto.sh's job and having two things resubmit the same
# arms is how duplicates arise -- four duplicate arms once held the entire node
# quota so their replacements could never start. This script only OBSERVES
# training; the single exception is submitting eval jobs, which take one
# interactive node and never contend with the batch arms.
#
# STALL DETECTION is the point of the health check. A job being "RUNNING" proves
# nothing -- an arm can sit in a requeue crash-loop, or hang after a NCCL
# timeout, and still look alive in squeue. So each tick records the newest
# checkpoint per arm and compares against the previous tick; an arm whose
# checkpoint has not moved in STALL_TICKS consecutive hours is reported as
# STALLED. Checkpoints land every ~2000 steps, well under an hour for every arm
# here, so one missed hour is already suspicious.
#
# ENV
#   HEALTH_INTERVAL  seconds between health checks   (default 3600)
#   EVAL_INTERVAL    seconds between evals           (default 25200 = 7h)
#   EVAL_AT_START    1 to submit an eval immediately (default 0)
#   STALL_TICKS      ticks without progress -> STALL (default 2)
#   MAX_HOURS        stop after this many hours      (default 168 = 7 days)
# ============================================================================
set -uo pipefail

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
REPO="${REPO:-${DFW}/hainanx/NeMo_SCRIPT_cc}"
RESULTS="${RESULTS:-${DFW}/hainanx/results/SpeechlmDFW}"
STATE="${STATE:-$HOME/.dfw_monitor_state}"

HEALTH_INTERVAL="${HEALTH_INTERVAL:-3600}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25200}"
EVAL_AT_START="${EVAL_AT_START:-0}"
STALL_TICKS="${STALL_TICKS:-2}"
MAX_HOURS="${MAX_HOURS:-168}"

# "exp_name|slurm job-name fragment". The two are NOT derivable from each other:
# experiment names use underscores and the granary2 prefix, Slurm job names use
# dashes and drop it. Deriving one from the other silently matched nothing.
# Note "script-banded1-nodelay-v2" is not a substring of
# "script-banded1-both-nodelay-v2", so these fragments stay unambiguous.
ARMS=(
  "dfw_granary2_chat_banded1_nodelay_v2|dfw-chat-banded1-nodelay-v2"
  "dfw_granary2_chat_banded1_both_nodelay_v2|dfw-chat-banded1-both-nodelay-v2"
  "dfw_granary2_script_banded1_nodelay_v2|dfw-script-banded1-nodelay-v2"
  "dfw_granary2_script_banded1_both_nodelay_v2|dfw-script-banded1-both-nodelay-v2"
)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p "$STATE"

# Newest checkpoint for an arm, as a bare filename. Excludes the averaged
# artifact, which an eval run drops into the SCRIPT checkpoint directories and
# which would otherwise look like fresh training progress on the next tick.
newest_ckpt() {
    ls -1t "${RESULTS}/$1/$1/checkpoints"/*.ckpt 2>/dev/null \
        | grep -v -- '-averaged\.ckpt$' | head -1 | xargs -r basename
}

best_val_wer() {
    ls -1 "${RESULTS}/$1/$1/checkpoints"/*.ckpt 2>/dev/null \
        | grep -oE 'val_wer=[0-9.]+' | sed 's/val_wer=//' | sort -g | head -1
}

# Is an eval already queued or running? Submitting a second one would contend
# for the single interactive slot and average a moving checkpoint set twice.
eval_in_flight() {
    squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -q 'eval-v2-all'
}

health_check() {
    local unhealthy=0
    local snapshot
    snapshot=$(squeue -u "$USER" -h -o '%j %T' 2>/dev/null)
    for entry in "${ARMS[@]}"; do
        local arm="${entry%%|*}" jobpat="${entry#*|}"
        local short="${arm#dfw_granary2_}"
        local state ckpt prev best oom stall_file stalls
        state=$(echo "$snapshot" | grep -F "$jobpat" | head -1 | awk '{print $2}')
        ckpt=$(newest_ckpt "$arm")
        best=$(best_val_wer "$arm")
        oom=$(grep -lic 'out of memory' "${RESULTS}/${arm}"/error-*.out 2>/dev/null | wc -l)

        stall_file="${STATE}/${arm}.ckpt"
        prev=$(cat "$stall_file" 2>/dev/null)
        stalls=0
        if [[ -n "$ckpt" && "$ckpt" == "$prev" ]]; then
            stalls=$(( $(cat "${STATE}/${arm}.stalls" 2>/dev/null || echo 0) + 1 ))
        fi
        echo "$ckpt"   > "$stall_file"
        echo "$stalls" > "${STATE}/${arm}.stalls"

        local flag="ok"
        [[ -z "$ckpt" ]]                  && { flag="NO CHECKPOINTS"; unhealthy=1; }
        [[ "$oom" -gt 0 ]]                && { flag="OOM in logs";    unhealthy=1; }
        [[ "$stalls" -ge "$STALL_TICKS" ]] && { flag="STALLED (${stalls}h no new ckpt)"; unhealthy=1; }

        printf '    %-42s %-10s best=%-8s %-34s %s\n' \
            "$short" "${state:-ABSENT}" "${best:-?}" "${ckpt:-none}" "$flag"
    done

    if ! pgrep -f 'bash .*auto\.sh' >/dev/null 2>&1; then
        log "    WARNING: auto.sh is NOT running -- arms will stop at their next 4h wall."
        unhealthy=1
    fi
    return $unhealthy
}

submit_eval() {
    if eval_in_flight; then
        log "  eval already queued/running; skipping this cycle"
        return
    fi
    log "  submitting four-arm leaderboard eval (FORCE_AVERAGE=1)"
    local out
    out=$(cd "$REPO" && sbatch --export=ALL,FORCE_AVERAGE=1 launch/dfw_eval_v2_all.sh 2>&1)
    log "    $out"
}

log "======================================================================"
log "DFW monitor starting"
log "  health every ${HEALTH_INTERVAL}s, eval every ${EVAL_INTERVAL}s, max ${MAX_HOURS}h"
log "  arms: ${#ARMS[@]}   repo: ${REPO}"
log "======================================================================"

START=$(date +%s)
LAST_EVAL=$START
[[ "$EVAL_AT_START" == "1" ]] && { log "EVAL (at start)"; submit_eval; LAST_EVAL=$(date +%s); }

while true; do
    now=$(date +%s)
    if (( (now - START) > MAX_HOURS * 3600 )); then
        log "MAX_HOURS reached; exiting."
        break
    fi

    log "health check ($(( (now - START) / 3600 ))h elapsed)"
    health_check && log "  all arms healthy" || log "  ^^ ATTENTION: see flags above"

    if (( now - LAST_EVAL >= EVAL_INTERVAL )); then
        log "EVAL cycle ($(( (now - LAST_EVAL) / 3600 ))h since last)"
        submit_eval
        LAST_EVAL=$now
    fi

    sleep "$HEALTH_INTERVAL"
done
