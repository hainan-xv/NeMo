#!/bin/bash
# auto_resubmit.sh — Monitor SLURM leaf jobs and resubmit them when they finish.
#
# A "leaf job" is one that no other queued/running job depends on.
# When a leaf job terminates, it is resubmitted from its original sbatch script.
#
# Usage:
#   ./auto_resubmit.sh              # one resubmission per leaf job
#   ./auto_resubmit.sh --loop       # keep chaining: resubmit, then monitor the new job too
#   ./auto_resubmit.sh --dry-run    # show what would happen without actually resubmitting

USER="hainanx"
POLL_INTERVAL=300   # seconds between squeue checks

LOOP_MODE=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --loop)    LOOP_MODE=true ;;
        --dry-run) DRY_RUN=true ;;
        *)         echo "Unknown option: $arg"; exit 1 ;;
    esac
done

declare -A JOB_SCRIPT   # jobid -> sbatch script path
declare -A JOB_NAME     # jobid -> job name
declare -A JOB_WORKDIR  # jobid -> original submission working directory

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Populate JOB_SCRIPT and JOB_NAME for a given job ID via scontrol.
cache_job_info() {
    local jid=$1
    local info
    info=$(scontrol show job "$jid" 2>/dev/null) || return 1
    JOB_SCRIPT[$jid]=$(echo "$info" | grep -oP 'Command=\K\S+')
    JOB_NAME[$jid]=$(echo "$info"  | grep -oP 'JobName=\K\S+')
    JOB_WORKDIR[$jid]=$(echo "$info" | grep -oP 'WorkDir=\K\S+')
}

# Return all job IDs for the user that are still queued or running.
get_my_jobids() {
    squeue -u "$USER" -h -o "%i" | tr -d ' '
}

# Given the full set of user jobs, extract every job ID that appears as a
# dependency target of some other job.  These are "non-leaf" jobs.
get_dependency_targets() {
    squeue -u "$USER" -h -o "%E" \
        | tr ',:()' '\n' \
        | grep -oP '\d+' \
        | sort -u
}

# ── Step 1: Discover leaf jobs ──────────────────────────────────────────────
discover_leaf_jobs() {
    local all_ids dep_targets
    all_ids=$(get_my_jobids)

    if [[ -z "$all_ids" ]]; then
        log "No jobs found for user $USER."
        exit 0
    fi

    dep_targets=$(get_dependency_targets)

    local -a leaves=()
    for jid in $all_ids; do
        if echo "$dep_targets" | grep -qw "$jid"; then
            log "  $jid — has dependents, skipping"
        else
            if cache_job_info "$jid"; then
                leaves+=("$jid")
                log "  $jid — LEAF  name=${JOB_NAME[$jid]}  script=${JOB_SCRIPT[$jid]}"
            else
                log "  $jid — could not query scontrol, skipping"
            fi
        fi
    done

    LEAF_JOBS=("${leaves[@]}")
}

# ── Step 2: Monitor and resubmit ───────────────────────────────────────────
monitor_and_resubmit() {
    while [[ ${#LEAF_JOBS[@]} -gt 0 ]]; do
        log "Monitoring ${#LEAF_JOBS[@]} leaf job(s): ${LEAF_JOBS[*]}"
        sleep "$POLL_INTERVAL"

        local -a remaining=()
        for jid in "${LEAF_JOBS[@]}"; do
            if squeue -j "$jid" -h 2>/dev/null | grep -q "$jid"; then
                remaining+=("$jid")
            else
                local script="${JOB_SCRIPT[$jid]}"
                local name="${JOB_NAME[$jid]}"
                local workdir="${JOB_WORKDIR[$jid]}"
                local script_path="$script"

                if [[ -z "$workdir" || ! -d "$workdir" ]]; then
                    log "Job $jid ($name) finished but workdir not found: $workdir"
                    continue
                fi

                if [[ "$script" != /* ]]; then
                    script_path="$workdir/$script"
                fi

                if [[ -z "$script" || ! -f "$script_path" ]]; then
                    log "Job $jid ($name) finished but script not found: $script_path"
                    continue
                fi

                if $DRY_RUN; then
                    log "[DRY-RUN] Would resubmit job $jid ($name) from: $script_path"
                    continue
                fi

                log "Job $jid ($name) finished. Resubmitting from: $script_path"
                local sbatch_out
                sbatch_out=$(cd "$workdir" && sbatch "$script" 2>&1)
                local new_jid
                if [[ "$sbatch_out" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]([0-9]+) ]]; then
                    new_jid="${BASH_REMATCH[1]}"
                else
                    new_jid=""
                fi

                if [[ -n "$new_jid" ]]; then
                    log "  -> New job $new_jid submitted"
                    if $LOOP_MODE; then
                        cache_job_info "$new_jid"
                        remaining+=("$new_jid")
                        log "  -> (loop mode) now monitoring $new_jid as well"
                    fi
                else
                    log "  -> sbatch failed: $sbatch_out"
                fi
            fi
        done

        LEAF_JOBS=("${remaining[@]}")
    done
}

# ── Main ────────────────────────────────────────────────────────────────────
echo "========================================="
echo "  SLURM Auto-Resubmit Monitor"
echo "  User:     $USER"
echo "  Interval: ${POLL_INTERVAL}s"
echo "  Loop:     $LOOP_MODE"
echo "  Dry-run:  $DRY_RUN"
echo "========================================="
echo ""

log "Scanning current jobs..."
declare -a LEAF_JOBS
discover_leaf_jobs

if [[ ${#LEAF_JOBS[@]} -eq 0 ]]; then
    log "No leaf jobs to monitor."
    exit 0
fi

echo ""
monitor_and_resubmit
log "Done — all monitored jobs have been resubmitted."