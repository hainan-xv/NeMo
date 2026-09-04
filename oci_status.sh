#!/bin/bash
# ============================================================================
# State and log for a job submitted by oci_launch.sh / oci_launch_interactive.sh.
#
#   ./oci_status.sh 13089420           # state + the last 60 log lines
#   ./oci_status.sh 13089420 200       # ... last 200 lines
#   ./oci_status.sh                    # your whole queue
#
# WHY: after a launch the useful question is almost always "did it get past
# the first validation, and if not what was the traceback" -- which otherwise
# takes a scontrol call to find StdOut, then a tail. `--follow` on the launcher
# only helps while you are still attached; this works after the fact, and works
# for a job that already exited (where scontrol has forgotten the path and only
# sacct still knows the job existed).
# ============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ./oci_env.sh

JOB_ID="${1:-}"
TAIL_LINES="${2:-60}"

if [[ -z "$JOB_ID" ]]; then
    echo "==> queue for ${OCI_USER}@${OCI_HOST}"
    oci_ssh "squeue -u '${OCI_USER}' -o '%.12i %.34j %.10T %.10M %.6D %R'"
    exit 0
fi

oci_ssh bash -s -- "$JOB_ID" "$TAIL_LINES" "$OCI_REPO" <<'REMOTE'
set -uo pipefail
job_id="$1"; n="$2"; repo="$3"

echo "==> job ${job_id}"
scontrol show job "$job_id" 2>/dev/null | tr ' ' '\n' \
    | sed -n 's/^JobState=/  state:    /p;s/^RunTime=/  runtime:  /p;s/^Reason=/  reason:   /p'
sacct -j "$job_id" --format=JobID%14,State%22,Elapsed,ExitCode --noheader 2>/dev/null \
    | head -4 | sed 's/^/  sacct: /'

# scontrol knows StdOut only while the job is still in Slurm's memory; once it
# has aged out, fall back to the newest matching file the launchers produce
# (slurm_out/%x=%j, so the id is in the NAME).
out="$(scontrol show job "$job_id" 2>/dev/null | tr ' ' '\n' | sed -n 's/^StdOut=//p' | head -1)"
[[ -f "$out" ]] || out="$(ls -t "${repo}"/slurm_out/*="$job_id" 2>/dev/null | head -1)"

if [[ ! -f "$out" ]]; then
    echo "  (no log found for ${job_id} under ${repo}/slurm_out)"
    exit 0
fi

echo ""
echo "==> ${out}  (last ${n} lines)"
tail -n "$n" "$out"

# A traceback scrolls off the bottom when a rank keeps logging after the
# failure, so surface it explicitly rather than making the reader scroll.
if grep -qE '^(Traceback|\[rank[0-9]+\].*(Error|Exception))' "$out"; then
    echo ""
    echo "==> first traceback in this log"
    grep -nE -m1 -A 25 '^(Traceback|\[rank[0-9]+\].*(Error|Exception))' "$out"
fi
REMOTE
