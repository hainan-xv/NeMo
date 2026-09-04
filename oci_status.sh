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
else
    echo ""
    echo "==> ${out}  (sbatch log, last 5 lines)"
    tail -n 5 "$out"
fi

# The sbatch log is nearly EMPTY by design: the recipes run srun with
# `-o $RESULTS_DIR/slurm-%j-%n.out`, so everything the training prints -- the
# vocabulary size, the losses, the traceback -- goes to a per-rank file under the
# results dir instead. Looking only at the sbatch log makes a job that is
# training happily and a job that is wedged look identical (one line, forever).
# Rank 0 is the one worth reading; the others mostly echo it.
res_root="/lustre/fsw/portfolios/nemotron/users/hainanx/results/SpeechlmScriptCC"
rank0="$(ls -t "${res_root}"/*/slurm-"${job_id}"-0.out 2>/dev/null | head -1)"
[[ -f "$rank0" ]] || rank0="$(ls -t "${res_root}"/*/slurm-"${job_id}"-*.out 2>/dev/null | head -1)"

if [[ ! -f "$rank0" ]]; then
    echo ""
    echo "  (no srun rank log yet under ${res_root}/*/slurm-${job_id}-*.out --"
    echo "   normal for the first ~10 min while the container image is pulled)"
    exit 0
fi

echo ""
echo "==> ${rank0}  (rank log, last ${n} lines)"
# Lightning redraws its progress bar with a CARRIAGE RETURN, so a plain
# `tail -45` on a training log is ~785KB of near-identical bar frames and tells
# you nothing. Split on \r and keep only the last frame of each redraw run, so
# the tail is actual log lines plus one current progress line.
tr '\r' '\n' < "$rank0" | grep -avE '^\s*(Validation|Sanity|Testing) DataLoader' \
    | awk '!/^Epoch [0-9]+:/ {print; prev=""; next} {prev=$0} END {if (prev) print prev}' \
    | tail -n "$n"
out="$rank0"

# The headline numbers, which otherwise scroll past: checkpoint names carry the
# monitored val_wer, so they are the cheapest history of whether it is learning.
ckpt_dir="$(dirname "$rank0")"
# [0-9]+\.[0-9]+ rather than [0-9.]+, which greedily eats the '.' of '.ckpt';
# uniq then collapses the duplicate that '-last.ckpt' makes of its own step.
ckpts="$(ls -1 "${ckpt_dir}"/*/checkpoints/*.ckpt 2>/dev/null | xargs -r -n1 basename \
         | grep -aoE 'step=[0-9]+-val_wer=[0-9]+\.[0-9]+' | sort -t= -k2 -n | uniq)"
if [[ -n "$ckpts" ]]; then
    echo ""
    echo "==> val_wer by checkpoint"
    echo "$ckpts" | sed 's/^/  /'
fi

# A traceback scrolls off the bottom when a rank keeps logging after the
# failure, so surface it explicitly rather than making the reader scroll.
if grep -qE '^(Traceback|\[rank[0-9]+\].*(Error|Exception))' "$out"; then
    echo ""
    echo "==> first traceback in this log"
    grep -nE -m1 -A 25 '^(Traceback|\[rank[0-9]+\].*(Error|Exception))' "$out"
fi
REMOTE
