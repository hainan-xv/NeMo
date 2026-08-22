#!/bin/bash
# ============================================================================
# Same as ./oci_launch.sh, but submits to the INTERACTIVE queue on ONE node.
#
# For quick debug runs: the interactive partition usually schedules far sooner
# than the batch blocks, and a single node keeps the job small. The submission
# path is otherwise identical (same sync check, same script, same arguments), so
# a run that works here works unchanged under ./oci_launch.sh at full scale.
#
# Two differences from the batch launcher:
#   * --partition=interactive --nodes=1 override the script's own #SBATCH lines.
#     The launcher passes trainer.num_nodes=$SLURM_JOB_NUM_NODES, so the training
#     job correctly sees 1 node without editing anything.
#   * --follow defaults ON, so you watch the log live — which is the point of
#     using the interactive queue. Ctrl-C stops the tail, NOT the job.
#
# Because the node count differs from what the launch script's "#SBATCH -N" asks
# for, the script appends _n1 to EXP_NAME (e.g. granary2_script_baseline_n1).
# That keeps a debug run from resuming out of, and then overwriting, the
# full-scale run's checkpoints — RESULTS_DIR is derived from EXP_NAME and the
# recipe sets resume_if_exists=true. Pass SKIP_NODE_SUFFIX=1 to opt out.
#
# USAGE (identical to oci_launch.sh)
#   ./oci_launch_interactive.sh launch/script_baseline.sh
#   ./oci_launch_interactive.sh DELAY=6 launch/script_baseline.sh 123
#   ./oci_launch_interactive.sh --no-follow launch/script_baseline.sh
#
# The interactive partition caps wall time well below the batch queue's 4 h, so
# override it when a run needs longer (or shorter):
#   SBATCH_OPTS="--time=01:00:00" ./oci_launch_interactive.sh launch/script_baseline.sh
#
# ENV
#   OCI_PARTITION   default "interactive"; set to change the queue
#   OCI_NODES       default 1
#   FOLLOW=0        do not tail (same as passing --no-follow)
# ============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export OCI_PARTITION="${OCI_PARTITION:-interactive}"
export OCI_NODES="${OCI_NODES:-1}"
export FOLLOW="${FOLLOW:-1}"

# oci_launch.sh has no --no-follow (its default is off), so strip it here.
ARGS=()
for a in "$@"; do
    if [[ "$a" == "--no-follow" ]]; then
        FOLLOW=0
        export FOLLOW
    else
        ARGS+=("$a")
    fi
done

echo "==> interactive queue: partition=${OCI_PARTITION} nodes=${OCI_NODES} follow=${FOLLOW}"
exec ./oci_launch.sh ${ARGS[@]+"${ARGS[@]}"}
