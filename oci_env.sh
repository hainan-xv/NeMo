#!/bin/bash
# ============================================================================
# Shared OCI connection settings.
#
# Sourced by sync_to_oci.sh and oci_launch.sh so the host, key and grid path are
# defined exactly once. CODE_DIR in launch/script_baseline.sh must match
# OCI_REPO here — that is the directory mounted into the container as /code.
#
# Every value is overridable from the environment, e.g.
#   OCI_REPO=/lustre/.../NeMo_experiment ./oci_launch.sh launch/script_baseline.sh
# ============================================================================

# Branch that local HEAD is published to (and that the grid checkout tracks).
BRANCH="${BRANCH:-SCRIPT_cc}"
GITHUB_URL="${GITHUB_URL:-https://github.com/hainan-xv/NeMo.git}"

OCI_HOST="${OCI_HOST:-draco-oci-dc-03.draco-oci-iad.nvidia.com}"
OCI_USER="${OCI_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"

# The grid checkout. Deliberately distinct from the older NeMo_script_clean tree
# so a sync (which does `reset --hard`) can never clobber it.
OCI_REPO="${OCI_REPO:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_SCRIPT_cc}"

# Standard ssh invocation used by both scripts.
oci_ssh() {
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${OCI_USER}@${OCI_HOST}" "$@"
}
