#!/bin/bash
# ============================================================================
# Shared connection settings for the CW DFW cluster.
#
# The counterpart of oci_env.sh, and deliberately the same shape: one place that
# defines host, user, key and grid path, sourced by sync_to_dfw.sh and
# dfw_launch.sh so they cannot drift apart.
#
# The SSH alias `dfw` is expected in ~/.ssh/config and carries the User and
# IdentityFile, so this file does not repeat them -- the key is the SAME
# draco-rno key the OCI grid uses, authorised on both clusters.
#
# Every value is overridable from the environment, e.g.
#   DFW_REPO=/lustre/.../NeMo_experiment ./dfw_launch.sh launch/dfw_script_banded1.sh
# ============================================================================

# Branch that local HEAD is published to (shared with the OCI grid -- the same
# code runs on both, only the launch scripts differ).
BRANCH="${BRANCH:-SCRIPT_cc}"
GITHUB_URL="${GITHUB_URL:-https://github.com/hainan-xv/NeMo.git}"

# The ~/.ssh/config alias, not a raw hostname: it holds User, IdentityFile and
# any ProxyJump this cluster needs, so none of that has to be duplicated here.
DFW_ALIAS="${DFW_ALIAS:-dfw}"

# The grid checkout. Filled in once the cluster's scratch layout is confirmed --
# DFW does NOT share a filesystem with OCI (verified: the example paths under
# .../users/heh/ are not readable from the OCI login node).
# NOTE: users/ under this project is NOT writable by me, and users/hainanx
# cannot be created -- verified, mkdir returns EACCES. The project ROOT is
# writable, so scratch lives directly under it. Must match MYDIR in
# launch/dfw_script_banded1.sh.
DFW_REPO="${DFW_REPO:-/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/hainanx/NeMo_SCRIPT_cc}"

# Standard ssh invocation, mirroring oci_ssh. BatchMode is NOT forced here so an
# interactive first connection can still accept a host key; the callers that poll
# in the background pass -o BatchMode=yes themselves.
dfw_ssh() {
    ssh -o StrictHostKeyChecking=accept-new "${DFW_ALIAS}" "$@"
}
