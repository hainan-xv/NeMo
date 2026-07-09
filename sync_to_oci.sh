#!/bin/bash
# Commit/push this worktree to GitHub, then update it on OCI (draco-oci-iad).
# This is the OCI analog of sync_to_ord.sh: it syncs the FULL NeMo repo to the
# grid checkout that streaming_stt_finetune.sh mounts as /code.
#
# The grid checkout lives in a UNIQUE per-branch dir (NeMo_ord_sync_d146_current)
# so it never interferes with the shared .../hainanx/NeMo used by older scripts.
# Keep OCI_REPO here in sync with CODE_DIR in oci/streaming_stt_finetune.sh.
#
# Usage:
#   ./sync_to_oci.sh [commit message]

set -euo pipefail

BRANCH="ord_sync_d146_current"
REMOTE_URL="https://github.com/hainan-xv/NeMo.git"
OCI_HOST="${OCI_HOST:-draco-oci-login-01.draco-oci-iad.nvidia.com}"
OCI_USER="${OCI_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"
OCI_REPO="${OCI_REPO:-/lustre/fsw/portfolios/llmservice/users/hainanx/NeMo_ord_sync_d146_current}"

cd "$(dirname "$0")"

if [ "$(git branch --show-current)" != "$BRANCH" ]; then
    echo "ERROR: expected branch $BRANCH, got $(git branch --show-current)" >&2
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    msg="${1:-Sync OCI code $(date +%Y%m%d_%H%M%S)}"
    echo "==> Committing local changes: $msg"
    git add -A
    git commit -m "$msg"
else
    echo "==> No local changes to commit"
fi

echo "==> Pushing $BRANCH to $REMOTE_URL"
git push "$REMOTE_URL" "HEAD:$BRANCH"

echo "==> Updating $BRANCH on OCI: ${OCI_USER}@${OCI_HOST}:${OCI_REPO}"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${OCI_USER}@${OCI_HOST}" \
    "if [ -d '$OCI_REPO/.git' ]; then \
         cd '$OCI_REPO' && rm -f .git/index.lock && rm -f .git/FETCH_HEAD 2>/dev/null || true && \
         git fetch -f '$REMOTE_URL' +$BRANCH:refs/tmp/oci_sync_head && \
         git checkout -f '$BRANCH' && git reset --hard refs/tmp/oci_sync_head; \
     else \
         echo 'Repo not found -- cloning (first-time setup)'; \
         mkdir -p \"\$(dirname '$OCI_REPO')\" && git clone --branch '$BRANCH' '$REMOTE_URL' '$OCI_REPO'; \
     fi"

echo "==> Done"
