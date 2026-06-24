#!/bin/bash
# Commit/push this worktree to GitHub, then pull it on OCI (draco-oci-iad).
# Clones the repo on the first run if it doesn't exist yet.
#
# The grid-side checkout lives in a UNIQUE directory named after the branch
# (NeMo_<branch>) so it never interferes with the shared .../hainanx/NeMo
# used by older scripts. Keep OCI_REPO here in sync with CODE_DIR in
# streaming_stt_finetune.sh.
#
# Usage:
#   ./oci/sync.sh [commit message]

set -euo pipefail

BRANCH="ord_sync_d146_current"
REMOTE_URL="http://github.com/hainan-xv/NeMo.git"
OCI_HOST="${OCI_HOST:-draco-oci-login-01.draco-oci-iad.nvidia.com}"
OCI_USER="${OCI_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"
# Unique per-branch checkout -- does NOT touch the shared .../hainanx/NeMo.
OCI_REPO="${OCI_REPO:-/lustre/fsw/portfolios/llmservice/users/hainanx/NeMo_ord_sync_d146_current}"

# Run git from the repo root regardless of where this script is invoked from.
cd "$(dirname "$0")/.."

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
         cd '$OCI_REPO' && git fetch '$REMOTE_URL' '$BRANCH' && git checkout '$BRANCH' && git reset --hard FETCH_HEAD; \
     else \
         echo 'Repo not found -- cloning (first-time setup)'; \
         mkdir -p \"\$(dirname '$OCI_REPO')\" && git clone --branch '$BRANCH' '$REMOTE_URL' '$OCI_REPO'; \
     fi"

echo "==> Done"
