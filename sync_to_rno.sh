#!/bin/bash
# Commit/push this worktree to GitHub, then pull it on RNO.
# Usage:
#   ./sync_to_rno.sh [commit message]

set -euo pipefail

BRANCH="ord_sync_d146_current"
REMOTE_URL="http://github.com/hainan-xv/NeMo.git"
RNO_HOST="${RNO_HOST:-draco-rno-login}"
RNO_USER="${RNO_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"
RNO_REPO="${RNO_REPO:-/gpfs/fs1/projects/ent_aiapps/users/hainanx/NeMo}"

cd "$(dirname "$0")"

if [ "$(git branch --show-current)" != "$BRANCH" ]; then
    echo "ERROR: expected branch $BRANCH, got $(git branch --show-current)" >&2
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    msg="${1:-Sync RNO code $(date +%Y%m%d_%H%M%S)}"
    echo "==> Committing local changes: $msg"
    git add -A
    git commit -m "$msg"
else
    echo "==> No local changes to commit"
fi

echo "==> Pushing $BRANCH to $REMOTE_URL"
git push "$REMOTE_URL" "HEAD:$BRANCH"

echo "==> Pulling $BRANCH on RNO: ${RNO_USER}@${RNO_HOST}:${RNO_REPO}"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${RNO_USER}@${RNO_HOST}" \
    "cd '$RNO_REPO' && git pull '$REMOTE_URL' '$BRANCH'"

echo "==> Done"
