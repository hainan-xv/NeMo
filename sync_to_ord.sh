#!/bin/bash
# Commit/push this worktree to GitHub, then pull it on ORD.
# Usage:
#   ./sync_to_ord.sh [commit message]

set -euo pipefail

BRANCH="ord_sync_d146_current"
REMOTE_URL="http://github.com/hainan-xv/NeMo.git"
ORD_HOST="cs-oci-ord-login-01.nvidia.com"
ORD_USER="hainanx"
SSH_KEY="$HOME/.ssh/draco-rno"
ORD_REPO="/lustre/fsw/portfolios/llmservice/users/hainanx/NeMo"

cd "$(dirname "$0")"

if [ "$(git branch --show-current)" != "$BRANCH" ]; then
    echo "ERROR: expected branch $BRANCH, got $(git branch --show-current)" >&2
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    msg="${1:-Sync ORD code $(date +%Y%m%d_%H%M%S)}"
    echo "==> Committing local changes: $msg"
    git add -A
    git commit -m "$msg"
else
    echo "==> No local changes to commit"
fi

echo "==> Pushing $BRANCH to $REMOTE_URL"
git push "$REMOTE_URL" "HEAD:$BRANCH"

echo "==> Pulling $BRANCH on ORD: ${ORD_USER}@${ORD_HOST}:${ORD_REPO}"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${ORD_USER}@${ORD_HOST}" \
    "cd '$ORD_REPO' && git pull '$REMOTE_URL' '$BRANCH'"

echo "==> Done"
