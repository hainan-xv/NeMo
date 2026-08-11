#!/usr/bin/env bash
# Commit the current clean SCRIPT repo, push it to GitHub, then update OCI from
# GitHub. Mirrors the workflow of the original NeMo repo's sync_to_oci.sh, but
# points at this repo's branch, fork, and OCI storage location.
set -euo pipefail

BRANCH="${BRANCH:-script}"
GITHUB_URL="${GITHUB_URL:-https://github.com/hainan-xv/NeMo.git}"
OCI_HOST="${OCI_HOST:-draco-oci-dc-03.draco-oci-iad.nvidia.com}"
OCI_USER="${OCI_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"
OCI_REPO="${OCI_REPO:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_script_clean}"

cd "$(dirname "${BASH_SOURCE[0]}")"

# The local working branch (e.g. heh/streaming_speechlm_automodel) is decoupled
# from the remote branch we publish to. We always push the current HEAD to the
# remote branch $BRANCH (default: script) and check that same branch out on the
# grid, no matter what the local branch is called.
current_branch="$(git branch --show-current)"
echo "==> Local branch: ${current_branch:-<detached HEAD>}  ->  remote branch: $BRANCH"

# Stage existing tracked changes plus the project-owned launch/sync scripts.
# Avoid `git add -A`, which could accidentally include local credentials.
# launch/ is tracked normally, but new files still need an explicit add
# (git add -u only stages already-tracked paths).
git add -u
git add .gitignore sync_to_oci.sh launch/script_baseline.sh

if ! git diff --cached --quiet; then
  message="${1:-Sync OCI code $(date +%Y%m%d_%H%M%S)}"
  echo "==> Committing: $message"
  git commit -m "$message"
else
  echo "==> No staged code changes to commit"
fi

echo "==> Pushing $BRANCH to $GITHUB_URL"
git push "$GITHUB_URL" "HEAD:$BRANCH"

echo "==> Updating ${OCI_USER}@${OCI_HOST}:$OCI_REPO"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no \
  "${OCI_USER}@${OCI_HOST}" bash -s -- "$GITHUB_URL" "$BRANCH" "$OCI_REPO" <<'REMOTE'
set -euo pipefail

url="$1"
branch="$2"
repo="$3"
git_auth=()

# Needed only for a private GitHub repository. The token remains on OCI and is
# used as a transient HTTP header, never stored in the repository config.
if [[ -r "$HOME/.github_token" ]]; then
  github_token="$(tr -d '\r\n' < "$HOME/.github_token")"
  basic_auth="$(printf 'x-access-token:%s' "$github_token" | base64 | tr -d '\r\n')"
  git_auth=(-c "http.extraHeader=Authorization: Basic $basic_auth")
  unset github_token
fi

if [[ -d "$repo/.git" ]]; then
  # Clear stale lock files left by a previously interrupted git operation
  # (single-user sync, so no concurrent git process should legitimately hold one).
  find "$repo/.git" -name '*.lock' -type f -delete 2>/dev/null || true
  git -C "$repo" "${git_auth[@]}" fetch --force "$url" \
    "$branch:refs/remotes/github/$branch"
  git -C "$repo" checkout -B "$branch" "refs/remotes/github/$branch"
  git -C "$repo" reset --hard "refs/remotes/github/$branch"
else
  if [[ -e "$repo" ]]; then
    backup="${repo}.pre-git.$(date +%Y%m%d_%H%M%S)"
    echo "Existing non-Git directory moved to $backup"
    mv "$repo" "$backup"
  fi
  mkdir -p "$(dirname "$repo")"
  git "${git_auth[@]}" clone --branch "$branch" --single-branch "$url" "$repo"
fi

echo "OCI checkout: $(git -C "$repo" rev-parse --short HEAD)"
REMOTE

echo "==> OCI is updated at $OCI_REPO"
