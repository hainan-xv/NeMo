#!/usr/bin/env bash
# Commit the current nemo79 code, push it to GitHub, then update ORD from GitHub.
set -euo pipefail

BRANCH="${BRANCH:-nemo79}"
GITHUB_URL="${GITHUB_URL:-https://github.com/hainan-xv/NeMo.git}"
ORD_HOST="${ORD_HOST:-cs-oci-ord-login-01.nvidia.com}"
ORD_USER="${ORD_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"
ORD_REPO="${ORD_REPO:-/lustre/fsw/portfolios/llmservice/users/hainanx/NeMo79}"

cd "$(dirname "${BASH_SOURCE[0]}")"

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$BRANCH" ]]; then
  echo "ERROR: expected branch '$BRANCH', found '$current_branch'." >&2
  exit 1
fi

# Track modifications to existing code plus the project-owned launch/sync
# scripts. Do not use `git add -A`: unrelated local files and secrets must never
# be swept into a GitHub commit.
git add -u
git add .gitignore sync_to_ord.sh
# Model launch + eval scripts now live under the gitignored /launch/ dir (local-
# only scratch); copy most of them to the cluster manually if needed. The project-
# owned launchers are force-added so they sync through git. This covers the two
# SCRIPT training launchers, the two SCRIPT inference launchers, and the shared
# pooled-shard eval backend they exec (eval_leaderboard_slurm.sh).
git add -f launch/script_baseline.sh
git add -f launch/script_promptctl.sh
git add -f launch/eval_script_baseline.sh
git add -f launch/eval_script_promptctl.sh
git add -f launch/eval_leaderboard_slurm.sh

if ! git diff --cached --quiet; then
  message="${1:-Sync ORD code $(date +%Y%m%d_%H%M%S)}"
  echo "==> Committing: $message"
  git commit -m "$message"
else
  echo "==> No staged code changes to commit"
fi

echo "==> Pushing $BRANCH to $GITHUB_URL"
git push "$GITHUB_URL" "HEAD:$BRANCH"

echo "==> Updating ${ORD_USER}@${ORD_HOST}:$ORD_REPO"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no \
  "${ORD_USER}@${ORD_HOST}" bash -s -- "$GITHUB_URL" "$BRANCH" "$ORD_REPO" <<'REMOTE'
set -euo pipefail

url="$1"
branch="$2"
repo="$3"
git_auth=()

# Needed only when the GitHub repository is private. The token stays on ORD and
# is passed as a transient HTTP header; it is not written into .git/config.
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

echo "ORD checkout: $(git -C "$repo" rev-parse --short HEAD)"
REMOTE

echo "==> ORD is updated at $ORD_REPO"
