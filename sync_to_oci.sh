#!/usr/bin/env bash
# Commit the current nemo79 code, push it to GitHub, then update OCI from GitHub.
set -euo pipefail

BRANCH="${BRANCH:-nemo79}"
GITHUB_URL="${GITHUB_URL:-https://github.com/hainan-xv/NeMo.git}"
OCI_HOST="${OCI_HOST:-draco-oci-dc-03.draco-oci-iad.nvidia.com}"
OCI_USER="${OCI_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"
OCI_REPO="${OCI_REPO:-/lustre/fsw/portfolios/llmservice/users/hainanx/NeMo79}"

cd "$(dirname "${BASH_SOURCE[0]}")"

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$BRANCH" ]]; then
  echo "ERROR: expected branch '$BRANCH', found '$current_branch'." >&2
  exit 1
fi

# Stage existing tracked changes and the project-owned launch/sync scripts only.
# Avoid `git add -A`, which could accidentally include local credentials.
git add -u
git add .gitignore sync_to_oci.sh sync_to_ord.sh launch_with_interactive.sh
# New tracked-dir scripts must be added explicitly (git add -u only stages
# already-tracked files).
git add scripts/asr_leaderboard_shard_decode.py
git add scripts/eval_wandb_report.py
# Model launch + eval scripts now live under the gitignored /launch/ dir (local-
# only scratch). Most stay untracked -- copy them to the cluster manually if needed
# (e.g. rsync launch/ to $OCI_REPO/launch/). The project-owned baseline launcher is
# force-added so they DO sync to the grid through git. This covers the two SCRIPT
# training launchers, the two SCRIPT inference launchers, and the shared pooled-
# shard eval backend they exec (eval_leaderboard_slurm.sh).
git add -f launch/script_baseline.sh
git add -f launch/script_promptctl.sh
git add -f launch/eval_script_baseline.sh
git add -f launch/eval_script_promptctl.sh
git add -f launch/eval_leaderboard_slurm.sh

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
