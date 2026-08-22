#!/bin/bash
# ============================================================================
# Push local SCRIPT code to the OCI grid.
#
# Git-based, not rsync: commit locally -> push to the GitHub fork -> SSH into the
# OCI login node and force-fetch + hard-reset the grid checkout to match. The
# grid copy is therefore a strict MIRROR of what you pushed; any edit made
# directly on OCI is destroyed by design.
#
# Usage:
#   ./sync_to_oci.sh                    # auto timestamp commit message
#   ./sync_to_oci.sh "fix delay logic"  # explicit message
#
# Overridable via env: BRANCH, GITHUB_URL, OCI_HOST, OCI_USER, SSH_KEY, OCI_REPO.
#
# SAFETY: this never runs `git add -A`, which could sweep in local credentials or
# scratch files. Tracked edits go in via `git add -u`; new files must match the
# SCRIPT_PATHS allowlist below. Anything untracked that is NOT staged is reported
# at the end, so a new file can never silently fail to reach the grid (a real
# failure mode of the previous setup: untracked launch scripts were missing on
# OCI while the job appeared to run fine against stale code).
# ============================================================================
set -euo pipefail

BRANCH="${BRANCH:-SCRIPT_cc}"
GITHUB_URL="${GITHUB_URL:-https://github.com/hainan-xv/NeMo.git}"
OCI_HOST="${OCI_HOST:-draco-oci-dc-03.draco-oci-iad.nvidia.com}"
OCI_USER="${OCI_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"
# Deliberately a DIFFERENT directory from the older NeMo_script_clean checkout so
# this sync cannot hard-reset that tree out from under a running job. Must match
# CODE_DIR in launch/script_baseline.sh.
OCI_REPO="${OCI_REPO:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_SCRIPT_cc}"

cd "$(dirname "${BASH_SOURCE[0]}")"

# The local working branch is decoupled from the published one: whatever HEAD is,
# it gets pushed to $BRANCH.
current_branch="$(git branch --show-current || true)"
echo "==> Local branch: ${current_branch:-<detached HEAD>}  ->  remote branch: $BRANCH"

# --- Stage: tracked edits, plus the SCRIPT surface if newly created ---
git add -u

SCRIPT_PATHS=(
    sync_to_oci.sh
    launch/
    nemo/collections/speechlm2/parts/script.py
    nemo/collections/speechlm2/parts/script_messages.py
    nemo/collections/speechlm2/data/script_dataset.py
    nemo/collections/speechlm2/models/script_model.py
    nemo/collections/speechlm2/__init__.py
    nemo/collections/speechlm2/data/__init__.py
    nemo/collections/speechlm2/models/__init__.py
    examples/speechlm2/script_train.py
    'examples/speechlm2/conf/streaming_stt_granary2_lora_script*.yaml'
    tests/collections/speechlm2/test_script.py
)
for p in "${SCRIPT_PATHS[@]}"; do
    # Globs may legitimately match nothing; a missing path is not an error.
    git add $p 2>/dev/null || true
done

# --- Commit (only if something is actually staged) ---
if ! git diff --cached --quiet; then
    message="${1:-Sync OCI code $(date +%Y%m%d_%H%M%S)}"
    git commit -m "$message"
else
    echo "==> No staged code changes to commit"
fi

# --- Warn about anything untracked that will NOT reach the grid ---
untracked="$(git ls-files --others --exclude-standard)"
if [[ -n "$untracked" ]]; then
    echo ""
    echo "WARNING: these files are untracked and were NOT synced to OCI:"
    printf '  %s\n' $untracked
    echo "  (add them to SCRIPT_PATHS in this script, or 'git add' them, if the grid needs them)"
    echo ""
fi

# --- Push ---
echo "==> Pushing HEAD to $BRANCH"
git push "$GITHUB_URL" "HEAD:$BRANCH"

# --- Update the grid checkout over SSH ---
# Quoted heredoc: nothing expands locally; the three args carry everything.
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "${OCI_USER}@${OCI_HOST}" bash -s -- "$GITHUB_URL" "$BRANCH" "$OCI_REPO" <<'REMOTE'
set -euo pipefail
url="$1"; branch="$2"; repo="$3"

# Private-repo auth, if configured. The token stays on OCI and is passed as a
# transient header — never written into the repo's git config.
git_auth=()
if [[ -r "$HOME/.github_token" ]]; then
    github_token="$(tr -d '\r\n' < "$HOME/.github_token")"
    basic_auth="$(printf 'x-access-token:%s' "$github_token" | base64 | tr -d '\r\n')"
    git_auth=(-c "http.extraHeader=Authorization: Basic $basic_auth")
    unset github_token
fi

if [[ -d "$repo/.git" ]]; then
    # A killed job can leave stale lock files behind and wedge every later fetch.
    find "$repo/.git" -name '*.lock' -type f -delete 2>/dev/null || true
    git -C "$repo" "${git_auth[@]}" fetch --force "$url" "$branch:refs/remotes/github/$branch"
    git -C "$repo" checkout -B "$branch" "refs/remotes/github/$branch"
    git -C "$repo" reset --hard "refs/remotes/github/$branch"
else
    # Never clobber a non-git directory that happens to be in the way.
    if [[ -e "$repo" ]]; then
        backup="${repo}.pre-git.$(date +%Y%m%d_%H%M%S)"
        echo "==> $repo exists but is not a git repo; moving it to $backup"
        mv "$repo" "$backup"
    fi
    mkdir -p "$(dirname "$repo")"
    git "${git_auth[@]}" clone --branch "$branch" --single-branch "$url" "$repo"
fi

echo "OCI checkout: $(git -C "$repo" rev-parse --short HEAD)  ($repo)"
REMOTE

echo "==> Sync complete."
