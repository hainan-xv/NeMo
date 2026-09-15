#!/bin/bash
# ============================================================================
# Publish local HEAD and update the DFW grid checkout.
#
#   ./sync_to_dfw.sh ["commit message"]
#
# The DFW counterpart of sync_to_oci.sh, deliberately the same shape and using
# the SAME published branch: the code is identical on both clusters, only the
# launch scripts differ. So a plain ./sync_to_oci.sh followed by this is safe --
# the second push is a no-op and only the remote checkout moves.
#
# Everything about WHICH files reach the grid is delegated to sync_to_oci.sh's
# SCRIPT_PATHS allowlist by calling it, rather than maintaining a second copy
# that would quietly drift out of date. That drift is not hypothetical: a CHAT
# config missing from the allowlist killed job 13381311 in 45 s on 8 nodes with
# a Hydra MissingConfigException.
# ============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ./dfw_env.sh

# --- Stage, commit and push via the OCI script (shared allowlist + branch) ---
# SKIP_OCI_CHECKOUT: publish the branch, but do NOT advance the OCI grid
# checkout. OCI has long-running arms whose queued jobs read the checkout at
# run time; a DFW-only change must not alter what they do.
SKIP_OCI_CHECKOUT=1 ./sync_to_oci.sh "${1:-Sync code $(date +%Y%m%d_%H%M%S)}"

# --- Update the DFW checkout over SSH ---
# Quoted heredoc: nothing expands locally; the three args carry everything.
dfw_ssh bash -s -- "$GITHUB_URL" "$BRANCH" "$DFW_REPO" <<'REMOTE'
set -euo pipefail
url="$1"; branch="$2"; repo="$3"

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
    if [[ -e "$repo" ]]; then
        backup="${repo}.pre-git.$(date +%Y%m%d_%H%M%S)"
        echo "==> $repo exists but is not a git repo; moving it to $backup"
        mv "$repo" "$backup"
    fi
    mkdir -p "$(dirname "$repo")"
    git "${git_auth[@]}" clone --branch "$branch" --single-branch "$url" "$repo"
fi

echo "DFW checkout: $(git -C "$repo" rev-parse --short HEAD)  ($repo)"
REMOTE

echo "==> DFW sync complete."
