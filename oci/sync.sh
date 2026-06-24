#!/bin/bash
set -e

REMOTE_HOST="draco-oci-login-01.draco-oci-iad.nvidia.com"
REMOTE_USER="hainanx"
SSH_KEY="$HOME/.ssh/draco-rno"
# UNIQUE per-branch launcher dir -- does NOT collide with older OCI scripts or
# the shared .../hainanx/NeMo code checkout (that is synced by sync_to_oci.sh).
REMOTE_DIR="/lustre/fsw/portfolios/llmservice/users/hainanx/oci_speechlm_d146"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no"
RSYNC_SSH="ssh $SSH_OPTS"

# Make sure the remote dir exists so a first-time push/diff doesn't fail.
ensure_remote_dir() {
    ssh $SSH_OPTS "$REMOTE_USER@$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"
}

EXCLUDE=(
    --exclude='.git/'
    --exclude='sync.sh'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='.nemo/'
    --exclude='*.swp'
    --exclude='*.swo'
    --exclude='slurm_out/'
)

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  pull        Pull remote -> local (dry-run first, then confirm)
  push        Push local -> remote (dry-run first, then confirm)
  pull -y     Pull without confirmation
  push -y     Push without confirmation
  status      Show what differs between local and remote
  watch       Watch local for changes and auto-push (requires inotifywait)

Remote: $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR
Local:  $LOCAL_DIR
EOF
    exit 1
}

do_rsync() {
    local direction="$1"
    local dry_run="$2"
    local flags=(-avz --delete "${EXCLUDE[@]}" -e "$RSYNC_SSH")

    if [ "$dry_run" = "true" ]; then
        flags+=(--dry-run --itemize-changes)
    fi

    if [ "$direction" = "pull" ]; then
        rsync "${flags[@]}" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"
    elif [ "$direction" = "push" ]; then
        rsync "${flags[@]}" "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"
    fi
}

show_diffs() {
    local tmpdir=$(mktemp -d)
    trap "rm -rf $tmpdir" RETURN

    rsync -avz "${EXCLUDE[@]}" -e "$RSYNC_SSH" \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$tmpdir/" 2>/dev/null || true

    local all_files=$(cd "$LOCAL_DIR" && find . -type f ! -name sync.sh | sort)
    local remote_files=$(cd "$tmpdir" && find . -type f 2>/dev/null | sort)
    local merged=$(echo -e "$all_files\n$remote_files" | sort -u)

    for f in $merged; do
        local local_f="$LOCAL_DIR/$f"
        local remote_f="$tmpdir/$f"
        if [ ! -f "$remote_f" ]; then
            echo "[NEW]     $f"
        elif [ ! -f "$local_f" ]; then
            echo "[REMOTE ONLY] $f"
        elif diff -q "$local_f" "$remote_f" >/dev/null 2>&1; then
            echo "[same]    $f"
        else
            echo "[CHANGED] $f"
            diff --color=auto -u "$remote_f" "$local_f" \
                --label "remote: $f" --label "local: $f" || true
            echo ""
        fi
    done
}

confirm() {
    echo ""
    read -rp "Proceed with $1? [y/N] " answer
    case "$answer" in
        [yY]*) return 0 ;;
        *) echo "Aborted."; exit 0 ;;
    esac
}

cmd_pull() {
    local auto_yes="$1"
    ensure_remote_dir
    echo "=== DRY RUN: pull (remote -> local) ==="
    do_rsync pull true
    echo ""
    echo "=== Diffs ==="
    show_diffs
    if [ "$auto_yes" = "-y" ]; then
        echo "=== Pulling... ==="
        do_rsync pull false
    else
        confirm "pull"
        do_rsync pull false
    fi
    echo "=== Pull complete ==="
}

cmd_push() {
    local auto_yes="$1"
    ensure_remote_dir
    echo "=== DRY RUN: push (local -> remote) ==="
    do_rsync push true
    echo ""
    echo "=== Diffs ==="
    show_diffs
    if [ "$auto_yes" = "-y" ]; then
        echo "=== Pushing... ==="
        do_rsync push false
    else
        confirm "push"
        do_rsync push false
    fi
    echo "=== Push complete ==="
}

cmd_status() {
    ensure_remote_dir
    echo "=== Differences (remote -> local) ==="
    do_rsync pull true
    echo ""
    echo "=== Diffs ==="
    show_diffs
}

cmd_watch() {
    if ! command -v inotifywait &>/dev/null; then
        echo "inotifywait not found. Install with: sudo apt install inotify-tools"
        exit 1
    fi
    echo "Watching $LOCAL_DIR for changes, auto-pushing to remote..."
    while inotifywait -r -e modify,create,delete,move "$LOCAL_DIR" --exclude 'sync\.sh|__pycache__|\.pyc|\.swp'; do
        echo "[$(date)] Change detected, pushing..."
        do_rsync push false
        echo "[$(date)] Push complete."
    done
}

case "${1:-}" in
    pull)   cmd_pull "$2" ;;
    push)   cmd_push "$2" ;;
    status) cmd_status ;;
    watch)  cmd_watch ;;
    *)      usage ;;
esac
