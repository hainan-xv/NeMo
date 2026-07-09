#!/bin/bash
# Auto-resume helper: waits until OCI connectivity returns (VPN/DNS), then
#   1) pushes the updated launcher (oci/sync.sh push -y)
#   2) updates the OCI /code checkout from GitHub (https)
#   3) scancels the old CHAT CE job and relaunches
# Safe to re-run; it only relaunches once it can reach OCI.
set -uo pipefail

HOST="draco-oci-login-01.draco-oci-iad.nvidia.com"
USER_="hainanx"
KEY="$HOME/.ssh/draco-rno"
REPO_LOCAL="/home/hainanx/Workplace/NeMo_ord_sync_d146_current"
OCI_REPO="/lustre/fsw/portfolios/llmservice/users/hainanx/NeMo_ord_sync_d146_current"
OCI_LAUNCH_DIR="/lustre/fsw/portfolios/llmservice/users/hainanx/oci_speechlm_d146"
BRANCH="ord_sync_d146_current"
GITHUB="https://github.com/hainan-xv/NeMo.git"
OLD_JOB="10799525"

SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15"

echo "[finish] waiting for OCI connectivity to $HOST ..."
tries=0
until getent hosts "$HOST" >/dev/null 2>&1 && $SSH "$USER_@$HOST" 'true' 2>/dev/null; do
    tries=$((tries+1))
    if [ $((tries % 10)) -eq 0 ]; then echo "[finish] still waiting ($tries) ..."; fi
    sleep 30
done
echo "[finish] CONNECTIVITY_RESTORED"

echo "[finish] pushing launcher via oci/sync.sh"
( cd "$REPO_LOCAL/oci" && ./sync.sh push -y ) || { echo "[finish] launcher push FAILED"; exit 1; }

echo "[finish] updating OCI /code checkout + scancel + relaunch"
$SSH "$USER_@$HOST" "
    set -e
    cd '$OCI_REPO'
    rm -f .git/index.lock .git/FETCH_HEAD 2>/dev/null || true
    git fetch -f '$GITHUB' +$BRANCH:refs/tmp/oci_sync_head
    git checkout -f '$BRANCH'
    git reset --hard refs/tmp/oci_sync_head
    echo '=== OCI HEAD ==='; git rev-parse HEAD
    echo '=== stats line present? ==='
    grep -n 'stats: loss=%.4f  train_wer=%.4f  best_val_wer' nemo/collections/asr/models/rnnt_models.py || echo 'MISSING_STATS_LINE'
    echo '=== scancel old job $OLD_JOB ==='
    scancel $OLD_JOB 2>/dev/null || true
    sleep 3
    cd '$OCI_LAUNCH_DIR'
    echo '=== relaunch ==='
    sbatch chat_extaligner_ce_granary.sh
"
echo "[finish] DONE"
