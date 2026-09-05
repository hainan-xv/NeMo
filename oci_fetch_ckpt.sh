#!/bin/bash
# ============================================================================
# Copy a training checkpoint from the OCI grid to this machine.
#
#   ./oci_fetch_ckpt.sh <exp_name> [dest_dir]      # best val_wer checkpoint
#   ./oci_fetch_ckpt.sh <exp_name> --list          # just show what is available
#   CKPT=step=16000-val_wer=0.0953.ckpt ./oci_fetch_ckpt.sh <exp_name>
#
# Picks the checkpoint with the LOWEST val_wer by default, rather than the
# newest: with save_top_k the newest is often not the best, and `-last.ckpt`
# carries whatever the metric was at the moment training stopped -- which for a
# cancelled job can be a step where validation had never run.
#
# These files are ~7 GB because they carry optimizer state (save_weights_only is
# false, so training can resume). Inference needs only the weights, so the copy
# is worth doing once and reusing.
# ============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ./oci_env.sh

EXP="${1:-}"
DEST="${2:-$HOME/checkpoints_chat}"
PROJECT="${PROJECT:-SpeechlmScriptCC}"
REMOTE_ROOT="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}/results/${PROJECT}"

if [[ -z "$EXP" ]]; then
    echo "usage: ./oci_fetch_ckpt.sh <exp_name> [dest_dir|--list]" >&2
    echo "" >&2
    echo "experiments on the grid:" >&2
    oci_ssh "ls -1 '${REMOTE_ROOT}' 2>/dev/null" | sed 's/^/  /' >&2
    exit 1
fi

CKPT_DIR="${REMOTE_ROOT}/${EXP}/${EXP}/checkpoints"
listing="$(oci_ssh "ls -1 '${CKPT_DIR}'/*.ckpt 2>/dev/null | xargs -r -n1 basename")"
if [[ -z "$listing" ]]; then
    echo "ERROR: no checkpoints under ${CKPT_DIR}" >&2
    exit 1
fi

if [[ "$DEST" == "--list" ]]; then
    echo "checkpoints for ${EXP}:"
    echo "$listing" | sed 's/^/  /'
    exit 0
fi

if [[ -n "${CKPT:-}" ]]; then
    name="$CKPT"
else
    # Lowest val_wer, ignoring -last (its metric may predate any validation).
    name="$(echo "$listing" | grep -v -- '-last\.ckpt$' | grep -E 'val_wer=[0-9]+\.[0-9]+' \
            | sed -E 's/.*val_wer=([0-9.]+)\.ckpt/\1 &/' | sort -g | head -1 | cut -d' ' -f2-)"
    [[ -z "$name" ]] && name="$(echo "$listing" | tail -1)"
fi

mkdir -p "$DEST"
out="${DEST}/${EXP}__${name}"
echo "==> ${EXP}"
echo "    remote: ${CKPT_DIR}/${name}"
echo "    local:  ${out}"
if [[ -s "$out" ]]; then
    echo "    already present ($(du -h "$out" | cut -f1)); delete it to re-fetch."
    exit 0
fi
echo "    (~7 GB, includes optimizer state)"
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no "${OCI_USER}@${OCI_HOST}:${CKPT_DIR}/${name}" "$out"
echo "==> done: $(du -h "$out" | cut -f1)"
