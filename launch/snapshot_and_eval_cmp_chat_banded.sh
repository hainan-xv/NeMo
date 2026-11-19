#!/bin/bash
# ============================================================================
# Snapshot cmp_chat_banded's checkpoints and submit BOTH evals.
#
# Not an sbatch script -- run it on the login node; it only hard-links and
# submits. Separate from the two eval launchers so they stay bare-sbatch-able.
#
# WHY A SNAPSHOT: the arm is still training with save_top_k=5, so its checkpoint
# set rotates, and the eval harness also WRITES <exp>-averaged.ckpt back into
# the directory it reads. Hard links cost no disk (same filesystem).
#
# WHY SERIALIZED: both evals average the same top-5 into the same cached path.
# Submitted together they race; the MCV run therefore waits on the leaderboard
# run with afterok and reuses the cached average, which also makes the two
# numbers come from a byte-identical model.
# ============================================================================
set -euo pipefail
P=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/hainanx
SRC="$P/results/SpeechlmCmp/cmp_chat_banded/cmp_chat_banded/checkpoints"
DST="$P/results/SpeechlmCmp/cmp_chat_banded_lb/cmp_chat_banded_lb/checkpoints"
MIN_EPOCH="${MIN_EPOCH:-80}"

mapfile -t CK < <(ls -1 "$SRC"/*.ckpt 2>/dev/null | grep -v -- '-last.ckpt' | grep -v -- '-averaged.ckpt' || true)
[[ ${#CK[@]} -ge 1 ]] || { echo "ERROR: no checkpoints at $SRC" >&2; exit 1; }

TOP_EPOCH=$(printf '%s\n' "${CK[@]}" | grep -oE 'epoch=[0-9]+' | cut -d= -f2 | sort -n | tail -1)
echo "==> cmp_chat_banded: ${#CK[@]} checkpoints, highest epoch=${TOP_EPOCH} (threshold ${MIN_EPOCH})"
if [[ "${TOP_EPOCH:-0}" -lt "$MIN_EPOCH" ]]; then
    echo "NOT READY: epoch ${TOP_EPOCH} < ${MIN_EPOCH}. Re-run later, or override with MIN_EPOCH=<n>." >&2
    exit 2
fi

mkdir -p "$DST"
for f in "${CK[@]}"; do ln -f "$f" "$DST/$(basename "$f")"; echo "    linked $(basename "$f")"; done

cd "$P/NeMo_SCRIPT_cc"
LB=$(sbatch --parsable launch/dfw_eval_cmp_chat_banded.sh)
echo "==> leaderboard eval: $LB"
MCV=$(sbatch --parsable --dependency=afterok:"$LB" launch/dfw_evalmcv_cmp_chat_banded.sh)
echo "==> MCV11-dev eval:   $MCV  (waits on $LB, reuses its cached average)"
