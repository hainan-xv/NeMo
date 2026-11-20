#!/bin/bash
# ============================================================================
# Snapshot every SpeechlmCmp arm's CURRENT top-5 and submit both evals for each.
#
# Run on the login node: it only hard-links and submits.
#
# WHY IT WIPES THE SNAPSHOT FIRST. Two caches would otherwise make this
# silently re-score an OLD model:
#   1. Re-linking into an existing snapshot ACCUMULATES checkpoints, so the
#      average would span every round ever taken, not this round's top-5.
#   2. eval_leaderboard.sh caches <exp>/averaged/top5-averaged.nemo and reuses
#      it unless FORCE_AVERAGE=1 -- so yesterday's averaged weights would be
#      evaluated against today's label.
# Deleting the whole <arm>_lb tree kills both. It holds nothing but hard links
# and derived artifacts; the training directory is never touched.
#
# WHY LB THEN MCV WITH afterok. Both runs average the same checkpoints into the
# same cached path; submitted together they race. Serialised, the MCV run reuses
# the leaderboard run's average, so the two numbers come from identical weights.
# ============================================================================
set -euo pipefail
P=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/hainanx
B="$P/results/SpeechlmCmp"
ARMS="${ARMS:-cmp_chat cmp_chat_banded cmp_chat_fa1k cmp_chat_banded1k cmp_chat_rnnt1k cmp_script cmp_script_band cmp_script_bandchunk}"

cd "$P/NeMo_SCRIPT_cc"
for a in $ARMS; do
    SRC="$B/$a/$a/checkpoints"
    DST_ROOT="$B/${a}_lb"
    case "$DST_ROOT" in *_lb) ;; *) echo "REFUSING to delete $DST_ROOT (not a _lb snapshot)" >&2; exit 1;; esac

    mapfile -t CK < <(ls -1 "$SRC"/*.ckpt 2>/dev/null | grep -v -- '-last.ckpt' | grep -v -- '-averaged.ckpt' || true)
    if [[ ${#CK[@]} -eq 0 ]]; then echo "SKIP $a: no checkpoints at $SRC" >&2; continue; fi

    rm -rf "$DST_ROOT"
    mkdir -p "$DST_ROOT/${a}_lb/checkpoints"
    for f in "${CK[@]}"; do ln -f "$f" "$DST_ROOT/${a}_lb/checkpoints/$(basename "$f")"; done

    LB=$(sbatch --parsable "launch/dfw_eval_${a}.sh")
    MCV=$(sbatch --parsable --dependency=afterok:"$LB" "launch/dfw_evalmcv_${a}.sh")
    printf '%-24s %d ckpts  lb=%s  mcv=%s\n' "$a" "${#CK[@]}" "$LB" "$MCV"
done
