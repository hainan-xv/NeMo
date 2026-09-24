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

cd "$(dirname "${BASH_SOURCE[0]}")"
# Host / key / grid path live in one place, shared with oci_launch.sh.
source ./oci_env.sh

# The local working branch is decoupled from the published one: whatever HEAD is,
# it gets pushed to $BRANCH.
current_branch="$(git branch --show-current || true)"
echo "==> Local branch: ${current_branch:-<detached HEAD>}  ->  remote branch: $BRANCH"

# --- Stage: tracked edits, plus the SCRIPT surface if newly created ---
git add -u

SCRIPT_PATHS=(
    sync_to_oci.sh
    oci_env.sh
    oci_launch.sh
    oci_launch_interactive.sh
    dfw_env.sh
    sync_to_dfw.sh
    eval_sweep.sh
    eval_all.sh
    eval_promptctl_sweep.sh
    eval_readwrite.sh
    launch/
    'nemo/collections/speechlm2/parts/script*.py'
    nemo/collections/speechlm2/data/script_dataset.py
    nemo/collections/speechlm2/models/script_model.py
    nemo/collections/speechlm2/__init__.py
    nemo/collections/speechlm2/data/__init__.py
    nemo/collections/speechlm2/models/__init__.py
    examples/speechlm2/script_train.py
    'examples/speechlm2/conf/streaming_stt_granary2_lora_script*.yaml'
    # CHAT recipes live under the ASR collection, not speechlm2. Without this
    # a NEW chat config is untracked, silently skipped, and the job dies at
    # startup with Hydra's MissingConfigException -- which is exactly how
    # granary2_chat_banded1_qwenvocab_win28_lr1e4 (job 13381311) failed in 45 s
    # on 8 nodes. The existing chat YAMLs only reach the grid because they are
    # tracked in git.
    'examples/asr/conf/fastconformer/cache_aware_streaming/*.yaml'
    examples/asr/asr_transducer/speech_to_text_chat_bpe.py
    'nemo/collections/asr/models/chat_bpe_models.py'
    'nemo/collections/asr/parts/utils/chat_alignment.py'
    'nemo/collections/asr/losses/banded_rnnt.py'
    # The multi-vocab model and the banded CUDA kernels it trains with.
    # Both were tracked by hand and NOT listed here, so every later edit
    # had to be scp'd -- the exact drift the CHAT-config note above warns
    # about, just with a slower failure mode.
    nemo/collections/asr/models/chat_multivocab_models.py
    nemo/collections/asr/models/__init__.py
    'nemo/collections/asr/parts/numba/banded_rnnt/*.py'
    scripts/chat_multivocab_smoke.py
    scripts/chat_multivocab_joint_decode_check.py
    scripts/chat_band_layout_check.py
    'nemo/collections/asr/parts/submodules/multivocab_joint_decoding.py'
    tests/collections/speechlm2/test_script.py
    scripts/script_leaderboard_eval.py
    scripts/script_asr_shim.py
    scripts/speechlm_asr_shim.py
    scripts/merge_shard_manifests.py
    scripts/build_chat_spe_corpus.py
    'scripts/tokenizers/process_asr_text_tokenizer.py'
    scripts/nemotron_leaderboard_eval.py
    scripts/speechlm_leaderboard_eval.py
    scripts/leaderboard_common.py
    scripts/run_eval_shards.sh
    scripts/analyze_eval_errors.py
    scripts/flex_attention_spike.py
    scripts/average_script_ckpts.py
    scripts/leaderboard_wer.py
    scripts/leaderboard_normalizer/
    scripts/stage_leaderboard_cache.py
    scripts/eval_wandb_report.py
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
# SKIP_OCI_CHECKOUT=1 stages, commits and pushes but leaves the OCI grid checkout
# where it is. sync_to_dfw.sh sets this: the two clusters share a branch, but a
# queued OCI job reads its config from the checkout at RUN time, so advancing it
# silently changes what an already-submitted OCI run will do. That is wanted when
# fixing an OCI arm and unwanted when the change is for DFW only.
if [[ "${SKIP_OCI_CHECKOUT:-0}" == "1" ]]; then
    echo "==> SKIP_OCI_CHECKOUT=1; leaving the OCI checkout untouched"
    echo "==> Push complete."
    exit 0
fi

# Quoted heredoc: nothing expands locally; the three args carry everything.
oci_ssh bash -s -- "$GITHUB_URL" "$BRANCH" "$OCI_REPO" <<'REMOTE'
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
