#!/usr/bin/env bash
# Commit the current clean SCRIPT repo, push it to GitHub, then update OCI from
# GitHub. Mirrors the workflow of the original NeMo repo's sync_to_oci.sh, but
# points at this repo's branch, fork, and OCI storage location.
set -euo pipefail

BRANCH="${BRANCH:-script}"
GITHUB_URL="${GITHUB_URL:-https://github.com/hainan-xv/NeMo.git}"
OCI_HOST="${OCI_HOST:-draco-oci-dc-03.draco-oci-iad.nvidia.com}"
OCI_USER="${OCI_USER:-hainanx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/draco-rno}"
OCI_REPO="${OCI_REPO:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_script_clean}"

cd "$(dirname "${BASH_SOURCE[0]}")"

# The local working branch (e.g. heh/streaming_speechlm_automodel) is decoupled
# from the remote branch we publish to. We always push the current HEAD to the
# remote branch $BRANCH (default: script) and check that same branch out on the
# grid, no matter what the local branch is called.
current_branch="$(git branch --show-current)"
echo "==> Local branch: ${current_branch:-<detached HEAD>}  ->  remote branch: $BRANCH"

# Stage existing tracked changes plus the project-owned launch/sync scripts.
# Avoid `git add -A`, which could accidentally include local credentials.
# launch/ is tracked normally, but new files still need an explicit add
# (git add -u only stages already-tracked paths).
git add -u
git add .gitignore sync_to_oci.sh launch/script_baseline.sh launch/script_selfcorrect.sh
# Leaderboard eval: the pooled-shard SLURM launchers + the self-contained decode
# driver and its helpers (checkpoint averaging, wandb reporting). New/untracked, so
# they need an explicit add (git add -u only stages already-tracked paths).
git add launch/eval_leaderboard.sh launch/eval_script_baseline.sh launch/eval_promptctl.sh launch/stage_leaderboard_cache.sh
# Flush model on the OLD/EARLIER 8-set suite (uncleaned + tedlium): same clean-repo
# code + flush ckpt as eval_flush_model.sh, only the DATASETS differ. Untracked -> add.
git add launch/eval_flush_model_oldsuite.sh
# Nemotron base ASR (cache-aware streaming) leaderboard eval + its old-suite wrapper.
# Decodes with NeMo's cache-aware streaming entrypoint (the offline path SIGFPEs on
# this streaming model) + the same leaderboard re-scorer. Untracked -> add.
git add launch/eval_nemotron_streaming.sh launch/eval_nemotron_nemo_oldsuite.sh
# Legacy A/B launchers: live in the clean repo (so they sync here) but run the
# PREVIOUS repo's code + model (mount OLD_CODE_DIR as /code, exec the old
# backend). Untracked, so add them explicitly.
#   eval_promptctl_legacy.sh              - old granary2_script_promptctl[_d8]
#   eval_promptctl_all_legacy.sh          - best prior model (promptctl_all), ORIGINAL 8-set suite -> 6.93
#   eval_promptctl_all_legacy_cleaned.sh  - same model, CLEANED 7-set suite -> comparable to 6.06
git add launch/eval_promptctl_legacy.sh launch/eval_promptctl_all_legacy.sh launch/eval_promptctl_all_legacy_cleaned.sh
git add scripts/speechlm_leaderboard_eval.py
# Prompt-controlled SCRIPT eval: builds the decode prompt from (chunk_size, delay,
# cap, punct) knobs so it matches the model's training render, then reuses the
# speechlm_leaderboard_eval driver unchanged.
git add scripts/speechlm_promptctl_eval.py
git add scripts/average_sslm_ckpts.py
git add scripts/eval_wandb_report.py
# Reference/sanity eval of a STANDARD NeMo ASR model (Parakeet TDT v2) through the
# SAME cache + scorer, to validate the pipeline against the public board's numbers.
git add launch/eval_parakeet_leaderboard.sh
git add scripts/parakeet_leaderboard_eval.py
# BACKEND=nemo twin of parakeet_leaderboard_eval.py: same pooled shards + generations
# schema, but decodes each GPU's shard with NeMo's stock speech_to_text_eval.py so the
# two decode backends can be compared over identical shards on the same GPUs.
git add scripts/parakeet_nemo_shard_eval.py
# Single-process/single-GPU desktop sanity variant (no sharding/pooling) to
# cross-check the grid numbers against the public board.
git add scripts/parakeet_eval_local.py
# NeMo-native reference eval: decode with NeMo's own speech_to_text_eval.py
# (single GPU, no sharding), then rescore predictions with the leaderboard scorer.
git add launch/eval_parakeet_nemo.sh
git add scripts/rescore_nemo_eval.py
# Desktop twin of eval_parakeet_nemo.sh (uv run, no container).
git add scripts/run_parakeet_nemo_local.sh
# Shared one-call log quieting used by the two local parakeet eval scripts.
git add scripts/quiet_logs.py
# Vendored Open-ASR-Leaderboard scoring (normalizer fork + kaldialign WER) so the
# eval matches the public board's numbers.
git add scripts/leaderboard_normalizer/
git add scripts/leaderboard_wer.py
# One-shot cache staging (downloads the leaderboard test sets into the on-disk
# layout the eval reads).
git add scripts/stage_leaderboard_cache.py

# The SCRIPT model itself: brand-new files added on top of the clean automodel
# base. They are untracked, so `git add -u` above does NOT stage them -- without
# these explicit adds the grid's /code cannot import ScriptSTTModel. Keep this
# list in sync with the SCRIPT source surface.
git add nemo/collections/speechlm2/models/script_model.py
git add nemo/collections/speechlm2/data/script_dataset.py
git add nemo/collections/speechlm2/parts/script.py
git add nemo/collections/speechlm2/parts/shared_audio_chunk.py
git add nemo/collections/speechlm2/parts/script_messages.py
git add examples/speechlm2/script_train.py
git add examples/speechlm2/conf/streaming_stt_granary2_lora_script*.yaml
git add tests/collections/speechlm2/test_script.py
# SCRIPT design/usage docs (new dir on this branch).
git add nemo/collections/speechlm2/docs/

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
