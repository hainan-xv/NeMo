#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:stage-leaderboard-cache
#SBATCH -p interactive,batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH --time-min 01:00:00
# Modest RAM: staging is CPU/network-bound, and requesting 1 GPU caps RAM (the
# scheduler rejects --mem=0 here as it would strand the node's other 7 GPUs).
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# One-shot staging of the Open ASR Leaderboard test sets into the on-disk cache
# our eval reads. Downloads each config from the consolidated hub dataset
# `hf-audio/open-asr-leaderboard` and materializes 16 kHz mono wavs plus a
# `_cache_manifest.jsonl` per split in the layout speechlm_leaderboard_eval.py
# expects:
#     <CACHE_DIR>/<dataset>/<split>/_cache_manifest.jsonl + 000000.wav ...
#
# Run this ONCE to populate the shared lustre cache; the leaderboard eval
# (launch/eval_leaderboard.sh) then runs fully OFFLINE off that cache. Idempotent:
# already-staged splits are skipped (REFRESH=1 forces a rebuild).
#
# NOTE: unlike the eval, this step needs INTERNET + a valid HF token (the dataset
# is gated -- accept its terms once on the Hub with the token's account). It is
# CPU/network-bound; the 1 GPU request only satisfies the GPU-only partitions.
#
# Usage (from the clean repo root on OCI, AFTER ./sync_to_oci.sh):
#   MAX_SAMPLES=10 sbatch launch/stage_leaderboard_cache.sh   # smoke test (10 utts/split)
#   sbatch launch/stage_leaderboard_cache.sh                  # full stage
#   REFRESH=1 sbatch launch/stage_leaderboard_cache.sh        # re-stage everything
#
# Key env:
#   CACHE_DIR      cache root to populate (MUST match the eval's CACHE_DIR;
#                  default /lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache)
#   DATASETS       space/comma 'name:split' list (default = current public suite)
#   DATASET_PATH   hub dataset to pull from (default hf-audio/open-asr-leaderboard)
#   MAX_SAMPLES    cap utts per split (0 = all; e.g. 10 for a smoke test)
#   REFRESH=1      re-stage even splits already cached
#   HF_ENDPOINT    optional HF mirror endpoint
#   HF_TOKEN       gated-dataset token; if exported it is used directly (sbatch
#                  forwards the submitting env), else read from HF_TOKEN_FILE
#   HF_TOKEN_FILE  path to a one-line token file (default ~/.hf_token). Put it on
#                  lustre if the compute node does not mount your home dir.
# ============================================================================

read_optional_token() { [[ -r "$1" ]] && tr -d '\r\n' < "$1" || true; }
# Token resolution order (the batch body runs on a COMPUTE node, where $HOME may
# not be mounted the same as the login node, so an on-disk file can be invisible):
#   1) an HF_TOKEN already exported into the environment (sbatch forwards the
#      submitting env by default, so `export HF_TOKEN=...; sbatch ...` works);
#   2) an explicit ${HF_TOKEN_FILE} (put it on lustre to guarantee it's mounted);
#   3) the standard ~/.hf_token used by the other launch scripts.
if [[ -z "${HF_TOKEN:-}" ]]; then
    HF_TOKEN="$(read_optional_token "${HF_TOKEN_FILE:-$HOME/.hf_token}")"
fi
if [[ -z "${HF_TOKEN}" ]]; then
    echo "ERROR: no HF token available -- staging needs it to download the gated dataset." >&2
    echo "Fix with ONE of (from the OCI login node):" >&2
    echo "  A) export HF_TOKEN=hf_xxx && sbatch launch/stage_leaderboard_cache.sh" >&2
    echo "  B) printf %s hf_xxx > ~/.hf_token && chmod 600 ~/.hf_token   # if home is mounted on compute" >&2
    echo "  C) put the token on lustre and point at it:" >&2
    echo "     HF_TOKEN_FILE=/lustre/.../.hf_token sbatch launch/stage_leaderboard_cache.sh" >&2
    exit 1
fi

mkdir -p slurm_out

DATASET_PATH="${DATASET_PATH:-hf-audio/open-asr-leaderboard}"
DATASETS="${DATASETS:-librispeech:test.clean librispeech:test.other ami_cleaned:test earnings22:test gigaspeech_cleaned:test spgispeech:test voxpopuli_cleaned_aa:test}"
DATASETS_CSV="$(echo "$DATASETS" | tr -s ' ' ',')"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
REFRESH="${REFRESH:-0}"
REFRESH_FLAG=""
[[ "$REFRESH" == 1 || "$REFRESH" == true ]] && REFRESH_FLAG="--refresh"

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
# The clean SCRIPT repo, git-synced via sync_to_oci.sh -> mounted as /code.
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_SCRIPT_cc}"
# Cache root to populate (keep in sync with launch/eval_leaderboard.sh CACHE_DIR).
CACHE_DIR="${CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache}"
# Writable HF cache on lustre (home is often read-only on compute nodes).
HFCACHE="${HFCACHE:-${OUTPUT_PREFIX}/hf_cache}"
mkdir -p "$HFCACHE" "$CACHE_DIR"

OUTFILE=${OUTPUT_PREFIX}/results/stage_leaderboard_cache-%j-%n.out
ERRFILE=${OUTPUT_PREFIX}/results/stage_leaderboard_cache-%j-%n.err
mkdir -p "${OUTPUT_PREFIX}/results"
# Mount each needed lustre leaf DIRECTLY (source==target) instead of relying on a
# broad /lustre/fsw bind. /lustre/fsw is an autofs tree: its sub-paths are mounted
# lazily in the HOST namespace, but the container gets a private mount namespace,
# so those sub-mounts do NOT appear inside the container under a broad bind (this
# is why an earlier run's cmd file at /lustre/.../results was "No such file").
# A direct bind of the exact dir forces autofs to resolve it at mount time -- the
# same reason /code and /hfcache already work. CACHE_DIR is bound at its real path
# so writes land on lustre and eval later reads the same location.
MOUNTS="--container-mounts=${CODE_DIR}:/code,${CACHE_DIR}:${CACHE_DIR},${HFCACHE}:/hfcache/"

HF_ENDPOINT_EXPORT=""
[[ -n "${HF_ENDPOINT:-}" ]] && HF_ENDPOINT_EXPORT="export HF_ENDPOINT='${HF_ENDPOINT}'; "

echo "==> Staging leaderboard cache"
echo "    dataset_path: ${DATASET_PATH}"
echo "    datasets:     ${DATASETS_CSV}"
echo "    cache_dir:    ${CACHE_DIR}"
echo "    max_samples:  ${MAX_SAMPLES}  refresh:${REFRESH}"

read -r -d '' cmd <<EOF || true
echo "*******Staging Open ASR Leaderboard cache********" \
&& cd /code \
&& git rev-parse HEAD 2>/dev/null || true \
&& export HF_TOKEN=${HF_TOKEN} \
&& export HF_HOME=/hfcache/ \
&& export HF_HUB_ENABLE_HF_TRANSFER=0 \
&& export TOKENIZERS_PARALLELISM=false \
&& ${HF_ENDPOINT_EXPORT}python -c "import datasets, soundfile; print('datasets', datasets.__version__)" \
&& python /code/scripts/stage_leaderboard_cache.py \
      --cache_dir "${CACHE_DIR}" \
      --dataset_path "${DATASET_PATH}" \
      --datasets "${DATASETS_CSV}" \
      --max_samples ${MAX_SAMPLES} \
      ${REFRESH_FLAG} \
&& echo "==> Staging complete: ${CACHE_DIR}"
EOF

# Run via a script file so any quoting in dataset names cannot break the shell.
# Write it INSIDE the checkout (bind-mounted directly at /code) so the container
# can always open it -- a path under the broad /lustre/fsw tree may be invisible
# in the container (autofs; see MOUNTS note above).
mkdir -p "${CODE_DIR}/slurm_out"
CMD_BASENAME="stage_cache_cmd_${SLURM_JOB_ID:-local$$}.sh"
printf '%s\n' "$cmd" > "${CODE_DIR}/slurm_out/${CMD_BASENAME}"
chmod +x "${CODE_DIR}/slurm_out/${CMD_BASENAME}"

srun -o "$OUTFILE" -e "$ERRFILE" --container-image="$CONTAINER" $MOUNTS bash "/code/slurm_out/${CMD_BASENAME}"
