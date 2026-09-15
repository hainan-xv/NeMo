#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-stage-leaderboard-cache
#SBATCH -p cpu_datamover,cpu
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH --time-min 01:00:00
# Modest RAM: staging is CPU/network-bound, and requesting 1 GPU caps RAM (the
# scheduler rejects --mem=0 here as it would strand the node's other 7 GPUs).
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# One-shot staging of the Open ASR Leaderboard test sets on the CW DFW cluster.
#
#   sbatch launch/dfw_stage_leaderboard_cache.sh          <- no arguments
#
# The DFW counterpart of stage_leaderboard_cache.sh. DFW does NOT share a
# filesystem with OCI -- verified -- so OCI's 43 GB leaderboard_cache is not
# reachable here and the suite has to be materialised again. Once this has run,
# every DFW eval reads it OFFLINE, exactly as on OCI.
#
# DFW differs from the OCI script only in:
#   account    nemotron_speechprod_asr
#   partition  cpu_datamover,cpu -- staging is CPU/network bound and wants no
#              GPU at all. Slurm REJECTS a GPU-less job submitted to a GPU
#              partition, and it validates the whole -p list, so mixing in
#              `interactive` fails outright. cpu_datamover has no time limit,
#              which suits a 43 GB download.
#   paths      cache, code and container all under the DFW project
#
# Confirmed before writing this: DFW login can reach huggingface.co (HTTP 200),
# which is what makes downloading here possible at all.
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

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/hainanx}"
# The clean SCRIPT repo, git-synced via sync_to_oci.sh -> mounted as /code.
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/hainanx/NeMo_SCRIPT_cc}"
# Cache root to populate (keep in sync with launch/eval_leaderboard.sh CACHE_DIR).
CACHE_DIR="${CACHE_DIR:-/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/hainanx/leaderboard_cache}"
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
