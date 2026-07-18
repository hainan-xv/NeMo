#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:check-spaces-granary2
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 01:00:00            # wall time
#SBATCH --time-min 00:30:00
#SBATCH --overcommit
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Scan a dataset's transcripts for whitespace problems (esp. runs of 2+ spaces,
# which SentencePiece encodes as spurious word-boundary tokens) on the grid.
#
# Runs scripts/check_multiple_spaces.py (MY code at /code) inside the training
# container on ONE node of the interactive partition. It reads the SAME lhotse
# input_cfg training uses (Granary 2.0 by default) and iterates cut metadata only
# (no audio is fetched), so it is light -- a single interactive node is plenty.
#
# Submit from an OCI login node:
#   ./sync_to_oci.sh
#   sbatch oci_chat/check_spaces.sh
#   # scan everything (slower, streams all metadata shards):
#   MAX=0 sbatch oci_chat/check_spaces.sh
#   # a specific manifest / a different input_cfg:
#   INPUT_CFG=/path/to/other.yaml sbatch oci_chat/check_spaces.sh
#   MANIFEST=/data/.../mcv11_dev.json sbatch oci_chat/check_spaces.sh
# ============================================================================

# Optional AIS token (needed to read shar metadata served over AIStore).
AIS_AUTHN_TOKEN=""
if [[ -r "$HOME/.ais_authn_token" ]]; then
    AIS_AUTHN_TOKEN="$(tr -d '\r\n' < "$HOME/.ais_authn_token")"
fi

mkdir -p slurm_out
CLUSTER="oci"
SLURM_ACCOUNT='llmservice'

# Latest container with the current SpeechLM/ASR deps (we only use its
# environment; the actual NeMo code comes from /code below).
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"

# ---------------------------------------------------------------------------
# What to scan (default: the Granary 2.0 input_cfg the launchers train on).
# ---------------------------------------------------------------------------
GRANARY2_CFG=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/dongjig/aligned_amos/granary_v2_en_pnc_qwen_aligned_filtered/granary_v2_en_pnc_qwen_aligned_filtered_safe_iad_s3_audio.yaml
INPUT_CFG="${INPUT_CFG:-$GRANARY2_CFG}"
# Optional: scan a plain NeMo manifest instead of a lhotse input_cfg.
MANIFEST="${MANIFEST:-}"
# Max cuts to scan (0 = all). Default is a fast ~200k spot-check.
MAX="${MAX:-200000}"
EXAMPLES="${EXAMPLES:-30}"
TEXT_FIELD="${TEXT_FIELD:-text}"

# Choose the source argument for the checker.
if [ -n "${MANIFEST}" ]; then
    SRC_ARG="--manifest ${MANIFEST}"
    SRC_DESC="manifest=${MANIFEST}"
else
    SRC_ARG="--input_cfg ${INPUT_CFG}"
    SRC_DESC="input_cfg=${INPUT_CFG}"
fi

# ---------------------------------------------------------------------------
# Paths / mounts (same layout as oci_chat/tdt.sh).
# ---------------------------------------------------------------------------
SPEECHLM_PROJECT_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm
DATA_DIR=${SPEECHLM_PROJECT_DIR}/data
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh
HAINAN_DIR=/lustre/fsw/portfolios/llmservice/users/hainanx
# MY code (synced via sync_to_oci.sh) -> mounted as /code.
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo79/}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
RESULTS_DIR=${OUTPUT_PREFIX}/results/checks
mkdir -p ${RESULTS_DIR}
OUTFILE=${RESULTS_DIR}/check_spaces-%j.out
ERRFILE=${RESULTS_DIR}/check_spaces-%j.err

MOUNTS="--container-mounts=${SPEECHLM_PROJECT_DIR}:${SPEECHLM_PROJECT_DIR},${H_DIR}:${H_DIR},$HAINAN_DIR:$HAINAN_DIR,$CODE_DIR:/code,$RESULTS_DIR:/results,$DATA_DIR:/data,/lustre/fsw:/lustre/fsw,/lustre/fs12:/lustre/fs12"

read -r -d '' cmd <<EOF
echo "*******Whitespace scan (2+ spaces etc.) ********" \
&& echo "*** SOURCE: ${SRC_DESC} ***" \
&& echo "*** MAX=${MAX} EXAMPLES=${EXAMPLES} TEXT_FIELD=${TEXT_FIELD} ***" \
&& cd /code \
&& git rev-parse HEAD \
&& export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& python -c "import nemo, nemo.collections.common.data.lhotse; print('USING NeMo FROM:', nemo.__file__)" \
&& export HYDRA_FULL_ERROR=1 \
&& export TOKENIZERS_PARALLELISM=false \
&& export AIS_ENDPOINT=http://asr.iad.oci.aistore.nvidia.com:51080 \
&& export AIS_AUTHN_TOKEN="${AIS_AUTHN_TOKEN}" \
&& export NEMO_DATA_STORE_CACHE_DIR=/lustre/fsw/portfolios/llmservice/users/heh/nemo_cache \
&& echo "Starting whitespace scan (running MY code at /code)" \
&& python /code/scripts/check_multiple_spaces.py \
    ${SRC_ARG} \
    --text_field ${TEXT_FIELD} \
    --max ${MAX} \
    --examples ${EXAMPLES}
EOF

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"

set +x
