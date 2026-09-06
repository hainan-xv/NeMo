#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:chat-standard
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 8
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 04:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=8
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# STANDARD CHAT transducer -- the marginalised-loss control.
#
#   sbatch launch/chat_train_standard.sh
#   ./oci_launch.sh launch/chat_train_standard.sh
#
# WHAT THIS IS FOR. Our ChatSTTModel trains the PR's CHAT architecture against a
# single FORCED alignment. This trains the same architecture the ordinary way --
# EncDecRNNTBPEModel with the full RNN-T loss, summing over every alignment --
# on the same Granary 2.0 data, from the same nemotron encoder, with the same
# 1,024-piece tokenizer and the same 14-frame chunk grid. The gap between the
# two runs is therefore the cost (or benefit) of fixing the alignment, which is
# the one assumption in the forced-alignment design we have never tested.
#
# It also gives a second reading on the leaderboard-vs-val_wer inversion: this
# model has the same 7M-parameter LSTM language model as the forced-alignment
# arm, so if the inversion is about LM strength rather than the loss, this run
# should show it too.
#
# Uses the standard NeMo entry point (speech_to_text_rnnt_bpe.py), so unlike
# ChatSTTModel it produces a .nemo and works with NeMo's own averaging and
# evaluation tooling.
#
# ENV
#   MAX_STEPS, LR, WARMUP_STEPS, EPOCH_STEPS  training schedule
#   EXP_NAME                                   results directory
#   INIT_NEMO                                  encoder donor (.nemo)
# ============================================================================

read_required_token() {
    local path="$1"
    if [[ ! -r "$path" ]]; then
        echo "ERROR: required token file is missing or unreadable: $path" >&2
        exit 1
    fi
    tr -d '\r\n' < "$path"
}
WANDB="$(read_required_token "$HOME/.wandb_token")"
HF_TOKEN="$(read_required_token "$HOME/.hf_token")"
AIS_AUTHN_TOKEN=""
[[ -r "$HOME/.ais_authn_token" ]] && AIS_AUTHN_TOKEN="$(tr -d '\r\n' < "$HOME/.ais_authn_token")"

mkdir -p slurm_out

GPUS_PER_NODE=8
PROJECT_NAME=SpeechlmScriptCC
LUSTRE_ACCOUNT_PREFIX=/lustre/fsw/portfolios/llmservice
CONTAINER="/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh"

MAX_STEPS="${MAX_STEPS:-300000}"
EPOCH_STEPS="${EPOCH_STEPS:-4000}"
LR="${LR:-0.001}"
WARMUP_STEPS="${WARMUP_STEPS:-2500}"

CONFIG_PATH=/code/examples/asr/conf/fastconformer/cache_aware_streaming
CONFIG_NAME="${CONFIG_NAME:-nemotron_chat_transducer_granary2}"
EXP_NAME="${EXP_NAME:-granary2_chat_standard_asrvocab}"

# Same _n<N> guard as chat_train.sh: a scaled-down debug run must not resume from
# and then overwrite the full-scale run's checkpoints.
DESIGN_NODES="$(grep -m1 -E '^#SBATCH[[:space:]]+-N[[:space:]]+[0-9]+' "$0" | grep -oE '[0-9]+$')"
ACTUAL_NODES="${SLURM_JOB_NUM_NODES:-$DESIGN_NODES}"
if [[ "${SKIP_NODE_SUFFIX:-0}" != "1" && "$ACTUAL_NODES" -ne "$DESIGN_NODES" ]]; then
    EXP_NAME="${EXP_NAME}_n${ACTUAL_NODES}"
    echo "==> Allocation is ${ACTUAL_NODES} node(s), not ${DESIGN_NODES}; EXP_NAME -> ${EXP_NAME}"
fi

OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
RESULTS_DIR=${OUTPUT_PREFIX}/results/$PROJECT_NAME/$EXP_NAME
HFCACHE=${OUTPUT_PREFIX}/hf_cache
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_SCRIPT_cc}"
DATA_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh
DONGJI_ROOT=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/dongjig
OCI_TMP_DIR="${OCI_TMP_DIR:-/results/tmp}"

# The encoder donor. Its 1,024-piece SentencePiece model is also the tokenizer,
# extracted below -- using any other vocabulary would make this incomparable to
# the 1k forced-alignment arm.
INIT_NEMO="${INIT_NEMO:-${H_DIR}/pretrained_models/huggingface/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo}"
TOKENIZER_DIR="${RESULTS_DIR}/tokenizer"

mkdir -p "${RESULTS_DIR}" "${HFCACHE}"
OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

MOUNTS="--container-mounts=${DATA_DIR}:${DATA_DIR},${H_DIR}:${H_DIR},$CODE_DIR:/code,$RESULTS_DIR:/results,${HFCACHE}:/hfcache/,$DONGJI_ROOT:$DONGJI_ROOT"

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "*** STANDARD CHAT: RNNTAttJoint + FULL RNN-T loss (marginalised over alignments) ***" \
&& echo "*** CONTROL FOR: ChatSTTModel, same architecture trained on ONE forced alignment ***" \
&& echo "*** encoder init: ${INIT_NEMO} ***" \
&& echo "*** schedule: epoch=${EPOCH_STEPS} lr=${LR} warmup=${WARMUP_STEPS} max_steps=${MAX_STEPS} ***" \
&& nvidia-smi \
&& export WANDB_API_KEY=${WANDB} \
&& cd /code && export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& echo "CODE COMMIT:" && git rev-parse HEAD \
&& export HF_HOME="/hfcache/" HF_TOKEN=${HF_TOKEN} HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 \
&& export OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
&& export TMPDIR=${OCI_TMP_DIR} && mkdir -p ${OCI_TMP_DIR} \
&& export AIS_ENDPOINT=http://asr.iad.oci.aistore.nvidia.com:51080 AIS_AUTHN_TOKEN="${AIS_AUTHN_TOKEN}" \
&& export NEMO_DATA_STORE_CACHE_DIR=${H_DIR}/nemo_cache \
&& if [ ! -f '${TOKENIZER_DIR}/tokenizer.model' ]; then \
     echo '==> extracting the donor SentencePiece vocabulary'; \
     python -c "
import os, tarfile, shutil, sys
src, dst = '${INIT_NEMO}', '${TOKENIZER_DIR}'
os.makedirs(dst, exist_ok=True)
with tarfile.open(src, 'r:') as tf:
    for m in tf.getmembers():
        if m.name.endswith('tokenizer.model'):
            m.name = 'tokenizer.model'; tf.extract(m, dst)
        elif m.name.endswith('tokenizer.vocab'):
            m.name = 'tokenizer.vocab'; tf.extract(m, dst)
        elif m.name.endswith('vocab.txt'):
            m.name = 'vocab.txt'; tf.extract(m, dst)
assert os.path.isfile(os.path.join(dst,'tokenizer.model')), 'no tokenizer.model in ' + src
print('    tokenizer ->', dst)
"; \
   else echo '==> reusing extracted tokenizer'; fi \
&& python /code/examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    model.tokenizer.dir=${TOKENIZER_DIR} \
    model.optim.lr=${LR} \
    model.optim.sched.warmup_steps=${WARMUP_STEPS} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.limit_train_batches=${EPOCH_STEPS} \
    trainer.val_check_interval=${EPOCH_STEPS} \
    trainer.devices=${GPUS_PER_NODE} \
    trainer.num_nodes=\${SLURM_JOB_NUM_NODES} \
    +init_from_nemo_model.model0.path=${INIT_NEMO} \
    +init_from_nemo_model.model0.include=["encoder"] \
    ++exp_manager.exp_dir=/results/ \
    ++exp_manager.name=${EXP_NAME} \
    ++exp_manager.max_time_per_run=00:03:55:00 \
    ++exp_manager.create_wandb_logger=true \
    ++exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME}
EOF

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
