#!/bin/bash
# ============================================================================
# Shared body for the CHAT arms on the CW DFW cluster. NEVER launched directly --
# each arm has its own wrapper (dfw_chat_rnnt.sh, dfw_chat_forced.sh) that names
# its settings and is `sbatch`-able with no arguments.
#
# The DFW counterpart of chat_train.sh. That script is NOT reused because it is
# threaded with OCI-specific paths (H_DIR under llmservice/users/heh, DONGJI_ROOT,
# a per-user HAINAN_DIR, an OCI aistore endpoint) and refactoring it to be
# cluster-agnostic would touch the script four live OCI arms requeue into. The
# duplication is deliberate and contained.
#
# WHAT DIFFERS FROM OCI, and why
#   data        DFW has exactly ONE granary input_cfg, and it is the same one the
#               SCRIPT arms use. The OCI CHAT arms read
#               granary_v2_en_pnc_qwen_aligned_filtered under users/dongjig, which
#               does NOT exist here. DFW's manifests are
#               granary_v2_en_tn_raw_qwen_aligned instead.
#               *** DFW CHAT NUMBERS ARE THEREFORE NOT DIRECTLY COMPARABLE TO THE
#               OCI CHAT NUMBERS. *** They ARE comparable to each other and to the
#               DFW SCRIPT arms, which read the same data.
#               Verified present in these manifests: an `alignments` field with
#               per-word start/end times, which the forced and banded losses
#               require and without which they train on nothing.
#   text_field  left UNSET, exactly as on OCI. The granary cuts carry their
#               transcript in the default field; setting it to "answer" once made
#               every training target empty, the RNN-T loss went to zero within a
#               few steps by emitting blanks, and training WER was NaN.
#   mounts      identity mounts of the two project roots the input_cfg names, so
#               its absolute lustre paths resolve identically inside the
#               container. Mounting only one of them is what failed job 18615569.
#   aistore     NOT configured. The OCI body points at an IAD aistore endpoint
#               that does not serve this cluster; DFW reads tarred audio straight
#               off lustre.
#
# ENV (set by the wrapper)
#   LOSS_TYPE            rnnt | forced_alignment | banded
#   EXP_NAME             experiment name, also the wandb run name
#   CHUNK_SIZES          encoder chunk size; fixed 14 for every current CHAT arm
#   DELAY_FRAMES / RECOVER_WORDS / HISTORY_CHUNKS / BAND_CHUNKS
#   TARGET_CONSTRUCTION / DELAY_PUNCT
#   LR / WARMUP_STEPS / MAX_STEPS / EPOCH_STEPS
#   CONFIG_NAME / TOKENIZER_DIR / INIT_EXCLUDE
# ============================================================================
set -uo pipefail

mkdir -p slurm_out

GPUS_PER_NODE=8
LUSTRE=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
HEH=${LUSTRE}/users/heh
MYDIR=${LUSTRE}/hainanx
DATA_ROOT=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm

# New wandb project for the DFW era, separate from OCI's SpeechlmScriptCC. It
# also separates the results tree, so a DFW run can never be confused with -- or
# resumed from -- an OCI one of the same arm name.
PROJECT_NAME="${PROJECT_NAME:-SpeechlmDFW}"

CONTAINER="${CONTAINER:-${HEH}/containers/nemo-26.02-streaming-speechlm.sqsh}"
CODE_DIR="${CODE_DIR:-${MYDIR}/NeMo_SCRIPT_cc}"
CONFIG_PATH=/code/examples/asr/conf/fastconformer/cache_aware_streaming
CONFIG_NAME="${CONFIG_NAME:-nemotron_chat_transducer_granary2_qwen}"

LOSS_TYPE="${LOSS_TYPE:-rnnt}"
DELAY_FRAMES="${DELAY_FRAMES:-0}"
RECOVER_WORDS="${RECOVER_WORDS:-0}"
HISTORY_CHUNKS="${HISTORY_CHUNKS:-0}"
MAX_DELAY_FRAMES="${MAX_DELAY_FRAMES:-0}"
BAND_CHUNKS="${BAND_CHUNKS:-1}"
BAND_SIDE="${BAND_SIDE:-later}"
DELAY_PUNCT="${DELAY_PUNCT:-true}"
TARGET_CONSTRUCTION="${TARGET_CONSTRUCTION:-partition}"
INFER_DELAY_FRAMES="${INFER_DELAY_FRAMES:-null}"

LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
MAX_STEPS="${MAX_STEPS:-500000}"
EPOCH_STEPS="${EPOCH_STEPS:-2000}"
# 4, NOT the config's 8. The reference DFW recipe flags this explicitly ("note the
# reduction in num_workers to 4"), and it is not cosmetic: with 8 the CHAT arms
# pass validation and then hang FOREVER entering the training dataloader --
# jobs 18617256 and 18620418 both sat at step 0/2000 for 20 minutes, while the
# SCRIPT arms, which were already passing 4, trained normally on the same data.
NUM_WORKERS="${NUM_WORKERS:-4}"

EXP_NAME="${EXP_NAME:-dfw_granary2_chat_${LOSS_TYPE}}"

# DFW data and model paths.
QWEN_TOK=${HEH}/pretrained_models/huggingface/Qwen/Qwen3-1.7B
INIT_NEMO="${INIT_NEMO:-${HEH}/pretrained_models/huggingface/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo}"
TRAIN_INPUT_CFG="${TRAIN_INPUT_CFG:-${HEH}/data_configs/granary_v2_en_full_d0.5_b0.5_dfw_qwen_aligned.yaml}"
VAL_MANIFEST="${VAL_MANIFEST:-${HEH}/data/mcv11_en_dev_aligned/mcv11_dev_clean_pcstrip_en_2k_qwen_aligned.json}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${QWEN_TOK}}"

# A 1-node allocation writes to a DIFFERENT EXP_NAME so a smoke test can never
# resume from, or overwrite, the real 8-node run's checkpoints.
# Set explicitly, NOT grepped from $0: this body is exec'd from a wrapper, so $0
# is this file, which carries no #SBATCH lines -- the grep would always miss and
# fall back, wrongly suffixing every full-size run. Wrappers may override.
DESIGN_NODES="${DESIGN_NODES:-4}"
ACTUAL_NODES="${SLURM_JOB_NUM_NODES:-$DESIGN_NODES}"
if [[ "${SKIP_NODE_SUFFIX:-0}" != "1" && "$ACTUAL_NODES" -ne "$DESIGN_NODES" ]]; then
    EXP_NAME="${EXP_NAME}_n${ACTUAL_NODES}"
    echo "==> Allocation is ${ACTUAL_NODES} node(s), not the designed ${DESIGN_NODES}; EXP_NAME -> ${EXP_NAME}"
fi

RESULTS_DIR=${MYDIR}/results/${PROJECT_NAME}/${EXP_NAME}
HFCACHE=${MYDIR}/hf_cache
mkdir -p "$RESULTS_DIR" "$HFCACHE"
OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

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

MOUNTS="--container-mounts=${CODE_DIR}:/code,${RESULTS_DIR}:/results,${HFCACHE}:/hfcache,${LUSTRE}:${LUSTRE},${DATA_ROOT}:${DATA_ROOT}"

# Do NOT enable xtrace: the command below contains expanded token values.
read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "*** DFW CHAT transducer (RNNTAttJoint), loss_type=${LOSS_TYPE} ***" \
&& echo "*** knobs: delay=${DELAY_FRAMES} recover=${RECOVER_WORDS} history_chunks=${HISTORY_CHUNKS} band=${BAND_CHUNKS} targets=${TARGET_CONSTRUCTION} punct_delay=${DELAY_PUNCT} ***" \
&& echo "*** schedule: epoch=${EPOCH_STEPS} lr=${LR} warmup=${WARMUP_STEPS} max_steps=${MAX_STEPS} ***" \
&& nvidia-smi \
&& export WANDB_API_KEY=${WANDB} \
&& cd /code && export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& echo "CODE COMMIT:" && git rev-parse HEAD \
&& export HF_HOME="/hfcache/" HF_TOKEN=${HF_TOKEN} HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 \
&& export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
&& export PYTORCH_ALLOC_CONF=expandable_segments:True \
&& export TORCH_NCCL_TIMEOUT_SEC=3600 \
&& export TORCH_NCCL_USE_COMM_NONBLOCKING=0 \
&& export TORCH_FR_BUFFER_SIZE=0 \
&& export TORCH_NCCL_ENABLE_MONITORING=0 \
&& export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=240 \
&& export NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=10 \
&& if [ ! -f '${TOKENIZER_DIR}/tokenizer.model' ] && [ ! -f '${TOKENIZER_DIR}/tokenizer.json' ]; then \
     echo '==> extracting the donor SentencePiece vocabulary'; \
     python -c "
import os, tarfile
src, dst = '${INIT_NEMO}', '${TOKENIZER_DIR}'
os.makedirs(dst, exist_ok=True)
with tarfile.open(src, 'r:') as tf:
    for m in tf.getmembers():
        for want in ('tokenizer.model', 'tokenizer.vocab', 'vocab.txt'):
            if m.name.endswith(want):
                m.name = want; tf.extract(m, dst)
assert os.path.isfile(os.path.join(dst,'tokenizer.model')), 'no tokenizer.model in ' + src
print('    tokenizer ->', dst)
"; \
   else echo '==> reusing existing tokenizer'; fi \
&& python /code/examples/asr/asr_transducer/speech_to_text_chat_bpe.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    model.tokenizer.dir=${TOKENIZER_DIR} \
    model.loss_type=${LOSS_TYPE} \
    model.forced_alignment.num_delay_frames=${DELAY_FRAMES} \
    model.forced_alignment.max_delay_frames=${MAX_DELAY_FRAMES} \
    model.forced_alignment.band_chunks=${BAND_CHUNKS} \
    ++model.forced_alignment.band_side=${BAND_SIDE} \
    model.forced_alignment.delay_word_final_punctuation=${DELAY_PUNCT} \
    model.forced_alignment.target_construction=${TARGET_CONSTRUCTION} \
    model.forced_alignment.inference_delay_frames=${INFER_DELAY_FRAMES} \
    model.forced_alignment.recover_history_words=${RECOVER_WORDS} \
    model.joint.history_chunks=${HISTORY_CHUNKS} \
    model.train_ds.input_cfg=${TRAIN_INPUT_CFG} \
    model.train_ds.num_workers=${NUM_WORKERS} \
    model.validation_ds.manifest_filepath=${VAL_MANIFEST} \
    model.optim.lr=${LR} \
    model.optim.sched.warmup_steps=${WARMUP_STEPS} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.limit_train_batches=${EPOCH_STEPS} \
    trainer.val_check_interval=${EPOCH_STEPS} \
    trainer.devices=${GPUS_PER_NODE} \
    trainer.num_nodes=\${SLURM_JOB_NUM_NODES} \
    +init_from_nemo_model.model0.path=${INIT_NEMO} \
    +init_from_nemo_model.model0.include=["encoder.","decoder.","joint.enc.","joint.pred."] \
    +init_from_nemo_model.model0.exclude=${INIT_EXCLUDE:-'["prediction.embed"]'} \
    ++exp_manager.exp_dir=/results/ \
    ++exp_manager.name=${EXP_NAME} \
    ++exp_manager.max_time_per_run=00:03:55:00 \
    ++exp_manager.create_wandb_logger=true \
    ++exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    ++exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME}
EOF

srun -o "$OUTFILE" -e "$ERRFILE" --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
