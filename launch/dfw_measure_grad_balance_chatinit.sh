#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-grad-balance-chatinit
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 00:30:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Gradient balance at the SHARED encoder, for a TRAINED checkpoint. BATCH copy.
#
# Identical to dfw_measure_grad_balance.sh except for the partition. Interactive
# QOS caps at 2 nodes per user and the summed training run holds both, so a
# 1-node measurement queued there waits on a job with hours left -- and freeing
# the slot would mean killing training to run a diagnostic.
#
# At init, ||dL_chat/d enc|| was ~3000x smaller than ||dL_script/d enc||, so the
# encoder was being shaped almost entirely by SCRIPT. val_wer_chat has since
# fallen 0.750 -> 0.665 while val_wer sits flat at ~0.109, which is consistent
# either with CHAT catching up or with it merely adapting to a representation it
# does not influence. This measures which.
#
# Reads a SNAPSHOT of the checkpoint: '-last.ckpt' is rewritten at every
# validation, and a 15GB file being overwritten mid-read is its own failure mode.
# ============================================================================
set -uo pipefail
mkdir -p slurm_out

LUSTRE=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
HEH=${LUSTRE}/users/heh
MYDIR=${LUSTRE}/hainanx
CONTAINER="${CONTAINER:-${HEH}/containers/nemo-26.02-streaming-speechlm.sqsh}"
CODE_DIR="${CODE_DIR:-${MYDIR}/NeMo_SCRIPT_cc}"
RUN=dfw_joint_chatinit_p84_interactive
# Resolve the NEWEST checkpoint at run time. Hard-coding a "-last.ckpt" name
# does not work against a live run: that file is renamed at every validation
# as the step changes, so a pinned name goes stale within minutes (job
# 21277913 died on exactly that, 12s in).
CKPT_DIR=${MYDIR}/results/SpeechlmDFW/${RUN}/${RUN}/checkpoints
CKPT_SRC="${CKPT_SRC:-$(ls -t ${CKPT_DIR}/*.ckpt 2>/dev/null | grep -v unfinished | head -1)}"
if [[ -z "${CKPT_SRC}" || ! -f "${CKPT_SRC}" ]]; then
    echo "ERROR: no checkpoint found under ${CKPT_DIR}" >&2; exit 1
fi
SNAP_DIR=${MYDIR}/grad_balance
# Snapshot name carries the source name, so a later step does not silently
# reuse an earlier snapshot.
CKPT=${SNAP_DIR}/$(basename "${CKPT_SRC}" .ckpt)__snapshot.ckpt
CONFIG=/code/examples/speechlm2/conf/streaming_stt_granary2_full_joint_chat_script_p84.yaml
VAL_MANIFEST=${HEH}/data/mcv11_en_dev_aligned/mcv11_dev_clean_pcstrip_en_2k_qwen_aligned.json

mkdir -p "$SNAP_DIR"
if [[ ! -f "$CKPT" ]]; then
    echo "==> snapshotting $(basename "$CKPT_SRC")"
    cp "$CKPT_SRC" "$CKPT" || { echo "ERROR: snapshot failed" >&2; exit 1; }
fi
echo "==> using snapshot: $CKPT"

DATA_ROOT=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm
NEMO_USERS=/lustre/fsw/portfolios/nemotron/users
MOUNTS="--container-mounts=${CODE_DIR}:/code,${LUSTRE}:${LUSTRE},${DATA_ROOT}:${DATA_ROOT},${NEMO_USERS}:${NEMO_USERS}"

read -r -d '' cmd <<EOF
export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& export HF_HUB_OFFLINE=1 \
&& export HYDRA_FULL_ERROR=1 \
&& cd /code \
&& python /code/scripts/measure_grad_balance.py \
    --config ${CONFIG} \
    --ckpt ${CKPT} \
    --manifest ${VAL_MANIFEST} \
    --chunks 7,14,28 \
    --chat_donor ${MYDIR}/results/SpeechlmDFW/dfw_granary2_chat_banded1_both_nodelay_v2/averaged/top5-averaged.nemo \
    --batches 4 \
    --batch_size 4 \
    --overrides model.attn_backend=dense model.prompt_include_chunk_size=false \
                data.dataset.prompt_include_chunk_size=false \
                model.pretrained_llm=${HEH}/pretrained_models/huggingface/Qwen/Qwen3-1.7B \
                model.pretrained_asr=${HEH}/pretrained_models/huggingface/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo \
                model.chat.tokenizer.dir=${HEH}/pretrained_models/huggingface/Qwen/Qwen3-1.7B \
                model.chat.forced_alignment.num_delay_frames=0 \
                model.chat.forced_alignment.target_construction=partition \
                model.chat.forced_alignment.delay_word_final_punctuation=true \
                data.dataset.delay_candidates=[0] data.dataset.num_delay_frames=0
EOF
srun --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
