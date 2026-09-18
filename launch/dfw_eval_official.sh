#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-eval-official
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 04:00:00
#SBATCH --time-min 02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# Evaluate our models with the OFFICIAL Open-ASR-Leaderboard harness.
#
#   sbatch launch/dfw_eval_official.sh
#   MODELS="fullctx parakeet" sbatch launch/dfw_eval_official.sh    <- a subset
#
# WHY. Our own harness diverged from the leaderboard in four ways at once --
# batch size (8 vs 128), padding, WER implementation, and the dataset list --
# which made every discrepancy impossible to attribute. Running their
# nemo_asr/run_eval.py removes all four at a stroke: same normalizer, same WER,
# same batching, same dataset configs (including earnings22_cleaned_aa_chunked,
# which lives in a DIFFERENT hub repo and we were missing entirely).
#
# THE ONE DELIBERATE DEVIATION IS PADDING, and it is per-model, not global.
# Our streaming CHAT/SCRIPT arms are TRAINED with data.dataset.pad_extra_duration
# =0.5; their emission lags the audio by design, so with no trailing silence the
# final words are never flushed -- measured, streaming CHAT scores 4.00 on
# test-other with the pad and 5.57 without, on the same checkpoint. Offline
# models have nothing to flush and get 0. run_eval.py was patched to take
# --pad_extra_seconds (and to key its audio cache on it, so a padded run cannot
# silently reuse unpadded wavs); duration/RTFx still use the ORIGINAL audio.
#
# SCRIPT arms ARE covered, via scripts/script_asr_shim.py: they are speechlm2
# ScriptSTTModel Lightning checkpoints with no .transcribe(), so the shim exposes
# the ASRModel surface run_eval.py needs while reusing SCRIPT's own decode path.
#
# ENV
#   MODELS   space-separated subset of the keys below (default: all)
#
# ONE GPU, ONE MODEL PER JOB, on the BATCH partition. Each model is an
# independent 8-dataset sweep with no shared state, so running them as five
# single-GPU jobs finishes in the time of the slowest model rather than the
# sum of all five -- and a 0.6B model decoding at batch 128 does not need a
# whole 8-GPU node. NOT --exclusive for the same reason.
# ============================================================================
set -uo pipefail
mkdir -p slurm_out

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
MY=${DFW}/hainanx
CONTAINER="${CONTAINER:-${DFW}/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
CODE_DIR="${CODE_DIR:-${MY}/NeMo_SCRIPT_cc}"
OASR="${MY}/open_asr_leaderboard"
# num2words is imported by the official normalizer and is NOT in the container.
# pip-installed once into ${MY}/pylibs (inside a container job, not on the login
# node) and put on PYTHONPATH rather than installed per job.
PYLIBS="${MY}/pylibs"
OUT="${MY}/results/official_eval/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

# key | .nemo | pad  (pad>0 ONLY for models trained with trailing silence)
ALL=(
  "fullctx|${MY}/results/SpeechlmDFW/dfw_granary2_chat_banded1_both_fullctx_parakeet/averaged/top5-averaged.nemo|0"
  "chat_both|${MY}/results/SpeechlmDFW/dfw_granary2_chat_banded1_both_nodelay_v2/averaged/top5-averaged.nemo|0.5"
  "chat_later|${MY}/results/SpeechlmDFW/dfw_granary2_chat_banded1_nodelay_v2/averaged/top5-averaged.nemo|0.5"
  "parakeet|${MY}/pretrained_models/nvidia/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo|0"
  "nemotron|${DFW}/users/heh/pretrained_models/huggingface/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo|0.5"
  # SCRIPT arms are Lightning .ckpt, not .nemo: run_eval.py routes them through
  # scripts/script_asr_shim.py, which reuses script_leaderboard_eval's own loader
  # and generate path. pad 0.5 because SCRIPT trains with
  # data.dataset.pad_extra_duration and its emission lags the audio.
  "script_later|${MY}/results/SpeechlmDFW/dfw_granary2_script_banded1_nodelay_v2/dfw_granary2_script_banded1_nodelay_v2/checkpoints/dfw_granary2_script_banded1_nodelay_v2-averaged.ckpt|0.5"
  "script_both|${MY}/results/SpeechlmDFW/dfw_granary2_script_banded1_both_nodelay_v2/dfw_granary2_script_banded1_both_nodelay_v2/checkpoints/dfw_granary2_script_banded1_both_nodelay_v2-averaged.ckpt|0.5"
)

# Exactly the configs nemo_asr/run_parakeet.sh evaluates, including the chunked
# earnings22 that lives in a separate hub repo.
DATASETS=(
  "ami_cleaned test"
  "gigaspeech_cleaned test"
  "voxpopuli_cleaned_aa test"
  "earnings22 test"
  "librispeech test.clean"
  "librispeech test.other"
  "spgispeech test"
  "earnings22_cleaned_aa_chunked test ArtificialAnalysis/Earnings22-Cleaned-AA-chunked"
)
DEFAULT_PATH="hf-audio/open-asr-leaderboard"
BATCH_SIZE="${BATCH_SIZE:-128}"

read_token() { [[ -r "$1" ]] || { echo "ERROR: missing $1" >&2; exit 1; }; tr -d '\r\n' < "$1"; }
HF_TOKEN="$(read_token "$HOME/.hf_token")"

WANT="${MODELS:-}"
echo "==> official leaderboard harness | out=${OUT} | batch=${BATCH_SIZE}"

for entry in "${ALL[@]}"; do
    IFS='|' read -r key nemo pad <<< "$entry"
    [[ -n "$WANT" && " $WANT " != *" $key "* ]] && continue
    if [[ ! -f "$nemo" ]]; then
        echo "### ${key}: SKIP, missing ${nemo}" >&2; continue
    fi
    echo; echo "############ ${key}  (pad=${pad}s)"; echo "###   ${nemo}"

    for cfg in "${DATASETS[@]}"; do
        read -r DS SPLIT DSPATH <<< "$cfg"
        DSPATH="${DSPATH:-$DEFAULT_PATH}"
        # Marker goes into the per-model LOG too, not just job stdout: without it
        # the log is a bare list of WERs with no way to tell which dataset each
        # belongs to, or which ones produced nothing at all.
        echo "--- ${key} / ${DS} ${SPLIT}" | tee -a "${OUT}/${key}.log"
        srun --container-image="$CONTAINER" \
             --container-mounts="${DFW}:${DFW},${CODE_DIR}:/code,${OASR}:/oasr" \
             bash -c "export PYTHONPATH=/code:/code/scripts:/oasr:${MY}/pylibs:\${PYTHONPATH:-} HF_HOME=${MY}/hf_cache HF_TOKEN=${HF_TOKEN} && \
                      cd /oasr/nemo_asr && \
                      python run_eval.py --model_id='${nemo}' --dataset_path='${DSPATH}' \
                        --dataset='${DS}' --split='${SPLIT}' --device=0 \
                        --batch_size=${BATCH_SIZE} --max_eval_samples=-1 \
                        --pad_extra_seconds=${pad}" \
            2>&1 | tee -a "${OUT}/${key}.log" | grep -E "WER|RTFx|Error|Traceback" | tail -3
    done
done

echo; echo "############ summary"
for f in "${OUT}"/*.log; do
    [[ -f "$f" ]] || continue
    echo "### $(basename "$f" .log)"
    grep -hoE "WER: [0-9.]+|wer: [0-9.]+" "$f" | tail -8
done
