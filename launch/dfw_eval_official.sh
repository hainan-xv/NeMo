#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-eval-official
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --mem=0
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
# ONE MODEL PER JOB, and within a job the 8 DATASETS RUN CONCURRENTLY, one per
# GPU. Wall-clock is then the slowest single dataset (spgispeech, 39k utts)
# rather than the sum of all eight -- measured serially at ~30 min per model on
# one GPU with seven idle.
#
# Utterances are NOT split within a dataset: run_eval.py has no sharding support,
# and adding it would mean patching the official code at the point that decides
# which utterances are scored. Across datasets needs no code change at all.
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
  "fullctx|${MY}/results/SpeechlmDFW/dfw_granary2_chat_banded1_both_fullctx_parakeet/averaged/top5-averaged.nemo|0|dfw_eval_chat_fullctx_parakeet.sh|${MY}/results/SpeechlmDFW/dfw_granary2_chat_banded1_both_fullctx_parakeet/dfw_granary2_chat_banded1_both_fullctx_parakeet/checkpoints"
  "chat_both|${MY}/results/SpeechlmDFW/dfw_granary2_chat_banded1_both_nodelay_v2/averaged/top5-averaged.nemo|0.5"
  "chat_later|${MY}/results/SpeechlmDFW/dfw_granary2_chat_banded1_nodelay_v2/averaged/top5-averaged.nemo|0.5"
  "parakeet|${MY}/pretrained_models/nvidia/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo|0"
  "nemotron|${DFW}/users/heh/pretrained_models/huggingface/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo|0.5"
  # SCRIPT arms are Lightning .ckpt, not .nemo: run_eval.py routes them through
  # scripts/script_asr_shim.py, which reuses script_leaderboard_eval's own loader
  # and generate path. pad 0.5 because SCRIPT trains with
  # data.dataset.pad_extra_duration and its emission lags the audio.
  "script_later|${MY}/results/SpeechlmDFW/dfw_granary2_script_banded1_nodelay_v2/dfw_granary2_script_banded1_nodelay_v2/checkpoints/dfw_granary2_script_banded1_nodelay_v2-averaged.ckpt|0.5"
  # Purpose-built 8k SentencePiece vocabulary. STREAMING, so pad 0.5 like the
  # other CHAT arms; its own eval launcher carries the tokenizer override that
  # averaging needs.
  "spe8k|${MY}/results/SpeechlmDFW/dfw_granary2_chat_spe8k_both/averaged/top5-averaged.nemo|0.5|dfw_eval_chat_spe8k.sh|${MY}/results/SpeechlmDFW/dfw_granary2_chat_spe8k_both/dfw_granary2_chat_spe8k_both/checkpoints"
  # Purpose-built 16k SentencePiece vocabulary. STREAMING, so pad 0.5 like the
  # other CHAT arms; its own eval launcher carries the tokenizer override that
  # averaging needs.
  "spe16k|${MY}/results/SpeechlmDFW/dfw_granary2_chat_spe16k_both/averaged/top5-averaged.nemo|0.5|dfw_eval_chat_spe16k.sh|${MY}/results/SpeechlmDFW/dfw_granary2_chat_spe16k_both/dfw_granary2_chat_spe16k_both/checkpoints"
  # Purpose-built 32k SentencePiece vocabulary. STREAMING, so pad 0.5 like the
  # other CHAT arms; its own eval launcher carries the tokenizer override that
  # averaging needs.
  "spe32k|${MY}/results/SpeechlmDFW/dfw_granary2_chat_spe32k_both/averaged/top5-averaged.nemo|0.5|dfw_eval_chat_spe32k.sh|${MY}/results/SpeechlmDFW/dfw_granary2_chat_spe32k_both/dfw_granary2_chat_spe32k_both/checkpoints"
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
# MAX_SYMBOLS: tokens the greedy decoder may emit per step. For CHAT a step is
# a CHUNK, and measurement showed 8.9 inner iterations against a cap of 10 --
# so chunks were plausibly being truncated and losing words. 0 keeps each
# model's own value; frame-synchronous baselines never approach the cap, so
# setting it is safe to apply uniformly.
MAX_SYMBOLS="${MAX_SYMBOLS:-0}"
MAX_SYM_ARG=""
[[ "${MAX_SYMBOLS}" != "0" ]] && MAX_SYM_ARG="--max_symbols=${MAX_SYMBOLS}"

read_token() { [[ -r "$1" ]] || { echo "ERROR: missing $1" >&2; exit 1; }; tr -d '\r\n' < "$1"; }
HF_TOKEN="$(read_token "$HOME/.hf_token")"

# Under sbatch $0 is a spool copy, and SLURM_SUBMIT_DIR is unreliable on a
# requeue -- hence the absolute fallback to the grid checkout.
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${LAUNCH_DIR}/eval_chat.sh" ]] || LAUNCH_DIR="${CODE_DIR}/launch"

WANT="${MODELS:-}"
echo "==> official leaderboard harness | out=${OUT} | batch=${BATCH_SIZE}"

for entry in "${ALL[@]}"; do
    IFS='|' read -r key nemo pad avg_launcher ckpt_dir <<< "$entry"

    # MODEL FILTER FIRST. It used to sit AFTER the averaging block, so a job
    # asked for one model still re-averaged every arm that has an averaging
    # launcher -- burning time on models it would not score, and racing the
    # trainers of arms it was never asked about.
    [[ -n "$WANT" && " $WANT " != *" $key "* ]] && continue

    # Refresh the average IN THIS JOB when asked, or refuse to score a stale one.
    # Previously averaging lived only in eval_chat.sh, reached from a different
    # launcher -- so refreshing meant running a second job that ALSO ran its own
    # redundant eval, and FORCE_AVERAGE=1 passed here was silently ignored while
    # 32 minutes were spent re-scoring weights from 50 epochs earlier.
    if [[ -n "${avg_launcher:-}" && -n "${ckpt_dir:-}" && -d "$ckpt_dir" ]]; then
        newest="$(ls -t "${ckpt_dir}"/*.ckpt 2>/dev/null | head -1)"
        stale=0
        [[ -n "$newest" && -f "$nemo" && "$newest" -nt "$nemo" ]] && stale=1
        if [[ "${FORCE_AVERAGE:-0}" == "1" || "$stale" == "1" ]]; then
            echo "### ${key}: rebuilding average (force=${FORCE_AVERAGE:-0} stale=${stale})"
            AVERAGE_ONLY=1 FORCE_AVERAGE=1 bash "${LAUNCH_DIR:-$(dirname "${BASH_SOURCE[0]}")}/${avg_launcher}" \
                || { echo "### ${key}: SKIP, averaging failed" >&2; continue; }
        elif [[ "$stale" == "1" ]]; then
            echo "### ${key}: REFUSING -- ${nemo} is older than its checkpoints. FORCE_AVERAGE=1 to rebuild." >&2
            continue
        fi
    fi
    if [[ ! -f "$nemo" ]]; then
        echo "### ${key}: SKIP, missing ${nemo}" >&2; continue
    fi
    echo; echo "############ ${key}  (pad=${pad}s)"; echo "###   ${nemo}"

    # ONE DATASET PER GPU, all 8 concurrently. run_eval.py has no sharding
    # support (no --num_shards/--shard_index), so utterances cannot be split
    # within a dataset without patching the official code -- which is exactly the
    # code we adopted BECAUSE it is authoritative, and the place where a mistake
    # silently loses or double-counts utterances. Across datasets is free: each
    # was already an independent invocation writing its own manifest, so this
    # changes only WHEN they run, not WHAT they compute.
    #
    # Wall-clock becomes the slowest single dataset (spgispeech, 39k utts)
    # instead of the sum of all eight.
    gpu=0
    pids=()
    for cfg in "${DATASETS[@]}"; do
        read -r DS SPLIT DSPATH <<< "$cfg"
        DSPATH="${DSPATH:-$DEFAULT_PATH}"
        # Per-dataset log: concurrent writers would interleave a shared file and
        # make the WER-to-dataset mapping unrecoverable.
        DLOG="${OUT}/${key}.${DS}_${SPLIT}.log"
        echo "--- ${key} / ${DS} ${SPLIT} -> gpu ${gpu}" | tee "${DLOG}"
        (
            # --overlap, NOT --exclusive. On an srun STEP, --exclusive means "do
            # not share this allocation with other steps", so Slurm SERIALISES the
            # eight backgrounded steps instead of running them side by side --
            # observed directly: only step .1 was ever live. --overlap lets them
            # share the node, and CUDA_VISIBLE_DEVICES pins each to its own GPU.
            CUDA_VISIBLE_DEVICES=${gpu} \
            srun --overlap -n1 -N1 \
                 --container-image="$CONTAINER" \
                 --container-mounts="${DFW}:${DFW},${CODE_DIR}:/code,${OASR}:/oasr" \
                 bash -c "export CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=/code:/code/scripts:/oasr:${MY}/pylibs:\${PYTHONPATH:-} HF_HOME=${MY}/hf_cache HF_TOKEN=${HF_TOKEN} && \
                          cd /oasr/nemo_asr && \
                          python run_eval.py --model_id='${nemo}' --dataset_path='${DSPATH}' \
                            --dataset='${DS}' --split='${SPLIT}' --device=0 \
                            --batch_size=${BATCH_SIZE} --max_eval_samples=-1 \
                            --pad_extra_seconds=${pad} ${MAX_SYM_ARG}" >> "${DLOG}" 2>&1
        ) &
        pids+=($!)
        gpu=$((gpu + 1))
    done

    # Wait for every dataset and report which ones failed. A silent failure here
    # loses that dataset's result entirely -- run_eval.py writes its manifest
    # BEFORE printing the WER, so a crash costs the number, not just the file.
    fail=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "  [FAIL] dataset index ${i} for ${key}" >&2; fail=$((fail + 1)); }
    done
    # Concatenate per-dataset logs into the per-model log the summary reads.
    cat "${OUT}/${key}."*.log > "${OUT}/${key}.log" 2>/dev/null
    n_wer=$(grep -hc "^WER: " "${OUT}/${key}.log" 2>/dev/null || echo 0)
    echo "### ${key}: ${n_wer}/${#DATASETS[@]} datasets scored, ${fail} process failures"
done

echo; echo "############ summary (printed WER -- NOT the leaderboard metric)"
for f in "${OUT}"/*.log; do
    [[ -f "$f" ]] || continue
    echo "### $(basename "$f" .log)"
    grep -hoE "WER: [0-9.]+|wer: [0-9.]+" "$f" | tail -8
done

# ---------------------------------------------------------------------------
# OFFICIAL SCORE, in this job.
#
# run_eval.py PRINTS evaluate/jiwer WER; the leaderboard PUBLISHES what
# normalizer/eval_utils.score_results() computes -- kaldialign batch_error_rate
# with merge_compounds=True, which counts split compounds ("white paper" vs
# "whitepaper") as zero errors. That is 0.3-0.5 WER more lenient, so the numbers
# above are systematically pessimistic and not comparable to published figures.
#
# Scoring re-reads the saved manifests, so it costs seconds and needs no GPU --
# there was never a reason for it to be a separate job, and running it by hand
# is how a table of jiwer numbers got reported as if it were leaderboard-
# comparable.
# ---------------------------------------------------------------------------
echo; echo "############ OFFICIAL SCORE (kaldialign, merge_compounds=True)"
srun --overlap -n1 -N1 --container-image="$CONTAINER" \
     --container-mounts="${DFW}:${DFW},${CODE_DIR}:/code,${OASR}:/oasr" \
     bash -c "export PYTHONPATH=${PYLIBS}:/oasr:\${PYTHONPATH:-} && python - <<'PYEOF'
import sys
sys.path.insert(0, '/oasr')
from normalizer import eval_utils
score, results = eval_utils.score_results('/oasr/nemo_asr/results')
print()
for k, v in sorted((results or {}).items()):
    model, _, ds = k.partition(' | ')
    print(f'{ds}\t{v.get(\"wer\")}\t{model[-60:]}')
PYEOF
" 2>&1 | grep -vE "^srun:|CSV Summary|^\*{4,}|^model,|^$" | tail -60
