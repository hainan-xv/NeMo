#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-eval-multivocab
#SBATCH -p interactive
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
# Open-ASR-Leaderboard eval of the MULTI-VOCAB arm: three heads, ONE job.
#
#   sbatch launch/dfw_eval_multivocab.sh
#
# Averages the arm's top-5 checkpoints ONCE, then decodes the whole benchmark
# three times -- head 0 (1k), head 1 (2k), head 2 (4k) -- sequentially, with the
# eight datasets running in parallel across the eight GPUs within each head.
# One average, three decodes: the heads share an encoder, so re-averaging per
# head would be identical work and would also risk the three rows differing for
# a reason other than the vocabulary.
#
# Selecting a head swaps its TOKENIZER as well as its decoder and joint, so each
# pass produces text from that vocabulary. run_eval.py appends "-headK" to the
# manifest name because all three share one .nemo path -- without it each head
# would overwrite the last and the scorer would print three identical rows.
#
# max_symbols 15 throughout: a 1k vocabulary needs ~23 tokens where 16k needs 13
# on the same sentence, and the cap is PER CHUNK, so the small heads are the ones
# at risk of truncation.
# ============================================================================
set -uo pipefail
mkdir -p slurm_out

DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
MY=${DFW}/hainanx
CODE_DIR="${CODE_DIR:-${MY}/NeMo_SCRIPT_cc}"
OASR="${OASR:-${MY}/open_asr_leaderboard}"
CONTAINER="${CONTAINER:-${DFW}/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"
ARM=dfw_granary2_chat_spe_multivocab
NEMO="${MY}/results/SpeechlmDFW/${ARM}/averaged/top5-averaged.nemo"
OUT="${MY}/results/official_eval/multivocab_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_SYMBOLS="${MAX_SYMBOLS:-15}"
HEADS="${HEADS:-0 1 2}"
# Streaming arm: trained with data.dataset.pad_extra_duration, so the tail needs
# flushing exactly as for every other streaming CHAT arm.
PAD="${PAD:-0.5}"
mkdir -p "$OUT"

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

read_token() { [[ -r "$1" ]] || { echo "ERROR: missing $1" >&2; exit 1; }; tr -d '\r\n' < "$1"; }
HF_TOKEN="$(read_token "$HOME/.hf_token")"

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${LAUNCH_DIR}/eval_chat.sh" ]] || LAUNCH_DIR="${CODE_DIR}/launch"

echo "==> multi-vocab eval | heads=${HEADS} | out=${OUT} | max_symbols=${MAX_SYMBOLS} | pad=${PAD}"

# ---- 1. average ONCE -------------------------------------------------------
echo "### rebuilding the top-5 average"
AVERAGE_ONLY=1 FORCE_AVERAGE=1 bash "${LAUNCH_DIR}/dfw_eval_chat_spe_multivocab.sh" \
    || { echo "### averaging FAILED" >&2; exit 1; }
[[ -f "$NEMO" ]] || { echo "### no averaged model at $NEMO" >&2; exit 1; }

# ---- 2. one decode per head, eight datasets in parallel each ---------------
for HEAD in ${HEADS}; do
    echo; echo "############ HEAD ${HEAD}"
    gpu=0; pids=()
    for cfg in "${DATASETS[@]}"; do
        read -r DS SPLIT DSPATH <<< "$cfg"
        DSPATH="${DSPATH:-$DEFAULT_PATH}"
        DLOG="${OUT}/head${HEAD}.${DS}_${SPLIT}.log"
        echo "--- head ${HEAD} / ${DS} ${SPLIT} -> gpu ${gpu}" | tee "${DLOG}"
        (
            CUDA_VISIBLE_DEVICES=${gpu} \
            srun --overlap -n1 -N1 \
                 --container-image="$CONTAINER" \
                 --container-mounts="${DFW}:${DFW},${CODE_DIR}:/code,${OASR}:/oasr" \
                 bash -c "export CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=/code:/code/scripts:/oasr:${MY}/pylibs:\${PYTHONPATH:-} HF_HOME=${MY}/hf_cache HF_TOKEN=${HF_TOKEN} && \
                          cd /oasr/nemo_asr && \
                          python run_eval.py --model_id='${NEMO}' --dataset_path='${DSPATH}' \
                            --dataset='${DS}' --split='${SPLIT}' --device=0 \
                            --batch_size=${BATCH_SIZE} --max_eval_samples=-1 \
                            --pad_extra_seconds=${PAD} --max_symbols=${MAX_SYMBOLS} \
                            --multivocab_head=${HEAD}" >> "${DLOG}" 2>&1
        ) &
        pids+=($!); gpu=$((gpu + 1))
    done
    fail=0
    for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "  [FAIL] dataset ${i} head ${HEAD}" >&2; fail=$((fail+1)); }; done
    n_wer=$(cat "${OUT}/head${HEAD}."*.log 2>/dev/null | grep -c "^WER: " || echo 0)
    echo "### head ${HEAD}: ${n_wer}/${#DATASETS[@]} datasets scored, ${fail} process failures"
    echo "### head ${HEAD} vocab line: $(grep -h 'MULTIVOCAB_HEAD' "${OUT}/head${HEAD}."*.log 2>/dev/null | head -1)"
done

# ---- 3. official scorer, all heads together -------------------------------
echo; echo "############ OFFICIAL SCORE (kaldialign, merge_compounds=True)"
srun --overlap -n1 -N1 --container-image="$CONTAINER" \
     --container-mounts="${DFW}:${DFW},${CODE_DIR}:/code,${OASR}:/oasr" \
     bash -c "export PYTHONPATH=${MY}/pylibs:/oasr:\${PYTHONPATH:-} && python - <<'PYEOF'
import sys
sys.path.insert(0, '/oasr')
from collections import defaultdict
from normalizer import eval_utils
score, results = eval_utils.score_results('/oasr/nemo_asr/results')
per = defaultdict(dict)
for k, v in (results or {}).items():
    m, _, ds = k.partition(' | ')
    per[m][ds] = v.get('wer')
SEVEN = [d for d in sorted({d for m in per for d in per[m]}) if d != 'hf-audio-open-asr-leaderboard_earnings22_test']
rows = []
for m, dd in per.items():
    got = [dd[d] for d in SEVEN if d in dd]
    if len(got) == len(SEVEN):
        rows.append((sum(got) / len(got), ' '.join(f'{dd[d]:5.2f}' for d in SEVEN), m))
for a, c, m in sorted(rows):
    print(f'ROW {a:6.3f} | {c} | {m[-58:]}')
PYEOF
" 2>&1 | grep -E "^ROW"
