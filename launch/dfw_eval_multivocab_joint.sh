#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-eval-mvjoint
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
# Open-ASR-Leaderboard eval of JOINT decoding over the multi-vocab arm's heads.
#
#   sbatch launch/dfw_eval_multivocab_joint.sh
#
# Per chunk, every head proposes text; each candidate is re-tokenised under all
# heads and force-scored; the text with the best SUMMED log-prob wins. See
# nemo/collections/asr/parts/submodules/multivocab_joint_decoding.py.
#
# DOES NOT RE-AVERAGE, and that is the whole point. The per-head rows already on
# the scoreboard (head0 4.893, head1 5.144, head2 5.257) were produced from the
# .nemo below; re-averaging would fold in the epochs trained since, and a joint
# row that beat 4.893 would then be unattributable -- better decoding, or just a
# newer model? Reusing the exact artifact keeps the only difference the decode.
#
# TWO CONFIGURATIONS, and the first is a CONTROL:
#   1,0,0  head 0 alone, but through the joint decoder's search. Joint decoding
#          compares whole chunk-local PATH scores where greedy stops at the
#          first blank-argmax, so this is NOT expected to reproduce 4.893
#          exactly. Its gap to 4.893 measures the SEARCH change; the gap between
#          it and 1,1,1 measures the ENSEMBLE. Without it the two are confounded.
#   1,1,1  uniform product of experts across 1k/2k/4k.
#
# Heads 1 and 2 are 0.25 and 0.36 WER-points behind head 0, so uniform weighting
# may well LOSE to head 0 alone. That is a real result about undertrained heads,
# not a bug -- the weights are the knob, and the control says which way to turn.
#
# Manifests are suffixed -joint<weights> by run_eval.py, so these rows land
# alongside the existing per-head rows and the scorer prints one table.
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
OUT="${MY}/results/official_eval/mvjoint_$(date +%Y%m%d_%H%M%S)"
# Joint decoding walks chunks in Python, one utterance at a time, so the batch
# size buys encoder efficiency rather than decode throughput. Smaller than the
# per-head eval's 128 because a batch's encoder output is held for the whole
# sequential decode.
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_SYMBOLS="${MAX_SYMBOLS:-15}"
JOINT_BEAM="${JOINT_BEAM:-4}"
WEIGHT_SETS="${WEIGHT_SETS:-1,0,0 1,1,1}"
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

[[ -f "$NEMO" ]] || {
    echo "### no averaged model at $NEMO" >&2
    echo "### build it with: sbatch launch/dfw_eval_multivocab.sh" >&2
    exit 1
}
echo "==> multi-vocab JOINT eval | weights='${WEIGHT_SETS}' | beam=${JOINT_BEAM} | out=${OUT}"
echo "==> model: $NEMO ($(stat -c %y "$NEMO" | cut -d. -f1)) -- NOT re-averaged, on purpose"

for W in ${WEIGHT_SETS}; do
    TAG="${W//,/_}"
    echo; echo "############ WEIGHTS ${W}"
    START=$(date +%s)
    gpu=0; pids=()
    for cfg in "${DATASETS[@]}"; do
        read -r DS SPLIT DSPATH <<< "$cfg"
        DSPATH="${DSPATH:-$DEFAULT_PATH}"
        DLOG="${OUT}/w${TAG}.${DS}_${SPLIT}.log"
        echo "--- w=${W} / ${DS} ${SPLIT} -> gpu ${gpu}" | tee "${DLOG}"
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
                            --joint_decode --joint_weights='${W}' --joint_beam=${JOINT_BEAM}" >> "${DLOG}" 2>&1
        ) &
        pids+=($!); gpu=$((gpu + 1))
    done
    fail=0
    for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "  [FAIL] dataset ${i} w=${W}" >&2; fail=$((fail+1)); }; done
    n_wer=$(cat "${OUT}/w${TAG}."*.log 2>/dev/null | grep -c "^WER: " || echo 0)
    echo "### w=${W}: ${n_wer}/${#DATASETS[@]} datasets scored, ${fail} process failures, $(( ($(date +%s) - START) / 60 )) min"
    echo "### w=${W} joint line: $(grep -h 'JOINT_DECODE' "${OUT}/w${TAG}."*.log 2>/dev/null | head -1)"
done

# ---- official scorer: joint rows land beside the existing per-head rows -----
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
        rows.append((sum(got) / len(got), got, m))
for avg, got, m in sorted(rows):
    print('ROW %6.3f | %s | %s' % (avg, ' '.join('%5.2f' % g for g in got), m[-60:]))
PYEOF"

echo; echo "==> logs in $OUT"
