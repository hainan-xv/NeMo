#!/bin/bash
# ============================================================================
# SHARED BACKEND for the official Open-ASR-Leaderboard eval on OCI.
#
# NOT submitted directly -- every model has its own bare-sbatch launcher
# (oci_eval_official_<arm>.sh) that execs this with five positional arguments:
#
#   oci_eval_official_backend.sh KEY MODEL PAD MAX_SYMBOLS CHUNK_SIZE
#
# WHY THIS EXISTS AT ALL. Our OCI numbers (parakeet 5.49, nemotron 5.73, SCRIPT
# multi@cs14 6.01, CHAT fullctx-1k 6.09, ...) come from scripts/leaderboard_common.py's
# WER(normalize=True). The DFW table (parakeet 4.839, fullctx SPE-1k 4.316, ...)
# comes from normalizer/eval_utils.score_results -- kaldialign batch_error_rate
# with merge_compounds=True, over a DIFFERENT dataset list. Comparing identical
# reference checkpoints across the two showed gigaspeech/ls-clean/ls-other/
# voxpopuli agreeing to <=0.03 while ami diverged by +2.52 and earnings22 by
# +1.86, so the gap is the SCORER plus genuinely different earnings22 audio --
# not the models. These scripts run the DFW harness unchanged on OCI so the two
# tables become directly comparable.
#
# THE EXISTING INTERNAL-AGGREGATE SCRIPTS ARE UNTOUCHED, by request: this is an
# addition, not a replacement, and both tables stay on the board.
#
# MODELS ARE PINNED BY HARD LINK, NOT READ IN PLACE. All three arms are STILL
# TRAINING, and their averages are the ones the internal table was built from.
# Re-averaging now would fold in the epochs since and make any difference
# unattributable -- better scorer, or newer weights? A hard link also means a
# concurrent averaging job cannot swap the file out mid-eval, which is the same
# save_top_k race that killed a DFW eval once already. Costs no disk (same fs).
# ============================================================================
set -uo pipefail

KEY="${1:?usage: $0 KEY MODEL PAD MAX_SYMBOLS CHUNK_SIZE}"
MODEL="${2:?}"
PAD="${3:?}"
MAX_SYMBOLS="${4:?}"
CHUNK_SIZE="${5:?}"

LUSTRE=/lustre/fsw/portfolios/nemotron
MY=${LUSTRE}/users/hainanx
HEH=/lustre/fsw/portfolios/llmservice/users/heh
CONTAINER="${CONTAINER:-${HEH}/containers/nemo-26.02-streaming-speechlm.sqsh}"
CODE_DIR="${CODE_DIR:-${MY}/NeMo_SCRIPT_cc}"
OASR="${MY}/open_asr_leaderboard"
# num2words (and its docopt dep) are imported by the official normalizer and are
# NOT in the container; staged here rather than pip-installed per job.
PYLIBS="${MY}/pylibs"
RESULTS="${OASR}/nemo_asr/results"
OUT="${MY}/results/official_eval/${KEY}_$(date +%Y%m%d_%H%M%S)"
PINDIR="${MY}/results/official_eval/pinned/${KEY}"
BATCH_SIZE="${BATCH_SIZE:-128}"
mkdir -p "$OUT" "$PINDIR" "$RESULTS"

# Exactly the DFW dataset list, including the chunked earnings22 that lives in a
# separate hub repo. That repo carries NO transcript column -- the harness joins
# each chunk's text from the parent repo ArtificialAnalysis/Earnings22-Cleaned-AA
# on parent_id (data_utils.load_chunked_data), then merges chunks back into
# sessions before scoring. Nothing to stage locally; it streams from HF.
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

[[ -f "$MODEL" ]] || { echo "ERROR: missing model $MODEL" >&2; exit 1; }
for p in "${OASR}/nemo_asr/run_eval.py" "${OASR}/normalizer/eval_utils.py" "${PYLIBS}/num2words"; do
    [[ -e "$p" ]] || { echo "ERROR: missing prerequisite $p" >&2; exit 1; }
done

# --- pin the artifact -------------------------------------------------------
PINNED="${PINDIR}/$(basename "$MODEL")"
rm -f "$PINNED"
if ln "$MODEL" "$PINNED" 2>/dev/null; then
    echo "==> pinned $(basename "$MODEL") by hard link (source mtime $(stat -c %y "$MODEL" | cut -d. -f1))"
    MODEL="$PINNED"
else
    echo "==> WARNING: could not hard-link $MODEL; reading it in place, so a" >&2
    echo "    concurrent re-average could swap it mid-eval." >&2
fi

MAX_SYM_ARG=""
[[ "${MAX_SYMBOLS}" != "0" ]] && MAX_SYM_ARG="--max_symbols=${MAX_SYMBOLS}"
# --chunk_size is read only by the SCRIPT .ckpt shim; .nemo arms take their chunk
# size from their own config and ignore it.
CHUNK_ARG=""
[[ "$MODEL" == *.ckpt ]] && CHUNK_ARG="--chunk_size=${CHUNK_SIZE}"

echo "==> OFFICIAL harness | key=${KEY} pad=${PAD}s batch=${BATCH_SIZE} max_symbols=${MAX_SYMBOLS} ${CHUNK_ARG:-(chunk n/a)}"
echo "==> model  : ${MODEL}"
echo "==> logs   : ${OUT}"
echo "==> results: ${RESULTS}  (shared across arms; filenames are keyed per arm)"

# ONE DATASET PER GPU, all 8 concurrently -- wall-clock becomes the slowest
# single dataset (spgispeech, 39k utts) instead of the sum of eight. run_eval.py
# has no sharding, so utterances are never split WITHIN a dataset; doing that
# would mean patching the authoritative code at the exact point that decides
# which utterances get scored.
#
# --overlap, NOT --exclusive: on an srun STEP --exclusive means "do not share
# this allocation", which SERIALISES the eight backgrounded steps.
gpu=0; pids=()
for cfg in "${DATASETS[@]}"; do
    read -r DS SPLIT DSPATH <<< "$cfg"
    DSPATH="${DSPATH:-$DEFAULT_PATH}"
    DLOG="${OUT}/${KEY}.${DS}_${SPLIT}.log"
    echo "--- ${KEY} / ${DS} ${SPLIT} -> gpu ${gpu}" | tee "${DLOG}"
    (
        CUDA_VISIBLE_DEVICES=${gpu} \
        srun --overlap -n1 -N1 \
             --container-image="$CONTAINER" \
             --container-mounts="${LUSTRE}:${LUSTRE},${HEH}:${HEH},${CODE_DIR}:/code,${OASR}:/oasr" \
             bash -c "export CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=/code:/code/scripts:/oasr:${PYLIBS}:\${PYTHONPATH:-} HF_HOME=${MY}/hf_cache HF_TOKEN=${HF_TOKEN} && \
                      cd /oasr/nemo_asr && \
                      python run_eval.py --model_id='${MODEL}' --dataset_path='${DSPATH}' \
                        --dataset='${DS}' --split='${SPLIT}' --device=0 \
                        --batch_size=${BATCH_SIZE} --max_eval_samples=-1 \
                        --pad_extra_seconds=${PAD} ${MAX_SYM_ARG} ${CHUNK_ARG}" >> "${DLOG}" 2>&1
    ) &
    pids+=($!); gpu=$((gpu + 1))
done

# run_eval.py writes its manifest BEFORE printing the WER, so a crash costs the
# number rather than just the file -- report every failure explicitly.
fail=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "  [FAIL] dataset index ${i} (${DATASETS[$i]}) for ${KEY}" >&2; fail=$((fail + 1)); }
done
cat "${OUT}/${KEY}."*.log > "${OUT}/${KEY}.log" 2>/dev/null
n_wer=$(grep -hc "^WER: " "${OUT}/${KEY}.log" 2>/dev/null || echo 0)
echo; echo "### ${KEY}: ${n_wer}/${#DATASETS[@]} datasets scored, ${fail} process failures"

echo; echo "############ OFFICIAL SCORE so far (kaldialign, merge_compounds=True)"
echo "### partial by design -- the other arms write into the same results dir as"
echo "### they finish. Run launch/oci_eval_official_score.sh for the final table."
bash "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/oci_eval_official_score.sh" --inline
