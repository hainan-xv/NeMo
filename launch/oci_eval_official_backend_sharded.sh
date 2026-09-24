#!/bin/bash
# ============================================================================
# SHARED BACKEND (SHARDED) for the official Open-ASR-Leaderboard eval on OCI.
#
# Supersedes oci_eval_official_backend.sh, which gave each GPU a whole dataset.
# Kept as a separate file only while the unsharded interleaved jobs are still
# executing the original -- overwriting a running bash script truncates it under
# the interpreter. Fold this over the old name once those land.
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

KEY="${1:?usage: $0 KEY MODEL PAD MAX_SYMBOLS CHUNK_SIZE [MODEL_TYPE]}"
MODEL="${2:?}"
PAD="${3:?}"
MAX_SYMBOLS="${4:?}"
CHUNK_SIZE="${5:?}"
# Which shim a .ckpt is routed through. Defaults to script, so every existing
# caller is unchanged; 'speechlm' selects the interleaved StreamingSTTModel.
MODEL_TYPE="${6:-script}"

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

# The interleaved model's base LLM and encoder. Its checkpoint records bare hub
# ids that do not resolve offline on the grid, and pointing these at the same
# snapshots our own arms load also keeps the base model byte-identical across
# every row of the table.
TYPE_ARG=""
if [[ "$MODEL_TYPE" != "script" ]]; then
    HF_PRE="${HEH}/pretrained_models/huggingface"
    P_LLM="${PRETRAINED_LLM:-${HF_PRE}/Qwen/Qwen3-1.7B}"
    P_ASR="${PRETRAINED_ASR:-${HF_PRE}/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo}"
    for p in "$P_LLM" "$P_ASR"; do
        [[ -e "$p" ]] || { echo "ERROR: missing base model $p" >&2; exit 1; }
    done
    TYPE_ARG="--model_type=${MODEL_TYPE} --pretrained_llm='${P_LLM}' --pretrained_asr='${P_ASR}'"
    echo "==> model_type=${MODEL_TYPE} | base llm ${P_LLM}"
    echo "==> base asr ${P_ASR}"
fi

echo "==> OFFICIAL harness | key=${KEY} pad=${PAD}s batch=${BATCH_SIZE} max_symbols=${MAX_SYMBOLS} ${CHUNK_ARG:-(chunk n/a)}"
echo "==> model  : ${MODEL}"
echo "==> logs   : ${OUT}"
echo "==> results: ${RESULTS}  (shared across arms; filenames are keyed per arm)"

# ONE DATASET AT A TIME, SHARDED ACROSS ALL 8 GPUs.
#
# The previous layout gave each GPU a whole dataset, but the eight differ in size
# by 115x -- spgispeech is 39,341 utterances (52% of all 75,078) while
# earnings22-chunked is 341. So spgispeech sat alone on one GPU while seven idled,
# and eight GPUs bought ~1.9x over serial instead of 8x. Measured: the cs2 job
# took 2h11m, almost all of it spgispeech.
#
# Now every GPU gets an equal 1/8 slice of whichever dataset is in flight, so
# wall-clock is (total work)/8 rather than (largest dataset). run_eval.py assigns
# utterance i to shard i % 8, so all shards see the same duration mix.
#
# THE COST is reloading the model once per dataset per GPU (64 loads instead of
# 8). That is 7 extra load rounds, a few minutes -- bought back many times over
# on the long datasets. Pooling all datasets into one sharded pass would avoid it,
# as scripts/leaderboard_common.py build_global_items does for the internal
# harness, but run_eval.py is per-dataset by construction and restructuring that
# means rewriting the authoritative runner rather than extending it.
#
# --overlap, NOT --exclusive: on an srun STEP --exclusive means "do not share this
# allocation", which SERIALISES the backgrounded steps.
NGPU="${NGPU:-8}"
overall_fail=0
for cfg in "${DATASETS[@]}"; do
    read -r DS SPLIT DSPATH <<< "$cfg"
    DSPATH="${DSPATH:-$DEFAULT_PATH}"
    echo; echo "--- ${KEY} / ${DS} ${SPLIT}: ${NGPU} shards across ${NGPU} GPUs"
    DS_START=$(date +%s)
    pids=()
    for gpu in $(seq 0 $((NGPU - 1))); do
        DLOG="${OUT}/${KEY}.${DS}_${SPLIT}.shard${gpu}.log"
        : > "${DLOG}"
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
                            --num_shards=${NGPU} --shard_index=${gpu} \
                            --pad_extra_seconds=${PAD} ${MAX_SYM_ARG} ${CHUNK_ARG} ${TYPE_ARG}" >> "${DLOG}" 2>&1
        ) &
        pids+=($!)
    done
    fail=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "  [FAIL] ${DS} ${SPLIT} shard ${i}" >&2; fail=$((fail + 1)); }
    done
    overall_fail=$((overall_fail + fail))
    echo "### ${DS} ${SPLIT}: ${fail} shard failures, $(( ($(date +%s) - DS_START) / 60 )) min"
done

cat "${OUT}/${KEY}."*.log > "${OUT}/${KEY}.log" 2>/dev/null
echo; echo "### ${KEY}: ${overall_fail} total shard failures across ${#DATASETS[@]} datasets"

# ---- merge shards back to one manifest per dataset -------------------------
# MUST happen before scoring: score_results derives the model id from the
# filename, so unmerged shards become eight one-eighth-sized rows. The merger
# refuses to write a canonical manifest unless all 8 shards are present and no
# audio id appears twice -- a partial merge would score as ~1/8 deletions and
# look merely "a bit poor" rather than broken.
echo; echo "############ MERGING SHARDS"
srun --overlap -n1 -N1 --container-image="$CONTAINER" \
     --container-mounts="${LUSTRE}:${LUSTRE},${HEH}:${HEH},${CODE_DIR}:/code,${OASR}:/oasr" \
     bash -c "export PYTHONPATH=/code:/code/scripts:/oasr:${PYLIBS}:\${PYTHONPATH:-} && \
              python /code/scripts/merge_shard_manifests.py ${RESULTS}" \
  || { echo "### MERGE REPORTED PROBLEMS -- see above; affected datasets were NOT written" >&2; }

echo; echo "############ OFFICIAL SCORE so far (kaldialign, merge_compounds=True)"
echo "### partial by design -- the other arms write into the same results dir as"
echo "### they finish. Run launch/oci_eval_official_score.sh for the final table."
bash "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/oci_eval_official_score.sh" --inline
