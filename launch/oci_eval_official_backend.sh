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
# -1 = the whole dataset. Set a small value (with a throwaway KEY) to smoke-test
# the plumbing without writing real manifests or burning a full decode.
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"
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

# ONE PROCESS PER GPU, COVERING EVERY DATASET.
#
# Three layouts have been measured here:
#
#   1. one dataset per GPU        -- spgispeech is 52% of all 75,078 utterances,
#                                    so it ran alone while seven GPUs idled.
#                                    ~1.9x over serial instead of 8x.
#   2. one dataset at a time,     -- balanced (gigaspeech split 2343-2346 per
#      sharded 8 ways                shard, all finishing within a minute), but
#                                    the model was loaded 64 times per arm and
#                                    the RTFx warm-up ran 8 times. voxpopuli's
#                                    628 utterances took 5 min against
#                                    gigaspeech's 12 for 30x the data -- almost
#                                    pure startup.
#   3. THIS: one process per GPU, each decoding its 1/8 stride of ALL datasets.
#                                    8 model loads instead of 64, one warm-up
#                                    instead of 8, and the same even split.
#
# The balance is unchanged from (2) -- run_eval.py still assigns utterance i to
# shard i % NGPU within each dataset -- so this is purely the removal of repeated
# fixed cost, not a different division of work.
NGPU="${NGPU:-8}"

# Dataset specs as run_eval.py --datasets wants them: name:split[:dataset_path].
SPECS=""
for cfg in "${DATASETS[@]}"; do
    read -r DS SPLIT DSPATH <<< "$cfg"
    if [[ -n "${DSPATH:-}" ]]; then entry="${DS}:${SPLIT}:${DSPATH}"; else entry="${DS}:${SPLIT}"; fi
    SPECS="${SPECS:+${SPECS},}${entry}"
done
echo "==> ${#DATASETS[@]} datasets in ONE process per GPU: ${SPECS}"

# WARM THE HUB RESOLUTION ONCE PER DATASET, then decode offline.
#
# The audio is cached (19G of open-asr-leaderboard on lustre; shards fetch zero
# bytes). What is NOT cached is the repo FILE LISTING: load_dataset re-resolves
# /api/datasets/<repo>/tree/<rev>/<config> on every invocation. Sharding turned
# that into NGPU identical metadata calls per dataset and blew HF's quota of
# 1000 API requests / 5 min -- a 429 that killed whole datasets while the data
# sat on local disk. Resolve once here, then run every shard with
# HF_HUB_OFFLINE=1 so they make no API calls at all.
# ALL AT ONCE, not one after another. These were 8 sequential srun calls, each
# paying container start plus a Hub round-trip -- ~5-6 minutes of dead time before
# a single GPU did any work, which on a 19-minute arm is most of the startup. They
# are independent, so they run concurrently; the quota problem was 64 SIMULTANEOUS
# resolvers from the shard fan-out, and 8 is comfortably inside it.
_warm_pids=()
for cfg in "${DATASETS[@]}"; do
    read -r DS SPLIT DSPATH <<< "$cfg"
    DSPATH="${DSPATH:-$DEFAULT_PATH}"
    (
    srun --overlap -n1 -N1 --container-image="$CONTAINER" \
         --container-mounts="${LUSTRE}:${LUSTRE},${HEH}:${HEH},${CODE_DIR}:/code,${OASR}:/oasr" \
         bash -c "export PYTHONPATH=/code:/code/scripts:/oasr:${PYLIBS}:\${PYTHONPATH:-} HF_HOME=${MY}/hf_cache HF_TOKEN=${HF_TOKEN} && \
                  python -c \"
import sys; sys.path.insert(0,'/oasr')
from normalizer import data_utils
class A:
    dataset_path='${DSPATH}'; dataset='${DS}'; split='${SPLIT}'
    max_eval_samples=1; streaming=False
data_utils.load_data(A())
print('hub resolution warmed: ${DS} ${SPLIT}')
\"" >> "${OUT}/${KEY}.warmup.log" 2>&1 \
        || echo "  [warn] hub warm-up failed for ${DS} ${SPLIT}; shards will need online resolution"
    ) &
    _warm_pids+=($!)
done
for _p in "${_warm_pids[@]}"; do wait "$_p" || true; done
echo "==> hub resolution warmed for all ${#DATASETS[@]} datasets (in parallel)"

DECODE_START=$(date +%s)
pids=()
for gpu in $(seq 0 $((NGPU - 1))); do
    DLOG="${OUT}/${KEY}.shard${gpu}.log"
    : > "${DLOG}"
    (
        CUDA_VISIBLE_DEVICES=${gpu} \
        srun --overlap -n1 -N1 \
             --container-image="$CONTAINER" \
             --container-mounts="${LUSTRE}:${LUSTRE},${HEH}:${HEH},${CODE_DIR}:/code,${OASR}:/oasr" \
             bash -c "export CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=/code:/code/scripts:/oasr:${PYLIBS}:\${PYTHONPATH:-} HF_HOME=${MY}/hf_cache HF_TOKEN=${HF_TOKEN} HF_HUB_OFFLINE=${HF_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_OFFLINE:-1} && \
                      cd /oasr/nemo_asr && \
                      python run_eval.py --model_id='${MODEL}' --dataset_path='${DEFAULT_PATH}' \
                        --datasets='${SPECS}' --device=0 \
                        --batch_size=${BATCH_SIZE} --max_eval_samples=${MAX_EVAL_SAMPLES} \
                        --num_shards=${NGPU} --shard_index=${gpu} --run_tag='${KEY}' \
                        --pad_extra_seconds=${PAD} ${MAX_SYM_ARG} ${CHUNK_ARG} ${TYPE_ARG}" >> "${DLOG}" 2>&1
    ) &
    pids+=($!)
done

overall_fail=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "  [FAIL] shard ${i}" >&2; overall_fail=$((overall_fail + 1)); }
done
echo "### ${KEY}: decode finished in $(( ($(date +%s) - DECODE_START) / 60 )) min, ${overall_fail} shard failures"

cat "${OUT}/${KEY}."*.log > "${OUT}/${KEY}.log" 2>/dev/null
if [[ "${overall_fail}" -gt 0 ]]; then
    # A dead GPU poisons its CUDA context, so one "unspecified launch failure"
    # costs that shard on EVERY later dataset -- observed as 7 of 8 datasets
    # losing the same shard index. Say so plainly: the affected datasets have no
    # manifest at all now, and the arm must be re-run rather than scored.
    echo "### WARNING: ${KEY} has failed shards. Datasets missing a shard were NOT written," >&2
    echo "###          so this arm is INCOMPLETE and must be re-run before it is scored." >&2
    echo "###          Per-shard failures cluster on one GPU when its context is poisoned." >&2
fi

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
              python /code/scripts/merge_shard_manifests.py ${RESULTS} --tag '${KEY}'" \
  || { echo "### MERGE REPORTED PROBLEMS -- see above; affected datasets were NOT written" >&2; }

# ---- score in this job, right after merging -------------------------------
# The scorer runs against a SNAPSHOT of merged manifests (see
# scripts/score_leaderboard_snapshot.py), so it cannot trip over other arms'
# in-flight shards the way the old inline scorer did -- that version globbed the
# live directory and died with FileNotFoundError when a peer's merge deleted a
# file mid-scan, marking finished jobs FAILED.
#
# The table printed here includes every arm that has merged so far, so it is a
# running view rather than only this arm's row; ours is marked with <<<.
echo; echo "############ OFFICIAL SCORE (kaldialign, merge_compounds=True)"
if [[ "${overall_fail}" -gt 0 ]]; then
    echo "### NOT scoring ${KEY}: it has failed shards, so some datasets have no" >&2
    echo "###   manifest and its row would be computed from a subset." >&2
fi
srun --overlap -n1 -N1 --container-image="$CONTAINER" \
     --container-mounts="${LUSTRE}:${LUSTRE},${HEH}:${HEH},${CODE_DIR}:/code,${OASR}:/oasr" \
     bash -c "export PYTHONPATH=/code:/code/scripts:/oasr:${PYLIBS}:\${PYTHONPATH:-} && \
              python /code/scripts/score_leaderboard_snapshot.py ${RESULTS} --highlight '${KEY}' --oasr /oasr" \
  2>&1 | grep -vE "^srun:|CSV Summary|^\*{4,}|^model,"

echo
echo "==> ${KEY} done. Manifests in ${RESULTS}"
echo "==> Whole-table rescore any time: sbatch launch/oci_eval_official_score.sh"
