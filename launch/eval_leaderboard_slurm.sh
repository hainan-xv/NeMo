#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:sslm-leaderboard-eval
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH --time-min 02:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1   # ONE task; it fans 8 python procs across the 8 GPUs
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Parallel Open-ASR-Leaderboard eval for a SpeechLM model, ON OCI: one node,
# work BALANCED across 8 GPUs. Instead of one dataset per GPU (imbalanced, since
# datasets differ hugely in size), we POOL every utterance across all datasets,
# shuffle with a fixed seed, and give each GPU an even 1/8 slice. Total wall time
# ~= sum(all)/8 instead of the single largest dataset.
#
# DEFAULT DECODE BACKEND = heh (BACKEND=heh): the checkpoint is converted ONCE to
# HF format (examples/speechlm2/to_hf.py) and each GPU decodes its shard with
# heh's engine examples/speechlm2/streaming_stt_generate.py (state-machine
# inference, pad_extra_duration, etc. — exact parity with the local heh runs).
# scripts/leaderboard_heh_shards.py builds the pooled per-shard NeMo manifests
# (tagging each utt with dataset_key) and reduces the shard generations into
# per-dataset WER. Set BACKEND=sslm to instead use the in-process driver
# scripts/speechlm_leaderboard_eval.py (pooled-shard mode, tqdm progress bars).
#
# Uses the TRACKED, offline driver scripts/speechlm_leaderboard_eval.py (synced
# to /code), which reads a PRE-STAGED wav/manifest cache on lustre (no HF
# download on compute nodes). Stage the cache once (the run_eval_sslm layout:
#   <CACHE_DIR>/<dataset>/<split>/_cache_manifest.jsonl  + 16kHz wavs
# e.g. rsync your local ~/leaderboard_run/cache to lustre).
#
# The checkpoint is already on lustre (the training run wrote it) -> no rsync.
#
# Usage:
#   cd /lustre/fsw/portfolios/llmservice/users/hainanx/NeMo79
#   sbatch oci/eval_leaderboard_slurm.sh <EXP_NAME>
#   # (defaults CACHE_DIR to /lustre/.../users/hainanx/leaderboard_cache; override with CACHE_DIR=)
#
# Key env:
#   CACHE_DIR         pre-staged leaderboard cache root on lustre
#                     (default /lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache)
#   PROJECT           results project dir (default Speechlm79)
#   BACKEND           heh (default) | sslm
#   MODEL_CLASS       model class (default ScriptSTTModel)
#   SYSTEM_PROMPT     decode prompt (match the model's training prompt!)
#   CHUNK_SIZE        decode chunk size override (encoder frames)
#   BATCH_SIZE        per-shard batch size (default 32; sslm OOM auto-halves)
#   MAX_NEW_TOKENS    per-chunk decode cap (default 64; heh uses HEH_MAX_NEW_TOKENS)
#   HEH_MAX_NEW_TOKENS / HEH_USE_STATE_MACHINE / HEH_USE_OFFLINE_EMBS /
#   HEH_PAD_DURATION  heh decode knobs (defaults mirror the local heh runs:
#                     256 / true / false / 0.5)
#   FORCE_CONVERT=1   rebuild the HF model even if one is cached (heh backend)
#   RUN_AVERAGING     1 (default) -> average the top-k non-last ckpts into
#                     <CKPT_DIR>/<EXP>-averaged.ckpt (stored in the model folder,
#                     reused across runs); 0 -> use a single checkpoint.
#   FORCE_AVERAGE=1   recompute the averaged ckpt even if it's cached
#   USE_LAST=1        eval the rolling -last.ckpt (disables averaging)
#   STEP=<n>          eval step=<n>.ckpt explicitly (disables averaging)
#   CKPT=<path>       eval this exact .ckpt (overrides EXP resolution; disables averaging)
#   DATASETS          space-separated 'name:split' list (default: full 8-set suite)
#   SHUFFLE_SEED      seed for the pooled global shuffle (default 1234)
#   MAX_EVAL_SAMPLES  cap samples per dataset (fast smoke test)
#   OUTPUT_PREFIX     results root (default nemotron users/hainanx)
#   EVAL_TAG          optional label spliced into RESULTS_DIR (e.g. d2_cap_punct);
#                     set by oci/eval_leaderboard_promptctl.sh
#
# Concurrent jobs for the SAME exp (e.g. many promptctl delay/cap/punct evals)
# share <EXP>-averaged.ckpt and hf_model_* . Those prep steps are guarded by
# mkdir-based locks on lustre so only one writer runs; others wait and reuse.
# Per-run outputs live under a unique RESULTS_DIR (timestamp + job id + EVAL_TAG).
# ============================================================================

read_optional_token() { [[ -r "$1" ]] && tr -d '\r\n' < "$1" || true; }
HF_TOKEN="$(read_optional_token "$HOME/.hf_token")"
WANDB_TOKEN="$(read_optional_token "$HOME/.wandb_token")"

mkdir -p slurm_out
SLURM_ACCOUNT='llmservice'
USERID='users/hainanx'
OLDUSERID='users/heh'
LUSTRE_ACCOUNT_PREFIX=/lustre/fsw/portfolios/${SLURM_ACCOUNT}

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"

PROJECT="${PROJECT:-Speechlm79}"
EXP_NAME="${1:-${EXP_NAME:-granary2_script}}"
BACKEND="${BACKEND:-heh}"
# Long-form mode: when LONGFORM_DIR is set the shard source switches from the
# short-form leaderboard cache to the long-form manifests discovered under it
# (build_longform: resolve relative audio paths, duration-balance the few huge
# recordings across the GPUs). Everything downstream (convert, decode fan-out,
# aggregate, wandb) is identical. Only the heh backend is supported here (the
# sslm in-process driver is hard-wired to the leaderboard cache layout). Set by
# launch/eval_longform.sh. Must live under a mounted lustre root (/lustre/fsw|fs12).
LONGFORM_DIR="${LONGFORM_DIR:-}"
# Windowed long-form: when >0, split each recording into ~this-many-second windows
# (snapped to a whole number of chunks), decode each window independently, and the
# window aggregator stitches a recording's per-window hyps back together before
# scoring. 0 = whole-recording streaming decode. Only meaningful with LONGFORM_DIR.
LONGFORM_WINDOW_SEC="${LONGFORM_WINDOW_SEC:-0}"
# Range-stratified "small" long-form set: shortest LONGFORM_PER_RANGE utts per
# minute bucket (2-5,5-10,10-20,20-40,... up to LONGFORM_MAX_RANGE_MIN), one utt
# per bucket on each GPU. A small, representative set spanning short..long clips
# without the multi-hour tail. Set by launch/eval_longform*.sh --stratified.
LONGFORM_STRATIFIED="${LONGFORM_STRATIFIED:-0}"
LONGFORM_PER_RANGE="${LONGFORM_PER_RANGE:-8}"
LONGFORM_MAX_RANGE_MIN="${LONGFORM_MAX_RANGE_MIN:-60}"
if [[ -n "$LONGFORM_DIR" && "$BACKEND" != "heh" ]]; then
    echo "ERROR: long-form eval (LONGFORM_DIR set) requires BACKEND=heh." >&2
    exit 1
fi
MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.script_model.ScriptSTTModel}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-Transcribe the audio into text.}"
# Local-mirror overrides for the HF conversion (heh backend) when an EXTERNAL
# checkpoint's exp_config references a bare Hub id (e.g. Qwen/Qwen3-1.7B) that the
# offline container can't fetch. patch_exp_config.py only swaps these in when the
# config value isn't already a valid local path. Empty = no override (our models
# already use mounted local paths). Set by launch/eval_interleave.sh.
PRETRAINED_LLM_OVERRIDE="${PRETRAINED_LLM_OVERRIDE:-}"
PRETRAINED_ASR_OVERRIDE="${PRETRAINED_ASR_OVERRIDE:-}"
CHUNK_SIZE="${CHUNK_SIZE:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
# heh decode knobs (mirror the local heh runs). Only used when BACKEND=heh.
HEH_MAX_NEW_TOKENS="${HEH_MAX_NEW_TOKENS:-256}"
HEH_USE_STATE_MACHINE="${HEH_USE_STATE_MACHINE:-true}"
HEH_USE_OFFLINE_EMBS="${HEH_USE_OFFLINE_EMBS:-false}"
HEH_PAD_DURATION="${HEH_PAD_DURATION:-0.5}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
# Self-correction: also dump the RAW emission stream with <del> markers (e.g.
# "the cat <del> cat sat") as a raw_text field in each generations record, so you
# can see exactly where the model corrected. heh backend only. Default off.
SAVE_RAW="${SAVE_RAW:-false}"
# Self-correction (SCRIPT windowed-re-decoding / redecode models only): decode the
# self-corrected LOCKED stream (re-decode each chunk with lookahead) instead of the
# default non-corrective j=0 stream (decode each chunk once, append). Default off.
# Accepts 1/true. Forwarded to both backends (heh: self_correct=true; sslm:
# --self_correct); ignored by non-redecode models.
SELF_CORRECT="${SELF_CORRECT:-0}"
SELF_CORRECT_HEH=""
SELF_CORRECT_SSLM=""
if [[ "$SELF_CORRECT" == 1 || "$SELF_CORRECT" == true ]]; then
    SELF_CORRECT_HEH="self_correct=true"
    SELF_CORRECT_SSLM="--self_correct"
fi
# Report per-word emission latency (proxy: end-of-chunk time of each word's last
# subword, averaged). SCRIPT only; default on for it, off otherwise.
if [[ -z "${REPORT_LATENCY:-}" ]]; then
    case "$MODEL_CLASS" in
        *ScriptSTTModel) REPORT_LATENCY=true ;;
        *) REPORT_LATENCY=false ;;
    esac
fi
# Checkpoint averaging (DEFAULT ON): average the top-k (non '-last') checkpoints
# — the ones exp_manager keeps by val_wer — into <CKPT_DIR>/<EXP>-averaged.ckpt,
# stored in the model folder and REUSED on later runs. FORCE_AVERAGE=1 recomputes
# it. Setting CKPT=/STEP=/USE_LAST=1 selects a specific ckpt and disables averaging.
RUN_AVERAGING="${RUN_AVERAGING:-1}"
FORCE_AVERAGE="${FORCE_AVERAGE:-0}"
USE_LAST="${USE_LAST:-0}"
STEP="${STEP:-}"
CKPT="${CKPT:-}"
DATASETS="${DATASETS:-librispeech:test.clean librispeech:test.other ami:test earnings22:test gigaspeech:test spgispeech:test tedlium:test voxpopuli:test}"
SHUFFLE_SEED="${SHUFFLE_SEED:-1234}"
# Comma-joined form for the driver's --datasets (which splits on commas).
DATASETS_CSV="$(echo "$DATASETS" | tr -s ' ' ',')"

# Pre-staged leaderboard cache on lustre (rsync of ~/leaderboard_run/cache).
# Not needed in long-form mode (the shards come from LONGFORM_DIR instead).
CACHE_DIR="${CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache}"
if [[ -n "$LONGFORM_DIR" ]]; then
    if [[ ! -d "$LONGFORM_DIR" ]]; then
        echo "ERROR: LONGFORM_DIR not found: $LONGFORM_DIR (stage the long-form manifests there, or set LONGFORM_DIR=)." >&2
        exit 1
    fi
elif [[ ! -d "$CACHE_DIR" ]]; then
    echo "ERROR: CACHE_DIR not found: $CACHE_DIR (stage the leaderboard cache there, or set CACHE_DIR=)." >&2
    exit 1
fi

OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
CKPT_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/${EXP_NAME}/checkpoints"

# A specific-checkpoint request wins over (and disables) averaging.
if [[ -n "$CKPT" || -n "$STEP" || "$USE_LAST" == "1" ]]; then
    RUN_AVERAGING=0
fi

# ---- Resolve the checkpoint on lustre (no rsync; it's already here) ----
# When RUN_AVERAGING=1 the shared target is <EXP>-averaged.ckpt. Concurrent jobs
# for the same exp all point at it; the in-container ensure step (mkdir lock)
# serializes writers so only one averages and the rest reuse.
AVG_INPUTS_QUOTED=""
ENSURE_AVG=0
if [[ "$RUN_AVERAGING" == "1" ]]; then
    AVG_CKPT="${CKPT_DIR}/${EXP_NAME}-averaged.ckpt"
    mapfile -t _AVG_IN < <(ls -t "${CKPT_DIR}"/*.ckpt 2>/dev/null | grep -v -- '-last\.ckpt$' | grep -v -- '-averaged\.ckpt$')
    if [[ ${#_AVG_IN[@]} -eq 0 ]]; then
        echo "ERROR: RUN_AVERAGING=1 but no non-last checkpoints under ${CKPT_DIR}." >&2
        exit 1
    fi
    for f in "${_AVG_IN[@]}"; do AVG_INPUTS_QUOTED+=" '${f}'"; done
    CKPT="$AVG_CKPT"
    ENSURE_AVG=1
    if [[ "$FORCE_AVERAGE" == "1" ]]; then
        echo "==> Will (re)average ${#_AVG_IN[@]} checkpoint(s) -> ${AVG_CKPT} (locked; concurrent jobs wait/reuse)."
    elif [[ -f "$AVG_CKPT" ]]; then
        echo "==> Averaged checkpoint present: ${AVG_CKPT} (locked ensure will reuse unless another job is mid-write)."
    else
        echo "==> Averaged checkpoint missing: will create ${AVG_CKPT} under lock (${#_AVG_IN[@]} inputs)."
    fi
else
    if [[ -z "$CKPT" ]]; then
        if [[ -n "$STEP" ]]; then
            CKPT="${CKPT_DIR}/step=${STEP}.ckpt"
        elif [[ "$USE_LAST" == "1" ]]; then
            CKPT="$(ls -t "${CKPT_DIR}"/*-last.ckpt 2>/dev/null | head -1)"
        else
            CKPT="$(ls -t "${CKPT_DIR}"/*.ckpt 2>/dev/null | grep -v -- '-last\.ckpt$' | grep -v -- '-averaged\.ckpt$' | head -1)"
            [[ -z "$CKPT" ]] && CKPT="$(ls -t "${CKPT_DIR}"/*-last.ckpt 2>/dev/null | head -1)"
        fi
    fi
fi
# Validate now unless the ckpt is produced by the in-container averaging step.
if [[ "$ENSURE_AVG" != "1" && ( -z "$CKPT" || ! -f "$CKPT" ) ]]; then
    echo "ERROR: could not resolve a checkpoint under ${CKPT_DIR} (set CKPT=, STEP=, USE_LAST=1, or RUN_AVERAGING=1)." >&2
    exit 1
fi

# Timestamp + job id (+ optional EVAL_TAG) so concurrent evals never share a results dir.
RUN_TS="$(date +%Y%m%d_%H%M%S)"
JOB_TAG="${SLURM_JOB_ID:-local$$}"
CHUNK_TAG=""; [[ -n "$CHUNK_SIZE" ]] && CHUNK_TAG="_chunk${CHUNK_SIZE}"
EVAL_TAG_SUFFIX=""; [[ -n "${EVAL_TAG:-}" ]] && EVAL_TAG_SUFFIX="_${EVAL_TAG}"
RESULTS_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/leaderboard_eval${CHUNK_TAG}${EVAL_TAG_SUFFIX}_${BACKEND}_${RUN_TS}_${JOB_TAG}"
mkdir -p "$RESULTS_DIR"
# Pooled shard manifests are cheap + depend on seed/datasets, so keep them in the
# timestamped run dir (fresh each run).
SHARD_DIR="${RESULTS_DIR}/shards"; mkdir -p "$SHARD_DIR"
# The HF conversion depends only on the (averaged) checkpoint, not on the decode
# config, so cache it per-ckpt OUTSIDE the timestamped dir and reuse across runs
# (rebuilt under lock when missing, FORCE_CONVERT=1, or ckpt newer than HF).
CKPT_STEM="$(basename "${CKPT%.ckpt}")"
HF_CKPT_DIR="${OUTPUT_PREFIX}/results/${PROJECT}/${EXP_NAME}/hf_model_${CKPT_STEM}"
# The exp config (with a top-level model:) sits next to the checkpoints dir on
# lustre; to_hf.py needs it to instantiate the model class. Env-overridable so an
# EXTERNAL checkpoint (e.g. a colleague's model via CKPT=, not under our
# results/<PROJECT>/<EXP> layout) can point EXP_CFG at its own exp_config.yaml.
EXP_CFG="${EXP_CFG:-${CKPT_DIR%/checkpoints}/exp_config.yaml}"
if [[ "$BACKEND" == "heh" && ! -f "$EXP_CFG" ]]; then
    echo "ERROR: heh backend needs the exp config at ${EXP_CFG} (set BACKEND=sslm to skip conversion)." >&2
    exit 1
fi
HFCACHE="${OUTPUT_PREFIX}/hf_cache"; mkdir -p "$HFCACHE"
CODE_DIR="${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo79/}"
H_DIR=/lustre/fsw/portfolios/llmservice/users/heh
# TMPDIR must stay SHORT: Python multiprocessing (the heh DataLoader workers)
# creates AF_UNIX sockets at ${TMPDIR}/pymp-*/listener-*, and Linux caps the
# socket path at ~108 bytes. The deep, timestamped RESULTS_DIR blows past that
# ("OSError: AF_UNIX path too long"), so default to a short node-local path.
OCI_TMP_DIR="${OCI_TMP_DIR:-/tmp/sslm_eval_${SLURM_JOB_ID:-$$}}"

OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out
# Broad lustre mounts cover ckpt (nemotron), pretrained (llmservice/heh) and the
# staged cache wherever it lives; /code is our synced checkout.
MOUNTS="--container-mounts=${CODE_DIR}:/code,${H_DIR}:${H_DIR},${HFCACHE}:/hfcache/,/lustre/fsw:/lustre/fsw,/lustre/fs12:/lustre/fs12"

NGPU=8

# Shard builder invocation (heh backend). Long-form pulls from LONGFORM_DIR with
# duration balancing; short-form pulls from the leaderboard cache with a seeded
# global shuffle. Both emit the same shard{k}_of{n}.json (with dataset_key), so
# the decode fan-out + aggregate below are identical.
# Aggregator invocation mirrors the builder: windowed long-form stitches windows
# back per recording (needs the utt_map sidecar); everything else uses the plain
# per-row per-dataset reduce.
AGG_INVOCATION="python /code/scripts/leaderboard_heh_shards.py aggregate --out_dir '${SHARD_DIR}'"
if [[ -n "$LONGFORM_DIR" ]]; then
    BUILD_INVOCATION="python /code/scripts/leaderboard_heh_shards.py build_longform --longform_dir '${LONGFORM_DIR}' --out_dir '${SHARD_DIR}' --num_shards ${NGPU} --max_eval_samples ${MAX_EVAL_SAMPLES}"
    if [[ "${LONGFORM_STRATIFIED}" == "1" ]]; then
        BUILD_INVOCATION="${BUILD_INVOCATION} --range_stratified --per_range ${LONGFORM_PER_RANGE} --max_range_min ${LONGFORM_MAX_RANGE_MIN}"
    fi
    if [[ "${LONGFORM_WINDOW_SEC}" != "0" ]]; then
        # chunk_size may be blank (model default); windowing needs a concrete value.
        BUILD_INVOCATION="${BUILD_INVOCATION} --window_sec ${LONGFORM_WINDOW_SEC} --chunk_size ${CHUNK_SIZE:-14}"
        AGG_INVOCATION="python /code/scripts/leaderboard_heh_shards.py aggregate_longform_windows --out_dir '${SHARD_DIR}' --utt_map '${SHARD_DIR}/longform_utt_map.json'"
    fi
else
    BUILD_INVOCATION="python /code/scripts/leaderboard_heh_shards.py build --cache_dir '${CACHE_DIR}' --datasets '${DATASETS_CSV}' --out_dir '${SHARD_DIR}' --num_shards ${NGPU} --shuffle_seed ${SHUFFLE_SEED} --max_eval_samples ${MAX_EVAL_SAMPLES}"
fi

# ---- Weights & Biases reporting of the final per-dataset WER (default: auto) ----
# After aggregation, scripts/eval_wandb_report.py logs ONE run per eval to a
# SEPARATE project (WANDB_EVAL_PROJECT, default <PROJECT>_leaderboard_eval), named
# by the decode config (WANDB_RUN_NAME) so runs are self-describing/comparable.
#   REPORT_WANDB: auto (report iff ~/.wandb_token exists) | 1 (force) | 0 (off)
# Reporting is best-effort and NEVER fails the eval (the reporter no-ops on any
# wandb/network/key error). Set WANDB_MODE=offline to log locally without a key.
REPORT_WANDB="${REPORT_WANDB:-auto}"
WANDB_EVAL_PROJECT="${WANDB_EVAL_PROJECT:-${PROJECT}_leaderboard_eval}"
# Default run name encodes the decode config: <exp><_eval_tag><_chunkN>_<backend>
# (the model wrappers usually pass a cleaner WANDB_RUN_NAME explicitly).
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXP_NAME}${EVAL_TAG_SUFFIX}${CHUNK_TAG}_${BACKEND}}"
# Always stamp the run with its start time (the same RUN_TS used for RESULTS_DIR)
# so repeated evals of the SAME config are distinct + time-ordered in wandb, and
# the run name maps 1:1 to its results dir. Applied whether the name came from a
# wrapper (WANDB_RUN_NAME=) or the default above.
WANDB_RUN_NAME="${WANDB_RUN_NAME}_${RUN_TS}"
DO_WANDB=0
case "$REPORT_WANDB" in
    auto)          [[ -n "${WANDB_TOKEN:-}" ]] && DO_WANDB=1 ;;
    1|true|yes|on)  DO_WANDB=1 ;;
    0|false|no|off) DO_WANDB=0 ;;
    *) echo "WARNING: unknown REPORT_WANDB='${REPORT_WANDB}' (expected auto|1|0); treating as off." >&2 ;;
esac
WANDB_MODE_EXPORT=""
[[ -n "${WANDB_MODE:-}" ]] && WANDB_MODE_EXPORT="export WANDB_MODE='${WANDB_MODE}'; "
WANDB_CLAUSE=""
if [[ "$DO_WANDB" == "1" ]]; then
    if [[ -z "${WANDB_TOKEN:-}" && "${WANDB_MODE:-}" != "offline" ]]; then
        echo "==> NOTE: wandb reporting requested but no ~/.wandb_token found; the reporter will skip (set WANDB_MODE=offline to log locally)."
    fi
    echo "==> wandb reporting ON -> project='${WANDB_EVAL_PROJECT}' run='${WANDB_RUN_NAME}'"
    # Best-effort: '|| true' + the reporter's own guards keep a wandb hiccup from
    # failing an otherwise-successful eval. The token is embedded like HF_TOKEN
    # below (same container_cmd.sh) -- keep RESULTS_DIR private.
    WANDB_CLAUSE="&& { ${WANDB_MODE_EXPORT}export WANDB_API_KEY='${WANDB_TOKEN}'; python /code/scripts/eval_wandb_report.py --project '${WANDB_EVAL_PROJECT}' --run_name '${WANDB_RUN_NAME}' --results_dir '${RESULTS_DIR}' --group '${EXP_NAME}' --job_type '${BACKEND}' 2>&1 | tee '${RESULTS_DIR}/wandb_report.log' || true; }"
fi

# ---- Record this run's configuration under the (timestamped) eval folder ----
{
    echo "# SpeechLM leaderboard eval run config"
    echo "timestamp:            ${RUN_TS}"
    echo "slurm_job_id:         ${JOB_TAG}"
    echo "eval_tag:             ${EVAL_TAG:-}"
    echo "exp_name:             ${EXP_NAME}"
    echo "project:              ${PROJECT}"
    echo "backend:              ${BACKEND}"
    echo "model_class:          ${MODEL_CLASS}"
    [[ -n "$PRETRAINED_LLM_OVERRIDE" ]] && echo "pretrained_llm_override: ${PRETRAINED_LLM_OVERRIDE}"
    [[ -n "$PRETRAINED_ASR_OVERRIDE" ]] && echo "pretrained_asr_override: ${PRETRAINED_ASR_OVERRIDE}"
    echo "checkpoint:           ${CKPT}"
    echo "run_averaging:        ${RUN_AVERAGING}"
    echo "force_average:        ${FORCE_AVERAGE}"
    echo "ensure_avg:           ${ENSURE_AVG}"
    if [[ "$RUN_AVERAGING" == "1" ]]; then
        echo "averaged_num_inputs:  ${#_AVG_IN[@]}"
        echo "averaged_inputs:"
        for f in "${_AVG_IN[@]}"; do echo "  - ${f}"; done
    fi
    echo "hf_model_dir:         ${HF_CKPT_DIR}"
    echo "force_convert:        ${FORCE_CONVERT}"
    echo "system_prompt:        \"${SYSTEM_PROMPT}\""
    echo "chunk_size:           ${CHUNK_SIZE:-<model default>}"
    echo "self_correct:         $( [[ "$SELF_CORRECT" == 1 || "$SELF_CORRECT" == true ]] && echo 'true (locked/corrected stream)' || echo 'false (non-corrective j=0 stream)')"
    echo "batch_size:           ${BATCH_SIZE}"
    echo "max_new_tokens:       ${MAX_NEW_TOKENS}"
    if [[ "$BACKEND" == "heh" ]]; then
        echo "heh_max_new_tokens:   ${HEH_MAX_NEW_TOKENS}"
        echo "heh_use_state_machine: ${HEH_USE_STATE_MACHINE}"
        echo "heh_use_offline_embs: ${HEH_USE_OFFLINE_EMBS}"
        echo "heh_pad_duration:     ${HEH_PAD_DURATION}"
        echo "save_raw:             ${SAVE_RAW}"
    fi
    echo "num_gpus:             ${NGPU}"
    echo "shuffle_seed:         ${SHUFFLE_SEED}"
    echo "max_eval_samples:     ${MAX_EVAL_SAMPLES}"
    if [[ -n "$LONGFORM_DIR" ]]; then
        echo "longform_dir:         ${LONGFORM_DIR}"
        echo "longform_window_sec:  ${LONGFORM_WINDOW_SEC}"
        echo "longform_stratified:  ${LONGFORM_STRATIFIED}"
        if [[ "${LONGFORM_STRATIFIED}" == "1" ]]; then
            echo "longform_per_range:   ${LONGFORM_PER_RANGE}"
            echo "longform_max_range_min: ${LONGFORM_MAX_RANGE_MIN}"
        fi
    else
        echo "datasets:             ${DATASETS_CSV}"
        echo "cache_dir:            ${CACHE_DIR}"
    fi
    echo "results_dir:          ${RESULTS_DIR}"
    echo "report_wandb:         ${DO_WANDB}"
    echo "wandb_eval_project:   ${WANDB_EVAL_PROJECT}"
    echo "wandb_run_name:       ${WANDB_RUN_NAME}"
} > "${RESULTS_DIR}/run_config.yaml"
echo "==> Wrote run config: ${RESULTS_DIR}/run_config.yaml"

# System prompt may contain spaces / apostrophes (e.g. "chunk's"). Embedding it
# as system_prompt='...' breaks Hydra on the apostrophe; bash -c "..." also
# fights with nested quotes. Write the raw prompt + a Hydra override line to
# files and have the container command read them.
printf '%s' "$SYSTEM_PROMPT" > "${SHARD_DIR}/system_prompt.txt"
python3 -c "
from pathlib import Path
p = Path(r'${SHARD_DIR}/system_prompt.txt').read_text()
e = p.replace('\\\\', '\\\\\\\\').replace('\"', '\\\\\"')
Path(r'${SHARD_DIR}/system_prompt.hydra_override').write_text('system_prompt=\"' + e + '\"')
"
echo "==> Wrote system prompt override: ${SHARD_DIR}/system_prompt.hydra_override"

# ---- Shared-prep helpers (mkdir locks; safe across concurrent jobs/nodes) ----
# Written into SHARD_DIR and invoked inside the container. mkdir is atomic on
# lustre; flock is not reliably cross-client. Writers stage to a temp path then
# mv/rename into place so waiters never observe a partial file.
ENSURE_AVG_SH="${SHARD_DIR}/ensure_averaged_ckpt.sh"
ENSURE_HF_SH="${SHARD_DIR}/ensure_hf_model.sh"

if [[ "$ENSURE_AVG" == "1" ]]; then
    cat > "$ENSURE_AVG_SH" <<AVG_EOF
#!/bin/bash
set -euo pipefail
OUT='${CKPT}'
FORCE='${FORCE_AVERAGE}'
LOCK="\${OUT}.preparing"
TMP="\${OUT}.tmp.\$\$"
STALE_SEC=7200  # steal lock if holder died mid-write (>2h)
acquire() {
  while ! mkdir "\$LOCK" 2>/dev/null; do
    if [[ -d "\$LOCK" ]]; then
      # portable mtime age (busybox/gnu)
      lock_mtime=\$(stat -c %Y "\$LOCK" 2>/dev/null || stat -f %m "\$LOCK" 2>/dev/null || echo 0)
      now=\$(date +%s)
      if [[ "\$lock_mtime" -gt 0 && \$(( now - lock_mtime )) -gt \$STALE_SEC ]]; then
        echo "==> stale avg lock (>\${STALE_SEC}s); removing \$LOCK"
        rmdir "\$LOCK" 2>/dev/null || rm -rf "\$LOCK"
        continue
      fi
    fi
    echo "==> waiting for another job to finish averaging -> \$OUT"
    for _ in \$(seq 1 60); do
      if [[ "\$FORCE" != "1" && -f "\$OUT" ]]; then
        echo "==> averaged ckpt ready (reusing): \$OUT"
        exit 0
      fi
      [[ -d "\$LOCK" ]] || break
      sleep 10
    done
  done
}
release() { rmdir "\$LOCK" 2>/dev/null || true; }
acquire
trap release EXIT
if [[ "\$FORCE" == "1" || ! -f "\$OUT" ]]; then
  echo "==> Averaging ${#_AVG_IN[@]} checkpoints -> \$OUT"
  python /code/scripts/average_sslm_ckpts.py --output "\$TMP"${AVG_INPUTS_QUOTED}
  mv -f "\$TMP" "\$OUT"
  echo "==> wrote \$OUT"
else
  echo "==> Reusing cached averaged checkpoint: \$OUT"
fi
release
trap - EXIT
AVG_EOF
    chmod +x "$ENSURE_AVG_SH"
fi

if [[ "$BACKEND" == "heh" ]]; then
    cat > "$ENSURE_HF_SH" <<HF_EOF
#!/bin/bash
set -euo pipefail
HF_DIR='${HF_CKPT_DIR}'
CKPT='${CKPT}'
FORCE='${FORCE_CONVERT}'
MODEL_CLASS='${MODEL_CLASS}'
EXP_CFG_LOCAL='${SHARD_DIR}/exp_config.local.yaml'
LOCK="\${HF_DIR}.preparing"
STALE_SEC=7200
acquire() {
  while ! mkdir "\$LOCK" 2>/dev/null; do
    if [[ -d "\$LOCK" ]]; then
      lock_mtime=\$(stat -c %Y "\$LOCK" 2>/dev/null || stat -f %m "\$LOCK" 2>/dev/null || echo 0)
      now=\$(date +%s)
      if [[ "\$lock_mtime" -gt 0 && \$(( now - lock_mtime )) -gt \$STALE_SEC ]]; then
        echo "==> stale HF lock (>\${STALE_SEC}s); removing \$LOCK"
        rmdir "\$LOCK" 2>/dev/null || rm -rf "\$LOCK"
        continue
      fi
    fi
    echo "==> waiting for another job to finish HF convert -> \$HF_DIR"
    for _ in \$(seq 1 60); do
      if [[ "\$FORCE" != "1" && -f "\${HF_DIR}/config.json" ]]; then
        # Reuse only if HF is at least as new as the ckpt (covers the case where
        # a sibling job just re-averaged and we must not keep a stale HF tree).
        if [[ ! -f "\$CKPT" ]] || [[ ! "\$CKPT" -nt "\${HF_DIR}/config.json" ]]; then
          echo "==> HF model ready (reusing): \$HF_DIR"
          exit 0
        fi
      fi
      [[ -d "\$LOCK" ]] || break
      sleep 10
    done
  done
}
release() { rmdir "\$LOCK" 2>/dev/null || true; }
acquire
trap release EXIT
NEED=0
if [[ "\$FORCE" == "1" || ! -f "\${HF_DIR}/config.json" ]]; then
  NEED=1
elif [[ -f "\$CKPT" && "\$CKPT" -nt "\${HF_DIR}/config.json" ]]; then
  echo "==> ckpt newer than HF cache; rebuilding"
  NEED=1
fi
if [[ "\$NEED" == "1" ]]; then
  echo "==> Converting ckpt -> HF format: \$HF_DIR"
  OVERRIDE_PRETRAINED_LLM='${PRETRAINED_LLM_OVERRIDE}' OVERRIDE_PRETRAINED_ASR='${PRETRAINED_ASR_OVERRIDE}' \
    python /code/scripts/patch_exp_config.py '${EXP_CFG}' "\$EXP_CFG_LOCAL" "\$CKPT"
  TMP_HF="\${HF_DIR}.tmp.\$\$"
  rm -rf "\$TMP_HF"
  CUDA_VISIBLE_DEVICES=0 python /code/examples/speechlm2/to_hf.py \\
    class_path="\$MODEL_CLASS" \\
    ckpt_path="'\$CKPT'" \\
    ckpt_config="'\$EXP_CFG_LOCAL'" \\
    output_dir="'\$TMP_HF'" \\
    weights_only=false
  [[ -f "\${TMP_HF}/config.json" ]] || { echo "ERROR: to_hf.py did not produce \${TMP_HF}/config.json"; exit 1; }
  rm -rf "\$HF_DIR"
  mv -f "\$TMP_HF" "\$HF_DIR"
  echo "==> wrote \$HF_DIR"
else
  echo "==> Reusing existing HF model: \$HF_DIR"
fi
release
trap - EXIT
HF_EOF
    chmod +x "$ENSURE_HF_SH"
fi

AVG_CLAUSE=""
[[ "$ENSURE_AVG" == "1" ]] && AVG_CLAUSE="&& bash '${ENSURE_AVG_SH}' "
HF_CLAUSE=""
[[ "$BACKEND" == "heh" ]] && HF_CLAUSE="&& bash '${ENSURE_HF_SH}' "

if [[ "$BACKEND" == "heh" ]]; then
# ===== heh backend: convert ckpt -> HF once, then decode each shard with =====
# =====             streaming_stt_generate.py (state-machine inference). =====
read -r -d '' cmd <<EOF
echo "*******SpeechLM leaderboard eval (heh engine, pooled shards over ${NGPU} GPUs)********" \
&& echo "*** EXP=${EXP_NAME} | MODEL_CLASS=${MODEL_CLASS} ***" \
&& echo "*** CKPT=${CKPT} ***" \
&& echo "*** CACHE_DIR=${CACHE_DIR} | CHUNK_SIZE=${CHUNK_SIZE:-<default>} | BATCH_SIZE=${BATCH_SIZE} | SEED=${SHUFFLE_SEED} ***" \
&& echo "*** heh: max_tok=${HEH_MAX_NEW_TOKENS} state_machine=${HEH_USE_STATE_MACHINE} offline=${HEH_USE_OFFLINE_EMBS} pad=${HEH_PAD_DURATION}s ***" \
&& echo "*** system_prompt (auto-set for ${MODEL_CLASS##*.}): [\$(cat '${SHARD_DIR}/system_prompt.txt')] ***" \
&& nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
&& cd /code \
&& git rev-parse HEAD \
&& export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& export OMP_NUM_THREADS=1 \
&& export HF_HOME="/hfcache/" \
&& export HF_TOKEN=${HF_TOKEN} \
&& export HF_HUB_OFFLINE=1 \
&& export TOKENIZERS_PARALLELISM=false \
&& export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
&& export HYDRA_FULL_ERROR=1 \
&& export TMPDIR=${OCI_TMP_DIR} && mkdir -p ${OCI_TMP_DIR} \
&& python -c "import nemo, nemo.collections.speechlm2; print('NeMo:', nemo.__file__)" \
${AVG_CLAUSE} \
${HF_CLAUSE} \
&& [ -f "${HF_CKPT_DIR}/config.json" ] || { echo "ERROR: HF model missing at ${HF_CKPT_DIR}/config.json"; exit 1; } \
&& echo "==> Building ${NGPU} shard manifests ..." \
&& ${BUILD_INVOCATION} \
&& rm -f ${SHARD_DIR}/shard*_of*.generations.jsonl \
&& echo "Fanning ${NGPU} shards across ${NGPU} GPUs with the heh engine..." \
&& SP_HYDRA=\$(cat '${SHARD_DIR}/system_prompt.hydra_override') \
&& pids=() \
&& for gpu in \$(seq 0 \$(( ${NGPU} - 1 ))); do \
      log="${RESULTS_DIR}/shard_\${gpu}.log"; \
      man="${SHARD_DIR}/shard\${gpu}_of${NGPU}.json"; \
      gen="${SHARD_DIR}/shard\${gpu}_of${NGPU}.generations.jsonl"; \
      echo "  [gpu \$gpu] \${man} -> \${log}"; \
      CUDA_VISIBLE_DEVICES=\$gpu python /code/examples/speechlm2/streaming_stt_generate.py \
        pretrained_name="'${HF_CKPT_DIR}'" \
        model_class="${MODEL_CLASS}" \
        inputs="'\${man}'" \
        batch_size=${BATCH_SIZE} \
        max_new_tokens=${HEH_MAX_NEW_TOKENS} \
        "\$SP_HYDRA" \
        use_offline_embs=${HEH_USE_OFFLINE_EMBS} \
        use_state_machine_inference=${HEH_USE_STATE_MACHINE} \
        pad_extra_duration=${HEH_PAD_DURATION} \
        report_word_latency=${REPORT_LATENCY} \
        save_raw=${SAVE_RAW} \
        ${SELF_CORRECT_HEH} \
        output_manifest="'\${gen}'" \
        verbose=false \
        device=cuda \
        ${CHUNK_SIZE:+chunk_size=${CHUNK_SIZE}} \
        > "\${log}" 2>&1 & \
      pids+=(\$!); \
   done \
&& fail=0 && for p in "\${pids[@]}"; do wait "\$p" || fail=1; done \
&& echo "" \
&& echo "==================== Leaderboard WER (heh) ====================" \
&& ${AGG_INVOCATION} 2>&1 | tee "${RESULTS_DIR}/aggregate.log" \
${WANDB_CLAUSE} \
&& echo "" \
&& echo "Per-shard logs: ${RESULTS_DIR} | manifests+generations: ${SHARD_DIR}" \
&& exit \$fail
EOF

else
# ===== sslm backend: in-process pooled-shard driver (tqdm bars, OOM backoff) =====
read -r -d '' cmd <<EOF
echo "*******SpeechLM leaderboard eval (sslm in-process, pooled shards over ${NGPU} GPUs)********" \
&& echo "*** EXP=${EXP_NAME} | MODEL_CLASS=${MODEL_CLASS} ***" \
&& echo "*** CKPT=${CKPT} ***" \
&& echo "*** CACHE_DIR=${CACHE_DIR} | CHUNK_SIZE=${CHUNK_SIZE:-<default>} | BATCH_SIZE=${BATCH_SIZE} | SEED=${SHUFFLE_SEED} ***" \
&& echo "*** system_prompt (auto-set for ${MODEL_CLASS##*.}): [\$(cat '${SHARD_DIR}/system_prompt.txt')] ***" \
&& nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
&& cd /code \
&& git rev-parse HEAD \
&& export PYTHONPATH="/code/.:\${PYTHONPATH}" \
&& export OMP_NUM_THREADS=1 \
&& export HF_HOME="/hfcache/" \
&& export HF_TOKEN=${HF_TOKEN} \
&& export HF_HUB_OFFLINE=1 \
&& export TOKENIZERS_PARALLELISM=false \
&& export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
&& export HYDRA_FULL_ERROR=1 \
&& export TMPDIR=${OCI_TMP_DIR} && mkdir -p ${OCI_TMP_DIR} \
&& python -c "import nemo, nemo.collections.speechlm2; print('NeMo:', nemo.__file__)" \
${AVG_CLAUSE} \
&& echo "Pooled datasets: ${DATASETS_CSV}" \
&& echo "Fanning ${NGPU} balanced shards across ${NGPU} GPUs (seed=${SHUFFLE_SEED})..." \
&& rm -f ${RESULTS_DIR}/shard*_of*.generations.jsonl \
&& SP_TEXT=\$(cat '${SHARD_DIR}/system_prompt.txt') \
&& pids=() \
&& for gpu in \$(seq 0 \$(( ${NGPU} - 1 ))); do \
      log="${RESULTS_DIR}/shard_\${gpu}.log"; \
      echo "  [gpu \$gpu] shard \$gpu/${NGPU} -> \${log}"; \
      python /code/scripts/speechlm_leaderboard_eval.py \
        --ckpt_path "${CKPT}" \
        --model_class "${MODEL_CLASS}" \
        --datasets "${DATASETS_CSV}" \
        --num_shards ${NGPU} \
        --shard_index \$gpu \
        --shuffle_seed ${SHUFFLE_SEED} \
        --device \$gpu \
        --cache_dir "${CACHE_DIR}" \
        --batch_size ${BATCH_SIZE} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --max_eval_samples ${MAX_EVAL_SAMPLES} \
        --system_prompt "\$SP_TEXT" \
        --output_dir "${RESULTS_DIR}" \
        ${CHUNK_SIZE:+--chunk_size ${CHUNK_SIZE}} \
        ${SELF_CORRECT_SSLM} \
        > "\${log}" 2>&1 & \
      pids+=(\$!); \
   done \
&& fail=0 && for p in "\${pids[@]}"; do wait "\$p" || fail=1; done \
&& echo "" \
&& echo "==================== Leaderboard WER ====================" \
&& python /code/scripts/speechlm_leaderboard_eval.py --aggregate --output_dir "${RESULTS_DIR}" 2>&1 | tee "${RESULTS_DIR}/aggregate.log" \
${WANDB_CLAUSE} \
&& echo "" \
&& echo "Per-shard logs + generations under: ${RESULTS_DIR}" \
&& exit \$fail
EOF
fi

# Run via a script file (not bash -c "...") so nested quotes in the prompt /
# hydra override cannot break the outer shell quoting.
CMD_FILE="${RESULTS_DIR}/container_cmd.sh"
printf '%s\n' "$cmd" > "$CMD_FILE"
chmod +x "$CMD_FILE"
echo "==> Container command: ${CMD_FILE}"

srun -o "$OUTFILE" -e "$ERRFILE" --container-image="$CONTAINER" $MOUNTS bash "$CMD_FILE"

set +x
