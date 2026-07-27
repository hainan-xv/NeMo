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
#   MODEL_CLASS       model class (default ChunkCompletionSTTModel)
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

mkdir -p slurm_out
SLURM_ACCOUNT='llmservice'
USERID='users/hainanx'
OLDUSERID='users/heh'
LUSTRE_ACCOUNT_PREFIX=/lustre/fsw/portfolios/${SLURM_ACCOUNT}

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh}"

PROJECT="${PROJECT:-Speechlm79}"
EXP_NAME="${1:-${EXP_NAME:-granary2_chunkcompletion}}"
BACKEND="${BACKEND:-heh}"
MODEL_CLASS="${MODEL_CLASS:-nemo.collections.speechlm2.models.chunk_completion_model.ChunkCompletionSTTModel}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-Transcribe the audio into text.}"
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
# Report per-word emission latency (proxy: end-of-chunk time of each word's last
# subword, averaged). Chunk-completion only; default on for it, off otherwise.
if [[ -z "${REPORT_LATENCY:-}" ]]; then
    case "$MODEL_CLASS" in
        *ChunkCompletionSTTModel) REPORT_LATENCY=true ;;
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
CACHE_DIR="${CACHE_DIR:-/lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache}"
if [[ ! -d "$CACHE_DIR" ]]; then
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
# lustre; to_hf.py needs it to instantiate the model class.
EXP_CFG="${CKPT_DIR%/checkpoints}/exp_config.yaml"
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
    echo "batch_size:           ${BATCH_SIZE}"
    echo "max_new_tokens:       ${MAX_NEW_TOKENS}"
    if [[ "$BACKEND" == "heh" ]]; then
        echo "heh_max_new_tokens:   ${HEH_MAX_NEW_TOKENS}"
        echo "heh_use_state_machine: ${HEH_USE_STATE_MACHINE}"
        echo "heh_use_offline_embs: ${HEH_USE_OFFLINE_EMBS}"
        echo "heh_pad_duration:     ${HEH_PAD_DURATION}"
    fi
    echo "num_gpus:             ${NGPU}"
    echo "shuffle_seed:         ${SHUFFLE_SEED}"
    echo "max_eval_samples:     ${MAX_EVAL_SAMPLES}"
    echo "datasets:             ${DATASETS_CSV}"
    echo "cache_dir:            ${CACHE_DIR}"
    echo "results_dir:          ${RESULTS_DIR}"
} > "${RESULTS_DIR}/run_config.yaml"
echo "==> Wrote run config: ${RESULTS_DIR}/run_config.yaml"

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
&& echo "==> Building ${NGPU} pooled shard manifests (seed=${SHUFFLE_SEED}) ..." \
&& python /code/scripts/leaderboard_heh_shards.py build \
      --cache_dir "${CACHE_DIR}" \
      --datasets "${DATASETS_CSV}" \
      --out_dir "${SHARD_DIR}" \
      --num_shards ${NGPU} \
      --shuffle_seed ${SHUFFLE_SEED} \
      --max_eval_samples ${MAX_EVAL_SAMPLES} \
&& rm -f ${SHARD_DIR}/shard*_of*.generations.jsonl \
&& echo "Fanning ${NGPU} shards across ${NGPU} GPUs with the heh engine..." \
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
        system_prompt="'${SYSTEM_PROMPT}'" \
        use_offline_embs=${HEH_USE_OFFLINE_EMBS} \
        use_state_machine_inference=${HEH_USE_STATE_MACHINE} \
        pad_extra_duration=${HEH_PAD_DURATION} \
        report_word_latency=${REPORT_LATENCY} \
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
&& python /code/scripts/leaderboard_heh_shards.py aggregate --out_dir "${SHARD_DIR}" 2>&1 | tee "${RESULTS_DIR}/aggregate.log" \
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
        --system_prompt "${SYSTEM_PROMPT}" \
        --output_dir "${RESULTS_DIR}" \
        ${CHUNK_SIZE:+--chunk_size ${CHUNK_SIZE}} \
        > "\${log}" 2>&1 & \
      pids+=(\$!); \
   done \
&& fail=0 && for p in "\${pids[@]}"; do wait "\$p" || fail=1; done \
&& echo "" \
&& echo "==================== Leaderboard WER ====================" \
&& python /code/scripts/speechlm_leaderboard_eval.py --aggregate --output_dir "${RESULTS_DIR}" 2>&1 | tee "${RESULTS_DIR}/aggregate.log" \
&& echo "" \
&& echo "Per-shard logs + generations under: ${RESULTS_DIR}" \
&& exit \$fail
EOF
fi

srun -o "$OUTFILE" -e "$ERRFILE" --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"

set +x
