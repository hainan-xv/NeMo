#!/bin/bash
set -e
set -o pipefail

#
# Eval a StreamingSTTModel checkpoint on the Open ASR Leaderboard datasets.
#
# This script lives at the NeMo repo root *intentionally* -- it must NOT be
# under speechllm_oci/ because that directory's sync.sh pulls *.sh files
# from a remote and silently overwrites this script's bug fixes.  Keep it
# here so local edits survive.
#
# Usage:
#   ./eval_leaderboard_ord.sh <EXP_NAME> [STEP] [DEVICE_ID]
#
# When STEP is omitted, the *best* (non '-last') checkpoint is picked
# automatically -- this matches what ModelCheckpoint(save_top_k=1, monitor=...)
# writes as the val-WER snapshot.  Set USE_LAST=1 to fall back to -last.ckpt
# instead (useful very early in training when no best snapshot exists yet).
#
# Examples:
#   # Auto-pick best (val-WER) checkpoint
#   ./eval_leaderboard_ord.sh oci_streaming_stt_..._full_context
#
#   # Force using the last checkpoint
#   USE_LAST=1 ./eval_leaderboard_ord.sh oci_streaming_stt_..._full_context
#
#   # Specify a step explicitly
#   ./eval_leaderboard_ord.sh oci_streaming_stt_..._full_context 8000
#
#   # Skip download if checkpoint already exists locally
#   SKIP_DOWNLOAD=1 ./eval_leaderboard_ord.sh oci_streaming_stt_..._full_context 8000
#
#   # Force a specific project dir on ORD (otherwise tries defaults in order)
#   PROJECT=Streaming_SLM_chunk14 ./eval_leaderboard_ord.sh oci_streaming_stt_..._im_end_prompt
#
#   # For two-pass refinement models: pick which pass scores the headline WER.
#   # Defaults to 'streaming' because checkpoints trained before the
#   # refinement-EOS-supervision fix produce truncated / paraphrased refined
#   # outputs.  Override with USE_PASS=refined once retrained.
#   USE_PASS=streaming ./eval_leaderboard_ord.sh oci_..._2pass_v2
#
#   # Re-evaluate just one (or a few) datasets without touching previously
#   # recorded results -- e.g. to refresh the tedlium number after a fix.
#   # ONLY accepts a comma-separated list of "dataset[:split]" entries; the
#   # split defaults to whatever appears in the master DATASETS list.
#   # Partial runs write to eval_log_only_<datasets>.txt and emit a separate
#   # summary so the full-run eval_log.txt is left untouched.
#   ONLY=tedlium ./eval_leaderboard_ord.sh oci_..._full_context
#   ONLY=tedlium:test,librispeech:test.clean ./eval_leaderboard_ord.sh ...
#
#   # Average every non '-last' checkpoint on the grid (i.e. all the top-k
#   # val_wer snapshots), then evaluate the averaged model.  Useful for
#   # squeezing a little extra WER out of late-stage training.  The averaged
#   # ckpt is cached at checkpoints/<EXP>/avg<N>.ckpt and reused on subsequent
#   # runs unless FORCE_AVERAGE=1 is set.
#   RUN_AVERAGING=1 ./eval_leaderboard_ord.sh oci_..._full_context
#   RUN_AVERAGING=1 FORCE_DOWNLOAD=1 FORCE_AVERAGE=1 ./eval_leaderboard_ord.sh ...
#
#   # Parallel multi-token chunk decoding (requires a checkpoint trained with
#   # parallel_loss_weight > 0).  Uses ParallelChunkHeads to emit up to K
#   # tokens in parallel per chunk instead of the AR depth loop.  Results
#   # land in a separate eval_results/<EXP>_step<STEP>_parDec/ dir so AR and
#   # parallel numbers don't overwrite each other.
#   PARALLEL_DECODE=1 ./eval_leaderboard_ord.sh oci_..._parHeads10_w0.5
#

# ---------- ORD connection ----------
REMOTE_HOST="cs-oci-ord-login-01.nvidia.com"
REMOTE_USER="hainanx"
SSH_KEY="$HOME/.ssh/draco-rno"
REMOTE_RESULTS_ROOT="/lustre/fsw/portfolios/llmservice/users/hainanx/results"
# Project-name directory under ${REMOTE_RESULTS_ROOT}.  Different training
# scripts use different project names (e.g. Streaming_SLM_chunk14 for the
# chunk-14 experiments).  We try them in order and pick the first one that
# actually contains the experiment.  Override with PROJECT=foo to force one.
PROJECT_CANDIDATES_DEFAULT=("Streaming_SLM_ord2")
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no"

# ---------- Local paths ----------
NEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
# Keep the eval launcher and driver at repo root so speechllm_oci/sync.sh
# cannot silently overwrite local eval fixes.
EVAL_DIR="${NEMO_ROOT}"
LOCAL_CKPT_DIR="${EVAL_DIR}/checkpoints"
RUN_EVAL_PY="${EVAL_DIR}/run_eval_sslm.py"

if [ ! -f "$RUN_EVAL_PY" ]; then
    echo "ERROR: cannot find eval driver at ${RUN_EVAL_PY}" >&2
    exit 1
fi

export PYTHONPATH="${NEMO_ROOT}:${PYTHONPATH}"

# ---------- Arguments ----------
EXP_NAME="${1:?Usage: $0 <EXP_NAME> [STEP] [DEVICE_ID]}"
STEP="${2:-}"
DEVICE_ID="${3:-0}"

# Allow STEP="last" (or "LAST") as a friendly alias for USE_LAST=1. New
# checkpoint filenames now embed val_wer (e.g. step=12000-val_wer=0.1787-last.ckpt),
# so a literal STEP=last would otherwise try to scp ``step=last.ckpt`` and fail.
if [ -n "$STEP" ]; then
    _step_lc=$(echo "$STEP" | tr '[:upper:]' '[:lower:]')
    if [ "$_step_lc" = "last" ]; then
        echo "==> STEP='${STEP}' → routing through USE_LAST=1"
        USE_LAST=1
        STEP=""
    fi
fi
BATCH_SIZE="${BATCH_SIZE:-128}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# MAX_NEW_TOKENS is the *per-chunk* cap; the model's
# ``_generate_chunked_streaming`` multiplies it by
# ``inference_audio_chunks_per_turn`` internally, so callers don't need to.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-0}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
USE_OFFLINE_EMBS="${USE_OFFLINE_EMBS:-0}"
USE_GENERATION_HISTORY="${USE_GENERATION_HISTORY:-0}"
INFERENCE_AUDIO_CHUNKS_PER_TURN="${INFERENCE_AUDIO_CHUNKS_PER_TURN:-1}"
DISABLE_MODALITY_POSITION_IDS="${DISABLE_MODALITY_POSITION_IDS:-0}"
# Cap the number of samples evaluated per dataset (forwarded to
# run_eval_sslm.py as --max_eval_samples). Useful for fast iteration on a
# subset, e.g. when isolating parallel-decode regressions. Unset / 0 = full.
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
# Parallel multi-token chunk decoding via ParallelChunkHeads. Requires a ckpt
# trained with parallel_loss_weight > 0. Tagged with _parDec in the results
# dir so AR and parallel numbers don't clobber each other.
PARALLEL_DECODE="${PARALLEL_DECODE:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
# RUN_AVERAGING=1: download every non '-last' checkpoint from the grid and
# average their state_dicts into a single ckpt for eval.  See header docstring.
RUN_AVERAGING="${RUN_AVERAGING:-0}"
QUICK_TEST="${QUICK_TEST:-0}"
# For two-pass refinement models: which pass scores the headline WER?
# Streaming-pass is the safer default until the refinement-EOS-supervision
# fix lands in retrained checkpoints (see streaming_stt_dataset.py).
USE_PASS="${USE_PASS:-streaming}"
# Optional filter: comma-separated "dataset[:split]" entries to evaluate.
# Empty = run the full leaderboard suite.
ONLY="${ONLY:-}"
# TED-LIUM was deleted from hf-audio/open-asr-leaderboard (alias of
# hf-audio/esb-datasets-test-only-sorted) on 2026-05-27 (PR #9).  We pin to
# the last commit that still has the parquet so historical TED-LIUM numbers
# remain reproducible.  Override with TEDLIUM_REVISION=<sha|branch> if needed,
# or TEDLIUM_REVISION="" to disable pinning (will yield 0 samples until HF
# restores the dataset).
TEDLIUM_REVISION="${TEDLIUM_REVISION:-20a009a}"

# Resolve which project directory hosts this experiment.
if [ -n "${PROJECT:-}" ]; then
    PROJECT_CANDIDATES=("$PROJECT")
else
    PROJECT_CANDIDATES=("${PROJECT_CANDIDATES_DEFAULT[@]}")
fi

REMOTE_CKPT_DIR=""
for proj in "${PROJECT_CANDIDATES[@]}"; do
    candidate="${REMOTE_RESULTS_ROOT}/${proj}/${EXP_NAME}/${EXP_NAME}/checkpoints"
    if ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "[ -d '${candidate}' ]" 2>/dev/null; then
        REMOTE_CKPT_DIR="$candidate"
        REMOTE_PROJECT="$proj"
        echo "==> Resolved project: ${proj}"
        break
    fi
done
if [ -z "$REMOTE_CKPT_DIR" ]; then
    echo "ERROR: experiment '${EXP_NAME}' not found under any of:" >&2
    for proj in "${PROJECT_CANDIDATES[@]}"; do
        echo "       ${REMOTE_RESULTS_ROOT}/${proj}/${EXP_NAME}/${EXP_NAME}/checkpoints" >&2
    done
    echo "       Override with PROJECT=<name> to specify a different project dir." >&2
    exit 1
fi

# ---------- Step 1: Download checkpoint ----------
#
# Two checkpoint filename schemes coexist in the wild on the grid:
#
#   - Legacy (monitor=val_loss, default filename):
#         step=NNNNN-last.ckpt   <- rolling last
#         step=MMMMM.ckpt        <- best by monitored metric
#
#   - New (monitor=val_wer + filename='{step}-{val_wer:.4f}'):
#         step=NNNNN-val_wer=0.XXXX-last.ckpt   <- rolling last
#         step=MMMMM-val_wer=0.YYYY.ckpt        <- top-k by val_wer
#
# When STEP is unspecified we prefer the *best* checkpoint over -last:
#   1) If any file carries a "val_wer=" tag, pick the one with the lowest WER.
#   2) Otherwise, fall back to most-recent-by-mtime (works for the legacy
#      single-best layout where there is exactly one "best" file).
# USE_LAST=1 forces -last.ckpt regardless.
#
# RUN_AVERAGING=1 branches into a separate flow that downloads every non
# '-last' ckpt and averages their state_dicts into checkpoints/<EXP>/avg<N>.ckpt
# (cached; FORCE_AVERAGE=1 to recompute).
if [ "$RUN_AVERAGING" = "1" ]; then
    if [ -n "$STEP" ]; then
        echo "WARNING: RUN_AVERAGING=1 ignores explicit STEP=$STEP" >&2
        STEP=""
    fi
    echo "==> RUN_AVERAGING=1: listing non '-last' checkpoints on ORD..."
    REMOTE_AVG_LIST_CMD="ls ${REMOTE_CKPT_DIR}/*.ckpt 2>/dev/null | grep -v -- '-last\\.ckpt$' | xargs -r -n1 basename"
    REMOTE_CKPT_FILES=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "$REMOTE_AVG_LIST_CMD")
    if [ -z "$REMOTE_CKPT_FILES" ]; then
        echo "ERROR: No non '-last' checkpoints found in ${REMOTE_CKPT_DIR}" >&2
        echo "       (the trainer's save_top_k snapshots are what gets averaged.)" >&2
        exit 1
    fi
    NUM_CKPTS=$(echo "$REMOTE_CKPT_FILES" | wc -l)
    echo "    Found ${NUM_CKPTS} checkpoint(s) to average:"
    echo "$REMOTE_CKPT_FILES" | sed 's/^/      - /'

    mkdir -p "${LOCAL_CKPT_DIR}/${EXP_NAME}"
    LOCAL_AVG_INPUTS=()
    while IFS= read -r fname; do
        [ -z "$fname" ] && continue
        local_path="${LOCAL_CKPT_DIR}/${EXP_NAME}/${fname}"
        remote_path="${REMOTE_CKPT_DIR}/${fname}"
        if [ -f "$local_path" ] && [ "${FORCE_DOWNLOAD:-0}" != "1" ]; then
            echo "==> Cached: ${fname} ($(du -h "$local_path" | cut -f1))"
        elif [ "$SKIP_DOWNLOAD" = "1" ]; then
            echo "ERROR: SKIP_DOWNLOAD=1 but missing: $local_path" >&2
            exit 1
        else
            echo "==> Downloading ${fname}..."
            if ! scp $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}:${remote_path}" "$local_path"; then
                echo "ERROR: scp failed for ${remote_path}" >&2
                [ -f "$local_path" ] && [ ! -s "$local_path" ] && rm -f "$local_path"
                exit 1
            fi
        fi
        LOCAL_AVG_INPUTS+=("$local_path")
    done <<< "$REMOTE_CKPT_FILES"

    STEP="avg${NUM_CKPTS}"
    CKPT_FILENAME="${STEP}.ckpt"
    LOCAL_CKPT_PATH="${LOCAL_CKPT_DIR}/${EXP_NAME}/${CKPT_FILENAME}"

    # Cache the averaged ckpt; recompute only on FORCE_AVERAGE=1 or input
    # changes (we re-average whenever any input is newer than the output).
    NEED_AVG=0
    if [ ! -f "$LOCAL_CKPT_PATH" ] || [ "${FORCE_AVERAGE:-0}" = "1" ]; then
        NEED_AVG=1
    else
        for src in "${LOCAL_AVG_INPUTS[@]}"; do
            if [ "$src" -nt "$LOCAL_CKPT_PATH" ]; then
                NEED_AVG=1
                break
            fi
        done
    fi

    if [ "$NEED_AVG" = "0" ]; then
        echo "==> Using cached averaged checkpoint: $LOCAL_CKPT_PATH"
        echo "    (set FORCE_AVERAGE=1 to recompute)"
    else
        echo "==> Averaging ${NUM_CKPTS} checkpoint(s) -> $LOCAL_CKPT_PATH"
        # Inline Python helper.  Floats are accumulated in fp64 for numerical
        # safety, then cast back to the original dtype.  Non-float tensors
        # (int counters, masks) are copied verbatim from the first ckpt --
        # averaging them is undefined.  The eval driver (run_eval_sslm.py)
        # only reads state_dict + hyper_parameters, so we drop optimizer /
        # scheduler / callback state (which is meaningless after averaging
        # and tends to mismatch in shape between ckpts).
        python3 - "$LOCAL_CKPT_PATH" "${LOCAL_AVG_INPUTS[@]}" <<'PYEOF'
import pickle as _pickle
import sys
import types as _types
from collections import OrderedDict

import torch

# ---------------------------------------------------------------------------
# Tolerant checkpoint loading.
#
# These checkpoints stash *training-only* objects in ``hyper_parameters``
# (notably ``hyper_parameters['forced_aligner']``, a
# ``nemo.collections.speechlm2.modules.qwen_forced_aligner.QwenForcedAligner``
# instance).  That module only ships in the training image and is absent from
# this clean checkout, so a vanilla ``torch.load`` dies with::
#
#     ModuleNotFoundError: No module named
#     'nemo.collections.speechlm2.modules.qwen_forced_aligner'
#
# Averaging only needs the tensors in ``state_dict`` (and, for the output,
# the primitive ``cfg``) -- never the aligner -- so we unpickle through an
# Unpickler that swaps any unimportable class for a throwaway stub.  The stubs
# are dropped before we re-save (see the hyper_parameters sanitization below),
# so the averaged ckpt is self-contained and loads with a plain ``torch.load``
# at eval time.
_stub_cache = {}


class _MissingClass:
    """Placeholder for classes whose defining module isn't importable here."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


def _make_stub(module, name):
    key = (module, name)
    cls = _stub_cache.get(key)
    if cls is None:
        cls = type(name, (_MissingClass,), {"__module__": module, "__qualname__": name})
        _stub_cache[key] = cls
    return cls


class _TolerantUnpickler(_pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            print(f"    (stubbing unimportable {module}.{name})", flush=True)
            return _make_stub(module, name)


_stub_pickle = _types.ModuleType("_stub_pickle")
_stub_pickle.__dict__.update(_pickle.__dict__)
_stub_pickle.Unpickler = _TolerantUnpickler


def _load_ckpt(path):
    return torch.load(
        path, map_location="cpu", weights_only=False, pickle_module=_stub_pickle
    )
# ---------------------------------------------------------------------------

out_path = sys.argv[1]
inputs = sys.argv[2:]
print(f"  averaging {len(inputs)} checkpoint(s) -> {out_path}", flush=True)

acc = None
orig_dtypes = None
template = None
n = 0
for p in inputs:
    print(f"    + loading {p}", flush=True)
    ckpt = _load_ckpt(p)
    sd = ckpt["state_dict"]
    if template is None:
        template = ckpt
        acc = OrderedDict()
        orig_dtypes = {}
        for k, v in sd.items():
            orig_dtypes[k] = v.dtype
            if torch.is_floating_point(v):
                acc[k] = v.detach().to(torch.float64).clone()
            else:
                acc[k] = v.detach().clone()
    else:
        missing = [k for k in acc if k not in sd]
        extra = [k for k in sd if k not in acc]
        if missing or extra:
            print(
                f"    ! key mismatch vs. first ckpt: "
                f"missing={len(missing)} extra={len(extra)} (using intersection)",
                flush=True,
            )
        for k in acc:
            if k not in sd:
                continue
            v = sd[k]
            if torch.is_floating_point(v):
                acc[k] = acc[k] + v.detach().to(torch.float64)
    n += 1

assert n > 0, "no checkpoints loaded"
print(f"  finalizing average over {n} checkpoint(s)...", flush=True)
final_sd = OrderedDict()
for k, v in acc.items():
    if torch.is_floating_point(v):
        final_sd[k] = (v / n).to(orig_dtypes[k])
    else:
        final_sd[k] = v

out = {"state_dict": final_sd}
for key in (
    "epoch",
    "global_step",
    "pytorch-lightning_version",
    "hparams_name",
):
    if key in template:
        out[key] = template[key]

# Sanitize hyper_parameters before saving.  The source checkpoints carry
# sibling entries ('forced_aligner', 'dataset_cls', 'data_cfg') that reference
# training-only modules; keeping them would re-pickle a dangling reference to
# nemo...qwen_forced_aligner and break a plain torch.load at eval time.  The
# eval driver (run_eval_sslm.py) only consumes hyper_parameters['cfg'], which
# is a plain primitive dict, so we keep exactly that and drop the rest.
hp = template.get("hyper_parameters")
if isinstance(hp, dict) and "cfg" in hp:
    out["hyper_parameters"] = {"cfg": hp["cfg"]}
elif hp is not None:
    out["hyper_parameters"] = hp

torch.save(out, out_path)
print(f"  wrote {out_path}", flush=True)
PYEOF
        echo "==> Average complete ($(du -h "$LOCAL_CKPT_PATH" | cut -f1))"
    fi
elif [ -z "$STEP" ]; then
    if [ "${USE_LAST:-0}" = "1" ]; then
        echo "==> No step specified (USE_LAST=1), finding most recent -last.ckpt on ORD..."
        REMOTE_LIST_CMD="ls -t ${REMOTE_CKPT_DIR}/*-last.ckpt"
        CKPT_FILENAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
            "${REMOTE_LIST_CMD} | head -1 | xargs -r basename")
    else
        echo "==> No step specified, finding best (non '-last') checkpoint on ORD..."
        # Prefer val_wer-tagged files; pick lowest WER. Fall back to mtime.
        # The remote awk extracts the numeric WER and sorts ascending.
        REMOTE_PICK_CMD="\
            files=\$(ls ${REMOTE_CKPT_DIR}/*.ckpt 2>/dev/null | grep -v -- '-last\\.ckpt$'); \
            wer_files=\$(echo \"\$files\" | grep -E 'val_wer=[0-9]+\\.[0-9]+' || true); \
            if [ -n \"\$wer_files\" ]; then \
                echo \"\$wer_files\" | awk -F'val_wer=' '{ print \$2, \$0 }' | sort -k1,1n | head -1 | awk '{ print \$2 }'; \
            else \
                echo \"\$files\" | head -1; \
            fi"
        CKPT_FILENAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
            "${REMOTE_PICK_CMD} | xargs -r basename")
    fi
    if [ -z "$CKPT_FILENAME" ]; then
        if [ "${USE_LAST:-0}" != "1" ]; then
            echo "    No best-WER checkpoint found; falling back to -last.ckpt..."
            CKPT_FILENAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" \
                "ls -t ${REMOTE_CKPT_DIR}/*-last.ckpt 2>/dev/null | head -1 | xargs -r basename")
        fi
    fi
    if [ -z "$CKPT_FILENAME" ]; then
        echo "ERROR: No checkpoints found in ${REMOTE_CKPT_DIR}" >&2
        exit 1
    fi
    # Derive a short STEP token for display + downstream path naming.
    # Supports both legacy (step=NNNN.ckpt) and new (step=NNNN-val_wer=0.XXXX.ckpt).
    STEP="${CKPT_FILENAME%.ckpt}"
    STEP="${STEP#step=}"
    # Drop any "-val_wer=..." suffix so we keep just the step number.
    STEP="${STEP%%-val_wer=*}"
    # Drop trailing "-last" tag for display purposes.
    STEP="${STEP%-last}"
    echo "    Found: ${CKPT_FILENAME} (step=${STEP})"
else
    # Explicit STEP=NNNN. Try the legacy bare filename first; if it doesn't
    # exist on the remote, glob for the val_wer-tagged variant
    # (step=NNNN-val_wer=0.XXXX.ckpt). This makes the explicit-step path
    # work for new-style checkpoints without forcing callers to type the
    # full filename including the WER.
    echo "==> Looking up STEP=${STEP} checkpoint on ORD..."
    REMOTE_STEP_LOOKUP="\
        if [ -f ${REMOTE_CKPT_DIR}/step=${STEP}.ckpt ]; then \
            echo step=${STEP}.ckpt; \
        else \
            ls ${REMOTE_CKPT_DIR}/step=${STEP}-*.ckpt 2>/dev/null \
                | grep -v -- '-last\\.ckpt$' \
                | head -1 \
                | xargs -r basename; \
        fi"
    CKPT_FILENAME=$(ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "${REMOTE_STEP_LOOKUP}")
    if [ -z "$CKPT_FILENAME" ]; then
        echo "ERROR: No checkpoint matching step=${STEP}*.ckpt found in ${REMOTE_CKPT_DIR}" >&2
        exit 1
    fi
    echo "    Found: ${CKPT_FILENAME}"
fi

# In RUN_AVERAGING mode LOCAL_CKPT_PATH was already set above and the
# averaged ckpt is on disk; skip the single-ckpt download path entirely.
if [ "$RUN_AVERAGING" != "1" ]; then
    REMOTE_CKPT_PATH="${REMOTE_CKPT_DIR}/${CKPT_FILENAME}"
    LOCAL_CKPT_PATH="${LOCAL_CKPT_DIR}/${EXP_NAME}/${CKPT_FILENAME}"

    # Cache-first download.  Trainer's save_top_k=1 means an older step's
    # checkpoint may have already been rotated off the remote -- a re-scp will
    # then 404 even though we already pulled the file once.  So:
    #   - If the local ckpt exists, use it by default (no scp attempt).
    #   - Set FORCE_DOWNLOAD=1 to overwrite the local copy from the remote.
    #   - SKIP_DOWNLOAD=1 is kept as a no-op alias for back-compat; it now
    #     additionally errors out if the local file is missing instead of
    #     silently scp'ing.
    if [ -f "$LOCAL_CKPT_PATH" ] && [ "${FORCE_DOWNLOAD:-0}" != "1" ]; then
        echo "==> Using cached local checkpoint (set FORCE_DOWNLOAD=1 to refresh): $LOCAL_CKPT_PATH"
    elif [ "$SKIP_DOWNLOAD" = "1" ]; then
        echo "ERROR: SKIP_DOWNLOAD=1 but no local checkpoint at: $LOCAL_CKPT_PATH" >&2
        echo "       Re-run without SKIP_DOWNLOAD to fetch from the remote." >&2
        exit 1
    else
        echo "==> Downloading checkpoint from ORD..."
        echo "    Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_PATH}"
        echo "    Local:  ${LOCAL_CKPT_PATH}"
        mkdir -p "$(dirname "$LOCAL_CKPT_PATH")"
        if ! scp $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CKPT_PATH}" "$LOCAL_CKPT_PATH"; then
            echo "ERROR: scp failed for ${REMOTE_CKPT_PATH}." >&2
            echo "       The remote may have rotated this step (trainer keeps save_top_k=1)." >&2
            echo "       If you already have the ckpt locally elsewhere, copy it to:" >&2
            echo "           ${LOCAL_CKPT_PATH}" >&2
            echo "       Or re-run without an explicit STEP to auto-pick the current best on the remote." >&2
            # Clean up a partial / truncated file scp may have left behind.
            if [ -f "$LOCAL_CKPT_PATH" ] && [ ! -s "$LOCAL_CKPT_PATH" ]; then
                rm -f "$LOCAL_CKPT_PATH"
            fi
            exit 1
        fi
        echo "==> Download complete ($(du -h "$LOCAL_CKPT_PATH" | cut -f1))"
    fi
fi

# ---------- Step 2: Run evaluation on all leaderboard datasets ----------
DATASETS=(
    "ami:test"
    "earnings22:test"
    "gigaspeech:test"
    "librispeech:test.clean"
    "librispeech:test.other"
    "spgispeech:test"
    "tedlium:test"
    "voxpopuli:test"
)

# Build a suffix that distinguishes runs with different eval-time knobs so
# different settings (e.g. chunks-per-turn) get their own results directory
# and don't silently overwrite each other.  An empty suffix preserves the
# default-knob directory name for backwards compatibility.
RUN_SUFFIX=""
if [ "$INFERENCE_AUDIO_CHUNKS_PER_TURN" != "1" ]; then
    RUN_SUFFIX="${RUN_SUFFIX}_chunks${INFERENCE_AUDIO_CHUNKS_PER_TURN}"
fi
if [ "$USE_OFFLINE_EMBS" = "1" ]; then
    RUN_SUFFIX="${RUN_SUFFIX}_offlineEmbs"
fi
if [ "$USE_GENERATION_HISTORY" = "1" ]; then
    RUN_SUFFIX="${RUN_SUFFIX}_genHist"
fi
if [ "$DISABLE_MODALITY_POSITION_IDS" = "1" ]; then
    RUN_SUFFIX="${RUN_SUFFIX}_noPosIds"
fi
if [ "$PARALLEL_DECODE" = "1" ]; then
    RUN_SUFFIX="${RUN_SUFFIX}_parDec"
fi
if [ -n "$USE_PASS" ] && [ "$USE_PASS" != "streaming" ]; then
    RUN_SUFFIX="${RUN_SUFFIX}_${USE_PASS}"
fi

RESULTS_DIR="${EVAL_DIR}/eval_results/${EXP_NAME}_step${STEP}${RUN_SUFFIX}"
mkdir -p "$RESULTS_DIR"

# Apply ONLY filter (comma-separated "dataset[:split]" list).  When set,
# partial runs write to a per-filter log file so the master eval_log.txt
# (which holds the full-suite numbers) is not clobbered.
if [ -n "$ONLY" ]; then
    declare -a _split_lookup
    for entry in "${DATASETS[@]}"; do
        _split_lookup[${#_split_lookup[@]}]="$entry"
    done
    declare -a _filtered=()
    IFS=',' read -r -a _only_entries <<< "$ONLY"
    for raw in "${_only_entries[@]}"; do
        raw="${raw// /}"
        [ -z "$raw" ] && continue
        if [[ "$raw" == *:* ]]; then
            _filtered+=("$raw")
            continue
        fi
        match=""
        for entry in "${_split_lookup[@]}"; do
            ds="${entry%%:*}"
            if [ "$ds" = "$raw" ]; then
                match="$match $entry"
            fi
        done
        if [ -z "$match" ]; then
            echo "ERROR: ONLY entry '$raw' does not match any dataset in DATASETS." >&2
            exit 1
        fi
        for m in $match; do
            _filtered+=("$m")
        done
    done
    DATASETS=("${_filtered[@]}")
    SAFE_ONLY=$(echo "$ONLY" | tr ',:' '__' | tr -cd 'A-Za-z0-9_-')
    EVAL_LOG="${RESULTS_DIR}/eval_log_only_${SAFE_ONLY}.txt"
    echo ""
    echo "==> ONLY filter active: running ${DATASETS[*]}"
    echo "    Log (separate from full-run eval_log.txt): ${EVAL_LOG}"
else
    EVAL_LOG="${RESULTS_DIR}/eval_log.txt"
fi

# Clear the (filtered or full) log so the summary only reflects this run.
> "$EVAL_LOG"

EXTRA_ARGS=()
if [ "$QUICK_TEST" = "1" ]; then
    DATASETS=("ami:test")
    EXTRA_ARGS+=(--max_eval_samples 10 --verbose)
    echo ""
    echo "==> QUICK TEST: 10 samples from ami/test only"
elif [ -n "$MAX_EVAL_SAMPLES" ] && [ "$MAX_EVAL_SAMPLES" != "0" ]; then
    EXTRA_ARGS+=(--max_eval_samples "$MAX_EVAL_SAMPLES")
    echo ""
    echo "==> MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES} per dataset"
fi
if [ -n "$SYSTEM_PROMPT" ]; then
    EXTRA_ARGS+=(--system_prompt "$SYSTEM_PROMPT")
fi
if [ "$USE_OFFLINE_EMBS" = "1" ]; then
    EXTRA_ARGS+=(--use_offline_embs)
fi
if [ "$USE_GENERATION_HISTORY" = "1" ]; then
    EXTRA_ARGS+=(--use_generation_history)
fi
if [ "$DISABLE_MODALITY_POSITION_IDS" = "1" ]; then
    EXTRA_ARGS+=(--disable_modality_position_ids)
fi
if [ "$PARALLEL_DECODE" = "1" ]; then
    EXTRA_ARGS+=(--parallel_chunk_decode)
fi
EXTRA_ARGS+=(--inference_audio_chunks_per_turn "$INFERENCE_AUDIO_CHUNKS_PER_TURN")
# --use_pass is silently ignored by single-pass models, so it's safe to
# always forward it.
EXTRA_ARGS+=(--use_pass "$USE_PASS")

echo ""
echo "==> Running ASR leaderboard evaluation"
echo "    Checkpoint:        $LOCAL_CKPT_PATH"
echo "    Device:            cuda:${DEVICE_ID}"
echo "    Batch size:        ${BATCH_SIZE}"
echo "    Chunks/turn:       ${INFERENCE_AUDIO_CHUNKS_PER_TURN}"
echo "    Max new tokens:    ${MAX_NEW_TOKENS} per chunk -> ${INFERENCE_AUDIO_CHUNKS_PER_TURN}x = $((MAX_NEW_TOKENS * INFERENCE_AUDIO_CHUNKS_PER_TURN)) per turn (model scales)"
if [ "$PARALLEL_DECODE" = "1" ]; then
    echo "    Decode mode:       PARALLEL (ParallelChunkHeads K-token-per-chunk)"
else
    echo "    Decode mode:       autoregressive"
fi
echo "    Results:           ${RESULTS_DIR}"
echo ""

# Wall-clock start so we can report a total run time alongside the per-dataset
# RTFx numbers (useful for AR vs parallel-decode comparison).
EVAL_T0=$(date +%s)

cd "$RESULTS_DIR"

# Terminal verbosity.  By default we keep the console quiet: the eval driver's
# verbose per-decode chatter (model load logs, tqdm bars, REF/HYP dumps) is
# redirected to $EVAL_LOG, and only a one-line status + WER per dataset is
# printed.  The full log still feeds the summary table below and stays
# available for debugging.  Set VERBOSE_EVAL=1 (or QUICK_TEST=1) to stream the
# raw driver output to the terminal instead.
VERBOSE_EVAL="${VERBOSE_EVAL:-0}"
if [ "$QUICK_TEST" = "1" ]; then
    VERBOSE_EVAL=1
fi

for ds_entry in "${DATASETS[@]}"; do
    IFS=':' read -r DATASET SPLIT <<< "$ds_entry"

    # Per-dataset revision pins.  TED-LIUM was deleted from the leaderboard
    # dataset on 2026-05-27; pin to TEDLIUM_REVISION so historical numbers
    # remain reproducible.  Set TEDLIUM_REVISION="" to opt out.
    DATASET_EXTRA_ARGS=()
    if [ "$DATASET" = "tedlium" ] && [ -n "$TEDLIUM_REVISION" ]; then
        DATASET_EXTRA_ARGS+=(--dataset_revision "$TEDLIUM_REVISION")
    fi

    echo "running inference on ${DATASET}/${SPLIT}..."

    # Section header in the log so each dataset is easy to locate.
    {
        echo "----------------------------------------------------------------------"
        echo "[$(date '+%H:%M:%S')] Evaluating: ${DATASET}/${SPLIT}"
        if [ ${#DATASET_EXTRA_ARGS[@]} -gt 0 ]; then
            echo "    (pinning hf-audio dataset to revision ${TEDLIUM_REVISION} for tedlium)"
        fi
        echo "----------------------------------------------------------------------"
    } >> "$EVAL_LOG"

    run_log=$(mktemp)
    # Disable errexit/pipefail aborts around the driver so we can capture its
    # exit status and surface a clean per-dataset message even on failure.
    set +e
    if [ "$VERBOSE_EVAL" = "1" ]; then
        python "$RUN_EVAL_PY" \
            --ckpt_path "$LOCAL_CKPT_PATH" \
            --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --device "$DEVICE_ID" \
            --batch_size "$BATCH_SIZE" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --no_repeat_ngram_size "$NO_REPEAT_NGRAM_SIZE" \
            "${EXTRA_ARGS[@]}" \
            "${DATASET_EXTRA_ARGS[@]}" \
            2>&1 | tee "$run_log"
        rc=${PIPESTATUS[0]}
    else
        python "$RUN_EVAL_PY" \
            --ckpt_path "$LOCAL_CKPT_PATH" \
            --dataset_path "hf-audio/esb-datasets-test-only-sorted" \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --device "$DEVICE_ID" \
            --batch_size "$BATCH_SIZE" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --no_repeat_ngram_size "$NO_REPEAT_NGRAM_SIZE" \
            "${EXTRA_ARGS[@]}" \
            "${DATASET_EXTRA_ARGS[@]}" \
            > "$run_log" 2>&1
        rc=$?
    fi
    set -e

    cat "$run_log" >> "$EVAL_LOG"

    if [ "$rc" -ne 0 ]; then
        echo "  ${DATASET} FAILED (exit ${rc}) -- see ${EVAL_LOG}"
    else
        wer=$(grep -oE 'WER:[[:space:]]*[0-9.]+' "$run_log" | tail -1 | grep -oE '[0-9.]+$')
        rtfx=$(grep -oE 'RTFX:[[:space:]]*[0-9.]+' "$run_log" | tail -1 | grep -oE '[0-9.]+$')
        if [ -n "$wer" ]; then
            echo "  ${DATASET} WER: ${wer} | RTFx: ${rtfx:-?}"
        else
            echo "  ${DATASET}: no WER parsed -- see ${EVAL_LOG}"
        fi
    fi
    rm -f "$run_log"
done

EVAL_T1=$(date +%s)
EVAL_ELAPSED=$((EVAL_T1 - EVAL_T0))
EVAL_ELAPSED_FMT=$(printf '%02dh%02dm%02ds' $((EVAL_ELAPSED/3600)) $(((EVAL_ELAPSED%3600)/60)) $((EVAL_ELAPSED%60)))

# ---------- Step 3: Summary table ----------
echo "======================================================================"
echo "Evaluation complete. Results in: ${RESULTS_DIR}"
echo "Log: ${EVAL_LOG}"
if [ "$PARALLEL_DECODE" = "1" ]; then
    echo "Decode mode: PARALLEL (ParallelChunkHeads K-token-per-chunk)"
else
    echo "Decode mode: autoregressive"
fi
echo "Total wall time (all datasets): ${EVAL_ELAPSED_FMT} (${EVAL_ELAPSED}s)"
echo "======================================================================"
echo ""

# Build the dataset list shown in the summary.  In ONLY mode we want to
# summarize *just* the requested datasets so the user sees a focused table
# (and isn't misled by stale "--" rows for datasets they didn't run).
SUMMARY_DATASETS=""
for ds_entry in "${DATASETS[@]}"; do
    IFS=':' read -r DATASET SPLIT <<< "$ds_entry"
    SUMMARY_DATASETS+="${DATASET},${SPLIT}|"
done

EVAL_LOG="$EVAL_LOG" SUMMARY_DATASETS="$SUMMARY_DATASETS" python3 -c "
import os
import re

raw = os.environ['SUMMARY_DATASETS']
datasets = [tuple(s.split(',')) for s in raw.split('|') if s]

log = open(os.environ['EVAL_LOG']).read()

# Parse 'Dataset: X/Y' followed by 'RTFX: A' and 'WER: B %' triples.
# RTFX is printed before WER in run_eval_sslm.py, so we capture both per
# dataset block. A missing field falls back to None.
entries = re.findall(
    r'Dataset:\s*(\S+/\S+)[\s\S]*?(?:RTFX:\s*([\d.]+)[\s\S]*?)?WER:\s*([\d.]+)\s*%',
    log,
)
wer_map = {}
rtfx_map = {}
for ds, rtfx, wer in entries:
    wer_map[ds] = float(wer)
    if rtfx:
        rtfx_map[ds] = float(rtfx)

print(f'  {\"Dataset\":<25} {\"WER (%)\":>8} {\"RTFx\":>8}')
print(f'  {\"-\" * 25} {\"-\" * 8} {\"-\" * 8}')
total_wer = 0.0
total_rtfx = 0.0
n_wer = 0
n_rtfx = 0
for ds, split in datasets:
    key = f'{ds}/{split}'
    wer_cell = f'{wer_map[key]:>8.2f}' if key in wer_map else f'{\"--\":>8}'
    rtfx_cell = f'{rtfx_map[key]:>8.2f}' if key in rtfx_map else f'{\"--\":>8}'
    print(f'  {key:<25} {wer_cell} {rtfx_cell}')
    if key in wer_map:
        total_wer += wer_map[key]
        n_wer += 1
    if key in rtfx_map:
        total_rtfx += rtfx_map[key]
        n_rtfx += 1

if n_wer > 0:
    print(f'  {\"-\" * 25} {\"-\" * 8} {\"-\" * 8}')
    avg_wer_cell = f'{total_wer / n_wer:>8.2f}'
    avg_rtfx_cell = f'{total_rtfx / n_rtfx:>8.2f}' if n_rtfx > 0 else f'{\"--\":>8}'
    print(f'  {\"Average\":<25} {avg_wer_cell} {avg_rtfx_cell}')
print()
"
