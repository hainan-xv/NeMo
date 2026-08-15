#!/bin/bash
# DESKTOP (no SLURM / no container) NeMo-native leaderboard eval.
#
# Loops NeMo's own examples/asr/speech_to_text_eval.py over the staged cache (one
# dataset at a time, single GPU, no sharding), then rescores the predictions with
# the leaderboard-faithful WER (scripts/rescore_nemo_eval.py). This is the desktop
# twin of launch/eval_parakeet_nemo.sh -- same decode + scoring, run via `uv run`
# instead of inside the grid container.
#
# PREREQ (once): the uv env must be on Python >= 3.12, else NeMo's restore_from hits
#   TarFile.extract(filter=...) -> TypeError:
#     uv python install 3.12 && uv sync -p 3.12
#
# The cache must already be staged locally (e.g. by scripts/parakeet_eval_local.py,
# which auto-downloads). speech_to_text_eval.py does NOT download; it reads the
# staged <cache_dir>/<name>/<split>/_cache_manifest.jsonl.
#
# Usage (from the clean repo root):
#   bash scripts/run_parakeet_nemo_local.sh [NEMO_MODEL] [CACHE_DIR]
#   MAX_SAMPLES=50 bash scripts/run_parakeet_nemo_local.sh          # quick check
#   DATASETS="librispeech:test.clean" bash scripts/run_parakeet_nemo_local.sh
#   bash scripts/run_parakeet_nemo_local.sh --gpu 1                 # pin to cuda:1
#
# GPU: pass `--gpu N` (or `--gpu=N`, or `GPU=N` env) to pin this run to a single
# CUDA device. Pick different N (and different OUT_DIR) for two runs so they share
# the box without fighting over one GPU.
#
# PYTHON: defaults to plain `python` (i.e. run inside your activated env -- a conda
# env with Python 3.12 + `pip install -e ".[asr]"`, or a `source .venv/bin/activate`
# uv env). To use uv WITHOUT activating, set PYRUN="uv run python".
set -euo pipefail

PYRUN="${PYRUN:-python}"

# GPU selection: `--gpu N` / `--gpu=N` (or GPU=N env) pins this run to one CUDA
# device (passed to speech_to_text_eval.py as cuda=N). Extracted before the
# positional $1/$2 (NEMO_MODEL / CACHE_DIR) so those still work.
GPU="${GPU:-0}"
_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --gpu=*) GPU="${1#*=}"; shift ;;
    *) _args+=("$1"); shift ;;
  esac
done
set -- ${_args[@]+"${_args[@]}"}

# Pin imports to THIS repo. The Cursor terminal injects the workspace root
# (a different NeMo checkout) onto PYTHONPATH, which shadows this repo's editable
# `nemo` and drags in a heavier import chain (e.g. pyannote). Prepend our own root so
# `import nemo` always resolves here.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Run NeMo's stock eval with the log spam silenced in-process (scripts/quiet_logs.py)
# while leaving the live tqdm progress bar intact -- a shell pipe can't do this since
# the bar and NeMo's warnings share stderr. Hydra still reads the key=value overrides
# from sys.argv as usual; we add examples/asr to sys.path for `import transcribe_speech`.
_quiet_eval() {
  $PYRUN - "$@" <<'PY'
import os, runpy, sys

repo = os.environ["REPO_ROOT"]
sys.path.insert(0, os.path.join(repo, "scripts"))
sys.path.insert(0, os.path.join(repo, "examples", "asr"))
from quiet_logs import add_transcribe_total, silence


def _argv_val(key):
    for a in sys.argv[1:]:
        if a.startswith(key + "="):
            return a.split("=", 1)[1]
    return None


silence()
add_transcribe_total(_argv_val("dataset_manifest"), int(_argv_val("batch_size") or 1))
script = os.path.join(repo, "examples", "asr", "speech_to_text_eval.py")
sys.argv = [script, *sys.argv[1:]]
runpy.run_path(script, run_name="__main__")
PY
}

# cuDNN precedence: pip torch bundles its own cuDNN (e.g. 9.20), but a system cuDNN
# on LD_LIBRARY_PATH (/usr/lib/x86_64-linux-gnu on this box ships 9.2) shadows it and
# triggers "cuDNN version incompatibility". Prepend torch's bundled nvidia/*/lib dirs
# (resolved from the active interpreter) so the correct libs load first.
# NB: `nvidia` is a PEP-420 namespace package, so nvidia.__file__ is None -- resolve
# the on-disk dir via nvidia.__path__ instead.
NV_ROOT="$($PYRUN -c 'import nvidia; print(list(nvidia.__path__)[0])' 2>/dev/null || true)"
if [ -n "${NV_ROOT}" ]; then
  for d in "${NV_ROOT}"/*/lib; do
    [ -d "$d" ] && LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
  done
  export LD_LIBRARY_PATH
fi

NEMO_MODEL="${1:-${NEMO_MODEL:-/home/hainanx/pretrained/parakeet-tdt-0.6b-v2.nemo}}"
CACHE_DIR="${2:-${CACHE_DIR:-$HOME/leaderboard_cache_smoke}}"
DATASETS="${DATASETS:-librispeech:test.clean librispeech:test.other ami_cleaned:test earnings22:test gigaspeech_cleaned:test spgispeech:test voxpopuli_cleaned_aa:test}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OUT_DIR="${OUT_DIR:-./parakeet_nemo_local_$(date +%Y%m%d_%H%M%S)}"
PRED_DIR="${OUT_DIR}/preds"
TMP_DIR="${OUT_DIR}/tmp"
mkdir -p "$PRED_DIR" "$TMP_DIR"

[[ -f "$NEMO_MODEL" ]] || { echo "ERROR: model not found: $NEMO_MODEL" >&2; exit 1; }
[[ -d "$CACHE_DIR"  ]] || { echo "ERROR: cache_dir not found: $CACHE_DIR (stage it first with parakeet_eval_local.py)." >&2; exit 1; }

# Offline: read only the local cache, no hub calls.
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "==> NeMo-native local eval (single GPU, no sharding)"
echo "    model:     $NEMO_MODEL"
echo "    cache_dir: $CACHE_DIR"
echo "    datasets:  $DATASETS"
echo "    gpu:       cuda:$GPU"
echo "    out_dir:   $OUT_DIR"

for entry in $DATASETS; do
    name="${entry%%:*}"; split="${entry#*:}"; [[ "$split" == "$entry" ]] && split=test
    man="${CACHE_DIR}/${name}/${split}/_cache_manifest.jsonl"
    out="${PRED_DIR}/${name}__${split}.json"
    if [[ ! -f "$man" ]]; then echo "  [skip] missing manifest: $man"; continue; fi
    use_man="$man"
    if [[ "$MAX_SAMPLES" -gt 0 ]]; then
        use_man="${TMP_DIR}/${name}__${split}.head.jsonl"
        head -n "$MAX_SAMPLES" "$man" > "$use_man"
    fi
    echo "==> [${name}/${split}] decoding (cuda:${GPU}) ..."
    # Quiet decode (NeMo logs -> ERROR, tqdm bar kept). On success, immediately
    # rescore just this dataset so the leaderboard WER prints as it finishes.
    if _quiet_eval \
        model_path="$NEMO_MODEL" \
        dataset_manifest="$use_man" \
        output_filename="$out" \
        gt_text_attr_name=reference \
        cuda="$GPU" \
        batch_size="$BATCH_SIZE" \
        amp=True \
        use_cer=False \
        text_processing.do_lowercase=true \
        text_processing.rm_punctuation=true ; then
        $PYRUN scripts/rescore_nemo_eval.py --pred_file "$out" --no_summary
    else
        echo "  WARN: speech_to_text_eval failed for ${name}/${split}"
    fi
done

echo ""
echo "==================== Leaderboard WER (rescored) ===================="
$PYRUN scripts/rescore_nemo_eval.py --pred_dir "$PRED_DIR" | tee "${OUT_DIR}/leaderboard_rescore.log"
echo ""
echo "Predictions + logs under: $OUT_DIR"
