#!/usr/bin/env bash
# Patched clone of the image's /workspace/convert.sh.
#
# Difference: it rewrites BOTH `pretrained_asr` AND `pretrained_llm` in the
# extracted checkpoint config to HF repo ids. The stock convert.sh only rewrites
# pretrained_asr, so checkpoints trained on the cluster (whose config stores an
# absolute /lustre/... path for the LLM, e.g. Qwen/Qwen3-1.7B) fail to convert
# offline with an HFValidationError. This maps any absolute LLM path to its HF
# repo id (last two path components), leaving already-repo-id values untouched.
#
# Everything else (extract_ckpt_cfg -> to_hf -> setup_chunk_generic) is identical
# to the baked convert.sh, so the produced vLLM SALM dir is the same.
#
# Usage (inside the container):
#   convert_patched.sh <ckpt_or_hf_dir> <out_vllm_salm_dir> [pretrained_asr_repo]
set -euo pipefail

SRC="${1:?Usage: convert_patched.sh <ckpt_or_hf_dir> <out_vllm_salm_dir> [asr_repo]}"
OUT="${2:?Usage: convert_patched.sh <ckpt_or_hf_dir> <out_vllm_salm_dir> [asr_repo]}"
ASR_REPO="${3:-nvidia/nemotron-speech-streaming-en-0.6b}"

export PYTHONPATH=/opt/heh-nemo
CONVERT=/opt/convert
HEH_NEMO=/opt/heh-nemo
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$OUT"

if [[ -f "$SRC" && "$SRC" == *.ckpt ]]; then
    echo "[convert] (1/3) raw .ckpt -> exp_config.yaml"
    CK="$SRC" OUT="$WORK/exp_config.yaml" python "$CONVERT/extract_ckpt_cfg.py"
    # Rewrite pretrained_asr + pretrained_llm to HF repo ids resolvable inside
    # this image (the training-cluster lustre paths don't exist here).
    python - "$WORK/exp_config.yaml" "$ASR_REPO" <<'PY'
import os, sys
from omegaconf import OmegaConf

yaml_path, asr_repo = sys.argv[1], sys.argv[2]
cfg = OmegaConf.load(yaml_path)


def to_repo_id(p):
    """Map an absolute local model path to its HF repo id (last two path
    components), stripping a trailing *.nemo file. Repo-id-looking values
    (relative, no leading slash) are returned unchanged."""
    p = str(p)
    if not os.path.isabs(p):
        return p
    if p.endswith(".nemo"):
        p = os.path.dirname(p)
    parts = [x for x in p.split("/") if x]
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


cfg.model.pretrained_asr = asr_repo
print(f"[convert]     pretrained_asr -> {asr_repo}")

llm = cfg.model.get("pretrained_llm", None)
if llm is not None:
    new_llm = to_repo_id(llm)
    if new_llm != str(llm):
        cfg.model.pretrained_llm = new_llm
        print(f"[convert]     pretrained_llm -> {new_llm}   (was {llm})")
    else:
        print(f"[convert]     pretrained_llm kept as-is: {llm}")

OmegaConf.save(cfg, yaml_path)
PY
    echo "[convert] (2/3) .ckpt -> HF export"
    HF_DIR="$WORK/hf_export"
    # weights_only=false: our Lightning .ckpt stores OmegaConf DictConfig objects
    # in its payload, which torch>=2.6's default weights_only=True refuses to
    # unpickle. These checkpoints are self-produced (trusted), so disable it.
    python "$HEH_NEMO/examples/speechlm2/to_hf.py" \
        class_path=nemo.collections.speechlm2.models.StreamingSTTModel \
        ckpt_path="$SRC" ckpt_config="$WORK/exp_config.yaml" output_dir="$HF_DIR" \
        weights_only=false
else
    echo "[convert] (1-2/3) using existing HF export: $SRC"
    HF_DIR="$SRC"
fi

echo "[convert] (3/3) HF export -> vLLM SALM dir (+ streaming markers)"
SETUP_SRC="$HF_DIR" SETUP_DST="$OUT" SETUP_MARKERS="$OUT/markers.json" \
    python "$CONVERT/setup_chunk_generic.py"

# Materialize the weights so the output dir is self-contained.
if [[ -L "$OUT/model.safetensors" ]]; then
    tgt="$(readlink -f "$OUT/model.safetensors")"
    cp --remove-destination "$tgt" "$OUT/model.safetensors"
fi

echo "[convert] DONE -> $OUT"
