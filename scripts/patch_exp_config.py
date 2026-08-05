# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Patch a training exp_config.yaml so ``to_hf.py`` can re-instantiate the model
for the heh leaderboard eval.

Usage:  python patch_exp_config.py <src_cfg> <dst_cfg> <ckpt_path>

Edits (mirrors scripts/speechlm_leaderboard_eval.load_model):
  * pretrained_llm / pretrained_asr: if the config value is NOT a valid local
    path (e.g. a bare Hub id ``Qwen/Qwen3-1.7B`` from an external/colleague model),
    and an override is supplied via env (OVERRIDE_PRETRAINED_LLM /
    OVERRIDE_PRETRAINED_ASR) that DOES exist locally, swap it in so the offline
    (HF_HUB_OFFLINE=1) conversion can load it from the mounted lustre mirror. A
    value that already exists locally is left untouched.
  * load_llm_weights=False (LLM weights come from the checkpoint).
  * pretrained_llm / pretrained_asr: kept as-is when the path EXISTS locally
    (OCI lustre paths are mounted -> stay offline); only mapped to a Hub id
    ``org/name`` when the absolute path is missing (off-cluster fallback).
  * lora.target_modules: filled from the checkpoint's LoRA keys when absent
    (newer PEFT requires it for a strict load).
"""
import os
import sys

from omegaconf import OmegaConf


def _is_local(path) -> bool:
    return isinstance(path, str) and path.startswith("/") and os.path.exists(path)


def _hubify(path: str) -> str:
    if not isinstance(path, str) or not path.startswith("/") or os.path.exists(path):
        return path
    parts = [p for p in path.rstrip("/").split("/") if p]
    if "huggingface" in parts:
        rem = [p for p in parts[parts.index("huggingface") + 1 :] if not p.endswith((".nemo", ".ckpt", ".bin"))]
        if len(rem) >= 2:
            return f"{rem[0]}/{rem[1]}"
        if rem:
            return rem[0]
    nonfile = [p for p in parts if not p.endswith((".nemo", ".ckpt", ".bin"))]
    if len(nonfile) >= 2:
        return f"{nonfile[-2]}/{nonfile[-1]}"
    return path


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: patch_exp_config.py <src_cfg> <dst_cfg> <ckpt_path>", file=sys.stderr)
        return 2
    src, dst, ckpt_path = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = OmegaConf.load(src)
    m = cfg.model

    # Local-mirror overrides for external checkpoints whose config references a
    # bare Hub id (unreachable under HF_HUB_OFFLINE=1). Only applied when the
    # existing value is NOT already a valid local path AND the override exists.
    overrides = {
        "pretrained_llm": os.environ.get("OVERRIDE_PRETRAINED_LLM", "").strip(),
        "pretrained_asr": os.environ.get("OVERRIDE_PRETRAINED_ASR", "").strip(),
    }
    for key, ov in overrides.items():
        if not ov or key not in m:
            continue
        cur = m[key]
        if _is_local(cur):
            continue  # config already points at a valid local mirror; keep it
        if not _is_local(ov):
            print(f"  {key}: override '{ov}' not found locally; leaving '{cur}' as-is")
            continue
        print(f"  {key}: {cur} -> {ov} (local-mirror override)")
        m[key] = ov

    for key in ("pretrained_llm", "pretrained_asr"):
        if key in m and m[key]:
            old = m[key]
            new = _hubify(old)
            if new != old:
                print(f"  {key}: {old} -> {new} (not found locally; using Hub id)")
                m[key] = new
    if "load_llm_weights" in m:
        m["load_llm_weights"] = False

    lora = m.get("lora") if hasattr(m, "get") else (m.lora if "lora" in m else None)
    if lora is not None and ("target_modules" not in lora or not lora.get("target_modules")):
        import torch

        try:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        mods = set()
        for k in sd:
            if ".lora_A." in k:
                parts = k.split(".")
                i = parts.index("lora_A")
                if i > 0:
                    mods.add(parts[i - 1])
        del ck, sd
        lora["target_modules"] = sorted(mods) if mods else "all-linear"
        print(f"  lora.target_modules: {lora['target_modules']}")

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    OmegaConf.save(cfg, dst)
    print(f"  wrote patched config: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
