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
