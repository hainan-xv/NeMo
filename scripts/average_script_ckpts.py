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
"""Average several Lightning checkpoints into one.

Averaging the top-k checkpoints of a run is usually worth a few tenths of WER over
picking the single best one, and costs nothing at inference.

    python scripts/average_script_ckpts.py --output avg.ckpt a.ckpt b.ckpt c.ckpt

Only floating-point tensors are averaged. Integer buffers (position ids, step
counters) are taken verbatim from the first checkpoint, since averaging them is
meaningless and would corrupt them.
"""

import argparse
import os
from typing import List

import torch

# Lightning's checkpoint migration refuses to load a file without these.
_META_KEYS = ("pytorch-lightning_version", "epoch", "global_step")


def _plain(obj):
    """Convert OmegaConf containers to plain Python so the result is loadable
    under ``torch.load(weights_only=True)``. Anything exotic becomes ``None``."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, (DictConfig, ListConfig)):
            obj = OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return None


def average_checkpoints(input_paths: List[str], output_path: str) -> None:
    if not input_paths:
        raise ValueError("no input checkpoints given")

    acc = None
    dtypes = {}
    first = None
    n = len(input_paths)

    for idx, path in enumerate(input_paths):
        print(f"==> [{idx + 1}/{n}] {path}", flush=True)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        if acc is None:
            first = ckpt
            acc = {}
            for k, v in sd.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    dtypes[k] = v.dtype
                    acc[k] = v.to(torch.float32).clone()
                else:
                    acc[k] = v  # non-float: keep the first checkpoint's value
        else:
            for k in acc:
                if k in dtypes and k in sd:
                    acc[k] += sd[k].to(torch.float32)
        del ckpt, sd

    for k, dt in dtypes.items():
        acc[k] = (acc[k] / n).to(dt)

    out = {"state_dict": acc}
    for key in _META_KEYS:
        if key in first:
            out[key] = first[key]
    if "hyper_parameters" in first:
        out["hyper_parameters"] = _plain(first["hyper_parameters"])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(out, output_path)
    print(f"==> wrote averaged checkpoint ({len(dtypes)} float tensors over {n} inputs): {output_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=str, required=True, help="output .ckpt path")
    ap.add_argument("inputs", nargs="+", help="input .ckpt files to average")
    args = ap.parse_args()
    average_checkpoints(args.inputs, args.output)
