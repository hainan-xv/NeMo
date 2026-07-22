#!/usr/bin/env python3
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
"""
Average several StreamingSTTModel Lightning checkpoints into one.

StreamingSTT training writes plain Lightning .ckpt files (dict with
"state_dict" + "hyper_parameters"); there is no .nemo export, so NeMo's
checkpoint_averaging.py (which needs a .nemo) does not apply. This averages the
floating-point tensors of "state_dict" across the inputs and keeps every other
field (non-float buffers, "hyper_parameters") from the first input, then writes
a minimal .ckpt that the eval drivers can load directly.

Usage:
    python average_sslm_ckpts.py --output avg.ckpt in1.ckpt in2.ckpt [...]
"""
import argparse
import os

import torch


def average_checkpoints(input_paths, output_path):
    if not input_paths:
        raise ValueError("no input checkpoints given")

    acc = None          # float32 running sum of float tensors (CPU)
    dtypes = {}         # original dtype per averaged key
    hyper_parameters = None
    n = len(input_paths)

    for i, path in enumerate(input_paths):
        print(f"  [{i + 1}/{n}] loading {path}", flush=True)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        if acc is None:
            hyper_parameters = ckpt.get("hyper_parameters")
            acc = {}
            for k, v in sd.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    acc[k] = v.detach().to(torch.float32).clone()
                    dtypes[k] = v.dtype
                else:
                    # Non-float (int buffers, position ids, ...): keep as-is.
                    acc[k] = v
        else:
            for k in dtypes:
                if k in sd:
                    acc[k] += sd[k].detach().to(torch.float32)
        del ckpt, sd

    for k in dtypes:
        acc[k] = (acc[k] / n).to(dtypes[k])

    out = {"state_dict": acc}
    if hyper_parameters is not None:
        # Store hyper_parameters as plain containers so the checkpoint loads under
        # torch.load(weights_only=True) (PyTorch 2.6+ default) — e.g. the vLLM
        # container's convert.sh/to_hf. An OmegaConf DictConfig/ListConfig here
        # would otherwise be rejected as an unsupported global.
        try:
            from omegaconf import DictConfig, ListConfig, OmegaConf

            _OC = True
        except ImportError:
            _OC = False

        def _plain(x):
            if _OC and isinstance(x, (DictConfig, ListConfig)):
                return _plain(OmegaConf.to_container(x, resolve=True))
            if isinstance(x, dict):
                return {k: _plain(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_plain(v) for v in x]
            if isinstance(x, (str, int, float, bool)) or x is None:
                return x
            return None  # drop non-plain objects (class refs, dataset instances, ...)

        hyper_parameters = _plain(hyper_parameters)
        out["hyper_parameters"] = hyper_parameters

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(out, output_path)
    print(f"==> wrote averaged checkpoint ({len(dtypes)} float tensors over {n} inputs): {output_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Average StreamingSTTModel Lightning checkpoints.")
    p.add_argument("--output", required=True, help="Output averaged .ckpt path")
    p.add_argument("inputs", nargs="+", help="Input .ckpt files to average")
    args = p.parse_args()
    average_checkpoints(args.inputs, args.output)
