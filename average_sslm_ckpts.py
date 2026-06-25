#!/usr/bin/env python3
"""
Average several StreamingSTTModel Lightning checkpoints into one.

StreamingSTT training writes plain Lightning .ckpt files (dict with
"state_dict" + "hyper_parameters"); there is no .nemo export, so NeMo's
checkpoint_averaging.py (which needs a .nemo) does not apply. This averages the
floating-point tensors of "state_dict" across the inputs and keeps every other
field (non-float buffers, "hyper_parameters") from the first input, then writes
a minimal .ckpt that run_eval_sslm.py can load directly.

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
