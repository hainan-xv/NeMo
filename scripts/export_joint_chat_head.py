# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Export a joint model's CHAT head as a standalone ``EncDecCHATBPEModel`` .nemo.

The joint model's CHAT side is exactly a CHAT transducer mounted on the shared
encoder, so lifting (encoder, decoder, joint) back out yields a plain CHAT model.
Doing that lets the EXISTING, tested CHAT leaderboard harness evaluate it, and
makes the numbers directly comparable to the standalone CHAT baselines instead of
to a bespoke driver written for this one experiment.

The donor .nemo supplies the architecture/config; only the WEIGHTS come from the
joint checkpoint. Feature normalisation and xscaling therefore come from the
donor's own config, which is the pairing the weights were trained under.
"""
import argparse
import io
import os
import shutil
import tarfile
import tempfile

import torch
from omegaconf import OmegaConf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint_ckpt", required=True, help="Lightning checkpoint of the joint model")
    ap.add_argument("--donor_nemo", required=True, help="CHAT .nemo supplying config + tokenizer artifacts")
    ap.add_argument("--out", required=True, help="output .nemo path")
    ap.add_argument("--tokenizer_dir", default=None, help="override tokenizer dir in the exported config")
    args = ap.parse_args()

    ckpt = torch.load(args.joint_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    # joint model -> plain CHAT names
    groups = {"perception.encoder.": "encoder.", "chat.decoder.": "decoder.", "chat.joint.": "joint."}
    out_sd, counts = {}, {g: 0 for g in groups}
    for k, v in sd.items():
        for src, dst in groups.items():
            if k.startswith(src):
                out_sd[dst + k[len(src) :]] = v
                counts[src] += 1
                break
    empty = [g for g, c in counts.items() if c == 0]
    if empty:
        raise SystemExit(
            f"no tensors found for {empty} in {args.joint_ckpt}; "
            f"top-level prefixes present: {sorted({k.split('.')[0] for k in sd})}"
        )
    print(f"lifted: encoder {counts['perception.encoder.']}, "
          f"decoder {counts['chat.decoder.']}, joint {counts['chat.joint.']}")

    tmp = tempfile.mkdtemp()
    try:
        with tarfile.open(args.donor_nemo) as tf:
            tf.extractall(tmp)
        cfg_path = None
        for root, _, files in os.walk(tmp):
            for f in files:
                if f.endswith("model_config.yaml"):
                    cfg_path = os.path.join(root, f)
        if cfg_path is None:
            raise SystemExit(f"{args.donor_nemo} has no model_config.yaml")
        cfg = OmegaConf.load(cfg_path)
        if args.tokenizer_dir:
            cfg.tokenizer.dir = args.tokenizer_dir
        for k in ("train_ds", "validation_ds", "test_ds"):
            if k in cfg:
                cfg.pop(k)
        OmegaConf.save(cfg, cfg_path)
        print(f"config: xscaling={cfg.encoder.get('xscaling')} "
              f"normalize={cfg.preprocessor.get('normalize')} tokenizer={cfg.tokenizer.dir}")

        wpath = None
        for root, _, files in os.walk(tmp):
            for f in files:
                if f.endswith("model_weights.ckpt"):
                    wpath = os.path.join(root, f)
        donor_sd = torch.load(wpath, map_location="cpu", weights_only=False)
        donor_sd = donor_sd.get("state_dict", donor_sd)
        # keep the donor's preprocessor buffers; replace everything we lifted
        merged = dict(donor_sd)
        missing = [k for k in out_sd if k not in donor_sd]
        if missing:
            raise SystemExit(f"{len(missing)} lifted tensors absent from the donor, e.g. {missing[:4]}")
        bad = [k for k in out_sd if donor_sd[k].shape != out_sd[k].shape]
        if bad:
            raise SystemExit(f"shape mismatch on {len(bad)} tensors, e.g. {bad[:4]}")
        merged.update(out_sd)
        torch.save(merged, wpath)

        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with tarfile.open(args.out, "w") as tf:
            for name in sorted(os.listdir(tmp)):
                tf.add(os.path.join(tmp, name), arcname="./" + name)
        print(f"wrote {args.out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
