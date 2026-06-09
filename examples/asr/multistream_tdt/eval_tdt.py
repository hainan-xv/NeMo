# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Unified WER evaluation for (TDT) transducer models.

Supports BOTH:
  * the regular TDT model (e.g. a fine-tuned ``nvidia/parakeet-tdt-0.6b-v2``,
    ``EncDecRNNTBPEModel``), and
  * the 2-stream factorized spelling+capitalization model
    (``EncDecMultiStreamTDTBPEModel``).

The right model class is selected automatically from the checkpoint's stored
``target`` field, and each model's own (validated) decoding path is reused via
``Trainer.test`` -> ``multi_test_epoch_end`` -> ``test_wer``. The factorized
model decodes with its non-batched greedy multistream decoder; the regular TDT
model uses its configured (batched) RNNT/TDT decoding.

Examples
--------
  # one .nemo, two test manifests
  python examples/asr/multistream_tdt/eval_tdt.py \
      --model /results/.../checkpoints/model.nemo \
      --test_manifest /data/.../mcv11_test.json,/data/.../ami_test.json \
      --batch_size 16

  # a mid-training .ckpt for the factorized model (tokenizer dir persists on lustre)
  python examples/asr/multistream_tdt/eval_tdt.py \
      --model /results/EXP/checkpoints/EXP--val_wer=0.1-step=20000.ckpt \
      --tokenizer_dir /results/EXP/tokenizer \
      --test_manifest /data/.../mcv11_test.json
"""

import argparse
import json
import os
from typing import List, Optional

import torch
from omegaconf import OmegaConf, open_dict

import lightning.pytorch as pl

from nemo.collections.asr.models import ASRModel
from nemo.utils import logging
from nemo.utils.model_utils import import_class_by_path


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate WER for TDT / multistream-TDT models.")
    p.add_argument("--model", required=True, help="Path to a .nemo or .ckpt checkpoint.")
    p.add_argument(
        "--test_manifest",
        required=True,
        help="Comma-separated list of test manifest json paths (each evaluated separately).",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--use_cer", action="store_true", help="Report CER instead of WER.")
    p.add_argument("--cuda", type=int, default=0, help="GPU index (use -1 for CPU).")
    p.add_argument("--precision", default="bf16", help="Trainer precision (e.g. 32, bf16, 16).")
    p.add_argument(
        "--tokenizer_dir",
        default=None,
        help="Override tokenizer dir when loading a .ckpt (needed if the trained-time path is gone).",
    )
    p.add_argument(
        "--max_symbols_per_step",
        type=int,
        default=None,
        help="Override greedy symbols-per-step cap (multistream model only).",
    )
    p.add_argument("--output", default=None, help="Optional path to write a JSON results summary.")
    return p.parse_args()


def _load_model(model_path: str, map_location, tokenizer_dir: Optional[str]):
    """Load a .nemo or .ckpt into the correct model subclass (auto-detected)."""
    if model_path.endswith(".nemo"):
        cfg = ASRModel.restore_from(restore_path=model_path, return_config=True)
        target = cfg.get("target", "nemo.collections.asr.models.ASRModel")
        cls = import_class_by_path(target)
        logging.info(f"Restoring {cls.__name__} from {model_path}")
        model = cls.restore_from(restore_path=model_path, map_location=map_location)
        return model

    if model_path.endswith(".ckpt"):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if "hyper_parameters" not in ckpt or "cfg" not in ckpt["hyper_parameters"]:
            raise ValueError(
                f"{model_path} does not embed a model config under hyper_parameters['cfg']; "
                "please evaluate a .nemo export instead."
            )
        cfg = OmegaConf.create(ckpt["hyper_parameters"]["cfg"])
        if tokenizer_dir is not None:
            with open_dict(cfg):
                if "tokenizer" in cfg:
                    cfg.tokenizer.dir = tokenizer_dir
                    cfg.tokenizer.update_tokenizer = False
        target = cfg.get("target", None)
        if target is None:
            raise ValueError("Checkpoint config has no `target`; cannot determine the model class.")
        cls = import_class_by_path(target)
        logging.info(f"Instantiating {cls.__name__} from .ckpt config and loading weights")
        model = cls(cfg=cfg)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            logging.warning(f"Missing keys when loading state_dict: {missing}")
        if unexpected:
            logging.warning(f"Unexpected keys when loading state_dict: {unexpected}")
        return model

    raise ValueError(f"Unsupported checkpoint extension for {model_path} (expected .nemo or .ckpt).")


def main():
    args = parse_args()

    use_gpu = args.cuda is not None and args.cuda >= 0 and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.set_device(args.cuda)
        map_location = torch.device(f"cuda:{args.cuda}")
        accelerator, devices = "gpu", [args.cuda]
    else:
        map_location = torch.device("cpu")
        accelerator, devices = "cpu", 1

    model = _load_model(args.model, map_location, args.tokenizer_dir)
    model = model.to(map_location)
    model.eval()

    # Optional override for the multistream greedy decoder.
    if args.max_symbols_per_step is not None and hasattr(model, "ms_greedy"):
        model.ms_greedy.max_symbols = args.max_symbols_per_step

    # Detect by capability so any multistream variant (2-stream cap, 3-stream cap+punct, ...) works.
    is_multistream = hasattr(model, "_decode_hyp_texts")
    metric_name = "CER" if args.use_cer else "WER"
    if hasattr(model, "use_cer"):
        model.use_cer = args.use_cer
    elif hasattr(model, "wer"):
        # Regular RNNT/TDT model: toggle its WER metric to CER if requested.
        try:
            model.wer.use_cer = args.use_cer
        except Exception:
            pass

    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        precision=args.precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    manifests: List[str] = [m.strip() for m in args.test_manifest.split(",") if m.strip()]
    results = {}
    for manifest in manifests:
        if not os.path.isfile(manifest):
            logging.warning(f"Manifest not found, skipping: {manifest}")
            continue

        test_ds = OmegaConf.create(
            {
                "manifest_filepath": manifest,
                "sample_rate": args.sample_rate,
                "batch_size": args.batch_size,
                "shuffle": False,
                "num_workers": args.num_workers,
                "pin_memory": True,
                "use_lhotse": False,
            }
        )
        model.setup_test_data(test_ds)

        logging.info(f"==> Evaluating on {manifest} (model={'multistream' if is_multistream else 'tdt'})")
        test_out = trainer.test(model, verbose=False)

        wer = None
        if test_out and "test_wer" in test_out[0]:
            wer = float(test_out[0]["test_wer"])
        elif "test_wer" in trainer.callback_metrics:
            wer = float(trainer.callback_metrics["test_wer"])
        results[manifest] = wer
        logging.info(f"==> {metric_name} on {os.path.basename(manifest)} = {wer * 100:.2f}%" if wer is not None
                     else f"==> {metric_name} on {os.path.basename(manifest)} = N/A")

    print("\n================ Evaluation summary ================")
    print(f"model     : {args.model}")
    print(f"type      : {'multistream-TDT (spelling+cap)' if is_multistream else 'regular TDT'}")
    print(f"metric    : {metric_name}")
    valid = [w for w in results.values() if w is not None]
    for manifest, wer in results.items():
        line = f"  {os.path.basename(manifest):50s} {wer * 100:6.2f}%" if wer is not None else f"  {os.path.basename(manifest):50s}    N/A"
        print(line)
    if valid:
        print("  " + "-" * 58)
        print(f"  {'AVERAGE':50s} {sum(valid) / len(valid) * 100:6.2f}%")
    print("====================================================\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "type": "multistream" if is_multistream else "tdt",
                    "metric": metric_name,
                    "results": {m: (None if w is None else w) for m, w in results.items()},
                    "average": (sum(valid) / len(valid)) if valid else None,
                },
                f,
                indent=2,
            )
        logging.info(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
