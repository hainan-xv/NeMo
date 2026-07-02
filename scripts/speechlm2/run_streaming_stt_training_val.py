#!/usr/bin/env python3
"""Run StreamingSTT validation WER through the training validation code path.

This script is intentionally different from ``run_eval_sslm.py``. It constructs
the model/datamodule like ``examples/speechlm2/streaming_stt_train.py`` and lets
Lightning call:

  StreamingSTTModel.validation_step()
  StreamingSTTModel.on_validation_epoch_end()

so the reported ``val_wer`` is computed by the exact path used during training.

Examples:

  # Use the resolved training config saved by exp_manager.
  python scripts/speechlm2/run_streaming_stt_training_val.py \
    --ckpt checkpoints/imend_flush_notrunc/step=6001.ckpt \
    --exp-config debug_decode_configs/imend_flush_notrunc/exp_config.yaml \
    --device 1 --limit-val-batches 10

  # Use a local validation manifest directly.
  python scripts/speechlm2/run_streaming_stt_training_val.py \
    --ckpt checkpoints/imend_flush_notrunc/step=6001.ckpt \
    --validation-manifest debug_decode_configs/imend_flush_notrunc/mcv11_dev_local_32.json \
    --device 1 --limit-val-batches 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nemo.collections.speechlm2 import DataModule, StreamingSTTDataset, StreamingSTTModel
from nemo.core.classes.common import Serialization
from nemo.utils.trainer_utils import resolve_trainer_cfg

import run_eval_sslm


def _safe_name(dataset: str, split: str) -> str:
    return f"{dataset}_{split}".replace("/", "_").replace(".", "_").replace("-", "_")


def materialize_leaderboard_manifests(args) -> dict[str, dict[str, str]]:
    """Create local NeMo manifests for the open-ASR leaderboard splits."""
    out_dir = Path(args.leaderboard_manifest_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_items: dict[str, dict[str, str]] = {}
    prep_args = argparse.Namespace(
        dataset_path=args.dataset_path,
        streaming=args.streaming,
        max_eval_samples=args.max_eval_samples,
    )
    for dataset, split in run_eval_sslm.LEADERBOARD_DATASETS:
        name = _safe_name(dataset, split)
        manifest_path = out_dir / f"{name}.json"
        print(f"materializing {dataset}/{split} -> {manifest_path}", flush=True)
        all_data = run_eval_sslm.prepare_samples(prep_args, dataset, split)
        with open(manifest_path, "w", encoding="utf-8") as f:
            for idx, (path, dur, ref) in enumerate(
                zip(all_data["audio_filepaths"], all_data["durations"], all_data["references"])
            ):
                f.write(
                    json.dumps(
                        {
                            "id": f"{name}_{idx}",
                            "audio_filepath": str(Path(path).resolve()),
                            "duration": dur,
                            "text": ref,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        manifest_items[name] = {"manifest_filepath": str(manifest_path)}
    return manifest_items


def _resolve_model_paths(model_cfg: DictConfig) -> None:
    with open_dict(model_cfg):
        model_cfg.load_llm_weights = False
        for key in ("pretrained_llm", "pretrained_asr"):
            val = str(model_cfg.get(key, ""))
            resolved = run_eval_sslm._resolve_remote_path(val)
            if resolved != val:
                print(f"resolved {key}: {val} -> {resolved}", flush=True)
                model_cfg[key] = resolved


def _patch_lora_target_modules(model_cfg: DictConfig, state_dict: dict) -> None:
    if "lora" not in model_cfg or "target_modules" in model_cfg.lora:
        return
    modules = set()
    for key in state_dict:
        if ".lora_A." not in key:
            continue
        parts = key.split(".")
        try:
            idx = parts.index("lora_A")
        except ValueError:
            continue
        if idx > 0:
            modules.add(parts[idx - 1])
    with open_dict(model_cfg.lora):
        model_cfg.lora.target_modules = sorted(modules) if modules else "all-linear"
    print(f"patched lora.target_modules={model_cfg.lora.target_modules}", flush=True)


def load_cfgs(args):
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    ckpt_model_cfg = OmegaConf.create(hp["cfg"])
    ckpt_data_cfg = hp.get("data_cfg")
    forced_aligner_cfg = hp.get("forced_aligner", None)
    del ckpt

    if args.exp_config:
        full_cfg = OmegaConf.load(args.exp_config)
        model_cfg = ckpt_model_cfg
        data_cfg = full_cfg.data
        with open_dict(data_cfg):
            # The checkpoint data_cfg has the model-propagated dataset flags that
            # matter for validation construction (<flush>, compact template, etc.).
            data_cfg.dataset = ckpt_data_cfg if ckpt_data_cfg is not None else full_cfg.data.dataset
    else:
        if ckpt_data_cfg is None:
            raise SystemExit("Checkpoint has no data_cfg; pass --exp-config or --validation-manifest/input-cfg.")
        if args.leaderboard:
            datasets = materialize_leaderboard_manifests(args)
        else:
            val_item = {}
            if args.validation_manifest:
                val_item["manifest_filepath"] = args.validation_manifest
            if args.validation_input_cfg:
                val_item["input_cfg"] = args.validation_input_cfg
            if not val_item:
                raise SystemExit("Pass --exp-config, --leaderboard, or --validation-manifest/--validation-input-cfg.")
            datasets = {args.val_name: val_item}
        model_cfg = ckpt_model_cfg
        data_cfg = OmegaConf.create(
            {
                "dataset": OmegaConf.to_container(ckpt_data_cfg, resolve=True),
                "validation_ds": {
                    "datasets": datasets,
                    "sample_rate": int(ckpt_data_cfg.sample_rate),
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "seed": 42,
                    "shard_seed": "randomized",
                    "shuffle": False,
                    # Matches the resolved imend_flush_notrunc training config.
                    "pad_extra_duration": args.pad_extra_duration,
                    "pad_extra_duration_prob": args.pad_extra_duration_prob,
                },
            }
        )

    with open_dict(data_cfg.validation_ds):
        data_cfg.validation_ds.batch_size = args.batch_size
        data_cfg.validation_ds.num_workers = args.num_workers
        data_cfg.validation_ds.shuffle = False
        if args.limit_val_batches is not None:
            # Lightning owns limiting; keep config finite/map-style.
            pass
    return model_cfg, data_cfg, forced_aligner_cfg


def load_model_training_style(args, model_cfg, data_cfg, forced_aligner_cfg):
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    _resolve_model_paths(model_cfg)
    _patch_lora_target_modules(model_cfg, state_dict)

    forced_aligner = Serialization.from_config_dict(forced_aligner_cfg) if forced_aligner_cfg is not None else None
    model = StreamingSTTModel(
        OmegaConf.to_container(model_cfg, resolve=True),
        forced_aligner=forced_aligner,
        data_cfg=data_cfg.dataset,
        dataset_cls=StreamingSTTDataset,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"missing keys ({len(missing)}): {missing[:5]}", flush=True)
    if unexpected:
        print(f"unexpected keys ({len(unexpected)}): {unexpected[:5]}", flush=True)
    del ckpt
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--exp-config", default=None)
    parser.add_argument("--validation-manifest", default=None)
    parser.add_argument("--validation-input-cfg", default=None)
    parser.add_argument("--leaderboard", action="store_true",
                        help="Evaluate all run_eval_sslm.py leaderboard datasets via training validation_step.")
    parser.add_argument("--dataset-path", default="hf-audio/esb-datasets-test-only-sorted")
    parser.add_argument("--leaderboard-manifest-dir", default="debug_decode_configs/leaderboard_manifests")
    parser.add_argument("--max-eval-samples", type=int, default=None,
                        help="Optional cap per leaderboard split while debugging.")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false",
                        help="Disable HF streaming when materializing leaderboard manifests.")
    parser.set_defaults(streaming=True)
    parser.add_argument("--val-name", default="debug")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=float, default=None)
    parser.add_argument("--precision", default="bf16-true", choices=["bf16-true", "32-true"])
    parser.add_argument("--pad-extra-duration", type=float, default=0.5)
    parser.add_argument("--pad-extra-duration-prob", type=float, default=1.0)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_float32_matmul_precision("medium")

    model_cfg, data_cfg, forced_aligner_cfg = load_cfgs(args)
    model = load_model_training_style(args, model_cfg, data_cfg, forced_aligner_cfg)

    encoder_reuse_k = int(getattr(model.core_cfg, "encoder_reuse_k", 1) or 1)
    defer_get_batch = model.forced_aligner is not None or encoder_reuse_k > 1
    dataset = StreamingSTTDataset(cfg=data_cfg.dataset, tokenizer=model.tokenizer, defer_get_batch=defer_get_batch)
    datamodule = DataModule(data_cfg, tokenizer=model.tokenizer, dataset=dataset)

    trainer_cfg = OmegaConf.create(
        {
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": [args.device] if torch.cuda.is_available() else 1,
            "num_nodes": 1,
            "precision": args.precision,
            "logger": False,
            "enable_checkpointing": False,
            "use_distributed_sampler": False,
            "limit_val_batches": args.limit_val_batches if args.limit_val_batches is not None else 1.0,
            "num_sanity_val_steps": 0,
        }
    )
    trainer = Trainer(**resolve_trainer_cfg(trainer_cfg))
    metrics = trainer.validate(model=model, datamodule=datamodule, verbose=False)
    for row in metrics:
        for key in sorted(row):
            if key == "val_wer" or key.startswith("val_wer_"):
                print(f"{key}: {float(row[key]):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
