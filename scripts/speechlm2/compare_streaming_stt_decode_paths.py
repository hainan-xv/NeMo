#!/usr/bin/env python3
"""Compare StreamingSTT validation-style decode against run_eval_sslm native decode.

This is a diagnostic for train/inference mismatches. It:

1. Loads a checkpoint in the same style as training validation, including the
   training data config and validation dataloader.
2. Runs the model's validation WER decode path on real validation utterances:
   ``_generate_hypotheses_for_wer``.
3. Frees that model, reloads the same checkpoint via ``run_eval_sslm.load_model``,
   and runs ``run_eval_sslm.transcribe_sslm`` on the exact same audio tensors.
4. Prints paired outputs and exits non-zero if any hypothesis differs.

Typical usage:

  python scripts/speechlm2/compare_streaming_stt_decode_paths.py \
      --ckpt checkpoints/imend_flush_notrunc/step=6001.ckpt \
      --exp-config /path/to/imend_flush_notrunc/exp_config.yaml \
      --device 1 --max-samples 8

If you do not have the full exp_config.yaml, pass the validation source directly:

  python scripts/speechlm2/compare_streaming_stt_decode_paths.py \
      --ckpt checkpoints/imend_flush_notrunc/step=6001.ckpt \
      --validation-input-cfg /path/to/validation_input_cfg.yaml \
      --device 1 --max-samples 8
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.speechlm2 import DataModule, StreamingSTTDataset, StreamingSTTModel
from nemo.collections.speechlm2.data.streaming_stt_dataset import get_word_alignments_for_batch
from nemo.core.classes.common import Serialization

import run_eval_sslm


@dataclass
class DecodeSample:
    ref: str
    offline_direct_hyp: str
    direct_before_hyp: str
    train_wrapper_before_hyp: str
    train_hyp: str
    direct_after_hyp: str
    train_wrapper_after_hyp: str
    audio: torch.Tensor
    audio_len: int


class TensorBatchLoader:
    def __init__(self, audios: list[torch.Tensor], audio_lens: list[int], batch_size: int) -> None:
        self.audios = audios
        self.audio_lens = audio_lens
        self.batch_size = batch_size

    def __iter__(self):
        for start in range(0, len(self.audios), self.batch_size):
            wavs = self.audios[start : start + self.batch_size]
            lens = self.audio_lens[start : start + self.batch_size]
            max_len = max(lens)
            batch = torch.stack([F.pad(w[:n], (0, max_len - n)) for w, n in zip(wavs, lens)])
            yield {
                "audios": batch,
                "audio_lens": torch.tensor(lens, dtype=torch.long),
            }


def _resolve_model_paths(model_cfg: DictConfig) -> None:
    with open_dict(model_cfg):
        model_cfg.load_llm_weights = False
        for key in ("pretrained_llm", "pretrained_asr"):
            val = str(model_cfg.get(key, ""))
            resolved = run_eval_sslm._resolve_remote_path(val)
            if resolved != val:
                print(f"resolved {key}: {val} -> {resolved}")
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
    print(f"patched lora.target_modules={model_cfg.lora.target_modules}")


def _load_training_style_model(
    ckpt_path: str,
    device: torch.device,
    model_cfg: DictConfig,
    data_cfg: DictConfig,
    forced_aligner_cfg=None,
    dtype=torch.bfloat16,
) -> StreamingSTTModel:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    _resolve_model_paths(model_cfg)
    _patch_lora_target_modules(model_cfg, state_dict)

    forced_aligner = Serialization.from_config_dict(forced_aligner_cfg) if forced_aligner_cfg is not None else None
    model = StreamingSTTModel(
        OmegaConf.to_container(model_cfg, resolve=True),
        forced_aligner=forced_aligner,
        data_cfg=data_cfg,
        dataset_cls=StreamingSTTDataset,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"training-style load missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"training-style load unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    del ckpt
    model = model.eval().to(dtype).to(device)
    model._ensure_inference_cache()
    return model


def _load_configs(args) -> tuple[DictConfig, DictConfig, DictConfig | None]:
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    model_cfg = OmegaConf.create(hp["cfg"])
    ckpt_data_cfg = hp.get("data_cfg")
    forced_aligner_cfg = hp.get("forced_aligner", None)
    del ckpt

    if args.exp_config:
        full = OmegaConf.load(args.exp_config)
        data_root = full.data
        data_cfg = full.data.dataset
        if "model" in full:
            # The checkpoint is still the source of truth for model weights/config,
            # but use the saved data config because it has the validation loader.
            with open_dict(data_cfg):
                pass
    else:
        if ckpt_data_cfg is None:
            raise SystemExit(
                "Checkpoint has no data_cfg. Pass --exp-config or --validation-input-cfg/--validation-manifest."
            )
        data_cfg = ckpt_data_cfg
        val_item = {}
        if args.validation_input_cfg:
            val_item["input_cfg"] = args.validation_input_cfg
        if args.validation_manifest:
            val_item["manifest_filepath"] = args.validation_manifest
        if not val_item:
            raise SystemExit("Pass --exp-config or --validation-input-cfg/--validation-manifest for val utterances.")
        data_root = OmegaConf.create(
            {
                "dataset": OmegaConf.to_container(data_cfg, resolve=True),
                "validation_ds": {
                    "datasets": {"debug": val_item},
                    "sample_rate": int(data_cfg.sample_rate),
                    "batch_size": args.val_batch_size,
                    "num_workers": args.num_workers,
                    "seed": 42,
                    "shard_seed": "randomized",
                },
            }
        )

    with open_dict(data_root.validation_ds):
        data_root.validation_ds.batch_size = args.val_batch_size
        data_root.validation_ds.num_workers = args.num_workers
    with open_dict(data_root):
        data_root.dataset = data_cfg
    return model_cfg, data_root, forced_aligner_cfg


def _first_loader(val_loader, name: str | None):
    if hasattr(val_loader, "iterables"):
        loaders = val_loader.iterables
        if name:
            return name, loaders[name]
        first_name = next(iter(loaders))
        return first_name, loaders[first_name]
    if isinstance(val_loader, dict):
        if name:
            return name, val_loader[name]
        first_name = next(iter(val_loader))
        return first_name, val_loader[first_name]
    return name or "val", val_loader


def _prepare_like_validation(model: StreamingSTTModel, batch):
    if isinstance(batch, dict) and "audios" not in batch:
        # CombinedLoader can yield {name: batch}; select the first non-empty batch.
        batch = next(v for v in batch.values() if v is not None)
    if batch.input_tokens is None and model.dataset is not None:
        if model.forced_aligner is not None:
            alignments = model.forced_aligner.align(batch.audios, batch.audio_lens, batch.text)
        else:
            alignments = get_word_alignments_for_batch(batch.cuts)
        batch = model.dataset.get_batch_data(
            cuts=batch.cuts,
            audios=batch.audios,
            audio_lens=batch.audio_lens,
            alignments=alignments,
            text=batch.text,
            randomize_fixed_chunk_groups=False,
            apply_random_delay=False,
        )
    return move_data_to_device(batch, model.device)


def collect_training_decode_samples(args, device: torch.device) -> tuple[list[DecodeSample], DictConfig]:
    model_cfg, data_root, forced_aligner_cfg = _load_configs(args)
    data_cfg = data_root.dataset
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = _load_training_style_model(args.ckpt, device, model_cfg, data_cfg, forced_aligner_cfg, dtype=dtype)

    encoder_reuse_k = int(getattr(model.core_cfg, "encoder_reuse_k", 1) or 1)
    defer_get_batch = model.forced_aligner is not None or encoder_reuse_k > 1
    dataset = StreamingSTTDataset(cfg=data_cfg, tokenizer=model.tokenizer, defer_get_batch=defer_get_batch)
    datamodule = DataModule(data_root, tokenizer=model.tokenizer, dataset=dataset)
    loader_name, loader = _first_loader(datamodule.val_dataloader(), args.val_name)
    print(f"using validation loader: {loader_name}")

    samples: list[DecodeSample] = []
    with torch.inference_mode():
        for batch in loader:
            batch = _prepare_like_validation(model, batch)
            batch_samples: list[DecodeSample] = []
            for i in range(len(batch.text)):
                n = int(batch.audio_lens[i].item())
                batch_samples.append(
                    DecodeSample(
                        ref=batch.text[i] if batch.text is not None else "",
                        offline_direct_hyp="",
                        direct_before_hyp="",
                        train_wrapper_before_hyp="",
                        train_hyp="",
                        direct_after_hyp="",
                        train_wrapper_after_hyp="",
                        audio=batch.audios[i, :n].detach().cpu(),
                        audio_len=n,
                    )
                )

            offline_direct = model.generate(
                audios=batch.audios,
                audio_lens=batch.audio_lens,
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens
                if args.max_new_tokens is not None
                else int(getattr(model.core_cfg, "max_new_tokens_per_chunk", 10)),
                use_offline_embs=True,
            )
            direct_before = model.generate(
                audios=batch.audios,
                audio_lens=batch.audio_lens,
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens
                if args.max_new_tokens is not None
                else int(getattr(model.core_cfg, "max_new_tokens_per_chunk", 10)),
            )
            wrapper_before = run_eval_sslm.transcribe_sslm(
                model,
                TensorBatchLoader(
                    [s.audio for s in batch_samples],
                    [s.audio_len for s in batch_samples],
                    batch_size=args.eval_batch_size,
                ),
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                warmup_decode=args.warmup_wrapper,
            )
            hyps = model._generate_hypotheses_for_wer(batch, parallel=False)
            direct_after = model.generate(
                audios=batch.audios,
                audio_lens=batch.audio_lens,
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens
                if args.max_new_tokens is not None
                else int(getattr(model.core_cfg, "max_new_tokens_per_chunk", 10)),
            )
            wrapper_after = run_eval_sslm.transcribe_sslm(
                model,
                TensorBatchLoader(
                    [s.audio for s in batch_samples],
                    [s.audio_len for s in batch_samples],
                    batch_size=args.eval_batch_size,
                ),
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                warmup_decode=args.warmup_wrapper,
            )
            for sample, offline, direct_b, before, hyp, direct_a, after in zip(
                batch_samples, offline_direct, direct_before, wrapper_before, hyps, direct_after, wrapper_after
            ):
                sample.offline_direct_hyp = offline
                sample.direct_before_hyp = direct_b
                sample.train_wrapper_before_hyp = before
                sample.train_hyp = hyp
                sample.direct_after_hyp = direct_a
                sample.train_wrapper_after_hyp = after
                samples.append(sample)
                if len(samples) >= args.max_samples:
                    break
            if len(samples) >= args.max_samples:
                break

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return samples, model_cfg


def run_eval_decode(args, samples: list[DecodeSample], device: torch.device) -> list[str]:
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = run_eval_sslm.load_model(args.ckpt, device, dtype=dtype)
    model._ensure_inference_cache()
    loader = TensorBatchLoader(
        [s.audio for s in samples],
        [s.audio_len for s in samples],
        batch_size=args.eval_batch_size,
    )
    hyps = run_eval_sslm.transcribe_sslm(
        model,
        loader,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        warmup_decode=args.warmup_wrapper,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hyps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Lightning checkpoint path")
    parser.add_argument("--exp-config", default=None, help="Training exp_config.yaml with data.validation_ds")
    parser.add_argument("--validation-input-cfg", default=None, help="Fallback validation input_cfg yaml")
    parser.add_argument("--validation-manifest", default=None, help="Fallback validation manifest")
    parser.add_argument("--val-name", default=None, help="Validation loader name when multiple are configured")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--system-prompt", default="Transcribe the audio into text.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--warmup-wrapper", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    samples, _model_cfg = collect_training_decode_samples(args, device)
    if not samples:
        raise SystemExit("No validation samples collected.")
    eval_hyps = run_eval_decode(args, samples, device)

    mismatches = 0
    print(
        "\nidx | match | REF | DIRECT_BEFORE | WRAPPER_BEFORE | TRAIN_VAL_DECODE | "
        "DIRECT_AFTER | WRAPPER_AFTER | RUN_EVAL_LOAD_WRAPPER | OFFLINE_EMBS"
    )
    print("-" * 100)
    for i, (sample, eval_hyp) in enumerate(zip(samples, eval_hyps)):
        match = (
            sample.direct_before_hyp
            == sample.train_wrapper_before_hyp
            == sample.train_hyp
            == sample.direct_after_hyp
            == sample.train_wrapper_after_hyp
            == eval_hyp
        )
        mismatches += int(not match)
        print(f"[{i}] match={match}")
        print(f"  REF  : {sample.ref}")
        print(f"  OFFLINE_EMBS : {sample.offline_direct_hyp}")
        print(f"  DIRECT_BEFORE: {sample.direct_before_hyp}")
        print(f"  WRAP_BEFORE  : {sample.train_wrapper_before_hyp}")
        print(f"  TRAIN_VAL    : {sample.train_hyp}")
        print(f"  DIRECT_AFTER : {sample.direct_after_hyp}")
        print(f"  WRAP_AFTER   : {sample.train_wrapper_after_hyp}")
        print(f"  EVAL_WRAPPER : {eval_hyp}")

    print(f"\nCompared {len(samples)} samples; mismatches={mismatches}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
