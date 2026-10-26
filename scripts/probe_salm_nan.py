# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Localize where a non-finite value enters the offline SALM audio pathway.

Training showed ``loss=nan`` with ``finite embeds=False``, i.e. the NaN is present
before the LLM runs. That leaves four candidate stages, and this probe checks each
one separately on real batches so the answer is a measurement rather than a guess:

    raw audio  ->  preprocessor (log-mel)  ->  SpecAugment  ->  encoder  ->  projector

For the first batch that goes non-finite it prints the stage, the per-example
finiteness mask and the offending cut ids, so the bad data (or the bad op) can be
named exactly. Single GPU, no DDP, no optimizer -- this is a read-only pass.
"""
import argparse

import torch
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


def _per_example_finite(t: torch.Tensor) -> list[bool]:
    flat = t.reshape(t.shape[0], -1)
    return [bool(x) for x in torch.isfinite(flat).all(dim=1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train_input_cfg", required=True)
    p.add_argument("--val_manifest", required=True)
    p.add_argument("--pretrained_llm", required=True)
    p.add_argument("--pretrained_asr", required=True)
    p.add_argument("--batches", type=int, default=400)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # MUST default to bf16: training runs precision="bf16-true", which casts the whole
    # stack -- preprocessor included -- to bfloat16. A float32 probe passes cleanly and
    # proves nothing about the run it is meant to explain.
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.model.pretrained_llm = args.pretrained_llm
    cfg.model.pretrained_asr = args.pretrained_asr
    cfg.data.train_ds.input_cfg[0].input_cfg = args.train_input_cfg
    cfg.data.train_ds.num_workers = args.num_workers
    cfg.data.train_ds.seed = args.seed
    cfg.data.validation_ds.datasets.mcv_11_dev.input_cfg[0].manifest_filepath = args.val_manifest
    OmegaConf.resolve(cfg)

    torch.set_float32_matmul_precision("medium")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"### probing in {args.dtype} (training uses {cfg.trainer.precision})", flush=True)
    model = SALM(OmegaConf.to_container(cfg.model, resolve=True)).cuda().eval()
    model = model.to(dtype)
    # SpecAugment only runs in training mode, and it is one of the suspects, so the
    # perception stack is probed in the same mode training uses.
    model.perception.train()

    dataset = SALMDataset(tokenizer=model.tokenizer, multispeaker_cfg=cfg.data.get("multispeaker_cfg", None))
    dm = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)
    dm.setup("fit")

    perception = model.perception
    n_bad = 0
    for i, batch in enumerate(dm.train_dataloader()):
        if i >= args.batches:
            break
        audios = batch["audios"].cuda().to(dtype)
        audio_lens = batch["audio_lens"].cuda()
        cut_ids = [getattr(c, "id", "?") for c in batch.get("cuts", [])]

        stages = {}
        with torch.no_grad():
            stages["audio"] = audios
            mel, mel_len = perception.maybe_preprocess_audio(audios, audio_lens, None, None)
            stages["mel"] = mel
            aug = mel
            if perception.spec_augmentation is not None:
                aug = perception.spec_augmentation(input_spec=mel, length=mel_len)
                stages["specaug"] = aug
            enc, enc_len = perception.encoder(audio_signal=aug, length=mel_len)
            stages["encoder"] = enc
            stages["projector"] = perception.proj(enc.transpose(1, 2))

        first_bad = next((name for name, t in stages.items() if not _finite(t)), None)
        if first_bad is None:
            if i % 50 == 0:
                print(f"[{i}] ok  B={audios.shape[0]} max_dur={audio_lens.max().item()/16000:.1f}s", flush=True)
            continue

        n_bad += 1
        mask = _per_example_finite(stages[first_bad])
        bad_rows = [j for j, ok in enumerate(mask) if not ok]
        print(f"\n*** BATCH {i}: first non-finite stage = {first_bad}", flush=True)
        print(f"    stage finiteness: {{{', '.join(f'{k}:{_finite(v)}' for k, v in stages.items())}}}", flush=True)
        for j in bad_rows:
            dur = audio_lens[j].item() / 16000
            raw = audios[j, : audio_lens[j]]
            print(
                f"    row {j}: dur={dur:.3f}s samples={audio_lens[j].item()} "
                f"finite_raw={_finite(raw)} min={raw.min().item():.4g} max={raw.max().item():.4g} "
                f"cut={cut_ids[j] if j < len(cut_ids) else '?'}",
                flush=True,
            )
        if n_bad >= 3:
            print("\nthree bad batches seen; stopping early.", flush=True)
            break

    print(f"\ndone: {n_bad} non-finite batches in {min(i + 1, args.batches)} inspected.", flush=True)


if __name__ == "__main__":
    main()
