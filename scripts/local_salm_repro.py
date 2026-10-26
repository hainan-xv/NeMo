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
"""Reproduce the offline-SALM NaN on one local GPU, with real optimizer steps.

The 8-node runs go NaN within a handful of steps, but each diagnosis round costs an
allocation. Everything needed to reproduce is available locally: Qwen3-1.7B, the
nemotron .nemo the perception stack is built from, and LibriSpeech. This runs the
real SALM training_step -- forward, backward, optimizer -- and reports, per step,
whether the loss, the activations, the gradients and the parameters are finite, so
the first thing to break is visible directly instead of inferred from a log tail.

It deliberately runs the SAME code path as training (SALM.training_step), not a
hand-rolled imitation, so a clean result here is evidence about the real thing.
"""
import argparse
import json

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset


def first_bad(named):
    return [n for n, t in named if t is not None and not torch.isfinite(t).all()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="examples/speechlm2/conf/salm_nocoder.yaml")
    p.add_argument("--llm", default="Qwen/Qwen3-1.7B")
    p.add_argument("--asr", default="/home/hainanx/Workplace/chat_diag/nemotron-speech-streaming-en-0.6b.nemo")
    p.add_argument("--manifest", default="/home/hainanx/Workplace/data/librispeech/train_960.json")
    p.add_argument("--val_manifest", default="/home/hainanx/Workplace/data/librispeech/test_clean.json")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--precision", default=None, help="override trainer.precision, e.g. bf16-true / 32")
    # Keep the YAML's data pipeline (granary input_cfg + token bucketing) instead of a
    # plain local manifest. LibriSpeech at matched length and batch does NOT reproduce
    # the NaN, so the remaining suspects are the data and the sampler; this flag runs
    # the real ones on a single GPU rather than on 8 nodes.
    p.add_argument("--train_input_cfg", default=None)
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.model.pretrained_llm = args.llm
    cfg.model.pretrained_asr = args.asr
    if args.precision:
        cfg.trainer.precision = args.precision

    # Plain manifest instead of the cluster's granary input_cfg, and a fixed batch
    # size instead of bucketing -- the NaN appears within ~10 steps, so the sampler
    # is not the variable under test.
    if args.train_input_cfg:
        cfg.data.train_ds.input_cfg[0].input_cfg = args.train_input_cfg
        cfg.data.validation_ds.datasets.mcv_11_dev.input_cfg[0].manifest_filepath = args.val_manifest
    else:
        for ds in (cfg.data.train_ds, cfg.data.validation_ds.datasets.mcv_11_dev):
            ds.input_cfg[0].pop("input_cfg", None)
            ds.input_cfg[0].manifest_filepath = args.manifest
        cfg.data.train_ds.manifest_filepath = args.manifest
        for k in (
            "use_bucketing",
            "use_multimodal_sampling",
            "bucket_duration_bins",
            "bucket_batch_size",
            "max_tokens",
        ):
            cfg.data.train_ds.pop(k, None)
        cfg.data.train_ds.batch_size = args.batch_size
        cfg.data.validation_ds.datasets.mcv_11_dev.input_cfg[0].manifest_filepath = args.val_manifest
    cfg.data.train_ds.num_workers = 2
    OmegaConf.resolve(cfg)

    dtype = torch.bfloat16 if "bf16" in str(cfg.trainer.precision) else torch.float32
    print(f"### precision={cfg.trainer.precision} -> {dtype}, lr={args.lr}, bs={args.batch_size}", flush=True)

    model = SALM(OmegaConf.to_container(cfg.model, resolve=True)).to(args.device).to(dtype)
    model.train()
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    print(f"### trainable tensors: {len(trainable)} / {sum(1 for _ in model.parameters())}", flush=True)

    dataset = SALMDataset(tokenizer=model.tokenizer, multispeaker_cfg=cfg.data.get("multispeaker_cfg", None))
    dm = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)
    dm.setup("fit")
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    for i, batch in enumerate(dm.train_dataloader()):
        if i >= args.steps:
            break
        batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        inputs = model.prepare_inputs(batch)
        out = model(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
        n = (inputs["target_ids"] != -100).long().sum()
        loss = torch.nn.functional.cross_entropy(
            out["logits"].flatten(0, 1).float(),
            inputs["target_ids"].flatten(0, 1),
            reduction="sum",
            ignore_index=-100,
        ) / n.clamp(min=1)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        bad_g = first_bad(model.named_parameters() and [(n_, p_.grad) for n_, p_ in model.named_parameters()])
        opt.step()
        bad_p = first_bad([(n_, p_) for n_, p_ in model.named_parameters()])

        flag = ""
        if not torch.isfinite(loss):
            flag += " LOSS_NAN"
        if bad_g:
            flag += f" GRAD_NAN({len(bad_g)}) first={bad_g[0]}"
        if bad_p:
            flag += f" PARAM_NAN({len(bad_p)})"
        print(
            f"[{i:3d}] loss={loss.item():9.4f} n_tgt={int(n):5d} T={inputs['input_embeds'].shape[1]:5d} "
            f"emb_finite={bool(torch.isfinite(inputs['input_embeds']).all())} "
            f"logit_absmax={out['logits'].float().abs().max().item():.4g}{flag}",
            flush=True,
        )
        if flag:
            print("\n*** first failure above; stopping.", flush=True)
            break
    else:
        print(f"\nno failure in {args.steps} steps.", flush=True)


if __name__ == "__main__":
    main()
