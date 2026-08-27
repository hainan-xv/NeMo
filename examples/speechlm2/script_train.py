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
"""Train the SCRIPT streaming SpeechLM (packed spine + per-chunk branches).

Mirrors ``streaming_stt_train.py`` but instantiates ``ScriptSTTModel`` /
``ScriptSTTDataset``. Run with, e.g.::

    python examples/speechlm2/script_train.py \
        --config-path=examples/speechlm2/conf \
        --config-name=streaming_stt_granary2_lora_script
"""

import os
from copy import deepcopy

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.speechlm2 import DataModule, ScriptSTTDataset, ScriptSTTModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="streaming_stt_granary2_lora_script")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    dataset_cfg = cfg.data.dataset

    # The audio-window settings must agree between the model and the dataset: the
    # dataset builds each branch's window from them, and the model rebuilds the
    # same window at inference. A mismatch trains and decodes on different
    # conditioning, which is silent and hard to spot, so fail loudly here.
    # prompt_control is in the same category: the dataset writes the control
    # sentence into the instruction, the model rewrites it at inference. If only
    # one side has it on, every decode is out of distribution.
    for key in (
        "audio_history_chunks",
        "audio_window_frames",
        "twod_layout",
        "prompt_control",
        "read_write",
        "gate_in_history",
    ):
        model_val = int(cfg.model.get(key, 0) or 0)
        data_val = int(dataset_cfg.get(key, 0) or 0)
        if model_val != data_val:
            raise ValueError(
                f"model.{key} ({model_val}) != data.dataset.{key} ({data_val}); they must match. "
                f"Set data.dataset.{key}: ${{model.{key}}} in the config."
            )

    # position_scheme is a STRING, so it cannot ride the int() loop above.
    ps_model = str(cfg.model.get("position_scheme", "branch"))
    ps_data = str(dataset_cfg.get("position_scheme", "branch"))
    if ps_model != ps_data:
        raise ValueError(
            f"model.position_scheme ({ps_model}) != data.dataset.position_scheme ({ps_data}); "
            "they must match, or training and inference lay out RoPE positions differently."
        )
    if ps_model not in ("branch", "continuous", "sampled"):
        raise ValueError(f"position_scheme must be 'branch', 'continuous' or 'sampled', got {ps_model!r}")

    # Validation dataset config = training config with val_dataset_overrides on
    # top (e.g. pinning a single chunk_size for the decode-only WER pass).
    val_dataset_overrides = cfg.data.get("val_dataset_overrides", None)
    if val_dataset_overrides is not None:
        val_dataset_cfg = deepcopy(dataset_cfg)
        if not isinstance(val_dataset_cfg, DictConfig):
            val_dataset_cfg = OmegaConf.create(val_dataset_cfg)
        with open_dict(val_dataset_cfg):
            val_dataset_cfg.update(val_dataset_overrides)
    else:
        val_dataset_cfg = None

    with trainer.init_module():
        model = ScriptSTTModel(
            OmegaConf.to_container(cfg.model, resolve=True),
            data_cfg=dataset_cfg,
            val_data_cfg=val_dataset_cfg,
            dataset_cls=ScriptSTTDataset,
        )

    dataset = ScriptSTTDataset(cfg=dataset_cfg, tokenizer=model.tokenizer)
    val_dataset = (
        ScriptSTTDataset(cfg=val_dataset_cfg, tokenizer=model.tokenizer) if val_dataset_cfg is not None else None
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset, val_dataset=val_dataset)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
