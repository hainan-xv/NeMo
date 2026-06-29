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
import os

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf, open_dict

from nemo.collections.speechlm2 import DataModule, StreamingSTTDataset, StreamingSTTModel
from nemo.core.classes.common import Serialization
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="streaming_stt")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    # Only global rank 0 writes the resolved config. Letting all ranks write the
    # same file into the shared (Lustre) log_dir races exp_manager's own rank-0
    # `run_{N}` rename of this exact file, which deadlocks the Lustre MDS: rank 0
    # wedges in `ptlrpc_set_wait` on the rename while the other ranks pile up in
    # `rwsem_down_write_slowpath` on the same directory. The stuck ranks never
    # reach the first NCCL collective, so the job dies with a 600s
    # `store->get('0')` timeout and leaves un-killable D-state processes.
    if is_global_rank_zero():
        OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    dataset_cfg = cfg.data.dataset
    with open_dict(dataset_cfg):
        dataset_cfg.supervise_im_end_in_loss = cfg.model.get("supervise_im_end_in_loss", False)
        dataset_cfg.project_unaligned_text_to_chunks = cfg.model.get("project_unaligned_text_to_chunks", False)
        dataset_cfg.max_audio_chunks_per_turn = cfg.model.get("max_audio_chunks_per_turn", 1)
    forced_aligner_cfg = cfg.get("forced_aligner", None)
    if forced_aligner_cfg is not None:
        forced_aligner = Serialization.from_config_dict(forced_aligner_cfg)
        defer_get_batch = True
        logging.info(f"Using online forced alignment: {forced_aligner_cfg}")
    else:
        forced_aligner = None
        defer_get_batch = False

    with trainer.init_module():
        model = StreamingSTTModel(
            OmegaConf.to_container(cfg.model, resolve=True),
            forced_aligner=forced_aligner,
            data_cfg=dataset_cfg,
            dataset_cls=StreamingSTTDataset,
        )

    dataset = StreamingSTTDataset(cfg=dataset_cfg, tokenizer=model.tokenizer, defer_get_batch=defer_get_batch)
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
