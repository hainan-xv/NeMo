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
"""Train the CHAT transducer on a forced alignment.

Two arms are meant to be run and compared -- a ~1k ASR vocabulary and a 151,936
Qwen one -- differing ONLY in the tokenizer, to ask whether a transducer gains
from a large vocabulary the way the SpeechLM does.

    python chat_train.py --config-name streaming_stt_granary2_chat_asrvocab
    python chat_train.py --config-name streaming_stt_granary2_chat_qwenvocab
"""

import os

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2.data import DataModule
from nemo.collections.speechlm2.data.chat_dataset import ChatAlignedDataset
from nemo.collections.speechlm2.models.chat_model import ChatSTTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def _build_tokenizer(cfg):
    """The tokenizer under test: the ASR encoder's, or the LLM's.

    This is the only intended difference between the two arms, so it is resolved
    in one place and its size is logged -- a silent fallback to the wrong
    vocabulary would make the comparison meaningless while still training.
    """
    if cfg.model.get("text_vocab_from_asr", True):
        import tempfile

        from nemo.collections.speechlm2.parts.asr_vocab import AsrVocabTokenizer, extract_spm_from_nemo

        spm = extract_spm_from_nemo(cfg.model.pretrained_asr, tempfile.mkdtemp(prefix="chat_vocab_"))
        tok = AsrVocabTokenizer(spm)
        logging.info(f"CHAT vocabulary: ASR SentencePiece from {cfg.model.pretrained_asr} -> {len(tok)} pieces")
        return tok

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

    tok = AutoTokenizer(cfg.model.pretrained_llm, use_fast=True)
    logging.info(f"CHAT vocabulary: {cfg.model.pretrained_llm} -> {len(tok.tokenizer)} pieces")
    return tok


@hydra_runner(config_path="conf", config_name="streaming_stt_granary2_chat_asrvocab")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    tokenizer = _build_tokenizer(cfg)
    vocab = len(tokenizer) if hasattr(tokenizer, "__len__") else len(tokenizer.tokenizer)

    # The joint's output layer and the prediction network's embedding are both
    # sized from this. If it disagreed with the tokenizer the model would still
    # train, just against wrong or unreachable classes, so pin it here.
    if int(cfg.model.vocab_size) != int(vocab):
        logging.warning(f"model.vocab_size={cfg.model.vocab_size} != tokenizer size {vocab}; using the tokenizer's.")
        cfg.model.vocab_size = int(vocab)

    dataset_cfg = cfg.data.dataset
    val_dataset_cfg = cfg.data.get("val_dataset", dataset_cfg)

    # chunk_size is FIXED for CHAT: the dataset's chunk indices and the joint's
    # own re-chunking must agree, so a sampled chunk size would silently
    # misalign them.
    if isinstance(dataset_cfg.chunk_size, (list, tuple)):
        raise ValueError(f"CHAT needs a single chunk_size, got {dataset_cfg.chunk_size}")
    if int(dataset_cfg.chunk_size) != int(cfg.model.chunk_size):
        raise ValueError(
            f"data.dataset.chunk_size={dataset_cfg.chunk_size} != model.chunk_size={cfg.model.chunk_size}; "
            "the dataset's chunk indices would not match the joint's chunking."
        )

    with trainer.init_module():
        model = ChatSTTModel(OmegaConf.to_container(cfg.model, resolve=True))

    dataset = ChatAlignedDataset(cfg=dataset_cfg, tokenizer=tokenizer, blank_id=model.blank_id)
    val_dataset = (
        ChatAlignedDataset(cfg=val_dataset_cfg, tokenizer=tokenizer, blank_id=model.blank_id)
        if val_dataset_cfg is not None
        else None
    )
    datamodule = DataModule(cfg.data, tokenizer=tokenizer, dataset=dataset, val_dataset=val_dataset)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
