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
"""CHAT transducer trained on a forced alignment.

THE EXPERIMENT. A larger text vocabulary clearly helps the SpeechLM: swapping
Qwen's 151,936 pieces for the ASR encoder's 1,024 cost ~0.14 WER on the
leaderboard. Whether a TRANSDUCER benefits the same way is unknown, and normally
untestable: the RNN-T loss marginalises over all alignments and needs a
[B, T, U, V] tensor, which at V=151,936 does not fit.

Fixing the alignment removes the marginalisation. Each word is emitted at the
chunk holding its last token -- the same assignment the SpeechLM trains on -- so
the loss is plain cross-entropy along a single path of U + T steps, scored by
``RNNTAttJoint.joint_on_path``. Memory then grows with the path, not with
T * U * V, and the two vocabularies become directly comparable.

WHY A FIXED ALIGNMENT IS NOT A HANDICAP. Marginalising over alignments exists to
LEARN an alignment you do not have; given a good one, conditioning on it is hard
EM rather than soft EM. The SpeechLM is trained exactly this way -- same forced
alignment, same delay, same chunk assignment -- and reaches 5.96 macro against
the RNN-T baseline's 5.82, so the recipe demonstrably works at this scale.

Crucially, DECODING MIRRORS TRAINING: RNNTAttJoint's CHAT greedy decode walks
chunks and emits tokens within a chunk until a blank, with the prediction state
advancing only on real tokens -- precisely the path built here. There is no
search over alignments at inference, so there is no train/test mismatch of the
kind that would otherwise punish single-path training.

What IS fixed by construction is emission latency: a word is emitted at its
aligned chunk plus the configured delay, rather than wherever the model finds
convenient. That is the same deliberate trade the SpeechLM makes to keep
streaming latency controllable.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from nemo.collections.asr.modules import RNNTAttJoint, RNNTDecoder
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers
from nemo.collections.speechlm2.parts.pretrained import setup_perception
from nemo.collections.speechlm2.parts.utils import to_dataclass
from nemo.utils import logging


@dataclass
class ChatSTTModelConfig:
    """Config for the forced-alignment CHAT transducer.

    Attributes:
        pretrained_asr: ``.nemo`` whose encoder (and, for the small-vocabulary
            arm, whose tokenizer) is used.
        chunk_size: encoder frames per chunk. Fixed, not sampled: the dataset's
            chunk indices and the joint's own chunking must agree exactly.
        vocab_size: number of text classes, EXCLUDING blank. Blank is class
            ``vocab_size`` (NeMo's convention).
        text_vocab_from_asr: use the ASR encoder's SentencePiece vocabulary
            (~1,024) instead of the LLM tokenizer named by ``pretrained_llm``.
            This is the knob the experiment turns.
    """

    pretrained_asr: str
    chunk_size: int = 14
    vocab_size: int = 1024
    pretrained_llm: str = ""
    text_vocab_from_asr: bool = True
    pred_hidden: int = 640
    pred_rnn_layers: int = 2
    joint_hidden: int = 640
    att_context_size: Optional[list] = None
    load_asr_weights: bool = True
    freeze_speech_encoder: bool = False
    audio_pad_to: int = 0


class ChatSTTModel(LightningModule):
    """Encoder + prediction network + chunk-attention joint, forced-alignment CE."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)
        self.core_cfg: ChatSTTModelConfig = to_dataclass(ChatSTTModelConfig, cfg)

        self.chunk_size = int(self.core_cfg.chunk_size)
        # Pin the encoder's look-ahead to this chunk size so a frame never
        # depends on audio past its own chunk boundary -- the same constraint the
        # SpeechLM encoder runs under, which is what keeps the comparison honest.
        att = self.core_cfg.att_context_size or [70, self.chunk_size - 1]
        self.perception = setup_perception(
            cfg=self.cfg,
            output_dim=self.core_cfg.joint_hidden,
            pretrained_asr=self.core_cfg.pretrained_asr,
            pretrained_weights=self.core_cfg.load_asr_weights,
            audio_pad_to=self.core_cfg.audio_pad_to or None,
            att_context_size=att,
        )

        V = int(self.core_cfg.vocab_size)
        self.blank_id = V  # blank is the extra class at the end
        self.decoder = RNNTDecoder(
            prednet={
                "pred_hidden": self.core_cfg.pred_hidden,
                "pred_rnn_layers": self.core_cfg.pred_rnn_layers,
                "dropout": 0.1,
            },
            vocab_size=V,
        )
        self.joint = RNNTAttJoint(
            jointnet={
                "encoder_hidden": self.core_cfg.joint_hidden,
                "pred_hidden": self.core_cfg.pred_hidden,
                "joint_hidden": self.core_cfg.joint_hidden,
                "activation": "relu",
                "dropout": 0.1,
            },
            num_classes=V,
            chunk_size=self.chunk_size,
        )
        logging.info(
            f"ChatSTTModel: vocab={V} (+1 blank), chunk_size={self.chunk_size}, "
            f"att_context_size={att}, joint_hidden={self.core_cfg.joint_hidden}"
        )

    def forward_loss(self, batch) -> torch.Tensor:
        """Cross-entropy along the forced path."""
        enc, enc_len = self.perception(input_signal=batch.audios, input_signal_length=batch.audio_lens)
        if enc.shape[1] != enc_len.max():  # perception may return (B, D, T)
            enc = enc.transpose(1, 2)

        # (B, D, U+1) -> (B, U+1, D); the decoder prepends its own SOS.
        g, _, _ = self.decoder(targets=batch.pred_input, target_length=batch.pred_lens)
        g = g.transpose(1, 2)

        logits = self.joint.joint_on_path(enc, g, batch.b_idx, batch.t_idx, batch.u_idx, enc_len)

        # The dataset numbers chunks from the message structure; the joint numbers
        # them by re-chunking the encoder output. If those disagree, t_idx points
        # at the wrong chunk -- or out of range -- and the model simply fails to
        # learn, with nothing in the logs to say why. Check rather than trust.
        produced = self.joint.num_chunks_per_utterance
        if produced is not None:
            expected = batch.n_chunks.to(produced.device)
            if not torch.equal(produced, expected):
                bad = (produced != expected).nonzero(as_tuple=True)[0][:5].tolist()
                raise RuntimeError(
                    "chunk-count mismatch between the dataset's alignment and the joint's chunking: "
                    f"joint={produced[bad].tolist()} vs dataset={expected[bad].tolist()} (utterances {bad}). "
                    "The two must use the same chunk_size and the same frame count."
                )

        return F.cross_entropy(logits.float(), batch.labels)

    def training_step(self, batch, batch_idx):
        loss = self.forward_loss(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.forward_loss(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return configure_optimizers(self)


__all__ = ["ChatSTTModel", "ChatSTTModelConfig"]
