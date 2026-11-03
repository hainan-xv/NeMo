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

"""SCRIPT and CHAT trained jointly on ONE shared encoder.

The two families already agree on everything that matters for sharing: the same
ConformerEncoder donor, the same Qwen3 vocabulary, the same ``forced_alignment``
machinery deriving word->chunk assignments from the same lhotse cuts, and the
same 80 ms encoder frame. They differ only in the HEAD mounted on the encoder:

- SCRIPT  -- projector + Qwen3 LLM, scoring ``p(words_k | text_history, audio_k)``
- CHAT    -- RNNTDecoder + RNNTAttJoint, scoring a banded transducer lattice

so a single encoder forward can feed both, and the encoder receives gradient
from both objectives::

    L = L_script + chat_loss_weight * L_chat

LATENCY IS SHARED, NOT MERELY MATCHED. SCRIPT draws ``chunk_size`` per batch;
that same draw is pushed into ``joint.chunk_size`` before the CHAT loss runs and
into the encoder's look-ahead (``right = chunk - 1``) by the inherited
``_set_encoder_att_context``. Both heads therefore see the identical acoustic
window on every step, which is the whole point of training them together.

WHY THE REAL ``EncDecCHATBPEModel`` IS MOUNTED rather than its loss methods
copied onto this class: ``_banded_loss`` reaches a dozen sibling helpers and
~17 attributes set up by CHAT's own ``__init__`` (band geometry, the alignment
config, the tokenizer's word-prefix convention, the mismatch counters). Copying
them would fork that setup and let the two drift apart silently -- exactly the
failure the joint model exists to rule out. Mounting the real class keeps CHAT's
semantics authoritative; we simply never call its ``forward``.
"""

from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.models.chat_bpe_models import EncDecCHATBPEModel
from nemo.collections.speechlm2.data.script_dataset import ScriptBatch, ScriptSTTDataset
from nemo.collections.speechlm2.models.script_model import ScriptSTTModel
from nemo.utils import logging


class JointChatScriptModel(ScriptSTTModel):
    """:class:`ScriptSTTModel` with a CHAT transducer head on the same encoder."""

    def __init__(
        self,
        cfg: dict,
        forced_aligner=None,
        data_cfg=None,
        val_data_cfg=None,
        dataset_cls=ScriptSTTDataset,
    ) -> None:
        super().__init__(
            cfg,
            forced_aligner=forced_aligner,
            data_cfg=data_cfg,
            val_data_cfg=val_data_cfg,
            dataset_cls=dataset_cls,
        )

        chat_cfg = cfg.get("chat", None) if hasattr(cfg, "get") else None
        if chat_cfg is None:
            raise ValueError(
                "JointChatScriptModel requires a `model.chat` config block holding the CHAT "
                "head (tokenizer / decoder / joint / loss_type / forced_alignment)."
            )
        if not isinstance(chat_cfg, DictConfig):
            chat_cfg = OmegaConf.create(chat_cfg)

        self.chat_loss_weight = float(cfg.get("chat_loss_weight", 1.0))

        # ModelPT only builds dataloaders when these are present; the joint model
        # drives data from the SCRIPT side, so refuse a config that would make the
        # CHAT submodule open its own (silently doubling the input pipeline).
        for key in ("train_ds", "validation_ds", "test_ds"):
            if chat_cfg.get(key, None) is not None:
                raise ValueError(
                    f"model.chat.{key} must be unset: the CHAT head consumes the SCRIPT "
                    "dataloader's batches, and a second data config would build a second, "
                    "differently-shuffled input pipeline."
                )

        self.chat = EncDecCHATBPEModel(chat_cfg, trainer=None)

        # Drop the CHAT submodule's own audio front-end. It is a byte-identical
        # second copy of the shared one; left in place it would be saved in every
        # checkpoint, counted by the optimizer, and -- because it never runs --
        # flagged by DDP as an unused parameter. The shared `self.perception`
        # supplies `encoded` instead (see `_chat_loss`).
        self.chat.encoder = None
        self.chat.preprocessor = None
        self.chat.spec_augmentation = None

        # Capture the SHARED encoder's output as it is produced. A forward hook
        # rather than a plumbed return value because `self.perception(...)` is
        # reached from several call paths in the SCRIPT step (indexed fill,
        # cumsum fill, grouped-branch recompute); hooking the encoder itself
        # covers all of them and cannot fall out of sync with a future one.
        self._cap_enc: Optional[torch.Tensor] = None
        self._cap_enc_len: Optional[torch.Tensor] = None
        self.perception.encoder.register_forward_hook(self._capture_encoder_output)

    def _capture_encoder_output(self, _module, _args, output) -> None:
        """Stash the Conformer's ``(B, D, T)`` output and its lengths."""
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            self._cap_enc, self._cap_enc_len = output[0], output[1]

    def _chat_loss(self, batch: ScriptBatch) -> Optional[torch.Tensor]:
        """Banded transducer loss on the frames the SCRIPT step just encoded."""
        enc, enc_len = self._cap_enc, self._cap_enc_len
        if enc is None or enc_len is None:
            raise RuntimeError(
                "No encoder output was captured for this step. The CHAT head scores the "
                "SAME frames the SCRIPT head consumed, so a missing capture means the two "
                "losses would disagree about the audio rather than merely about the model."
            )
        if batch.cuts is None:
            raise ValueError("batch.cuts is required: the CHAT banded loss reads word alignments from them.")

        # Same latency as the SCRIPT branch this step. `_set_encoder_att_context`
        # has already pinned the encoder's look-ahead to `chunk - 1`; this pins
        # the transducer's chunking to the identical grid.
        self.chat.joint.chunk_size = int(batch.chunk_size)
        return self.chat._banded_loss(enc, enc_len, batch.cuts)

    def _training_step_inner(self, batch: ScriptBatch, batch_idx: int):
        # Clear first: a stale tensor from the previous step would be graph-free
        # (backward already ran) and would fail with a confusing autograd error
        # rather than pointing at the missing capture.
        self._cap_enc = self._cap_enc_len = None

        out = super()._training_step_inner(batch, batch_idx)
        script_loss = out["loss"]

        chat_loss = self._chat_loss(batch)
        loss = script_loss + self.chat_loss_weight * chat_loss

        self.log_dict(
            {
                "train_loss_script": script_loss.detach(),
                "train_loss_chat": chat_loss.detach(),
                "train_loss_joint": loss.detach(),
            },
            on_step=True,
        )
        out["loss"] = loss
        return out
