# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Jointly trained GPT-LM + TDT transducer with detached log-linear fusion.

This model is a standard parakeet-style TDT (``EncDecRNNTBPEModel``) whose prediction network and
joint are the fusion-aware :class:`TDTGPTFusionDecoder` / :class:`TDTGPTFusionJoint` (selected via the
config ``_target_``). The total training loss is::

    L = L_TDT + lm_loss_weight * L_LM

where ``L_TDT`` is the usual transducer loss computed on logits whose non-blank token slots have been
log-linearly combined with the GPT LM's (detached) log-probs, and ``L_LM`` is the GPT LM's own
next-token cross-entropy. Because the LM contribution to the joint is detached, the transducer loss
never updates the LM; the LM learns only from ``L_LM``.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.mixins import AccessMixin
from nemo.utils import logging


class EncDecTDTGPTFusionModel(EncDecRNNTBPEModel):
    """TDT transducer + GPT LM trained jointly with detached log-linear fusion."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.lm_loss_weight = float(self.cfg.get('lm_loss_weight', 1.0))
        # How to reduce the LM next-token CE loss:
        #   'token_mean'  -> mean over ALL valid tokens in the batch (default; back-compat).
        #   'sample_mean' -> per-utterance per-token mean, then averaged over the batch. This matches
        #                    the TDT loss when `rnnt_reduction=mean`, so both terms are length-normalized
        #                    and weight each utterance equally.
        self.lm_loss_reduction = str(self.cfg.get('lm_loss_reduction', 'token_mean')).lower()
        if self.lm_loss_reduction not in ('token_mean', 'sample_mean'):
            raise ValueError(
                f"`lm_loss_reduction` must be 'token_mean' or 'sample_mean', got '{self.lm_loss_reduction}'."
            )
        if not hasattr(self.decoder, 'get_last_lm_logits'):
            raise ValueError(
                "EncDecTDTGPTFusionModel requires a fusion-aware decoder "
                "(nemo.collections.asr.modules.tdt_gpt_fusion.TDTGPTFusionDecoder). "
                "Set model.decoder._target_ accordingly."
            )

    def _compute_lm_loss(self, transcript: torch.Tensor, transcript_len: torch.Tensor):
        """Next-token CE loss + perplexity from the LM logits stashed by the last decoder forward."""
        lm_logits = self.decoder.get_last_lm_logits()  # [B, U+1, V]
        if lm_logits is None:
            zero = torch.zeros((), device=transcript.device)
            return zero, torch.ones((), device=transcript.device)

        batch, seq_len_p1, vocab = lm_logits.shape
        # Position u predicts token y_u given y_<u; we score positions 0..U-1 against transcript[:, :U].
        steps = min(seq_len_p1 - 1, transcript.size(1))
        logits = lm_logits[:, :steps, :].reshape(-1, vocab).float()
        targets = transcript[:, :steps].reshape(-1).long()
        ce = F.cross_entropy(logits, targets, reduction='none').view(batch, steps)

        device = transcript.device
        mask = (torch.arange(steps, device=device)[None, :] < transcript_len[:, None].to(device)).float()
        ce = ce * mask

        # Perplexity is always reported as a per-token quantity (independent of the loss reduction).
        token_mean = ce.sum() / mask.sum().clamp(min=1.0)
        ppl = torch.exp(token_mean.detach())

        if self.lm_loss_reduction == 'sample_mean':
            per_sample = ce.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # per-utterance per-token loss
            lm_loss = per_sample.mean()
        else:  # 'token_mean'
            lm_loss = token_mean
        return lm_loss, ppl

    def training_step(self, batch, batch_nb):
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len = batch

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        # Decoder forward also runs the GPT LM and stashes its logits for the LM loss below.
        decoder, target_length, _states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # Non-fused joint (1024-class vocab fits): the joint adds the detached LM term internally.
        joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder, encoder_lengths=encoded_len)
        tdt_loss = self.loss(
            log_probs=joint,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=target_length,
        )

        lm_loss, lm_ppl = self._compute_lm_loss(transcript, transcript_len)
        loss_value = tdt_loss + self.lm_loss_weight * lm_loss
        loss_value = self.add_auxiliary_losses(loss_value)

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        tensorboard_logs = {
            'train_loss': loss_value,
            'train_tdt_loss': tdt_loss.detach(),
            'train_lm_loss': lm_loss.detach(),
            'train_lm_ppl': lm_ppl,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }

        if (sample_id + 1) % log_every_n_steps == 0:
            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            _, scores, words = self.wer.compute()
            self.wer.reset()
            tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        self.log_dict(tensorboard_logs)

        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        tensorboard_logs = super().validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)

        # When compute_eval_loss is set, super() ran the decoder forward on this batch, so the LM
        # logits are stashed and aligned with the batch transcript.
        if self.compute_eval_loss:
            _signal, _signal_len, transcript, transcript_len = batch
            lm_loss, lm_ppl = self._compute_lm_loss(transcript, transcript_len)
            tensorboard_logs['val_lm_loss'] = lm_loss
            tensorboard_logs['val_lm_ppl'] = lm_ppl

        return tensorboard_logs

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return []
