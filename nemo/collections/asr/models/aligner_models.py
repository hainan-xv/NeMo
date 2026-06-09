# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Aligner-Encoder ASR models.

This implements the "Aligner-Encoder" architecture from Stooke et al.,
"Aligner-Encoders: Self-Attention Transformers Can Be Self-Transducers"
(https://arxiv.org/abs/2502.05232).

The model reuses NeMo's Conformer encoder and RNN-T prediction network, but:

* the encoder is expected to perform audio-to-text alignment internally, so the
  joint network combines the acoustic embedding ``h_i`` with the text embedding
  ``g_i`` strictly one-to-one (see :class:`AlignerJoint`);
* there is no blank token -- the vocabulary is the real tokens plus a single EOS
  token -- and training uses a frame-wise cross-entropy loss
  (:class:`AlignerCrossEntropyLoss`);
* decoding emits one token per encoder frame until EOS (autoregressive ``ar``
  variant) or classifies each frame independently (non-autoregressive ``nonar``
  variant).

Two model classes are provided:

* :class:`EncDecAlignerModel` -- character/grapheme vocabulary (``cfg.labels``).
* :class:`EncDecAlignerBPEModel` -- sub-word (BPE/WPE) tokenizer (``cfg.tokenizer``).
"""

from typing import Dict, List, Optional, Tuple, Union

import editdistance
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.losses.aligner_loss import AlignerCrossEntropyLoss
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.modules.aligner import AlignerCTCHead, AlignerJoint
from nemo.collections.asr.parts.submodules.aligner_decoding import AlignerDecoding
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.mixins import AccessMixin
from nemo.utils import logging, model_utils


class EncDecAlignerModel(EncDecRNNTModel):
    """Aligner-Encoder ASR model with a character/grapheme vocabulary."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        # Initialize the ModelPT/ASRModel machinery directly, bypassing the RNN-T
        # specific wiring in EncDecRNNTModel.__init__ (we build aligner components below).
        super(EncDecRNNTModel, self).__init__(cfg=cfg, trainer=trainer)

        self.preprocessor = EncDecAlignerModel.from_config_dict(self.cfg.preprocessor)
        self.encoder = EncDecAlignerModel.from_config_dict(self.cfg.encoder)

        # Vocabulary: real tokens come from cfg.labels; EOS is appended as one extra class.
        vocabulary = list(self.cfg.labels)
        num_real_tokens = len(vocabulary)
        self.eos_id = num_real_tokens
        num_classes = num_real_tokens + 1  # real tokens + EOS, no blank

        self.aligner_type = self.cfg.get('aligner_type', 'ar')
        if self.aligner_type not in ('ar', 'nonar'):
            raise ValueError(f"model.aligner_type must be 'ar' or 'nonar', got '{self.aligner_type}'.")

        # The prediction network embedding must cover all output ids (incl. EOS);
        # vocab_size becomes the SOS/pad slot, so set it one above the EOS index.
        with open_dict(self.cfg.decoder):
            self.cfg.decoder.vocab_size = num_classes

        with open_dict(self.cfg.joint):
            self.cfg.joint.num_classes = num_classes
            self.cfg.joint.vocabulary = ListConfig(list(vocabulary))
            self.cfg.joint.jointnet.encoder_hidden = self.cfg.model_defaults.enc_hidden
            self.cfg.joint.jointnet.pred_hidden = self.cfg.model_defaults.pred_hidden

        # Prediction network is only needed for the autoregressive variant, but it is
        # cheap and keeps checkpoints uniform, so we always build it.
        self.decoder = EncDecAlignerModel.from_config_dict(self.cfg.decoder)
        self.joint = AlignerJoint(
            jointnet=self.cfg.joint.jointnet,
            num_classes=num_classes,
            vocabulary=list(vocabulary),
            log_softmax=self.cfg.joint.get('log_softmax', None),
        )

        # Optional / required per-frame head for the non-autoregressive variant.
        self.aux_nonar_loss_weight = float(self.cfg.get('aux_nonar_loss_weight', 0.0))
        self.ctc_head = None
        if self.aligner_type == 'nonar' or self.aux_nonar_loss_weight > 0:
            head_cfg = self.cfg.get('ctc_head', {}) or {}
            self.ctc_head = AlignerCTCHead(
                feat_in=self.cfg.model_defaults.enc_hidden,
                num_classes=num_classes,
                hidden=head_cfg.get('hidden', None),
                activation=head_cfg.get('activation', 'relu'),
                dropout=head_cfg.get('dropout', 0.0),
            )

        self.loss = AlignerCrossEntropyLoss(
            num_classes=num_classes,
            label_smoothing=self.cfg.get('label_smoothing', 0.1),
        )

        if hasattr(self.cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecAlignerModel.from_config_dict(self.cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.decoding = AlignerDecoding(
            decoding_cfg=self.cfg.get('decoding', None),
            decoder=self.decoder,
            joint=self.joint,
            eos_id=self.eos_id,
            vocabulary=list(vocabulary),
            tokenizer=getattr(self, 'tokenizer', None),
            ctc_head=self.ctc_head,
        )

        self.compute_eval_loss = self.cfg.get('compute_eval_loss', True)

        # Reuse the RNN-T optimization helpers (variational noise / grad normalization).
        self.setup_optim_normalization()
        self.setup_optimization_flags()
        self.setup_adapters()

    # ------------------------------------------------------------------ #
    # Target preparation
    # ------------------------------------------------------------------ #
    def _append_eos(self, transcript: torch.Tensor, transcript_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append the EOS token to each (length-truncated) transcript.

        Returns ``(targets, target_lengths)`` where ``targets`` has shape
        ``(B, U0 + 1)`` and includes one EOS token per sample, and
        ``target_lengths == transcript_len + 1``.
        """
        targets = F.pad(transcript, (0, 1), value=0)
        batch_idx = torch.arange(transcript.size(0), device=transcript.device)
        targets[batch_idx, transcript_len.long()] = self.eos_id
        return targets, transcript_len + 1

    # ------------------------------------------------------------------ #
    # Loss computation (shared by training and validation)
    # ------------------------------------------------------------------ #
    def _aligner_loss(
        self,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
        transcript: torch.Tensor,
        transcript_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        targets_eos, target_eos_len = self._append_eos(transcript, transcript_len)
        logs: Dict[str, torch.Tensor] = {}
        total_loss = encoded.new_zeros(())

        if self.aligner_type == 'ar' or self.aux_nonar_loss_weight > 0:
            # Teacher-forced prediction network (SOS is prepended internally).
            decoder_outputs, _, _ = self.decoder(targets=transcript, target_length=transcript_len)
            ar_logits = self.joint(encoder_outputs=encoded, decoder_outputs=decoder_outputs)
            ar_loss = self.loss(log_probs=ar_logits, targets=targets_eos, target_lengths=target_eos_len)
            logs['ar_loss'] = ar_loss.detach()
            if self.aligner_type == 'ar':
                total_loss = total_loss + ar_loss

        if self.ctc_head is not None:
            nonar_logits = self.ctc_head(encoder_output=encoded)
            nonar_loss = self.loss(log_probs=nonar_logits, targets=targets_eos, target_lengths=target_eos_len)
            logs['nonar_loss'] = nonar_loss.detach()
            if self.aligner_type == 'nonar':
                total_loss = total_loss + nonar_loss
            elif self.aux_nonar_loss_weight > 0:
                total_loss = total_loss + self.aux_nonar_loss_weight * nonar_loss

        return total_loss, logs

    # ------------------------------------------------------------------ #
    # WER helpers
    # ------------------------------------------------------------------ #
    def _references_from_targets(self, transcript: torch.Tensor, transcript_len: torch.Tensor) -> List[str]:
        transcript = transcript.long().cpu()
        transcript_len = transcript_len.long().cpu()
        refs = []
        for b in range(transcript.size(0)):
            ids = transcript[b, : int(transcript_len[b].item())].tolist()
            refs.append(self.decoding.decode_ids_to_str(ids))
        return refs

    @staticmethod
    def _wer_counts(hypotheses: List[str], references: List[str]) -> Tuple[int, int]:
        scores = 0
        words = 0
        for hyp, ref in zip(hypotheses, references):
            hyp_words = hyp.split()
            ref_words = ref.split()
            scores += editdistance.eval(hyp_words, ref_words)
            words += len(ref_words)
        return scores, words

    # ------------------------------------------------------------------ #
    # PTL steps
    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_nb):
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len = batch

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        loss_value, extra_logs = self._aligner_loss(encoded, encoded_len, transcript, transcript_len)
        loss_value = self.add_auxiliary_losses(loss_value)

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        tensorboard_logs = {
            'train_loss': loss_value,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }
        tensorboard_logs.update({f'train_{k}': v for k, v in extra_logs.items()})

        self.log_dict(tensorboard_logs)
        return {'loss': loss_value}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, sample_id = batch

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        texts, _ = self.decoding.decode_encoder_output(encoded, encoded_len)

        if isinstance(sample_id, torch.Tensor):
            sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, texts))

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len = batch

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        tensorboard_logs = {}

        if self.compute_eval_loss:
            loss_value, _ = self._aligner_loss(encoded, encoded_len, transcript, transcript_len)
            tensorboard_logs['val_loss'] = loss_value

        hypotheses, _ = self.decoding.decode_encoder_output(encoded, encoded_len)
        references = self._references_from_targets(transcript, transcript_len)
        scores, words = self._wer_counts(hypotheses, references)

        tensorboard_logs['val_wer_num'] = torch.tensor(scores, dtype=torch.float32, device=encoded.device)
        tensorboard_logs['val_wer_denom'] = torch.tensor(words, dtype=torch.float32, device=encoded.device)
        tensorboard_logs['val_wer'] = torch.tensor(
            scores / max(words, 1), dtype=torch.float32, device=encoded.device
        )

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))
        return tensorboard_logs

    # ------------------------------------------------------------------ #
    # Transcription
    # ------------------------------------------------------------------ #
    def _transcribe_output_processing(self, outputs, trcfg) -> Union[List['Hypothesis'], List[List['Hypothesis']]]:
        encoded = outputs.pop('encoded')
        encoded_len = outputs.pop('encoded_len')

        texts, token_ids = self.decoding.decode_encoder_output(encoded, encoded_len)
        del encoded, encoded_len

        hypotheses = []
        for text, ids in zip(texts, token_ids):
            hypotheses.append(Hypothesis(score=0.0, y_sequence=torch.tensor(ids, dtype=torch.long), text=text))
        return hypotheses

    # ------------------------------------------------------------------ #
    # Vocabulary management
    # ------------------------------------------------------------------ #
    def change_vocabulary(self, new_vocabulary: List[str], decoding_cfg: Optional[DictConfig] = None):
        """Rebuild the prediction net, joint, loss and decoding for a new vocabulary."""
        if self.joint.vocabulary == new_vocabulary:
            logging.warning(f"Old vocabulary == new vocabulary ({new_vocabulary}). Not changing anything.")
            return

        num_real_tokens = len(new_vocabulary)
        self.eos_id = num_real_tokens
        num_classes = num_real_tokens + 1

        new_decoder_config = OmegaConf.create(OmegaConf.to_container(self.cfg.decoder))
        new_decoder_config.vocab_size = num_classes
        self.decoder = EncDecAlignerModel.from_config_dict(new_decoder_config)

        new_joint_config = OmegaConf.create(OmegaConf.to_container(self.cfg.joint))
        new_joint_config.num_classes = num_classes
        new_joint_config.vocabulary = ListConfig(list(new_vocabulary))
        self.joint = AlignerJoint(
            jointnet=new_joint_config.jointnet,
            num_classes=num_classes,
            vocabulary=list(new_vocabulary),
            log_softmax=new_joint_config.get('log_softmax', None),
        )

        if self.ctc_head is not None:
            self.ctc_head = AlignerCTCHead(feat_in=self.cfg.model_defaults.enc_hidden, num_classes=num_classes)

        self.loss = AlignerCrossEntropyLoss(
            num_classes=num_classes, label_smoothing=self.cfg.get('label_smoothing', 0.1)
        )

        decoding_cfg = decoding_cfg if decoding_cfg is not None else self.cfg.get('decoding', None)
        self.decoding = AlignerDecoding(
            decoding_cfg=decoding_cfg,
            decoder=self.decoder,
            joint=self.joint,
            eos_id=self.eos_id,
            vocabulary=list(new_vocabulary),
            tokenizer=getattr(self, 'tokenizer', None),
            ctc_head=self.ctc_head,
        )

        with open_dict(self.cfg):
            self.cfg.labels = ListConfig(list(new_vocabulary))
            self.cfg.decoder = new_decoder_config
            self.cfg.joint = new_joint_config
            if decoding_cfg is not None:
                self.cfg.decoding = decoding_cfg

        logging.info(f"Changed the vocabulary to {num_real_tokens} tokens (+EOS).")

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []


class EncDecAlignerBPEModel(EncDecAlignerModel, EncDecRNNTBPEModel):
    """Aligner-Encoder ASR model with a sub-word (BPE/WPE) tokenizer."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create an Aligner-Encoder BPE model.")

        # Setup the BPE/WPE tokenizer (from ASRBPEMixin) and inject the vocabulary.
        self._setup_tokenizer(cfg.tokenizer)
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        with open_dict(cfg):
            cfg.labels = ListConfig(list(vocabulary))

        # EncDecAlignerModel.__init__ builds all aligner components from cfg.labels;
        # because self.tokenizer is already set, decoding will detokenize via BPE.
        super().__init__(cfg=cfg, trainer=trainer)

    def change_vocabulary(
        self,
        new_tokenizer_dir: Union[str, DictConfig],
        new_tokenizer_type: str,
        decoding_cfg: Optional[DictConfig] = None,
    ):
        """Rebuild components for a new sub-word tokenizer."""
        if isinstance(new_tokenizer_dir, DictConfig):
            tokenizer_cfg = OmegaConf.create(new_tokenizer_dir)
        else:
            tokenizer_cfg = OmegaConf.create({'dir': new_tokenizer_dir, 'type': new_tokenizer_type})

        self._setup_tokenizer(tokenizer_cfg)
        new_vocabulary = list(self.tokenizer.tokenizer.get_vocab())

        with open_dict(self.cfg):
            self.cfg.tokenizer = tokenizer_cfg

        # Reuse the character-model rebuild logic with the new token list.
        EncDecAlignerModel.change_vocabulary(self, new_vocabulary=new_vocabulary, decoding_cfg=decoding_cfg)

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []
