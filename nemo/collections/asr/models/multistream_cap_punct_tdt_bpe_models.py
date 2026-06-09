# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""3-stream (spelling + capitalization + punctuation) TDT BPE model.

This extends the 2-stream :class:`EncDecMultiStreamTDTBPEModel` with a third,
**word-ending punctuation** stream emitted jointly at every step (on top of the standard TDT
duration stream):

* **spelling**       - a lowercased, punctuation-stripped BPE sub-word (this stream owns blank).
* **capitalization** - a per-token casing class.
* **punctuation**    - a per-word word-ending punctuation class (attached to the word's last
  sub-word; see :mod:`multistream_cap_punct_factorization`).

Design (mirrors the 2-stream model):

* The data pipeline carries a single *product-space* id
  ``(punct*V_cap + cap)*V_spell + spell`` via :class:`MultiStreamCapPunctTokenizer`.
* The **prediction network is standard** and consumes only the spelling id; capitalization and
  punctuation are side outputs predicted by the joint and not fed back.
* The **joint** output is laid out (sum space) as
  ``[ punct(V_punct) | cap(V_cap) | spell(V_spell) | blank | durations ]`` so ``MultistreamTDTLoss``
  sees the canonical layout (contiguous label streams, shared blank as the last label index,
  durations last).
"""

import copy
from typing import Optional

import torch
from omegaconf import DictConfig, ListConfig, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.losses.rnnt_pytorch import MultistreamTDTLoss
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.submodules.multistream_cap_punct_tdt_greedy_decoding import (
    GreedyBatchedMultiStreamCapPunctTDTInfer,
    GreedyMultiStreamCapPunctTDTInfer,
)
from nemo.collections.asr.parts.utils.multistream_cap_punct_factorization import (
    DEFAULT_PUNCT_MARKS,
    cap_punct_dividers,
)
from nemo.collections.asr.parts.utils.multistream_cap_punct_tokenizer import MultiStreamCapPunctTokenizer
from nemo.collections.asr.parts.utils.multistream_factorization import NUM_CAP
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

try:
    from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs

    HAVE_DALI = True
except (ImportError, ModuleNotFoundError):
    HAVE_DALI = False


class EncDecMultiStreamCapPunctTDTBPEModel(EncDecRNNTBPEModel):
    """Encoder-Decoder 3-stream (spelling + capitalization + punctuation) TDT model (BPE)."""

    def _setup_tokenizer(self, tokenizer_cfg: DictConfig):
        # Build the standard BPE tokenizer first, then wrap it so the data pipeline emits
        # product-space (spelling x capitalization x punctuation) ids.
        super()._setup_tokenizer(tokenizer_cfg)
        # `_setup_tokenizer` runs inside super().__init__ before self._cfg exists, so read the
        # multistream settings stashed at the top of __init__.
        num_cap = getattr(self, '_ms_num_cap', NUM_CAP)
        punct_marks = getattr(self, '_ms_punct_marks', list(DEFAULT_PUNCT_MARKS))
        self.tokenizer = MultiStreamCapPunctTokenizer(self.tokenizer, num_cap=num_cap, punct_marks=punct_marks)

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Stash multistream settings before super() (which calls _setup_tokenizer).
        self._ms_num_cap = int(cfg.model_defaults.get('num_cap', NUM_CAP))
        self._ms_punct_marks = list(cfg.model_defaults.get('punct_marks', list(DEFAULT_PUNCT_MARKS)))
        self._ms_durations = list(cfg.model_defaults.tdt_durations)

        # super() builds: encoder, prediction net (vocab = V_spell, correct),
        # joint (num_classes = V_spell, to be rebuilt), and a placeholder loss.
        super().__init__(cfg=cfg, trainer=trainer)

        self.num_spell = self.tokenizer.num_spell
        self.num_cap = self.tokenizer.num_cap
        self.num_punct = self.tokenizer.num_punct
        self.punct_marks = self.tokenizer.punct_marks
        self.durations = self._ms_durations
        n_dur = len(self.durations)

        self.dividers, self.ms_blank = cap_punct_dividers(self.num_spell, self.num_cap, self.num_punct)

        # ----- rebuild the joint with the sum-space layout -----
        # [ punct(V_punct) | cap(V_cap) | spell(V_spell) | blank | durations(n_dur) ]
        joint_cfg = copy.deepcopy(self.joint.to_config_dict())
        spell_vocab = [self.tokenizer.base.ids_to_tokens([i])[0] for i in range(self.num_spell)]
        cap_vocab = [f'<cap{i}>' for i in range(self.num_cap)]
        punct_vocab = [f'<punct{i}>' for i in range(self.num_punct)]
        joint_cfg['num_classes'] = self.num_punct + self.num_cap + self.num_spell
        joint_cfg['num_extra_outputs'] = n_dur
        joint_cfg['vocabulary'] = ListConfig(punct_vocab + cap_vocab + spell_vocab)
        del self.joint
        self.joint = EncDecMultiStreamCapPunctTDTBPEModel.from_config_dict(joint_cfg)
        self.joint._fuse_loss_wer = False  # we drive loss explicitly

        # ----- multistream TDT loss -----
        del self.loss
        self.loss = MultistreamTDTLoss(
            blank=self.ms_blank,
            durations=self.durations,
            dividers=self.dividers,
            reduction=self.cfg.get("rnnt_reduction", "mean_batch"),
            sigma=float(self.cfg.get("model_defaults", {}).get("tdt_sigma", 0.0)),
        )

        # keep cfg in sync
        with open_dict(self.cfg.joint):
            self.cfg.joint = joint_cfg

        self.compute_eval_loss = self.cfg.get('compute_eval_loss', True)

        # ----- greedy decoder for train/val WER -----
        # Default to the batched (label-looping) decoder; set `batched_decoding: false` to use the
        # slower per-utterance greedy decoder.
        self.max_symbols_per_step = int(self.cfg.get('max_symbols_per_step', 10))
        self.batched_decoding = bool(self.cfg.get('batched_decoding', True))
        use_cuda_graph_decoder = bool(self.cfg.get('greedy_use_cuda_graph_decoder', False))
        if self.batched_decoding:
            self.ms_greedy = GreedyBatchedMultiStreamCapPunctTDTInfer(
                decoder_model=self.decoder,
                joint_model=self.joint,
                blank_index=self.ms_blank,
                durations=self.durations,
                num_punct=self.num_punct,
                num_cap=self.num_cap,
                num_spell=self.num_spell,
                max_symbols_per_step=self.max_symbols_per_step,
                use_cuda_graph_decoder=use_cuda_graph_decoder,
            )
        else:
            self.ms_greedy = GreedyMultiStreamCapPunctTDTInfer(
                decoder_model=self.decoder,
                joint_model=self.joint,
                blank_index=self.ms_blank,
                durations=self.durations,
                num_punct=self.num_punct,
                num_cap=self.num_cap,
                num_spell=self.num_spell,
                max_symbols_per_step=self.max_symbols_per_step,
            )
        self.use_cer = self.cfg.get('use_cer', False)

        logging.info(
            "Initialized EncDecMultiStreamCapPunctTDTBPEModel: "
            f"V_spell={self.num_spell}, V_cap={self.num_cap}, V_punct={self.num_punct}, "
            f"punct_marks={self.punct_marks}, durations={self.durations}, "
            f"joint dim={self.joint.num_classes_with_blank} (blank={self.ms_blank}), dividers={self.dividers}, "
            f"decoding={'batched' if self.batched_decoding else 'sequential'}"
        )

    # ------------------------------------------------------------------ #
    # factorization helpers
    # ------------------------------------------------------------------ #
    def _split_combined(self, transcript: torch.Tensor):
        """[B, U] product ids -> (spell_ids, cap_ids, punct_ids), each [B, U]."""
        spell = transcript % self.num_spell
        rest = transcript // self.num_spell
        cap = rest % self.num_cap
        punct = rest // self.num_cap
        return spell, cap, punct

    def _factorized_targets(self, spell: torch.Tensor, cap: torch.Tensor, punct: torch.Tensor) -> torch.Tensor:
        """-> [B, U, 3] absolute indices into the joint label part: [punct, cap+V_punct, spell+V_punct+V_cap]."""
        return torch.stack(
            [punct, cap + self.num_punct, spell + self.num_punct + self.num_cap], dim=-1
        )

    # ------------------------------------------------------------------ #
    # decoding -> text (for WER)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _decode_hyp_texts(self, encoded: torch.Tensor, encoded_len: torch.Tensor):
        """Greedy-decode an encoder batch into a list of cased, punctuated hypothesis strings."""
        hypotheses = self.ms_greedy(encoder_output=encoded, encoded_lengths=encoded_len)[0]
        texts = []
        for hyp in hypotheses:
            seq = hyp.y_sequence
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            texts.append(self.tokenizer.ids_to_text([int(x) for x in seq]))
        return texts

    def _ref_texts(self, transcript: torch.Tensor, transcript_len: torch.Tensor):
        """Convert (padded) product-id targets into cased, punctuated reference strings."""
        refs = []
        for b in range(transcript.size(0)):
            n = int(transcript_len[b])
            seq = transcript[b, :n].tolist()
            refs.append(self.tokenizer.ids_to_text([int(x) for x in seq]))
        return refs

    # ------------------------------------------------------------------ #
    # PTL steps
    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len = batch

        if HAVE_DALI and isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        spell, cap, punct = self._split_combined(transcript)
        # Prediction network sees the spelling stream only.
        decoder, target_length, states = self.decoder(targets=spell, target_length=transcript_len)
        joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

        targets = self._factorized_targets(spell, cap, punct)
        loss_value = self.loss(acts=joint, labels=targets, act_lens=encoded_len, label_lens=target_length)
        loss_value = self.add_auxiliary_losses(loss_value)

        tensorboard_logs = {
            'train_loss': loss_value,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }

        log_every_n_steps = self._trainer.log_every_n_steps if getattr(self, '_trainer', None) is not None else 1
        if (self.trainer.global_step + 1) % log_every_n_steps == 0:
            hyps = self._decode_hyp_texts(encoded, encoded_len)
            refs = self._ref_texts(transcript, transcript_len)
            wer = word_error_rate(hyps, refs, use_cer=self.use_cer)
            tensorboard_logs['training_batch_wer'] = torch.tensor(wer, device=self.device, dtype=torch.float32)

        self.log_dict(tensorboard_logs)
        return {'loss': loss_value}

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len = batch

        if HAVE_DALI and isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        tensorboard_logs = {}
        if self.compute_eval_loss:
            spell, cap, punct = self._split_combined(transcript)
            decoder, target_length, states = self.decoder(targets=spell, target_length=transcript_len)
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            targets = self._factorized_targets(spell, cap, punct)
            loss_value = self.loss(acts=joint, labels=targets, act_lens=encoded_len, label_lens=target_length)
            tensorboard_logs['val_loss'] = loss_value

        # Greedy decode for corpus-level WER (aggregated at epoch end).
        tensorboard_logs['val_hyps'] = self._decode_hyp_texts(encoded, encoded_len)
        tensorboard_logs['val_refs'] = self._ref_texts(transcript, transcript_len)
        return tensorboard_logs

    def _aggregate_epoch(self, outputs, prefix: str):
        logs = {}
        losses = [x[f'{prefix}_loss'] for x in outputs if f'{prefix}_loss' in x]
        if losses:
            logs[f'{prefix}_loss'] = torch.stack(losses).mean()
        hyps, refs = [], []
        for x in outputs:
            hyps.extend(x.get(f'{prefix}_hyps', []))
            refs.extend(x.get(f'{prefix}_refs', []))
        if refs:
            # Must be on the model device: logged metrics are DDP all-reduced over the NCCL (GPU)
            # backend, which cannot sync CPU tensors.
            wer = word_error_rate(hyps, refs, use_cer=self.use_cer)
            logs[f'{prefix}_wer'] = torch.tensor(wer, device=self.device, dtype=torch.float32)
        return logs

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        if not outputs:
            return {}
        logs = self._aggregate_epoch(outputs, 'val')
        return {**logs, 'log': logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        if not outputs:
            return {}
        logs = self._aggregate_epoch(outputs, 'test')
        return {**logs, 'log': logs}

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return []
