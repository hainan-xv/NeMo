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

"""Bidirectional TDT transducer: one shared encoder, two (prediction-net, joint) pairs.

A standard parakeet-style TDT (:class:`EncDecRNNTBPEModel`) is extended with a second
prediction-network + joint pair that is trained in the *backward* (right-to-left) direction. The
encoder is shared and runs once per step; the backward branch consumes the **time-reversed** encoder
output aligned to the **reversed** label sequence. Reversing *both* time and labels keeps the
transducer's monotonic alignment valid (reversed-frame 0 = end of audio, reversed-label 0 = last
token), so the backward branch is a genuine right-to-left acoustic model rather than a broken
forward-time / reversed-label alignment.

The total training loss is::

    L = L_fwd + backward_loss_weight * L_bwd

Both branches use the same TDT loss object (it is stateless w.r.t. direction). Only the training
objective is implemented here; inference-time combination of the two directions is intentionally left
for later (the monitored ``val_wer`` is still computed from the forward branch).
"""

from typing import Optional, Tuple

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.mixins import AccessMixin
from nemo.utils import logging


class EncDecBidirectionalTDTBPEModel(EncDecRNNTBPEModel):
    """TDT transducer trained jointly left-to-right and right-to-left over a shared encoder."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        if self.loss_type != 'rnnt':
            raise ValueError(
                f"EncDecBidirectionalTDTBPEModel only supports loss_type='rnnt' (TDT/RNN-T), got "
                f"'{self.loss_type}'."
            )
        if getattr(self.joint, 'fuse_loss_wer', False):
            raise ValueError(
                "EncDecBidirectionalTDTBPEModel requires model.joint.fuse_loss_wer=false (the forward "
                "and backward losses are computed explicitly outside the fused joint path)."
            )

        # Weight on the backward (right-to-left) transducer loss added to the forward loss.
        self.backward_loss_weight = float(self.cfg.get('backward_loss_weight', 1.0))

        # HAINAN stochastic predictor masking probability (Xu et al., "HAINAN: ... Hybrid-Autoregressive
        # Inference Transducer"). During training, the predictor (prediction-net) output is zeroed at a
        # random subset of text indices with this probability, so the joint also learns to predict from
        # the encoder alone -> enables non-/semi-autoregressive inference. 0.0 == plain bidirectional TDT
        # (unchanged behavior); 0.5 == HAINAN. Applied identically to BOTH the forward and backward
        # branches (training only; never during validation/decoding).
        self.hainan_predictor_mask_prob = float(self.cfg.get('hainan_predictor_mask_prob', 0.0))
        if not 0.0 <= self.hainan_predictor_mask_prob <= 1.0:
            raise ValueError(
                f"`hainan_predictor_mask_prob` must be in [0, 1], got {self.hainan_predictor_mask_prob}."
            )

        # Second (prediction-net, joint) pair, built from the SAME sub-configs as the forward pair so
        # the architecture (incl. TDT durations via joint.num_extra_outputs) is identical. These are
        # fresh modules: by default the entrypoint warm-starts only the forward pair, leaving these
        # trained from scratch.
        self.decoder_bwd = self.from_config_dict(self.cfg.decoder)
        self.joint_bwd = self.from_config_dict(self.cfg.joint)

        # Decoding object for the backward branch (greedy/beam over decoder_bwd + joint_bwd), used only
        # to report a backward WER. self.wer / self.decoding (built by the base class) cover the
        # forward branch. The backward branch decodes the time-reversed encoder output, so its
        # predicted token sequence is un-reversed before scoring (see _backward_wer_counts), making the
        # reported backward WER directly comparable to the forward WER.
        self.decoding_bwd = RNNTBPEDecoding(
            decoding_cfg=self.cfg.decoding,
            decoder=self.decoder_bwd,
            joint=self.joint_bwd,
            tokenizer=self.tokenizer,
        )

    # -- helpers ---------------------------------------------------------------------------------

    @staticmethod
    def _reverse_time(encoded: torch.Tensor, encoded_len: torch.Tensor) -> torch.Tensor:
        """Reverse the encoder output along time, per-sample, respecting valid lengths.

        Args:
            encoded: ``[B, D, T]`` encoder output (channel-first, as fed to the joint).
            encoded_len: ``[B]`` number of valid frames per sample.

        Returns:
            ``[B, D, T]`` with frames ``0..L-1`` reversed and padding frames ``L..T-1`` left in place.
        """
        B, D, T = encoded.shape
        device = encoded.device
        idx = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        lengths = encoded_len.to(device).long().unsqueeze(1)  # [B, 1]
        src = torch.where(idx < lengths, lengths - 1 - idx, idx).clamp_(min=0, max=max(T - 1, 0))  # [B, T]
        src = src.unsqueeze(1).expand(B, D, T)
        return torch.gather(encoded, 2, src)

    @staticmethod
    def _reverse_labels(transcript: torch.Tensor, transcript_len: torch.Tensor) -> torch.Tensor:
        """Reverse the label sequence, per-sample, respecting valid lengths.

        Args:
            transcript: ``[B, U]`` token ids.
            transcript_len: ``[B]`` number of valid tokens per sample.

        Returns:
            ``[B, U]`` with tokens ``0..L-1`` reversed and padding tokens left in place.
        """
        B, U = transcript.shape
        if U == 0:
            return transcript
        device = transcript.device
        idx = torch.arange(U, device=device).unsqueeze(0)  # [1, U]
        lengths = transcript_len.to(device).long().unsqueeze(1)  # [B, 1]
        src = torch.where(idx < lengths, lengths - 1 - idx, idx).clamp_(min=0, max=U - 1)  # [B, U]
        return torch.gather(transcript, 1, src)

    def _transducer_loss(
        self,
        decoder_module,
        joint_module,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
        transcript: torch.Tensor,
        transcript_len: torch.Tensor,
    ) -> torch.Tensor:
        """Standard (non-fused) TDT loss for one (prediction-net, joint) pair.

        With ``hainan_predictor_mask_prob > 0`` the predictor output is stochastically masked (HAINAN)
        during training before the joint, so the joint also learns to predict from the encoder alone.
        """
        dec, target_length, _ = decoder_module(targets=transcript, target_length=transcript_len)
        dec = self._maybe_mask_predictor(dec)
        joint = joint_module(encoder_outputs=encoded, decoder_outputs=dec, encoder_lengths=encoded_len)
        return self.loss(
            log_probs=joint,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=target_length,
        )

    def _maybe_mask_predictor(self, dec: torch.Tensor) -> torch.Tensor:
        """HAINAN stochastic predictor masking (training only).

        ``dec`` is the prediction-network output ``[B, D, U+1]``. With probability
        ``hainan_predictor_mask_prob``, zero the predictor vector at each text index (per [batch,
        text-index], broadcast over the feature dim), matching ``joint(E_t, D_u * 0)`` from the paper.
        No-op outside training or when the probability is 0 (plain bidirectional TDT).
        """
        if not self.training or self.hainan_predictor_mask_prob <= 0.0:
            return dec
        keep = torch.rand(dec.size(0), 1, dec.size(2), device=dec.device) >= self.hainan_predictor_mask_prob
        return dec * keep.to(dec.dtype)

    def _forward_encoder(self, batch):
        signal, signal_len, transcript, transcript_len = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        return encoded, encoded_len, transcript, transcript_len

    def _backward_wer_counts(
        self,
        enc_rev: torch.Tensor,
        encoded_len: torch.Tensor,
        transcript: torch.Tensor,
        transcript_len: torch.Tensor,
    ) -> Tuple[int, int]:
        """Greedy-decode the backward branch and return (edit_distance, ref_words) vs forward refs.

        The backward branch decodes the time-reversed encoder output, so its predicted token sequence
        is in reverse (R2L) order; we un-reverse it before detokenizing so the resulting text is in
        normal forward order and directly comparable to the forward references.
        """
        hyps = self.decoding_bwd.rnnt_decoder_predictions_tensor(
            encoder_output=enc_rev, encoded_lengths=encoded_len, return_hypotheses=True
        )
        if isinstance(hyps, tuple):
            hyps = hyps[0]

        hyp_texts = []
        for hyp in hyps:
            ids = hyp.y_sequence
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            ids = list(ids)[::-1]  # un-reverse R2L prediction -> forward token order
            hyp_texts.append(self.decoding_bwd.decode_ids_to_str(ids))

        references = self._references_from_targets(transcript, transcript_len)
        return self._wer_counts(hyp_texts, references)

    # -- non-autoregressive (NAR) inference ------------------------------------------------------

    @staticmethod
    def _nar_joint_logits(joint, enc_btd):
        """Full masked-predictor joint logits ``[B, T, V]`` for one branch (self-consistent).

        Runs the branch's OWN enc projection + a single zero/masked predictor step through its OWN
        output head, so the returned token AND duration logits both live in this branch's joint-hidden
        basis. ``enc_btd`` is ``[B, T, D]``.
        """
        f = joint.project_encoder(enc_btd)  # [B, T, H]
        g0 = joint.project_prednet(
            torch.zeros(enc_btd.size(0), 1, joint.pred.in_features, device=enc_btd.device, dtype=enc_btd.dtype)
        )  # [B, 1, H]  (== projected zero/masked predictor)
        return joint.joint_after_projection(f, g0)[:, :, 0, :]  # [B, T, V]

    @torch.no_grad()
    def nar_greedy_decode(
        self,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
        backward: bool = False,
        swap_token_embed: bool = False,
        ensemble_weight: float = None,
    ):
        """Non-autoregressive greedy decode -- no prediction-network / decoder is run.

        HAINAN trains the joint to also predict from the encoder alone (the prediction-net output is
        stochastically zero-masked). At inference we exploit that: feed the joint a SINGLE masked
        (all-zero) predictor step and compute the token + duration distribution for **every encoder
        frame in parallel** with one joint call. A pure-Python loop (no neural-net compute) then walks
        the frames, using each frame's predicted TDT duration to jump, emitting the per-frame argmax
        token when it is not blank.

        Args:
            encoded:     ``[B, D, T]`` forward encoder output (channel-first), from ``self.forward``.
            encoded_len: ``[B]`` valid frame counts.
            backward:    if True, decode with the backward (R2L) branch -- the encoder output is
                         time-reversed and the R2L token stream is un-reversed before detokenizing, so
                         the returned text is in normal forward order (comparable to forward NAR).
            swap_token_embed: if True, take the per-frame TOKEN logits from the OTHER branch's joiner
                         (forward tokens for a backward run, and vice versa) while keeping the running
                         branch's own DURATION logits. Each branch's joiner is run in full and
                         self-consistently (its own enc projection + own output head), then the token
                         block is spliced from the other branch -- so there is no joint-hidden basis
                         mismatch. Since the two branches agree ~99% per-frame on tokens, this should
                         decode about as well as the running branch alone. Ignored if ``ensemble_weight``
                         is set.
            ensemble_weight: if not None, log-linearly (geometrically) pool BOTH branches' NAR
                         distributions: ``logP = (1-w)*logP_own + w*logP_other``, applied separately to
                         the token block and the duration block (each a proper softmax). ``w=0`` == own
                         branch only, ``w=1`` == other branch only, ``w=0.5`` == equal 2-branch ensemble.
                         Both joiners are run fully self-consistently (no basis mismatch).

        Returns:
            list[str] of length B (forward-order transcripts).
        """
        joint = self.joint_bwd if backward else self.joint
        decoding = self.decoding_bwd if backward else self.decoding
        durations = list(decoding.durations) if decoding.durations else [0, 1]
        if backward:
            encoded = self._reverse_time(encoded, encoded_len)

        # ---- parallel joint over all frames with a masked (zero) predictor ----
        enc_btd = encoded.transpose(1, 2)  # [B, D, T] -> [B, T, D]
        num_dur = len(durations)
        logits = self._nar_joint_logits(joint, enc_btd)  # [B, T, V]  (own branch, self-consistent)
        if ensemble_weight is not None:
            other = self.joint if backward else self.joint_bwd
            logits_other = self._nar_joint_logits(other, enc_btd)  # [B, T, V] (other branch, self-consistent)
            w = float(ensemble_weight)
            # log-linear pooling, per TDT block (token + blank, and durations) normalized separately
            tok = (1.0 - w) * torch.log_softmax(logits[..., :-num_dur], dim=-1) + w * torch.log_softmax(
                logits_other[..., :-num_dur], dim=-1
            )
            dur = (1.0 - w) * torch.log_softmax(logits[..., -num_dur:], dim=-1) + w * torch.log_softmax(
                logits_other[..., -num_dur:], dim=-1
            )
            logits = torch.cat([tok, dur], dim=-1)
        elif swap_token_embed:
            other = self.joint if backward else self.joint_bwd
            logits_other = self._nar_joint_logits(other, enc_btd)  # [B, T, V]  (other branch, self-consistent)
            # token block (real tokens + blank) from the OTHER branch; duration block from the OWN branch.
            # Both blocks come from a fully self-consistent joiner (own enc_proj+head), so no basis mismatch.
            logits = torch.cat([logits_other[..., :-num_dur], logits[..., -num_dur:]], dim=-1)

        dur_values = torch.tensor(durations, device=logits.device)
        token_logits = logits[..., :-num_dur]  # [B, T, num_tokens(+blank)]
        dur_logits = logits[..., -num_dur:]  # [B, T, num_durations]
        blank_id = token_logits.size(-1) - 1  # blank is the last token class
        tokens = token_logits.argmax(dim=-1)  # [B, T]
        jumps = dur_values[dur_logits.argmax(dim=-1)]  # [B, T]

        # ---- pure-Python extraction loop (no neural-net compute inside) ----
        tokens_l = tokens.tolist()
        jumps_l = jumps.tolist()
        lens_l = encoded_len.tolist()
        results = []
        for b in range(len(lens_l)):
            T = int(lens_l[b])
            tok_b, jmp_b = tokens_l[b], jumps_l[b]
            ids, t = [], 0
            while t < T:
                k = tok_b[t]
                if k != blank_id:
                    ids.append(k)
                d = jmp_b[t]
                t += d if d > 0 else 1  # always progress (blank-with-dur-0 -> +1; matches TDT rule)
            if backward:
                ids = ids[::-1]  # un-reverse R2L prediction -> forward token order
            results.append(decoding.decode_ids_to_str(ids))
        return results

    # -- training / validation -------------------------------------------------------------------

    def training_step(self, batch, batch_nb):
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        encoded, encoded_len, transcript, transcript_len = self._forward_encoder(batch)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # Backward branch operates on the time-reversed encoder output + reversed labels.
        enc_rev = self._reverse_time(encoded, encoded_len)
        y_rev = self._reverse_labels(transcript, transcript_len)

        fwd_loss = self._transducer_loss(self.decoder, self.joint, encoded, encoded_len, transcript, transcript_len)
        bwd_loss = self._transducer_loss(
            self.decoder_bwd, self.joint_bwd, enc_rev, encoded_len, y_rev, transcript_len
        )
        loss_value = fwd_loss + self.backward_loss_weight * bwd_loss
        loss_value = self.add_auxiliary_losses(loss_value)

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        tensorboard_logs = {
            'train_loss': loss_value,
            'train_fwd_loss': fwd_loss.detach(),
            'train_bwd_loss': bwd_loss.detach(),
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }

        if (sample_id + 1) % log_every_n_steps == 0:
            # Forward-branch WER (self.decoding wraps self.decoder/self.joint).
            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            _, scores, words = self.wer.compute()
            self.wer.reset()
            tensorboard_logs['training_batch_wer'] = scores.float() / words

            # Backward-branch WER (un-reversed predictions -> comparable to forward).
            scores_b, words_b = self._backward_wer_counts(enc_rev, encoded_len, transcript, transcript_len)
            tensorboard_logs['training_batch_wer_bwd'] = torch.tensor(
                scores_b / max(words_b, 1), dtype=torch.float32, device=encoded.device
            )

        self.log_dict(tensorboard_logs)

        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        encoded, encoded_len, transcript, transcript_len = self._forward_encoder(batch)

        enc_rev = self._reverse_time(encoded, encoded_len)
        y_rev = self._reverse_labels(transcript, transcript_len)

        tensorboard_logs = {}
        if self.compute_eval_loss:
            fwd_loss = self._transducer_loss(
                self.decoder, self.joint, encoded, encoded_len, transcript, transcript_len
            )
            bwd_loss = self._transducer_loss(
                self.decoder_bwd, self.joint_bwd, enc_rev, encoded_len, y_rev, transcript_len
            )
            tensorboard_logs['val_loss'] = fwd_loss + self.backward_loss_weight * bwd_loss
            tensorboard_logs['val_fwd_loss'] = fwd_loss.detach()
            tensorboard_logs['val_bwd_loss'] = bwd_loss.detach()

        # Forward-branch WER (the monitored metric for checkpointing).
        self.wer.update(
            predictions=encoded,
            predictions_lengths=encoded_len,
            targets=transcript,
            targets_lengths=transcript_len,
        )
        wer, wer_num, wer_denom = self.wer.compute()
        self.wer.reset()
        tensorboard_logs['val_wer_num'] = wer_num
        tensorboard_logs['val_wer_denom'] = wer_denom
        tensorboard_logs['val_wer'] = wer

        # Backward-branch WER, emitted as summed num/denom so multi_validation_epoch_end micro-averages
        # it into 'val_wer_bwd' automatically (it aggregates any extra *_num/*_denom pair).
        scores_b, words_b = self._backward_wer_counts(enc_rev, encoded_len, transcript, transcript_len)
        tensorboard_logs['val_wer_bwd_num'] = torch.tensor(float(scores_b), dtype=torch.float32, device=encoded.device)
        tensorboard_logs['val_wer_bwd_denom'] = torch.tensor(
            float(words_b), dtype=torch.float32, device=encoded.device
        )

        return tensorboard_logs

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return []
