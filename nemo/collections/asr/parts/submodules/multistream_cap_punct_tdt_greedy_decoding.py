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

"""Greedy decoding for the 3-stream (spelling + capitalization + punctuation) TDT model.

The label part of the joint output is laid out as
``[ punct(num_punct) | cap(num_cap) | spell(num_spell) | blank ]`` (followed by the duration
logits). At each step we pick the spelling token (incl. blank) by argmax over the spelling
slice, and the capitalization / punctuation classes by argmax over their own slices.

The emitted token stored in the hypothesis is the *product* id
``(punct * num_cap + cap) * num_spell + spell`` (so ``tokenizer.ids_to_text`` reconstructs cased,
punctuated text), while the token fed back to the prediction network is the spelling id only.

This module provides both the per-utterance :class:`GreedyMultiStreamCapPunctTDTInfer` and the
batched :class:`GreedyBatchedMultiStreamCapPunctTDTInfer` (label-looping) decoders.
"""

from typing import List, Optional

import torch

from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyTDTInfer, pack_hypotheses
from nemo.collections.asr.parts.submodules.transducer_decoding.multistream_cap_punct_tdt_label_looping import (
    GreedyBatchedMultiStreamCapPunctTDTLabelLoopingComputer,
)
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.common.parts.rnn import label_collate
from nemo.core.classes.common import typecheck
from nemo.utils import logging


class GreedyMultiStreamCapPunctTDTInfer(GreedyTDTInfer):
    """Greedy (non-batched) decoder for spelling + capitalization + punctuation TDT."""

    def __init__(
        self,
        decoder_model,
        joint_model,
        blank_index: int,
        durations: list,
        num_punct: int,
        num_cap: int,
        num_spell: int,
        max_symbols_per_step: Optional[int] = 10,
        preserve_alignments: bool = False,
        preserve_frame_confidence: bool = False,
        include_duration: bool = False,
        include_duration_confidence: bool = False,
        confidence_method_cfg=None,
    ):
        super().__init__(
            decoder_model=decoder_model,
            joint_model=joint_model,
            blank_index=blank_index,
            durations=durations,
            max_symbols_per_step=max_symbols_per_step,
            preserve_alignments=preserve_alignments,
            preserve_frame_confidence=preserve_frame_confidence,
            include_duration=include_duration,
            include_duration_confidence=include_duration_confidence,
            confidence_method_cfg=confidence_method_cfg,
        )
        self.num_punct = num_punct
        self.num_cap = num_cap
        self.num_spell = num_spell
        # The prediction network operates on the spelling vocabulary, whose SOS/blank index is
        # num_spell (NOT the joint blank index).
        self._SOS = num_spell

    @torch.no_grad()
    def _greedy_decode(
        self, x: torch.Tensor, out_len: torch.Tensor, partial_hypotheses: Optional[rnnt_utils.Hypothesis] = None
    ):
        if partial_hypotheses is not None:
            raise NotImplementedError("`partial_hypotheses` is not supported for cap+punct TDT greedy decoding.")

        hypothesis = rnnt_utils.Hypothesis(
            score=0.0, y_sequence=[], dec_state=None, timestamp=[], token_duration=[], last_token=None
        )

        n_dur = len(self.durations)
        spell_blank = self.num_spell  # blank index within the spelling slice
        cap_start = self.num_punct
        spell_start = self.num_punct + self.num_cap

        time_idx = 0
        while time_idx < out_len:
            f = x.narrow(dim=0, start=time_idx, length=1)

            symbols_added = 0
            need_loop = True
            while need_loop and (self.max_symbols is None or symbols_added < self.max_symbols):
                if hypothesis.last_token is None and hypothesis.dec_state is None:
                    last_label = self._SOS
                else:
                    last_label = label_collate([[hypothesis.last_token]])

                g, hidden_prime = self._pred_step(last_label, hypothesis.dec_state)
                logits = self._joint_step(f, g, log_normalize=False)[0, 0, 0]
                del g

                label_logp = logits[:-n_dur]  # [punct | cap | spell | blank]
                if label_logp.dtype != torch.float32:
                    label_logp = label_logp.float()

                punct_logp = label_logp[:cap_start]
                cap_logp = label_logp[cap_start:spell_start]
                spell_logp = label_logp[spell_start:]  # [spell..., blank] (blank at index num_spell)

                v, spell_k = spell_logp.max(0)
                spell_k = spell_k.item()
                _, cap_k = cap_logp.max(0)
                cap_k = cap_k.item()
                _, punct_k = punct_logp.max(0)
                punct_k = punct_k.item()

                duration_logp = torch.log_softmax(logits[-n_dur:].float(), dim=-1)
                _, d_k = duration_logp.max(0)
                skip = self.durations[d_k.item()]

                if spell_k != spell_blank:
                    combined = (punct_k * self.num_cap + cap_k) * self.num_spell + spell_k
                    hypothesis.y_sequence.append(combined)
                    hypothesis.score += float(v)
                    hypothesis.timestamp.append(time_idx)
                    hypothesis.dec_state = hidden_prime
                    hypothesis.last_token = spell_k
                    if self.include_duration:
                        hypothesis.token_duration.append(skip)

                symbols_added += 1
                time_idx += skip
                need_loop = skip == 0

            if skip == 0:
                skip = 1

            if symbols_added == self.max_symbols:
                time_idx += 1

        hypothesis.dec_state = self.decoder.batch_select_state(hypothesis.dec_state, 0)
        return hypothesis


class GreedyBatchedMultiStreamCapPunctTDTInfer(GreedyTDTInfer):
    """Batched greedy (label-looping) decoder for spelling + capitalization + punctuation TDT.

    Decodes a whole batch in lockstep using
    :class:`GreedyBatchedMultiStreamCapPunctTDTLabelLoopingComputer`. CUDA graphs are not supported
    yet: if ``use_cuda_graph_decoder`` is set, a warning is logged and decoding proceeds with CUDA
    graphs disabled.
    """

    def __init__(
        self,
        decoder_model,
        joint_model,
        blank_index: int,
        durations: list,
        num_punct: int,
        num_cap: int,
        num_spell: int,
        max_symbols_per_step: Optional[int] = 10,
        include_duration: bool = False,
        use_cuda_graph_decoder: bool = False,
    ):
        super().__init__(
            decoder_model=decoder_model,
            joint_model=joint_model,
            blank_index=blank_index,
            durations=durations,
            max_symbols_per_step=max_symbols_per_step,
            include_duration=include_duration,
        )
        self.num_punct = num_punct
        self.num_cap = num_cap
        self.num_spell = num_spell
        self._SOS = num_spell

        if use_cuda_graph_decoder:
            logging.warning(
                "CUDA graph decoding is not supported yet for the multistream cap+punct TDT batched decoder; "
                "running with CUDA graphs disabled."
            )

        self.decoding_computer = GreedyBatchedMultiStreamCapPunctTDTLabelLoopingComputer(
            decoder=decoder_model,
            joint=joint_model,
            blank_index=blank_index,
            durations=durations,
            num_punct=num_punct,
            num_cap=num_cap,
            num_spell=num_spell,
            max_symbols_per_step=max_symbols_per_step,
            include_duration=include_duration,
        )

    @property
    def max_symbols(self):
        return self._max_symbols

    @max_symbols.setter
    def max_symbols(self, value):
        self._max_symbols = value
        computer = self.__dict__.get("decoding_computer", None)
        if computer is not None:
            computer.max_symbols = value

    @typecheck()
    def forward(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        partial_hypotheses: Optional[List[rnnt_utils.Hypothesis]] = None,
    ):
        """Greedy-decode an encoder batch into a list of (packed) Hypotheses with product ids."""
        if partial_hypotheses is not None:
            raise NotImplementedError(
                "`partial_hypotheses` is not supported for batched cap+punct TDT greedy decoding."
            )

        decoder_training_state = self.decoder.training
        joint_training_state = self.joint.training

        with torch.inference_mode():
            encoder_output = encoder_output.transpose(1, 2)  # (B, D, T) -> (B, T, D)
            self.decoder.eval()
            self.joint.eval()

            batched_hyps, _alignments, _state = self.decoding_computer(x=encoder_output, out_len=encoded_lengths)
            hypotheses = rnnt_utils.batched_hyps_to_hypotheses(
                batched_hyps, None, batch_size=encoder_output.shape[0]
            )
            packed_result = pack_hypotheses(hypotheses, encoded_lengths)

        self.decoder.train(decoder_training_state)
        self.joint.train(joint_training_state)

        return (packed_result,)
