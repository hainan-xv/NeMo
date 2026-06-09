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

"""Greedy decoding for the Aligner-Encoder ASR model.

Two decoding modes are supported, matching the two model variants:

* ``ar`` (autoregressive): the prediction network is seeded with a start-of-
  sentence (SOS) embedding and then, for each consecutive encoder frame, the
  one-to-one joint produces a distribution over the vocabulary; the arg-max token
  is emitted and fed back into the prediction network. Decoding stops when EOS is
  produced or the encoder frames are exhausted. Complexity is ``O(U)``.

* ``nonar`` (non-autoregressive / CTC-like): every encoder frame is classified
  independently by the per-frame head; tokens after the first EOS are discarded.

Reference: Stooke et al., "Aligner-Encoders: Self-Attention Transformers Can Be
Self-Transducers" (https://arxiv.org/abs/2502.05232).
"""

from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig

__all__ = ['AlignerDecoding']


class AlignerDecoding:
    """Holds references to the model sub-modules and performs greedy decoding.

    Args:
        decoding_cfg: Dict-like config. Recognized keys are ``aligner_type``
            (``ar`` or ``nonar``) and ``max_symbols`` (an optional hard cap on the
            number of emitted tokens; defaults to the number of encoder frames).
        decoder: The RNN-T-style prediction network (only used for ``ar``).
        joint: The :class:`AlignerJoint` (only used for ``ar``).
        eos_id: Index of the EOS token in the joint's output space.
        vocabulary: List of tokens for character models (``None`` for BPE).
        tokenizer: Tokenizer for BPE models (``None`` for character models).
        ctc_head: The :class:`AlignerCTCHead` (only used for ``nonar``).
    """

    def __init__(
        self,
        decoding_cfg: DictConfig,
        decoder,
        joint,
        eos_id: int,
        vocabulary: Optional[List[str]] = None,
        tokenizer=None,
        ctc_head=None,
    ):
        self.cfg = decoding_cfg
        self.decoder = decoder
        self.joint = joint
        self.ctc_head = ctc_head
        self.eos_id = eos_id
        self.blank_id = eos_id  # kept for API symmetry; the Aligner has no blank
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer

        self.aligner_type = decoding_cfg.get('aligner_type', 'ar') if decoding_cfg is not None else 'ar'
        self.max_symbols = decoding_cfg.get('max_symbols', None) if decoding_cfg is not None else None

        if self.aligner_type not in ('ar', 'nonar'):
            raise ValueError(f"aligner_type must be 'ar' or 'nonar', got '{self.aligner_type}'.")

    # ------------------------------------------------------------------ #
    # Token id <-> string helpers
    # ------------------------------------------------------------------ #
    def decode_ids_to_str(self, ids: List[int]) -> str:
        """Convert a list of token ids to a string, dropping EOS/padding."""
        ids = [int(i) for i in ids if int(i) != self.eos_id]
        if self.tokenizer is not None:
            return self.tokenizer.ids_to_text(ids)
        if self.vocabulary is not None:
            return ''.join(self.vocabulary[i] for i in ids if 0 <= i < len(self.vocabulary))
        return ' '.join(str(i) for i in ids)

    # ------------------------------------------------------------------ #
    # Decoding entry point
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decode_encoder_output(
        self, encoder_output: torch.Tensor, encoded_lengths: torch.Tensor
    ) -> Tuple[List[str], List[List[int]]]:
        """Decode a batch of encoder outputs into hypothesis strings and ids.

        Args:
            encoder_output: Encoder output of shape ``(B, D, T)``.
            encoded_lengths: Valid frame counts per sample, ``(B,)``.

        Returns:
            A tuple ``(texts, token_ids)`` where ``texts`` is a list of decoded
            strings and ``token_ids`` is the corresponding list of id lists.
        """
        if self.aligner_type == 'nonar':
            token_ids = self._nonar_greedy(encoder_output, encoded_lengths)
        else:
            token_ids = self._ar_greedy(encoder_output, encoded_lengths)

        texts = [self.decode_ids_to_str(ids) for ids in token_ids]
        return texts, token_ids

    # ------------------------------------------------------------------ #
    # Autoregressive greedy decoding
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _ar_greedy(self, encoder_output: torch.Tensor, encoded_lengths: torch.Tensor) -> List[List[int]]:
        if self.decoder is None or self.joint is None:
            raise RuntimeError("Autoregressive decoding requires both a prediction network and a joint network.")

        device = encoder_output.device
        batch_size = encoder_output.size(0)
        # (B, D, T) -> (B, T, D)
        h = encoder_output.transpose(1, 2)
        max_frames = h.size(1)
        lengths = encoded_lengths.to(device).long()

        max_symbols = self.max_symbols if self.max_symbols is not None else max_frames

        hypotheses: List[List[int]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        states = None
        last_label = None  # None triggers the SOS (zero) embedding inside predict().

        for t in range(min(max_frames, max_symbols)):
            if last_label is None:
                g, states = self.decoder.predict(None, states, add_sos=False, batch_size=batch_size)
            else:
                g, states = self.decoder.predict(last_label, states, add_sos=False)

            # g: (B, 1, H) -> (B, H, 1); frame: (B, 1, D) -> (B, D, 1)
            g_step = g.transpose(1, 2)
            frame = h[:, t : t + 1, :].transpose(1, 2)
            logits = self.joint(encoder_outputs=frame, decoder_outputs=g_step)  # (B, 1, V)
            k = logits[:, 0, :].argmax(dim=-1)  # (B,)

            beyond_len = t >= lengths
            emit_eos = k == self.eos_id
            newly_done = beyond_len | emit_eos

            for b in range(batch_size):
                if finished[b] or newly_done[b]:
                    continue
                hypotheses[b].append(int(k[b].item()))

            finished = finished | newly_done
            last_label = k.unsqueeze(1)  # (B, 1); ignored for finished samples.

            if bool(finished.all()):
                break

        return hypotheses

    # ------------------------------------------------------------------ #
    # Non-autoregressive greedy decoding
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _nonar_greedy(self, encoder_output: torch.Tensor, encoded_lengths: torch.Tensor) -> List[List[int]]:
        if self.ctc_head is None:
            raise RuntimeError("Non-autoregressive decoding requires a per-frame CTC head (ctc_head).")

        logits = self.ctc_head(encoder_output=encoder_output)  # (B, T, V)
        preds = logits.argmax(dim=-1)  # (B, T)
        lengths = encoded_lengths.long().cpu()

        hypotheses: List[List[int]] = []
        for b in range(preds.size(0)):
            seq = preds[b, : int(lengths[b].item())].tolist()
            out: List[int] = []
            for tok in seq:
                if tok == self.eos_id:
                    break
                out.append(int(tok))
            hypotheses.append(out)

        return hypotheses
