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

"""Greedy decoding for the streaming Chunked-Aligner ASR model.

The Chunked Aligner uses a standard RNN-T predictor + joint (with a blank that
doubles as an end-of-chunk / EOC signal). Decoding walks the encoder frames in
fixed-size chunks and replays the training lattice greedily:

* at the current frame ``t`` and predictor state ``u`` the joint produces a
  distribution over the vocabulary (blank included);
* if the arg-max is the blank/EOC, the rest of the current chunk is skipped and
  decoding jumps to the start of the next chunk (the predictor is *not* advanced);
* otherwise the token is emitted, the predictor is advanced (fed the token), and
  ``t`` moves forward by one (the within-chunk diagonal). A chunk that fills with
  ``chunk_size`` tokens rolls over to the next chunk without an explicit blank.

Decoding stops when the encoder frames are exhausted. Each frame is visited at
most once, so the cost is ``O(T)`` joint steps per utterance.

This mirrors the full-sum training objective implemented by
``ChunkedAlignerLossNumba`` / ``chunked_aligner_pytorch.py``.

Two equivalent implementations are provided:

* :meth:`ChunkedAlignerDecoding._chunked_greedy` -- a simple per-utterance walk
  (one joint call per visited frame, looping over the batch in Python). Easy to
  read; used as the correctness reference.
* :meth:`ChunkedAlignerDecoding._chunked_greedy_batched` -- a batched
  "label-looping" walk (https://arxiv.org/abs/2406.06220) that processes the
  whole batch in lock-step: each outer iteration calls the prediction network at
  most once for the whole batch (only for utterances that emitted a token), while
  an inner loop vectorizes the blank/EOC chunk-jumps. This is the default and is
  numerically identical to the per-utterance version (greedy arg-max). CUDA-graph
  support is intentionally out of scope for now.
"""

from typing import List, Optional, Tuple

import torch

from nemo.utils import logging

__all__ = ['ChunkedAlignerDecoding', 'ChunkedAlignerNarDecoding']


class ChunkedAlignerDecoding:
    """Holds references to the model sub-modules and performs chunked greedy decoding.

    Args:
        decoding_cfg: Dict-like config. Recognized keys are ``max_symbols`` (an
            optional hard cap on the number of emitted tokens per utterance;
            defaults to the number of encoder frames) and ``loop_labels`` (use the
            batched label-looping decoder; defaults to ``True``).
        decoder: The RNN-T prediction network.
        joint: The RNN-T joint network (output space includes the blank/EOC).
        blank_id: Index of the blank / end-of-chunk symbol in the joint output.
        chunk_size: Number of encoder frames per chunk ``C``.
        vocabulary: List of tokens for character models (``None`` for BPE).
        tokenizer: Tokenizer for BPE models (``None`` for character models).
    """

    def __init__(
        self,
        decoding_cfg,
        decoder,
        joint,
        blank_id: int,
        chunk_size: int,
        vocabulary: Optional[List[str]] = None,
        tokenizer=None,
    ):
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
        self.cfg = decoding_cfg
        self.decoder = decoder
        self.joint = joint
        self.blank_id = blank_id
        self.chunk_size = chunk_size
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_symbols = decoding_cfg.get('max_symbols', None) if decoding_cfg is not None else None
        self.loop_labels = decoding_cfg.get('loop_labels', True) if decoding_cfg is not None else True

        # CUDA-graph decoding is not implemented for the Chunked Aligner. If a
        # config requests it (top-level or under a ``greedy`` block), force it off
        # and warn rather than silently ignoring the flag.
        if _requests_cuda_graph(decoding_cfg):
            logging.warning(
                "[ChunkedAlignerDecoding] CUDA-graph decoding (use_cuda_graph_decoder=true) is not "
                "supported by the Chunked-Aligner greedy decoder; disabling it for this run."
            )

    # ------------------------------------------------------------------ #
    # Token id <-> string helpers
    # ------------------------------------------------------------------ #
    def decode_ids_to_str(self, ids: List[int]) -> str:
        """Convert a list of token ids to a string, dropping blank/EOC ids."""
        ids = [int(i) for i in ids if int(i) != self.blank_id]
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

        Uses the batched label-looping walk by default (``loop_labels=True``);
        set ``loop_labels=False`` to fall back to the per-utterance reference walk
        (both produce identical greedy arg-max hypotheses).

        Args:
            encoder_output: Encoder output of shape ``(B, D, T)``.
            encoded_lengths: Valid frame counts per sample, ``(B,)``.

        Returns:
            A tuple ``(texts, token_ids)``.
        """
        if self.loop_labels:
            token_ids = self._chunked_greedy_batched(encoder_output, encoded_lengths)
        else:
            token_ids = self._chunked_greedy(encoder_output, encoded_lengths)
        texts = [self.decode_ids_to_str(ids) for ids in token_ids]
        return texts, token_ids

    # ------------------------------------------------------------------ #
    # Chunked greedy decoding
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _chunked_greedy(self, encoder_output: torch.Tensor, encoded_lengths: torch.Tensor) -> List[List[int]]:
        if self.decoder is None or self.joint is None:
            raise RuntimeError("Chunked-Aligner decoding requires both a prediction network and a joint network.")

        device = encoder_output.device
        batch_size = encoder_output.size(0)
        # (B, D, T) -> (B, T, D)
        h = encoder_output.transpose(1, 2)
        lengths = encoded_lengths.to(device).long()
        C = self.chunk_size

        hypotheses: List[List[int]] = []
        # Per-utterance greedy walk (the EOC jumps make a lock-step batched walk
        # awkward; per-utterance keeps the predictor-state bookkeeping simple).
        for b in range(batch_size):
            T_b = int(lengths[b].item())
            max_symbols = self.max_symbols if self.max_symbols is not None else T_b

            hyp: List[int] = []
            # Seed the predictor with the SOS (zero) embedding for u = 0.
            g, states = self.decoder.predict(None, None, add_sos=False, batch_size=1)
            g_step = g.transpose(1, 2)  # (1, H, 1)

            t = 0
            while t < T_b:
                frame = h[b, t : t + 1, :].unsqueeze(0).transpose(1, 2)  # (1, D, 1)
                logits = self.joint(encoder_outputs=frame, decoder_outputs=g_step)  # (1, 1, 1, V)
                v = int(logits.reshape(-1).argmax().item())

                if v == self.blank_id:
                    # End-of-chunk: skip to the start of the next chunk.
                    t = ((t // C) + 1) * C
                else:
                    hyp.append(v)
                    if len(hyp) >= max_symbols:
                        break
                    last_label = torch.full((1, 1), v, dtype=torch.long, device=device)
                    g, states = self.decoder.predict(last_label, states, add_sos=False)
                    g_step = g.transpose(1, 2)
                    t += 1

            hypotheses.append(hyp)

        return hypotheses

    # ------------------------------------------------------------------ #
    # Batched (label-looping) chunked greedy decoding
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _chunked_greedy_batched(
        self, encoder_output: torch.Tensor, encoded_lengths: torch.Tensor
    ) -> List[List[int]]:
        """Batched label-looping walk; numerically identical to :meth:`_chunked_greedy`.

        The whole batch advances in lock-step. Each outer iteration calls the
        prediction network at most once (after every still-active utterance has
        landed on a token-emitting frame), while an inner loop vectorizes the
        blank/EOC chunk-jumps -- on a blank the predictor state is unchanged, so no
        prediction-network call is needed to skip a chunk. See
        https://arxiv.org/abs/2406.06220 for the label-looping idea.
        """
        if self.decoder is None or self.joint is None:
            raise RuntimeError("Chunked-Aligner decoding requires both a prediction network and a joint network.")

        device = encoder_output.device
        B = encoder_output.size(0)
        # (B, D, T) -> (B, T, D)
        h = encoder_output.transpose(1, 2)
        T_max = h.size(1)
        C = self.chunk_size
        lengths = encoded_lengths.to(device).long()

        if B == 0 or T_max == 0:
            return [[] for _ in range(B)]

        # Per-utterance emission cap (defaults to the number of valid frames).
        if self.max_symbols is not None:
            cap = torch.full((B,), int(self.max_symbols), dtype=torch.long, device=device)
        else:
            cap = lengths.clone()

        batch_idx = torch.arange(B, device=device)
        t = torch.zeros(B, dtype=torch.long, device=device)
        emitted = torch.zeros(B, dtype=torch.long, device=device)
        active = (t < lengths) & (lengths > 0) & (emitted < cap)

        hypotheses: List[List[int]] = [[] for _ in range(B)]

        # Seed the predictor with the SOS (zero) embedding for u = 0 (whole batch).
        g, states = self.decoder.predict(None, None, add_sos=False, batch_size=B)
        g_step = g.transpose(1, 2)  # (B, H, 1)

        v = torch.zeros(B, dtype=torch.long, device=device)
        while bool(active.any()):
            # Inner loop: skip blank/EOC frames (jumping chunk-by-chunk) until every
            # still-active utterance sits on a token-emitting frame. The predictor
            # state is unchanged on a blank, so g_step stays fixed here.
            while True:
                t_safe = t.clamp(max=T_max - 1)
                frame = h[batch_idx, t_safe, :].unsqueeze(-1)  # (B, D, 1)
                logits = self.joint(encoder_outputs=frame, decoder_outputs=g_step)
                v = logits.reshape(B, -1).argmax(dim=-1)  # (B,)

                is_blank = (v == self.blank_id) & active
                next_chunk_start = ((t // C) + 1) * C
                t = torch.where(is_blank, next_chunk_start, t)
                ran_off = is_blank & (t >= lengths)
                active = active & ~ran_off
                # Utterances that jumped to a valid frame must be re-evaluated.
                if not bool((is_blank & active).any()):
                    break

            if not bool(active.any()):
                break

            # Every active utterance now has a (non-blank) token at its current frame.
            emit = active
            v_list = v.tolist()
            emit_list = emit.tolist()
            for b in range(B):
                if emit_list[b]:
                    hypotheses[b].append(int(v_list[b]))
            emitted = emitted + emit.long()

            # Advance the predictor for emitting utterances (one batched call). For
            # non-emitting (finished) utterances feed the blank/pad id, whose
            # embedding is the zero/pad vector, so their state update is harmless.
            labels = torch.where(emit, v, torch.full_like(v, self.blank_id))
            g, states = self.decoder.predict(labels.unsqueeze(1), states, add_sos=False)
            g_step = g.transpose(1, 2)

            # Token arc advances the frame by one; recompute who is still active.
            t = torch.where(emit, t + 1, t)
            active = emit & (t < lengths) & (emitted < cap)

        return hypotheses


def _requests_cuda_graph(decoding_cfg) -> bool:
    """True if the decoding config asks for CUDA-graph decoding (top-level or greedy)."""
    if decoding_cfg is None:
        return False
    try:
        if bool(decoding_cfg.get('use_cuda_graph_decoder', False)):
            return True
        greedy_cfg = decoding_cfg.get('greedy', None)
        if greedy_cfg is not None and bool(greedy_cfg.get('use_cuda_graph_decoder', False)):
            return True
    except (AttributeError, TypeError):
        return False
    return False


class ChunkedAlignerNarDecoding:
    """Greedy decoding for the *non-autoregressive* (NAR) Chunked-Aligner.

    In NAR mode there is no prediction network and no joint: a single per-frame
    projection head maps each (token-extracted) encoder frame to a distribution
    over the vocabulary (blank/EOC included). Because the per-frame distributions
    do not depend on previously emitted tokens, all logits can be computed in one
    batched matmul and the greedy walk is a cheap arg-max scan that replays the
    chunk lattice (blank/EOC jumps to the next chunk; a token advances one frame).

    Args:
        decoding_cfg: Dict-like config. Recognized key is ``max_symbols`` (optional
            hard cap on emitted tokens per utterance; defaults to #frames).
        head: The per-frame projection head, a callable mapping ``[B, T, D]`` ->
            ``[B, T, V]`` (e.g. ``torch.nn.Linear``).
        blank_id: Index of the blank / end-of-chunk symbol in the head output.
        chunk_size: Number of (extracted) frames per chunk ``C``.
        vocabulary: List of tokens for character models (``None`` for BPE).
        tokenizer: Tokenizer for BPE models (``None`` for character models).
    """

    def __init__(
        self,
        decoding_cfg,
        head,
        blank_id: int,
        chunk_size: int,
        vocabulary: Optional[List[str]] = None,
        tokenizer=None,
    ):
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
        self.cfg = decoding_cfg
        self.head = head
        self.blank_id = blank_id
        self.chunk_size = chunk_size
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_symbols = decoding_cfg.get('max_symbols', None) if decoding_cfg is not None else None

    def decode_ids_to_str(self, ids: List[int]) -> str:
        """Convert a list of token ids to a string, dropping blank/EOC ids."""
        ids = [int(i) for i in ids if int(i) != self.blank_id]
        if self.tokenizer is not None:
            return self.tokenizer.ids_to_text(ids)
        if self.vocabulary is not None:
            return ''.join(self.vocabulary[i] for i in ids if 0 <= i < len(self.vocabulary))
        return ' '.join(str(i) for i in ids)

    @torch.no_grad()
    def decode_encoder_output(
        self, encoder_output: torch.Tensor, encoded_lengths: torch.Tensor
    ) -> Tuple[List[str], List[List[int]]]:
        """Decode a batch of (token-extracted) encoder outputs.

        Args:
            encoder_output: Encoder/extracted output of shape ``(B, D, T)``.
            encoded_lengths: Valid frame counts per sample, ``(B,)``.

        Returns:
            A tuple ``(texts, token_ids)``.
        """
        if self.head is None:
            raise RuntimeError("NAR Chunked-Aligner decoding requires a projection head.")
        # (B, D, T) -> (B, T, D) -> head -> (B, T, V); arg-max once (no AR dependency).
        logits = self.head(encoder_output.transpose(1, 2))
        preds = logits.argmax(dim=-1)  # (B, T)
        token_ids = self._greedy(preds, encoded_lengths.to(preds.device).long())
        texts = [self.decode_ids_to_str(ids) for ids in token_ids]
        return texts, token_ids

    @torch.no_grad()
    def _greedy(self, preds: torch.Tensor, lengths: torch.Tensor) -> List[List[int]]:
        C = self.chunk_size
        hypotheses: List[List[int]] = []
        for b in range(preds.size(0)):
            T_b = int(lengths[b].item())
            max_symbols = self.max_symbols if self.max_symbols is not None else T_b
            hyp: List[int] = []
            t = 0
            while t < T_b:
                v = int(preds[b, t].item())
                if v == self.blank_id:
                    # End-of-chunk: skip to the start of the next chunk.
                    t = ((t // C) + 1) * C
                else:
                    hyp.append(v)
                    if len(hyp) >= max_symbols:
                        break
                    t += 1
            hypotheses.append(hyp)
        return hypotheses
