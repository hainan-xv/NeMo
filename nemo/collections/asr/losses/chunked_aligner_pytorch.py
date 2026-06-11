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

"""Reference (loop-based) PyTorch implementation of the Chunked-Aligner loss.

This is the *alignment-free* training objective for the streaming "Chunked
Aligner": a full-sum (forward algorithm) over every way to distribute the label
sequence across fixed-size encoder chunks, so no external timestamps / forced
alignment are needed (unlike Chunkwise Aligners, arXiv:2605.11422, which fix the
label->chunk assignment with an external aligner).

It is the chunked, streaming generalization of the offline Aligner-Encoder
(Stooke et al., arXiv:2502.05232): within a chunk the encoder is expected to
self-transduce the chunk's tokens onto the chunk's left-most frames (the diagonal
one-to-one joint), while a blank symbol acts as an end-of-chunk (EOC) signal that
advances to the next chunk.

------------------------------------------------------------------------------
Lattice / topology
------------------------------------------------------------------------------
For simplicity this reference uses ``chunk_size == max_tokens_per_chunk``: every
encoder frame is an emission slot (higher memory than the sparse "first K of C
frames" variant, but easier to reason about). The activations therefore have the
exact same layout as the RNN-T / TDT references:

    acts: (B, T, U + 1, V)   raw logits (log_softmax is applied internally)
    labels: (B, U)           target token ids (real tokens only, NO appended EOS)
    act_lens: (B,)           number of valid encoder frames T_b per sample
    label_lens: (B,)         number of valid labels U_b per sample

where ``V`` includes the blank/EOC symbol at index ``blank`` and the ``U + 1``
axis indexes the predictor state ``u`` = number of tokens emitted so far.

Let ``C = chunk_size`` and chunk ``n`` cover frames ``[n*C, (n+1)*C)``. Define
``log_alpha[t, u]`` = log-prob of *arriving at frame t with u tokens emitted*.
Because emissions are left-packed within a chunk, being at frame ``t`` implies
exactly ``t % C`` tokens have been emitted in the current chunk. Two arcs feed a
state:

* token arc ``(t-1, u-1) -> (t, u)``: emit ``y_u`` at frame ``t-1`` using
  predictor state ``u-1``. Always advances the frame by one (the within-chunk
  diagonal); when ``t`` is a chunk boundary this is the "chunk was completely
  filled with tokens" case (no blank).
* blank/EOC arc ``(t', u) -> (n*C, u)``: a blank emitted at *any* frame ``t'`` of
  the previous chunk ``n-1`` ends that chunk and jumps to the start of chunk
  ``n`` (the frames of chunk ``n-1`` after ``t'`` are skipped). Hence blank arcs
  only ever land on chunk-start frames (``t % C == 0``).

Base case ``log_alpha[0, 0] = 0``.

Termination (an utterance is complete once the final chunk is consumed with all
``U`` tokens emitted), summing:

* blank-EOC at any frame ``t'`` of the last chunk with ``u = U``
  (``log_alpha[t', U] + blank``); and
* a token at the very last frame ``T-1`` (``log_alpha[T-1, U-1] + y_U``) -- the
  "last chunk filled exactly to the end of the audio, no trailing blank" case.

------------------------------------------------------------------------------
NOTE
------------------------------------------------------------------------------
This module is a *correctness reference* written with explicit Python loops; it
is not optimized and does not (yet) provide a hand-written backward pass. The
forward is composed of differentiable ops, so autograd would work, but gradients
are intentionally out of scope for this first version.
"""

from typing import List

import torch

from nemo.core.classes import Loss
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType

__all__ = [
    'ChunkedAlignerLossPytorch',
    'chunked_aligner_loss_bruteforce',
    'ChunkedAlignerNarLossPytorch',
    'chunked_aligner_nar_loss_bruteforce',
]

# Large finite stand-in for log(0); mirrors the other PyTorch loss references
# which avoid -inf so that logsumexp never produces NaNs.
NEG_INF = -1.0e9


class ChunkedAlignerLossPytorch(Loss):
    """Loop-based forward-algorithm Chunked-Aligner loss (no backward yet).

    Args:
        blank: index of the blank / end-of-chunk (EOC) symbol within ``V``.
        chunk_size: number of encoder frames per chunk ``C`` (also the maximum
            number of tokens a chunk can emit in this reference variant).
        reduction: one of ``'none'``, ``'sum'``, ``'mean'`` (per-sample loss
            divided by label length, then averaged over the batch), ``'mean_batch'``
            (mean over the batch) or ``'mean_volume'`` (batch sum divided by the
            total number of labels).
    """

    @property
    def input_types(self):
        return {
            "acts": NeuralType(('B', 'T', 'T', 'D'), LogprobsType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
            "act_lens": NeuralType(tuple('B'), LengthsType()),
            "label_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, blank: int, chunk_size: int, reduction: str = 'mean_batch'):
        super().__init__()
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
        if reduction not in ('none', 'sum', 'mean', 'mean_batch', 'mean_volume'):
            raise ValueError(
                "reduction must be one of "
                "['none', 'sum', 'mean', 'mean_batch', 'mean_volume'], "
                f"got '{reduction}'."
            )
        self.blank = blank
        self.chunk_size = chunk_size
        self.reduction = reduction

    def forward(self, acts, labels, act_lens, label_lens):
        # CPU patch for FP16 (log_softmax on CPU is not implemented for half).
        if not acts.is_cuda and acts.dtype == torch.float16:
            acts = acts.float()

        acts = torch.log_softmax(acts, dim=-1)

        forward_logprob = self.compute_forward_prob(acts, labels, act_lens, label_lens)
        losses = -forward_logprob

        if self.reduction == 'none':
            return losses
        if self.reduction == 'sum':
            return losses.sum()
        if self.reduction == 'mean':
            return torch.div(losses, label_lens.clamp(min=1)).mean()
        if self.reduction == 'mean_batch':
            return losses.mean()
        if self.reduction == 'mean_volume':
            return losses.sum() / label_lens.sum().clamp(min=1)
        return losses

    def compute_forward_prob(self, acts, labels, act_lens, label_lens):
        """Forward algorithm; returns per-sample log P(labels | acts), shape (B,)."""
        B, T, Up1, _ = acts.shape
        C = self.chunk_size
        blank = self.blank

        log_probs = []
        for b in range(B):
            T_b = int(act_lens[b])
            U_b = int(label_lens[b])

            log_alpha = torch.full((T_b, U_b + 1), NEG_INF, device=acts.device, dtype=acts.dtype)

            for t in range(T_b):
                n = t // C  # current chunk index
                j = t % C  # slot within the chunk == #tokens emitted so far in this chunk
                for u in range(U_b + 1):
                    if t == 0 and u == 0:
                        log_alpha[t, u] = 0.0
                        continue

                    contribs: List[torch.Tensor] = []

                    # token arc: emit y_u at frame (t-1) from predictor state (u-1)
                    if t >= 1 and u >= 1:
                        contribs.append(log_alpha[t - 1, u - 1] + acts[b, t - 1, u - 1, labels[b, u - 1]])

                    # blank/EOC arc: a blank anywhere in the previous chunk lands here
                    # (only possible on a chunk-start frame).
                    if j == 0 and n >= 1:
                        for t_prev in range((n - 1) * C, n * C):
                            contribs.append(log_alpha[t_prev, u] + acts[b, t_prev, u, blank])

                    if contribs:
                        log_alpha[t, u] = torch.logsumexp(torch.stack(contribs), dim=0)
                    # else: unreachable, stays NEG_INF

            # ---- termination ----
            n_chunks = (T_b + C - 1) // C
            last_chunk_start = (n_chunks - 1) * C

            terms: List[torch.Tensor] = []
            # (a) blank/EOC at any frame of the last chunk, with all tokens emitted.
            for t_prev in range(last_chunk_start, T_b):
                terms.append(log_alpha[t_prev, U_b] + acts[b, t_prev, U_b, blank])
            # (b) last token emitted exactly on the final audio frame (full last chunk).
            if U_b >= 1:
                terms.append(log_alpha[T_b - 1, U_b - 1] + acts[b, T_b - 1, U_b - 1, labels[b, U_b - 1]])

            log_probs.append(torch.logsumexp(torch.stack(terms), dim=0))

        return torch.stack(log_probs)


def chunked_aligner_loss_bruteforce(
    acts: torch.Tensor,
    labels: torch.Tensor,
    act_lens: torch.Tensor,
    label_lens: torch.Tensor,
    blank: int,
    chunk_size: int,
) -> torch.Tensor:
    """Brute-force path enumeration of the Chunked-Aligner log-prob (validation only).

    Explicitly enumerates every chunk segmentation ``(k_0, ..., k_{N-1})`` with
    ``sum_n k_n == U`` (each ``0 <= k_n <= #frames in chunk n``) and log-sum-exps
    the path scores. Exponential in the number of chunks -- use only for tiny
    correctness checks against :class:`ChunkedAlignerLossPytorch`.

    Returns per-sample ``log P(labels | acts)`` of shape ``(B,)`` (acts are raw
    logits; log_softmax is applied internally, matching the loss class).
    """
    acts = torch.log_softmax(acts.float(), dim=-1)
    B = acts.shape[0]
    C = chunk_size

    out = []
    for b in range(B):
        T_b = int(act_lens[b])
        U_b = int(label_lens[b])
        n_chunks = (T_b + C - 1) // C
        chunk_frames = [min(C, T_b - n * C) for n in range(n_chunks)]

        path_logprobs: List[float] = []

        def recurse(chunk_idx: int, u: int, acc: float):
            if chunk_idx == n_chunks:
                if u == U_b:
                    path_logprobs.append(acc)
                return
            frames_here = chunk_frames[chunk_idx]
            base = chunk_idx * C
            # emit k tokens in this chunk (left-packed), then either a blank (k <
            # frames_here) or nothing (k == frames_here -> chunk completely filled).
            for k in range(0, frames_here + 1):
                if u + k > U_b:
                    break
                score = acc
                uu = u
                for i in range(k):
                    frame = base + i
                    score = score + float(acts[b, frame, uu, labels[b, uu]])
                    uu += 1
                if k < frames_here:
                    blank_frame = base + k
                    score_blank = score + float(acts[b, blank_frame, uu, blank])
                    recurse(chunk_idx + 1, uu, score_blank)
                else:
                    # full chunk: no blank, fall through to the next chunk (or to
                    # the end of audio if this was the last chunk).
                    recurse(chunk_idx + 1, uu, score)

        recurse(0, 0, 0.0)

        if path_logprobs:
            out.append(torch.logsumexp(torch.tensor(path_logprobs), dim=0))
        else:
            out.append(torch.tensor(NEG_INF))

    return torch.stack(out)


# ============================================================================
# Non-autoregressive (NAR) variant
# ============================================================================
# Identical lattice / topology to the AR Chunked-Aligner above, but the per-frame
# token distribution does NOT depend on the predictor state ``u`` (there is no
# prediction network / joint). The activations are therefore ``acts: (B, T, V)``
# -- a per-frame projection head output -- instead of the AR ``(B, T, U+1, V)``
# joint tensor. This removes the ``U`` axis entirely (no joint step), which is a
# large training-memory win. Every arc score simply reads ``acts[b, t, x]``
# (``x`` = a target token id, or ``blank``) with no ``u`` index.


class ChunkedAlignerNarLossPytorch(Loss):
    """Loop-based forward-algorithm NAR Chunked-Aligner loss (autograd backward).

    Same chunked full-sum objective as :class:`ChunkedAlignerLossPytorch`, but the
    activations are per-frame logits ``(B, T, V)`` (no ``U`` axis / no joint). The
    forward is composed of differentiable ops, so the backward is provided by
    autograd.

    Args:
        blank: index of the blank / end-of-chunk (EOC) symbol within ``V``.
        chunk_size: number of frames per chunk ``C`` (also the maximum number of
            tokens a chunk can emit in this variant).
        reduction: one of ``'none'``, ``'sum'``, ``'mean'`` (per-sample loss
            divided by label length, then averaged over the batch), ``'mean_batch'``
            (mean over the batch) or ``'mean_volume'`` (batch sum divided by the
            total number of labels).
    """

    @property
    def input_types(self):
        return {
            "acts": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
            "act_lens": NeuralType(tuple('B'), LengthsType()),
            "label_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, blank: int, chunk_size: int, reduction: str = 'mean_batch'):
        super().__init__()
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
        if reduction not in ('none', 'sum', 'mean', 'mean_batch', 'mean_volume'):
            raise ValueError(
                "reduction must be one of "
                "['none', 'sum', 'mean', 'mean_batch', 'mean_volume'], "
                f"got '{reduction}'."
            )
        self.blank = blank
        self.chunk_size = chunk_size
        self.reduction = reduction

    def forward(self, acts, labels, act_lens, label_lens):
        # CPU patch for FP16 (log_softmax on CPU is not implemented for half).
        if not acts.is_cuda and acts.dtype == torch.float16:
            acts = acts.float()

        acts = torch.log_softmax(acts, dim=-1)

        forward_logprob = self.compute_forward_prob(acts, labels, act_lens, label_lens)
        losses = -forward_logprob

        if self.reduction == 'none':
            return losses
        if self.reduction == 'sum':
            return losses.sum()
        if self.reduction == 'mean':
            return torch.div(losses, label_lens.clamp(min=1)).mean()
        if self.reduction == 'mean_batch':
            return losses.mean()
        if self.reduction == 'mean_volume':
            return losses.sum() / label_lens.sum().clamp(min=1)
        return losses

    def compute_forward_prob(self, acts, labels, act_lens, label_lens):
        """Forward algorithm; returns per-sample log P(labels | acts), shape (B,).

        ``acts`` are per-frame log-probs of shape ``(B, T, V)``.
        """
        B, T, _ = acts.shape
        C = self.chunk_size
        blank = self.blank

        log_probs = []
        for b in range(B):
            T_b = int(act_lens[b])
            U_b = int(label_lens[b])

            log_alpha = torch.full((T_b, U_b + 1), NEG_INF, device=acts.device, dtype=acts.dtype)

            for t in range(T_b):
                n = t // C  # current chunk index
                j = t % C  # slot within the chunk == #tokens emitted so far in this chunk
                for u in range(U_b + 1):
                    if t == 0 and u == 0:
                        log_alpha[t, u] = 0.0
                        continue

                    contribs: List[torch.Tensor] = []

                    # token arc: emit y_u at frame (t-1) -- per-frame, no u index.
                    if t >= 1 and u >= 1:
                        contribs.append(log_alpha[t - 1, u - 1] + acts[b, t - 1, labels[b, u - 1]])

                    # blank/EOC arc: a blank anywhere in the previous chunk lands here
                    # (only possible on a chunk-start frame).
                    if j == 0 and n >= 1:
                        for t_prev in range((n - 1) * C, n * C):
                            contribs.append(log_alpha[t_prev, u] + acts[b, t_prev, blank])

                    if contribs:
                        log_alpha[t, u] = torch.logsumexp(torch.stack(contribs), dim=0)
                    # else: unreachable, stays NEG_INF

            # ---- termination ----
            n_chunks = (T_b + C - 1) // C
            last_chunk_start = (n_chunks - 1) * C

            terms: List[torch.Tensor] = []
            # (a) blank/EOC at any frame of the last chunk, with all tokens emitted.
            for t_prev in range(last_chunk_start, T_b):
                terms.append(log_alpha[t_prev, U_b] + acts[b, t_prev, blank])
            # (b) last token emitted exactly on the final audio frame (full last chunk).
            if U_b >= 1:
                terms.append(log_alpha[T_b - 1, U_b - 1] + acts[b, T_b - 1, labels[b, U_b - 1]])

            log_probs.append(torch.logsumexp(torch.stack(terms), dim=0))

        return torch.stack(log_probs)


def chunked_aligner_nar_loss_bruteforce(
    acts: torch.Tensor,
    labels: torch.Tensor,
    act_lens: torch.Tensor,
    label_lens: torch.Tensor,
    blank: int,
    chunk_size: int,
) -> torch.Tensor:
    """Brute-force path enumeration of the NAR Chunked-Aligner log-prob (validation).

    Same enumeration as :func:`chunked_aligner_loss_bruteforce` but with per-frame
    activations ``(B, T, V)`` (no ``u`` index). Exponential in the number of
    chunks -- use only for tiny correctness checks.
    """
    acts = torch.log_softmax(acts.float(), dim=-1)
    B = acts.shape[0]
    C = chunk_size

    out = []
    for b in range(B):
        T_b = int(act_lens[b])
        U_b = int(label_lens[b])
        n_chunks = (T_b + C - 1) // C
        chunk_frames = [min(C, T_b - n * C) for n in range(n_chunks)]

        path_logprobs: List[float] = []

        def recurse(chunk_idx: int, u: int, acc: float):
            if chunk_idx == n_chunks:
                if u == U_b:
                    path_logprobs.append(acc)
                return
            frames_here = chunk_frames[chunk_idx]
            base = chunk_idx * C
            for k in range(0, frames_here + 1):
                if u + k > U_b:
                    break
                score = acc
                uu = u
                for i in range(k):
                    frame = base + i
                    score = score + float(acts[b, frame, labels[b, uu]])
                    uu += 1
                if k < frames_here:
                    blank_frame = base + k
                    score_blank = score + float(acts[b, blank_frame, blank])
                    recurse(chunk_idx + 1, uu, score_blank)
                else:
                    recurse(chunk_idx + 1, uu, score)

        recurse(0, 0, 0.0)

        if path_logprobs:
            out.append(torch.logsumexp(torch.tensor(path_logprobs), dim=0))
        else:
            out.append(torch.tensor(NEG_INF))

    return torch.stack(out)
