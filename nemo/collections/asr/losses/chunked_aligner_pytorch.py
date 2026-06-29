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
from nemo.utils import logging


def _is_global_rank0() -> bool:
    """True on rank 0 (or when distributed is not initialized)."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True

__all__ = [
    'ChunkedAlignerLossPytorch',
    'chunked_aligner_loss_bruteforce',
    'ChunkedAlignerNarLossPytorch',
    'chunked_aligner_nar_loss_bruteforce',
    'ChunkwiseAlignerLoss',
    'chunkwise_aligner_single_path_logprob',
    'ChatExternalAlignerCELoss',
    'chat_external_aligner_single_path_logprob',
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
# Chunkwise-Aligner baseline (external alignment, single fixed path)
# ============================================================================
# This is the *external-alignment* counterpart of the alignment-free Chunked
# Aligner above. It implements the training objective of "Chunkwise Aligners for
# Streaming Speech Recognition" (arXiv:2605.11422): instead of full-summing over
# every way to distribute the labels across chunks, an EXTERNAL aligner fixes the
# label->chunk assignment, and training maximizes the probability of that single
# path through the SAME lattice / topology as ``ChunkedAlignerLossPytorch``.
#
# Given the per-token chunk assignment (which chunk each label is emitted in),
# left-packing within a chunk fully determines the path: the j-th token assigned
# to chunk ``c`` (j = 0..k_c-1) is emitted at frame ``c*C + j`` from predictor
# state = its global token index, and a blank/EOC closes chunk ``c`` at frame
# ``c*C + k_c`` whenever ``k_c < frames_in_chunk_c`` (a completely filled chunk
# rolls over with no blank, exactly as in the full-sum lattice). The loss is then
# just the negative sum of the token + EOC log-probs along that one path -- a sum
# of differentiable ``gather`` ops, so autograd provides the backward (no DP, no
# CUDA kernel needed). This makes it directly comparable to the full-sum loss:
# both score the same arcs, the baseline simply commits to one segmentation.


def chunkwise_aligner_single_path_logprob(
    acts: torch.Tensor,
    labels: torch.Tensor,
    act_lens: torch.Tensor,
    label_lens: torch.Tensor,
    chunk_counts: torch.Tensor,
    blank: int,
    chunk_size: int,
) -> torch.Tensor:
    """Reference single-path log-prob for a *fixed* chunk segmentation (loop-based).

    ``chunk_counts[b, c]`` is the number of labels assigned to chunk ``c`` for
    sample ``b`` (``sum_c chunk_counts[b, c] == U_b``). Returns per-sample
    ``log P(path | acts)`` of shape ``(B,)`` for the single left-packed path
    implied by that segmentation, or ``NEG_INF`` if the segmentation is infeasible
    (a chunk is assigned more tokens than it has frames, or the counts do not sum
    to ``U_b``). ``acts`` are raw logits (log_softmax applied internally), matching
    :class:`ChunkwiseAlignerLoss`. Mirrors a single ``recurse`` branch of
    :func:`chunked_aligner_loss_bruteforce` -- used for correctness tests.
    """
    acts = torch.log_softmax(acts.float(), dim=-1)
    B = acts.shape[0]
    C = chunk_size

    out = []
    for b in range(B):
        T_b = int(act_lens[b])
        U_b = int(label_lens[b])
        n_chunks = (T_b + C - 1) // C
        counts = [int(chunk_counts[b, c]) for c in range(n_chunks)]

        feasible = sum(counts) == U_b
        score = 0.0
        u = 0
        if feasible:
            for c in range(n_chunks):
                base = c * C
                frames_here = min(C, T_b - base)
                k = counts[c]
                if k > frames_here or u + k > U_b:
                    feasible = False
                    break
                for i in range(k):
                    score = score + float(acts[b, base + i, u, labels[b, u]])
                    u += 1
                if k < frames_here:
                    score = score + float(acts[b, base + k, u, blank])

        if feasible and u == U_b:
            out.append(torch.tensor(score))
        else:
            out.append(torch.tensor(NEG_INF))

    return torch.stack(out)


class ChunkwiseAlignerLoss(Loss):
    """Single fixed-path (external-alignment) Chunkwise-Aligner loss.

    Same lattice / topology and ``acts`` layout as :class:`ChunkedAlignerLossPytorch`
    (``acts: (B, T, U+1, V)`` raw logits, ``blank`` doubles as end-of-chunk), but
    instead of full-summing over segmentations the loss scores the SINGLE path
    fixed by an external aligner via the per-token chunk assignment
    ``token_chunk_ids``. The forward is a sum of ``gather`` ops, so autograd
    provides the backward.

    Args:
        blank: index of the blank / end-of-chunk (EOC) symbol within ``V``.
        chunk_size: number of encoder frames per chunk ``C``.
        reduction: one of ``'none'``, ``'sum'``, ``'mean'`` (per-sample loss
            divided by label length, then averaged), ``'mean_batch'`` (mean over
            the valid samples) or ``'mean_volume'`` (sum divided by the total
            number of labels in the valid samples).
    """

    @property
    def input_types(self):
        return {
            "acts": NeuralType(('B', 'T', 'T', 'D'), LogprobsType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
            "act_lens": NeuralType(tuple('B'), LengthsType()),
            "label_lens": NeuralType(tuple('B'), LengthsType()),
            "token_chunk_ids": NeuralType(('B', 'T'), LabelsType()),
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, blank: int, chunk_size: int, reduction: str = 'mean_volume'):
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

    def _sample_path_logprob(self, logp_b, labels_b, cids_b, T_b, U_b):
        """Differentiable log-prob of the fixed path for one sample.

        Returns ``(logprob, feasible)`` where ``logprob`` is a scalar tensor (a
        zero tensor when infeasible) and ``feasible`` is a Python bool.
        """
        C = self.chunk_size
        device = logp_b.device
        n_chunks = (T_b + C - 1) // C

        chunk_range = torch.arange(n_chunks, device=device)
        frames_here = torch.clamp(T_b - chunk_range * C, max=C)  # [n_chunks]

        if U_b == 0:
            # Degenerate "all blanks" path; only reached for samples that are
            # already flagged invalid upstream, so feasibility is reported False.
            return logp_b.new_zeros(()), False

        cids = cids_b[:U_b].long()
        # Monotonic, in-range assignment is required for left-packing.
        if cids.min().item() < 0 or cids.max().item() >= n_chunks:
            return logp_b.new_zeros(()), False
        if U_b > 1 and bool((cids[1:] < cids[:-1]).any().item()):
            return logp_b.new_zeros(()), False

        counts = torch.bincount(cids, minlength=n_chunks)[:n_chunks]  # [n_chunks]
        if bool((counts > frames_here).any().item()):
            return logp_b.new_zeros(()), False

        prefix_incl = torch.cumsum(counts, dim=0)  # tokens emitted through chunk c
        prefix_excl = prefix_incl - counts  # global index of first token in chunk c

        # ---- token arcs: token u emitted at frame (chunk*C + within-chunk pos) ----
        u_idx = torch.arange(U_b, device=device)
        within = u_idx - prefix_excl[cids]
        frames = cids * C + within
        tok_logp = logp_b[frames, u_idx, labels_b[:U_b].long()]

        # ---- blank/EOC arcs: one per chunk that is not completely filled ----
        blank_mask = counts < frames_here
        score = tok_logp.sum()
        if bool(blank_mask.any().item()):
            blank_frames = (chunk_range * C + counts)[blank_mask]
            blank_states = prefix_incl[blank_mask]
            bl_logp = logp_b[blank_frames, blank_states, self.blank]
            score = score + bl_logp.sum()
        return score, True

    def forward(self, acts, labels, act_lens, label_lens, token_chunk_ids, valid_mask=None):
        # CPU patch for FP16 (log_softmax on CPU is not implemented for half).
        if not acts.is_cuda and acts.dtype == torch.float16:
            acts = acts.float()

        logp = torch.log_softmax(acts, dim=-1)
        B = acts.shape[0]

        losses = []
        valids = []
        for b in range(B):
            T_b = int(act_lens[b])
            U_b = int(label_lens[b])
            externally_valid = True if valid_mask is None else bool(valid_mask[b])
            if not externally_valid:
                losses.append(logp.new_zeros(()))
                valids.append(False)
                continue
            logprob, feasible = self._sample_path_logprob(logp[b], labels[b], token_chunk_ids[b], T_b, U_b)
            losses.append(-logprob)
            valids.append(feasible)

        losses = torch.stack(losses)
        valids_t = torch.tensor(valids, device=losses.device, dtype=torch.bool)

        if self.reduction == 'none':
            return losses

        # Zero out invalid samples so they never contribute to the (masked) reduction.
        losses = torch.where(valids_t, losses, torch.zeros_like(losses))
        label_lens = label_lens.to(losses.device)
        valid_lens = torch.where(valids_t, label_lens, torch.zeros_like(label_lens))

        if self.reduction == 'sum':
            return losses.sum()
        if self.reduction == 'mean_volume':
            return losses.sum() / valid_lens.sum().clamp(min=1)
        if self.reduction == 'mean_batch':
            denom = valids_t.sum().clamp(min=1)
            return losses.sum() / denom
        if self.reduction == 'mean':
            per_sample = torch.div(losses, valid_lens.clamp(min=1))
            denom = valids_t.sum().clamp(min=1)
            return per_sample.sum() / denom
        return losses


# ============================================================================
# CHAT external-alignment cross-entropy baseline (single fixed path, chunk-indexed)
# ============================================================================
# This is the external-alignment counterpart for the CHAT (Chunk-wise Attention
# Transducer, RNNTAttJoint) model. Unlike the additive-joint Chunkwise Aligner
# above -- whose ``acts`` are FRAME-indexed ``(B, T, U+1, V)`` and which left-packs
# the tokens of a chunk onto the chunk's left-most FRAMES -- the CHAT joint's
# cross-attention already pools each chunk's frames into a single per-chunk
# representation, so its ``acts`` are CHUNK-indexed ``(B, n_chunks, U+1, V)``.
#
# In CHAT the chunk axis plays the role of the RNN-T "time" axis: at chunk ``c``
# the model emits the tokens assigned to that chunk (advancing the predictor
# state ``u``) and then a single blank to advance to chunk ``c+1``. With an
# EXTERNAL aligner fixing the per-token chunk assignment, that path is unique, so
# training reduces to teacher-forced cross-entropy over it -- the standard RNN-T
# loss restricted to one alignment. Concretely, for sample ``b`` with ``n_b``
# valid chunks and ``U_b`` tokens whose chunk ids are ``cids`` (non-decreasing,
# in ``[0, n_b)``):
#
#   * token arc  for token ``u``:  ``logp[b, cids[u], u, labels[u]]``
#   * blank arc  for chunk ``c``:  ``logp[b, c, prefix_incl[c], blank]``  (one per
#     chunk; ``prefix_incl[c]`` = #tokens assigned to chunks ``<= c``)
#
# and the loss is the negative sum of those log-probs. Differences from
# :class:`ChunkwiseAlignerLoss`: (1) the frame index ``c*C + within`` collapses to
# the chunk index ``c``; (2) EVERY chunk emits exactly one blank (the chunk axis
# IS the time axis), not only "not completely filled" chunks; and (3) there is NO
# left-packing feasibility constraint -- a chunk may host arbitrarily many tokens
# (the cross-attention pools the whole chunk), so only monotonic, in-range chunk
# ids are required. The forward is a sum of ``gather`` ops, so autograd provides
# the backward.


def chat_external_aligner_single_path_logprob(
    acts: torch.Tensor,
    labels: torch.Tensor,
    act_lens: torch.Tensor,
    label_lens: torch.Tensor,
    token_chunk_ids: torch.Tensor,
    blank: int,
) -> torch.Tensor:
    """Reference single-path log-prob for the CHAT external-alignment path (loop-based).

    ``acts`` are CHUNK-indexed raw logits ``(B, n_chunks, U+1, V)`` (log_softmax is
    applied internally). ``act_lens[b]`` is the number of valid chunks ``n_b`` and
    ``token_chunk_ids[b, u]`` the chunk that token ``u`` is emitted in. Returns
    per-sample ``log P(path | acts)`` of shape ``(B,)``, or ``NEG_INF`` if the
    assignment is infeasible (out-of-range or non-monotonic chunk ids). Used for
    correctness tests against :class:`ChatExternalAlignerCELoss`.
    """
    logp = torch.log_softmax(acts.float(), dim=-1)
    B = acts.shape[0]

    out = []
    for b in range(B):
        n_b = int(act_lens[b])
        U_b = int(label_lens[b])

        feasible = U_b >= 1 and n_b >= 1
        score = 0.0
        u = 0
        if feasible:
            for c in range(n_b):
                # emit every token assigned to chunk c (left-to-right)
                while u < U_b and int(token_chunk_ids[b, u]) == c:
                    score += float(logp[b, c, u, int(labels[b, u])])
                    u += 1
                # one blank/EOC per chunk to advance the chunk (time) axis
                score += float(logp[b, c, u, blank])
            # any tokens not consumed -> ids were out of range / non-monotonic
            if u != U_b:
                feasible = False
            # all chunk ids must be valid and monotonic non-decreasing
            cids = [int(token_chunk_ids[b, i]) for i in range(U_b)]
            if any(c < 0 or c >= n_b for c in cids):
                feasible = False
            if any(cids[i] < cids[i - 1] for i in range(1, U_b)):
                feasible = False

        out.append(torch.tensor(score) if feasible else torch.tensor(NEG_INF))

    return torch.stack(out)


class ChatExternalAlignerCELoss(Loss):
    """Single fixed-path cross-entropy loss for the CHAT external-alignment baseline.

    Scores the SINGLE chunk->token path fixed by an external aligner (via the
    per-token chunk assignment ``token_chunk_ids``) on the CHAT joint's
    CHUNK-indexed activations ``acts: (B, n_chunks, U+1, V)`` (raw logits). The
    forward is a sum of ``gather`` ops, so autograd provides the backward.

    Delay-randomized, position-weighted training objective (optional)
    -----------------------------------------------------------------
    When ``num_delay_variants >= 1`` AND a within-chunk position tensor
    ``token_chunk_pos`` is supplied AND the module is in training mode, the loss
    is the average over ``num_delay_variants`` randomly *delayed* alignments of a
    *weighted* single-path CE on the SAME joint (so the encoder/joint are computed
    once). For each variant every token is pushed by an independent delay
    ``~ Uniform{0..max_delay}`` chunks, made monotonic by a running max and clamped
    ``< n_chunks``. Each token's CE term is weighted by ``w = weight_gamma ** p``
    where ``p`` is its within-chunk frame position (0 = chunk start): a token that
    stays in its base chunk keeps ``w = gamma**p`` (decays the later in the chunk
    its audio finishes), while a token moved to a later chunk is treated as
    available from frame 0 -> ``w = gamma**0 = 1`` (max). The blank/EOC arc closing
    each chunk takes the weight of the LAST token emitted in that chunk (1.0 if the
    chunk has no tokens). With ``weight_gamma=1`` and ``max_delay=0`` this reduces
    EXACTLY to the plain single-path CE. In eval mode (or when ``token_chunk_pos``
    is ``None``) the plain single-path CE is used so the validation loss is
    deterministic.

    Fixed-delay comparison mode (optional)
    --------------------------------------
    When ``fixed_delay >= 1`` the loss instead shifts EVERY token by that constant
    number of chunks (clamped ``< n_chunks``) and scores the single UNWEIGHTED path
    on the shifted assignment -- a deterministic "delay everything by a fixed
    amount, no sampling/weighting" baseline. It takes priority over the randomized
    objective and is applied in both train and eval (deterministic).

    Args:
        blank: index of the blank / end-of-chunk (EOC) symbol within ``V``.
        reduction: one of ``'none'``, ``'sum'``, ``'mean'`` (per-sample loss
            divided by label length, then averaged over valid samples),
            ``'mean_batch'`` (mean over valid samples) or ``'mean_volume'`` (sum
            divided by the total number of labels in the valid samples).
        num_delay_variants: number of random delayed alignments to average per step
            (``<=0`` disables the delay/weighting -> plain baseline).
        max_delay: maximum per-token delay in chunks (``0`` -> position-weighting
            only, no actual delay).
        weight_gamma: base of the within-chunk position weight ``gamma**p``
            (``(0, 1]``; ``1`` -> uniform weights).
        chunk_size: encoder frames per chunk (used to clamp positions).
        stat_log_every: log mean weight / fraction-delayed / mean-delay every this
            many training forwards (rank 0 only; ``<=0`` disables).
        fixed_delay: if ``>=1``, deterministic constant chunk shift applied to every
            token (single unweighted path, no sampling); overrides the randomized
            objective. ``0`` -> disabled.
    """

    @property
    def input_types(self):
        return {
            "acts": NeuralType(('B', 'T', 'T', 'D'), LogprobsType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
            "act_lens": NeuralType(tuple('B'), LengthsType()),
            "label_lens": NeuralType(tuple('B'), LengthsType()),
            "token_chunk_ids": NeuralType(('B', 'T'), LabelsType()),
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        blank: int,
        reduction: str = 'mean_volume',
        num_delay_variants: int = 0,
        max_delay: int = 0,
        weight_gamma: float = 1.0,
        chunk_size: int = 0,
        stat_log_every: int = 200,
        fixed_delay: int = 0,
    ):
        super().__init__()
        if reduction not in ('none', 'sum', 'mean', 'mean_batch', 'mean_volume'):
            raise ValueError(
                "reduction must be one of "
                "['none', 'sum', 'mean', 'mean_batch', 'mean_volume'], "
                f"got '{reduction}'."
            )
        self.blank = blank
        self.reduction = reduction
        self.num_delay_variants = int(num_delay_variants)
        self.max_delay = max(int(max_delay), 0)
        self.weight_gamma = float(weight_gamma)
        self.chunk_size = int(chunk_size)
        self.stat_log_every = int(stat_log_every)
        self.fixed_delay = max(int(fixed_delay), 0)
        self._delay_step = 0  # counts training forwards with the delay objective

    def _sample_path_logprob(self, logp_b, labels_b, cids_b, n_b, U_b):
        """Differentiable log-prob of the fixed CHAT path for one sample.

        ``logp_b`` is ``(n_chunks, U+1, V)`` log-probs. Returns ``(logprob,
        feasible)`` where ``logprob`` is a scalar tensor (zero when infeasible).
        """
        device = logp_b.device

        if U_b == 0 or n_b == 0:
            return logp_b.new_zeros(()), False

        cids = cids_b[:U_b].long()
        # In-range, monotonic non-decreasing assignment (no left-packing limit).
        if cids.min().item() < 0 or cids.max().item() >= n_b:
            return logp_b.new_zeros(()), False
        if U_b > 1 and bool((cids[1:] < cids[:-1]).any().item()):
            return logp_b.new_zeros(()), False

        counts = torch.bincount(cids, minlength=n_b)[:n_b]  # tokens per chunk
        prefix_incl = torch.cumsum(counts, dim=0)  # tokens emitted through chunk c

        # ---- token arcs: token u emitted at its chunk cids[u], predictor state u ----
        u_idx = torch.arange(U_b, device=device)
        tok_logp = logp_b[cids, u_idx, labels_b[:U_b].long()]

        # ---- blank/EOC arcs: one per chunk c at predictor state prefix_incl[c] ----
        chunk_range = torch.arange(n_b, device=device)
        bl_logp = logp_b[chunk_range, prefix_incl, self.blank]

        return tok_logp.sum() + bl_logp.sum(), True

    def _sample_delayed_path_logprob(self, logp_b, labels_b, cids_b, pos_b, n_b, U_b):
        """Weighted log-prob of ONE randomly delayed CHAT path for a sample.

        Pushes every token by an independent delay ``~ Uniform{0..max_delay}``
        chunks (running-max for monotonicity, clamped ``< n_chunks``), then weights
        each token arc by ``weight_gamma ** p_eff`` where ``p_eff`` is its
        within-chunk position (0 if it moved to a later chunk than its audio, or if
        the position is unknown). The blank/EOC arc that closes each chunk takes the
        weight of the LAST token emitted in that chunk (the "prior token"), or 1.0
        for a chunk with no tokens. Returns ``(weighted_logprob, feasible, stats)``
        where ``stats`` is ``(sum_weight, n_delayed, sum_delta, n_tokens)`` tensors/
        ints for periodic logging (``None`` when infeasible).
        """
        device = logp_b.device
        if U_b == 0 or n_b == 0:
            return logp_b.new_zeros(()), False, None

        cids = cids_b[:U_b].long()
        if cids.min().item() < 0 or cids.max().item() >= n_b:
            return logp_b.new_zeros(()), False, None
        if U_b > 1 and bool((cids[1:] < cids[:-1]).any().item()):
            return logp_b.new_zeros(()), False, None

        # Sample per-token chunk delays, then make monotonic non-decreasing and
        # clamp into range. cummax of a non-decreasing-plus-noise sequence stays
        # non-decreasing; clamping by a constant ceiling preserves that.
        if self.max_delay > 0:
            delays = torch.randint(0, self.max_delay + 1, (U_b,), device=device)
        else:
            delays = torch.zeros(U_b, dtype=torch.long, device=device)
        cnew = torch.cummax(cids + delays, dim=0).values.clamp(max=n_b - 1)
        delta = cnew - cids  # actual chunk displacement (>= 0)

        # Effective within-chunk position -> weight. Moved-to-a-later-chunk (delta>0)
        # or unknown position (<0) => p_eff=0 => max weight.
        pos = pos_b[:U_b].long()
        p_eff = torch.where((delta > 0) | (pos < 0), torch.zeros_like(pos), pos)
        if self.chunk_size > 0:
            p_eff = p_eff.clamp(min=0, max=self.chunk_size - 1)
        else:
            p_eff = p_eff.clamp(min=0)
        weights = self.weight_gamma ** p_eff.to(logp_b.dtype)

        u_idx = torch.arange(U_b, device=device)
        tok_logp = logp_b[cnew, u_idx, labels_b[:U_b].long()]
        weighted_tok = (weights * tok_logp).sum()

        counts = torch.bincount(cnew, minlength=n_b)[:n_b]
        prefix_incl = torch.cumsum(counts, dim=0)
        chunk_range = torch.arange(n_b, device=device)
        bl_logp = logp_b[chunk_range, prefix_incl, self.blank]

        # EOC/blank weight = weight of the last token emitted in that chunk (the
        # token sitting at index prefix_incl[c]-1); 1.0 for a chunk with no tokens.
        last_idx = (prefix_incl - 1).clamp(min=0)
        blank_weights = torch.where(
            counts > 0, weights[last_idx], torch.ones_like(weights[last_idx])
        )
        weighted_bl = (blank_weights * bl_logp).sum()

        stats = (weights.sum().detach(), (delta > 0).sum().detach(), delta.sum().detach(), U_b)
        return weighted_tok + weighted_bl, True, stats

    def forward(self, acts, labels, act_lens, label_lens, token_chunk_ids, valid_mask=None, token_chunk_pos=None):
        # CPU patch for FP16 (log_softmax on CPU is not implemented for half).
        if not acts.is_cuda and acts.dtype == torch.float16:
            acts = acts.float()

        logp = torch.log_softmax(acts, dim=-1)
        B = acts.shape[0]

        # Objective selection (priority: fixed-delay > randomized > plain):
        #   * fixed-delay: deterministic constant chunk shift, single unweighted
        #     path (no sampling). Applied in train AND eval since it is
        #     deterministic, so the validation loss reflects the same target.
        #   * randomized: training only, needs within-chunk positions.
        #   * plain single path otherwise (e.g. eval of the randomized model).
        use_fixed = self.fixed_delay >= 1
        use_delay = (
            not use_fixed and self.training and self.num_delay_variants >= 1 and token_chunk_pos is not None
        )
        K = self.num_delay_variants if use_delay else 1

        # Running stats over this batch (tokens x variants) for periodic logging.
        stat_sum_w = logp.new_zeros(())
        stat_n_delayed = logp.new_zeros(())
        stat_sum_delta = logp.new_zeros(())
        stat_n_tok = 0

        losses = []
        valids = []
        for b in range(B):
            n_b = int(act_lens[b])
            U_b = int(label_lens[b])
            externally_valid = True if valid_mask is None else bool(valid_mask[b])
            if not externally_valid:
                losses.append(logp.new_zeros(()))
                valids.append(False)
                continue
            if use_fixed:
                # Shift every (real) token by a constant number of chunks, then
                # score the single unweighted path on the shifted assignment.
                shifted = token_chunk_ids[b].clone()
                if U_b > 0:
                    shifted[:U_b] = (shifted[:U_b].long() + self.fixed_delay).clamp(max=max(n_b - 1, 0))
                logprob, feasible = self._sample_path_logprob(logp[b], labels[b], shifted, n_b, U_b)
                losses.append(-logprob)
                valids.append(feasible)
            elif use_delay:
                acc = logp.new_zeros(())
                feasible = False
                for _ in range(K):
                    lp, feas, st = self._sample_delayed_path_logprob(
                        logp[b], labels[b], token_chunk_ids[b], token_chunk_pos[b], n_b, U_b
                    )
                    acc = acc + (-lp)
                    feasible = feasible or feas
                    if st is not None:
                        sw, nd, sd, nt = st
                        stat_sum_w = stat_sum_w + sw
                        stat_n_delayed = stat_n_delayed + nd
                        stat_sum_delta = stat_sum_delta + sd
                        stat_n_tok += int(nt)
                losses.append(acc / K)
                valids.append(feasible)
            else:
                logprob, feasible = self._sample_path_logprob(logp[b], labels[b], token_chunk_ids[b], n_b, U_b)
                losses.append(-logprob)
                valids.append(feasible)

        if use_delay:
            self._delay_step += 1
            if (
                self.stat_log_every > 0
                and self._delay_step % self.stat_log_every == 0
                and stat_n_tok > 0
                and _is_global_rank0()
            ):
                ntok = float(stat_n_tok)
                logging.info(
                    f"[chat-aligner] delay-weight stats (step~{self._delay_step}, K={K}): "
                    f"mean_token_weight={float(stat_sum_w.item()) / ntok:.3f}, "
                    f"frac_delayed={float(stat_n_delayed.item()) / ntok:.3f}, "
                    f"mean_delta_chunks={float(stat_sum_delta.item()) / ntok:.3f} "
                    f"(over {stat_n_tok} token x variant slots)."
                )

        losses = torch.stack(losses)
        valids_t = torch.tensor(valids, device=losses.device, dtype=torch.bool)

        if self.reduction == 'none':
            return losses

        # Zero out invalid samples so they never contribute to the (masked) reduction.
        losses = torch.where(valids_t, losses, torch.zeros_like(losses))
        label_lens = label_lens.to(losses.device)
        valid_lens = torch.where(valids_t, label_lens, torch.zeros_like(label_lens))

        if self.reduction == 'sum':
            return losses.sum()
        if self.reduction == 'mean_volume':
            return losses.sum() / valid_lens.sum().clamp(min=1)
        if self.reduction == 'mean_batch':
            denom = valids_t.sum().clamp(min=1)
            return losses.sum() / denom
        if self.reduction == 'mean':
            per_sample = torch.div(losses, valid_lens.clamp(min=1))
            denom = valids_t.sum().clamp(min=1)
            return per_sample.sum() / denom
        return losses


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
