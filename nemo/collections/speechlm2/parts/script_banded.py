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
"""Banded forward algorithm for SCRIPT.

SCRIPT trains as conditional text completion: given the transcript so far (the
SPINE) and one chunk of audio, a BRANCH emits the words revealed by that chunk.
Today the assignment of words to chunks is a single path chosen by the forced
aligner, and the loss is plain cross-entropy over it -- the exact analogue of
CHAT's ``forced_alignment``. The aligner is not exact, so that path charges a
chunk for a word whose audio has not finished arriving, and the model is trained
to guess.

This module marginalises over a BAND of nearby paths instead, the way CHAT's
banded RNN-T loss does.

WHY A DYNAMIC PROGRAM IS EVEN LEGAL HERE
    The band is only tractable because the spine is PATH-INDEPENDENT. SCRIPT's
    attention rule gives branch ``j`` a prefix of the spine (``spine_idx <
    branch_prefix[j]``) and no access to any other branch, and the spine carries
    no audio. So the state "chunk ``t``, spine cut at token ``u``" fully
    determines a branch's input: the audio window follows from ``t`` and the
    history is ``spine[:u]``. Two different partitions that reach the same
    ``(t, u)`` therefore share a branch context and can be merged -- exactly the
    property RNN-T gets from its prediction network never seeing audio.

    That holds ONLY under ``target_construction="partition"``. Under ``legacy``
    each chunk's text is tokenized on its own, so moving a word re-tokenizes both
    neighbours and the spine ids change with the path; the states do not merge
    and this DP is invalid. It also fails under ``gate_in_history=True``, where
    the spine gets a READ/WRITE token whose identity is ``write_id if target_ids
    else read_id`` -- i.e. it flips precisely when the band empties a chunk. Both
    are rejected at construction time rather than silently mis-trained.

COST
    ``branch_prefix`` enters the model only through the branch mask's
    ``spine_idx < branch_prefix``, so one spine forward serves branches at
    arbitrarily different cuts. A band that admits ``C`` start cuts per chunk
    therefore pays the spine cost ONCE and the branch cost ``C`` times. And one
    branch forward at ``(t, u)`` yields every span length at once, because the
    score of emitting ``spine[u:u+k]`` and then stopping is a running prefix sum
    of that branch's own token log-probs.
"""

from typing import Optional

import torch
from torch import Tensor

# Finite sentinel rather than -inf: an all-masked logsumexp over -inf produces
# NaN in the backward pass, which then poisons every parameter. CHAT's banded
# RNN-T uses the same constant for the same reason.
NEG_INF = -1e30

# Anything at or below this is treated as "no path", not as a very small number.
_EMPTY = NEG_INF / 2


def span_scores(token_logprob: Tensor, stop_logprob: Tensor, span_valid: Optional[Tensor] = None) -> Tensor:
    """Score every in-band span length from one branch forward per start cut.

    Args:
        token_logprob: ``(B, T, C, K)`` log P(the k-th spine token after this
            branch's start cut | the branch). Position ``k`` scores the token at
            spine index ``cut + k``.
        stop_logprob: ``(B, T, C, K + 1)`` log P(end-of-chunk | the branch, after
            ``k`` tokens). ``k = 0`` is an empty chunk.
        span_valid: optional ``(B, T, C, K + 1)`` mask; False entries are scored
            ``NEG_INF``.

    Returns:
        ``(B, T, C, K + 1)`` where entry ``k`` is the log-probability of emitting
        exactly ``k`` tokens and then stopping.

    The cumulative sum is the whole point: emitting ``spine[u:u+k]`` shares its
    first ``k-1`` factors with emitting ``spine[u:u+k-1]``, so a single forward
    at cut ``u`` scores every span length that starts there.
    """
    b, t, c, k = token_logprob.shape
    if stop_logprob.shape != (b, t, c, k + 1):
        raise ValueError(f"stop_logprob must be {(b, t, c, k + 1)}, got {tuple(stop_logprob.shape)}")
    zero = token_logprob.new_zeros((b, t, c, 1))
    emitted = torch.cat([zero, torch.cumsum(token_logprob, dim=-1)], dim=-1)
    out = emitted + stop_logprob
    if span_valid is not None:
        out = torch.where(span_valid, out, out.new_full((), NEG_INF))
    return out


def _scatter_logsumexp(index: Tensor, values: Tensor, size: int) -> Tensor:
    """Numerically-stable log-sum-exp of ``values`` grouped by ``index``.

    ``index`` and ``values`` are ``(B, N)``; the result is ``(B, size)``. Groups
    that receive nothing come back as ``NEG_INF`` rather than NaN.

    The running max is detached. It is only a constant shift, so dropping its
    gradient changes nothing mathematically, and keeping it would route gradient
    through a ``scatter_reduce(amax)`` whose backward is both unnecessary and,
    for ties, arbitrary.
    """
    b = values.shape[0]
    base = values.new_full((b, size), NEG_INF)
    mx = base.scatter_reduce(1, index, values, reduce="amax", include_self=True).detach()
    empty = mx <= _EMPTY
    safe = torch.where(empty, torch.zeros_like(mx), mx)
    summed = torch.zeros_like(base).scatter_add(1, index, torch.exp(values - safe.gather(1, index)))
    return torch.where(empty, base, safe + torch.log(summed.clamp_min(torch.finfo(values.dtype).tiny)))


def banded_forward(
    span_logprob: Tensor,
    cut: Tensor,
    cut_valid: Tensor,
    n_chunks: Tensor,
    n_tokens: Tensor,
) -> Tensor:
    """Negative log-likelihood of the whole transcript, marginalised over the band.

    Args:
        span_logprob: ``(B, T, C, K + 1)`` from :func:`span_scores` -- log P(emit
            ``k`` tokens from this start cut, then stop).
        cut: ``(B, T, C)`` int64. ``cut[b, t, j]`` is the spine token index where
            candidate ``j`` of chunk ``t`` starts.
        cut_valid: ``(B, T, C)`` bool. False pads a chunk with fewer than ``C``
            candidates.
        n_chunks: ``(B,)`` int64, real chunk count per utterance.
        n_tokens: ``(B,)`` int64, spine tokens to be emitted (the final state).

    Returns:
        ``(B,)`` per-utterance NLL. An utterance no in-band path can complete
        comes back as ``-NEG_INF``; the caller must decide what to do with it
        rather than have it silently dominate a mean.

    The recursion is the RNN-T forward with the lattice's two arc types replaced
    by a single variable-length one::

        alpha[0, 0]  = 0,  alpha[0, u != 0] = -inf
        alpha[t, v]  = logsumexp over (j, k) with cut[t, j] + k == v of
                           alpha[t - 1, cut[t, j]] + span_logprob[t, j, k]
        NLL          = -alpha[T, U]

    Chunks past an utterance's own length carry ``alpha`` forward unchanged, so a
    short utterance in a long batch is neither advanced nor zeroed.
    """
    if span_logprob.dim() != 4:
        raise ValueError(f"span_logprob must be (B, T, C, K+1), got {tuple(span_logprob.shape)}")
    b, t_max, c, kp1 = span_logprob.shape
    if cut.shape != (b, t_max, c) or cut_valid.shape != (b, t_max, c):
        raise ValueError("cut and cut_valid must both be (B, T, C)")
    u_max = int(n_tokens.max().item()) if n_tokens.numel() else 0

    device = span_logprob.device
    ks = torch.arange(kp1, device=device).view(1, 1, kp1)

    alpha = span_logprob.new_full((b, u_max + 1), NEG_INF)
    alpha[:, 0] = 0.0

    for t in range(t_max):
        cut_t = cut[:, t]  # (B, C)
        valid_t = cut_valid[:, t]  # (B, C)

        src = alpha.gather(1, cut_t.clamp(0, u_max))  # (B, C)
        src = torch.where(valid_t, src, src.new_full((), NEG_INF))

        contrib = src.unsqueeze(-1) + span_logprob[:, t]  # (B, C, K+1)
        dest = cut_t.unsqueeze(-1) + ks  # (B, C, K+1)

        ok = valid_t.unsqueeze(-1) & (dest <= n_tokens.view(b, 1, 1))
        contrib = torch.where(ok, contrib, contrib.new_full((), NEG_INF))

        nxt = _scatter_logsumexp(dest.clamp(0, u_max).reshape(b, -1), contrib.reshape(b, -1), u_max + 1)

        # Utterances that have already run out of chunks keep their alpha.
        live = (torch.full_like(n_chunks, t) < n_chunks).view(b, 1)
        alpha = torch.where(live, nxt, alpha)

    return -alpha.gather(1, n_tokens.view(b, 1).clamp(0, u_max)).squeeze(1)
