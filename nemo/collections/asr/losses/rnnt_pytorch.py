# ! /usr/bin/python
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import math
from typing import List, Optional, Sequence

import torch

from nemo.core.classes import Loss
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType


class RNNTLossPytorch(Loss):
    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "acts": NeuralType(('B', 'T', 'T', 'D'), LogprobsType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
            "act_lens": NeuralType(tuple('B'), LengthsType()),
            "label_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, blank, reduction):
        super().__init__()
        self.blank = blank
        self.reduction = reduction

    def forward(self, acts, labels, act_lens, label_lens):
        # CPU patch for FP16
        if not acts.is_cuda and acts.dtype == torch.float16:
            acts = acts.float()

        acts = torch.log_softmax(acts, -1)

        forward_logprob = self.compute_forward_prob(acts, labels, act_lens, label_lens)
        losses = -forward_logprob
        if self.reduction == 'mean_batch':
            losses = losses.mean()  # global batch size average
        elif self.reduction == 'mean':
            losses = torch.div(losses, label_lens).mean()
        elif self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean_volume':
            losses = losses.sum() / label_lens.sum()  # same as above but longer samples weigh more

        return losses

    def compute_forward_prob(self, acts, labels, act_lens, label_lens):
        B, T, U, _ = acts.shape

        log_alpha = torch.zeros(B, T, U)
        log_alpha = log_alpha.to(acts.device)

        for t in range(T):
            for u in range(U):
                if u == 0:
                    if t == 0:
                        # this is the base case: (t=0, u=0) with log-alpha = 0.
                        log_alpha[:, t, u] = 0.0
                    else:
                        # this is case for (t = 0, u > 0), reached by (t, u - 1)
                        # emitting a blank symbol.
                        log_alpha[:, t, u] = log_alpha[:, t - 1, u] + acts[:, t - 1, 0, self.blank]
                else:
                    if t == 0:
                        # in case of (u > 0, t = 0), this is only reached from
                        # (t, u - 1) with a label emission.
                        gathered = torch.gather(
                            acts[:, t, u - 1], dim=1, index=labels[:, u - 1].view(-1, 1).type(torch.int64)
                        ).reshape(-1)
                        log_alpha[:, t, u] = log_alpha[:, t, u - 1] + gathered.to(log_alpha.device)
                    else:
                        # here both t and u are > 0, this state is reachable
                        # with two possibilities: (t - 1, u) with a blank emission
                        # or (t, u - 1) with a label emission.
                        log_alpha[:, t, u] = torch.logsumexp(
                            torch.stack(
                                [
                                    log_alpha[:, t - 1, u] + acts[:, t - 1, u, self.blank],
                                    log_alpha[:, t, u - 1]
                                    + torch.gather(
                                        acts[:, t, u - 1], dim=1, index=labels[:, u - 1].view(-1, 1).type(torch.int64)
                                    ).reshape(-1),
                                ]
                            ),
                            dim=0,
                        )

        log_probs = []
        for b in range(B):
            # here we need to add the final blank emission weights.
            to_append = (
                log_alpha[b, act_lens[b] - 1, label_lens[b]] + acts[b, act_lens[b] - 1, label_lens[b], self.blank]
            )
            log_probs.append(to_append)
        log_prob = torch.stack(log_probs)

        return log_prob


class TDTLossPytorch(Loss):
    """
    Pure Python implementation of TDT loss (https://arxiv.org/pdf/2304.06795.pdf)
    """

    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "acts": NeuralType(('B', 'T', 'T', 'D'), LogprobsType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
            "act_lens": NeuralType(tuple('B'), LengthsType()),
            "label_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, blank: int, durations: List[int] = [], reduction: str = 'sum', sigma: float = 0.0):
        super().__init__()
        self.blank = blank
        self.durations = durations
        self.n_durations = len(durations)
        self.reduction = reduction
        self.sigma = sigma

    def forward(self, acts, labels, act_lens, label_lens):
        label_acts = acts[:, :, :, : -self.n_durations]
        duration_acts = acts[:, :, :, -self.n_durations :]

        # the - self.sigma here is for logit-undernormalization. Check the paper for details.
        label_acts = torch.log_softmax(label_acts, -1) - self.sigma

        duration_acts = torch.log_softmax(duration_acts, -1)

        forward_logprob, _ = self.compute_forward_prob(label_acts, duration_acts, labels, act_lens, label_lens)
        losses = -forward_logprob
        if self.reduction == 'mean_batch':
            losses = losses.mean()  # global batch size average
        elif self.reduction == 'mean':
            losses = torch.div(losses, label_lens).mean()
        elif self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean_volume':
            losses = losses.sum() / label_lens.sum()  # same as above but longer samples weigh more

        return losses

    def logsumexp(self, a, b):
        ret = torch.logsumexp(torch.stack([a, b]), dim=0)
        return ret

    def compute_forward_prob(self, acts, duration_acts, labels, act_lens, label_lens):
        """This function implements Equation 7 in the TDT paper https://arxiv.org/pdf/2304.06795.pdf,
        Simply put, for each alpha(t, u), it sums over the contribution from all incoming blank arcs and non-blank arcs.
        """
        B, T, U, _ = acts.shape

        log_alpha = torch.zeros(B, T, U)
        log_alpha = log_alpha.cuda()
        for b in range(B):
            for t in range(T):
                for u in range(U):
                    if u == 0:
                        if t == 0:
                            # both t and u are 0, this is the base case for alphas.
                            log_alpha[b, t, u] = 0.0
                        else:
                            # u = 0 and t != 0: only considers blank emissions.
                            log_alpha[b, t, u] = -1000.0
                            for n, l in enumerate(self.durations):
                                if (
                                    t - l >= 0 and l > 0
                                ):  # checking conditions for blank emission, l has to be at least 1
                                    tmp = (
                                        log_alpha[b, t - l, u]
                                        + acts[b, t - l, u, self.blank]
                                        + duration_acts[b, t - l, u, n]
                                    )
                                    log_alpha[b, t, u] = self.logsumexp(tmp, 1.0 * log_alpha[b, t, u])

                    else:
                        # u != 0 here, need to consider both blanks and non-blanks.
                        log_alpha[b, t, u] = -1000.0
                        for n, l in enumerate(self.durations):
                            if t - l >= 0:
                                if l > 0:  # for blank emissions. Need to ensure index is not out-of-bound.
                                    tmp = (
                                        log_alpha[b, t - l, u]
                                        + acts[b, t - l, u, self.blank]
                                        + duration_acts[b, t - l, u, n]
                                    )
                                    log_alpha[b, t, u] = self.logsumexp(tmp, 1.0 * log_alpha[b, t, u])

                                # non-blank emissions.
                                tmp = (
                                    log_alpha[b, t - l, u - 1]
                                    + acts[b, t - l, u - 1, labels[b, u - 1]]
                                    + duration_acts[b, t - l, u - 1, n]
                                )
                                log_alpha[b, t, u] = self.logsumexp(tmp, 1.0 * log_alpha[b, t, u])

        log_probs = []
        for b in range(B):
            tt = torch.Tensor([-1000.0]).cuda()[0]

            # need to loop over all possible ways that blank with different durations contributes to the final loss.
            for n, l in enumerate(self.durations):
                if act_lens[b] - l >= 0 and l > 0:
                    bb = (
                        log_alpha[b, act_lens[b] - l, label_lens[b]]
                        + acts[b, act_lens[b] - l, label_lens[b], self.blank]
                        + duration_acts[b, act_lens[b] - l, label_lens[b], n]
                    )

                    tt = self.logsumexp(bb, 1.0 * tt)

            log_probs.append(tt)

        log_prob = torch.stack(log_probs)

        return log_prob, log_alpha


class MultiblankRNNTLossPytorch(Loss):
    """
    Pure Python implementation of multi-blank transducer loss (https://arxiv.org/pdf/2211.03541.pdf)
    """

    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "acts": NeuralType(('B', 'T', 'T', 'D'), LogprobsType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
            "act_lens": NeuralType(tuple('B'), LengthsType()),
            "label_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, blank, big_blank_durations, reduction: str = "sum", sigma: float = 0.0):
        super().__init__()
        self.blank = blank
        self.big_blank_durations = big_blank_durations
        self.reduction = reduction
        self.sigma = sigma

    def forward(self, acts, labels, act_lens, label_lens):
        acts = torch.log_softmax(acts, -1) - self.sigma
        forward_logprob, _ = self.compute_forward_prob(acts, labels, act_lens, label_lens)

        losses = -forward_logprob
        if self.reduction == 'mean_batch':
            losses = losses.mean()  # global batch size average
        elif self.reduction == 'mean':
            losses = torch.div(losses, label_lens).mean()
        elif self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean_volume':
            losses = losses.sum() / label_lens.sum()  # same as above but longer samples weigh more

        return losses

    def compute_forward_prob(self, acts, labels, act_lens, label_lens):
        B, T, U, _ = acts.shape

        log_alpha = torch.zeros(B, T, U, device=acts.device)
        for t in range(T):
            for u in range(U):
                if u == 0:
                    if t == 0:
                        # this is the base case: (t=0, u=0) with log-alpha = 0.
                        log_alpha[:, t, u] = 0.0
                    else:
                        # this is case for (t = 0, u > 0), reached by (t, u - d)
                        # emitting a blank symbol of duration d.
                        log_alpha[:, t, u] = log_alpha[:, t - 1, u] + acts[:, t - 1, 0, self.blank]
                        for i, d in enumerate(self.big_blank_durations):
                            if t >= d:
                                tt = log_alpha[:, t - d, u] + acts[:, t - d, 0, self.blank - 1 - i]
                                log_alpha[:, t, u] = torch.logsumexp(
                                    torch.stack([1.0 * log_alpha[:, t, u], tt]), dim=0
                                )

                else:
                    if t == 0:
                        # in case of (u > 0, t = 0), this is only reached from
                        # (t, u - 1) with a label emission.
                        gathered = torch.gather(
                            acts[:, t, u - 1], dim=1, index=labels[:, u - 1].view(-1, 1).type(torch.int64)
                        ).reshape(-1)
                        log_alpha[:, t, u] = log_alpha[:, t, u - 1] + gathered
                    else:
                        # here both t and u are > 0, this state is reachable
                        # with two possibilities: (t - d, u) with emission of
                        # blank with duration d, or (t, u - 1) with a label emission.

                        # first we take care of the standard blank.
                        log_alpha[:, t, u] = torch.logsumexp(
                            torch.stack(
                                [
                                    log_alpha[:, t - 1, u] + acts[:, t - 1, u, self.blank],
                                    log_alpha[:, t, u - 1]
                                    + torch.gather(
                                        acts[:, t, u - 1], dim=1, index=labels[:, u - 1].view(-1, 1).type(torch.int64)
                                    ).reshape(-1),
                                ]
                            ),
                            dim=0,
                        )

                        # now we go over all big blanks. They need to be considered if current t >= blank duration d.
                        for i, d in enumerate(self.big_blank_durations):
                            if t >= d:
                                tt = log_alpha[:, t - d, u] + acts[:, t - d, u, self.blank - 1 - i]
                                log_alpha[:, t, u] = torch.logsumexp(
                                    torch.stack([1.0 * log_alpha[:, t, u], tt]), dim=0
                                )

        log_probs = []
        for b in range(B):
            # here we need to add the final blank emission weights, which needs
            # to consider all possible blank durations.
            to_append = (
                log_alpha[b, act_lens[b] - 1, label_lens[b]] + acts[b, act_lens[b] - 1, label_lens[b], self.blank]
            )

            for i, d in enumerate(self.big_blank_durations):
                if act_lens[b] >= d:
                    tt = (
                        log_alpha[b, act_lens[b] - d, label_lens[b]]
                        + acts[b, act_lens[b] - d, label_lens[b], self.blank - 1 - i]
                    )
                    to_append = torch.logsumexp(torch.stack([1.0 * to_append, tt]), dim=0)

            log_probs.append(to_append)
        log_prob = torch.stack(log_probs)

        return log_prob, log_alpha


# ============================================================================
# Multistream TDT loss
# ============================================================================
#
# This combines two ideas:
#   1. TDT (Token-and-Duration Transducer, https://arxiv.org/pdf/2304.06795.pdf),
#      which augments the transducer with an explicit duration distribution so a
#      single step may advance the acoustic frame index by one of several
#      durations.
#   2. The "multistream" transducer factorization, in which a single emitted
#      token is decomposed into K parallel sub-labels (streams). Typical use:
#      treat punctuation / capitalization as independent "modifier" streams on
#      top of a canonical (lowercased, de-punctuated) sub-word stream. The joint
#      output vocabulary is partitioned into contiguous stream slices via
#      `dividers`, each slice is independently log-softmaxed, and the probability
#      of emitting a token factorizes as the product of the per-stream
#      probabilities (sum of per-stream log-probs).
#
# Layout of the joint activation tensor `acts` (shape [B, T, U, D]):
#   * label part : acts[..., : dividers[-1]]  (= acts[..., : -n_durations])
#       - partitioned into streams [dividers[i] : dividers[i+1])
#       - `blank` is the very last label index, i.e. dividers[-1] == blank + 1
#   * duration part : acts[..., -n_durations:]  (one logit per duration)
#
# Targets `labels` have shape [B, U, K] (K = number of streams): for each label
# position u, the K absolute vocabulary indices (one per stream) that are emitted
# jointly at that step.
#
# The transducer lattice is the ordinary TDT lattice over (t, u); only the
# non-blank emission score changes from a single log-prob lookup to the sum of
# the K stream log-probs.

NEG_INF = -1e30


def _lse(terms: List[float]) -> float:
    """logsumexp over a python list of floats, computed in float64 for accuracy."""
    if len(terms) == 0:
        return NEG_INF
    return float(torch.logsumexp(torch.tensor(terms, dtype=torch.float64), dim=0))


@torch.no_grad()
def multistream_tdt_alpha_beta_grad(
    label_logp: torch.Tensor,
    dur_logp: torch.Tensor,
    labels: torch.Tensor,
    act_lens: torch.Tensor,
    label_lens: torch.Tensor,
    durations: Sequence[int],
    blank: int,
):
    """Reference forward-backward for the multistream TDT loss.

    Operates on already-normalized log-probabilities and returns the per-sample
    negative log-likelihood together with the analytic gradients w.r.t. the
    log-prob tensors (i.e. ``d(-logP)/d(label_logp)`` and
    ``d(-logP)/d(dur_logp)``). This is the algorithmic reference a CUDA kernel
    would mirror, and is used to validate gradients against autograd.

    Args:
        label_logp: [B, T, U, V] per-stream log-softmaxed label log-probs
            (blank is index ``blank``).
        dur_logp: [B, T, U, n_durations] log-softmaxed duration log-probs.
        labels: [B, U, K] target indices (K parallel stream labels per position).
        act_lens: [B] acoustic lengths.
        label_lens: [B] label lengths.
        durations: list of integer durations (may include 0 for label emission).
        blank: blank index inside the label part.

    Returns:
        costs: [B] tensor of ``-logP`` (same dtype/device as ``label_logp``).
        grad_label_logp: [B, T, U, V] gradient of ``-logP`` w.r.t. ``label_logp``.
        grad_dur_logp: [B, T, U, n_durations] gradient w.r.t. ``dur_logp``.
    """
    B = label_logp.shape[0]
    K = labels.shape[-1]
    device = label_logp.device
    dtype = label_logp.dtype

    grad_label = torch.zeros_like(label_logp)
    grad_dur = torch.zeros_like(dur_logp)
    costs = torch.zeros(B, device=device, dtype=dtype)

    for b in range(B):
        Tb = int(act_lens[b])
        L = int(label_lens[b])

        def emis(t, u):
            s = 0.0
            for k in range(K):
                s += float(label_logp[b, t, u, int(labels[b, u, k])])
            return s

        def lblank(t, u):
            return float(label_logp[b, t, u, blank])

        def dlp(t, u, n):
            return float(dur_logp[b, t, u, n])

        # ---------------- alpha (forward) ----------------
        alpha = [[NEG_INF] * (L + 1) for _ in range(Tb)]
        alpha[0][0] = 0.0
        for t in range(Tb):
            for u in range(L + 1):
                if t == 0 and u == 0:
                    continue
                terms = []
                for n, l in enumerate(durations):
                    # blank arc into (t, u): from (t - l, u), only for l > 0
                    if l > 0 and t - l >= 0:
                        terms.append(alpha[t - l][u] + lblank(t - l, u) + dlp(t - l, u, n))
                    # label arc into (t, u): from (t - l, u - 1), any duration
                    if u > 0 and t - l >= 0:
                        terms.append(alpha[t - l][u - 1] + emis(t - l, u - 1) + dlp(t - l, u - 1, n))
                alpha[t][u] = _lse(terms)

        fin = []
        for n, l in enumerate(durations):
            if l > 0 and Tb - l >= 0:
                fin.append(alpha[Tb - l][L] + lblank(Tb - l, L) + dlp(Tb - l, L, n))
        logP = _lse(fin)
        costs[b] = -logP

        # ---------------- beta (backward) ----------------
        beta = [[NEG_INF] * (L + 1) for _ in range(Tb)]
        for t in range(Tb - 1, -1, -1):
            for u in range(L, -1, -1):
                terms = []
                for n, l in enumerate(durations):
                    if l > 0:
                        if t + l == Tb and u == L:
                            # terminal blank arc reaching exactly T at u == L
                            terms.append(lblank(t, L) + dlp(t, L, n))
                        elif t + l <= Tb - 1 and beta[t + l][u] > NEG_INF:
                            terms.append(lblank(t, u) + dlp(t, u, n) + beta[t + l][u])
                    if u < L and t + l <= Tb - 1 and beta[t + l][u + 1] > NEG_INF:
                        terms.append(emis(t, u) + dlp(t, u, n) + beta[t + l][u + 1])
                beta[t][u] = _lse(terms)

        # ---------------- gradients (arc posteriors) ----------------
        # d(-logP)/d(score of arc) = -exp(alpha[src] + score + beta[dst] - logP)
        for t in range(Tb):
            for u in range(L + 1):
                a_src = alpha[t][u]
                if a_src <= NEG_INF:
                    continue
                for n, l in enumerate(durations):
                    # blank arc
                    if l > 0:
                        valid = False
                        if t + l == Tb and u == L:
                            score = lblank(t, L) + dlp(t, L, n)
                            bdst = 0.0
                            valid = True
                        elif t + l <= Tb - 1 and beta[t + l][u] > NEG_INF:
                            score = lblank(t, u) + dlp(t, u, n)
                            bdst = beta[t + l][u]
                            valid = True
                        if valid:
                            post = math.exp(a_src + score + bdst - logP)
                            grad_label[b, t, u, blank] -= post
                            grad_dur[b, t, u, n] -= post
                    # label arc
                    if u < L and t + l <= Tb - 1 and beta[t + l][u + 1] > NEG_INF:
                        score = emis(t, u) + dlp(t, u, n)
                        bdst = beta[t + l][u + 1]
                        post = math.exp(a_src + score + bdst - logP)
                        for k in range(K):
                            grad_label[b, t, u, int(labels[b, u, k])] -= post
                        grad_dur[b, t, u, n] -= post

    return costs, grad_label, grad_dur


def _split_multistream_tdt_acts(acts, dividers, sigma, durations):
    """Split raw joint acts into per-stream label log-probs and duration log-probs.

    Returns (label_logp, dur_logp, label_softmax, dur_softmax) where the softmax
    tensors are kept so the analytic path can backprop through the log-softmax.
    """
    n_dur = len(durations)
    label_acts = acts[..., :-n_dur]
    dur_acts = acts[..., -n_dur:]

    label_logp = torch.empty_like(label_acts)
    label_p = torch.empty_like(label_acts)
    for i in range(len(dividers) - 1):
        sl = slice(dividers[i], dividers[i + 1])
        lp = torch.log_softmax(label_acts[..., sl], dim=-1)
        label_logp[..., sl] = lp - sigma
        label_p[..., sl] = lp.exp()

    dur_logp = torch.log_softmax(dur_acts, dim=-1)
    dur_p = dur_logp.exp()
    return label_logp, dur_logp, label_p, dur_p


class MultistreamTDTLossPytorch(Loss):
    """Pure-PyTorch, fully differentiable reference for the multistream TDT loss.

    The forward pass is written entirely with autograd-friendly ops, so the
    gradient is obtained by autograd. This serves both as a usable (slow)
    reference loss and as the ground truth in the gradient correctness test.
    """

    @property
    def input_types(self):
        return {
            "acts": NeuralType(('B', 'T', 'T', 'D'), LogprobsType()),
            "labels": NeuralType(('B', 'T', 'D'), LabelsType()),
            "act_lens": NeuralType(tuple('B'), LengthsType()),
            "label_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        blank: int,
        durations: List[int] = [],
        dividers: Optional[List[int]] = None,
        reduction: str = 'sum',
        sigma: float = 0.0,
    ):
        super().__init__()
        if dividers is None or len(dividers) < 2:
            raise ValueError("`dividers` must be a list with at least 2 entries marking the stream boundaries.")
        if dividers[-1] != blank + 1:
            raise ValueError(f"Expected dividers[-1] ({dividers[-1]}) == blank + 1 ({blank + 1}).")
        self.blank = blank
        self.durations = list(durations)
        self.n_durations = len(durations)
        self.dividers = list(dividers)
        self.reduction = reduction
        self.sigma = sigma

    def _reduce(self, losses, label_lens):
        if self.reduction == 'mean_batch':
            return losses.mean()
        elif self.reduction == 'mean':
            return torch.div(losses, label_lens).mean()
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'mean_volume':
            return losses.sum() / label_lens.sum()
        return losses  # 'none'

    def forward(self, acts, labels, act_lens, label_lens):
        if not acts.is_cuda and acts.dtype == torch.float16:
            acts = acts.float()

        n_dur = self.n_durations
        label_acts = acts[..., :-n_dur]
        dur_acts = acts[..., -n_dur:]

        parts = []
        for i in range(len(self.dividers) - 1):
            parts.append(torch.log_softmax(label_acts[..., self.dividers[i] : self.dividers[i + 1]], dim=-1) - self.sigma)
        label_logp = torch.cat(parts, dim=-1)
        dur_logp = torch.log_softmax(dur_acts, dim=-1)

        forward_logprob = self.compute_forward_prob(label_logp, dur_logp, labels, act_lens, label_lens)
        losses = -forward_logprob
        return self._reduce(losses, label_lens)

    def compute_forward_prob(self, label_logp, dur_logp, labels, act_lens, label_lens):
        """Differentiable forward (alpha) recursion returning [B] log-probabilities."""
        B = label_logp.shape[0]
        device = label_logp.device
        dtype = label_logp.dtype
        neg = torch.tensor(NEG_INF, device=device, dtype=dtype)

        out = []
        for b in range(B):
            Tb = int(act_lens[b])
            L = int(label_lens[b])
            alpha = [[neg for _ in range(L + 1)] for _ in range(Tb)]
            alpha[0][0] = torch.zeros((), device=device, dtype=dtype)
            for t in range(Tb):
                for u in range(L + 1):
                    if t == 0 and u == 0:
                        continue
                    terms = []
                    for n, l in enumerate(self.durations):
                        if l > 0 and t - l >= 0:
                            terms.append(alpha[t - l][u] + label_logp[b, t - l, u, self.blank] + dur_logp[b, t - l, u, n])
                        if u > 0 and t - l >= 0:
                            emis = label_logp[b, t - l, u - 1, labels[b, u - 1]].sum()
                            terms.append(alpha[t - l][u - 1] + emis + dur_logp[b, t - l, u - 1, n])
                    if terms:
                        alpha[t][u] = torch.logsumexp(torch.stack(terms), dim=0)
            fin = []
            for n, l in enumerate(self.durations):
                if l > 0 and Tb - l >= 0:
                    fin.append(alpha[Tb - l][L] + label_logp[b, Tb - l, L, self.blank] + dur_logp[b, Tb - l, L, n])
            out.append(torch.logsumexp(torch.stack(fin), dim=0))
        return torch.stack(out)


class _MultistreamTDTLossFunction(torch.autograd.Function):
    """autograd.Function with an explicit (analytic) backward for multistream TDT.

    Computes the loss and the analytic gradient w.r.t. the raw joint `acts` via
    the alpha/beta forward-backward plus the log-softmax Jacobian. The custom
    backward exists precisely so it can be validated against autograd.
    """

    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens, blank, durations, dividers, sigma):
        label_logp, dur_logp, label_p, dur_p = _split_multistream_tdt_acts(acts, dividers, sigma, durations)

        costs, g_label_logp, g_dur_logp = multistream_tdt_alpha_beta_grad(
            label_logp, dur_logp, labels, act_lens, label_lens, durations, blank
        )

        # Backprop through per-stream log-softmax (label) and duration log-softmax.
        # For y = log_softmax(x): dL/dx = dL/dy - softmax(x) * sum(dL/dy).
        g_label_acts = torch.empty_like(label_p)
        for i in range(len(dividers) - 1):
            sl = slice(dividers[i], dividers[i + 1])
            g = g_label_logp[..., sl]
            g_label_acts[..., sl] = g - label_p[..., sl] * g.sum(dim=-1, keepdim=True)
        g_dur_acts = g_dur_logp - dur_p * g_dur_logp.sum(dim=-1, keepdim=True)

        grad_acts = torch.cat([g_label_acts, g_dur_acts], dim=-1)
        ctx.save_for_backward(grad_acts)
        return costs

    @staticmethod
    def backward(ctx, grad_output):
        (grad_acts,) = ctx.saved_tensors
        grad = grad_acts * grad_output.view(-1, 1, 1, 1).to(grad_acts)
        return grad, None, None, None, None, None, None, None


class MultistreamTDTLoss(Loss):
    """Multistream TDT loss with an analytic (custom) gradient.

    Same math as :class:`MultistreamTDTLossPytorch`, but the gradient is computed
    by an explicit forward-backward instead of autograd. This is the reference
    for a future fused/CUDA implementation and is what gets validated against the
    differentiable reference / autograd.
    """

    @property
    def input_types(self):
        return {
            "acts": NeuralType(('B', 'T', 'T', 'D'), LogprobsType()),
            "labels": NeuralType(('B', 'T', 'D'), LabelsType()),
            "act_lens": NeuralType(tuple('B'), LengthsType()),
            "label_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        blank: int,
        durations: List[int] = [],
        dividers: Optional[List[int]] = None,
        reduction: str = 'sum',
        sigma: float = 0.0,
    ):
        super().__init__()
        if dividers is None or len(dividers) < 2:
            raise ValueError("`dividers` must be a list with at least 2 entries marking the stream boundaries.")
        if dividers[-1] != blank + 1:
            raise ValueError(f"Expected dividers[-1] ({dividers[-1]}) == blank + 1 ({blank + 1}).")
        self.blank = blank
        self.durations = tuple(durations)
        self.dividers = tuple(dividers)
        self.reduction = reduction
        self.sigma = float(sigma)

    def forward(self, acts, labels, act_lens, label_lens):
        if not acts.is_cuda and acts.dtype == torch.float16:
            acts = acts.float()

        losses = _MultistreamTDTLossFunction.apply(
            acts, labels, act_lens, label_lens, self.blank, self.durations, self.dividers, self.sigma
        )

        if self.reduction == 'mean_batch':
            return losses.mean()
        elif self.reduction == 'mean':
            return torch.div(losses, label_lens).mean()
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'mean_volume':
            return losses.sum() / label_lens.sum()
        return losses  # 'none'
