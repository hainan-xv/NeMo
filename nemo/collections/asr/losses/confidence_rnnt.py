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

"""Reference implementation of the two-head "confidence" RNN-T loss.

THE MODEL. Every lattice node (t, u) carries TWO heads instead of one:

    token head       P_y(k | t, u)   over  V + {blank}
    confidence head  P_c(1 | t, u),  P_c(0 | t, u) = 1 - P_c(1 | t, u)

"Confident" gates committing. From node (t, u):

    emit y_{u+1}            P_c(1) * P_y(y_{u+1})      -> (t,   u+1)
    blank, confident        P_c(1) * P_y(blank)        -> (t+1, u)
    defer (not confident)   P_c(0)                     -> (t+1, u)

The intent is to separate two things a single blank conflates in streaming:
"there is genuinely nothing here" from "I cannot tell yet and want more audio".

A KNOWN PROPERTY, STATED UP FRONT. Both blank branches lead to the SAME successor
(t+1, u), so the likelihood depends on the two heads only through

    P_eff(blank) = P_c(1) * P_y(blank) + P_c(0)
    P_eff(y)     = P_c(1) * P_y(y)

which is one equation in two unknowns: infinitely many (P_c, P_y) give identical
loss, so the confidence head is NOT identified by this objective alone. See
docs/reports/confidence_rnnt.tex for the proof. This module implements the
formulation anyway so the claim can be tested empirically rather than argued --
the split that training actually lands on is an experimental question, and shared
backbone / initialisation effects are outside the algebra.

That same identity is what makes this implementation simple and trustworthy: the
composition is done in log space and the standard RNN-T recursion is run on
P_eff. Autograd then routes gradients to both heads by the chain rule.
"""

from typing import Optional

import torch

__all__ = ["compose_effective_logprobs", "confidence_rnnt_loss"]

NEG_INF = -1e30


def compose_effective_logprobs(
    log_py: torch.Tensor,
    log_pc: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    """Fold the two heads into the effective per-node distribution.

    Args:
        log_py: (B, T, U+1, V+1) log P_y over tokens + blank, normalised over -1.
        log_pc: (B, T, U+1, 2) log P_c, normalised over -1; index 1 = confident.
        blank_id: index of blank in the last axis of ``log_py``.

    Returns:
        (B, T, U+1, V+1) log P_eff, a valid distribution over the same support.
    """
    if log_py.shape[:3] != log_pc.shape[:3]:
        raise ValueError(f"head shapes disagree: {tuple(log_py.shape)} vs {tuple(log_pc.shape)}")
    if log_pc.shape[-1] != 2:
        raise ValueError(f"confidence head must have 2 classes, got {log_pc.shape[-1]}")

    log_c1 = log_pc[..., 1:2]  # (B, T, U+1, 1) keepdim for broadcast
    log_c0 = log_pc[..., 0:1]

    # Every channel is scaled by P_c(1) ...
    log_eff = log_py + log_c1
    # ... and the blank channel additionally absorbs the whole P_c(0) mass.
    blank_col = torch.logaddexp(log_eff[..., blank_id], log_c0.squeeze(-1))
    log_eff = log_eff.clone()
    log_eff[..., blank_id] = blank_col
    return log_eff


def _rnnt_forward_logprob(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lens: torch.Tensor,
    target_lens: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    """Standard RNN-T forward recursion in log space. Returns log P(Y|X), (B,).

    Deliberately a plain double loop: this is the slow ground truth the fast
    kernels are checked against, so clarity beats speed.
    """
    B, T, Up1, _ = log_probs.shape
    device = log_probs.device
    alpha = log_probs.new_full((B, T, Up1), NEG_INF)
    alpha[:, 0, 0] = 0.0

    for t in range(T):
        for u in range(Up1):
            if t == 0 and u == 0:
                continue
            terms = []
            if t > 0:
                terms.append(alpha[:, t - 1, u] + log_probs[:, t - 1, u, blank_id])
            if u > 0:
                tgt = targets[:, u - 1]
                emit = log_probs[:, t, u - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                terms.append(alpha[:, t, u - 1] + emit)
            alpha[:, t, u] = torch.logsumexp(torch.stack(terms, dim=0), dim=0) if terms else NEG_INF

    # Terminate on a blank at (T_b - 1, U_b) per batch element.
    idx_t = (input_lens - 1).clamp(min=0)
    idx_u = target_lens
    bsel = torch.arange(B, device=device)
    return alpha[bsel, idx_t, idx_u] + log_probs[bsel, idx_t, idx_u, blank_id]


def confidence_rnnt_loss(
    log_py: torch.Tensor,
    log_pc: torch.Tensor,
    targets: torch.Tensor,
    input_lens: torch.Tensor,
    target_lens: torch.Tensor,
    blank_id: int,
    reduction: Optional[str] = "mean",
) -> torch.Tensor:
    """Negative log-likelihood of the two-head confidence transducer.

    Shapes follow the RNN-T convention: ``log_py`` is (B, T, U+1, V+1),
    ``log_pc`` is (B, T, U+1, 2), ``targets`` is (B, U).
    """
    log_eff = compose_effective_logprobs(log_py, log_pc, blank_id)
    nll = -_rnnt_forward_logprob(log_eff, targets, input_lens, target_lens, blank_id)
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll
