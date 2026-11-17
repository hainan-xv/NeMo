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

"""Reference two-head confidence RNN-T: anchored against plain RNN-T.

The strongest available check is a REDUCTION: pin the confidence head to
"always confident" (P_c(1) = 1) and the model must collapse to standard RNN-T
exactly, because the defer branch carries zero mass. NeMo's own rnnt_numpy
implementation is the ground truth for that.

The second check encodes the identifiability property directly: moving mass
between P_c(0) and P_y(blank) along the curve that holds P_eff fixed must leave
the loss NUMERICALLY UNCHANGED. That is the degeneracy, written as a test, so it
is visible rather than folklore -- and if a future change to the lattice breaks
the degeneracy on purpose, this test is what will catch it.
"""

import numpy as np
import pytest
import torch

from nemo.collections.asr.losses.confidence_rnnt import (
    compose_effective_logprobs,
    confidence_rnnt_loss,
)
from nemo.collections.asr.parts.numba.rnnt_loss import rnnt_numpy

B, T, U, V = 2, 5, 3, 6
BLANK = V  # last index


def _rand_heads(seed=0, device="cpu", dtype=torch.float64):
    g = torch.Generator(device="cpu").manual_seed(seed)
    py = torch.randn(B, T, U + 1, V + 1, generator=g, dtype=dtype).log_softmax(-1)
    pc = torch.randn(B, T, U + 1, 2, generator=g, dtype=dtype).log_softmax(-1)
    targets = torch.randint(0, V, (B, U), generator=g, dtype=torch.int64)
    in_lens = torch.full((B,), T, dtype=torch.int64)
    tgt_lens = torch.full((B,), U, dtype=torch.int64)
    return py.to(device), pc.to(device), targets.to(device), in_lens.to(device), tgt_lens.to(device)


def _always_confident_like(log_pc):
    """log P_c with all mass on class 1."""
    out = torch.full_like(log_pc, -1e30)
    out[..., 1] = 0.0
    return out


def test_reduces_to_plain_rnnt_when_always_confident():
    py, pc, tgt, il, tl = _rand_heads(seed=1)
    pc = _always_confident_like(pc)

    got = confidence_rnnt_loss(py, pc, tgt, il, tl, blank_id=BLANK, reduction="sum")

    # rnnt_numpy.RNNTLoss returns a batch-reduced scalar and requires int64 labels.
    ref = rnnt_numpy.RNNTLoss(blank=BLANK)(py.float(), tgt, il, tl)
    torch.testing.assert_close(got.float().reshape(()), ref.float().reshape(()), rtol=1e-4, atol=1e-4)


def test_composition_is_a_valid_distribution():
    py, pc, *_ = _rand_heads(seed=2)
    eff = compose_effective_logprobs(py, pc, BLANK)
    total = torch.logsumexp(eff, dim=-1)
    torch.testing.assert_close(total, torch.zeros_like(total), rtol=0, atol=1e-9)


def test_confidence_is_unidentifiable_loss_is_invariant_on_the_curve():
    """Shift mass between P_c(0) and P_y(blank) holding P_eff fixed -> same loss.

    This is the degeneracy proved in docs/reports/confidence_rnnt.tex, as code.
    """
    py, pc, tgt, il, tl = _rand_heads(seed=3)
    base = confidence_rnnt_loss(py, pc, tgt, il, tl, blank_id=BLANK, reduction=None)
    eff_ref = compose_effective_logprobs(py, pc, BLANK)

    # Pick a different confidence level, then solve for the P_y that reproduces
    # the SAME P_eff:   P_y(y) = P_eff(y)/P_c(1);  P_y(blank) = (P_eff(b)-P_c(0))/P_c(1)
    for c1 in (0.95, 0.80, 0.60):
        log_c1 = torch.log(torch.tensor(c1, dtype=py.dtype))
        log_c0 = torch.log(torch.tensor(1.0 - c1, dtype=py.dtype))
        pc2 = torch.stack([log_c0.expand(B, T, U + 1), log_c1.expand(B, T, U + 1)], dim=-1)

        py2 = eff_ref - log_c1
        # blank channel: log( (exp(eff_b) - c0) / c1 )
        eff_b = eff_ref[..., BLANK]
        inner = torch.exp(eff_b) - (1.0 - c1)
        if (inner <= 0).any():
            continue  # this c1 is not reachable for every node; skip
        py2 = py2.clone()
        py2[..., BLANK] = torch.log(inner) - log_c1

        # sanity: py2 must still be a distribution
        tot = torch.logsumexp(py2, dim=-1)
        torch.testing.assert_close(tot, torch.zeros_like(tot), rtol=0, atol=1e-8)

        got = confidence_rnnt_loss(py2, pc2, tgt, il, tl, blank_id=BLANK, reduction=None)
        torch.testing.assert_close(got, base, rtol=0, atol=1e-8)


def test_gradient_flows_to_both_heads():
    py, pc, tgt, il, tl = _rand_heads(seed=4)
    py = py.clone().requires_grad_(True)
    pc = pc.clone().requires_grad_(True)
    confidence_rnnt_loss(py, pc, tgt, il, tl, blank_id=BLANK).backward()
    assert py.grad is not None and torch.isfinite(py.grad).all()
    assert pc.grad is not None and torch.isfinite(pc.grad).all()
    assert pc.grad.abs().sum() > 0, "confidence head received no gradient at all"


@pytest.mark.parametrize("seed", [5, 6, 7])
def test_loss_is_finite_and_positive(seed):
    py, pc, tgt, il, tl = _rand_heads(seed=seed)
    nll = confidence_rnnt_loss(py, pc, tgt, il, tl, blank_id=BLANK, reduction=None)
    assert torch.isfinite(nll).all() and (nll > 0).all()
