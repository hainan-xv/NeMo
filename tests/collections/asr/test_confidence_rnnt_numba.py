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

"""The CUDA confidence RNN-T must agree with the PyTorch reference.

The reference in ``losses/confidence_rnnt.py`` is the ground truth: a plain
double loop over the lattice, slow but transparent. These tests check the kernel
against it on BOTH the loss and the gradient -- a kernel that gets the loss right
while getting alpha/beta occupancies wrong is a real and easy failure mode, and
only the gradient comparison catches it.

Ragged batches are included deliberately: off-by-one errors in the per-element
T and U bounds are the single most likely kernel bug, and uniform lengths hide
them completely.
"""

import pytest
import torch

from nemo.collections.asr.losses.confidence_rnnt import confidence_rnnt_loss

numba_cuda = pytest.importorskip("numba.cuda", reason="numba.cuda required")

from nemo.collections.asr.parts.numba.confidence_rnnt.confidence_rnnt_numba import (  # noqa: E402
    confidence_rnnt_loss_cuda,
    kernel_is_usable,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

V = 6
BLANK = V


def _make(B, T, U, seed, device="cuda", ragged=False):
    g = torch.Generator().manual_seed(seed)
    py = torch.randn(B, T, U + 1, V + 1, generator=g, dtype=torch.float32).log_softmax(-1)
    pc = torch.randn(B, T, U + 1, 2, generator=g, dtype=torch.float32).log_softmax(-1)
    targets = torch.randint(0, V, (B, U), generator=g, dtype=torch.int64)
    if ragged and B > 1:
        il = torch.tensor([T] + [max(1, T - 1 - i % max(1, T - 1)) for i in range(B - 1)], dtype=torch.int64)
        tl = torch.tensor([U] + [max(0, U - 1 - i % max(1, U)) for i in range(B - 1)], dtype=torch.int64)
    else:
        il = torch.full((B,), T, dtype=torch.int64)
        tl = torch.full((B,), U, dtype=torch.int64)
    return (py.to(device), pc.to(device), targets.to(device), il.to(device), tl.to(device))


@pytest.mark.parametrize("B,T,U", [(1, 4, 2), (2, 5, 3), (3, 7, 4), (2, 10, 1)])
def test_cuda_loss_matches_reference(B, T, U):
    py, pc, tgt, il, tl = _make(B, T, U, seed=B * 100 + T)
    ok, why = kernel_is_usable(py)
    assert ok, why

    got = confidence_rnnt_loss_cuda(py, pc, tgt, il, tl, BLANK, reduction=None)
    ref = confidence_rnnt_loss(
        py.double().cpu(), pc.double().cpu(), tgt.cpu(), il.cpu(), tl.cpu(), BLANK, reduction=None
    )
    torch.testing.assert_close(got.double().cpu(), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("B,T,U", [(2, 6, 3), (3, 8, 4)])
def test_cuda_loss_matches_reference_ragged(B, T, U):
    """Per-element T and U bounds: the likeliest place for an off-by-one."""
    py, pc, tgt, il, tl = _make(B, T, U, seed=7, ragged=True)
    got = confidence_rnnt_loss_cuda(py, pc, tgt, il, tl, BLANK, reduction=None)
    ref = confidence_rnnt_loss(
        py.double().cpu(), pc.double().cpu(), tgt.cpu(), il.cpu(), tl.cpu(), BLANK, reduction=None
    )
    torch.testing.assert_close(got.double().cpu(), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("B,T,U", [(2, 5, 3), (3, 6, 2)])
def test_cuda_gradient_matches_reference(B, T, U):
    """Analytic kernel gradient vs autograd through the reference, BOTH heads."""
    py, pc, tgt, il, tl = _make(B, T, U, seed=B * 31 + U)

    a_py = py.clone().requires_grad_(True)
    a_pc = pc.clone().requires_grad_(True)
    confidence_rnnt_loss_cuda(a_py, a_pc, tgt, il, tl, BLANK, reduction="sum").backward()

    b_py = py.double().cpu().clone().requires_grad_(True)
    b_pc = pc.double().cpu().clone().requires_grad_(True)
    confidence_rnnt_loss(b_py, b_pc, tgt.cpu(), il.cpu(), tl.cpu(), BLANK, reduction="sum").backward()

    torch.testing.assert_close(a_py.grad.double().cpu(), b_py.grad, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(a_pc.grad.double().cpu(), b_pc.grad, rtol=1e-3, atol=1e-4)


def test_confidence_head_receives_nonzero_gradient_from_kernel():
    py, pc, tgt, il, tl = _make(2, 5, 3, seed=11)
    pc = pc.clone().requires_grad_(True)
    confidence_rnnt_loss_cuda(py, pc, tgt, il, tl, BLANK, reduction="sum").backward()
    assert pc.grad is not None and pc.grad.abs().sum() > 0


def test_alpha_beta_agree_on_the_likelihood():
    """llForward and llBackward must agree; they are computed independently."""
    from nemo.collections.asr.parts.numba.confidence_rnnt import confidence_rnnt_numba as m

    py, pc, tgt, il, tl = _make(2, 6, 3, seed=13)
    from nemo.collections.asr.losses.confidence_rnnt import compose_effective_logprobs

    eff = compose_effective_logprobs(py, pc, BLANK)
    blank_lp, label_lp = m._gather_columns(eff, tgt, BLANK)

    B, T, U1 = blank_lp.shape
    dev = blank_lp.device
    from numba import cuda as _c

    alphas = torch.empty((B, T, U1), device=dev, dtype=torch.float32)
    betas = torch.empty((B, T, U1), device=dev, dtype=torch.float32)
    llf = torch.empty((B,), device=dev, dtype=torch.float32)
    llb = torch.empty((B,), device=dev, dtype=torch.float32)
    tlen = il.to(torch.int32)
    ulen = tl.to(torch.int32)
    bl = blank_lp.detach().contiguous()
    la = label_lp.detach().contiguous()
    st = _c.external_stream(torch.cuda.current_stream(dev).cuda_stream)
    from nemo.collections.asr.parts.numba.confidence_rnnt.confidence_rnnt_kernel import (
        compute_alphas_kernel,
        compute_betas_kernel,
    )

    compute_alphas_kernel[B, U1, st](
        _c.as_cuda_array(bl),
        _c.as_cuda_array(la),
        _c.as_cuda_array(alphas),
        _c.as_cuda_array(llf),
        _c.as_cuda_array(tlen),
        _c.as_cuda_array(ulen),
        T,
        U1,
    )
    compute_betas_kernel[B, U1, st](
        _c.as_cuda_array(bl),
        _c.as_cuda_array(la),
        _c.as_cuda_array(betas),
        _c.as_cuda_array(llb),
        _c.as_cuda_array(tlen),
        _c.as_cuda_array(ulen),
        T,
        U1,
    )
    torch.cuda.current_stream(dev).synchronize()
    torch.testing.assert_close(llf, llb, rtol=1e-4, atol=1e-4)
