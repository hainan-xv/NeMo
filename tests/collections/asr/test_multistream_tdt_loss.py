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

"""Gradient-correctness tests for the multistream TDT loss.

The multistream TDT loss has three implementations:

* :class:`MultistreamTDTLossPytorch` - fully differentiable; its gradient comes
  from autograd. Used here as the ground truth.
* :class:`MultistreamTDTLoss` - on CPU uses an explicit analytic forward-backward
  (custom ``autograd.Function``); on CUDA it dispatches to the fused Numba kernels
  (:class:`MultistreamTDTLossNumba`). Both consume already-normalized log-probs.
* :class:`MultistreamTDTLossNumba` - the CUDA/Numba kernels (tested directly on GPU).

These tests assert that (a) the loss values match the differentiable reference,
(b) the gradient matches the autograd gradient of the reference, (c) the CPU
custom backward passes ``torch.autograd.gradcheck``, and (d) on GPU the Numba
kernels match the reference for both loss and gradient (float64 and float32).
"""

import numpy as np
import pytest
import torch

from nemo.collections.asr.losses.rnnt_pytorch import MultistreamTDTLoss, MultistreamTDTLossPytorch

DEVICES = ['cpu']
if torch.cuda.is_available():
    DEVICES.append('cuda')


def _build_inputs(device, dtype=torch.float64, seed=0, small=False):
    """Construct a small but non-trivial multistream-TDT problem.

    Streams (label part):
        bow  : 2 classes      -> dividers slice [0, 2)
        cap  : 3 classes      -> dividers slice [2, 5)
        spell: Vspell + blank -> dividers slice [5, 5 + Vspell + 1)
    blank is the last label index. Duration part is appended after the labels.

    Set ``small=True`` for a tiny problem used by the (expensive) gradcheck test.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if small:
        durations = [0, 1, 2]
        bow, cap, vspell = 2, 2, 2
        B, T = 1, 4
        act_lens = torch.tensor([T], dtype=torch.long, device=device)
        label_lens = torch.tensor([2], dtype=torch.long, device=device)
    else:
        durations = [0, 1, 2, 3]
        bow, cap, vspell = 2, 3, 5
        B, T = 2, 6
        act_lens = torch.tensor([T, T - 1], dtype=torch.long, device=device)
        label_lens = torch.tensor([3, 2], dtype=torch.long, device=device)

    n_dur = len(durations)
    dividers = [0, bow, bow + cap, bow + cap + vspell + 1]
    label_dim = dividers[-1]
    blank = label_dim - 1
    K = len(dividers) - 1  # number of streams

    D = label_dim + n_dur
    acts = torch.randn(B, T, T + 1, D, dtype=dtype, device=device)

    U = int(label_lens.max())
    labels = torch.zeros(B, U, K, dtype=torch.long, device=device)
    for b in range(B):
        for u in range(int(label_lens[b])):
            labels[b, u, 0] = np.random.randint(dividers[0], dividers[1])
            labels[b, u, 1] = np.random.randint(dividers[1], dividers[2])
            labels[b, u, 2] = np.random.randint(dividers[2], dividers[3] - 1)  # exclude blank

    return acts, labels, act_lens, label_lens, durations, dividers, blank


class TestMultistreamTDTLoss:
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("sigma", [0.0, 0.05])
    def test_loss_value_matches_reference(self, device, sigma):
        acts, labels, act_lens, label_lens, durations, dividers, blank = _build_inputs(device)

        ref = MultistreamTDTLossPytorch(
            blank=blank, durations=durations, dividers=dividers, reduction='none', sigma=sigma
        )
        analytic = MultistreamTDTLoss(
            blank=blank, durations=durations, dividers=dividers, reduction='none', sigma=sigma
        )

        loss_ref = ref(acts, labels, act_lens, label_lens)
        loss_an = analytic(acts, labels, act_lens, label_lens)

        assert torch.allclose(loss_ref, loss_an, atol=1e-5, rtol=1e-5), (loss_ref, loss_an)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("sigma", [0.0, 0.05])
    def test_gradient_matches_autograd(self, device, sigma):
        acts, labels, act_lens, label_lens, durations, dividers, blank = _build_inputs(device)

        # ----- differentiable reference: gradient via autograd -----
        acts_ref = acts.clone().detach().requires_grad_(True)
        ref = MultistreamTDTLossPytorch(
            blank=blank, durations=durations, dividers=dividers, reduction='sum', sigma=sigma
        )
        loss_ref = ref(acts_ref, labels, act_lens, label_lens)
        loss_ref.backward()

        # ----- analytic implementation: gradient via custom backward -----
        acts_an = acts.clone().detach().requires_grad_(True)
        analytic = MultistreamTDTLoss(
            blank=blank, durations=durations, dividers=dividers, reduction='sum', sigma=sigma
        )
        loss_an = analytic(acts_an, labels, act_lens, label_lens)
        loss_an.backward()

        assert torch.allclose(loss_ref, loss_an, atol=1e-5, rtol=1e-5)
        assert torch.allclose(acts_ref.grad, acts_an.grad, atol=1e-5, rtol=1e-5), (
            (acts_ref.grad - acts_an.grad).abs().max().item()
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("sigma", [0.0, 0.05])
    def test_gradcheck_custom_backward(self, device, sigma):
        acts, labels, act_lens, label_lens, durations, dividers, blank = _build_inputs(device, small=True)
        analytic = MultistreamTDTLoss(
            blank=blank, durations=durations, dividers=dividers, reduction='sum', sigma=sigma
        )

        acts = acts.clone().detach().requires_grad_(True)

        def fn(a):
            return analytic(a, labels, act_lens, label_lens)

        assert torch.autograd.gradcheck(fn, (acts,), eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Numba multistream-TDT kernels require CUDA.")
class TestMultistreamTDTLossNumba:
    """Validate the fused Numba CUDA kernels against the differentiable reference."""

    @pytest.mark.unit
    @pytest.mark.parametrize("sigma", [0.0, 0.05])
    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
    def test_numba_matches_reference(self, sigma, dtype):
        from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import MultistreamTDTLossNumba

        acts, labels, act_lens, label_lens, durations, dividers, blank = _build_inputs('cuda', dtype=dtype)

        atol = 1e-5 if dtype == torch.float64 else 2e-3
        rtol = 1e-5 if dtype == torch.float64 else 2e-3

        # reference (autograd) gradient
        acts_ref = acts.clone().detach().requires_grad_(True)
        ref = MultistreamTDTLossPytorch(
            blank=blank, durations=durations, dividers=dividers, reduction='sum', sigma=sigma
        )
        loss_ref = ref(acts_ref, labels, act_lens, label_lens)
        loss_ref.backward()

        # numba kernel gradient
        acts_nb = acts.clone().detach().requires_grad_(True)
        numba_loss = MultistreamTDTLossNumba(
            blank=blank, durations=durations, dividers=dividers, reduction='sum', sigma=sigma
        )
        loss_nb = numba_loss(acts_nb, labels, act_lens, label_lens)
        loss_nb.backward()

        assert torch.allclose(loss_ref.float(), loss_nb.float().view_as(loss_ref), atol=atol, rtol=rtol), (
            loss_ref.item(),
            loss_nb.item(),
        )
        assert torch.allclose(acts_ref.grad.float(), acts_nb.grad.float(), atol=atol, rtol=rtol), (
            (acts_ref.grad - acts_nb.grad).abs().max().item()
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
