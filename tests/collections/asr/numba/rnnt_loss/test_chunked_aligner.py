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

import random

import numpy as np
import pytest
import torch

from nemo.collections.asr.losses.chunked_aligner_pytorch import (
    ChunkedAlignerLossPytorch,
    ChunkedAlignerNarLossPytorch,
    chunked_aligner_loss_bruteforce,
    chunked_aligner_nar_loss_bruteforce,
)
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import (
    ChunkedAlignerLossNumba,
    ChunkedAlignerNarLossNumba,
)
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__

CUDA_ONLY_DEVICE = ['cuda']


def wrap_and_call(fn, acts, labels, device):
    """Run ``fn`` over ``acts``/``labels`` and return (per-sample costs, acts grad).

    Mirrors the helper used by the RNN-T / TDT numba tests so the numba loss and
    the autograd PyTorch reference are exercised through identical plumbing.
    """
    if not torch.is_tensor(acts):
        acts = torch.tensor(acts)

    if 'cuda' in device:
        acts = acts.cuda()

    if not acts.requires_grad:
        acts.requires_grad = True

    lengths = [acts.shape[1]] * acts.shape[0]
    label_lengths = [len(l) for l in labels]
    labels = torch.LongTensor(labels)
    lengths = torch.LongTensor(lengths)
    label_lengths = torch.LongTensor(label_lengths)
    if 'cuda' in device:
        labels = labels.cuda()
        lengths = lengths.cuda()
        label_lengths = label_lengths.cuda()

    costs = fn(acts, labels, lengths, label_lengths)
    cost = torch.sum(costs)
    cost.backward()

    if 'cuda' in device:
        torch.cuda.synchronize()

    if acts.grad is not None:
        grad = acts.grad.data.cpu().numpy()
    else:
        grad = None

    return costs.data.cpu().numpy(), grad


class TestChunkedAlignerLoss:
    @pytest.mark.unit
    @pytest.mark.parametrize('device', CUDA_ONLY_DEVICE)
    @pytest.mark.parametrize('chunk_size', [2, 3, 4])
    def test_case_randomized_act_label(self, device, chunk_size):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        random.seed(0)
        torch.manual_seed(0)

        B, T, U, V = 4, 8, 4, 8  # V = number of non-blank labels; blank == V
        blank = V

        acts = torch.rand([B, T, U, V + 1])
        # U - 1 real tokens per sample, each strictly less than blank.
        labels = [[random.randrange(0, V) for _ in range(U - 1)] for _ in range(B)]

        fn_numba = ChunkedAlignerLossNumba(blank=blank, chunk_size=chunk_size, reduction='sum')
        numba_cost, numba_grads = wrap_and_call(fn_numba, acts, labels, device)

        fn_ag = ChunkedAlignerLossPytorch(blank=blank, chunk_size=chunk_size, reduction='sum')
        ag_cost, ag_grads = wrap_and_call(fn_ag, acts, labels, device)

        assert np.allclose(numba_cost, ag_cost, atol=1e-4, rtol=1e-5), "chunked-aligner costs mismatch."
        assert np.allclose(numba_grads, ag_grads, atol=1e-4, rtol=1e-3), "chunked-aligner gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', CUDA_ONLY_DEVICE)
    def test_case_variable_lengths(self, device):
        """Variable per-sample acoustic / label lengths and a partial last chunk.

        ``wrap_and_call`` derives lengths from rectangular inputs, so here the
        losses are driven directly with padded labels + explicit length vectors.
        """
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        random.seed(7)
        torch.manual_seed(7)

        B, T, U, V = 3, 10, 5, 6  # T = 10 not a multiple of chunk_size -> partial last chunk
        blank = V
        chunk_size = 4

        act_lens = torch.tensor([10, 7, 9], dtype=torch.int64, device=device)
        label_lens = torch.tensor([4, 2, 3], dtype=torch.int64, device=device)
        # Padded labels [B, U - 1]; entries beyond label_lens[b] are ignored by both losses.
        labels = torch.zeros((B, U - 1), dtype=torch.int64, device=device)
        for b in range(B):
            for u in range(int(label_lens[b])):
                labels[b, u] = random.randrange(0, V)

        acts = torch.rand([B, T, U, V + 1], device=device)

        def run(loss_fn):
            a = acts.clone().detach().requires_grad_(True)
            cost = loss_fn(a, labels, act_lens, label_lens)
            torch.sum(cost).backward()
            torch.cuda.synchronize()
            return cost.detach().cpu().numpy(), a.grad.detach().cpu().numpy()

        numba_cost, numba_grads = run(ChunkedAlignerLossNumba(blank=blank, chunk_size=chunk_size, reduction='sum'))
        ag_cost, ag_grads = run(ChunkedAlignerLossPytorch(blank=blank, chunk_size=chunk_size, reduction='sum'))

        assert np.allclose(numba_cost, ag_cost, atol=1e-4, rtol=1e-5), "chunked-aligner costs mismatch."
        assert np.allclose(numba_grads, ag_grads, atol=1e-4, rtol=1e-3), "chunked-aligner gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', CUDA_ONLY_DEVICE)
    @pytest.mark.parametrize('chunk_size', [2, 3])
    def test_forward_matches_bruteforce(self, device, chunk_size):
        """Cross-check the numba forward cost against exhaustive path enumeration."""
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        random.seed(1)
        torch.manual_seed(1)

        B, T, U, V = 2, 6, 3, 5  # small enough to brute-force all chunk segmentations
        blank = V

        acts = torch.rand([B, T, U, V + 1])
        labels = [[random.randrange(0, V) for _ in range(U - 1)] for _ in range(B)]

        fn_numba = ChunkedAlignerLossNumba(blank=blank, chunk_size=chunk_size, reduction='none')
        numba_cost, _ = wrap_and_call(fn_numba, acts, labels, device)

        act_lens = torch.LongTensor([T] * B)
        label_lens = torch.LongTensor([len(l) for l in labels])
        bf_logprob = chunked_aligner_loss_bruteforce(
            acts, torch.LongTensor(labels), act_lens, label_lens, blank=blank, chunk_size=chunk_size
        )
        bf_cost = (-bf_logprob).numpy()

        assert np.allclose(numba_cost, bf_cost, atol=1e-4, rtol=1e-4), "chunked-aligner vs brute-force mismatch."


class TestChunkedAlignerNarLoss:
    """Non-autoregressive (NAR) Chunked-Aligner loss.

    The NAR loss consumes per-frame logits ``(B, T, V+1)`` (no joint / no ``U``
    axis). It must be identical to the AR loss when the AR joint tensor is a
    "trivial join" -- the NAR logits broadcast over the ``U`` axis. We assert that
    equality for both the cost AND the gradient. Because both losses are fed a
    *view* of the same leaf tensor, the AR backward accumulates into the NAR
    leaf's ``.grad`` (summing over the broadcast ``U`` axis), so the two gradients
    can be compared element-wise directly.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize('chunk_size', [2, 3, 4])
    def test_nar_matches_ar_trivial_join(self, chunk_size):
        random.seed(0)
        torch.manual_seed(0)

        B, T, U, V = 4, 8, 4, 8  # V = number of non-blank labels; blank == V
        blank = V

        act_lens = torch.tensor([T, T - 1, T - 3, T], dtype=torch.int64)
        label_lens = torch.tensor([U, U - 1, 2, 1], dtype=torch.int64)
        labels = torch.zeros((B, U), dtype=torch.int64)
        for b in range(B):
            for u in range(int(label_lens[b])):
                labels[b, u] = random.randrange(0, V)

        # Single leaf -> both losses differentiate the same tensor.
        nar_acts = torch.rand([B, T, V + 1], dtype=torch.double, requires_grad=True)

        nar_loss_fn = ChunkedAlignerNarLossPytorch(blank=blank, chunk_size=chunk_size, reduction='sum')
        ar_loss_fn = ChunkedAlignerLossPytorch(blank=blank, chunk_size=chunk_size, reduction='sum')

        # ---- NAR path ----
        nar_cost = nar_loss_fn(nar_acts, labels, act_lens, label_lens)
        nar_cost.backward()
        nar_grad = nar_acts.grad.detach().clone()

        # ---- AR path on the "trivially joined" view of the same leaf ----
        nar_acts.grad = None
        ar_acts = nar_acts.unsqueeze(2).expand(B, T, U + 1, V + 1)  # broadcast over U
        ar_cost = ar_loss_fn(ar_acts, labels, act_lens, label_lens)
        ar_cost.backward()
        ar_grad = nar_acts.grad.detach().clone()  # AR backward summed over the U axis

        assert torch.allclose(nar_cost, ar_cost, atol=1e-6, rtol=1e-5), "NAR vs AR (trivial join) cost mismatch."
        assert torch.allclose(nar_grad, ar_grad, atol=1e-6, rtol=1e-5), "NAR vs AR (trivial join) gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('chunk_size', [2, 3])
    def test_nar_forward_matches_bruteforce(self, chunk_size):
        random.seed(1)
        torch.manual_seed(1)

        B, T, U, V = 2, 6, 3, 5  # small enough to brute-force all chunk segmentations
        blank = V

        act_lens = torch.tensor([T, T], dtype=torch.int64)
        label_lens = torch.tensor([U, U - 1], dtype=torch.int64)
        labels = torch.zeros((B, U), dtype=torch.int64)
        for b in range(B):
            for u in range(int(label_lens[b])):
                labels[b, u] = random.randrange(0, V)

        acts = torch.rand([B, T, V + 1], dtype=torch.double)

        nar_loss_fn = ChunkedAlignerNarLossPytorch(blank=blank, chunk_size=chunk_size, reduction='none')
        nar_cost = nar_loss_fn(acts, labels, act_lens, label_lens)

        bf_logprob = chunked_aligner_nar_loss_bruteforce(
            acts, labels, act_lens, label_lens, blank=blank, chunk_size=chunk_size
        )
        bf_cost = -bf_logprob.double()

        assert torch.allclose(nar_cost, bf_cost, atol=1e-6, rtol=1e-5), "NAR vs brute-force mismatch."


class TestChunkedAlignerNarLossNumba:
    """CUDA / Numba non-autoregressive (NAR) Chunked-Aligner loss.

    Validates the NAR numba kernels (acts ``[B, T, V]``, no joint) against the
    autograd PyTorch reference (cost + gradient), against brute-force path
    enumeration, and -- the key cross-check -- against the AR numba loss fed a
    "trivial join" (the NAR log-probs broadcast over the ``U`` axis), for both the
    cost and the (u-summed) gradient.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize('device', CUDA_ONLY_DEVICE)
    @pytest.mark.parametrize('chunk_size', [2, 3, 4])
    def test_nar_numba_matches_pytorch(self, device, chunk_size):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        random.seed(0)
        torch.manual_seed(0)

        B, T, U, V = 4, 8, 4, 8  # V = number of non-blank labels; blank == V
        blank = V

        acts = torch.rand([B, T, V + 1])  # [B, T, V+1] -- no U axis
        labels = [[random.randrange(0, V) for _ in range(U - 1)] for _ in range(B)]

        fn_numba = ChunkedAlignerNarLossNumba(blank=blank, chunk_size=chunk_size, reduction='sum')
        numba_cost, numba_grads = wrap_and_call(fn_numba, acts, labels, device)

        fn_ag = ChunkedAlignerNarLossPytorch(blank=blank, chunk_size=chunk_size, reduction='sum')
        ag_cost, ag_grads = wrap_and_call(fn_ag, acts, labels, device)

        assert np.allclose(numba_cost, ag_cost, atol=1e-4, rtol=1e-5), "NAR chunked-aligner costs mismatch."
        assert np.allclose(numba_grads, ag_grads, atol=1e-4, rtol=1e-3), "NAR chunked-aligner gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', CUDA_ONLY_DEVICE)
    def test_nar_numba_variable_lengths(self, device):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        random.seed(7)
        torch.manual_seed(7)

        B, T, U, V = 3, 10, 5, 6  # T = 10 -> partial last chunk
        blank = V
        chunk_size = 4

        act_lens = torch.tensor([10, 7, 9], dtype=torch.int64, device=device)
        label_lens = torch.tensor([4, 2, 3], dtype=torch.int64, device=device)
        labels = torch.zeros((B, U - 1), dtype=torch.int64, device=device)
        for b in range(B):
            for u in range(int(label_lens[b])):
                labels[b, u] = random.randrange(0, V)

        acts = torch.rand([B, T, V + 1], device=device)

        def run(loss_fn):
            a = acts.clone().detach().requires_grad_(True)
            cost = loss_fn(a, labels, act_lens, label_lens)
            torch.sum(cost).backward()
            torch.cuda.synchronize()
            return cost.detach().cpu().numpy(), a.grad.detach().cpu().numpy()

        numba_cost, numba_grads = run(ChunkedAlignerNarLossNumba(blank=blank, chunk_size=chunk_size, reduction='sum'))
        ag_cost, ag_grads = run(ChunkedAlignerNarLossPytorch(blank=blank, chunk_size=chunk_size, reduction='sum'))

        assert np.allclose(numba_cost, ag_cost, atol=1e-4, rtol=1e-5), "NAR chunked-aligner costs mismatch."
        assert np.allclose(numba_grads, ag_grads, atol=1e-4, rtol=1e-3), "NAR chunked-aligner gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', CUDA_ONLY_DEVICE)
    @pytest.mark.parametrize('chunk_size', [2, 3])
    def test_nar_numba_matches_bruteforce(self, device, chunk_size):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        random.seed(1)
        torch.manual_seed(1)

        B, T, U, V = 2, 6, 3, 5
        blank = V

        acts = torch.rand([B, T, V + 1])
        labels = [[random.randrange(0, V) for _ in range(U - 1)] for _ in range(B)]

        fn_numba = ChunkedAlignerNarLossNumba(blank=blank, chunk_size=chunk_size, reduction='none')
        numba_cost, _ = wrap_and_call(fn_numba, acts, labels, device)

        act_lens = torch.tensor([T] * B, dtype=torch.int64)
        label_lens = torch.tensor([len(l) for l in labels], dtype=torch.int64)
        bf_logprob = chunked_aligner_nar_loss_bruteforce(
            acts, torch.tensor(labels, dtype=torch.int64), act_lens, label_lens, blank=blank, chunk_size=chunk_size
        )
        bf_cost = (-bf_logprob).numpy()

        assert np.allclose(numba_cost, bf_cost, atol=1e-4, rtol=1e-4), "NAR chunked-aligner vs brute-force mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', CUDA_ONLY_DEVICE)
    @pytest.mark.parametrize('chunk_size', [2, 3, 4])
    def test_nar_numba_matches_ar_numba_trivial_join(self, device, chunk_size):
        """NAR numba loss == AR numba loss on a "trivial join" of the NAR log-probs.

        Broadcasting the per-frame NAR activations over the ``U`` axis turns them
        into a valid AR joint tensor whose loss must match the NAR loss; the AR
        gradient summed over ``U`` must match the NAR gradient.
        """
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        random.seed(3)
        torch.manual_seed(3)

        B, T, U, V = 4, 9, 4, 7  # V = number of non-blank labels; blank == V
        blank = V

        act_lens = torch.tensor([9, 8, 6, 9], dtype=torch.int64, device=device)
        label_lens = torch.tensor([U - 1, 2, 2, 1], dtype=torch.int64, device=device)
        labels = torch.zeros((B, U - 1), dtype=torch.int64, device=device)
        for b in range(B):
            for u in range(int(label_lens[b])):
                labels[b, u] = random.randrange(0, V)

        nar_acts = torch.rand([B, T, V + 1], device=device)

        # NAR numba path.
        a_nar = nar_acts.clone().detach().requires_grad_(True)
        nar_cost = ChunkedAlignerNarLossNumba(blank=blank, chunk_size=chunk_size, reduction='sum')(
            a_nar, labels, act_lens, label_lens
        )
        nar_cost.backward()
        torch.cuda.synchronize()
        nar_grad = a_nar.grad.detach().cpu().numpy()

        # AR numba path on the trivially-joined (broadcast) activations.
        maxU = U
        a_ar = nar_acts.detach().unsqueeze(2).expand(B, T, maxU, V + 1).contiguous().requires_grad_(True)
        ar_cost = ChunkedAlignerLossNumba(blank=blank, chunk_size=chunk_size, reduction='sum')(
            a_ar, labels, act_lens, label_lens
        )
        ar_cost.backward()
        torch.cuda.synchronize()
        ar_grad_summed = a_ar.grad.sum(dim=2).detach().cpu().numpy()  # sum over the U axis

        assert np.allclose(
            nar_cost.detach().cpu().numpy(), ar_cost.detach().cpu().numpy(), atol=1e-4, rtol=1e-5
        ), "NAR vs AR (trivial join) numba cost mismatch."
        assert np.allclose(nar_grad, ar_grad_summed, atol=1e-4, rtol=1e-3), "NAR vs AR (trivial join) numba grad mismatch."


if __name__ == "__main__":
    pytest.main([__file__])
