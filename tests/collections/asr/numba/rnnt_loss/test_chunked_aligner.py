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
    chunked_aligner_loss_bruteforce,
)
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import ChunkedAlignerLossNumba
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


if __name__ == "__main__":
    pytest.main([__file__])
