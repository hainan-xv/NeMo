# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.losses.rnnt import MultiblankRNNTLossPytorch, RNNTLossPytorch, TDTLossPytorch
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_numpy import RNNTLoss as RNNTLoss_Numpy
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import (
    MultiblankRNNTLossNumba,
    RNNTLossNumba,
    TDTLossNumba,
)
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__

DEVICES = ['cpu']

if torch.cuda.is_available():
    DEVICES.append('cuda')

CUDA_ONLY_DEVICE = ['cuda']

DTYPES = [np.float32]
if numba_utils.is_numba_cuda_fp16_supported():
    DTYPES.append(np.float16)


def wrap_and_call(fn, acts, labels, device):
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


class TestRNNTLoss:
    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_randomized_act_label(self, device):
        if device == 'cuda':
#            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

            B, T, U, V = 1, 15, 8, 4  # here V is number of non blank labels
            B, T, U, V = 1, 5, 3, 3  # here V is number of non blank labels
            sigma = 1

            is_terminal = [1.0 for i in range(V)]
#            is_terminal = [random.randint(0, 1) for _ in range(V)]
#            is_terminal = [0.0 for i in range(V)]

            acts = torch.rand([B, T, U, V + 1])
            labels = [[random.randrange(0, V) for i in range(U - 1)] for j in range(B)]

            is_terminal[labels[0][-1]] = 1

            fn_pt = RNNTLossNumba(blank=V, reduction='sum', sigma=sigma, is_terminal=is_terminal)
            pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

            pt_token_grads = pt_grads[:,:,:,:]

            fn_ag = RNNTLossPytorch(
                blank=V, reduction='sum', sigma=sigma, is_terminal=is_terminal
            )  # ag for automatic gradient computation
            ag_cost, ag_grads = wrap_and_call(fn_ag, acts, labels, device)

            ag_token_grads = ag_grads

            print("TWO losses", pt_cost, ag_cost)
            print("ISTERMINAL", is_terminal)
            print("GRAD IDFF", pt_token_grads - ag_token_grads)
            assert np.allclose(pt_cost, ag_cost, rtol=1e-6), "tdt costs mismatch."
            assert np.allclose(pt_token_grads, ag_token_grads, rtol=1e-2), "td token gradient mismatch."


if __name__ == "__main__":
    pytest.main([__file__])
