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

from nemo.collections.asr.losses.rnnt import (
    MultiblankRNNTLossPytorch,
    RNNTLossPytorch,
    TDTLossPytorch,
    WordawareMultiblankRNNTLossPytorch,
)
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_numpy import RNNTLoss as RNNTLoss_Numpy
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import (
    MultiblankRNNTLossNumba,
    RNNTLossNumba,
    TDTLossNumba,
    WordawareMultiblankRNNTLossNumba,
)
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__

DEVICES = ['cpu']

if torch.cuda.is_available():
    DEVICES.append('cuda')

DEVICES = ['cuda']


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


# class TestTDTLoss:
#    @pytest.mark.unit
#    @pytest.mark.parametrize('device', DEVICES)
#    def test_case_randomized_act_label(self, device):
#        if device == 'cuda':
##            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)
#
#            B, T, U, V = 1, 3, 2, 2  # here V is number of non blank labels
#            durations = [0, 1, 2]
#            sigma = 0.05
#
#            for t in range(25):
#                acts = torch.rand([B, T, U, V + 1 + len(durations)])
#                labels = [[random.randrange(0, V) for i in range(U - 1)] for j in range(B)]
##            labels[:,-1] = 5
#                for i in range(len(labels)):
#                    labels[i][-1] = 1
#
#                fn_pt = WordawareTDTLossNumba(blank=V, reduction='sum', durations=durations, sigma=sigma, vocab_file='/home/hainanx/nemo_exps_local/tokenizer_reversed/tokenizer_spe_bpe_v256/tokenizer.vocab')
#                pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)
#
#                fn_ag = WordawareTDTLossPytorch(
#                    blank=V, reduction='sum', durations=durations, sigma=sigma, vocab_file='/home/hainanx/nemo_exps_local/tokenizer_reversed/tokenizer_spe_bpe_v256/tokenizer.vocab'
#                )  # ag for automatic gradient computation
#                ag_cost, ag_grads = wrap_and_call(fn_ag, acts, labels, device)
#
##                print("two losses", pt_cost.item(), ag_cost)
##                print("two grads")
##                print("a", pt_grads)
##                print("b", ag_grads)
##                print("")
##            assert np.allclose(pt_cost, ag_cost, rtol=1e-6), "tdt costs mismatch."
#                assert np.allclose(pt_grads, ag_grads, rtol=1e-2), "td gradient mismatch."
#
#            assert(False)


class TestMultiblankRNNTLoss:
    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_randomized_act_label(self, device):
        if device == 'cuda':
            #            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

            B, T, U, V = 4, 8, 4, 8  # here V is number of non blank labels
            big_blank_durations = [2, 4, 8]
            sigma = 0.1

            acts = torch.rand([B, T, U, V + 1 + len(big_blank_durations)])
            labels = [[random.randrange(0, V) for i in range(U - 1)] for j in range(B)]

            for i in range(len(labels)):
                labels[i][-1] = 1

            fn_pt = WordawareMultiblankRNNTLossNumba(
                blank=V + len(big_blank_durations),
                reduction='sum',
                big_blank_durations=big_blank_durations,
                sigma=sigma,
                vocab_file='/home/hainanx/nemo_exps_local/tokenizer_reversed/tokenizer_spe_bpe_v256/tokenizer.vocab',
            )
            pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

            fn_ag = WordawareMultiblankRNNTLossPytorch(
                blank=V + len(big_blank_durations),
                reduction='sum',
                big_blank_durations=big_blank_durations,
                sigma=sigma,
                vocab_file='/home/hainanx/nemo_exps_local/tokenizer_reversed/tokenizer_spe_bpe_v256/tokenizer.vocab',
            )  # ag for automatic gradient computation
            ag_cost, ag_grads = wrap_and_call(fn_ag, acts, labels, device)

            assert np.allclose(pt_cost, ag_cost, rtol=1e-6), "multi-blank costs mismatch."
            assert np.allclose(pt_grads, ag_grads, rtol=1e-2), "multi-blank gradient mismatch."


# class TestTDTLoss:
#    @pytest.mark.unit
#    @pytest.mark.parametrize('device', DEVICES)
#    def test_case_randomized_act_label(self, device):
#        if device == 'cuda':
#            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)
#
#            B, T, U, V = 4, 8, 4, 8  # here V is number of non blank labels
#            durations = [0, 1, 2, 3, 4, 5]
#            sigma = 0.05
#
#            acts = torch.rand([B, T, U, V + 1 + len(durations)])
#            labels = [[random.randrange(0, V) for i in range(U - 1)] for j in range(B)]
#
#            fn_pt = TDTLossNumba(blank=V, reduction='sum', durations=durations, sigma=sigma)
#            pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)
#
#            fn_ag = TDTLossPytorch(
#                blank=V, reduction='sum', durations=durations, sigma=sigma
#            )  # ag for automatic gradient computation
#            ag_cost, ag_grads = wrap_and_call(fn_ag, acts, labels, device)
#
#            assert np.allclose(pt_cost, ag_cost, rtol=1e-6), "tdt costs mismatch."
#            assert np.allclose(pt_grads, ag_grads, rtol=1e-2), "td gradient mismatch."
#
#    @pytest.mark.unit
#    @pytest.mark.parametrize('device', DEVICES)
#    def test_case_fixed_case_act_label(self, device):
#        if device == 'cuda':
#            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)
#
#            B, T, U, V = 1, 3, 2, 3  # here V is number of non blank labels
#            durations = [0, 1, 2]
#            sigma = 0.05
#
#            acts = torch.zeros([B, T, U, V + 1 + len(durations)])
#            labels = [[(i + j) % (V - 1) for i in range(U - 1)] for j in range(B)]
#
#            fn_pt = TDTLossNumba(blank=V, reduction='sum', durations=durations, sigma=sigma)
#            pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)
#
#            expected_cost = 4.155739
#            expected_grads = [
#                [
#                    [
#                        [-0.64962804, 0.25, 0.25, 0.14962798, 0.2672583, -0.16792619, -0.09933221],
#                        [0.01651875, 0.01651875, 0.01651875, -0.04955626, 0.022025, -0.01227201, -0.009753],
#                    ],
#                    [
#                        [-0.04892651, 0.01714851, 0.01714851, 0.01462949, -0.01143234, -0.01143234, 0.02286467],
#                        [0.12531489, 0.12531489, 0.12531489, -0.37594467, 0.16708651, 0.13027048, -0.29735702],
#                    ],
#                    [
#                        [-0.02572276, 0.00857425, 0.00857425, 0.00857425, -0.02286468, 0.01143234, 0.01143234],
#                        [0.13388914, 0.13388914, 0.13388914, -0.40166742, 0.17851885, -0.35703772, 0.17851885],
#                    ],
#                ]
#            ]
#
#            assert np.allclose(pt_cost, expected_cost, rtol=1e-6), "tdt costs mismatch."
#            assert np.allclose(pt_grads, expected_grads, rtol=1e-2), "td gradient mismatch."


if __name__ == "__main__":
    pytest.main([__file__])
