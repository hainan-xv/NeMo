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

"""SqueezeExcite's lazy fp32 cast breaks optimizer-state resume under bf16-true.

``SqueezeExcite.forward_for_export`` flips its own ``fc`` parameters to float32
the first time it runs. Under ``bf16-true`` that means a parameter's dtype
changes DURING training -- after the optimizer state has already been restored
and cast to the (then bf16) parameter dtype. The next step combines an fp32
gradient with a bf16 ``exp_avg`` and dies.

This is why the conv-encoder SCRIPT arm trained a full 4-hour leg and then failed
on every resume at ~5 minutes, while every Conformer arm resumed fine: only
Citrinet carries SqueezeExcite.
"""

import pytest
import torch

from nemo.collections.asr.parts.submodules.jasper import SqueezeExcite


def _se(channels=16, bf16=True):
    se = SqueezeExcite(channels, reduction_ratio=2, context_window=-1)
    return se.to(torch.bfloat16) if bf16 else se


def test_squeeze_excite_mutates_its_own_param_dtype_on_first_forward():
    """The upstream behaviour this fix exists to tame -- lazy, mid-training."""
    se = _se()
    assert se.fc[0].weight.dtype == torch.bfloat16
    x = torch.randn(2, 16, 20, dtype=torch.bfloat16)
    se(x, torch.tensor([20, 15]))
    assert se.fc[0].weight.dtype == torch.float32  # flipped by forward()


def test_resume_dtype_mismatch_reproduces_without_eager_cast():
    """Restore state against bf16 params, then let forward flip them -> boom."""
    se = _se()
    opt = torch.optim.AdamW(se.parameters(), lr=1e-4, foreach=False)

    # State as a checkpoint would restore it: cast to the params' dtype NOW (bf16).
    for p in se.parameters():
        opt.state[p] = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros_like(p),
            "exp_avg_sq": torch.zeros_like(p),
        }
    assert opt.state[se.fc[0].weight]["exp_avg"].dtype == torch.bfloat16

    se(torch.randn(2, 16, 20, dtype=torch.bfloat16), torch.tensor([20, 15]))
    for p in se.parameters():
        p.grad = torch.ones_like(p)

    with pytest.raises(RuntimeError, match="BFloat16|dtype"):
        opt.step()


def test_eager_cast_makes_resume_safe():
    """Casting BEFORE the optimizer is built keeps param and state dtypes aligned."""
    se = _se()
    se.fc.float()  # what _force_squeeze_excite_fp32 does

    opt = torch.optim.AdamW(se.parameters(), lr=1e-4, foreach=False)
    for p in se.parameters():
        opt.state[p] = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros_like(p),
            "exp_avg_sq": torch.zeros_like(p),
        }
    assert opt.state[se.fc[0].weight]["exp_avg"].dtype == torch.float32

    se(torch.randn(2, 16, 20, dtype=torch.bfloat16), torch.tensor([20, 15]))
    assert se.fc[0].weight.dtype == torch.float32  # forward changed nothing
    for p in se.parameters():
        p.grad = torch.ones_like(p)
    opt.step()  # must not raise


def test_helper_finds_and_converts_nested_squeeze_excites():
    from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([_se(), _se(), _se()])

        _force_squeeze_excite_fp32 = StreamingSTTModel._force_squeeze_excite_fp32

    h = Holder()
    assert all(b.fc[0].weight.dtype == torch.bfloat16 for b in h.blocks)
    h._force_squeeze_excite_fp32()
    assert all(b.fc[0].weight.dtype == torch.float32 for b in h.blocks)


def test_helper_is_idempotent_and_noop_when_already_fp32():
    from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.se = _se(bf16=False)

        _force_squeeze_excite_fp32 = StreamingSTTModel._force_squeeze_excite_fp32

    h = Holder()
    h._force_squeeze_excite_fp32()
    h._force_squeeze_excite_fp32()
    assert h.se.fc[0].weight.dtype == torch.float32
