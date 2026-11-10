# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""ScaledAdam's heavy-clipping diagnostics must never kill a training run.

`_show_param_with_unusual_grad` fires only when the clipping scale drops below
0.5 -- i.e. exactly when a run is already in trouble and the log is most needed.
It sorted a list of ``(ratio, name, tensor)`` tuples WHOLE, so any tie on both
ratio and name fell through to comparing two multi-element tensors and raised
``RuntimeError: Boolean value of Tensor with more than one value is ambiguous``.
That took down job 21645690 (zipformer + ScaledAdam) 16 minutes in.

Ties are not exotic: a batched parameter group whose grads are all zero gives
every entry ratio 0.0, and icefall's batching repeats names across shapes.
"""

import pytest
import torch

from nemo.collections.asr.parts.submodules.zipformer.optim import ScaledAdam


def _opt_with_tuples(ratios, names):
    """Build a ScaledAdam plus the (p, state, names) tuples its diagnostic consumes."""
    n = len(ratios)
    p = torch.nn.Parameter(torch.zeros(n, 4))
    opt = ScaledAdam([p], lr=0.045)
    # grad**2 / exp_avg_sq, meaned over dims 1.. -> sqrt -> the ratio per row.
    p.grad = torch.tensor(ratios, dtype=torch.float32).unsqueeze(1).expand(n, 4).contiguous()
    state = {"exp_avg_sq": torch.ones(n, 4)}
    return opt, [(p, state, list(names))]


def test_tied_ratio_and_name_does_not_raise():
    """The exact shape of the production crash: identical ratio AND identical name."""
    opt, tuples = _opt_with_tuples([0.0, 0.0, 0.0], ["dup", "dup", "dup"])
    opt._show_param_with_unusual_grad(tuples)  # must not raise


def test_distinct_ratios_still_ordered_largest_first():
    opt, tuples = _opt_with_tuples([1.0, 9.0, 3.0], ["a", "b", "c"])
    captured = {}

    def fake_warning(msg):
        captured["msg"] = msg

    import nemo.collections.asr.parts.submodules.zipformer.optim as mod

    orig = mod.logging.warning
    mod.logging.warning = fake_warning
    try:
        opt._show_param_with_unusual_grad(tuples)
    finally:
        mod.logging.warning = orig

    msg = captured["msg"]
    # 'b' carries the largest ratio, so it must be reported before 'a' and 'c'.
    assert msg.index("'b'") < msg.index("'c'") < msg.index("'a'")


def test_nan_ratio_does_not_raise():
    opt, tuples = _opt_with_tuples([float("nan"), 1.0, float("nan")], ["a", "b", "c"])
    opt._show_param_with_unusual_grad(tuples)
