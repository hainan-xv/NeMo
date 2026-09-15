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
"""``init_from_ckpt`` is a warm start, and its dangerous failure mode is silence.

A no-op load looks identical to a successful one from the outside -- the job
runs, the loss falls, and only the step-for-step comparison against the donor
arm ever reveals that it trained from scratch. So the contract under test is as
much about REFUSING as about loading.
"""

import importlib.util
import pathlib

import pytest
import torch
from torch import nn

_SPEC = importlib.util.spec_from_file_location(
    "script_train_mod",
    pathlib.Path(__file__).parents[3] / "examples" / "speechlm2" / "script_train.py",
)


def _load_fn():
    # script_train.py calls torch.cuda.set_device at import time, which needs a
    # GPU and a LOCAL_RANK. Pull the function out without executing the module.
    src = _SPEC.origin
    ns = {"torch": torch}
    text = pathlib.Path(src).read_text()
    start = text.index("def _init_from_ckpt")
    end = text.index("@hydra_runner")

    class _Log:
        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

    ns["logging"] = _Log()
    exec(text[start:end], ns)
    return ns["_init_from_ckpt"]


_init_from_ckpt = _load_fn()


class _Model(nn.Module):
    def __init__(self, head_out=4):
        super().__init__()
        self.trunk = nn.Linear(8, 8)
        self.head = nn.Linear(8, head_out)


def _write_ckpt(tmp_path, model, key="state_dict"):
    path = tmp_path / "donor.ckpt"
    torch.save({key: model.state_dict()}, path)
    return str(path)


def test_loads_matching_weights(tmp_path):
    donor, target = _Model(), _Model()
    with torch.no_grad():
        donor.trunk.weight.fill_(0.5)
    _init_from_ckpt(target, _write_ckpt(tmp_path, donor))
    assert torch.allclose(target.trunk.weight, donor.trunk.weight)
    assert torch.allclose(target.head.weight, donor.head.weight)


def test_mismatched_head_is_skipped_but_trunk_still_loads(tmp_path):
    """The sibling-arm case: shared trunk, different head width."""
    donor = _Model(head_out=4)
    target = _Model(head_out=7)
    with torch.no_grad():
        donor.trunk.weight.fill_(0.25)
    before = target.head.weight.clone()

    _init_from_ckpt(target, _write_ckpt(tmp_path, donor))

    assert torch.allclose(target.trunk.weight, donor.trunk.weight)
    assert torch.allclose(target.head.weight, before), "mismatched head must be left at init"


def test_zero_matches_raises_rather_than_silently_training_from_scratch(tmp_path):
    class _Unrelated(nn.Module):
        def __init__(self):
            super().__init__()
            self.something_else = nn.Linear(3, 3)

    path = _write_ckpt(tmp_path, _Unrelated())
    with pytest.raises(ValueError, match="matched ZERO parameters"):
        _init_from_ckpt(_Model(), path)


def test_accepts_raw_state_dict_without_lightning_wrapper(tmp_path):
    donor, target = _Model(), _Model()
    with torch.no_grad():
        donor.trunk.bias.fill_(1.5)
    path = tmp_path / "raw.ckpt"
    torch.save(donor.state_dict(), path)
    _init_from_ckpt(target, str(path))
    assert torch.allclose(target.trunk.bias, donor.trunk.bias)
