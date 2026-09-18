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
"""The corrector's warm start must not write the SHARED encoder.

``ScriptCorrectorModel`` sets ``self.perception.encoder = self.chat.encoder``:
one module object under two names. A warm start from a donor SCRIPT arm carries
``perception.encoder.*`` keys whose shapes match perfectly, so a plain
``load_state_dict`` overwrites the frozen CHAT encoder in place -- the model
under verification stops being the model that was measured, with no error and no
log line. These tests pin the guard that holds those keys back.
"""

import importlib.util
import pathlib

import pytest
import torch
from torch import nn

_SRC = pathlib.Path(__file__).parents[3] / "examples" / "speechlm2" / "script_corrector_train.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("script_corrector_train", _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


class _Shared(nn.Module):
    """Minimal stand-in with the same aliasing as the real model."""

    def __init__(self):
        super().__init__()
        encoder = nn.Linear(4, 4, bias=False)
        self.chat = nn.Module()
        self.chat.encoder = encoder
        self.perception = nn.Module()
        self.perception.encoder = encoder  # SAME object, as in the real model
        self.perception.proj = nn.Linear(4, 4, bias=False)
        self.head = nn.Linear(4, 4, bias=False)


@pytest.fixture
def donor(tmp_path):
    """A donor checkpoint whose every tensor is 1.0, so any write is visible."""
    path = tmp_path / "donor.ckpt"
    sd = {
        "perception.encoder.weight": torch.ones(4, 4),
        "perception.proj.weight": torch.ones(4, 4),
        "head.weight": torch.ones(4, 4),
    }
    torch.save({"state_dict": sd}, path)
    return path


@pytest.mark.unit
def test_shared_encoder_is_not_overwritten(donor):
    m = _Shared()
    with torch.no_grad():
        m.perception.encoder.weight.fill_(0.5)
    mod._init_from_ckpt(m, str(donor))

    # The encoder keeps CHAT's weights...
    assert torch.allclose(m.perception.encoder.weight, torch.full((4, 4), 0.5))
    # ...and the alias really is the same tensor, not a lucky copy.
    assert m.chat.encoder.weight is m.perception.encoder.weight


@pytest.mark.unit
def test_non_shared_weights_do_load(donor):
    m = _Shared()
    mod._init_from_ckpt(m, str(donor))
    assert torch.allclose(m.perception.proj.weight, torch.ones(4, 4))
    assert torch.allclose(m.head.weight, torch.ones(4, 4))


@pytest.mark.unit
def test_encoder_only_checkpoint_raises_rather_than_no_op(tmp_path):
    """Holding every key back is a failed warm start, not a silent success."""
    path = tmp_path / "enc_only.ckpt"
    torch.save({"state_dict": {"perception.encoder.weight": torch.ones(4, 4)}}, path)
    with pytest.raises(ValueError, match="ZERO parameters"):
        mod._init_from_ckpt(_Shared(), str(path))


@pytest.mark.unit
def test_chat_prefixed_keys_are_held_back(tmp_path):
    path = tmp_path / "with_chat.ckpt"
    torch.save(
        {"state_dict": {"chat.encoder.weight": torch.ones(4, 4), "head.weight": torch.ones(4, 4)}},
        path,
    )
    m = _Shared()
    with torch.no_grad():
        m.chat.encoder.weight.fill_(0.25)
    mod._init_from_ckpt(m, str(path))
    assert torch.allclose(m.chat.encoder.weight, torch.full((4, 4), 0.25))
    assert torch.allclose(m.head.weight, torch.ones(4, 4))
