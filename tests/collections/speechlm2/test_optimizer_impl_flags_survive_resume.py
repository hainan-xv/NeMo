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

"""Optimizer implementation flags must survive a checkpoint restore.

torch's ``Optimizer.load_state_dict`` takes ``param_groups`` from the CHECKPOINT
and keeps only the live ``params`` list, so a flag changed in the config is
reverted on every resume. The conv-encoder arm hit this head-on: it mixes 46 fp32
parameters into an otherwise bf16 model, foreach AdamW groups state by
(device, dtype), and resuming died in ``_group_tensors_by_device_and_dtype``.
Setting ``foreach: false`` had no effect because the checkpoint said True.
"""

import pytest
import torch

from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel


def _opt_with(foreach):
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = torch.ones(3)
    opt = torch.optim.AdamW([p], lr=1e-4, foreach=foreach)
    opt.step()
    return opt


def test_torch_reverts_foreach_on_load_state_dict():
    """Documents the upstream behaviour the hook exists to counteract."""
    saved = _opt_with(True).state_dict()
    fresh = _opt_with(False)
    assert fresh.param_groups[0]["foreach"] is False
    fresh.load_state_dict(saved)
    # NOT False any more -- this is exactly the silent revert.
    assert fresh.param_groups[0]["foreach"] is True


class _Trainer:
    def __init__(self, optimizers):
        self.optimizers = optimizers


class _Model:
    """Minimal stand-in carrying only the hook and the config it reads."""

    on_train_start = StreamingSTTModel.on_train_start

    def __init__(self, cfg, optimizers):
        self.cfg = cfg
        self.trainer = _Trainer(optimizers)


def test_hook_reasserts_foreach_after_restore():
    opt = _opt_with(False)
    opt.load_state_dict(_opt_with(True).state_dict())
    assert opt.param_groups[0]["foreach"] is True  # clobbered

    _Model({"optimizer": {"foreach": False}}, [opt]).on_train_start()
    assert opt.param_groups[0]["foreach"] is False


def test_hook_does_not_touch_lr():
    """The LR schedule owns lr; re-applying the config's base LR would reset it."""
    opt = _opt_with(False)
    opt.param_groups[0]["lr"] = 3.7e-5  # as a scheduler would have left it
    _Model({"optimizer": {"foreach": False, "lr": 1e-4}}, [opt]).on_train_start()
    assert opt.param_groups[0]["lr"] == pytest.approx(3.7e-5)
    assert opt.param_groups[0]["foreach"] is False


def test_hook_is_a_noop_without_impl_keys():
    opt = _opt_with(True)
    _Model({"optimizer": {"lr": 1e-4, "weight_decay": 1e-3}}, [opt]).on_train_start()
    assert opt.param_groups[0]["foreach"] is True


def test_hook_is_a_noop_without_optimizer_config():
    opt = _opt_with(True)
    _Model({}, [opt]).on_train_start()
    assert opt.param_groups[0]["foreach"] is True
