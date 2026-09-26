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
"""Tests for the FORWARD-pass CUDA OOM guard in ScriptSTTModel.

Forward OOMs are real and frequent at small chunk sizes -- 41 of them on the
multi-lookahead arm, 100% at chunk_size=2. training_step catches them, votes
across ranks so every rank takes the same branch, and drops the batch coherently.

A backward-pass guard also existed briefly and was removed; see the note in
script_model.py for why. These tests cover what remains.
"""

import pytest
import torch

from nemo.collections.speechlm2.models.script_model import ScriptSTTModel


class _Cfg:
    oom_skip_limit = 3


class _Harness(ScriptSTTModel):
    """Drives the guard without building a Qwen LLM and a Conformer encoder."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.core_cfg = _Cfg()
        self.logged = {}

    def parameters(self, recurse: bool = True):
        return iter([torch.nn.Parameter(torch.zeros(2))])

    def log(self, name, value, **kw):
        self.logged[name] = value

    def _any_rank_oom(self, local):
        return local  # world size 1


@pytest.mark.unit
def test_forward_oom_is_counted_and_batch_dropped(monkeypatch):
    """An OOM in the forward must yield a zero loss, not propagate."""
    h = _Harness()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    out = h._skip_oom_batch(batch_idx=7)
    assert "loss" in out
    assert float(out["loss"]) == 0.0
    assert h.logged.get("train_batches_skipped_oom") == 1.0


@pytest.mark.unit
def test_dropped_batch_touches_every_trainable_param(monkeypatch):
    """The zero loss must be graph-connected to EVERY trainable parameter.

    DDP aborts on parameters that received no gradient, so a detached zero would
    turn a survivable OOM into a crash on the very next reduction.
    """
    h = _Harness()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    loss = h._skip_oom_batch(batch_idx=0)["loss"]
    assert loss.requires_grad, "zero loss is detached; DDP would see unused params"


@pytest.mark.unit
def test_consecutive_forward_ooms_eventually_raise(monkeypatch):
    """The safety limit must fire, or the job would skip forever.

    A run of skipped batches logs a zero loss, which is indistinguishable from a
    model that has learned the task perfectly -- so it must fail loudly instead.
    """
    h = _Harness()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    for _ in range(_Cfg.oom_skip_limit - 1):
        h._skip_oom_batch(batch_idx=0)
    with pytest.raises(RuntimeError, match="consecutive OOM batches"):
        h._skip_oom_batch(batch_idx=0)
