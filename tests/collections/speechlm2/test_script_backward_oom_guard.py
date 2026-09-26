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
"""Tests for the BACKWARD-pass CUDA OOM guard in ScriptSTTModel.

Lightning runs backward after training_step returns, inside the optimizer
closure, so the forward guard cannot see an OOM there. Uncaught, it kills one
rank and the peers block in the gradient collective until Slurm SIGKILLs the
step -- observed twice, at chunk_size=2 and chunk_size=7, the latter missing by
80 MB on a 79 GiB card once in ~8,280 steps.

This guard was once reverted for having zero catches and causing an NCCL
deadlock. The deadlock was a bug in _allreduce_grads (see the rank-invariance
test), not in the idea; both are covered here.
"""

import pytest
import torch

from nemo.collections.speechlm2.models.script_model import ScriptSTTModel


class _Cfg:
    oom_skip_limit = 3


class _Harness(ScriptSTTModel):
    """Drives the guard without building a Qwen LLM and a Conformer encoder."""

    def __init__(self, oom_in_backward: bool):
        torch.nn.Module.__init__(self)
        self.core_cfg = _Cfg()
        self._oom_in_backward = oom_in_backward
        self.optimizer_steps = 0
        self.logged = {}

    def parameters(self, recurse: bool = True):
        return iter([])

    def zero_grad(self, set_to_none=True):
        pass

    def log(self, name, value, **kw):
        self.logged[name] = value

    def _ddp_module(self):
        return None  # single process: no_sync path not exercised

    def _any_rank_oom(self, local):
        return local  # world size 1

    def _allreduce_grads(self):
        pass  # stubbed; the real one is exercised directly in the test below


def _make(oom_in_backward, monkeypatch):
    h = _Harness(oom_in_backward)

    def fake_super_backward(*a, **k):
        if h._oom_in_backward:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.27 GiB")

    def fake_super_optimizer_step(*a, **k):
        h.optimizer_steps += 1

    parent = ScriptSTTModel.__mro__[1]
    monkeypatch.setattr(parent, "backward", fake_super_backward, raising=False)
    monkeypatch.setattr(parent, "optimizer_step", fake_super_optimizer_step, raising=False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return h


@pytest.mark.unit
def test_backward_oom_is_caught_and_step_is_skipped(monkeypatch):
    """THE REGRESSION: an OOM in backward must not escape and kill the rank."""
    h = _make(True, monkeypatch)
    h.backward()  # must not raise
    assert h._skip_optimizer_step is True
    h.optimizer_step()
    assert h.optimizer_steps == 0, "optimizer ran on a batch whose backward OOMed"
    assert h.logged.get("train_batches_skipped_oom") == 1.0


@pytest.mark.unit
def test_healthy_backward_still_updates(monkeypatch):
    """Positive control: without an OOM the guard must be transparent.

    Without this, a guard that skipped EVERY step would pass the test above.
    """
    h = _make(False, monkeypatch)
    h.backward()
    assert h._skip_optimizer_step is False
    h.optimizer_step()
    assert h.optimizer_steps == 1


@pytest.mark.unit
def test_consecutive_backward_ooms_eventually_raise(monkeypatch):
    """The safety limit must fire for BACKWARD OOMs too.

    The streak used to reset at the end of a successful FORWARD; since a backward
    OOM is always preceded by one, that made the limit unreachable and the job
    would skip every step forever while logging a zero loss.
    """
    h = _make(True, monkeypatch)
    for _ in range(_Cfg.oom_skip_limit - 1):
        h.backward()
        h.optimizer_step()
    with pytest.raises(RuntimeError, match="consecutive OOM batches"):
        h.backward()


@pytest.mark.unit
def test_completed_update_clears_the_streak(monkeypatch):
    """A real update must break the streak, or transient OOMs would accumulate."""
    h = _make(True, monkeypatch)
    h.backward()
    h.optimizer_step()
    assert h._oom_streak == 1
    h._oom_in_backward = False
    h.backward()
    h.optimizer_step()
    assert h._oom_streak == 0


@pytest.mark.unit
def test_allreduce_buffer_shape_is_rank_invariant(monkeypatch):
    """THE DEADLOCK REGRESSION: the flat buffer must not depend on which
    parameters a rank's batch exercised.

    _allreduce_grads used to select `prm.grad is not None`, so a rank whose batch
    left a parameter untouched built a SMALLER buffer than its peers. They then
    all-reduced mismatched shapes and NCCL blocked until the watchdog fired --
    "collective operation timeout ... OpType=ALLREDUCE" at step 510 -- with no
    Python error anywhere. That killed an 8-node run and caused this guard to be
    reverted once.
    """
    calls = []

    class _FakeDist:
        is_available = staticmethod(lambda: True)
        is_initialized = staticmethod(lambda: True)
        get_world_size = staticmethod(lambda: 2)
        all_reduce = staticmethod(lambda t, *a, **k: calls.append(tuple(t.shape)))

    def _run(populate):
        del calls[:]
        h = _Harness(oom_in_backward=False)
        params = {n: torch.nn.Parameter(torch.ones(4)) for n in ("a", "b", "c")}
        for n, prm in params.items():
            if n in populate:
                prm.grad = torch.ones_like(prm)
        h.parameters = lambda recurse=True: iter(params.values())
        saved = torch.distributed
        torch.distributed = _FakeDist
        try:
            # The REAL method, not _Harness's no-op stub.
            ScriptSTTModel._allreduce_grads(h)
        finally:
            torch.distributed = saved
        return list(calls), params

    all_shapes, _ = _run({"a", "b", "c"})
    some_shapes, some_params = _run({"b"})
    assert all_shapes, "no all-reduce was issued"
    assert (
        all_shapes == some_shapes
    ), f"buffer shape depends on which grads exist: {all_shapes} vs {some_shapes} -- ranks would deadlock"
    assert all(p.grad is not None for p in some_params.values())
    assert torch.equal(some_params["a"].grad, torch.zeros(4))
    assert torch.allclose(some_params["b"].grad, torch.full((4,), 0.5))
