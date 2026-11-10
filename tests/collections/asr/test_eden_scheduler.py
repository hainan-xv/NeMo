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

"""The Eden adapter must reproduce upstream's schedule, not merely resemble it.

Eden is being added because the ScaledAdam arm underperformed on CosineAnnealing.
If the adapter's curve quietly differs from icefall's, that experiment answers a
question nobody asked -- so the central test here compares against the VENDORED
upstream Eden driven through its own step_batch/step_epoch API.
"""

import pytest
import torch

from nemo.collections.asr.parts.submodules.zipformer.nemo_sched import Eden
from nemo.collections.asr.parts.submodules.zipformer.optim import Eden as UpstreamEden

BASE_LR = 0.045
LR_BATCHES = 5000.0
LR_EPOCHS = 6.0
STEPS_PER_EPOCH = 16000.0


def _opt():
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=BASE_LR)


def _ours(**kw):
    kw.setdefault("lr_batches", LR_BATCHES)
    kw.setdefault("lr_epochs", LR_EPOCHS)
    kw.setdefault("steps_per_epoch", STEPS_PER_EPOCH)
    return Eden(_opt(), **kw)


def test_matches_upstream_eden_at_epoch_boundaries():
    """At every exact epoch boundary the continuous epoch axis equals the discrete one."""
    ours = _ours(warmup_batches=0.0)
    theirs = UpstreamEden(_opt(), lr_batches=LR_BATCHES, lr_epochs=LR_EPOCHS, warmup_batches=0.0, warmup_start=0.5)

    for epoch in range(0, 5):
        batch = int(epoch * STEPS_PER_EPOCH)
        # Upstream: set both axes explicitly.
        theirs.step_epoch(epoch)
        theirs.step_batch(batch)
        # Ours: one step axis; last_epoch IS the batch count.
        ours.last_epoch = batch
        mine = ours.get_lr()[0]
        ref = theirs.get_lr()[0]
        assert mine == pytest.approx(ref, rel=1e-9), f"epoch {epoch}, batch {batch}: {mine} vs {ref}"


def test_matches_upstream_formula_including_warmup():
    """Formula identity, warmup included.

    Upstream is fed the SAME (fractional) epoch this adapter derives, so any
    difference here is a difference in the formula itself rather than the
    documented continuous-vs-discrete epoch departure.
    """
    ours = _ours(warmup_batches=500.0, warmup_start=0.5)
    theirs = UpstreamEden(_opt(), lr_batches=LR_BATCHES, lr_epochs=LR_EPOCHS, warmup_batches=500.0, warmup_start=0.5)
    for batch in [0, 1, 100, 250, 499, 500, 501, 1000, 50_000, 400_000]:
        theirs.step_epoch(batch / STEPS_PER_EPOCH)
        theirs.step_batch(batch)
        ours.last_epoch = batch
        assert ours.get_lr()[0] == pytest.approx(theirs.get_lr()[0], rel=1e-9), f"batch {batch}"


def test_warmup_starts_at_half_and_reaches_one():
    """Eden warms up from HALF the base LR, not from zero -- easy to get wrong."""
    s = _ours(warmup_batches=500.0, warmup_start=0.5)
    s.last_epoch = 0
    assert s.get_lr()[0] == pytest.approx(0.5 * BASE_LR, rel=1e-6)
    s.last_epoch = 500
    # At the end of warmup the warmup factor is exactly 1.0; BOTH decay factors
    # remain -- the epoch axis has already advanced to 500/steps_per_epoch, so
    # leaving it out of the expectation is wrong by ~7e-6 relative.
    batch_factor = ((500.0**2 + LR_BATCHES**2) / LR_BATCHES**2) ** -0.25
    epoch_factor = (((500.0 / STEPS_PER_EPOCH) ** 2 + LR_EPOCHS**2) / LR_EPOCHS**2) ** -0.25
    assert s.get_lr()[0] == pytest.approx(BASE_LR * batch_factor * epoch_factor, rel=1e-9)


def test_lr_decreases_monotonically_after_warmup():
    s = _ours(warmup_batches=500.0)
    prev = None
    for batch in range(500, 200_000, 2_000):
        s.last_epoch = batch
        lr = s.get_lr()[0]
        if prev is not None:
            assert lr < prev, f"LR rose at batch {batch}"
        prev = lr


def test_min_lr_is_a_floor():
    s = _ours(warmup_batches=0.0, min_lr=0.01)
    s.last_epoch = 10_000_000
    assert s.get_lr()[0] == pytest.approx(0.01)


def test_resume_reproduces_identical_lr():
    """These runs restart every 4h -- a resumed job must land on the same LR."""
    a = _ours()
    for _ in range(1234):
        a.step()
    lr_before = a.get_last_lr()[0]
    state = a.state_dict()

    b = _ours()
    b.load_state_dict(state)
    assert b.get_last_lr()[0] == pytest.approx(lr_before, rel=1e-12)
    # And continues on the same trajectory, not from step 0.
    a.step()
    b.step()
    assert b.get_last_lr()[0] == pytest.approx(a.get_last_lr()[0], rel=1e-12)


def test_registered_under_name_eden():
    import nemo.collections.asr.parts.submodules.zipformer.nemo_sched  # noqa: F401
    from nemo.core.optim.lr_scheduler import AVAILABLE_SCHEDULERS

    assert AVAILABLE_SCHEDULERS.get("eden") is Eden


@pytest.mark.parametrize(
    "kw",
    [
        {"lr_batches": 0},
        {"lr_epochs": -1},
        {"steps_per_epoch": 0},
        {"warmup_start": 1.5},
        {"epochs_over_run": 0},
    ],
)
def test_invalid_params_rejected(kw):
    with pytest.raises(ValueError):
        _ours(**kw)


def test_steps_per_epoch_derived_from_max_steps():
    """NeMo injects max_steps; Eden sizes its epoch axis from it automatically."""
    s = Eden(_opt(), lr_batches=LR_BATCHES, lr_epochs=LR_EPOCHS, max_steps=600_000, epochs_over_run=30.0)
    assert s.steps_per_epoch == pytest.approx(20_000.0)


def test_explicit_steps_per_epoch_wins_over_max_steps():
    s = Eden(_opt(), steps_per_epoch=16_000.0, max_steps=600_000, epochs_over_run=30.0)
    assert s.steps_per_epoch == pytest.approx(16_000.0)


def test_requires_steps_per_epoch_or_max_steps():
    with pytest.raises(ValueError, match="steps_per_epoch or a max_steps"):
        Eden(_opt())
