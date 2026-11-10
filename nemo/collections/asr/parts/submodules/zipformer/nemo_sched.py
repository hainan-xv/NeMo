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

"""Eden, icefall's Zipformer LR schedule, as a torch-API scheduler NeMo can drive.

WHY. ScaledAdam was ported first and left on NeMo's CosineAnnealing, which is the
one remaining deviation from upstream's recipe. That arm descended far slower
than its AdamW control (loss ~6.3 at 18k steps against ~1.0), which is the
signature of a wrong LR schedule rather than a broken optimizer -- so Eden is the
next thing to rule out.

THE ADAPTER PROBLEM. Upstream's Eden is not a torch ``_LRScheduler``: it exposes
``step_batch`` / ``step_epoch`` and decays on BOTH axes. Lightning only ever calls
``step()``. This class keeps the formula exactly and drives both axes from the
global step count, so Lightning's one hook is enough.

    lr = base_lr
         * ((batch^2 + lr_batches^2) / lr_batches^2) ** -0.25
         * ((epoch^2 + lr_epochs^2) / lr_epochs^2) ** -0.25
         * warmup

TWO DELIBERATE DEPARTURES, both forced by how this project trains:

1. ``epoch`` is CONTINUOUS (``batch / steps_per_epoch``), not an integer bumped at
   epoch boundaries. Upstream steps it discretely; the values agree at every
   boundary and the continuous form avoids an LR discontinuity mid-run. More
   importantly it does not depend on Lightning's epoch hooks firing, which is
   what makes a single ``step()`` sufficient.

2. ``steps_per_epoch`` is EXPLICIT and has no default tied to the trainer. In this
   project an "epoch" is an artificial 2000-step validation leg
   (``limit_train_batches=2000``), NOT a pass over the data -- Granary is far
   larger than that. Taking Lightning's epoch counter literally would give ~250
   epochs over a 500k-step run, and upstream's ``lr_epochs=6`` (tuned for 20-40
   epochs) would then crush the LR to near zero. Set ``steps_per_epoch`` so the
   run spans roughly 20-40 Eden-epochs: for 500k steps that is ~16000.

RESUME. These runs restart every 4 hours, so the schedule must survive a reload.
``_LRScheduler`` already persists ``last_epoch``, which is the only state this
scheduler has -- ``batch`` and ``epoch`` are both derived from it, so a resumed
job recomputes the identical LR. That is covered by a test.
"""

from dataclasses import dataclass
from typing import Optional

from torch.optim.lr_scheduler import _LRScheduler

from nemo.core.config.schedulers import SchedulerParams
from nemo.core.optim.lr_scheduler import AVAILABLE_SCHEDULERS, register_scheduler
from nemo.utils import logging

__all__ = ["Eden", "EdenParams", "register_eden"]


class Eden(_LRScheduler):
    """icefall's Eden schedule, driven entirely from the global step.

    Args:
        optimizer: optimizer whose LRs are set.
        lr_batches: batches after which the LR starts decaying appreciably
            (upstream suggests 5000).
        lr_epochs: Eden-epochs after which the LR starts decaying appreciably
            (upstream suggests 6, for a 20-40 epoch run).
        steps_per_epoch: optimizer steps that count as one Eden-epoch. Leave it
            None to derive it as ``max_steps / epochs_over_run``, which keeps the
            epoch axis meaningful whatever the run length. See the module
            docstring -- this is NOT the trainer's epoch length here.
        epochs_over_run: how many Eden-epochs the whole run should span when
            steps_per_epoch is derived. Upstream's lr_epochs=6 is tuned for a
            20-40 epoch run, so 30 keeps us in that regime.
        max_steps: injected by NeMo's ``prepare_lr_scheduler``; used only to
            derive steps_per_epoch.
        warmup_batches: batches over which the warmup factor ramps.
        warmup_start: warmup factor at step 0 (upstream 0.5 -- note Eden warms up
            from HALF the base LR, not from zero).
        min_lr: floor applied after the Eden factors.
        last_epoch: standard torch resume hook.
    """

    def __init__(
        self,
        optimizer,
        *,
        lr_batches: float = 5000.0,
        lr_epochs: float = 6.0,
        steps_per_epoch: float = None,
        epochs_over_run: float = 30.0,
        max_steps: int = None,
        min_lr: float = 0.0,
        warmup_batches: float = 500.0,
        warmup_start: float = 0.5,
        last_epoch: int = -1,
    ):
        if lr_batches <= 0:
            raise ValueError(f"lr_batches must be positive, got {lr_batches}")
        if lr_epochs <= 0:
            raise ValueError(f"lr_epochs must be positive, got {lr_epochs}")
        if epochs_over_run <= 0:
            raise ValueError(f"epochs_over_run must be positive, got {epochs_over_run}")
        if steps_per_epoch is None:
            # Derive from the run length so the epoch axis scales with the run
            # rather than being a magic constant that silently stops matching it.
            if not max_steps:
                raise ValueError(
                    "Eden needs either an explicit steps_per_epoch or a max_steps to derive it from; "
                    "got neither. NeMo injects max_steps from trainer.max_steps."
                )
            steps_per_epoch = float(max_steps) / float(epochs_over_run)
        if steps_per_epoch <= 0:
            raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
        if warmup_batches < 0:
            raise ValueError(f"warmup_batches must be non-negative, got {warmup_batches}")
        if not 0.0 <= warmup_start <= 1.0:
            raise ValueError(f"warmup_start must be in [0, 1], got {warmup_start}")

        # Assigned BEFORE super().__init__, which calls step() -> get_lr() immediately.
        self.lr_batches = float(lr_batches)
        self.lr_epochs = float(lr_epochs)
        self.steps_per_epoch = float(steps_per_epoch)
        self.epochs_over_run = float(epochs_over_run)
        self.max_steps = max_steps
        self.warmup_batches = float(warmup_batches)
        self.warmup_start = float(warmup_start)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def _factors(self, batch: float):
        epoch = batch / self.steps_per_epoch
        batch_factor = ((batch**2 + self.lr_batches**2) / self.lr_batches**2) ** -0.25
        epoch_factor = ((epoch**2 + self.lr_epochs**2) / self.lr_epochs**2) ** -0.25
        if self.warmup_batches > 0 and batch < self.warmup_batches:
            warmup = self.warmup_start + (1.0 - self.warmup_start) * (batch / self.warmup_batches)
        else:
            warmup = 1.0
        return batch_factor, epoch_factor, warmup

    def get_lr(self):
        # last_epoch is the global optimizer-step count: Lightning steps this
        # scheduler once per batch (interval: step).
        batch = float(max(self.last_epoch, 0))
        batch_factor, epoch_factor, warmup = self._factors(batch)
        scale = batch_factor * epoch_factor * warmup
        return [max(base_lr * scale, self.min_lr) for base_lr in self.base_lrs]


@dataclass
class EdenParams(SchedulerParams):
    """Config schema for Eden. Defaults follow icefall's zipformer recipe."""

    lr_batches: float = 5000.0
    lr_epochs: float = 6.0
    steps_per_epoch: Optional[float] = None
    epochs_over_run: float = 30.0
    warmup_batches: float = 500.0
    warmup_start: float = 0.5
    min_lr: float = 0.0


def register_eden() -> None:
    """Idempotent registration, so importing this twice is harmless."""
    if "eden" in AVAILABLE_SCHEDULERS:
        return
    register_scheduler("eden", Eden, EdenParams())
    logging.info("Registered scheduler 'eden' (icefall Eden, step-driven)")


register_eden()
