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

"""Register icefall's ScaledAdam with NeMo's optimizer registry.

WHY THIS MATTERS. Zipformer is designed around ScaledAdam, and upstream reports
material degradation with plain Adam. Our first Zipformer CHAT run used AdamW and
plateaued at val_wer ~0.20 against a ~0.085 Conformer reference, with the descent
flattening by ~50k steps -- the signature of an optimizer mismatch rather than an
architecture limit.

WHAT SCALEDADAM DOES DIFFERENTLY. It scales each tensor's update by that tensor's
RMS, and learns a per-tensor scale factor alongside the direction. That makes the
learning rate roughly scale-invariant across parameters, which is why its base LR
(~0.045) is an order of magnitude larger than Adam's and NOT comparable to it.
Carrying over an Adam LR would be as wrong as the 5e-5 we started with.

NOT REGISTERED: Eden. Upstream's scheduler is not a torch ``_LRScheduler`` -- it
exposes ``step_batch``/``step_epoch`` rather than ``step()``, and decays on both
axes. Lightning drives schedulers through the torch API, so wiring Eden in needs
an adapter; NeMo's CosineAnnealing is used meanwhile. That is a real deviation
from the upstream recipe and is worth revisiting if ScaledAdam alone does not
close the gap.
"""

from dataclasses import dataclass
from typing import Optional

from nemo.core.config.optimizers import OptimizerParams
from nemo.core.optim.optimizers import AVAILABLE_OPTIMIZERS, register_optimizer
from nemo.utils import logging

from .optim import ScaledAdam

__all__ = ["ScaledAdam", "ScaledAdamParams", "register_scaled_adam"]


@dataclass
class ScaledAdamParams(OptimizerParams):
    """Config schema for ScaledAdam. Defaults follow icefall's zipformer recipe."""

    lr: float = 0.045
    clipping_scale: Optional[float] = 2.0
    betas: tuple = (0.9, 0.98)
    scalar_lr_scale: float = 0.1
    eps: float = 1.0e-8
    param_min_rms: float = 1.0e-5
    param_max_rms: float = 3.0
    scalar_max: float = 10.0
    size_update_period: int = 4
    clipping_update_period: int = 100


def register_scaled_adam() -> None:
    """Idempotent registration, so importing this twice is harmless."""
    if "scaled_adam" in AVAILABLE_OPTIMIZERS:
        return
    register_optimizer("scaled_adam", ScaledAdam, ScaledAdamParams())
    logging.info("Registered optimizer 'scaled_adam' (icefall ScaledAdam)")


register_scaled_adam()
