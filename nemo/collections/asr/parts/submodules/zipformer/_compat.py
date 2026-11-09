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

"""Shims so icefall's Zipformer runs unmodified-in-spirit inside NeMo.

The vendored ``zipformer.py``/``scaling.py`` are taken verbatim from icefall
(Apache-2.0) apart from imports and the k2 fallbacks below, so they stay easy to
diff against upstream.

Two upstream dependencies are replaced:

``torch_autocast``
    icefall.utils helper that picks the new or old autocast API. Reimplemented
    rather than depending on icefall, which would drag in k2, lhotse pinning and
    a parallel training stack.

``k2``
    Used ONLY for fused SwooshL/SwooshR CUDA kernels. k2 is a compiled package
    and is not installed in this container. The activations have exact
    pure-PyTorch definitions upstream (``SwooshLForward``/``SwooshRForward``,
    and the torch.jit branches), so without k2 we take those: the FORWARD maths
    is identical, and backward goes through autograd instead of k2's
    hand-written kernel. The cost is memory and a little speed, not accuracy.
"""

import contextlib
from typing import Tuple

import torch
from torch import Tensor, nn

try:  # pragma: no cover - exercised only where k2 is installed
    import k2  # noqa: F401

    HAVE_K2 = True
except Exception:  # noqa: BLE001
    HAVE_K2 = False


@contextlib.contextmanager
def torch_autocast(device_type: str = "cuda", **kwargs):
    """``torch.amp.autocast`` on new torch, ``torch.cuda.amp.autocast`` on old."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        with torch.amp.autocast(device_type=device_type, **kwargs):
            yield
    else:  # pragma: no cover - torch < 2.3
        with torch.cuda.amp.autocast(**kwargs):
            yield


class EncoderInterface(nn.Module):
    """Upstream's abstract encoder base; only the signature matters here."""

    def forward(self, x: Tensor, x_lens: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
