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

"""Chunk-local bidirectional mixing for a causal convolutional encoder.

WHY. A causal conv encoder sees unlimited left context and ZERO lookahead. The
FastConformer it is measured against uses ``chunked_limited`` attention: every
frame sees its left context AND everything up to the end of its chunk. Comparing
those two confounds ARCHITECTURE with LOOKAHEAD -- the conv arm is simply given
less information, so a WER gap cannot be attributed to removing attention.

This module closes that gap from the right side: it gives the conv stack the SAME
access the Conformer already has, rather than crippling the Conformer to match.

WHAT IT IS NOT. It is NOT a per-layer ``future_context`` budget. Stacking a fixed
f-frame lookahead over N blocks gives every frame a SLIDING N*f window, so the
latency is that window -- constant for every frame, and unrelated to where the
chunk boundary falls. Commit granularity is what actually costs latency, which is
why the additive right-context model was discarded earlier. Here the lookahead is
CHUNK-ALIGNED: a frame at the start of a chunk sees ~chunk_size frames ahead, a
frame at the chunk end sees none, and nothing ever crosses the boundary.

HOW. Reshape ``(B, D, T)`` into NON-OVERLAPPING chunks ``(B*n, D, C)`` and mix
within each chunk with a depthwise convolution wide enough to span it
(``kernel = 2C - 1`` with 'same' padding, so every position reaches every other).
Cross-chunk leakage is impossible by construction -- the chunks are separate rows
of the batch, not merely masked. Left context across chunks is unaffected: it
still flows through the causal stack this module is inserted into, at no extra
cost and with no recomputation.

IDENTITY AT INIT. The block is residual and its output projection is zero-init, so
a freshly built model is numerically identical to the causal baseline. That keeps
a Citrinet warm start meaningful -- the pretrained stack is not perturbed on step
one by randomly initialised mixers.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ChunkBidirMixer"]


class ChunkBidirMixer(nn.Module):
    """Mix bidirectionally within each chunk; never across chunk boundaries.

    Matches the ``([tensors], lengths)`` calling convention of
    :class:`~nemo.collections.asr.parts.submodules.jasper.JasperBlock` so it can be
    inserted directly into ``ConvASREncoder``'s ``nn.Sequential``.

    Args:
        channels: feature dimension (depthwise, so in == out).
        chunk_size: chunk length in frames AT THIS POINT IN THE STACK. Insert only
            after all subsampling, otherwise this must be rescaled by the
            remaining stride and the chunk grid will not line up with the joint's.
        kernel_size: defaults to ``2 * chunk_size - 1``, the smallest kernel for
            which every position in a chunk reaches every other position.
        dropout: applied to the mixed branch before the residual add.
    """

    def __init__(
        self,
        channels: int,
        chunk_size: int,
        kernel_size: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.chunk_size = int(chunk_size)
        k = int(kernel_size) if kernel_size else 2 * self.chunk_size - 1
        if k % 2 == 0:
            raise ValueError(f"kernel_size must be odd for symmetric 'same' padding, got {k}")
        self.kernel_size = k

        self.depthwise = nn.Conv1d(channels, channels, k, padding=k // 2, groups=channels, bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU()
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Zero-init the OUTPUT projection so the block starts as an exact
        # identity. A warm-started Citrinet stack then begins training as the
        # causal baseline rather than as the baseline plus random noise.
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def mix(self, x: torch.Tensor) -> torch.Tensor:
        """Apply chunk-local mixing to ``(B, D, T)`` and return the same shape."""
        B, D, T = x.shape
        C = self.chunk_size
        pad = (C - T % C) % C
        xp = F.pad(x, (0, pad)) if pad else x
        n = xp.shape[-1] // C

        # (B, D, n, C) -> (B, n, D, C) -> (B*n, D, C): chunks become batch rows,
        # so the convolution physically cannot reach across a boundary.
        chunks = xp.view(B, D, n, C).permute(0, 2, 1, 3).reshape(B * n, D, C)
        y = self.depthwise(chunks)
        y = self.activation(self.norm(y))
        y = self.pointwise(y)
        y = y.reshape(B, n, D, C).permute(0, 2, 1, 3).reshape(B, D, n * C)
        return y[..., :T] if pad else y

    def forward(self, xs_and_lengths: Tuple[List[torch.Tensor], torch.Tensor]):
        xs, lengths = xs_and_lengths
        x = xs[-1]
        y = x + self.dropout(self.mix(x))
        return [y], lengths
