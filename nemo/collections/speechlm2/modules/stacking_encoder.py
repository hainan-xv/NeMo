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
"""An encoder-free audio front-end: stack mel frames and hand them to the LLM.

Replaces the 609M-parameter Conformer with a reshape. The dimensions line up
exactly for the nemotron-speech-streaming-en-0.6b setup this repo uses::

    preprocessor: 128 mel features at a 10 ms hop  (100 frames/s)
    Conformer:    subsampling_factor 8  ->  d_model 1024  (12.5 frames/s)
    stacking:     128 mel x 8 frames    ->  1024          (12.5 frames/s)

so ``perception.proj`` (1024 -> LLM hidden) and every downstream frame-based
quantity -- ``chunk_size``, ``num_delay_frames``, the 0.08 s frame, the two-front
window arithmetic -- are unchanged. The module holds NO parameters: all acoustic
modelling moves into the LLM.

It is deliberately NOT a ``ConformerEncoder`` subclass, so
``AudioPerceptionModule.forward`` dispatches to its generic encoder branch and
never tries cache-aware streaming against it.

There is no look-ahead to configure: a stacked frame sees exactly its own
``subsampling_factor`` mel frames and nothing beyond them, so this front-end is
causal by construction and the chunk boundary is the only source of latency.
``att_context_size`` / ``att_chunk_offset`` may still be assigned by callers;
they are accepted and ignored.
"""

from typing import Optional, Tuple

import torch
from torch import nn

__all__ = ["StackingEncoder"]


class StackingEncoder(nn.Module):
    """Group ``subsampling_factor`` consecutive feature frames into one frame.

    Args:
        feat_in: input features per frame (mel bins).
        subsampling_factor: how many input frames become one output frame.
        d_model: optional output width. When None (default) the output is the
            bare stack, ``feat_in * subsampling_factor``. When given and it
            differs from that product, a single Linear maps onto it -- the "one
            linear transform to match dimension" case, for feature/LLM widths
            that do not already line up.
    """

    def __init__(self, feat_in: int = 128, subsampling_factor: int = 8, d_model: Optional[int] = None):
        super().__init__()
        if feat_in <= 0 or subsampling_factor <= 0:
            raise ValueError(f"feat_in and subsampling_factor must be positive, got {feat_in}, {subsampling_factor}")
        self.feat_in = int(feat_in)
        self.subsampling_factor = int(subsampling_factor)
        stacked = self.feat_in * self.subsampling_factor
        self.stacked_dim = stacked
        if d_model is None or int(d_model) == stacked:
            self.d_model = stacked
            self.out_proj = None
        else:
            self.d_model = int(d_model)
            self.out_proj = nn.Linear(stacked, self.d_model)

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(B, feat_in, T)`` mel -> ``(B, d_model, ceil(T / subsampling_factor))``.

        The tail is zero-padded rather than dropped, so the last partial frame
        still carries its audio -- dropping it would silently truncate the
        final up-to-70 ms of every utterance.
        """
        if audio_signal.dim() != 3:
            raise ValueError(f"expected (B, feat_in, T), got shape {tuple(audio_signal.shape)}")
        b, f, t = audio_signal.shape
        if f != self.feat_in:
            raise ValueError(f"expected {self.feat_in} features, got {f}")
        s = self.subsampling_factor

        pad = (-t) % s
        if pad:
            audio_signal = torch.nn.functional.pad(audio_signal, (0, pad))
        t_out = audio_signal.shape[-1] // s

        # (B, F, T) -> (B, F, T_out, S) -> (B, S, F, T_out) -> (B, S*F, T_out).
        # Channel index is s * F + f, so each contiguous block of F channels is
        # one sub-frame: the output really is "8 frames concatenated".
        x = audio_signal.view(b, f, t_out, s).permute(0, 3, 1, 2).reshape(b, s * f, t_out)

        if self.out_proj is not None:
            x = self.out_proj(x.transpose(1, 2)).transpose(1, 2)

        out_len = torch.div(length + s - 1, s, rounding_mode="floor").clamp(max=t_out)
        return x, out_len
