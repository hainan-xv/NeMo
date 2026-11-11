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

"""A causal ConvASREncoder with chunk-local bidirectional mixing inserted.

Gives a convolution-only encoder the same information access a ``chunked_limited``
FastConformer has -- left context plus everything to the end of the current chunk
-- so that conv-vs-attention can be compared without a lookahead handicap. See
:mod:`nemo.collections.asr.parts.submodules.chunk_bidir_conv` for the mechanism.
"""

from typing import List, Optional

import torch
from omegaconf import ListConfig

from nemo.collections.asr.modules.conv_asr import ConvASREncoder
from nemo.collections.asr.parts.submodules.chunk_bidir_conv import ChunkBidirMixer
from nemo.utils import logging

__all__ = ["ChunkBidirConvASREncoder"]


class ChunkBidirConvASREncoder(ConvASREncoder):
    """``ConvASREncoder`` plus chunk-local bidirectional mixers.

    Args:
        chunk_size: chunk length in OUTPUT frames, matching ``joint.chunk_size``.
        mixer_positions: block indices to insert a mixer AFTER. Must all be at or
            beyond the last stride-bearing block, so the mixers run at the final
            frame rate and their chunk grid coincides with the joint's. Defaults
            to three positions spread over the post-subsampling blocks.
        mixer_kernel_size: override the default ``2 * chunk_size - 1``.
        mixer_dropout: dropout on the mixed branch.
    """

    def __init__(
        self,
        *args,
        chunk_size: int = 14,
        mixer_positions: Optional[List[int]] = None,
        mixer_kernel_size: Optional[int] = None,
        mixer_dropout: float = 0.0,
        **kwargs,
    ):
        jasper_cfg = kwargs.get("jasper", args[0] if args else None)
        if jasper_cfg is None:
            raise ValueError("ChunkBidirConvASREncoder needs the 'jasper' block list to locate strides")

        super().__init__(*args, **kwargs)

        blocks = list(self.encoder)
        n_blocks = len(blocks)

        # Last block that changes the frame rate. Inserting a mixer before this
        # would have it operate at a different rate than `chunk_size` assumes, so
        # its chunk grid would not line up with the one the joint commits on.
        #
        # Read from the CONFIG, not from the built modules: a JasperBlock exposes
        # no `.stride` attribute, so getattr(blk, "stride", None) is always None
        # and the guard silently passes for every position. Mirrors how
        # ConvASREncoder computes _subsampling_factor.
        last_stride_idx = -1
        for i, lcfg in enumerate(jasper_cfg):
            raw = lcfg["stride"]
            s = int(raw[0]) if isinstance(raw, (list, tuple, ListConfig)) else int(raw)
            if s > 1:
                last_stride_idx = i

        if mixer_positions is None:
            first = last_stride_idx + 1
            span = n_blocks - first
            mixer_positions = sorted({first + max(0, (span * k) // 3 - 1) for k in (1, 2, 3)})
        mixer_positions = sorted(int(p) for p in mixer_positions)

        for p in mixer_positions:
            if not 0 <= p < n_blocks:
                raise ValueError(f"mixer position {p} out of range for {n_blocks} blocks")
            if p < last_stride_idx:
                raise ValueError(
                    f"mixer position {p} precedes the last stride-bearing block ({last_stride_idx}); "
                    "the mixer would run at a different frame rate than chunk_size assumes"
                )

        channels = self._feat_out
        rebuilt = []
        for i, blk in enumerate(blocks):
            rebuilt.append(blk)
            if i in mixer_positions:
                rebuilt.append(
                    ChunkBidirMixer(
                        channels=channels,
                        chunk_size=chunk_size,
                        kernel_size=mixer_kernel_size,
                        dropout=mixer_dropout,
                    )
                )
        self.encoder = torch.nn.Sequential(*rebuilt)

        self.chunk_size = int(chunk_size)
        self.mixer_positions = mixer_positions
        logging.info(
            f"ChunkBidirConvASREncoder: {len(mixer_positions)} chunk-local mixer(s) after blocks "
            f"{mixer_positions} (last stride-bearing block {last_stride_idx}), chunk_size={chunk_size}, "
            f"kernel={mixer_kernel_size or 2 * chunk_size - 1}, channels={channels}"
        )
