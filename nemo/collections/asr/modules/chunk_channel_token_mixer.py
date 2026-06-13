# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Chunk-wise reshape + channel-axis attention token mixer.

This optional layer sits at the very end of the encoder, as an alternative to the
parameter-free first-k frame selection and to the cross-attention
``ChunkTokenExtractor``. Instead of selecting / pooling along the time axis, it
mixes information across BOTH time and channels and emits a fixed number of tokens
per chunk:

For each fixed-size block of ``frames_per_chunk`` (= ``C``) encoder frames of size
``d_model`` (= ``D``):

1. flatten the chunk to ``[C * D]`` (row-major, time-major / channel-minor);
2. reshape to ``[C * D / M, M]`` where ``M = tokens_per_chunk`` (the max tokens a
   chunk may emit), then transpose to ``[M, new_d_model]`` with
   ``new_d_model = C * D / M`` -- i.e. ``M`` tokens, each a strided slice (mod ``M``)
   of the flattened (time x channel) chunk, so every token draws from all frames and
   a subset of channels;
3. self-attend over the ``M`` axis (treated as the new "time" axis), so the ``M``
   tokens can exchange information;
4. the ``M`` output rows ARE the per-chunk tokens.

Downstream the Chunked-Aligner loss / decoder operate on the extracted tokens with
``chunk_size == tokens_per_chunk`` and an encoder hidden size of ``new_d_model``.

NOTE: ``new_d_model`` is tied to the chunk geometry (``C * D / M``), so the module
weights are specific to a given ``chunk_size`` / ``d_model`` / ``M``.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['ChunkChannelTokenMixer']


class ChunkChannelTokenMixer(nn.Module):
    """Reshape each chunk into ``M`` tokens and self-attend over the token axis.

    Args:
        d_model: encoder / model hidden size ``D``.
        frames_per_chunk: number of encoder frames per chunk ``C`` (acoustic span).
        tokens_per_chunk: number of emitted tokens per chunk ``M`` (the "8").
        num_heads: requested number of attention heads for the token-axis attention.
            Reduced to the largest divisor of ``new_d_model`` that is ``<= num_heads``
            (down to 1) since ``new_d_model`` depends on the chunk geometry.
        dropout: attention dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        frames_per_chunk: int,
        tokens_per_chunk: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if frames_per_chunk < 1:
            raise ValueError(f"frames_per_chunk must be >= 1, got {frames_per_chunk}.")
        if tokens_per_chunk < 1:
            raise ValueError(f"tokens_per_chunk must be >= 1, got {tokens_per_chunk}.")
        if (frames_per_chunk * d_model) % tokens_per_chunk != 0:
            raise ValueError(
                f"frames_per_chunk * d_model ({frames_per_chunk * d_model}) must be divisible "
                f"by tokens_per_chunk ({tokens_per_chunk})."
            )

        self.d_model = d_model
        self.frames_per_chunk = frames_per_chunk
        self.tokens_per_chunk = tokens_per_chunk
        # New per-token feature dim after the reshape (the new "d_model").
        self.out_dim = (frames_per_chunk * d_model) // tokens_per_chunk

        # new_d_model depends on the geometry, so coerce num_heads to a valid divisor.
        heads = max(1, int(num_heads))
        while heads > 1 and self.out_dim % heads != 0:
            heads -= 1
        self.num_heads = heads

        # Learnable positional embedding over the M token slots.
        self.pos = nn.Parameter(torch.zeros(tokens_per_chunk, self.out_dim))

        # Self-attention over the M tokens (pre-norm + residual).
        self.norm = nn.LayerNorm(self.out_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.out_dim, num_heads=self.num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, encoded: torch.Tensor, encoded_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reshape + token-axis attention chunk-by-chunk.

        Args:
            encoded: encoder output of shape ``[B, D, T]``.
            encoded_len: valid frame counts per sample, ``[B]``.

        Returns:
            A tuple ``(mixed, mixed_len)`` where ``mixed`` has shape
            ``[B, new_d_model, n_chunks * tokens_per_chunk]`` and ``mixed_len`` holds
            the valid token counts (``ceil(encoded_len / C) * tokens_per_chunk``).
        """
        B, D, T = encoded.shape
        C = self.frames_per_chunk
        M = self.tokens_per_chunk
        device = encoded.device

        n_chunks = (T + C - 1) // C
        T_pad = n_chunks * C

        # [B, D, T] -> [B, T, D]; pad time up to a whole number of chunks.
        x = encoded.transpose(1, 2)
        if T_pad > T:
            x = F.pad(x, (0, 0, 0, T_pad - T))

        # Run the small attention in float32 for precision/dtype robustness
        # (encoder output may be bf16/fp16), then cast the result back.
        orig_dtype = x.dtype
        x = x.float()

        # [B, T_pad, D] -> [B, n_chunks, C, D] -> flatten chunk row-major -> [.., C*D].
        x = x.reshape(B, n_chunks, C * D)
        # Reshape to [.., C*D/M, M], then move the M axis to be the (new) time axis.
        x = x.reshape(B, n_chunks, self.out_dim, M).transpose(-1, -2).contiguous()  # [B, n_chunks, M, out_dim]
        tokens = x.reshape(B * n_chunks, M, self.out_dim)

        # Self-attention over the M token axis (pre-norm + residual).
        h = self.norm(tokens + self.pos.float().unsqueeze(0))
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        out = tokens + attn_out

        # [B*n_chunks, M, out_dim] -> [B, n_chunks*M, out_dim] -> [B, out_dim, n_chunks*M].
        out = out.reshape(B, n_chunks * M, self.out_dim).transpose(1, 2).contiguous()
        out = out.to(orig_dtype)

        mixed_len = ((encoded_len.to(device) + C - 1) // C) * M
        return out, mixed_len.to(encoded_len.dtype)
