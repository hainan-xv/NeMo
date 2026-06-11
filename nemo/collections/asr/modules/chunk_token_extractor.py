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

"""Chunk-wise learned token-extraction layer.

This optional layer sits at the very end of the encoder. It compresses every
fixed-size block of ``frames_per_chunk`` encoder frames into a smaller, fixed set
of ``tokens_per_chunk`` learned tokens via cross-attention:

* a trainable query bank ``Q in R^{tokens_per_chunk x d_model}`` (shared across
  chunks) attends to the ``frames_per_chunk`` encoder frames of each chunk (the
  keys/values), through standard trainable Q/K/V/output projections
  (``torch.nn.MultiheadAttention``);
* the output is ``tokens_per_chunk`` vectors per chunk, i.e. the chunk of
  ``[frames_per_chunk, d_model]`` is summarized into ``[tokens_per_chunk, d_model]``.

For the streaming Chunked-Aligner this decouples the *acoustic* chunk size
(``frames_per_chunk``, e.g. 12 encoder frames) from the maximum number of tokens
a chunk may emit (``tokens_per_chunk``, e.g. 5): downstream the Chunked-Aligner
loss / decoder simply operate on the extracted tokens with ``chunk_size ==
tokens_per_chunk``.

Padded frames in the (possibly partial) final chunk of each utterance are masked
out of the attention. Chunks that are entirely padding produce zero vectors and
fall outside the returned valid length.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['ChunkTokenExtractor']


class ChunkTokenExtractor(nn.Module):
    """Compress each chunk of encoder frames into a fixed set of learned tokens.

    Args:
        d_model: encoder / model hidden size ``D``.
        frames_per_chunk: number of encoder frames per chunk ``C`` (acoustic span).
        tokens_per_chunk: number of extracted tokens per chunk ``t`` (``t <= C``
            is typical but not required).
        num_heads: number of attention heads for the cross-attention.
        dropout: attention dropout probability.
        frame_pos_emb: if True, add a learnable positional embedding to the ``C``
            frame slots before attention (lets the queries distinguish in-chunk
            frame positions). Defaults to False.
    """

    def __init__(
        self,
        d_model: int,
        frames_per_chunk: int,
        tokens_per_chunk: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        frame_pos_emb: bool = False,
    ):
        super().__init__()
        if frames_per_chunk < 1:
            raise ValueError(f"frames_per_chunk must be >= 1, got {frames_per_chunk}.")
        if tokens_per_chunk < 1:
            raise ValueError(f"tokens_per_chunk must be >= 1, got {tokens_per_chunk}.")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        self.d_model = d_model
        self.frames_per_chunk = frames_per_chunk
        self.tokens_per_chunk = tokens_per_chunk

        # Trainable query bank, shared across all chunks: [t, D].
        self.query = nn.Parameter(torch.empty(tokens_per_chunk, d_model))
        nn.init.normal_(self.query, mean=0.0, std=0.02)

        # Cross-attention with trainable Q/K/V/output projections.
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

        if frame_pos_emb:
            self.frame_pos = nn.Parameter(torch.zeros(frames_per_chunk, d_model))
        else:
            self.register_parameter('frame_pos', None)

    def forward(self, encoded: torch.Tensor, encoded_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress encoder output chunk-by-chunk.

        Args:
            encoded: encoder output of shape ``[B, D, T]``.
            encoded_len: valid frame counts per sample, ``[B]``.

        Returns:
            A tuple ``(extracted, extracted_len)`` where ``extracted`` has shape
            ``[B, D, n_chunks * tokens_per_chunk]`` and ``extracted_len`` holds the
            valid extracted-token counts (``ceil(encoded_len / C) * tokens_per_chunk``).
        """
        B, D, T = encoded.shape
        C = self.frames_per_chunk
        t = self.tokens_per_chunk
        device = encoded.device

        n_chunks = (T + C - 1) // C
        T_pad = n_chunks * C

        # [B, D, T] -> [B, T, D]; pad time up to a whole number of chunks.
        x = encoded.transpose(1, 2)
        if T_pad > T:
            x = F.pad(x, (0, 0, 0, T_pad - T))

        # Run the small cross-attention in float32 for precision/dtype robustness
        # (encoder output may be bf16/fp16), then cast the result back.
        orig_dtype = x.dtype
        x = x.float()

        # [B, T_pad, D] -> [B*n_chunks, C, D] (keys/values per chunk).
        x = x.reshape(B, n_chunks, C, D)
        if self.frame_pos is not None:
            x = x + self.frame_pos.float().view(1, 1, C, D)
        kv = x.reshape(B * n_chunks, C, D)

        # Key padding mask: True == padded frame (ignored by attention).
        frame_idx = torch.arange(T_pad, device=device).view(1, n_chunks, C)
        valid = frame_idx < encoded_len.to(device).view(B, 1, 1)  # [B, n_chunks, C]
        key_padding_mask = ~valid.reshape(B * n_chunks, C)

        # Fully-padded chunks would make softmax see an all-masked row (-> NaN).
        # Temporarily unmask them, then zero their outputs below.
        fully_padded = key_padding_mask.all(dim=1)
        if fully_padded.any():
            key_padding_mask = key_padding_mask.masked_fill(fully_padded.unsqueeze(1), False)

        q = self.query.float().unsqueeze(0).expand(B * n_chunks, t, D)
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)  # [B*n_chunks, t, D]
        out = self.norm(out)

        if fully_padded.any():
            out = out.masked_fill(fully_padded.view(-1, 1, 1), 0.0)

        # [B*n_chunks, t, D] -> [B, n_chunks*t, D] -> [B, D, n_chunks*t].
        out = out.reshape(B, n_chunks * t, D).transpose(1, 2).contiguous()
        out = out.to(orig_dtype)

        extracted_len = ((encoded_len.to(device) + C - 1) // C) * t
        return out, extracted_len.to(encoded_len.dtype)
