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

"""Conformer encoder that alternates TIME-axis and CHANNEL-TOKEN-axis attention.

Standard Conformer layers attend over the time axis. Within the chunked-aligner
framework we also want to attend over the "reshaped channel-token" axis (the same
axis the ``ChunkChannelTokenMixer`` introduces): each ``chunk_size`` (= ``C``) block
of encoder frames of size ``d_model`` (= ``D``) is reshaped per chunk to
``[C*D/M, M]`` (``M = chunk_tokens``) and attention runs over the ``M`` axis.

``ChunkAlternatingConformerEncoder`` keeps the regular Conformer machinery
(pre-encode / subsampling, relative positional encoding, chunked-limited streaming
masks, stochastic depth) and simply replaces every other Conformer layer with a
``ChannelAxisConformerLayer``. The channel layers reshape ``[B, T, D]`` to the
per-chunk token view ``[B*n_chunks, M, new_d_model]`` (with
``new_d_model = C*D/M``), run a full Conformer block (feed-forward, self-attention
over the ``M`` tokens, depthwise conv over the ``M`` axis, feed-forward) in that
view, and reshape losslessly back to ``[B, T, D]``. Because the reshape is a
bijection of the ``C*D`` values per chunk, no information is dropped between layers;
any reduction to per-chunk tokens for the lattice still happens at the very end of
the encoder (e.g. first-k / ``ChunkChannelTokenMixer``).

NOTE:
* The channel-axis attention is per-chunk (the chunks live in the batch dim), so it
  never leaks across chunk boundaries -- it is causal at chunk granularity.
* ``new_d_model`` is tied to the chunk geometry (``C*D/M``); the feed-forward and
  convolution in the channel layers therefore run at ``new_d_model``, not ``D``.
* Cache-aware streaming export is not supported by the channel layers yet (the
  per-chunk reshape would need the streaming cache offset threaded through); this
  targets the full, non-cached training / validation forward.
"""

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer

__all__ = ['ChunkAlternatingConformerEncoder', 'ChannelAxisConformerLayer']


class ChannelAxisConformerLayer(nn.Module):
    """A Conformer block that attends over the per-chunk channel-token axis.

    Reshapes ``[B, T, D]`` to ``[B*n_chunks, M, new_d_model]`` (per chunk), runs a
    standard ``abs_pos`` ``ConformerLayer`` over the ``M`` (token) axis at feature dim
    ``new_d_model = chunk_size*d_model/M``, then reshapes losslessly back to
    ``[B, T, D]``. The signature mirrors ``ConformerLayer.forward`` so the encoder's
    layer loop can call it transparently; ``att_mask`` / ``pos_emb`` / ``pad_mask``
    are ignored (the channel block uses full attention over the ``M`` tokens with its
    own learned positional embedding).

    Args:
        d_model: time-view hidden size ``D``.
        chunk_size: encoder frames per chunk ``C``.
        chunk_tokens: tokens per chunk ``M`` (the new "time" axis length).
        n_heads: requested attention heads (coerced to a divisor of ``new_d_model``).
        ff_expansion_factor: feed-forward expansion (``d_ff = new_d_model * factor``).
        conv_kernel_size: depthwise conv kernel over the ``M`` axis (odd).
        conv_norm_type / dropout / dropout_att / use_bias: passed to the inner block.
    """

    def __init__(
        self,
        d_model: int,
        chunk_size: int,
        chunk_tokens: int,
        n_heads: int = 4,
        ff_expansion_factor: int = 4,
        conv_kernel_size: int = 3,
        conv_norm_type: str = 'batch_norm',
        dropout: float = 0.1,
        dropout_att: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()
        if (chunk_size * d_model) % chunk_tokens != 0:
            raise ValueError(
                f"chunk_size*d_model ({chunk_size * d_model}) must be divisible by chunk_tokens ({chunk_tokens})."
            )
        if conv_kernel_size % 2 == 0:
            raise ValueError(f"channel conv_kernel_size must be odd, got {conv_kernel_size}.")

        self.d_model = d_model
        self.chunk_size = chunk_size
        self.chunk_tokens = chunk_tokens
        self.new_d_model = (chunk_size * d_model) // chunk_tokens

        # new_d_model depends on geometry -> coerce heads to a valid divisor.
        heads = max(1, int(n_heads))
        while heads > 1 and self.new_d_model % heads != 0:
            heads -= 1
        self.num_heads = heads
        if self.num_heads != int(n_heads):
            from nemo.utils import logging

            logging.warning(
                f"[ChannelAxisConformerLayer] requested n_heads={int(n_heads)} does not divide "
                f"new_d_model={self.new_d_model} (= chunk_size*d_model/chunk_tokens); "
                f"coercing to n_heads={self.num_heads}. Pick chunk_size/chunk_tokens so that "
                f"new_d_model is divisible by the desired head count to avoid this."
            )

        # Learned positional embedding over the M token slots.
        self.pos = nn.Parameter(torch.zeros(chunk_tokens, self.new_d_model))

        # A standard Conformer block operating in the new_d_model space, attending
        # over the M tokens (abs_pos -> full self-attention with mask=None).
        self.layer = ConformerLayer(
            d_model=self.new_d_model,
            d_ff=self.new_d_model * ff_expansion_factor,
            self_attention_model='abs_pos',
            n_heads=self.num_heads,
            conv_kernel_size=conv_kernel_size,
            conv_norm_type=conv_norm_type,
            conv_context_size=(conv_kernel_size - 1) // 2,
            dropout=dropout,
            dropout_att=dropout_att,
            use_bias=use_bias,
        )

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        if cache_last_channel is not None or cache_last_time is not None:
            raise NotImplementedError(
                "ChannelAxisConformerLayer does not support cache-aware streaming inference."
            )

        B, T, D = x.shape
        C = self.chunk_size
        M = self.chunk_tokens
        new_d = self.new_d_model

        # Zero out padded time steps BEFORE the per-chunk reshape. The reshape folds C
        # consecutive frames into the channel-token axis and self-attention / conv then
        # mix every token in the chunk, so any non-zero values left at padded positions
        # leak into the *valid* frames of the boundary chunk -- and a subsequent
        # full-context time-axis layer spreads that contamination across the whole
        # sequence. Without this masking the encoder output at every valid frame depends
        # on how much padding the batch carries (i.e. on the batch's max length), which
        # breaks batching-invariance and causes a train/inference mismatch. Zeroing here
        # makes the padded frames behave like the F.pad zeros used to complete the last
        # chunk, restoring padding-invariance (matching the standard ConformerEncoder).
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        n_chunks = (T + C - 1) // C
        T_pad = n_chunks * C
        if T_pad > T:
            x = F.pad(x, (0, 0, 0, T_pad - T))

        # [B, T_pad, D] -> [B, n_chunks, C, D] -> flatten chunk row-major -> [.., C*D]
        # -> [.., new_d, M] -> [.., M, new_d] -> [B*n_chunks, M, new_d].
        x = x.reshape(B, n_chunks, C * D)
        x = x.reshape(B, n_chunks, new_d, M).transpose(-1, -2).contiguous()
        x = x.reshape(B * n_chunks, M, new_d)

        # Channel-token-axis Conformer block (full attention over the M tokens).
        x = x + self.pos.to(dtype=x.dtype).unsqueeze(0)
        x = self.layer(x=x)

        # Inverse reshape back to [B, T, D].
        x = x.reshape(B, n_chunks, M, new_d).transpose(-1, -2).contiguous()
        x = x.reshape(B, n_chunks, C * D).reshape(B, n_chunks, C, D).reshape(B, T_pad, D)
        return x[:, :T].contiguous()


class ChunkAlternatingConformerEncoder(ConformerEncoder):
    """ConformerEncoder that alternates time-axis and channel-token-axis layers.

    Identical to :class:`ConformerEncoder` except that every other Conformer layer is
    replaced with a :class:`ChannelAxisConformerLayer`. All other components
    (pre-encode, positional encoding, streaming masks, stochastic depth) are reused.

    Args:
        chunk_size: encoder frames per chunk ``C`` (should match the streaming /
            chunked-aligner chunk).
        chunk_tokens: tokens per chunk ``M`` for the channel-axis layers
            (``new_d_model = chunk_size*d_model/M``).
        channel_conv_kernel_size: depthwise conv kernel (odd) for the channel layers.
        channel_layers: which layer indices become channel-axis layers -- ``'odd'``
            (default: index 0 is time, then alternate) or ``'even'``.
        **kwargs: forwarded verbatim to :class:`ConformerEncoder`.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_tokens: int,
        channel_conv_kernel_size: int = 3,
        channel_layers: str = 'odd',
        **kwargs,
    ):
        super().__init__(**kwargs)

        if channel_layers not in ('odd', 'even'):
            raise ValueError(f"channel_layers must be 'odd' or 'even', got '{channel_layers}'.")

        D = self.d_model
        if (chunk_size * D) % chunk_tokens != 0:
            raise ValueError(
                f"chunk_size*d_model ({chunk_size * D}) must be divisible by chunk_tokens ({chunk_tokens})."
            )

        self.chunk_size = chunk_size
        self.chunk_tokens = chunk_tokens

        ff_expansion_factor = int(kwargs.get('ff_expansion_factor', 4))
        n_heads = int(kwargs.get('n_heads', 4))
        conv_norm_type = kwargs.get('conv_norm_type', 'batch_norm')
        dropout = float(kwargs.get('dropout', 0.1))
        dropout_att = float(kwargs.get('dropout_att', 0.0))
        use_bias = bool(kwargs.get('use_bias', True))

        channel_on_odd = channel_layers == 'odd'
        n_channel = 0
        for i in range(len(self.layers)):
            is_channel = (i % 2 == 1) if channel_on_odd else (i % 2 == 0)
            if is_channel:
                self.layers[i] = ChannelAxisConformerLayer(
                    d_model=D,
                    chunk_size=chunk_size,
                    chunk_tokens=chunk_tokens,
                    n_heads=n_heads,
                    ff_expansion_factor=ff_expansion_factor,
                    conv_kernel_size=channel_conv_kernel_size,
                    conv_norm_type=conv_norm_type,
                    dropout=dropout,
                    dropout_att=dropout_att,
                    use_bias=use_bias,
                )
                n_channel += 1

        new_d_model = (chunk_size * D) // chunk_tokens
        from nemo.utils import logging

        logging.info(
            f"[chunk-alternating-conformer] {n_channel}/{len(self.layers)} layers are channel-axis "
            f"(chunk_size={chunk_size}, chunk_tokens={chunk_tokens}, new_d_model={new_d_model}); "
            f"the rest attend over time."
        )
