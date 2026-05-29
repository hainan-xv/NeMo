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
"""Parallel chunk heads: emit K text tokens per audio chunk in one LLM forward.

Architecture (depth-transformer + tied lm_head):

    anchor_hidden  (M, H)                          M = total anchors in the batch
          │
          │  + slot_embeds[k]   (K, H)             K learned slot positions
          ▼
        (M, K, H)
          │
          │  depth_transformer    (1 layer, causal over slots)
          ▼
        (M, K, H)
          │
          │  shared lm_head  (H, V)                tied with the model's lm_head
          ▼
        (M, K, V)   ← K parallel next-token logits per anchor

Causality:
    Slot k can attend to slots 0..k. This lets slot k condition on slots
    0..k-1's depth-transformer output, mirroring within-chunk autoregression
    while still requiring a SINGLE forward pass (the K slots are computed
    jointly via a triangular attention mask).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class ParallelChunkHeads(nn.Module):
    """K-slot parallel prediction heads anchored at a hidden state.

    Args:
        hidden_size: LLM hidden dim H.
        num_slots: K — max tokens predicted per anchor.
        depth_layers: Number of transformer layers in the depth path. Default 1.
        num_heads: Attention heads in the depth transformer. Default 8.
        ffn_mult: FFN expansion (hidden → ffn_mult*hidden). Default 4.
        dropout: Dropout in the depth transformer. Default 0.0.
    """

    def __init__(
        self,
        hidden_size: int,
        num_slots: int,
        depth_layers: int = 1,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_slots = num_slots

        # Learned per-slot embeddings. Added to the anchor hidden state so each
        # slot starts from a distinguishable feature.
        self.slot_embeds = nn.Parameter(torch.empty(num_slots, hidden_size))
        nn.init.normal_(self.slot_embeds, mean=0.0, std=0.02)

        # Depth transformer over the K slots. We use TransformerEncoderLayer
        # with a causal mask to allow slot k to attend to slots 0..k.
        layers = []
        for _ in range(max(depth_layers, 0)):
            layers.append(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * ffn_mult,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )
        self.depth_layers = nn.ModuleList(layers)

        # Pre-built causal mask buffer (K, K). True = mask out (PyTorch convention
        # for attn_mask in TransformerEncoderLayer when boolean: True means
        # "ignore"). We register a buffer so it follows .to(device/dtype).
        causal = torch.triu(torch.ones(num_slots, num_slots, dtype=torch.bool), diagonal=1)
        self.register_buffer("_slot_causal_mask", causal, persistent=False)

    def forward(self, anchor_hidden: Tensor, lm_head: nn.Module) -> Tensor:
        """Compute K-slot logits for each anchor.

        Args:
            anchor_hidden: (M, H) hidden states gathered at the M anchor positions.
            lm_head: The model's lm_head (Linear or HF tied head). Called as
                ``lm_head(slot_hidden)`` and expected to return logits over the
                LLM vocab. Re-using the model's lm_head ties weights and avoids
                growing the model by K*H*V parameters.

        Returns:
            (M, K, V) logits.
        """
        if anchor_hidden.dim() != 2:
            raise ValueError(
                f"anchor_hidden must be (M, H); got shape {tuple(anchor_hidden.shape)}"
            )
        M, H = anchor_hidden.shape
        if H != self.hidden_size:
            raise ValueError(
                f"anchor_hidden hidden size {H} != ParallelChunkHeads.hidden_size {self.hidden_size}"
            )
        K = self.num_slots

        # Broadcast anchor across K slots and add slot embeddings.
        # slot_embeds: (K, H), anchor: (M, 1, H), → (M, K, H)
        slot_input = anchor_hidden.unsqueeze(1) + self.slot_embeds.unsqueeze(0).to(anchor_hidden.dtype)

        # Causal mask in the dtype/device of the slot_input.
        # nn.TransformerEncoderLayer accepts a boolean mask where True = masked.
        attn_mask = self._slot_causal_mask
        if attn_mask.device != slot_input.device:
            attn_mask = attn_mask.to(slot_input.device)

        x = slot_input
        for layer in self.depth_layers:
            x = layer(x, src_mask=attn_mask)

        # Apply the model's lm_head per slot. lm_head is a Linear-like module of
        # shape (V, H); calling lm_head(x) where x is (M, K, H) returns (M, K, V).
        logits = lm_head(x)
        return logits
