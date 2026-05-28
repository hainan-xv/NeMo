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

"""Helpers for chunk-local audio attention (``use_chunk_local_audio_attn``).

This module is intentionally self-contained and does NOT share code with the
separated-position implementation (``use_modality_position_ids``).

Design (v2)
-----------
* Every token in a streaming-STT sequence belongs to a *chunk*, identified by a
  per-token ``chunk_id``:

  - Tokens before the first audio span (system prompt, and the user-header
    tokens of the very first chunk) get ``chunk_id = -1``.
  - The K-th contiguous audio span and every token from that span up to (but
    not including) the next audio span — i.e. the K-th audio, its assistant
    template, its assistant content, its trailing ``<|im_end|>\\n``, and the
    user-header tokens of chunk K+1 — share ``chunk_id = K``.

  This is exactly ``cumsum(audio-span-starts) - 1`` per row, so it is fully
  derivable from the audio mask alone with no dataset-side cooperation.

* Attention mask (token ``i`` attends to token ``j``)::

      mask[i, j] = (j <= i)                                              # causal
                   AND attention_mask[j] == 1                            # key is valid
                   AND NOT ( is_audio[j]
                             AND chunk_id[i] - chunk_id[j] >= N )

  where ``N = num_visible_audio_chunks`` (1 = current chunk only, 2 = current
  plus previous chunk, ...). The default ``N = 1`` reproduces "only the current
  chunk's audio is visible to the LLM". Non-audio keys are always visible to
  any causal query regardless of chunk.

* Position IDs — **two independent contiguous counters per row**:

  - Audio tokens get ``audio_position_id = (cumsum of valid audio frames in
    row up to and including this token) - 1``. So for two audio chunks of 12
    frames each, the audio positions across the row are ``0, 1, …, 23``.
  - Non-audio valid tokens get ``text_position_id = (cumsum of valid non-audio
    tokens in row up to and including this token) - 1``. The system prompt
    occupies ``[0, S-1]`` and text continues contiguously across chunks.
  - The two counters share the RoPE position space (no offset). Audio and
    text content are distinguishable from their embeddings; the position
    collision in RoPE phase is therefore acceptable, and it keeps both
    counters in the most in-distribution range for Qwen3.
  - Pad positions get ``position_id = 0`` (they are masked out anyway).

Convention caveats
------------------
Because ``chunk_id`` is derived from audio spans, the user-header tokens for
chunk K+1 land in chunk K. The practical effect on training is negligible:
at inference the user-header is *fed* (not predicted), and its KV-cache K/V
values are linear projections of its embedding — independent of attention
masking. We accept this slight mismatch in exchange for keeping the helper
purely tensor-derived from ``input_ids`` / ``audio_mask`` without dataset
plumbing.

All helpers are fully vectorized over the batch and sequence dimensions; no
Python loops touch per-sample or per-token data.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_chunk_ids(
    is_audio: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Derive per-token ``chunk_id`` from the audio mask.

    Args:
        is_audio: ``(B, L)`` bool tensor, ``True`` where the token is an audio
            placeholder.
        attention_mask: optional ``(B, L)`` ``{0, 1}`` tensor marking valid
            (non-pad) positions. When provided, pad positions are excluded
            from chunk-start detection so a padded run does not spuriously
            increment ``chunk_id``.

    Returns:
        ``chunk_id``: ``(B, L)`` ``long`` tensor. ``-1`` for tokens before
        any audio span in a row; ``K`` for tokens from audio span ``K``
        onward up to (but not including) audio span ``K+1``.
    """
    if attention_mask is not None:
        is_audio = is_audio & attention_mask.to(torch.bool)
    prev_audio = F.pad(is_audio[:, :-1], (1, 0), value=False)
    audio_starts = is_audio & ~prev_audio
    return audio_starts.long().cumsum(dim=1) - 1


def build_chunk_local_attention_bias(
    chunk_id: torch.Tensor,
    is_audio: torch.Tensor,
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    num_visible_audio_chunks: int = 1,
) -> torch.Tensor:
    """Build the ``(B, 1, L, L)`` additive attention bias for HF self-attention.

    Args:
        chunk_id: ``(B, L)`` ``long``, per-token chunk id (see
            :func:`build_chunk_ids`).
        is_audio: ``(B, L)`` bool.
        attention_mask: ``(B, L)`` ``{0, 1}``, marks valid (non-pad) positions.
        dtype: dtype of the bias tensor (use the LLM's compute dtype, e.g.
            ``torch.bfloat16``).
        num_visible_audio_chunks: number ``N >= 1`` of recent audio chunks
            visible to the LLM. ``N = 1`` means a query can only attend to
            audio from its own chunk; ``N = 2`` adds the previous chunk's
            audio; etc. Non-audio keys (system prompt, text, templates,
            ``<|im_end|>``) are always visible regardless of this setting.

    Returns:
        ``bias``: ``(B, 1, L, L)`` tensor with ``0`` at allowed ``(i, j)``
        pairs and ``finfo(dtype).min`` at disallowed pairs. The head axis is
        broadcast-ready for HF self-attention; HF adds the bias directly to
        the pre-softmax scores.
    """
    if num_visible_audio_chunks < 1:
        raise ValueError(
            f"num_visible_audio_chunks must be >= 1, got {num_visible_audio_chunks}"
        )

    B, L = chunk_id.shape
    device = chunk_id.device

    causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))

    chunk_delta = chunk_id.unsqueeze(2) - chunk_id.unsqueeze(1)  # (B, L, L): query - key
    audio_key = is_audio.unsqueeze(1).expand(B, L, L)  # broadcast over queries
    kill_old_chunk_audio = audio_key & (chunk_delta >= num_visible_audio_chunks)
    valid_key = attention_mask.to(torch.bool).unsqueeze(1).expand(B, L, L)

    allowed = causal.unsqueeze(0) & valid_key & ~kill_old_chunk_audio  # (B, L, L)

    bias = torch.zeros((B, L, L), dtype=dtype, device=device)
    bias.masked_fill_(~allowed, torch.finfo(dtype).min)
    return bias.unsqueeze(1)


def build_chunk_local_inference_bias(
    chunk_id_history: torch.Tensor,
    is_audio_history: torch.Tensor,
    attention_mask_history: torch.Tensor,
    chunk_id_new: torch.Tensor,
    is_audio_new: torch.Tensor,
    dtype: torch.dtype,
    num_visible_audio_chunks: int = 1,
) -> torch.Tensor:
    """Build the ``(B, 1, L_q, L_k)`` chunk-local additive bias for streaming inference.

    Unlike :func:`build_chunk_local_attention_bias` (used in training where Q
    and K are the same sequence), inference attends *new* queries against a
    *history* of cached K/V tokens plus the new tokens themselves.

    Args:
        chunk_id_history: ``(B, L_past)`` ``long``, chunk id of each cached
            token. Use ``-1`` for tokens that precede any audio (e.g., system
            prompt and left-padding slots).
        is_audio_history: ``(B, L_past)`` bool.
        attention_mask_history: ``(B, L_past)`` ``{0, 1}``, ``1`` for valid
            cached tokens, ``0`` for left-padding slots.
        chunk_id_new: ``(B, L_q)`` ``long``, chunk id of each new query token.
        is_audio_new: ``(B, L_q)`` bool, ``True`` at audio-frame positions in
            the new tokens.
        dtype: dtype of the bias tensor.
        num_visible_audio_chunks: ``N >= 1``, same semantics as the training
            helper.

    Returns:
        ``bias``: ``(B, 1, L_q, L_k)`` where ``L_k = L_past + L_q``.
    """
    if num_visible_audio_chunks < 1:
        raise ValueError(
            f"num_visible_audio_chunks must be >= 1, got {num_visible_audio_chunks}"
        )

    B, L_past = chunk_id_history.shape
    L_q = chunk_id_new.shape[1]
    L_k = L_past + L_q
    device = chunk_id_history.device

    all_chunk_ids = torch.cat([chunk_id_history, chunk_id_new], dim=1)  # (B, L_k)
    all_is_audio = torch.cat([is_audio_history, is_audio_new], dim=1)  # (B, L_k)
    new_valid = torch.ones((B, L_q), dtype=attention_mask_history.dtype, device=device)
    all_valid = torch.cat([attention_mask_history, new_valid], dim=1).to(torch.bool)  # (B, L_k)

    q_abs = torch.arange(L_past, L_past + L_q, device=device).unsqueeze(1)  # (L_q, 1)
    k_idx = torch.arange(L_k, device=device).unsqueeze(0)  # (1, L_k)
    causal = k_idx <= q_abs  # (L_q, L_k)

    chunk_delta = chunk_id_new.unsqueeze(2) - all_chunk_ids.unsqueeze(1)  # (B, L_q, L_k)
    audio_key = all_is_audio.unsqueeze(1).expand(B, L_q, L_k)  # broadcast over queries
    kill_old_chunk_audio = audio_key & (chunk_delta >= num_visible_audio_chunks)
    valid_key = all_valid.unsqueeze(1).expand(B, L_q, L_k)

    allowed = causal.unsqueeze(0) & valid_key & ~kill_old_chunk_audio  # (B, L_q, L_k)

    bias = torch.zeros((B, L_q, L_k), dtype=dtype, device=device)
    bias.masked_fill_(~allowed, torch.finfo(dtype).min)
    return bias.unsqueeze(1)


def build_chunk_local_position_ids(
    is_audio: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Build per-token ``position_ids`` with one contiguous counter per modality.

    Two independent counters advance left-to-right per row:

    * Audio tokens get the count of valid audio frames in the row up to and
      including this token, minus 1. For two audio chunks of 12 frames each,
      the audio positions across the row are ``0, 1, …, 23``.
    * Non-audio valid tokens (system prompt, templates, text content,
      ``<|im_end|>``) get the count of valid non-audio tokens in the row up
      to and including this token, minus 1. System prompt occupies
      ``[0, S-1]``, text continues contiguously across chunks.
    * Pad positions get ``position_id = 0`` (they are masked out anyway).

    Args:
        is_audio: ``(B, L)`` bool.
        attention_mask: ``(B, L)`` ``{0, 1}``.

    Returns:
        ``position_ids``: ``(B, L)`` ``long``.
    """
    is_valid = attention_mask.to(torch.bool)
    is_audio_valid = is_audio & is_valid
    is_text_valid = is_valid & ~is_audio
    audio_pos = is_audio_valid.long().cumsum(dim=1) - 1  # -1 at non-audio positions in a streak
    text_pos = is_text_valid.long().cumsum(dim=1) - 1  # -1 at audio / pad positions in a streak
    position_ids = torch.where(is_audio.to(torch.bool), audio_pos, text_pos)
    return position_ids.clamp(min=0)
