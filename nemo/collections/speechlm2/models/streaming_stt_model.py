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
import math
import os
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf import DictConfig, ListConfig
from torch import Tensor, nn
from torch.distributed.tensor.parallel import loss_parallel
from transformers import AutoModel, GenerationConfig

try:
    # Used to build per-stage causal masks when driving the Qwen layer stack
    # manually for two-stream (last-layer) fusion.
    from transformers.masking_utils import create_causal_mask
except Exception:  # noqa: BLE001
    create_causal_mask = None

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # noqa: BLE001
    DynamicCache = None

from nemo.collections.asr.inference.streaming.buffering.cache_feature_bufferer import BatchedCacheFeatureBufferer
from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.utils.context_manager import CacheAwareContext
from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.streaming_stt_dataset import (
    AUDIO_TOKEN_IDX,
    IGNORE_INDEX,
    StreamingSTTBatch,
    StreamingSTTDataConfig,
    StreamingSTTDataset,
    build_compact_turn_markers,
    decode_with_blank,
    get_llm_messages_for_sample,
    parse_chat_template_ids,
)
from nemo.collections.speechlm2.parts.alignments import ForcedAligner, get_word_alignments_for_batch
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.wer import WER
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf, move_embedding, setup_perception
from nemo.collections.speechlm2.parts.utils import freeze_module, to_dataclass, unfreeze_module
from nemo.utils import logging


def token_in_vocab(token: str, tokenizer: AutoTokenizer) -> bool:
    token_pieces = tokenizer.text_to_tokens(token)
    if len(token_pieces) == 1:
        return True
    else:
        return False


def interleave_embeddings(
    input_tokens: Tensor,
    audio_mask: Tensor,
    text_embeds: Tensor,
    audio_embs: Tensor,
    pad_id: int,
) -> dict[str, Tensor]:
    """
    Merge pre-computed text and audio embeddings into a single sequence,
    guided by ``audio_mask``.

    All operations are fully batched (no Python loops over batch items):

    1. ``cumsum`` on the audio mask gives a 0-based frame index per audio position.
    2. ``torch.gather`` selects the correct audio frame for each position.
    3. ``torch.where`` picks audio or text embeddings per position.

    Args:
        input_tokens: (B, L) token IDs — only used to derive the attention mask
            (non-``pad_id`` positions).
        audio_mask: (B, L) bool — True at ``AUDIO_TOKEN_IDX`` positions.
        text_embeds: (B, L, H) embeddings produced by the text embedding layer.
            Values at audio positions are unused and may be arbitrary.
        audio_embs: (B, T_enc, H) frame-level embeddings from the audio encoder.
            If there are more audio tokens than encoder frames (last-chunk
            ceiling), the tensor is zero-padded automatically.
        pad_id: text token ID used for padding — these positions get
            ``attention_mask = False``.

    Returns:
        dict with:
            ``input_embeds`` — (B, L, H) interleaved embeddings.
            ``attention_mask`` — (B, L) bool, False only at padding positions.
    """
    B, L = input_tokens.shape

    if not audio_mask.any():
        # Pure text — nothing to interleave.
        attention_mask = input_tokens != pad_id
        return {"input_embeds": text_embeds, "attention_mask": attention_mask}

    # Sequential 0-based frame index for each audio-token position.
    frame_indices = audio_mask.long().cumsum(dim=1) - 1  # (B, L)

    # Pad encoder output if the dataset produced more audio tokens than
    # the encoder returned (last chunk ceiling).
    max_frame_idx = frame_indices.max().item()
    T_enc = audio_embs.shape[1]
    if max_frame_idx >= T_enc:
        audio_embs = F.pad(audio_embs, (0, 0, 0, max_frame_idx - T_enc + 1))

    # Gather the correct audio frame for every position in L.
    H = audio_embs.shape[2]
    gather_idx = frame_indices.clamp(min=0).unsqueeze(-1).expand(B, L, H)
    audio_at_all_pos = torch.gather(audio_embs, dim=1, index=gather_idx)  # (B, L, H)

    # Merge: audio embeddings at audio positions, text embeddings elsewhere.
    embeds = torch.where(audio_mask.unsqueeze(-1), audio_at_all_pos, text_embeds)

    # Attend to every non-padding position.
    # pad_id is ≥ 0 and AUDIO_TOKEN_IDX is −100, so this is safe.
    attention_mask = input_tokens != pad_id  # (B, L)

    return {"input_embeds": embeds, "attention_mask": attention_mask}


def audio_positions_and_chunk_ids(input_tokens: Tensor) -> tuple[Tensor, Tensor]:
    """Identify audio positions and assign a per-chunk id from ``input_tokens``.

    Each contiguous run of ``AUDIO_TOKEN_IDX`` is one chunk. ``chunk_id`` counts
    those runs (1-based) and only carries meaning at audio positions; text
    positions inherit the running count but are never compared as audio.

    Returns:
        ``(is_audio, chunk_id)`` — both ``(B, L)``; ``is_audio`` bool, ``chunk_id`` long.
    """
    is_audio = input_tokens == AUDIO_TOKEN_IDX  # (B, L)
    prev_is_audio = F.pad(is_audio[:, :-1], (1, 0), value=False)
    run_start = is_audio & ~prev_is_audio
    chunk_id = torch.cumsum(run_start.long(), dim=1)
    return is_audio, chunk_id


def build_chunk_restricted_mask(
    key_is_audio: Tensor,
    key_chunk_id: Tensor,
    key_valid: Tensor,
    query_is_audio: Tensor,
    query_chunk_id: Tensor,
    query_abs_pos: Tensor,
    key_abs_pos: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    """Build a 4D additive attention mask enforcing chunk-restricted audio attention.

    A position is allowed to attend iff: it is causal (``key_abs_pos <=
    query_abs_pos``), the key is not padding, and it is NOT the case that a
    **text** query attends to an **audio** key from a different chunk. In other
    words, text (transcription) queries may attend to all text/system tokens and
    only their own chunk's audio frames; audio queries keep full causal
    attention. Cross-chunk information therefore flows only through the emitted
    text.

    A text token's chunk id is the chunk of the audio run it follows (see
    :func:`audio_positions_and_chunk_ids`), so the transcription emitted after
    chunk ``k`` is bound to chunk ``k``'s audio.

    All tensors are batched: ``*_is_audio``/``key_valid`` bool, ``*_chunk_id``/
    ``*_abs_pos`` long. Query tensors are ``(B, Lq)`` and key tensors ``(B, Lk)``.

    Returns:
        Additive mask ``(B, 1, Lq, Lk)`` with ``0`` where attention is allowed and
        ``torch.finfo(dtype).min`` where it is blocked.
    """
    causal = query_abs_pos[:, :, None] >= key_abs_pos[:, None, :]  # (B, Lq, Lk)
    valid = key_valid[:, None, :]  # (B, 1, Lk)
    query_is_text = ~query_is_audio
    text_to_cross_chunk_audio = (
        query_is_text[:, :, None]
        & key_is_audio[:, None, :]
        & (query_chunk_id[:, :, None] != key_chunk_id[:, None, :])
    )
    allowed = causal & valid & ~text_to_cross_chunk_audio  # (B, Lq, Lk)

    additive = torch.zeros_like(allowed, dtype=dtype)
    additive = additive.masked_fill(~allowed, torch.finfo(dtype).min)
    return additive.unsqueeze(1)  # (B, 1, Lq, Lk)


def build_training_chunk_restricted_mask(input_tokens: Tensor, pad_id: int, dtype: torch.dtype) -> Tensor:
    """Full-sequence chunk-restricted 4D mask for teacher-forced training.

    ``query`` and ``key`` axes are the same length ``L``; absolute positions are
    ``0..L-1``.
    """
    is_audio, chunk_id = audio_positions_and_chunk_ids(input_tokens)
    key_valid = input_tokens != pad_id
    B, L = input_tokens.shape
    abs_pos = torch.arange(L, device=input_tokens.device).unsqueeze(0).expand(B, L)
    return build_chunk_restricted_mask(
        key_is_audio=is_audio,
        key_chunk_id=chunk_id,
        key_valid=key_valid,
        query_is_audio=is_audio,
        query_chunk_id=chunk_id,
        query_abs_pos=abs_pos,
        key_abs_pos=abs_pos,
        dtype=dtype,
    )


def _run_decoder_layer(layer, hidden, attn_mask, position_ids, position_embeddings):
    """Call one HF decoder layer, tolerating tuple/tensor return conventions."""
    out = layer(
        hidden,
        attention_mask=attn_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        past_key_value=None,
        use_cache=False,
    )
    return out[0] if isinstance(out, tuple) else out


def _causal_additive_mask(valid_mask: Tensor, dtype: torch.dtype) -> Tensor:
    """Build a (B, 1, S, S) additive causal mask honoring a (B, S) key-validity mask.

    Uses ``finfo(dtype).min`` (not ``-inf``) for blocked entries so a fully-blocked
    query row degrades to uniform attention instead of NaN.
    """
    B, S = valid_mask.shape
    device = valid_mask.device
    idx = torch.arange(S, device=device)
    causal = idx[:, None] >= idx[None, :]  # (S, S): key <= query
    allowed = causal[None] & valid_mask[:, None, :]  # (B, S, S)
    additive = torch.zeros(B, S, S, dtype=dtype, device=device)
    additive = additive.masked_fill(~allowed, torch.finfo(dtype).min)
    return additive.unsqueeze(1)  # (B, 1, S, S)


def two_stream_llm_forward(
    *,
    layers,
    norm,
    rotary_emb,
    lm_head,
    inputs_embeds: Tensor,
    audio_mask: Tensor,
    valid_mask: Tensor,
    num_fusion_layers: int = 1,
    compute_logits: bool = True,
) -> dict[str, Tensor]:
    """Two-stream last-layer fusion over a full (teacher-forced) sequence.

    The text stream (all non-audio, non-pad positions) is packed into a dense,
    contiguous sequence and run through the lower ``layers[:-num_fusion_layers]``
    alone, with contiguous text positions and a causal mask. The resulting text
    hidden states are then scattered back to their original positions and
    interleaved with the RAW audio encoder embeddings (which never passed through
    the lower layers). The top ``num_fusion_layers`` layers run over this full
    interleaved sequence (original absolute positions, causal mask) so text
    queries fuse with audio, followed by ``norm`` + ``lm_head``.

    ``num_fusion_layers=1`` (default) fuses only in the final layer. Larger values
    move the interleave point earlier (e.g. num_fusion_layers=4 interleaves at
    layer n-4, so the last 4 layers are cross-modal).

    Args:
        layers: ``nn.ModuleList`` of decoder layers (LoRA-injected linears are fine).
        norm: final RMSNorm.
        rotary_emb: rotary embedding module (called as ``rotary_emb(x, position_ids)``).
        lm_head: LM projection.
        inputs_embeds: (B, L, H) interleaved embeddings — audio-encoder embeds at
            audio positions, token embeds at text positions.
        audio_mask: (B, L) bool, True at audio positions.
        valid_mask: (B, L) bool, True at non-pad positions.
        compute_logits: when False, only ``hidden_states`` is returned.

    Returns:
        dict with ``logits`` (B, L, V) [when ``compute_logits``] and
        ``hidden_states`` (B, L, H) — the post-norm last hidden state.
    """
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    B, L, H = inputs_embeds.shape
    text_mask = valid_mask & ~audio_mask  # (B, L)

    # --- Stage 1: pack text into a dense per-row sequence, run layers[:-1] ---
    n_text = text_mask.long().sum(dim=1)  # (B,)
    T = int(n_text.max().item()) if L > 0 else 0
    text_rank = torch.cumsum(text_mask.long(), dim=1) - 1  # (B, L); meaningful at text pos
    b_idx, l_idx = text_mask.nonzero(as_tuple=True)
    dst_col = text_rank[b_idx, l_idx]

    packed = inputs_embeds.new_zeros(B, max(T, 1), H)
    packed[b_idx, dst_col] = inputs_embeds[b_idx, l_idx]
    text_valid = torch.arange(max(T, 1), device=device)[None, :] < n_text[:, None]  # (B, T)
    pos1 = torch.arange(max(T, 1), device=device)[None, :].expand(B, max(T, 1))
    mask1 = _causal_additive_mask(text_valid, dtype)
    pos_emb1 = rotary_emb(packed, pos1)

    n_layers = len(layers)
    n_fusion = max(1, min(int(num_fusion_layers), n_layers))
    text_layers = layers[: n_layers - n_fusion]
    fusion_layers = layers[n_layers - n_fusion:]

    h = packed
    for layer in text_layers:
        h = _run_decoder_layer(layer, h, mask1, pos1, pos_emb1)

    # --- Stage 2: scatter text hidden back, interleave raw audio, run fusion layers ---
    h_full = inputs_embeds.clone()  # audio positions keep raw encoder embeds
    h_full[b_idx, l_idx] = h[b_idx, dst_col]
    pos2 = (torch.cumsum(valid_mask.long(), dim=1) - 1).clamp(min=0)  # (B, L)
    mask2 = _causal_additive_mask(valid_mask, dtype)
    pos_emb2 = rotary_emb(h_full, pos2)
    for layer in fusion_layers:
        h_full = _run_decoder_layer(layer, h_full, mask2, pos2, pos_emb2)
    h_last = norm(h_full)

    ans: dict[str, Tensor] = {"hidden_states": h_last}
    if compute_logits:
        ans["logits"] = lm_head(h_last)
    return ans


def _run_decoder_layer_cached(layer, hidden, attn_mask, position_ids, position_embeddings, cache, cache_position):
    """Call one HF decoder layer with a KV cache (incremental decode)."""
    out = layer(
        hidden,
        attention_mask=attn_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
    )
    return out[0] if isinstance(out, tuple) else out


def _block_causal_additive_mask(q_len: int, past_len: int, dtype: torch.dtype, device, batch: int) -> Tensor:
    """(batch, 1, q_len, past_len+q_len) additive causal mask for a new block.

    New query ``i`` (0-based within the block) may attend to all ``past_len``
    cached keys plus new keys ``0..i``. No padding (two-stream fixed-chunk
    inference uses a shared, fully-valid sequence). ``finfo.min`` for blocked.
    """
    kv = past_len + q_len
    qi = torch.arange(q_len, device=device)
    kj = torch.arange(kv, device=device)
    allowed = kj[None, :] <= (past_len + qi)[:, None]  # (q_len, kv)
    additive = torch.zeros(q_len, kv, dtype=dtype, device=device)
    additive = additive.masked_fill(~allowed, torch.finfo(dtype).min)
    return additive[None, None].expand(batch, 1, q_len, kv).contiguous()


def two_stream_cached_step(
    *,
    layers,
    norm,
    rotary_emb,
    lm_head,
    input_embeds: Tensor,
    input_is_audio: Tensor,
    text_cache,
    last_cache,
    text_len: int,
    full_len: int,
    num_fusion_layers: int = 1,
    compute_logits: bool = True,
) -> tuple[Optional[Tensor], int, int]:
    """Incremental (KV-cached) two-stream step — the fast counterpart of
    :func:`two_stream_llm_forward`.

    Only the TEXT columns of the new block are pushed through ``layers[:-1]``
    (advancing ``text_cache`` with contiguous text positions); the raw audio
    columns skip the lower layers entirely. The final layer then runs over ALL
    new columns (advancing ``last_cache`` at absolute positions), fusing text and
    audio, followed by ``norm`` + ``lm_head``. This reproduces the full-recompute
    forward exactly (causal), at O(1) layer-passes per new token instead of a
    full re-forward.

    Assumes a batch-uniform audio/text column pattern (true for the static
    fixed-chunk path, where all streams process the same turn template in
    lockstep) and no left padding (shared system prompt).

    Args:
        input_embeds: (B, q, H) new block — audio-encoder embeds at audio
            columns, token embeds at text columns.
        input_is_audio: (B, q) bool (uniform across the batch dim).
        text_cache / last_cache: ``DynamicCache`` for ``layers[:-1]`` / ``layers[-1]``.
        text_len / full_len: positions already cached in each.

    Returns:
        ``(logits (B, q, V) | None, new_text_len, new_full_len)``.
    """
    device = input_embeds.device
    dtype = input_embeds.dtype
    B, q, H = input_embeds.shape
    col_is_audio = input_is_audio[0]  # (q,)
    assert bool((input_is_audio == col_is_audio.unsqueeze(0)).all()), (
        "two_stream_cached_step requires a batch-uniform audio/text pattern"
    )
    text_cols = (~col_is_audio).nonzero(as_tuple=True)[0]  # (t,)
    t = int(text_cols.numel())

    n_layers = len(layers)
    n_fusion = max(1, min(int(num_fusion_layers), n_layers))
    text_layers = layers[: n_layers - n_fusion]
    fusion_layers = layers[n_layers - n_fusion:]

    # --- Stage 1: lower (text-only) layers over the new TEXT columns only ---
    if t > 0:
        text_block = input_embeds.index_select(1, text_cols)  # (B, t, H)
        text_positions = torch.arange(text_len, text_len + t, device=device)
        pos_ids_text = text_positions.unsqueeze(0).expand(B, t)
        mask_text = _block_causal_additive_mask(t, text_len, dtype, device, B)
        pos_emb_text = rotary_emb(text_block, pos_ids_text)
        h = text_block
        for layer in text_layers:
            h = _run_decoder_layer_cached(layer, h, mask_text, pos_ids_text, pos_emb_text, text_cache, text_positions)
        text_hidden = h  # (B, t, H)
        new_text_len = text_len + t
    else:
        text_hidden = None
        new_text_len = text_len

    # --- Stage 2: fusion layers over ALL new columns (interleave raw audio) ---
    last_input = input_embeds.clone()
    if t > 0:
        last_input.index_copy_(1, text_cols, text_hidden)
    full_positions = torch.arange(full_len, full_len + q, device=device)
    pos_ids_full = full_positions.unsqueeze(0).expand(B, q)
    mask_full = _block_causal_additive_mask(q, full_len, dtype, device, B)
    pos_emb_full = rotary_emb(last_input, pos_ids_full)
    for layer in fusion_layers:
        last_input = _run_decoder_layer_cached(
            layer, last_input, mask_full, pos_ids_full, pos_emb_full, last_cache, full_positions
        )
    h_last = norm(last_input)
    new_full_len = full_len + q

    logits = lm_head(h_last) if compute_logits else None
    return logits, new_text_len, new_full_len


def _repr_chunk_size(chunk_size) -> int:
    """Representative scalar chunk size for a config value that may be a list.

    Returns the longest entry when ``chunk_size`` is a list/tuple/ListConfig
    (multi chunk-size training); otherwise the scalar unchanged. Used to build
    the default inference turn template and to dispatch the generation mode.
    """
    if isinstance(chunk_size, (list, tuple, ListConfig)):
        return max(int(x) for x in chunk_size)
    return int(chunk_size)


@dataclass
class StreamingSTTModelConfig:
    pretrained_llm: str
    pretrained_asr: str
    load_llm_weights: bool
    blank_token: str
    load_asr_weights: bool
    freeze_speech_encoder: bool
    freeze_modality_adapter: bool
    freeze_modality_proj: bool
    freeze_llm_model: bool
    freeze_llm_head: bool
    freeze_embed_tokens: bool
    # ``> 0`` fixed chunking, ``0`` dynamic, ``< 0`` offline. May also be a list
    # of positive ints (e.g. [2,4,7,10,14,28]) for multi chunk-size training:
    # one size is drawn per batch by the dataset, and the encoder look-ahead is
    # matched per batch as [att_context_size[0], chunk_size - 1]. Inference then
    # defaults to the longest size (override via generate(chunk_size_override=...)).
    chunk_size: Union[int, List[int]]
    audio_tag: str = "<audio>"
    att_context_size: Optional[List[int]] = None
    # When True (default), the encoder's attention look-ahead is coupled to the
    # per-batch LLM chunk size as [att_context_size[0], chunk_size - 1] (multi
    # chunk-size training). When False, the encoder keeps a FIXED look-ahead of
    # ``att_context_size`` regardless of the chunk size the dataset drew — e.g.
    # set att_context_size=[70, 0] for a strictly-causal encoder while the LLM
    # still trains on variable chunk sizes.
    couple_encoder_lookahead_to_chunk: bool = True
    audio_pad_to: Optional[int] = None
    sample_rate: int = 16000
    frame_length_in_secs: float = 0.08
    blank_loss_weight: float = 1.0
    log_every_n_steps: int = 10
    # Opt-in diagnostics for batch composition and loss decomposition. The
    # aggregate loss and learning rate are always logged.
    log_detailed_train_metrics: bool = False
    # Opt-in per-step check that input/target token ids are within the embedding
    # and LM-head vocab ranges. Turns an opaque async CUDA device-side assert
    # (out-of-range embedding index / cross-entropy target) into a precise Python
    # error. Adds a per-step CPU sync, so enable only for debugging.
    debug_validate_tokens: bool = False
    # Validation uses real autoregressive streaming decoding. None chooses the
    # number of frames per chunk for fixed chunking (or 64 otherwise).
    val_max_new_tokens_per_chunk: Optional[int] = None
    # For multi chunk-size training, decode validation at THIS fixed chunk size
    # (encoder frames) instead of the longest candidate. The longest candidate's
    # look-ahead (chunk-1) can be an unsupported/slow cache-aware streaming config
    # and stall the decode. None -> 14 if it's a candidate, else the longest.
    # Ignored for scalar-chunk configs (they decode at their single chunk_size).
    val_chunk_size: Optional[int] = None
    # Optional rank-zero autoregressive preview from the training stream.
    # The interval is measured in optimizer steps, not microbatches.
    train_decode_every_n_steps: int = 0
    train_decode_max_new_tokens_per_chunk: Optional[int] = None
    dtype: str = "bfloat16"
    # --- Chunk-restricted audio attention (fixed chunking only) ---
    # When True, a TEXT (transcription) query in the LLM may attend to all
    # text/system tokens but only to its own chunk's audio frames — not to audio
    # frames of earlier chunks. Audio-frame queries keep full causal attention.
    # This forces cross-chunk acoustic history to flow through the emitted text.
    # Applied identically in training and fixed-chunk inference via a custom 4D
    # additive mask. Only valid for chunk_size > 0.
    restrict_audio_to_own_chunk: bool = False
    # --- Two-stream last-layer fusion (fixed chunking only) ---
    # When True, the LLM processes the TEXT stream alone through layers[:-1]
    # (a contiguous text-only sequence with its own causal mask and positions).
    # Only in the FINAL layer are the audio-encoder frames interleaved back into
    # the sequence (at their original chunk positions) so text queries can attend
    # to audio via a single cross-modal layer, before final norm + LM head.
    # Audio frames therefore never pass through the lower layers, and the text
    # lower layers see a dense text-only sequence. Applied identically in training
    # and fixed-chunk streaming inference (two KV caches). Only valid for
    # chunk_size > 0 and mutually exclusive with use_chunk_classifier /
    # restrict_audio_to_own_chunk.
    two_stream_last_layer: bool = False
    # Number of TOP LLM layers that process the interleaved (text+audio) sequence
    # under two-stream fusion. 1 (default) fuses only in the final layer; larger
    # moves the interleave point earlier (e.g. 4 -> interleave at layer n-4, so the
    # last 4 layers are cross-modal). Must be >= 1 and leave >= 1 text-only layer.
    two_stream_num_fusion_layers: int = 1
    # --- Full fine-tuning of the top of the LLM stack (no LoRA) ---
    # When > 0 AND the LLM body is otherwise frozen (freeze_llm_model=True),
    # unfreeze the last N decoder layers so they train with full-rank weight
    # updates. Intended for two_stream_last_layer, where only the final layer
    # performs cross-modal fusion — training that one layer fully (instead of
    # LoRA) is both cheap and a natural fit. Leave 0 to disable.
    unfreeze_last_n_llm_layers: int = 0
    # Analogous to unfreeze_last_n_llm_layers, but for the speech encoder: when
    # > 0 AND the encoder is otherwise frozen (freeze_speech_encoder=True),
    # unfreeze the last N encoder (Conformer) layers for full-weight training.
    # The modality adapter / projection are governed separately by their own
    # freeze flags. Leave 0 to disable.
    unfreeze_last_n_encoder_layers: int = 0
    # --- Compact template ---
    # Compact template: use a write token to trigger text generation, and the EOS token
    # is automatically generated by the tokenizer.
    compact_template: bool = False
    write_token: str = "<|im_start|>"
    # --- Aux chunk-boundary classifier head ---
    # Master switch. Only valid in dynamic-chunking mode (chunk_size == 0).
    # When True, a small K-layer transformer head is built on top of the LLM's
    # last hidden state and trained with BCE at audio frame positions; the LM
    # head is no longer supervised at audio positions. When False (default),
    # the boundary decision falls back to the LM head signaling via the blank
    # / user_footer_first token (legacy behavior). The module is NOT built
    # unless this flag is True.
    use_chunk_classifier: bool = False
    chunk_classifier_loss_weight: float = 0.5
    chunk_classifier_num_layers: int = 2
    chunk_classifier_init_from_llm: bool = True
    chunk_classifier_threshold: float = 0.5
    chunk_classifier_use_at_inference: bool = False
    freeze_chunk_classifier: bool = False
    # Auto-balance the BCE: pos_weight = num_neg/num_pos per batch. Most audio
    # frames are "keep listening" (label=0); the few "emit" frames (label=1) get
    # drowned out without rebalancing.
    chunk_classifier_auto_balance: bool = True
    # Detach the LLM's last hidden state before feeding it to the aux backbone.
    # When True, the aux BCE loss does NOT propagate back into the LLM (body,
    # lm_head, or embed_tokens) — the LLM is supervised purely by the masked
    # LM CE on text positions, and the aux head learns on a fixed feature.
    # Recommended default for clean A/B against pure text-LLM training.
    chunk_classifier_stop_grad_to_llm: bool = True
    # When True, keep the LM-CE supervision at audio positions (blank /
    # user_footer_first targets) alongside the aux head's BCE. The LM head
    # learns the boundary signal too — useful when LoRA / lm_head needs more
    # gradient density per batch (default text-only positions can be sparse,
    # ~20% of input length, slowing convergence). When False (default), audio
    # positions are masked to IGNORE_INDEX in the LM CE — the aux head owns
    # the boundary decision exclusively.
    chunk_classifier_keep_lm_supervision_at_audio: bool = False


@dataclass
class StreamingState:
    """Holds the KV cache and other state for B streaming audio sessions.

    All tensors have batch dimension B (B=1 for single-stream inference).
    The LLM cache ``past_key_values`` has shape ``(layers, (B, heads, seq, dim))``
    for K and V.  The perception cache has batch dim on axis 1.
    """

    cache: tuple | None = None  # HF past_key_values with batch dim B
    generated_tokens: list[list[int]] = field(default_factory=list)  # B lists of per-chunk token IDs
    seq_lens: list[int] = field(default_factory=list)  # per-stream sequence lengths
    audio_cache: CacheAwareContext | None = None  # perception cache with batch dim B
    audio_feature_buffer: BatchedCacheFeatureBufferer | None = None
    attention_mask: Optional[Tensor] = None  # (B, seq_len) mask for left-padded prefill
    # (B, seq_len, H) running buffer of LLM last hidden states; used by the aux
    # chunk-boundary classifier when chunk_classifier_use_at_inference is True.
    # None when disabled — keeps the field cheap to always carry on the state.
    aux_hidden_buffer: Optional[Tensor] = None
    # Per-position bookkeeping for chunk-restricted audio attention. Only
    # populated when restrict_audio_to_own_chunk is True; otherwise None.
    is_audio: Optional[Tensor] = None  # (B, seq_len) bool: audio-frame positions
    chunk_id: Optional[Tensor] = None  # (B, seq_len) long: per-chunk id (audio positions)
    n_audio_chunks: int = 0  # number of audio chunks appended so far
    # Two-stream (last-layer fusion) inference buffers. When two_stream_last_layer
    # is on, the streaming path does NOT use ``cache``; instead it accumulates
    # every fed embedding and its audio flag here and recomputes the two-stream
    # forward over the full sequence each step (guaranteed parity with training).
    ts_embeds: Optional[Tensor] = None  # (B, seq_len, H) fed embeddings [recompute path]
    ts_is_audio: Optional[Tensor] = None  # (B, seq_len) bool: audio-frame positions [recompute path]
    # Two KV caches for the fast (incremental) two-stream path. ``ts_text_cache``
    # holds the text stream through layers[:-1] (text positions only); ``ts_last_cache``
    # holds the final layer over the full interleaved sequence. Counters track how
    # many positions each cache already holds.
    ts_text_cache: Any = None  # DynamicCache for layers[:-1]
    ts_last_cache: Any = None  # DynamicCache for layers[-1]
    ts_text_len: int = 0
    ts_full_len: int = 0
    batch_size: int = 1

    @property
    def seq_len(self) -> int:
        """Max seq_len across streams (= KV cache dimension)."""
        return max(self.seq_lens) if self.seq_lens else 0


class StreamingSTTModel(LightningModule, HFHubMixin):

    def __init__(
        self,
        cfg: dict,
        forced_aligner: Optional[ForcedAligner] = None,
        data_cfg: Optional[DictConfig] = None,
        dataset_cls=StreamingSTTDataset,
    ) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to StreamingSTTModel as a Python dict to support hyperparameter "
            f"serialization in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)
        self.core_cfg: StreamingSTTModelConfig = to_dataclass(StreamingSTTModelConfig, cfg)

        # --- Multi chunk-size ("contexts") ---
        # chunk_size may be a list (e.g. [2,4,7,10,14,28]) for multi chunk-size
        # training: the dataset draws one size per batch and the encoder look-ahead
        # is matched per batch to [att_context_size[0], chunk_size - 1]. We keep a
        # scalar representative (longest size) for inference/dispatch, and the raw
        # candidate list for logging. Read the RAW value from self.cfg (self.cfg
        # holds the model config dict directly) — everything else in the model uses
        # the scalar self._chunk_size_repr.
        _cs_raw = self.cfg.chunk_size
        self._chunk_size_candidates = (
            [int(x) for x in _cs_raw] if isinstance(_cs_raw, (list, tuple, ListConfig)) else None
        )
        self._chunk_size_repr = _repr_chunk_size(_cs_raw)
        if self._chunk_size_candidates is not None:
            if not self._chunk_size_candidates or any(x <= 0 for x in self._chunk_size_candidates):
                raise ValueError(
                    f"All chunk sizes in a list must be positive (fixed chunking), "
                    f"got {self._chunk_size_candidates}"
                )
            if self.core_cfg.att_context_size is None:
                logging.warning(
                    "chunk_size is a list but att_context_size is not set — the encoder's "
                    "attention look-ahead will NOT be matched to the per-batch chunk size."
                )
            logging.info(
                "Multi chunk-size training enabled: candidates=%s, inference default=%d",
                self._chunk_size_candidates,
                self._chunk_size_repr,
            )
        if self.core_cfg.att_context_size is not None and not self.core_cfg.couple_encoder_lookahead_to_chunk:
            logging.info(
                "FIXED encoder look-ahead: att_context_size=%s (decoupled from LLM chunk size%s)",
                list(self.core_cfg.att_context_size),
                "; strictly causal" if int(self.core_cfg.att_context_size[1]) == 0 else "",
            )

        # --- LLM ---
        self.tokenizer = AutoTokenizer(self.core_cfg.pretrained_llm, use_fast=True)
        self.llm = load_pretrained_hf(
            self.core_cfg.pretrained_llm,
            pretrained_weights=self.core_cfg.load_llm_weights,
        )

        # Ensure <blank> token is in the vocabulary.
        # Unescape Python escape sequences (e.g. "\\n" → "\n") because Hydra/OmegaConf
        # loads YAML strings literally without interpreting backslash escapes.
        # An empty blank_token ("") disables the blank mechanism entirely
        # (fixed chunking only — see StreamingSTTDataset for the guard).
        self.blank_token = self.core_cfg.blank_token.encode().decode('unicode_escape')

        if self.blank_token == "":
            logging.info("blank_token is empty: blank mechanism disabled")
        elif not token_in_vocab(self.blank_token, self.tokenizer):
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.blank_token]})
            self.llm.resize_token_embeddings(len(self.tokenizer.tokenizer))
            logging.info(f"Added blank token `{self.blank_token}` to tokenizer: {self.blank_token_id}")
        else:
            logging.info(f"Blank token `{str(self.blank_token)}` already in tokenizer: {self.blank_token_id}")

        # Compact-template write_token: only register if missing (default <|im_start|>
        # is already in Qwen3's vocab → uses pretrained embedding).
        if self.core_cfg.compact_template:
            wt = self.core_cfg.write_token
            if not token_in_vocab(wt, self.tokenizer):
                self.tokenizer.add_special_tokens({"additional_special_tokens": [wt]})
                self.llm.resize_token_embeddings(len(self.tokenizer.tokenizer))
                logging.info(f"compact_template: added write_token `{wt}` to tokenizer (random init)")
            else:
                logging.info(f"compact_template: using existing vocab token `{wt}` as write_token")

        # Separate embedding layer to avoid FSDP/TP conflicts (same pattern as SALM)
        self.embed_tokens = self.llm.model.embed_tokens
        del self.llm.model.embed_tokens

        # --- Speech encoder (perception module) ---
        self.perception = setup_perception(
            cfg=self.cfg,
            output_dim=self.llm.config.hidden_size,
            pretrained_asr=self.core_cfg.pretrained_asr,
            pretrained_weights=self.core_cfg.load_asr_weights,
            audio_pad_to=self.core_cfg.audio_pad_to,
            att_context_size=self.core_cfg.att_context_size,
        )

        # --- Aux chunk-boundary classifier (only built when enabled) ---
        # Only valid in dynamic-chunking mode (chunk_size == 0). When disabled,
        # no module is built and existing runs / checkpoints see no behavior
        # change. The boundary decision falls back to the LM head's blank /
        # user_footer_first signal.
        if self.core_cfg.use_chunk_classifier:
            assert self._chunk_size_repr == 0, (
                "use_chunk_classifier=True requires dynamic chunking "
                f"(chunk_size=0), got chunk_size={self._chunk_size_repr}"
            )
            self._build_chunk_classifier()
            # Aux training/eval reads self._user_footer_first_id (the BCE positive
            # label). It's normally set lazily by _ensure_inference_cache, but
            # training runs before any inference call — so prime the cache now.
            self._ensure_inference_cache()
        elif self.core_cfg.chunk_classifier_use_at_inference:
            raise ValueError("chunk_classifier_use_at_inference=True requires use_chunk_classifier=True")

        # --- Chunk-restricted audio attention (fixed chunking only) ---
        if self.core_cfg.restrict_audio_to_own_chunk:
            assert self._chunk_size_repr > 0, (
                "restrict_audio_to_own_chunk=True requires fixed chunking "
                f"(chunk_size>0), got chunk_size={self._chunk_size_repr}"
            )
            assert not self.core_cfg.use_chunk_classifier, (
                "restrict_audio_to_own_chunk is mutually exclusive with use_chunk_classifier "
                "(the latter requires chunk_size=0)."
            )
            # Custom 4D additive masks are only honored by the eager and sdpa
            # attention backends; flash-attention-2 cannot consume them.
            attn_impl = getattr(self.llm.config, "_attn_implementation", None)
            if attn_impl not in ("eager", "sdpa"):
                logging.warning(
                    "restrict_audio_to_own_chunk needs a 4D-mask-capable attention backend; "
                    "found _attn_implementation=%r. Forcing 'sdpa'.",
                    attn_impl,
                )
                self.llm.config._attn_implementation = "sdpa"
            # Distinctive banner so it is obvious in the logs (a) that this
            # checkout's code is running and (b) that the restricted-attention
            # mask is active for this run.
            logging.info("=" * 72)
            logging.info(
                "[restricted-attention] restrict_audio_to_own_chunk=True | chunk_size=%d | "
                "attn_impl=%s -- text queries attend to all text + only their own chunk's audio; "
                "audio queries stay causal.",
                self._chunk_size_repr,
                self.llm.config._attn_implementation,
            )
            logging.info("=" * 72)

        # --- Two-stream last-layer fusion (fixed chunking only) ---
        if self.core_cfg.two_stream_last_layer:
            assert self._chunk_size_repr > 0, (
                "two_stream_last_layer=True requires fixed chunking "
                f"(chunk_size>0), got chunk_size={self._chunk_size_repr}"
            )
            assert not self.core_cfg.use_chunk_classifier, (
                "two_stream_last_layer is mutually exclusive with use_chunk_classifier "
                "(the latter requires chunk_size=0)."
            )
            assert not self.core_cfg.restrict_audio_to_own_chunk, (
                "two_stream_last_layer is mutually exclusive with restrict_audio_to_own_chunk."
            )
            # Number of top layers that process the interleaved sequence.
            _n_llm_layers = int(self.llm.config.num_hidden_layers)
            self._two_stream_num_fusion_layers = int(self.core_cfg.two_stream_num_fusion_layers)
            assert 1 <= self._two_stream_num_fusion_layers < _n_llm_layers, (
                f"two_stream_num_fusion_layers must be in [1, {_n_llm_layers - 1}] "
                f"(need >=1 text-only layer), got {self._two_stream_num_fusion_layers}"
            )
            if create_causal_mask is None:
                raise ImportError(
                    "two_stream_last_layer requires transformers.masking_utils.create_causal_mask, "
                    "which is unavailable in this transformers version."
                )
            # Inference uses the fast incremental (two-KV-cache) stepper by
            # default. Set NEMO_TWO_STREAM_RECOMPUTE=1 to fall back to the slow
            # full-recompute path (parity reference; e.g. for per-sample prompts
            # or debugging).
            self._two_stream_use_cache = os.environ.get("NEMO_TWO_STREAM_RECOMPUTE", "0") != "1"
            if self._two_stream_use_cache and DynamicCache is None:
                logging.warning(
                    "two_stream_last_layer: DynamicCache unavailable; falling back to the "
                    "slow full-recompute inference path."
                )
                self._two_stream_use_cache = False
            # We drive the Qwen layer stack manually; sdpa/eager both honor the
            # boolean/additive masks we build. flash-attention-2 does not.
            attn_impl = getattr(self.llm.config, "_attn_implementation", None)
            if attn_impl not in ("eager", "sdpa"):
                logging.warning(
                    "two_stream_last_layer needs an sdpa/eager attention backend; "
                    "found _attn_implementation=%r. Forcing 'sdpa'.",
                    attn_impl,
                )
                self.llm.config._attn_implementation = "sdpa"
            logging.info("=" * 72)
            logging.info(
                "[two-stream] two_stream_last_layer=True | num_fusion_layers=%d/%d | chunk_size=%d "
                "| attn_impl=%s -- text stream runs through the lower layers alone; audio is "
                "interleaved into the top %d layer(s) for cross-modal fusion.",
                self._two_stream_num_fusion_layers,
                int(self.llm.config.num_hidden_layers),
                self._chunk_size_repr,
                self.llm.config._attn_implementation,
                self._two_stream_num_fusion_layers,
            )
            logging.info("=" * 72)

        self._apply_freeze_config()

        # --- LoRA ---
        if "lora" in self.cfg:
            # Install LoRA after freezing the LLM body to avoid freezing the LoRA weights
            maybe_install_lora(self)
            # huggingface PEFT library freezes the whole LLM, so we need to unfreeze the lm_head if needed
            if self.core_cfg.freeze_llm_head:
                freeze_module(self.llm.lm_head)
            else:
                unfreeze_module(self.llm.lm_head)

        self.val_system_prompt = (
            data_cfg.get("system_prompt", "Transcribe the audio into text.")
            if data_cfg is not None
            else "Transcribe the audio into text."
        )
        self.val_prompt_field = data_cfg.get("prompt_field", "system_prompt") if data_cfg is not None else "system_prompt"

        if forced_aligner is not None:
            assert data_cfg is not None, "Dataset config is required for online forced alignment"
            assert dataset_cls is not None, "Dataset class is required for online forced alignment"
            self.forced_aligner = forced_aligner
            self.dataset = dataset_cls(cfg=data_cfg, tokenizer=self.tokenizer)
        else:
            self.forced_aligner = None
            self.dataset = None

        # Standalone dataset config for the rank-zero training preview. In
        # precomputed mode there is no self.dataset, so keep a resolved copy of
        # the turn-construction settings to rebuild the reference transcript.
        self._preview_data_cfg = to_dataclass(StreamingSTTDataConfig, data_cfg) if data_cfg is not None else None
        if self._preview_data_cfg is not None:
            self._preview_data_cfg.blank_token = self.blank_token

        logging.info("\n" + str(ModelSummary(self, max_depth=2)))

    def _build_chunk_classifier(self) -> None:
        """Construct the aux backbone + linear head.

        The backbone reuses the LLM's architecture via ``AutoModel.from_config``
        (works for any modern HF decoder-only LLM — Llama/Qwen/Mistral/Phi/Gemma).
        ``embed_tokens`` is dropped since we always feed ``inputs_embeds``;
        same pattern as the main LLM at __init__.
        """
        K = max(int(self.core_cfg.chunk_classifier_num_layers), 1)
        aux_cfg = deepcopy(self.llm.config)
        aux_cfg.num_hidden_layers = K
        # Aux backbone is run as a full-sequence forward at both train and
        # inference time — no KV cache needed.
        aux_cfg.use_cache = False

        self.chunk_classifier_backbone = AutoModel.from_config(aux_cfg)
        # Drop V×H embedding table: we always feed inputs_embeds.
        # (Same trick as line 228-229 for the main LLM.)
        if hasattr(self.chunk_classifier_backbone, "embed_tokens"):
            del self.chunk_classifier_backbone.embed_tokens

        self.chunk_classifier_head = nn.Linear(aux_cfg.hidden_size, 1)
        nn.init.zeros_(self.chunk_classifier_head.bias)

        # Optional warm-start: copy the last K layers + final norm from the
        # main LLM. The aux backbone consumes the LLM's last hidden state, so
        # these layers operate on the right input distribution and converge
        # much faster than random init.
        if self.core_cfg.chunk_classifier_init_from_llm:
            try:
                src_layers = self.llm.model.layers[-K:]
                for i in range(K):
                    self.chunk_classifier_backbone.layers[i].load_state_dict(src_layers[i].state_dict())
                self.chunk_classifier_backbone.norm.load_state_dict(self.llm.model.norm.state_dict())
                logging.info(f"chunk_classifier: warm-started from last {K} LLM layers")
            except (AttributeError, KeyError) as e:
                logging.warning(
                    f"chunk_classifier_init_from_llm: warm-start failed ({e}); " "falling back to random init"
                )

    def _apply_freeze_config(self) -> None:
        if self.core_cfg.freeze_speech_encoder:
            freeze_module(self.perception.encoder)
        else:
            unfreeze_module(self.perception.encoder)

        # Optionally unfreeze just the last N encoder (Conformer) layers for
        # full-weight training of the top of the encoder, even when the encoder
        # is otherwise frozen. Mirrors unfreeze_last_n_llm_layers.
        n_unfreeze_enc = int(getattr(self.core_cfg, "unfreeze_last_n_encoder_layers", 0) or 0)
        if n_unfreeze_enc > 0:
            enc_layers = getattr(self.perception.encoder, "layers", None)
            if enc_layers is None:
                logging.warning(
                    "unfreeze_last_n_encoder_layers=%d requested but could not locate "
                    "self.perception.encoder.layers; skipping.",
                    n_unfreeze_enc,
                )
            else:
                num = len(enc_layers)
                k = max(1, min(n_unfreeze_enc, num))
                for layer in enc_layers[num - k:]:
                    unfreeze_module(layer)
                logging.info(
                    "Unfroze last %d/%d encoder layer(s) for full-weight fine-tuning.", k, num
                )

        if self.core_cfg.freeze_modality_adapter:
            freeze_module(self.perception.modality_adapter)
        else:
            unfreeze_module(self.perception.modality_adapter)

        if self.core_cfg.freeze_modality_proj:
            freeze_module(self.perception.proj)
        else:
            unfreeze_module(self.perception.proj)

        # Freeze the LLM body (lm_head and embed_tokens are handled separately)
        if self.core_cfg.freeze_llm_model:
            freeze_module(self.llm.model)
        else:
            unfreeze_module(self.llm.model)

        # Optionally unfreeze just the last N decoder layers for full-weight
        # fine-tuning of the top of the stack (e.g. the two-stream fusion layer),
        # even when the LLM body is otherwise frozen. Runs before LoRA install,
        # so self.llm is the raw CausalLM here (self.llm.model.layers valid).
        n_unfreeze = int(getattr(self.core_cfg, "unfreeze_last_n_llm_layers", 0) or 0)
        if n_unfreeze > 0:
            layers = getattr(getattr(self.llm, "model", None), "layers", None)
            if layers is None:
                logging.warning(
                    "unfreeze_last_n_llm_layers=%d requested but could not locate "
                    "self.llm.model.layers; skipping.",
                    n_unfreeze,
                )
            else:
                num = len(layers)
                k = max(1, min(n_unfreeze, num))
                for layer in layers[num - k:]:
                    unfreeze_module(layer)
                logging.info(
                    "Unfroze last %d/%d LLM decoder layer(s) for full-weight fine-tuning.", k, num
                )

        # lm_head is inside self.llm, so re-apply after the LLM-wide freeze
        if self.core_cfg.freeze_llm_head:
            freeze_module(self.llm.lm_head)
        else:
            unfreeze_module(self.llm.lm_head)

        # embed_tokens is a separate top-level module (moved out of llm)
        if self.core_cfg.freeze_embed_tokens:
            freeze_module(self.embed_tokens)
        else:
            unfreeze_module(self.embed_tokens)

        # Aux chunk-boundary classifier (backbone + linear head). Only present
        # when use_chunk_classifier is True.
        if self.core_cfg.use_chunk_classifier:
            if self.core_cfg.freeze_chunk_classifier:
                freeze_module(self.chunk_classifier_backbone)
                freeze_module(self.chunk_classifier_head)
            else:
                unfreeze_module(self.chunk_classifier_backbone)
                unfreeze_module(self.chunk_classifier_head)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return int(self.embed_tokens.num_embeddings)

    @property
    def text_pad_id(self) -> int:
        pad_id = self.tokenizer.pad_id
        if pad_id is None:
            pad_id = self.tokenizer.unk_id
        if pad_id is None:
            warnings.warn(
                "The text tokenizer has no <pad> or <unk> token; using id 0 for "
                "padding (this may lead to silent bugs)."
            )
            pad_id = 0
        return pad_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def sampling_rate(self) -> int:
        return self.perception.preprocessor.featurizer.sample_rate

    @property
    def sample_rate(self) -> int:
        return self.perception.preprocessor.featurizer.sample_rate

    @property
    def frame_duration(self) -> float:
        """Duration (in seconds) of one audio frame at the perception output."""
        return self.perception.token_equivalent_duration

    @property
    def blank_token_id(self) -> int:
        # Sentinel -1 when blank is disabled — guarantees `token == blank_token_id`
        # never matches a real vocab id, so stop-on-blank checks naturally no-op.
        if self.blank_token == "":
            return -1
        return self.tokenizer.text_to_ids(self.blank_token)[0]

    @property
    def has_blank(self) -> bool:
        return self.blank_token != ""

    # ------------------------------------------------------------------
    # Core: efficient audio-text embedding interleaving
    # ------------------------------------------------------------------

    def _build_input_embeds(
        self,
        input_tokens: Tensor,
        audios: Tensor,
        audio_lens: Tensor,
    ) -> dict[str, Tensor]:
        """
        Encode audio, embed text tokens, then interleave them.

        This is the high-level entry point used by ``training_step`` and
        ``_eval_step``.  The pure-tensor interleaving logic lives in
        :func:`interleave_embeddings` so it can be tested without a model.

        Args:
            input_tokens: (B, L) token IDs with ``AUDIO_TOKEN_IDX`` at audio
                positions and ``text_pad_id`` at left-padding positions.
            audios: (B, T_samples) raw waveforms.
            audio_lens: (B,) waveform lengths in samples.
        Returns:
            dict with keys ``input_embeds`` (B, L, H), ``attention_mask`` (B, L).
        """
        audio_mask = input_tokens == AUDIO_TOKEN_IDX  # (B, L)

        # --- text embeddings ---
        # Zero-out audio positions so embed_tokens gets valid indices.
        text_tokens = input_tokens.where(~audio_mask, torch.zeros_like(input_tokens))
        text_embeds = self.embed_tokens(text_tokens)  # (B, L, H)

        # --- audio embeddings ---
        audio_embs, _audio_emb_lens = self.perception(
            input_signal=audios,
            input_signal_length=audio_lens,
        )  # audio_embs: (B, T_enc, H)

        # --- interleave & build attention mask ---
        return interleave_embeddings(
            input_tokens=input_tokens,
            audio_mask=audio_mask,
            text_embeds=text_embeds,
            audio_embs=audio_embs,
            pad_id=self.text_pad_id,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_embeds: Tensor,
        attention_mask: Tensor | None = None,
        cache=None,
        output_hidden_states: bool = False,
    ) -> dict[str, Tensor]:
        """
        Forward pass:  embeddings → LLM → logits.

        When ``output_hidden_states=True`` the dict also contains
        ``hidden_states`` (B, L, H) — the LLM's last-layer hidden state, used
        as input to the aux chunk-boundary classifier.
        """
        out = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=cache is not None,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        ans = {"logits": out["logits"]}  # (B, L, V)
        if output_hidden_states:
            ans["hidden_states"] = out["hidden_states"][-1]  # (B, L, H)
        if cache is not None:
            ans["cache"] = out["past_key_values"]
        return ans

    def _two_stream_forward(
        self,
        input_embeds: Tensor,
        audio_mask: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Full-sequence two-stream forward (see :func:`two_stream_llm_forward`).

        ``self.llm.model`` / ``self.llm.lm_head`` resolve through the PEFT wrapper
        when LoRA is installed, so the injected adapters stay active.
        """
        layers, norm, rotary_emb, lm_head = self._resolve_llm_core()
        return two_stream_llm_forward(
            layers=layers,
            norm=norm,
            rotary_emb=rotary_emb,
            lm_head=lm_head,
            inputs_embeds=input_embeds,
            audio_mask=audio_mask,
            valid_mask=valid_mask,
            num_fusion_layers=getattr(self, "_two_stream_num_fusion_layers", 1),
            compute_logits=True,
        )

    def _resolve_llm_core(self):
        """Return ``(layers, norm, rotary_emb, lm_head)`` for the LLM.

        Unwraps the PEFT/LoRA wrapper (``get_base_model()`` returns the underlying
        ``*ForCausalLM`` with adapters still injected) and then walks down to the
        decoder body that owns ``layers`` / ``norm`` / ``rotary_emb``. With LoRA
        installed, ``self.llm.model`` resolves to the ``*ForCausalLM`` (not the
        decoder), so a naive ``self.llm.model.layers`` fails.
        """
        llm = self.llm
        base = llm.get_base_model() if hasattr(llm, "get_base_model") else llm
        lm_head = getattr(base, "lm_head", None)
        if lm_head is None:
            lm_head = self.llm.lm_head
        # Descend until we find the module holding the decoder layers.
        core = base
        while not hasattr(core, "layers") and hasattr(core, "model"):
            core = core.model
        if not hasattr(core, "layers"):
            raise AttributeError(
                f"Could not locate decoder layers on LLM of type {type(base).__name__}"
            )
        return core.layers, core.norm, core.rotary_emb, lm_head

    def _resolve_inference_chunk_size(self, chunk_size_override: Optional[int] = None) -> int:
        """Resolve the chunk size (frames) for a single inference call.

        Precedence: explicit ``chunk_size_override`` > longest size when the
        config ``chunk_size`` is a list > the scalar config value.
        """
        if chunk_size_override is not None:
            return int(chunk_size_override)
        return self._chunk_size_repr

    def _set_encoder_att_context(self, chunk_size: Optional[int], recompute_streaming: bool = False) -> None:
        """Set the Conformer encoder's attention look-ahead.

        Two modes, selected by ``couple_encoder_lookahead_to_chunk``:
          * coupled (default): match the look-ahead to the per-batch LLM chunk
            size as ``[att_context_size[0], chunk_size - 1]`` (multi chunk-size
            "contexts" training). No-op for dynamic (0) / offline (<0) chunking.
          * fixed: pin the encoder to the configured ``att_context_size`` (e.g.
            ``[70, 0]`` for a strictly-causal encoder) regardless of the chunk
            size the dataset drew.
        No-op when ``att_context_size`` is unset.

        Args:
            chunk_size: fixed-chunk size in encoder frames (per-batch for
                training; resolved single size for inference). Only used in the
                coupled mode.
            recompute_streaming: also recompute the cache-aware streaming config
                (needed for streaming inference buffer/cache sizing). Leave False
                during training (the non-cached forward only reads att_context_size).
        """
        if self.core_cfg.att_context_size is None:
            return
        if self.core_cfg.couple_encoder_lookahead_to_chunk:
            # Only couple look-ahead to chunk size in multi chunk-size mode; scalar
            # configs keep their configured att_context_size untouched.
            if not getattr(self, "_chunk_size_candidates", None):
                return
            if chunk_size is None or int(chunk_size) <= 0:
                return
            left = int(self.core_cfg.att_context_size[0])
            new_ctx = [left, int(chunk_size) - 1]
        else:
            # Fixed encoder look-ahead, independent of the LLM chunk size.
            new_ctx = [int(x) for x in self.core_cfg.att_context_size]
        encoder = self.perception.encoder
        # Set att_context_size directly (not set_default_att_context_size) so the
        # encoder's training forward does not randomly pick a look-ahead and
        # override this value.
        encoder.att_context_size = new_ctx
        if recompute_streaming:
            encoder.setup_streaming_params()

    def _two_stream_infer_step(
        self,
        input_embeds: Tensor,
        input_is_audio: Tensor,
        state: "StreamingState",
    ) -> SimpleNamespace:
        """Dispatch a two-stream streaming forward: fast KV-cached path by
        default, slow full-recompute path when ``_two_stream_use_cache`` is off."""
        if getattr(self, "_two_stream_use_cache", True):
            return self._two_stream_infer_step_cached(input_embeds, input_is_audio, state)
        return self._two_stream_infer_step_recompute(input_embeds, input_is_audio, state)

    def _two_stream_infer_step_cached(
        self,
        input_embeds: Tensor,
        input_is_audio: Tensor,
        state: "StreamingState",
    ) -> SimpleNamespace:
        """Fast incremental two-stream step using the two KV caches on ``state``."""
        layers, norm, rotary_emb, lm_head = self._resolve_llm_core()
        logits, state.ts_text_len, state.ts_full_len = two_stream_cached_step(
            layers=layers,
            norm=norm,
            rotary_emb=rotary_emb,
            lm_head=lm_head,
            input_embeds=input_embeds,
            input_is_audio=input_is_audio,
            text_cache=state.ts_text_cache,
            last_cache=state.ts_last_cache,
            text_len=state.ts_text_len,
            full_len=state.ts_full_len,
            num_fusion_layers=getattr(self, "_two_stream_num_fusion_layers", 1),
            compute_logits=True,
        )
        return SimpleNamespace(logits=logits, past_key_values=None, hidden_states=None)

    def _two_stream_infer_step_recompute(
        self,
        input_embeds: Tensor,
        input_is_audio: Tensor,
        state: "StreamingState",
    ) -> SimpleNamespace:
        """One streaming forward for two-stream inference (no KV cache).

        Appends ``input_embeds`` / ``input_is_audio`` to the running buffers on
        ``state`` and recomputes the two-stream forward over the FULL accumulated
        sequence. This is O(L) layers × O(L) length per step, but guarantees the
        inference math is byte-for-byte the same as the (tested) training forward
        — the whole point of the two-stream experiment is a clean train/infer
        comparison. ``state.attention_mask`` (the running validity mask) must
        already be extended to cover ``input_embeds`` before this call.

        Returns a ``SimpleNamespace`` mirroring the HF output (``logits`` for the
        newly appended positions, ``past_key_values=None``).
        """
        if state.ts_embeds is None:
            state.ts_embeds = input_embeds
            state.ts_is_audio = input_is_audio
        else:
            state.ts_embeds = torch.cat([state.ts_embeds, input_embeds], dim=1)
            state.ts_is_audio = torch.cat([state.ts_is_audio, input_is_audio], dim=1)

        assert state.ts_embeds.shape[1] == state.attention_mask.shape[1], (
            f"two-stream buffer length {state.ts_embeds.shape[1]} != attention_mask length "
            f"{state.attention_mask.shape[1]}"
        )
        out = self._two_stream_forward(
            state.ts_embeds,
            state.ts_is_audio,
            state.attention_mask.bool(),
        )
        q = input_embeds.shape[1]
        return SimpleNamespace(logits=out["logits"][:, -q:, :], past_key_values=None, hidden_states=None)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _validate_token_ranges(self, batch: StreamingSTTBatch) -> None:
        """Raise a clear error if any input/target token id is out of range.

        This is the CPU-side equivalent of the device-side asserts that
        ``embed_tokens`` (index >= num_embeddings) and ``cross_entropy`` (target
        outside ``[0, n_classes)``) raise asynchronously. Reports the offending
        ids and the embed vs. LM-head vocab sizes (a mismatch there is a common
        cause after adding special tokens like ``<blank>``).
        """
        embed_vocab = int(self.embed_tokens.num_embeddings)
        head_vocab = int(getattr(self.llm.lm_head, "out_features", embed_vocab))
        if embed_vocab != head_vocab:
            logging.warning(
                "[debug_validate_tokens] embed vocab (%d) != lm_head vocab (%d); "
                "targets in [%d,%d) will assert in cross_entropy.",
                embed_vocab,
                head_vocab,
                min(embed_vocab, head_vocab),
                max(embed_vocab, head_vocab),
            )

        input_tokens = batch.input_tokens
        non_audio = input_tokens != AUDIO_TOKEN_IDX
        bad_input = non_audio & ((input_tokens < 0) | (input_tokens >= embed_vocab))
        if bool(bad_input.any()):
            offending = input_tokens[bad_input].unique().tolist()[:10]
            raise ValueError(
                f"[debug_validate_tokens] input token id(s) out of embedding range "
                f"[0,{embed_vocab}): {offending}"
            )

        target_tokens = batch.target_tokens
        valid_target = target_tokens != IGNORE_INDEX
        bad_target = valid_target & ((target_tokens < 0) | (target_tokens >= head_vocab))
        if bool(bad_target.any()):
            offending = target_tokens[bad_target].unique().tolist()[:10]
            raise ValueError(
                f"[debug_validate_tokens] target token id(s) out of LM-head range "
                f"[0,{head_vocab}) (ignore_index={IGNORE_INDEX}): {offending}"
            )

    def training_step(self, batch: StreamingSTTBatch, batch_idx: int):
        # Keep frozen modules in eval mode (disables dropout / batch-norm updates).
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        if self.forced_aligner is not None:
            alignments = self.forced_aligner.align(batch.audios, batch.audio_lens, batch.text)
            self._latest_train_alignments = alignments
            batch = self.dataset.get_batch_data(
                cuts=batch.cuts,
                audios=batch.audios,
                audio_lens=batch.audio_lens,
                alignments=alignments,
                text=batch.text,
            )
            batch = move_data_to_device(batch, self.device)

        if self.core_cfg.debug_validate_tokens:
            self._validate_token_ranges(batch)

        # Match the encoder's attention look-ahead to the per-batch chunk size the
        # dataset drew (no-op unless chunk_size is a list and att_context_size is
        # set). The non-cached training forward only reads att_context_size.
        self._set_encoder_att_context(getattr(batch, "chunk_size", None))

        inputs = self._build_input_embeds(batch.input_tokens, batch.audios, batch.audio_lens)
        use_aux = self.core_cfg.use_chunk_classifier
        attention_mask = inputs["attention_mask"]
        if self.core_cfg.restrict_audio_to_own_chunk:
            attention_mask = build_training_chunk_restricted_mask(
                batch.input_tokens, self.text_pad_id, inputs["input_embeds"].dtype
            )
            if not getattr(self, "_restricted_mask_logged", False):
                self._restricted_mask_logged = True
                logging.info(
                    "[restricted-attention] applied 4D chunk-restricted mask in training_step "
                    "(shape=%s) -- this run uses restricted audio attention.",
                    tuple(attention_mask.shape),
                )
        if self.core_cfg.two_stream_last_layer:
            two_stream_audio_mask = batch.input_tokens == AUDIO_TOKEN_IDX  # (B, L)
            two_stream_valid = batch.input_tokens != self.text_pad_id  # (B, L)
            outputs = self._two_stream_forward(
                inputs["input_embeds"], two_stream_audio_mask, two_stream_valid
            )
            if not getattr(self, "_two_stream_logged", False):
                self._two_stream_logged = True
                logging.info(
                    "[two-stream] applied two-stream last-layer forward in training_step "
                    "(text stream len<=%d, full len=%d) -- this run fuses audio only in the "
                    "final layer.",
                    int(two_stream_valid.sum(dim=1).max().item()),
                    int(batch.input_tokens.shape[1]),
                )
        else:
            outputs = self.forward(
                inputs["input_embeds"],
                attention_mask=attention_mask,
                output_hidden_states=use_aux,
            )

        target_ids = batch.target_tokens

        # When the aux chunk classifier is active, strip audio-frame positions
        # from the LM CE so the LM head is only supervised on text. The aux
        # head (below) handles the boundary decision via BCE. Use the input-axis
        # audio mask — NOT a target-value mask — so end-of-chunk blanks at
        # text positions (line 1696/1701 in inference) remain supervised.
        if use_aux:
            audio_mask = batch.input_tokens == AUDIO_TOKEN_IDX  # (B, L)
            if not self.core_cfg.chunk_classifier_keep_lm_supervision_at_audio:
                target_ids = torch.where(audio_mask, torch.full_like(target_ids, IGNORE_INDEX), target_ids)
            # else: keep blank / user_footer_first targets at audio positions —
            # LM head is co-supervised alongside the aux BCE.
        else:
            audio_mask = None

        num_targets = (target_ids != IGNORE_INDEX).long().sum()

        if num_targets == 0:
            logging.warning("Batch %d: num_targets is 0 — skipping (returning zero loss).", batch_idx)
            return {"loss": torch.tensor(0.0, device=target_ids.device, requires_grad=True)}

        logits = outputs["logits"]

        # # Diagnose NaN sources (remove once stable).
        # if torch.isnan(inputs["input_embeds"]).any():
        #     logging.warning("Batch %d: NaN in input_embeds", batch_idx)
        # if torch.isnan(logits).any():
        #     logging.warning("Batch %d: NaN in logits", batch_idx)

        flat_logits = logits.flatten(0, 1)
        flat_targets = target_ids.flatten(0, 1)

        with loss_parallel():
            per_token_loss = F.cross_entropy(
                flat_logits,
                flat_targets,
                reduction="none",
                ignore_index=IGNORE_INDEX,
            )

        # --- Blank vs non-blank loss breakdown ---
        blank_id = self.blank_token_id
        valid_mask = flat_targets != IGNORE_INDEX
        # When blank is disabled (blank_id=-1), is_blank is always False →
        # everything counts as non-blank, and blank_weight has no effect.
        is_blank = valid_mask & (flat_targets == blank_id)
        is_nonblank = valid_mask & (flat_targets != blank_id)
        num_blank = is_blank.sum()
        num_nonblank = is_nonblank.sum()

        # Apply blank loss weight (< 1.0 to down-weight easy blank predictions)
        blank_weight = self.core_cfg.blank_loss_weight
        if num_blank > 0 and blank_weight != 1.0 and self.has_blank:
            effective_num_targets = num_blank * blank_weight + num_nonblank
            loss = (
                per_token_loss[is_nonblank].sum() + per_token_loss[is_blank].sum() * blank_weight
            ) / effective_num_targets
        else:
            loss = per_token_loss.sum() / num_targets

        with torch.no_grad():
            loss_blank = per_token_loss[is_blank].sum() / num_blank.clamp(min=1)
            loss_nonblank = per_token_loss[is_nonblank].sum() / num_nonblank.clamp(min=1)

        # --- Aux chunk-boundary classifier loss ---
        # BCE on the aux head's binary "ready to emit" prediction at audio frames.
        # Supervised positions: input is an audio frame AND original target was a
        # decision token (blank=keep listening, user_footer_first=emit). Using the
        # input-axis audio mask here exactly mirrors the supervision the LM head
        # used to provide at audio positions before §4(a) masked them out.
        cls_loss_log = torch.zeros((), device=loss.device)
        if use_aux and self.has_blank and audio_mask is not None and self._user_footer_first_id is not None:
            audio_mask_flat = audio_mask.flatten(0, 1)  # (B*L,)
            orig_targets_flat = batch.target_tokens.flatten(0, 1)  # pre-LM-CE-masking
            decision_mask = audio_mask_flat & (orig_targets_flat != IGNORE_INDEX)
            num_decisions = decision_mask.sum()
            if num_decisions > 0:
                aux_input = outputs["hidden_states"]  # (B, L, H)
                if self.core_cfg.chunk_classifier_stop_grad_to_llm:
                    aux_input = aux_input.detach()
                aux_out = self.chunk_classifier_backbone(
                    inputs_embeds=aux_input,
                    attention_mask=inputs["attention_mask"],
                    return_dict=True,
                )
                flat_aux = aux_out.last_hidden_state.flatten(0, 1)  # (B*L, H)
                cls_logits = self.chunk_classifier_head(flat_aux[decision_mask]).squeeze(-1)
                cls_targets = (orig_targets_flat[decision_mask] == self._user_footer_first_id).to(cls_logits.dtype)
                # Auto-balance: pos_weight = N_neg/N_pos. Skip when either class
                # is empty in this batch (pos_weight would zero out one side).
                num_pos = cls_targets.sum()
                num_neg = cls_targets.numel() - num_pos
                if self.core_cfg.chunk_classifier_auto_balance and num_pos > 0 and num_neg > 0:
                    pos_weight = (num_neg.float() / num_pos.float()).detach()
                else:
                    pos_weight = None
                cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_targets, pos_weight=pos_weight)
                cls_w = self.core_cfg.chunk_classifier_loss_weight
                loss = loss + cls_w * cls_loss
                cls_loss_log = cls_loss.detach()
                cls_pos_ratio = num_pos.float() / num_decisions.clamp(min=1).float()
                if self.core_cfg.log_detailed_train_metrics:
                    self.log_dict(
                        {"loss_chunk_cls_pos_ratio": cls_pos_ratio},
                        on_step=True,
                    )

        B, L = inputs["input_embeds"].shape[:2]
        train_metrics = {
            "loss": loss,
            "learning_rate": torch.as_tensor(
                self.trainer.optimizers[0].param_groups[0]["lr"] if self._trainer is not None else 0
            ),
        }
        if self.core_cfg.log_detailed_train_metrics:
            train_metrics.update(
                {
                    "loss_blank": loss_blank,
                    "loss_nonblank": loss_nonblank,
                    "loss_chunk_cls": cls_loss_log,
                    "blank_ratio": num_blank.float() / num_targets,
                    "sequence_length": float(L),
                    "num_targets": num_targets.float(),
                    "target_to_input_ratio": num_targets / (B * L),
                }
            )
        self.log_dict(train_metrics, on_step=True)
        return {"loss": loss}

    @property
    def train_decode_max_new_tokens_per_chunk(self) -> int:
        configured = self.core_cfg.train_decode_max_new_tokens_per_chunk
        if configured is not None:
            if configured <= 0:
                raise ValueError("train_decode_max_new_tokens_per_chunk must be positive")
            return configured
        if self._chunk_size_repr > 0:
            return self._chunk_size_repr
        return 64

    def _training_reference_with_chunk_boundaries(self, batch: StreamingSTTBatch) -> str:
        """Render the first transcript using the same aligned turns as training.

        Falls back to the plain reference (no chunk separators) whenever
        alignments are unavailable — e.g. precomputed mode without cuts — so the
        preview never fails just because it cannot draw boundaries.
        """
        alignments = getattr(self, "_latest_train_alignments", None)
        if alignments is None:
            if getattr(batch, "cuts", None) is None:
                return batch.text[0]
            try:
                alignments = get_word_alignments_for_batch(cuts=batch.cuts)
            except Exception:
                return batch.text[0]
        if not alignments or alignments[0] is None:
            return batch.text[0]

        cfg = self.dataset.cfg if self.dataset is not None else self._preview_data_cfg
        if cfg is None:
            return batch.text[0]
        prompts = self._validation_system_prompts(batch)
        prompt = prompts[0] if isinstance(prompts, list) else prompts
        messages = get_llm_messages_for_sample(
            system_role=cfg.system_role,
            system_prompt=prompt,
            audio_tag=cfg.audio_tag,
            blank_token=cfg.blank_token,
            chunk_size=cfg.chunk_size,
            num_delay_frames=cfg.num_delay_frames,
            audio_duration_secs=float(batch.audio_lens[0].item()) / cfg.sample_rate,
            frame_length_in_secs=cfg.frame_length_in_secs,
            alignments=alignments[0],
            transcript=batch.text[0],
            words_per_group=cfg.words_per_group,
            chunk_step=cfg.chunk_step,
        )
        chunk_text = [
            "" if message["content"] == cfg.blank_token else message["content"].strip()
            for message in messages
            if message["role"] == "assistant"
        ]
        return " | ".join(chunk_text)

    def on_train_batch_end(self, outputs, batch: StreamingSTTBatch, batch_idx: int) -> None:
        """Periodically print a real autoregressive hypothesis for one training utterance."""
        interval = self.core_cfg.train_decode_every_n_steps
        if interval <= 0 or self.trainer is None or not self.trainer.is_global_zero:
            return

        step = int(self.trainer.global_step)
        if step <= 0 or step % interval != 0 or getattr(self, "_last_train_decode_step", None) == step:
            return
        self._last_train_decode_step = step

        if not isinstance(batch, StreamingSTTBatch) or not batch.text:
            logging.warning("Skipping train autoregressive preview at step %d: unsupported/empty batch", step)
            return

        prompts = self._validation_system_prompts(batch)
        prompt = prompts[0] if isinstance(prompts, list) else prompts
        module_training_states = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            hyps = self.generate(
                audios=batch.audios[:1],
                audio_lens=batch.audio_lens[:1],
                system_prompt=prompt,
                max_new_tokens=self.train_decode_max_new_tokens_per_chunk,
                generation_config=GenerationConfig(do_sample=False),
                chunk_separator="|",
            )
            plain_hyp = " ".join(hyps[0].replace("|", " ").split())
            preview_wer = WER(normalize=True, verbose=False)
            preview_wer.update("preview", refs=[batch.text[0]], hyps=[plain_hyp])
            training_wer = preview_wer.compute()["wer"].to(self.device)
            self.log(
                "training_wer",
                training_wer,
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=False,
                rank_zero_only=True,
            )
            logging.info(
                "[train-ar] step %d (WER %.4f, max %d tokens/chunk)\n  ref: `%s`\n  hyp: `%s`",
                step,
                training_wer.item(),
                self.train_decode_max_new_tokens_per_chunk,
                self._training_reference_with_chunk_boundaries(batch),
                hyps[0],
            )
        except Exception as error:
            logging.warning("Train autoregressive preview failed at step %d: %s", step, error)
        finally:
            self._latest_train_alignments = None
            for module, was_training in module_training_states:
                module.training = was_training

    def configure_optimizers(self):
        return configure_optimizers(self)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @property
    def val_max_new_tokens_per_chunk(self) -> int:
        """Resolve the autoregressive validation cap."""
        configured = self.core_cfg.val_max_new_tokens_per_chunk
        if configured is not None:
            if configured <= 0:
                raise ValueError("val_max_new_tokens_per_chunk must be positive")
            return configured
        if self._chunk_size_repr > 0:
            return self._chunk_size_repr
        return 64

    def _validation_system_prompts(self, batch: StreamingSTTBatch) -> Union[str, List[str]]:
        if batch.cuts is None:
            return self.val_system_prompt
        return [
            (cut.custom or {}).get(self.val_prompt_field, self.val_system_prompt)
            for cut in batch.cuts
        ]

    def on_validation_epoch_start(self) -> None:
        self._partial_wer_refs: dict[str, list[str]] = defaultdict(list)
        self._partial_wer_hyps: dict[str, list[str]] = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        # Gather decoded strings, then compute true corpus WER on every rank.
        # Averaging rank-local WERs would be incorrect when ranks see different
        # numbers of reference words.
        local_wer_data = {
            name: {"refs": self._partial_wer_refs[name], "hyps": self._partial_wer_hyps[name]}
            for name in self._partial_wer_refs
        }
        if torch.distributed.is_initialized():
            gathered_wer_data = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered_wer_data, local_wer_data)
        else:
            gathered_wer_data = [local_wer_data]

        wer = WER(normalize=True, verbose=False)
        has_wer_data = False
        for rank_data in gathered_wer_data:
            for name, values in rank_data.items():
                has_wer_data = has_wer_data or bool(values["refs"])
                wer.update(name, refs=values["refs"], hyps=values["hyps"])
        if has_wer_data:
            for metric_name, metric_value in wer.compute().items():
                log_name = "val_wer" if metric_name == "wer" else f"val_{metric_name}"
                self.log(log_name, metric_value.to(self.device), on_epoch=True, sync_dist=False)

        self._partial_wer_refs.clear()
        self._partial_wer_hyps.clear()

    def validation_step(self, batch, batch_idx: int):
        # Support multiple validation dataloaders ({name: batch} dict).
        if isinstance(batch, dict):
            for name, dataset_batch in batch.items():
                if dataset_batch is not None:
                    self._eval_step(dataset_batch, name, batch_idx)
        else:
            self._eval_step(batch, "val", batch_idx)

    def _eval_step(self, batch: StreamingSTTBatch, name: str, batch_idx: int = 0) -> None:
        # Validation is decode-only: unlike teacher-forced loss, autoregressive
        # WER does not require word alignments or constructed target turns.
        refs = list(batch.text)
        # Multi chunk-size: decode validation at a fixed, well-supported chunk
        # (default 14) instead of the longest candidate, whose look-ahead
        # (chunk-1) may be an unsupported/slow streaming config that stalls the
        # decode. generate(chunk_size_override=...) pins the size and rebuilds the
        # turn template for it (then restores).
        val_chunk = None
        if getattr(self, "_chunk_size_candidates", None):
            val_chunk = self.core_cfg.val_chunk_size
            if val_chunk is None:
                val_chunk = 14 if 14 in self._chunk_size_candidates else max(self._chunk_size_candidates)
            val_chunk = int(val_chunk)
        hyps = self.generate(
            audios=batch.audios,
            audio_lens=batch.audio_lens,
            system_prompt=self._validation_system_prompts(batch),
            max_new_tokens=self.val_max_new_tokens_per_chunk,
            generation_config=GenerationConfig(do_sample=False),
            chunk_size_override=val_chunk,
        )
        self._partial_wer_refs[name].extend(refs)
        self._partial_wer_hyps[name].extend(hyps)

        if batch_idx % self.core_cfg.log_every_n_steps == 0 and refs and hyps:
            logging.info(
                "[%s] autoregressive batch %d (max %d tokens/chunk)\n  ref: `%s`\n  hyp: `%s`",
                name,
                batch_idx,
                self.val_max_new_tokens_per_chunk,
                refs[0],
                hyps[0],
            )

    # ------------------------------------------------------------------
    # Test (delegates to validation logic)
    # ------------------------------------------------------------------

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    # ------------------------------------------------------------------
    # Backward + OOMptimizer
    # ------------------------------------------------------------------

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    @property
    def oomptimizer_schema(self) -> dict:
        from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType

        return {
            "cls": StreamingSTTBatch,
            "inputs": [
                {
                    "name": "input_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": int(self.text_vocab_size),
                },
                {"name": "input_token_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "output"},
                {
                    "name": "target_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": int(self.text_vocab_size),
                },
                {"name": "target_token_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "output"},
                {"name": "audios", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
            ],
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _ensure_inference_cache(self) -> None:
        """Lazily cache token templates and IDs needed for inference.

        Uses ``apply_chat_template(tokenize=False)`` on a 4-message dummy
        conversation and splits the text around a sentinel to isolate
        user-header, user-footer + assistant-header, and assistant-footer tokens.

        The 4-message pattern (two user+assistant pairs) ensures the *first*
        assistant turn is not the last — this prevents Qwen3-style chat
        templates from injecting ``<think>``/``</think>`` tags, which only
        appear on the final assistant turn.
        """
        if hasattr(self, '_inference_cache_ready'):
            return

        hf_tok = self.tokenizer.tokenizer
        chunk_size = self._chunk_size_repr

        # --- Build turn template ---
        if self.core_cfg.compact_template:
            user_header_ids, user_footer_and_asst_header_ids, asst_footer_ids = build_compact_turn_markers(
                hf_tok, self.core_cfg.write_token
            )
            logging.info(
                f"compact_template: user_header={user_header_ids}, "
                f"write+mid={user_footer_and_asst_header_ids}, footer={asst_footer_ids}"
            )
        else:
            user_header_ids, user_footer_and_asst_header_ids, asst_footer_ids = parse_chat_template_ids(
                hf_tok, last_turn=(chunk_size < 0)
            )
        self._user_header_ids = user_header_ids
        self._user_footer_and_asst_header_ids = user_footer_and_asst_header_ids
        self._asst_footer_ids = asst_footer_ids

        # Always cache user_footer_first_id — needed by state machine inference
        # for both dynamic (chunk_size=0) and fixed chunking (use_state_machine_inference).
        self._user_footer_first_id = user_footer_and_asst_header_ids[0] if user_footer_and_asst_header_ids else None

        if chunk_size > 0:
            turn_ids = user_header_ids + [AUDIO_TOKEN_IDX] * chunk_size + user_footer_and_asst_header_ids
            self._turn_template_ids = turn_ids
            n_audio = turn_ids.count(AUDIO_TOKEN_IDX)
            logging.info(
                f"Streaming turn template ({len(turn_ids)} tokens, "
                f"{n_audio} audio slots, chunk_size={chunk_size}): {turn_ids}"
            )
        elif chunk_size == 0:
            # Dynamic chunking: no fixed turn template. Audio frames are fed
            # incrementally; the user header/footer are appended on demand.
            self._turn_template_ids = None
            logging.info(
                f"Dynamic chunking mode: user_footer_first_id={self._user_footer_first_id}, "
                f"user_header_ids={user_header_ids}"
            )
        else:
            self._turn_template_ids = None
            logging.info(f"Offline mode (chunk_size={chunk_size}): no fixed turn template")

        self._eos_id = getattr(hf_tok, 'eos_token_id', None)

        # When eos_token_id coincides with a token in the footer (e.g. Qwen3
        # where eos = <|im_end|> = footer[0]), detecting EOS acts as an
        # early-stop shortcut that avoids generating the remaining footer
        # tokens.  When eos_token_id is NOT in the footer it serves as a
        # safety-net stop only.
        self._eos_in_footer = self._eos_id is not None and self._eos_id in self._asst_footer_ids
        logging.info(
            f"Assistant footer IDs: {self._asst_footer_ids}, "
            f"blank ID: {self.blank_token_id}, EOS ID: {self._eos_id}, "
            f"EOS in footer: {self._eos_in_footer}"
        )
        self._inference_cache_ready = True

    def _sample_token(
        self,
        logits: Tensor,
        generated_ids: list[list[int]] | list[int] | None = None,
        generation_config: Optional[GenerationConfig] = None,
        **generation_kwargs,
    ) -> Tensor:
        """Select the next token from logits.

        Applies the following transforms in order (each is skipped when the
        corresponding parameter is at its default/off value):

        1. **Suppress tokens** — force listed token IDs to ``-inf``.
        2. **No-repeat-ngram** — block n-grams that already appear in
           *generated_ids*.
        3. **Repetition penalty** — scale logits for tokens that already appear
           in *generated_ids*.
        4. **Temperature** — divide logits by temperature.
        5. **Top-k** — keep only the *k* highest-scoring tokens.
        6. **Top-p (nucleus)** — keep the smallest set of tokens whose
           cumulative probability is ≥ *top_p*.
        7. If ``do_sample`` is ``True``, sample from the filtered distribution;
           otherwise return the argmax.

        Parameters are read from *generation_kwargs* first, falling back to
        *generation_config*, then to HuggingFace defaults.

        Args:
            logits: ``(B, vocab_size)`` logits for the last position.
            generated_ids: Token IDs generated so far.  For B=1, a flat list.
                For B>1, a list of B lists (one per stream).  Used for
                repetition-aware transforms.  May be ``None`` or empty.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            generation_kwargs: Per-call overrides.

        Returns:
            ``(B,)`` tensor with the selected token IDs.
        """
        # Fast path: no config → greedy
        if generation_config is None and not generation_kwargs:
            return logits.argmax(dim=-1)

        cfg = generation_config or GenerationConfig()
        do_sample = generation_kwargs.get('do_sample', cfg.do_sample)
        temperature = generation_kwargs.get('temperature', cfg.temperature)
        top_k = generation_kwargs.get('top_k', cfg.top_k)
        top_p = generation_kwargs.get('top_p', cfg.top_p)
        repetition_penalty = generation_kwargs.get('repetition_penalty', cfg.repetition_penalty)
        no_repeat_ngram_size = generation_kwargs.get('no_repeat_ngram_size', cfg.no_repeat_ngram_size)
        suppress_tokens = generation_kwargs.get('suppress_tokens', cfg.suppress_tokens)

        # --- logit manipulation (order matters) ---

        # 1. Suppress tokens
        if suppress_tokens:
            logits[..., suppress_tokens] = float('-inf')

        # 2. No-repeat-ngram blocking
        if no_repeat_ngram_size > 0 and generated_ids and len(generated_ids) >= no_repeat_ngram_size - 1:
            ngram_prefix = generated_ids[-(no_repeat_ngram_size - 1) :]
            for i in range(len(generated_ids) - no_repeat_ngram_size + 1):
                if generated_ids[i : i + no_repeat_ngram_size - 1] == ngram_prefix:
                    # The token that followed this prefix last time is banned
                    logits[..., generated_ids[i + no_repeat_ngram_size - 1]] = float('-inf')

        # 3. Repetition penalty
        if repetition_penalty != 1.0 and generated_ids:
            prev_token_ids = torch.tensor(list(set(generated_ids)), device=logits.device)
            scores = logits[..., prev_token_ids]
            # Penalize: divide positive scores, multiply negative scores
            logits[..., prev_token_ids] = torch.where(
                scores > 0, scores / repetition_penalty, scores * repetition_penalty
            )

        # Greedy fast path (no sampling-related transforms needed)
        if not do_sample:
            return logits.argmax(dim=-1)

        # 4. Temperature scaling
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        # 5. Top-k filtering
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            kth_val = torch.topk(logits, k, dim=-1)[0][..., -1:]
            logits = logits.masked_fill(logits < kth_val, float('-inf'))

        # 6. Top-p (nucleus) filtering
        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Mark tokens whose cumulative probability (excluding themselves) >= top_p
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            indices_to_remove = sorted_mask.scatter(dim=-1, index=sorted_indices, src=sorted_mask)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # 7. Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _autoregressive_decode(
        self,
        logits: Tensor,
        cache: tuple,
        state: Optional['StreamingState'],
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        stop_on_blank: Union[bool, str] = True,
        **generation_kwargs,
    ) -> tuple[list[list[int]], tuple, list[bool], int]:
        """Autoregressive decoding (supports B streams).

        Token selection is delegated to :meth:`_sample_token`, which supports
        greedy (default), sampling (temperature / top-k / top-p), repetition
        penalty, no-repeat-ngram blocking, and token suppression.

        Generation stops per stream when any of these conditions is met:

        1. **EOS** — the tokenizer's ``eos_token_id`` is predicted.
        2. **Blank** — the ``<blank>`` token is predicted (controlled by
           ``stop_on_blank``).
        3. **Footer sequence** — the last *N* tokens match ``self._asst_footer_ids``.
        4. **Max tokens** — ``max_new_tokens`` is reached.

        Args:
            logits: ``(B, L, V)`` logits from the LLM forward pass.
            cache: HF ``past_key_values`` with batch dim B.
            max_new_tokens: Maximum tokens to generate per stream.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            stop_on_blank: Controls blank-token stopping behavior:
                - ``True`` (default): stop whenever blank is predicted.
                  Use with dedicated ``<blank>`` special tokens.
                - ``"first"``: stop only if blank is the **first** token
                  generated (= "no speech this chunk").  Use when the blank
                  token is a natural text token (e.g. ``" "``).
                - ``False``: never stop on blank.
            generation_kwargs: Per-call overrides.

        Returns:
            ``(generated_per_stream, updated_cache, footer_consumed_per_stream, num_feed_steps)``
            where ``generated_per_stream`` is a list of B token-ID lists and
            ``num_feed_steps`` is how many tokens were fed to the LLM cache.
        """
        B = logits.shape[0]
        footer = self._asst_footer_ids
        flen = len(footer)
        generated: list[list[int]] = [[] for _ in range(B)]
        footer_consumed = [False] * B
        finished = [False] * B
        num_feed_steps = 0
        # Track which streams need their token fed to the LLM this step.
        # EOS tokens must NOT be fed; blank/footer/normal tokens must be fed.
        feed_mask = [False] * B

        next_tokens = self._sample_token(logits[:, -1, :], None, generation_config, **generation_kwargs)  # (B,)

        for _ in range(max_new_tokens):
            for b in range(B):
                feed_mask[b] = False
                if finished[b]:
                    continue
                tid = next_tokens[b].item()

                # EOS: stop WITHOUT feeding to LLM. Append a chunk separator so
                # decode_with_blank can join per-chunk outputs correctly.
                if self._eos_id is not None and tid == self._eos_id:
                    finished[b] = True
                    # Blank token when enabled, else EOS id itself (matches decode_with_blank).
                    generated[b].append(self.blank_token_id if self.has_blank else self._eos_id)
                    continue

                # All other tokens get appended and fed to LLM
                generated[b].append(tid)
                feed_mask[b] = True

                # Blank: stop (token IS fed to LLM, IS in generated).
                # When stop_on_blank == "first", only stop if blank is the
                # first generated token (= "no speech this chunk").  This
                # avoids false stops when the blank token collides with a
                # natural text token (e.g. " ") that appears mid-sentence.
                if tid == self.blank_token_id:
                    if stop_on_blank is True or (stop_on_blank == "first" and len(generated[b]) == 1):
                        finished[b] = True

                # Footer sequence match
                elif flen > 0 and len(generated[b]) >= flen and generated[b][-flen:] == footer:
                    generated[b] = generated[b][:-flen]
                    footer_consumed[b] = True
                    finished[b] = True

            # If no stream needs feeding, we're done
            if not any(feed_mask):
                break

            # Feed tokens to LLM. For finished streams, feed the blank token
            # (which the model was trained on) instead of a pad token, so the
            # KV cache stays clean — no foreign tokens that corrupt attention.
            # When blank is disabled, feed text_pad_id as a fallback.
            filler_id = self.blank_token_id if self.has_blank else self.text_pad_id
            tokens_to_feed = next_tokens.clone()
            for b in range(B):
                if not feed_mask[b]:
                    tokens_to_feed[b] = filler_id

            # All tokens are "real" (blank is a valid token), so all seq_lens grow
            if state is not None:
                for b in range(B):
                    state.seq_lens[b] += 1

            token_emb = self.embed_tokens(tokens_to_feed.unsqueeze(1))  # (B, 1, H)

            state.attention_mask = torch.cat(
                [
                    state.attention_mask,
                    torch.ones(B, 1, dtype=state.attention_mask.dtype, device=state.attention_mask.device),
                ],
                dim=1,
            )
            # Generated tokens are text. Under the chunk-restricted mask they may
            # attend to all text but only the current chunk's audio, so extend the
            # per-position bookkeeping and pass the 4D mask to match training.
            if self.core_cfg.restrict_audio_to_own_chunk and state.is_audio is not None:
                new_is_audio = torch.zeros(B, 1, dtype=torch.bool, device=state.attention_mask.device)
                # Generated transcription belongs to the current (most recent) chunk.
                new_chunk_id = torch.full(
                    (B, 1), state.n_audio_chunks, dtype=state.chunk_id.dtype, device=state.attention_mask.device
                )
                llm_attention_mask = self._extend_and_build_restricted_mask(
                    state, new_is_audio, new_chunk_id, token_emb.dtype
                )
            else:
                llm_attention_mask = state.attention_mask
            if self.core_cfg.two_stream_last_layer:
                # Generated/filler tokens are all text.
                text_is_audio = torch.zeros(B, 1, dtype=torch.bool, device=token_emb.device)
                out = self._two_stream_infer_step(token_emb, text_is_audio, state)
                cache = None
            else:
                out = self.llm(
                    inputs_embeds=token_emb,
                    past_key_values=cache,
                    attention_mask=llm_attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                cache = out.past_key_values
            num_feed_steps += 1

            if all(finished):
                break

            next_tokens = self._sample_token(out.logits[:, -1, :], None, generation_config, **generation_kwargs)

        return generated, cache, footer_consumed, num_feed_steps

    def get_audio_feature_buffer(
        self,
        batch_size: int,
        chunk_size_override: Optional[int] = None,
    ) -> BatchedCacheFeatureBufferer:
        """Get the audio feature buffer for the streaming state.

        Args:
            batch_size: Number of parallel streams.
            chunk_size_override: If provided, use this chunk size (in frames)
                instead of ``self._chunk_size_repr``.  Used by dynamic
                chunking inference where the inference step size differs
                from the config chunk_size.
        """
        preprocessor_cfg: DictConfig = self.perception.cfg.preprocessor
        window_stride_in_secs = preprocessor_cfg.window_stride
        pre_encode_cache_size = self.perception.encoder.streaming_cfg.pre_encode_cache_size
        if isinstance(pre_encode_cache_size, list):
            pre_encode_cache_size = pre_encode_cache_size[1]
        pre_encode_cache_size_in_secs = pre_encode_cache_size * window_stride_in_secs
        cs = chunk_size_override if chunk_size_override is not None else max(self._chunk_size_repr, 1)
        chunk_size_in_secs = cs * self.core_cfg.frame_length_in_secs
        buffer_size_in_secs = pre_encode_cache_size_in_secs + chunk_size_in_secs

        audio_feature_buffer = BatchedCacheFeatureBufferer(
            num_slots=batch_size,
            sample_rate=self.core_cfg.sample_rate,
            buffer_size_in_secs=buffer_size_in_secs,
            chunk_size_in_secs=buffer_size_in_secs,  # recalculate mel-spec for the whole buffer
            preprocessor_cfg=preprocessor_cfg,
            device=self.device,
        )
        return audio_feature_buffer

    def get_init_streaming_state(
        self,
        system_prompt: Union[str, List[str]],
        device: torch.device,
        batch_size: int = 1,
    ) -> StreamingState:
        """Forward the system prompt through the LLM and return a fresh :class:`StreamingState`.

        Args:
            system_prompt: System prompt string (shared) or list of B per-sample prompts.
            device: Target device.
            batch_size: Number of parallel streams (B).
        """
        hf_tok = self.tokenizer.tokenizer
        dtype = self.embed_tokens.weight.dtype

        if isinstance(system_prompt, str):
            prompts = [system_prompt] * batch_size
        else:
            prompts = system_prompt

        # Tokenize each prompt
        all_sys_ids = []
        for prompt in prompts:
            ids = hf_tok.apply_chat_template(
                [{"role": "system", "content": prompt}],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            all_sys_ids.append(ids)

        # Check if all prompts are the same length (common case: same prompt)
        sys_lens = [len(ids) for ids in all_sys_ids]
        needs_padding = len(set(sys_lens)) > 1

        # Capture hidden states from the prefill if the aux head will be used at
        # inference, so the aux backbone sees the same full-sequence context as
        # at training time.
        capture_hidden = self.core_cfg.use_chunk_classifier and self.core_cfg.chunk_classifier_use_at_inference

        if not needs_padding:
            # Fast path: all same length, no padding needed
            sys_embs = self.embed_tokens(
                torch.tensor(all_sys_ids[0], device=device, dtype=torch.long).unsqueeze(0)
            ).expand(batch_size, -1, -1)
            attention_mask = torch.ones(batch_size, sys_lens[0], dtype=torch.long, device=device)
            if self.core_cfg.two_stream_last_layer:
                out = None  # two-stream recomputes from buffers; no KV-cache prefill
            else:
                out = self.llm(
                    inputs_embeds=sys_embs,
                    attention_mask=attention_mask,
                    use_cache=True,
                    output_hidden_states=capture_hidden,
                    return_dict=True,
                )
            max_sys_len = sys_lens[0]
        else:
            # Per-sample prompts with different lengths: left-pad and use attention mask
            max_sys_len = max(sys_lens)
            H = self.embed_tokens.weight.shape[-1]
            sys_embs = torch.zeros(batch_size, max_sys_len, H, device=device, dtype=dtype)
            attention_mask = torch.zeros(batch_size, max_sys_len, dtype=torch.long, device=device)
            for b in range(batch_size):
                embs = self.embed_tokens(
                    torch.tensor(all_sys_ids[b], device=device, dtype=torch.long).unsqueeze(0)
                ).squeeze(
                    0
                )  # (L_b, H)
                offset = max_sys_len - sys_lens[b]
                sys_embs[b, offset:] = embs
                attention_mask[b, offset:] = 1
            if self.core_cfg.two_stream_last_layer:
                out = None  # two-stream recomputes from buffers; no KV-cache prefill
            else:
                out = self.llm(
                    inputs_embeds=sys_embs,
                    attention_mask=attention_mask,
                    use_cache=True,
                    output_hidden_states=capture_hidden,
                    return_dict=True,
                )

        aux_hidden_buffer = out.hidden_states[-1] if (out is not None and capture_hidden) else None

        # Two-stream inference: seed state with the (all-text) system prompt.
        ts_embeds = None
        ts_is_audio = None
        ts_text_cache = None
        ts_last_cache = None
        ts_text_len = 0
        ts_full_len = 0
        if self.core_cfg.two_stream_last_layer:
            sys_is_audio = torch.zeros(batch_size, max_sys_len, dtype=torch.bool, device=device)
            if getattr(self, "_two_stream_use_cache", True):
                if needs_padding:
                    raise NotImplementedError(
                        "Fast two-stream inference (KV-cached) requires a shared system prompt "
                        "(no left-padding). Use a single system_prompt string, or set "
                        "NEMO_TWO_STREAM_RECOMPUTE=1 for the slow per-sample-prompt path."
                    )
                ts_text_cache = DynamicCache()
                ts_last_cache = DynamicCache()
                layers, _norm, _rotary, _lm_head = self._resolve_llm_core()
                _logits, ts_text_len, ts_full_len = two_stream_cached_step(
                    layers=layers,
                    norm=_norm,
                    rotary_emb=_rotary,
                    lm_head=_lm_head,
                    input_embeds=sys_embs,
                    input_is_audio=sys_is_audio,
                    text_cache=ts_text_cache,
                    last_cache=ts_last_cache,
                    text_len=0,
                    full_len=0,
                    num_fusion_layers=getattr(self, "_two_stream_num_fusion_layers", 1),
                    compute_logits=False,
                )
            else:
                ts_embeds = sys_embs.contiguous()
                ts_is_audio = sys_is_audio

        # System-prompt positions are all text (chunk 0); seed per-position
        # bookkeeping only when the chunk-restricted attention mask is enabled.
        if self.core_cfg.restrict_audio_to_own_chunk:
            init_is_audio = torch.zeros(batch_size, max_sys_len, dtype=torch.bool, device=device)
            init_chunk_id = torch.zeros(batch_size, max_sys_len, dtype=torch.long, device=device)
        else:
            init_is_audio = None
            init_chunk_id = None

        cache_last_channel, cache_last_time, cache_last_channel_len = self.perception.get_initial_cache_state(
            batch_size=batch_size, dtype=dtype, device=device
        )
        audio_feature_buffer = self.get_audio_feature_buffer(batch_size=batch_size)
        audio_cache = CacheAwareContext(
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
        return StreamingState(
            cache=(None if out is None else out.past_key_values),
            generated_tokens=[[] for _ in range(batch_size)],
            seq_lens=[max_sys_len] * batch_size,
            audio_cache=audio_cache,
            audio_feature_buffer=audio_feature_buffer,
            attention_mask=attention_mask,
            aux_hidden_buffer=aux_hidden_buffer,
            is_audio=init_is_audio,
            chunk_id=init_chunk_id,
            n_audio_chunks=0,
            ts_embeds=ts_embeds,
            ts_is_audio=ts_is_audio,
            ts_text_cache=ts_text_cache,
            ts_last_cache=ts_last_cache,
            ts_text_len=ts_text_len,
            ts_full_len=ts_full_len,
            batch_size=batch_size,
        )

    def _extend_and_build_restricted_mask(
        self,
        state: 'StreamingState',
        input_is_audio: Tensor,
        input_chunk_id: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        """Append the new positions' audio/chunk bookkeeping and build the 4D mask.

        ``state.attention_mask`` must already be extended to include the new
        query positions. ``input_is_audio``/``input_chunk_id`` are ``(B, q_len)``.
        """
        state.is_audio = torch.cat([state.is_audio, input_is_audio], dim=1)
        state.chunk_id = torch.cat([state.chunk_id, input_chunk_id], dim=1)
        B, total = state.attention_mask.shape
        q_len = input_is_audio.shape[1]
        device = state.attention_mask.device
        key_abs = torch.arange(total, device=device).unsqueeze(0).expand(B, total)
        query_abs = torch.arange(total - q_len, total, device=device).unsqueeze(0).expand(B, q_len)
        return build_chunk_restricted_mask(
            key_is_audio=state.is_audio,
            key_chunk_id=state.chunk_id,
            key_valid=state.attention_mask.bool(),
            query_is_audio=input_is_audio,
            query_chunk_id=input_chunk_id,
            query_abs_pos=query_abs,
            key_abs_pos=key_abs,
            dtype=dtype,
        )

    @torch.no_grad()
    def _chunked_streaming_step(
        self,
        audio_chunks: Tensor,
        audio_chunk_lens: Optional[Tensor] = None,
        state: Optional[StreamingState] = None,
        max_new_tokens: int = 64,
        generation_config: Optional[GenerationConfig] = None,
        _audio_embs: Optional[Tensor] = None,
        **generation_kwargs,
    ) -> list[list[int]]:
        """
        Process B raw audio chunks and generate the assistant responses.

        Args:
            audio_chunks: ``(B, T_samples)`` raw waveforms for one chunk per stream.
            audio_chunk_lens: ``(B,)`` number of valid samples per stream.
            state: Mutable :class:`StreamingState` with ``batch_size=B`` (updated in place).
            max_new_tokens: Maximum tokens to generate per chunk per stream.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            _audio_embs: Optional pre-computed audio embeddings ``(B, chunk_size, H)``.
                Diagnostic use only.
            generation_kwargs: Per-call overrides for generation parameters.
        Returns:
            List of B token-ID lists (one per stream).
        """

        self._ensure_inference_cache()
        device = audio_chunks.device
        B = state.batch_size

        if _audio_embs is not None:
            audio_chunk_embs = _audio_embs.type_as(self.embed_tokens.weight)
        else:
            # 0. Update audio feature buffer — B frames, one per stream
            if audio_chunk_lens is None:
                audio_chunk_lens = torch.tensor([audio_chunks.shape[-1]] * B, device=device)
            frames = [
                Frame(
                    samples=audio_chunks[b] if audio_chunks.dim() == 2 else audio_chunks,
                    length=int(audio_chunk_lens[b].item()),
                    stream_id=b,
                )
                for b in range(B)
            ]
            features, right_paddings = state.audio_feature_buffer.update(frames)
            # Stack B feature buffers → (B, D, fbl)
            processed_signal = torch.stack(features).type_as(self.embed_tokens.weight)
            processed_signal_length = torch.tensor(
                [processed_signal.shape[-1] - int(rp) for rp in right_paddings],
                device=device,
            ).long()

            # 1. Encode audio chunks with streaming cache
            outputs = self.perception(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=state.audio_cache.cache_last_channel,
                cache_last_time=state.audio_cache.cache_last_time,
                cache_last_channel_len=state.audio_cache.cache_last_channel_len,
                streaming=True,
            )
            audio_chunk_embs, _, new_perception_cache = outputs

            # 2. Update streaming state with new perception cache
            if new_perception_cache is not None:
                state.audio_cache.cache_last_channel = new_perception_cache['cache_last_channel']
                state.audio_cache.cache_last_time = new_perception_cache['cache_last_time']
                state.audio_cache.cache_last_channel_len = new_perception_cache['cache_last_channel_len']

        # 3. Pad/trim to chunk_size frames
        chunk_size = self._chunk_size_repr
        n_frames = audio_chunk_embs.shape[1]
        if n_frames < chunk_size:
            audio_chunk_embs = F.pad(audio_chunk_embs, (0, 0, 0, chunk_size - n_frames))
        elif n_frames > chunk_size:
            audio_chunk_embs = audio_chunk_embs[:, :chunk_size, :]

        # 4. Build input embeddings from cached turn template — (B, L, H)
        turn_ids_t = torch.tensor(self._turn_template_ids, device=device).unsqueeze(0).expand(B, -1)  # (B, L)
        audio_mask = turn_ids_t == AUDIO_TOKEN_IDX  # (B, L)

        text_tokens = turn_ids_t.where(~audio_mask, torch.zeros_like(turn_ids_t))
        input_embeds = self.embed_tokens(text_tokens)  # (B, L, H)

        # Replace audio placeholder positions with actual audio embeddings
        input_embeds[audio_mask] = audio_chunk_embs.reshape(-1, audio_chunk_embs.shape[-1])

        # 5. Forward through LLM with cache
        input_len = input_embeds.shape[1]
        state.attention_mask = torch.cat(
            [state.attention_mask, torch.ones(B, input_len, dtype=state.attention_mask.dtype, device=device)],
            dim=1,
        )
        # Chunk-restricted attention: this turn's audio slots belong to a new
        # chunk. Text queries emitted for this chunk may only attend to this
        # chunk's audio (plus all text); audio queries keep full causal access.
        if self.core_cfg.restrict_audio_to_own_chunk and state.is_audio is not None:
            carry = state.n_audio_chunks
            new_chunk = carry + 1
            input_is_audio = audio_mask  # (B, input_len)
            # Match the training cumsum: positions before this turn's first audio
            # frame keep the previous chunk id; the audio run and everything after
            # it (write token / assistant header) belong to the new chunk.
            audio_cols = [i for i, t in enumerate(self._turn_template_ids) if t == AUDIO_TOKEN_IDX]
            first_audio = audio_cols[0] if audio_cols else input_len
            chunk_ids_row = torch.full((input_len,), carry, dtype=state.chunk_id.dtype, device=device)
            chunk_ids_row[first_audio:] = new_chunk
            input_chunk_id = chunk_ids_row.unsqueeze(0).expand(B, input_len)
            llm_attention_mask = self._extend_and_build_restricted_mask(
                state, input_is_audio, input_chunk_id, input_embeds.dtype
            )
            state.n_audio_chunks = new_chunk
        else:
            llm_attention_mask = state.attention_mask
        if self.core_cfg.two_stream_last_layer:
            out = self._two_stream_infer_step(input_embeds, audio_mask, state)
        else:
            out = self.llm(
                inputs_embeds=input_embeds,
                past_key_values=state.cache,
                attention_mask=llm_attention_mask,
                use_cache=True,
                return_dict=True,
            )
        state.cache = out.past_key_values
        for b in range(B):
            state.seq_lens[b] += input_len

        # 6. Autoregressive generation loop
        generated_per_stream, state.cache, footer_consumed, _ = self._autoregressive_decode(
            out.logits,
            state.cache,
            state,
            max_new_tokens,
            generation_config,
            **generation_kwargs,
        )

        # 7. Finalize turn — ensure end-of-turn tokens are in the cache.
        any_needs_footer = any(not fc for fc in footer_consumed)
        if any_needs_footer and self._asst_footer_ids:
            flen = len(self._asst_footer_ids)
            asst_footer_embs = self.embed_tokens(
                torch.tensor(self._asst_footer_ids, device=device).unsqueeze(0).expand(B, -1)
            )
            state.attention_mask = torch.cat(
                [state.attention_mask, torch.ones(B, flen, dtype=state.attention_mask.dtype, device=device)],
                dim=1,
            )
            # Footer tokens are text belonging to the current chunk.
            if self.core_cfg.restrict_audio_to_own_chunk and state.is_audio is not None:
                footer_is_audio = torch.zeros(B, flen, dtype=torch.bool, device=device)
                footer_chunk_id = torch.full(
                    (B, flen), state.n_audio_chunks, dtype=state.chunk_id.dtype, device=device
                )
                llm_attention_mask = self._extend_and_build_restricted_mask(
                    state, footer_is_audio, footer_chunk_id, asst_footer_embs.dtype
                )
            else:
                llm_attention_mask = state.attention_mask
            if self.core_cfg.two_stream_last_layer:
                footer_is_audio = torch.zeros(B, flen, dtype=torch.bool, device=device)
                out = self._two_stream_infer_step(asst_footer_embs, footer_is_audio, state)
            else:
                out = self.llm(
                    inputs_embeds=asst_footer_embs,
                    past_key_values=state.cache,
                    attention_mask=llm_attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
            state.cache = out.past_key_values
            for b in range(B):
                state.seq_lens[b] += flen
        elif all(footer_consumed):
            for b in range(B):
                state.seq_lens[b] += len(self._asst_footer_ids)

        # 8. Store and return
        for b in range(B):
            state.generated_tokens[b].extend(generated_per_stream[b])
        return generated_per_stream

    def _build_offline_emb_chunks(
        self,
        audio_wav: Tensor,
        n_samples: int,
        device: torch.device,
    ) -> list[Tensor]:
        """Pre-compute offline perception embeddings and slice into chunk_size groups.

        Runs the full perception module on the complete audio (the same path
        used during training), then splits the resulting embeddings into
        ``chunk_size``-frame groups that can be fed directly to the LLM turn
        template.  This bypasses both the feature buffer and the streaming
        encoder, isolating the LLM / generation logic from perception.

        Returns a list of ``(1, chunk_size, H)`` tensors, one per chunk.
        """
        chunk_size = self._chunk_size_repr
        with torch.no_grad():
            offline_embs, _ = self.perception(
                input_signal=audio_wav.unsqueeze(0),
                input_signal_length=torch.tensor([n_samples], device=device),
            )
        total_frames = offline_embs.shape[1]
        chunks: list[Tensor] = []
        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            chunk = offline_embs[:, start:end, :]
            if chunk.shape[1] < chunk_size:
                chunk = F.pad(chunk, (0, 0, 0, chunk_size - chunk.shape[1]))
            chunks.append(chunk)
        return chunks

    def _generate_offline(
        self,
        audios: Tensor,
        n_samples_list: list[int],
        system_prompt: Union[str, List[str]],
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        **generation_kwargs,
    ) -> list[str]:
        """Offline generation: process entire audio in a single LLM forward pass.

        Unlike the streaming path, this method runs offline perception on the
        full audio (no chunking, no streaming cache), builds one input sequence
        per sample (system prompt + user turn with all audio frames + assistant
        header), and prefills the LLM in a single forward pass before decoding.

        Args:
            audios: ``(B, T_samples)`` raw waveforms (zero-padded to max length).
            n_samples_list: List of B valid sample counts.
            system_prompt: System prompt string (shared) or list of B per-sample prompts.
            max_new_tokens: Maximum tokens to generate per sample.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            generation_kwargs: Per-call overrides for generation parameters.

        Returns:
            List of B transcription strings.
        """
        B = len(n_samples_list)
        if B == 0 or max(n_samples_list) == 0:
            return [""] * B
        device = audios.device
        dtype = self.embed_tokens.weight.dtype

        # 1. Encode system prompt(s)
        hf_tok = self.tokenizer.tokenizer
        prompts = [system_prompt] * B if isinstance(system_prompt, str) else system_prompt
        all_sys_embs = []
        for prompt in prompts:
            sys_ids = hf_tok.apply_chat_template(
                [{"role": "system", "content": prompt}],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            embs = self.embed_tokens(torch.tensor(sys_ids, device=device, dtype=torch.long).unsqueeze(0)).squeeze(
                0
            )  # (L_sys_b, H)
            all_sys_embs.append(embs)

        # 2. Embed turn template components (shared across batch)
        user_header_embs = self.embed_tokens(
            torch.tensor(self._user_header_ids, device=device, dtype=torch.long).unsqueeze(0)
        )  # (1, L_uh, H)
        uf_ah_embs = self.embed_tokens(
            torch.tensor(self._user_footer_and_asst_header_ids, device=device, dtype=torch.long).unsqueeze(0)
        )  # (1, L_uf, H)

        # 3. Run offline perception on the full batch
        audio_lens_t = torch.tensor(n_samples_list, device=device)
        batch_audio_embs, batch_emb_lens = self.perception(
            input_signal=audios,
            input_signal_length=audio_lens_t,
        )  # (B, T_enc_max, H), (B,)
        batch_audio_embs = batch_audio_embs.type_as(self.embed_tokens.weight)
        all_audio_embs = [batch_audio_embs[b, : int(batch_emb_lens[b].item())] for b in range(B)]

        # 4. Build per-sample input sequences:
        #    sys_embs[b] + user_header_embs + audio_embs[b] + user_footer_asst_header_embs
        sample_embs_list = []
        sample_lens = []
        for b in range(B):
            seq = torch.cat(
                [all_sys_embs[b], user_header_embs.squeeze(0), all_audio_embs[b], uf_ah_embs.squeeze(0)],
                dim=0,
            )  # (L_b, H)
            sample_embs_list.append(seq)
            sample_lens.append(seq.shape[0])

        # 5. Left-pad to max length and build attention mask
        max_len = max(sample_lens)
        H = sample_embs_list[0].shape[-1]
        input_embeds = torch.zeros(B, max_len, H, device=device, dtype=dtype)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for b in range(B):
            offset = max_len - sample_lens[b]
            input_embeds[b, offset:] = sample_embs_list[b]
            attention_mask[b, offset:] = 1

        # 6. LLM prefill (single forward pass)
        out = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        # 7. Autoregressive decode
        state = StreamingState(
            cache=out.past_key_values,
            generated_tokens=[[] for _ in range(B)],
            seq_lens=[max_len] * B,
            attention_mask=attention_mask,
            batch_size=B,
        )
        generated_per_stream, _, _, _ = self._autoregressive_decode(
            out.logits,
            out.past_key_values,
            state,
            max_new_tokens,
            generation_config,
            **generation_kwargs,
        )

        # 8. Decode tokens to text
        return [
            decode_with_blank(toks, self.blank_token, self.tokenizer, write_token=self.core_cfg.write_token)
            for toks in generated_per_stream
        ]

    def _generate_dynamic_streaming(
        self,
        audios: Tensor,
        n_samples_list: list[int],
        system_prompt: Union[str, List[str]],
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        inference_chunk_size: Optional[int] = None,
        dynamic_min_chunk_size: int = 0,
        dynamic_max_chunk_size: Optional[int] = None,
        lm_head_emit_threshold: Optional[float] = None,
        debug_logs: Optional[list] = None,
        chunk_separator: Optional[str] = None,
        **generation_kwargs,
    ) -> list[str]:
        """Batched dynamic-chunking generation.

        All B streams are processed in lockstep: each step feeds exactly 1
        embedding per stream to the LLM (audio frame, template token, or
        generated text token).  Perception runs with ``inference_chunk_size``
        frames and the resulting embeddings are buffered per-stream so the
        LLM still consumes them one at a time.

        Args:
            audios: ``(B, T_samples)`` raw waveforms.
            n_samples_list: List of B valid sample counts.
            system_prompt: System prompt string or list.
            max_new_tokens: Max tokens per text generation segment.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            inference_chunk_size: Number of encoder frames per perception call
                (default 1).  Embeddings are buffered and fed to the LLM one
                at a time.
            dynamic_min_chunk_size: Minimum frames before the model is allowed
                to trigger generation (default 0, no minimum).
            dynamic_max_chunk_size: Maximum frames before forcing generation.
                ``None`` means no upper bound (default).
            generation_kwargs: Per-call overrides.
        """
        if self.core_cfg.two_stream_last_layer:
            raise NotImplementedError(
                "two_stream_last_layer inference is only wired for the static fixed-chunk "
                "path (_generate_chunked_streaming). Do not set use_state_machine_inference=True "
                "with two_stream_last_layer."
            )
        B = len(n_samples_list)
        if B == 0:
            return []
        device = audios.device
        # Default inference_chunk_size: match the encoder granularity the
        # model was trained at — chunk_size for fixed chunking, 1 for dynamic
        # chunking. Even though the streaming encoder is conceptually causal,
        # cache-aware Conformer's internal chunking + lookahead window is NOT
        # identical at different batch sizes (train/inference mismatch causes
        # slight embedding drift → measurable WER regression).
        # Override via inference_chunk_size only when you've trained the
        # encoder for that granularity (e.g. chunk_step > 1 in dataset).
        if inference_chunk_size is None:
            N = max(self._chunk_size_repr, 1)
        else:
            N = inference_chunk_size
        chunk_samples = math.ceil(N * self.core_cfg.frame_length_in_secs * self.core_cfg.sample_rate)

        # --- Init state ---
        state = self.get_init_streaming_state(system_prompt, device=device, batch_size=B)
        state.audio_feature_buffer = self.get_audio_feature_buffer(
            batch_size=B,
            chunk_size_override=N,
        )

        # --- Per-stream state machine ---
        HEADER, LISTENING, FOOTER, GENERATING, BLANK_FEED, ASST_FOOTER, DONE = range(7)
        # When user_header_ids is empty (compact template), skip HEADER entirely.
        _initial_state = LISTENING if not self._user_header_ids else HEADER
        stream_state = [_initial_state] * B
        template_pos = [0] * B  # position within current template seq
        audio_sample_idx = [0] * B  # next audio sample offset for perception
        gen_token_count = [0] * B  # tokens generated in current GENERATING phase
        last_gen_token = [self.text_pad_id] * B  # last generated token per stream
        all_tokens: list[list[int]] = [[] for _ in range(B)]

        # Per-stream audio embedding buffer (filled by perception, consumed 1 at a time)
        audio_emb_buf: list[list[Tensor]] = [[] for _ in range(B)]

        # Fixed-chunk mode: count frames consumed per segment to transition
        # after exactly chunk_size frames (ignoring model predictions).
        fixed_chunk_mode = self._chunk_size_repr > 0
        fixed_chunk_size = self._chunk_size_repr if fixed_chunk_mode else 0
        frames_in_segment = [0] * B  # frames consumed in current LISTENING segment

        # --- Audio-frame debug logging ---
        # When debug_logs is provided (a list passed in by the caller), we
        # populate it with per-LISTENING-frame diagnostic records per stream
        # — used to investigate whether the model is overfitting to predict
        # blank, whether the aux head is well-calibrated, etc.
        log_frames = debug_logs is not None
        per_stream_frame_logs: list[list[dict]] = [[] for _ in range(B)] if log_frames else []
        total_frame_idx = [0] * B  # cumulative LISTENING frames per stream

        uf_ah_ids = self._user_footer_and_asst_header_ids
        uh_ids = self._user_header_ids
        af_ids = self._asst_footer_ids
        user_footer_first_id = self._user_footer_first_id

        # Max steps: LLM's max context length minus the system prompt already in KV cache.
        max_model_len = getattr(self.llm.config, 'max_position_embeddings', 40960)
        max_steps = max_model_len - max(state.seq_lens)

        # Padding embedding for DONE streams and empty-buffer LISTENING streams.
        # Use the blank token embedding when blank is enabled (a real token the
        # model knows). Otherwise fall back to the text pad id.
        pad_token_id = self.blank_token_id if self.has_blank else self.text_pad_id
        pad_emb = self.embed_tokens(torch.tensor([pad_token_id], device=device)).squeeze(0)  # (H,)

        for _step in range(max_steps):
            # --- Refill audio embedding buffers for LISTENING streams ---
            needs_refill = [
                b
                for b in range(B)
                if stream_state[b] == LISTENING
                and len(audio_emb_buf[b]) == 0
                and audio_sample_idx[b] < n_samples_list[b]
            ]
            if needs_refill:
                # Run perception only for streams that need refill.
                # The feature buffer selectively updates via stream_id.
                # The encoder cache is sliced to the subset, then scattered back.
                idx_t = torch.tensor(needs_refill, device=device)

                # Build frames (only for refill streams)
                frames = []
                for b in needs_refill:
                    start = audio_sample_idx[b]
                    end = min(start + chunk_samples, n_samples_list[b])
                    wav = audios[b, start:end]
                    if wav.shape[0] < chunk_samples:
                        wav = F.pad(wav, (0, chunk_samples - wav.shape[0]))
                    frames.append(Frame(samples=wav, stream_id=b, length=end - start))
                    audio_sample_idx[b] = end

                # Feature buffer selectively updates only the submitted stream_ids
                features, right_paddings = state.audio_feature_buffer.update(frames)
                processed_signal = torch.stack(features).type_as(self.embed_tokens.weight)  # (S, D, T)
                processed_signal_length = torch.tensor(
                    [processed_signal.shape[-1] - int(rp) for rp in right_paddings],
                    device=device,
                ).long()

                # Slice encoder cache to the subset
                sub_cache_lc = state.audio_cache.cache_last_channel.index_select(1, idx_t)
                sub_cache_lt = state.audio_cache.cache_last_time.index_select(1, idx_t)
                sub_cache_lcl = state.audio_cache.cache_last_channel_len[idx_t]

                outputs = self.perception(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=sub_cache_lc,
                    cache_last_time=sub_cache_lt,
                    cache_last_channel_len=sub_cache_lcl,
                    streaming=True,
                )
                batch_embs, _, new_cache = outputs

                # Scatter updated cache back into the full B-sized cache
                if new_cache is not None:
                    for i, b in enumerate(needs_refill):
                        state.audio_cache.cache_last_channel[:, b] = new_cache['cache_last_channel'][:, i]
                        state.audio_cache.cache_last_time[:, b] = new_cache['cache_last_time'][:, i]
                        state.audio_cache.cache_last_channel_len[b] = new_cache['cache_last_channel_len'][i]

                # Distribute embeddings into per-stream buffers.
                # Pad to exactly N frames (matching fast path's pad/trim behavior).
                H_enc = batch_embs.shape[-1]
                for i, b in enumerate(needs_refill):
                    n_enc = batch_embs[i].shape[0]
                    for f in range(n_enc):
                        audio_emb_buf[b].append(batch_embs[i, f])
                    # Pad with zeros if encoder returned fewer than N frames
                    for _ in range(N - n_enc):
                        audio_emb_buf[b].append(torch.zeros(H_enc, device=device, dtype=batch_embs.dtype))

            # --- Build (B, 1, H) input embeddings based on per-stream state ---
            # Each entry is (H,); we stack → (B, H) then unsqueeze → (B, 1, H).
            embs_list = []
            for b in range(B):
                if stream_state[b] == LISTENING:
                    if audio_emb_buf[b]:
                        embs_list.append(audio_emb_buf[b].pop(0))  # (H,)
                    else:
                        embs_list.append(pad_emb)
                elif stream_state[b] == FOOTER:
                    tid = uf_ah_ids[template_pos[b]]
                    embs_list.append(self.embed_tokens(torch.tensor([tid], device=device)).squeeze(0))  # (H,)
                elif stream_state[b] == GENERATING:
                    embs_list.append(
                        self.embed_tokens(torch.tensor([last_gen_token[b]], device=device)).squeeze(0)  # (H,)
                    )
                elif stream_state[b] == BLANK_FEED:
                    # Only reached when has_blank is True (guarded at transition sites).
                    embs_list.append(
                        self.embed_tokens(torch.tensor([self.blank_token_id], device=device)).squeeze(0)  # (H,)
                    )
                elif stream_state[b] == ASST_FOOTER:
                    tid = af_ids[template_pos[b]]
                    embs_list.append(self.embed_tokens(torch.tensor([tid], device=device)).squeeze(0))  # (H,)
                elif stream_state[b] == HEADER:
                    tid = uh_ids[template_pos[b]]
                    embs_list.append(self.embed_tokens(torch.tensor([tid], device=device)).squeeze(0))  # (H,)
                else:  # DONE
                    embs_list.append(pad_emb)

            input_embs = torch.stack(embs_list).unsqueeze(1)  # (B, H) → (B, 1, H)

            # --- Single LLM forward ---
            use_aux = self.core_cfg.use_chunk_classifier and self.core_cfg.chunk_classifier_use_at_inference
            llm_kwargs = dict(
                inputs_embeds=input_embs,
                past_key_values=state.cache,
                use_cache=True,
                output_hidden_states=use_aux,
                return_dict=True,
            )
            if state.attention_mask is not None:
                state.attention_mask = torch.cat(
                    [state.attention_mask, torch.ones(B, 1, dtype=state.attention_mask.dtype, device=device)],
                    dim=1,
                )
                llm_kwargs["attention_mask"] = state.attention_mask
            out = self.llm(**llm_kwargs)
            state.cache = out.past_key_values
            for b in range(B):
                state.seq_lens[b] += 1

            # --- Aux chunk-boundary classifier: full-sequence forward ---
            # Append the new LLM hidden state to the running buffer, then run
            # the K-layer aux backbone over the entire accumulated buffer
            # (no aux KV cache — matches training, where the aux backbone sees
            # the full sequence in one pass). Cost is K layers × current length
            # per step; cheap at K≈2.
            aux_last_hidden = None
            if use_aux:
                new_h = out.hidden_states[-1]  # (B, 1, H)
                if state.aux_hidden_buffer is None:
                    state.aux_hidden_buffer = new_h
                else:
                    state.aux_hidden_buffer = torch.cat([state.aux_hidden_buffer, new_h], dim=1)
                aux_out = self.chunk_classifier_backbone(
                    inputs_embeds=state.aux_hidden_buffer,
                    attention_mask=state.attention_mask,
                    return_dict=True,
                )
                aux_last_hidden = aux_out.last_hidden_state  # (B, L_so_far, H)

            # --- Per-stream state transitions ---
            for b in range(B):
                if stream_state[b] == DONE:
                    continue

                if stream_state[b] == LISTENING:
                    frames_in_segment[b] += 1
                    total_frame_idx[b] += 1
                    seg_idx_now = frames_in_segment[b]  # captured before any reset
                    decision_str = "keep_listening"
                    aux_p_log: Optional[float] = None  # only set when aux head consulted
                    if fixed_chunk_mode:
                        # Fixed chunking: transition after exactly chunk_size frames
                        if frames_in_segment[b] >= fixed_chunk_size:
                            stream_state[b] = FOOTER
                            template_pos[b] = 0
                            frames_in_segment[b] = 0
                            decision_str = "emit_forced_chunk_size"
                        elif not audio_emb_buf[b] and audio_sample_idx[b] >= n_samples_list[b]:
                            stream_state[b] = DONE
                            decision_str = "done_audio_end"
                    else:
                        # Dynamic chunking: transition when model predicts <user_footer>,
                        # subject to [min_chunk_size, max_chunk_size] bounds.
                        if dynamic_max_chunk_size is not None and frames_in_segment[b] >= dynamic_max_chunk_size:
                            # Forced transition — hit upper bound
                            stream_state[b] = FOOTER
                            template_pos[b] = 0
                            frames_in_segment[b] = 0
                            decision_str = "emit_forced_max"
                        elif frames_in_segment[b] < dynamic_min_chunk_size:
                            # Below minimum — ignore model prediction, keep listening
                            if not audio_emb_buf[b] and audio_sample_idx[b] >= n_samples_list[b]:
                                # Audio exhausted before reaching min — still emit
                                stream_state[b] = FOOTER
                                template_pos[b] = 0
                                frames_in_segment[b] = 0
                                decision_str = "emit_forced_audio_end_below_min"
                            else:
                                decision_str = "below_min_keep"
                        else:
                            # In [min, max] window — use model prediction.
                            # Either the aux classifier head (when enabled) or
                            # the LM head's vocab sample (legacy path).
                            if use_aux and aux_last_hidden is not None:
                                h_last = aux_last_hidden[b, -1, :]  # (H,)
                                aux_logit_b = self.chunk_classifier_head(h_last)
                                aux_p_log = float(torch.sigmoid(aux_logit_b).item())
                                emit = aux_p_log >= self.core_cfg.chunk_classifier_threshold
                            elif lm_head_emit_threshold is not None:
                                # Threshold-based LM-head decision: fire when
                                # p(user_footer_first_id) ≥ threshold. Lower
                                # values catch boundaries where the LM is
                                # moderately confident but loses argmax to blank.
                                lm_probs_emit = torch.softmax(out.logits[b, -1, :].float(), dim=-1)
                                p_ufid_emit = (
                                    float(lm_probs_emit[user_footer_first_id].item())
                                    if user_footer_first_id is not None
                                    else 0.0
                                )
                                emit = p_ufid_emit >= lm_head_emit_threshold
                            else:
                                token = self._sample_token(
                                    out.logits[b : b + 1, -1, :],
                                    None,
                                    generation_config,
                                    **generation_kwargs,
                                ).item()
                                emit = token == user_footer_first_id
                            if emit:
                                stream_state[b] = FOOTER
                                template_pos[b] = 0
                                frames_in_segment[b] = 0
                                decision_str = "emit_model"
                            elif (
                                not audio_emb_buf[b] and audio_sample_idx[b] >= n_samples_list[b] and not all_tokens[b]
                            ):
                                # Audio exhausted in [min, max] window AND no
                                # text emitted yet for this stream — force a
                                # final FOOTER → GENERATING sweep so we don't
                                # produce an empty prediction. Once any chunk
                                # has been emitted, the model's "keep listening
                                # at end of audio" signal is trustworthy ("I'm
                                # done"), so we go to DONE without forcing —
                                # avoiding the trailing-hallucination failure
                                # mode where forced-emit invents extra text.
                                stream_state[b] = FOOTER
                                template_pos[b] = 0
                                frames_in_segment[b] = 0
                                decision_str = "emit_forced_audio_end"
                            elif not audio_emb_buf[b] and audio_sample_idx[b] >= n_samples_list[b]:
                                # Already emitted at least once and model says
                                # blank at end-of-audio: trust it and stop.
                                stream_state[b] = DONE
                                decision_str = "done_audio_end"

                    # Per-frame debug log (LM head + aux head diagnostics).
                    if log_frames:
                        lm_logits_b = out.logits[b, -1, :]
                        lm_probs_b = torch.softmax(lm_logits_b.float(), dim=-1)
                        topk_p, topk_id = torch.topk(lm_probs_b, 5)
                        lm_top5 = [
                            {"id": int(topk_id[k].item()), "prob": float(topk_p[k].item())}
                            for k in range(topk_p.numel())
                        ]
                        p_ufid = (
                            float(lm_probs_b[user_footer_first_id].item())
                            if user_footer_first_id is not None
                            else None
                        )
                        p_blank = float(lm_probs_b[self.blank_token_id].item()) if self.has_blank else None
                        # If aux is on but we didn't consult it (below min / above
                        # max / fixed-chunk path), still compute it for visibility.
                        if aux_p_log is None and use_aux and aux_last_hidden is not None:
                            aux_logit_b = self.chunk_classifier_head(aux_last_hidden[b, -1, :])
                            aux_p_log = float(torch.sigmoid(aux_logit_b).item())
                        per_stream_frame_logs[b].append(
                            {
                                "step": _step,
                                "total_frame_idx": total_frame_idx[b],
                                "frame_idx_in_segment": seg_idx_now,
                                "lm_top5": lm_top5,
                                "lm_p_user_footer_first": p_ufid,
                                "lm_p_blank": p_blank,
                                "aux_p_emit": aux_p_log,
                                "decision": decision_str,
                            }
                        )

                elif stream_state[b] == FOOTER:
                    template_pos[b] += 1
                    if template_pos[b] >= len(uf_ah_ids):
                        stream_state[b] = GENERATING
                        gen_token_count[b] = 0
                        # Use the logit from this step as the first generation logit
                        first_token = self._sample_token(
                            out.logits[b : b + 1, -1, :],
                            None,
                            generation_config,
                            **generation_kwargs,
                        ).item()
                        first_is_stop = (
                            self._eos_id is not None and first_token == self._eos_id
                        ) or first_token == self.blank_token_id
                        if first_is_stop:
                            # Immediately done generating — append chunk separator
                            # (blank when enabled, else EOS so decode_with_blank splits chunks)
                            all_tokens[b].append(self.blank_token_id if self.has_blank else self._eos_id)
                            if fixed_chunk_mode and self.has_blank:
                                # Feed blank to LLM first (matches training sequence)
                                stream_state[b] = BLANK_FEED
                            elif af_ids:
                                stream_state[b] = ASST_FOOTER
                                template_pos[b] = 0
                            else:
                                self._dynamic_finish_generating(
                                    b,
                                    stream_state,
                                    template_pos,
                                    audio_emb_buf,
                                    audio_sample_idx,
                                    n_samples_list,
                                    _initial_state,
                                    DONE,
                                )
                        else:
                            all_tokens[b].append(first_token)
                            last_gen_token[b] = first_token
                            gen_token_count[b] = 1

                elif stream_state[b] == GENERATING:
                    token = self._sample_token(
                        out.logits[b : b + 1, -1, :],
                        None,
                        generation_config,
                        **generation_kwargs,
                    ).item()
                    # Stop on EOS, blank, or max tokens (matching _autoregressive_decode)
                    is_eos = self._eos_id is not None and token == self._eos_id
                    is_blank = token == self.blank_token_id
                    is_max = gen_token_count[b] >= max_new_tokens
                    if is_eos or is_blank or is_max:
                        # Append chunk separator (blank when enabled, else EOS).
                        # decode_with_blank splits per-chunk outputs on this.
                        all_tokens[b].append(self.blank_token_id if self.has_blank else self._eos_id)
                        # Do NOT feed <blank> here. Training for non-empty
                        # chunks ends as `text <asst_footer>` (no blank between
                        # text and asst_footer — blank only appears as the
                        # *content* of empty chunks, which exits via FOOTER's
                        # first_is_stop path above). Feeding <blank> at this
                        # position pollutes the KV cache with an OOD token,
                        # manifesting as premature EOS in subsequent chunks
                        # (heavy deletion errors, especially in compact mode
                        # where the single asst_footer token can't recover the
                        # context).
                        if af_ids:
                            stream_state[b] = ASST_FOOTER
                            template_pos[b] = 0
                        else:
                            self._dynamic_finish_generating(
                                b,
                                stream_state,
                                template_pos,
                                audio_emb_buf,
                                audio_sample_idx,
                                n_samples_list,
                                _initial_state,
                                DONE,
                            )
                    else:
                        all_tokens[b].append(token)
                        last_gen_token[b] = token
                        gen_token_count[b] += 1

                elif stream_state[b] == BLANK_FEED:
                    # Blank was fed to LLM this step. Transition to ASST_FOOTER.
                    if af_ids:
                        stream_state[b] = ASST_FOOTER
                        template_pos[b] = 0
                    else:
                        self._dynamic_finish_generating(
                            b,
                            stream_state,
                            template_pos,
                            audio_emb_buf,
                            audio_sample_idx,
                            n_samples_list,
                            _initial_state,
                            DONE,
                        )

                elif stream_state[b] == ASST_FOOTER:
                    template_pos[b] += 1
                    if template_pos[b] >= len(af_ids):
                        self._dynamic_finish_generating(
                            b,
                            stream_state,
                            template_pos,
                            audio_emb_buf,
                            audio_sample_idx,
                            n_samples_list,
                            _initial_state,
                            DONE,
                        )

                elif stream_state[b] == HEADER:
                    template_pos[b] += 1
                    if template_pos[b] >= len(uh_ids):
                        stream_state[b] = LISTENING

            if all(s == DONE for s in stream_state):
                break

        if log_frames:
            debug_logs.extend(per_stream_frame_logs)
        return [
            decode_with_blank(
                toks,
                self.blank_token,
                self.tokenizer,
                replace_blank=chunk_separator,
                write_token=self.core_cfg.write_token,
            )
            for toks in all_tokens
        ]

    @staticmethod
    def _dynamic_finish_generating(
        b,
        stream_state,
        template_pos,
        audio_emb_buf,
        audio_sample_idx,
        n_samples_list,
        next_listen_state,
        DONE,
    ):
        """Transition stream b from GENERATING to next_listen_state (HEADER or LISTENING) or DONE.

        ``next_listen_state`` is HEADER when user_header_ids is non-empty, or LISTENING
        when user_header_ids is empty (compact template) — the HEADER state is skipped
        since there are no header tokens to feed.
        """
        has_more_audio = bool(audio_emb_buf[b]) or audio_sample_idx[b] < n_samples_list[b]
        if has_more_audio:
            stream_state[b] = next_listen_state
            template_pos[b] = 0
        else:
            stream_state[b] = DONE

    def _generate_chunked_streaming(
        self,
        audios: Tensor,
        n_samples_list: list[int],
        system_prompt: Union[str, List[str]],
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        use_offline_embs: bool = False,
        chunk_separator: Optional[str] = None,
        **generation_kwargs,
    ) -> list[str]:
        """Chunk-by-chunk streaming generation for B samples in lockstep.

        Args:
            audios: ``(B, T_samples)`` raw waveforms (zero-padded to max length).
            n_samples_list: List of B valid sample counts.
            system_prompt: System prompt string (shared) or list of B per-sample prompts.
            max_new_tokens: Maximum tokens to generate per chunk per stream.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            use_offline_embs: When True, bypass streaming perception with offline embeddings.
            generation_kwargs: Per-call overrides for generation parameters.

        Returns:
            List of B transcription strings.
        """
        assert self._chunk_size_repr > 0, (
            f"chunk_size must be positive for streaming mode, got {self._chunk_size_repr}. "
            f"Use generate() which dispatches to _generate_offline() for chunk_size < 0."
        )
        B = len(n_samples_list)
        if B == 0 or max(n_samples_list) == 0:
            return [""] * B
        device = audios.device
        chunk_size = self._chunk_size_repr
        chunk_samples = math.ceil(chunk_size * self.core_cfg.frame_length_in_secs * self.core_cfg.sample_rate)
        state = self.get_init_streaming_state(system_prompt, device=device, batch_size=B)

        offline_emb_chunks_list = None
        if use_offline_embs:
            offline_emb_chunks_list = [
                self._build_offline_emb_chunks(audios[b, : n_samples_list[b]], n_samples_list[b], device)
                for b in range(B)
            ]

        num_chunks_per_stream = [math.ceil(ns / chunk_samples) if ns > 0 else 0 for ns in n_samples_list]
        max_chunks = max(num_chunks_per_stream)
        all_token_ids: list[list[int]] = [[] for _ in range(B)]
        all_chunk_token_ids: list[list[list[int]]] = [[] for _ in range(B)]

        for chunk_i in range(max_chunks):
            # Build B audio chunks (zero-pad finished streams)
            chunks = []
            chunk_lens = []
            for b in range(B):
                start = chunk_i * chunk_samples
                end = min(start + chunk_samples, n_samples_list[b])
                if start >= n_samples_list[b]:
                    # Stream b has finished — send zeros with zero valid length
                    chunks.append(torch.zeros(chunk_samples, device=device, dtype=audios.dtype))
                    chunk_lens.append(0)
                else:
                    wav = audios[b, start:end]
                    if wav.shape[0] < chunk_samples:
                        wav = F.pad(wav, (0, chunk_samples - wav.shape[0]))
                    chunks.append(wav)
                    chunk_lens.append(end - start)

            audio_batch = torch.stack(chunks)  # (B, chunk_samples)
            lens_batch = torch.tensor(chunk_lens, device=device)

            extra_kwargs = {}
            if offline_emb_chunks_list is not None:
                emb_chunks = []
                for b in range(B):
                    if chunk_i < len(offline_emb_chunks_list[b]):
                        emb_chunks.append(offline_emb_chunks_list[b][chunk_i])
                    else:
                        H = offline_emb_chunks_list[0][0].shape[-1]
                        emb_chunks.append(torch.zeros(1, chunk_size, H, device=device, dtype=audios.dtype))
                extra_kwargs["_audio_embs"] = torch.cat(emb_chunks, dim=0)

            chunk_tokens = self._chunked_streaming_step(
                audio_batch,
                lens_batch,
                state,
                max_new_tokens,
                generation_config,
                **extra_kwargs,
                **generation_kwargs,
            )
            for b in range(B):
                # Only collect tokens for streams that are still active
                if chunk_i < num_chunks_per_stream[b]:
                    all_token_ids[b].extend(chunk_tokens[b])
                    all_chunk_token_ids[b].append(chunk_tokens[b])

        if chunk_separator is not None:
            return [
                f" {chunk_separator} ".join(
                    decode_with_blank(
                        chunk_tokens,
                        self.blank_token,
                        self.tokenizer,
                        strip_whitespace=True,
                        write_token=self.core_cfg.write_token,
                    )
                    for chunk_tokens in stream_chunks
                )
                for stream_chunks in all_chunk_token_ids
            ]

        return [
            decode_with_blank(
                toks,
                self.blank_token,
                self.tokenizer,
                write_token=self.core_cfg.write_token,
            )
            for toks in all_token_ids
        ]

    @torch.no_grad()
    def generate(
        self,
        audios: Tensor,
        audio_lens: Tensor,
        system_prompt: Union[str, List[str]] = "Transcribe the audio into text.",
        max_new_tokens: int = 64,
        generation_config: Optional[GenerationConfig] = None,
        use_offline_embs: bool = False,
        use_state_machine_inference: bool = False,
        dynamic_min_chunk_size: int = 0,
        dynamic_max_chunk_size: Optional[int] = None,
        lm_head_emit_threshold: Optional[float] = None,
        debug_logs: Optional[list] = None,
        chunk_separator: Optional[str] = None,
        chunk_size_override: Optional[int] = None,
        **generation_kwargs,
    ) -> list[str]:
        """
        Transcribe full audio(s).

        Args:
            audios: (B, T_samples) raw waveforms.
            audio_lens: (B,) waveform lengths in samples.
            system_prompt: System prompt string (shared) or list of B per-sample prompts.
            max_new_tokens: Maximum tokens to generate per chunk per stream.
            generation_config: Optional HuggingFace GenerationConfig object.
            use_offline_embs: When True, bypass streaming perception with
                offline embeddings. Diagnostic use only.
            dynamic_min_chunk_size: For dynamic chunking — minimum frames before
                the model is allowed to trigger generation (default 0).
            dynamic_max_chunk_size: For dynamic chunking — maximum frames before
                forcing generation. ``None`` means no upper bound (default).
            chunk_separator: If set, include this marker at decoded streaming
                chunk boundaries.
            chunk_size_override: Decode at this fixed chunk size (encoder frames)
                instead of the config default. For multi chunk-size models the
                default is the longest candidate; pass e.g. 2/4/7/... to measure a
                specific latency. Ignored for dynamic/offline configs.
            generation_kwargs: Per-call overrides for generation parameters.

        Returns:
            List of transcription strings, one per sample.
        """
        # Resolve the decode chunk size (override > longest-candidate/scalar) and
        # pin it: every inference path reads self._chunk_size_repr. For fixed
        # chunking also rebuild the turn template for this size (its audio-slot
        # count == chunk size). Restore both afterwards so a later call at a
        # different size (e.g. training val vs a sweep) is unaffected.
        cs = self._resolve_inference_chunk_size(chunk_size_override)
        saved_repr = getattr(self, "_chunk_size_repr", None)
        saved_template = getattr(self, "_turn_template_ids", None)
        self._chunk_size_repr = cs
        try:
            self._ensure_inference_cache()  # size-independent ids (header/footer/eos)
            if cs > 0:
                self._turn_template_ids = (
                    list(self._user_header_ids)
                    + [AUDIO_TOKEN_IDX] * cs
                    + list(self._user_footer_and_asst_header_ids)
                )
            # Match the encoder's streaming look-ahead to the decode chunk size.
            self._set_encoder_att_context(cs, recompute_streaming=True)

            with move_embedding(self):
                B = audios.shape[0]
                n_samples_list = [int(audio_lens[b].item()) for b in range(B)]

                if cs < 0:
                    results = self._generate_offline(
                        audios,
                        n_samples_list,
                        system_prompt,
                        max_new_tokens,
                        generation_config,
                        **generation_kwargs,
                    )
                elif cs == 0 or use_state_machine_inference:
                    # Dynamic chunking (chunk_size=0) or state machine inference opted in for chunk_size > 0.
                    # Note that for chunk_size > 0, use_state_machine_inference is not recommended.
                    results = self._generate_dynamic_streaming(
                        audios,
                        n_samples_list,
                        system_prompt,
                        max_new_tokens,
                        generation_config,
                        dynamic_min_chunk_size=dynamic_min_chunk_size,
                        dynamic_max_chunk_size=dynamic_max_chunk_size,
                        lm_head_emit_threshold=lm_head_emit_threshold,
                        debug_logs=debug_logs,
                        chunk_separator=chunk_separator,
                        **generation_kwargs,
                    )
                else:
                    # Static chunking (chunk_size > 0): bulk prefill + auto-regressive decode.
                    results = self._generate_chunked_streaming(
                        audios,
                        n_samples_list,
                        system_prompt,
                        max_new_tokens,
                        generation_config,
                        use_offline_embs=use_offline_embs,
                        chunk_separator=chunk_separator,
                        **generation_kwargs,
                    )
        finally:
            self._chunk_size_repr = saved_repr
            self._turn_template_ids = saved_template

        return results
