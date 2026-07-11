# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for chunk-restricted audio attention masks (restrict_audio_to_own_chunk)."""

import torch

from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX
from nemo.collections.speechlm2.models.streaming_stt_model import (
    audio_positions_and_chunk_ids,
    build_chunk_restricted_mask,
    build_training_chunk_restricted_mask,
)

PAD = 0
TXT = 5  # arbitrary non-audio, non-pad text token id
A = AUDIO_TOKEN_IDX

# Sequence layout (indices):
#   0,1   system text            (chunk 0)
#   2,3,4 chunk 1 audio          (chunk 1)
#   5,6   text turn 1            (carry 1)
#   7,8,9 chunk 2 audio          (chunk 2)
#   10,11 text turn 2            (carry 2)
#   12    padding
SEQ = [TXT, TXT, A, A, A, TXT, TXT, A, A, A, TXT, TXT, PAD]


def _blocked(value: torch.Tensor) -> bool:
    return float(value) == torch.finfo(torch.float32).min


def _allowed(value: torch.Tensor) -> bool:
    return float(value) == 0.0


def test_audio_positions_and_chunk_ids():
    input_tokens = torch.tensor([SEQ], dtype=torch.long)
    is_audio, chunk_id = audio_positions_and_chunk_ids(input_tokens)

    assert is_audio[0].tolist() == [False, False, True, True, True, False, False, True, True, True, False, False, False]
    # Audio runs get incrementing chunk ids; text carries the running count.
    assert chunk_id[0, 2:5].tolist() == [1, 1, 1]
    assert chunk_id[0, 7:10].tolist() == [2, 2, 2]


def test_text_query_blocks_previous_chunk_audio():
    input_tokens = torch.tensor([SEQ], dtype=torch.long)
    mask = build_training_chunk_restricted_mask(input_tokens, PAD, torch.float32)
    assert mask.shape == (1, 1, len(SEQ), len(SEQ))

    # Query = chunk-2 transcription text (index 10): only its own chunk's audio.
    assert _blocked(mask[0, 0, 10, 2])  # chunk-1 audio -> blocked
    assert _blocked(mask[0, 0, 10, 3])
    assert _blocked(mask[0, 0, 10, 4])
    assert _allowed(mask[0, 0, 10, 7])  # chunk-2 audio -> allowed (own chunk)
    assert _allowed(mask[0, 0, 10, 8])
    assert _allowed(mask[0, 0, 10, 9])
    assert _allowed(mask[0, 0, 10, 0])  # system text -> allowed
    assert _allowed(mask[0, 0, 10, 5])  # text turn 1 -> allowed
    assert _allowed(mask[0, 0, 10, 10])  # self -> allowed
    assert _blocked(mask[0, 0, 10, 11])  # future text -> causal block

    # Chunk-1 text (index 5) may attend to chunk-1 audio (its own chunk).
    assert _allowed(mask[0, 0, 5, 2])
    assert _allowed(mask[0, 0, 5, 0])


def test_audio_query_keeps_full_causal_attention():
    input_tokens = torch.tensor([SEQ], dtype=torch.long)
    mask = build_training_chunk_restricted_mask(input_tokens, PAD, torch.float32)

    # Audio queries are NOT restricted: chunk-2 audio may still attend to chunk-1 audio.
    assert _allowed(mask[0, 0, 7, 2])  # chunk-2 audio -> chunk-1 audio allowed
    assert _allowed(mask[0, 0, 9, 4])
    assert _allowed(mask[0, 0, 7, 5])  # -> text turn 1 allowed
    assert _allowed(mask[0, 0, 7, 7])  # self
    assert _blocked(mask[0, 0, 7, 8])  # future -> causal block
    assert _blocked(mask[0, 0, 7, 12])  # padding blocked


def test_padding_key_blocked_everywhere():
    input_tokens = torch.tensor([SEQ], dtype=torch.long)
    mask = build_training_chunk_restricted_mask(input_tokens, PAD, torch.float32)
    for query in range(len(SEQ)):
        assert _blocked(mask[0, 0, query, 12])


def test_inference_incremental_mask_matches_training():
    """The incremental (key-cache) construction must equal the full-sequence training mask.

    Emulates the chunk-2 prefill: keys span positions 0..9 (system + chunk1 +
    text1 + chunk2), queries are the chunk-2 audio turn (positions 7..9). The
    resulting allowed/blocked pattern must match the corresponding rows of the
    full training mask.
    """
    input_tokens = torch.tensor([SEQ], dtype=torch.long)
    train_mask = build_training_chunk_restricted_mask(input_tokens, PAD, torch.float32)

    is_audio, chunk_id = audio_positions_and_chunk_ids(input_tokens)
    key_valid = input_tokens != PAD

    kv_end = 12  # keys 0..11 (system + both chunks + both text turns; pad excluded)
    q_start, q_end = 7, 12  # chunk-2 audio + chunk-2 transcription text queries
    abs_pos = torch.arange(len(SEQ)).unsqueeze(0)

    incremental = build_chunk_restricted_mask(
        key_is_audio=is_audio[:, :kv_end],
        key_chunk_id=chunk_id[:, :kv_end],
        key_valid=key_valid[:, :kv_end],
        query_is_audio=is_audio[:, q_start:q_end],
        query_chunk_id=chunk_id[:, q_start:q_end],
        query_abs_pos=abs_pos[:, q_start:q_end],
        key_abs_pos=abs_pos[:, :kv_end],
        dtype=torch.float32,
    )

    expected = train_mask[:, :, q_start:q_end, :kv_end]
    assert torch.equal(incremental, expected)
