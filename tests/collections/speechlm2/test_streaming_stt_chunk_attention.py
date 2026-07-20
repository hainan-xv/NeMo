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


# ---------------------------------------------------------------------------
# Strict variant: restrict_audio_cross_chunk (restrict_audio_queries=True).
# NO query (text OR audio) may attend to another chunk's audio. Models
# p(text_k | text_<k, audio_k).
# ---------------------------------------------------------------------------


def test_strict_audio_query_blocks_previous_chunk_audio():
    input_tokens = torch.tensor([SEQ], dtype=torch.long)
    mask = build_training_chunk_restricted_mask(input_tokens, PAD, torch.float32, restrict_audio_queries=True)
    assert mask.shape == (1, 1, len(SEQ), len(SEQ))

    # Query = chunk-2 audio (index 7): cross-chunk audio is now blocked too.
    assert _blocked(mask[0, 0, 7, 2])  # chunk-1 audio -> BLOCKED (was allowed in non-strict)
    assert _blocked(mask[0, 0, 9, 4])  # chunk-1 audio -> BLOCKED
    # But audio may still attend to all prior text and its own chunk's audio.
    assert _allowed(mask[0, 0, 7, 0])  # system text -> allowed
    assert _allowed(mask[0, 0, 7, 5])  # text turn 1 -> allowed
    assert _allowed(mask[0, 0, 7, 7])  # self -> allowed
    assert _allowed(mask[0, 0, 9, 7])  # same-chunk earlier audio frame -> allowed
    assert _blocked(mask[0, 0, 7, 8])  # future -> causal block


def test_strict_text_query_matches_nonstrict():
    """Text queries are restricted identically in both variants (audio queries differ)."""
    input_tokens = torch.tensor([SEQ], dtype=torch.long)
    strict = build_training_chunk_restricted_mask(input_tokens, PAD, torch.float32, restrict_audio_queries=True)

    # Chunk-2 transcription text (index 10): own chunk audio only, same as non-strict.
    assert _blocked(strict[0, 0, 10, 2])  # chunk-1 audio blocked
    assert _allowed(strict[0, 0, 10, 7])  # chunk-2 audio (own) allowed
    assert _allowed(strict[0, 0, 10, 0])  # system text allowed
    assert _allowed(strict[0, 0, 10, 5])  # text turn 1 allowed
    assert _allowed(strict[0, 0, 10, 10])  # self allowed
    assert _blocked(strict[0, 0, 10, 11])  # future text causal block


def test_strict_only_differs_on_audio_query_cross_chunk_audio():
    """Strict and non-strict masks differ EXACTLY at audio-query x cross-chunk-audio-key cells."""
    input_tokens = torch.tensor([SEQ], dtype=torch.long)
    nonstrict = build_training_chunk_restricted_mask(input_tokens, PAD, torch.float32)
    strict = build_training_chunk_restricted_mask(input_tokens, PAD, torch.float32, restrict_audio_queries=True)

    is_audio, chunk_id = audio_positions_and_chunk_ids(input_tokens)
    diff = ~torch.isclose(nonstrict, strict)  # (1, 1, L, L)
    for q in range(len(SEQ)):
        for k in range(len(SEQ)):
            if bool(diff[0, 0, q, k]):
                # Only audio-query -> different-chunk audio-key cells may differ,
                # and only when causal (q >= k).
                assert bool(is_audio[0, q]) and bool(is_audio[0, k])
                assert int(chunk_id[0, q]) != int(chunk_id[0, k])
                assert q >= k
                # Non-strict allowed it; strict blocks it.
                assert _allowed(nonstrict[0, 0, q, k]) and _blocked(strict[0, 0, q, k])
    # There is at least one such differing cell in this layout (chunk2 audio -> chunk1 audio).
    assert bool(diff.any())


def test_strict_padding_key_blocked_everywhere():
    input_tokens = torch.tensor([SEQ], dtype=torch.long)
    mask = build_training_chunk_restricted_mask(input_tokens, PAD, torch.float32, restrict_audio_queries=True)
    for query in range(len(SEQ)):
        assert _blocked(mask[0, 0, query, 12])


def test_strict_inference_incremental_mask_matches_training():
    """Incremental (key-cache) construction must equal the full-sequence training mask (strict)."""
    input_tokens = torch.tensor([SEQ], dtype=torch.long)
    train_mask = build_training_chunk_restricted_mask(input_tokens, PAD, torch.float32, restrict_audio_queries=True)

    is_audio, chunk_id = audio_positions_and_chunk_ids(input_tokens)
    key_valid = input_tokens != PAD

    kv_end = 12
    q_start, q_end = 7, 12
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
        restrict_audio_queries=True,
    )

    expected = train_mask[:, :, q_start:q_end, :kv_end]
    assert torch.equal(incremental, expected)
