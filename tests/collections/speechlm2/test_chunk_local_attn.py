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

"""Unit tests for chunk-local audio attention helpers.

Run with ``pytest -s`` to see the printed mask/position-id visualizations.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from nemo.collections.speechlm2.parts.chunk_local_attn import (
    build_chunk_ids,
    build_chunk_local_attention_bias,
    build_chunk_local_inference_bias,
    build_chunk_local_position_ids,
)


# ---------------------------------------------------------------------------
# Synthetic-batch builder
# ---------------------------------------------------------------------------


# Token-kind glyphs used purely for visualization.
SYS = "s"
USER_HDR = "u"
AUDIO = "A"
ASST_HDR = "h"
CONTENT = "c"
IM_END = "E"
PAD = "."


def _build_sample(sys_len: int, chunks: List[dict]) -> List[str]:
    """Build a list of per-token glyphs for one sequence.

    ``chunks`` is a list of dicts each with keys ``user_hdr``, ``audio``,
    ``asst_hdr``, ``content``, ``im_end`` giving the number of tokens of
    each kind. This mirrors the actual training-time layout per chunk.
    """
    seq: List[str] = [SYS] * sys_len
    for c in chunks:
        seq += [USER_HDR] * c.get("user_hdr", 2)
        seq += [AUDIO] * c.get("audio", 3)
        seq += [ASST_HDR] * c.get("asst_hdr", 2)
        seq += [CONTENT] * c.get("content", 2)
        seq += [IM_END] * c.get("im_end", 1)
    return seq


def _pad_to(seq: List[str], length: int) -> List[str]:
    if len(seq) > length:
        raise ValueError(f"Sequence longer than pad length: {len(seq)} > {length}")
    return seq + [PAD] * (length - len(seq))


def _make_batch(samples: List[List[str]], device: str = "cpu"):
    """Pack a list of per-sample glyph sequences into batched tensors.

    Returns ``(is_audio, attention_mask, glyphs)``:

    * ``is_audio``: ``(B, L)`` bool.
    * ``attention_mask``: ``(B, L)`` long, ``1`` for valid tokens, ``0`` for pad.
    * ``glyphs``: ``(B, L)`` Python lists, kept for pretty-printing.
    """
    L = max(len(s) for s in samples)
    padded = [_pad_to(s, L) for s in samples]
    is_audio = torch.tensor(
        [[g == AUDIO for g in row] for row in padded],
        dtype=torch.bool,
        device=device,
    )
    attention_mask = torch.tensor(
        [[g != PAD for g in row] for row in padded],
        dtype=torch.long,
        device=device,
    )
    return is_audio, attention_mask, padded


# ---------------------------------------------------------------------------
# Pretty-printers
# ---------------------------------------------------------------------------


def _print_row_summary(
    name: str,
    glyphs: List[str],
    chunk_id: torch.Tensor,
    position_ids: torch.Tensor,
) -> None:
    print(f"\n  {name}")
    print(f"    idx    : {' '.join(f'{i:>2d}' for i in range(len(glyphs)))}")
    print(f"    token  : {' '.join(f'{g:>2s}' for g in glyphs)}")
    print(f"    chunk  : {' '.join(f'{int(x):>2d}' for x in chunk_id.tolist())}")
    print(f"    pos_id : {' '.join(f'{int(x):>2d}' for x in position_ids.tolist())}")


def _print_mask(
    name: str,
    glyphs: List[str],
    allowed_2d: torch.Tensor,
) -> None:
    """Print the (L, L) attention mask as a human-readable grid.

    Columns are keys, rows are queries. ``.`` = allowed, ``X`` = blocked.
    """
    L = allowed_2d.shape[0]
    print(f"\n  {name} (rows=queries, cols=keys; '.'=allowed, 'X'=blocked)")
    header_idx = "         " + " ".join(f"{i:>2d}" for i in range(L))
    header_tok = "         " + " ".join(f"{g:>2s}" for g in glyphs)
    print(header_idx)
    print(header_tok)
    for i in range(L):
        cells = []
        for j in range(L):
            cells.append(" ." if bool(allowed_2d[i, j]) else " X")
        print(f"  q{i:>2d} {glyphs[i]:>2s} {''.join(cells)}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chunk_ids_and_positions_single_chunk():
    """One row, one chunk: chunk-ids, audio counter, text counter all tidy."""
    glyphs = _build_sample(
        sys_len=3,
        chunks=[dict(user_hdr=2, audio=3, asst_hdr=2, content=2, im_end=1)],
    )
    is_audio, attn_mask, batch_glyphs = _make_batch([glyphs])
    chunk_id = build_chunk_ids(is_audio, attn_mask)
    pos = build_chunk_local_position_ids(is_audio, attn_mask)

    print("\n[test_chunk_ids_and_positions_single_chunk]")
    _print_row_summary("row 0", batch_glyphs[0], chunk_id[0], pos[0])

    expected_chunk = torch.tensor([[-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0]])
    # text counter: sys 0..2, u_hdr 3..4, h_hdr 5..6, content 7..8, im_end 9
    # audio counter: 0..2 on the 3 audio frames
    expected_pos = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 5, 6, 7, 8, 9]])
    assert torch.equal(chunk_id, expected_chunk)
    assert torch.equal(pos, expected_pos)


def test_positions_audio_counter_contiguous_across_chunks():
    """Audio counter must be globally contiguous: 12-frame x 2 chunks → 0..23."""
    glyphs = _build_sample(
        sys_len=2,
        chunks=[
            dict(user_hdr=1, audio=12, asst_hdr=1, content=2, im_end=1),
            dict(user_hdr=1, audio=12, asst_hdr=1, content=2, im_end=1),
        ],
    )
    is_audio, attn_mask, batch_glyphs = _make_batch([glyphs])
    pos = build_chunk_local_position_ids(is_audio, attn_mask)

    audio_positions = pos[0][is_audio[0]]
    print("\n[test_positions_audio_counter_contiguous_across_chunks]")
    print(f"  audio_positions = {audio_positions.tolist()}")
    assert torch.equal(audio_positions, torch.arange(24))

    text_positions = pos[0][(~is_audio[0]) & attn_mask[0].bool()]
    print(f"  text_positions  = {text_positions.tolist()}")
    assert torch.equal(text_positions, torch.arange(text_positions.numel()))


def test_mask_blocks_cross_chunk_audio_only_default_n1():
    """N=1 (default): chunk-1 tokens must not see chunk-0 audio, and vice versa."""
    glyphs = _build_sample(
        sys_len=3,
        chunks=[
            dict(user_hdr=2, audio=3, asst_hdr=2, content=2, im_end=1),
            dict(user_hdr=2, audio=3, asst_hdr=2, content=2, im_end=1),
        ],
    )
    is_audio, attn_mask, batch_glyphs = _make_batch([glyphs])
    chunk_id = build_chunk_ids(is_audio, attn_mask)
    pos = build_chunk_local_position_ids(is_audio, attn_mask)
    bias = build_chunk_local_attention_bias(
        chunk_id, is_audio, attn_mask, dtype=torch.float32
    )
    allowed = bias[0, 0] == 0  # (L, L) bool

    print("\n[test_mask_blocks_cross_chunk_audio_only_default_n1]")
    _print_row_summary("row 0", batch_glyphs[0], chunk_id[0], pos[0])
    _print_mask("attention mask", batch_glyphs[0], allowed)

    L = len(glyphs)
    audio = is_audio[0]
    cid = chunk_id[0]

    # Causal lower-triangular property holds.
    for i in range(L):
        for j in range(i + 1, L):
            assert not bool(allowed[i, j]), f"non-causal allowed at ({i},{j})"

    # Cross-chunk audio is always blocked; same-chunk causal audio is allowed.
    for i in range(L):
        for j in range(i + 1):
            if audio[j] and cid[j].item() != cid[i].item():
                assert not bool(allowed[i, j]), (
                    f"cross-chunk audio incorrectly allowed: q={i} k={j}"
                )

    # Non-audio causal keys are always allowed.
    for i in range(L):
        for j in range(i + 1):
            if not audio[j]:
                assert bool(allowed[i, j]), (
                    f"non-audio causal key incorrectly blocked: q={i} k={j}"
                )


def test_mask_with_two_visible_audio_chunks():
    """N=2: each query sees its own chunk's audio + the previous chunk's audio."""
    glyphs = _build_sample(
        sys_len=2,
        chunks=[
            dict(user_hdr=1, audio=2, asst_hdr=1, content=1, im_end=1),
            dict(user_hdr=1, audio=2, asst_hdr=1, content=1, im_end=1),
            dict(user_hdr=1, audio=2, asst_hdr=1, content=1, im_end=1),
        ],
    )
    is_audio, attn_mask, batch_glyphs = _make_batch([glyphs])
    chunk_id = build_chunk_ids(is_audio, attn_mask)
    pos = build_chunk_local_position_ids(is_audio, attn_mask)
    bias_n2 = build_chunk_local_attention_bias(
        chunk_id, is_audio, attn_mask, dtype=torch.float32, num_visible_audio_chunks=2
    )
    allowed = bias_n2[0, 0] == 0

    print("\n[test_mask_with_two_visible_audio_chunks]")
    _print_row_summary("row 0", batch_glyphs[0], chunk_id[0], pos[0])
    _print_mask("attention mask (N=2)", batch_glyphs[0], allowed)

    L = len(glyphs)
    audio = is_audio[0]
    cid = chunk_id[0]

    # Rule: audio key j is visible to query i iff cid[i] - cid[j] < 2 (and causal).
    for i in range(L):
        for j in range(i + 1):
            if audio[j]:
                delta = cid[i].item() - cid[j].item()
                if delta < 2:
                    assert bool(allowed[i, j]), (
                        f"N=2 audio key incorrectly blocked: q={i} k={j} delta={delta}"
                    )
                else:
                    assert not bool(allowed[i, j]), (
                        f"N=2 audio key incorrectly allowed: q={i} k={j} delta={delta}"
                    )

    # Non-audio causal keys remain always allowed.
    for i in range(L):
        for j in range(i + 1):
            if not audio[j]:
                assert bool(allowed[i, j]), (
                    f"non-audio causal key incorrectly blocked: q={i} k={j}"
                )


def test_padded_batch_two_rows_different_lengths():
    """Padded right-end positions must be masked as keys and ignored elsewhere."""
    row0 = _build_sample(
        sys_len=2,
        chunks=[
            dict(user_hdr=1, audio=2, asst_hdr=1, content=2, im_end=1),
            dict(user_hdr=1, audio=2, asst_hdr=1, content=1, im_end=1),
        ],
    )
    row1 = _build_sample(
        sys_len=2,
        chunks=[
            dict(user_hdr=1, audio=2, asst_hdr=1, content=2, im_end=1),
        ],
    )
    is_audio, attn_mask, batch_glyphs = _make_batch([row0, row1])
    chunk_id = build_chunk_ids(is_audio, attn_mask)
    pos = build_chunk_local_position_ids(is_audio, attn_mask)
    bias = build_chunk_local_attention_bias(
        chunk_id, is_audio, attn_mask, dtype=torch.float32
    )

    print("\n[test_padded_batch_two_rows_different_lengths]")
    for b in (0, 1):
        _print_row_summary(f"row {b}", batch_glyphs[b], chunk_id[b], pos[b])
        allowed = bias[b, 0] == 0
        _print_mask(f"row {b} attention mask", batch_glyphs[b], allowed)

    # Pad positions are masked as keys for every query in their row.
    L = is_audio.shape[1]
    for b in range(2):
        is_valid = attn_mask[b].bool()
        for j in range(L):
            if not is_valid[j]:
                for i in range(L):
                    assert bias[b, 0, i, j] != 0, (
                        f"row {b}: pad key j={j} not masked for query i={i}"
                    )

    # Position_ids are non-negative everywhere (no out-of-range LLM indexing).
    assert (pos >= 0).all()

    # Audio and text counters are independent and contiguous on each row.
    for b in (0, 1):
        audio_mask_b = is_audio[b] & attn_mask[b].bool()
        text_mask_b = (~is_audio[b]) & attn_mask[b].bool()
        ap = pos[b][audio_mask_b]
        tp = pos[b][text_mask_b]
        assert torch.equal(ap, torch.arange(ap.numel(), device=ap.device))
        assert torch.equal(tp, torch.arange(tp.numel(), device=tp.device))


def test_sys_only_no_chunks():
    """No audio at all → every token is sys-prompt-like; chunk_id stays at -1."""
    glyphs = [SYS] * 4
    is_audio, attn_mask, batch_glyphs = _make_batch([glyphs])
    chunk_id = build_chunk_ids(is_audio, attn_mask)
    pos = build_chunk_local_position_ids(is_audio, attn_mask)
    bias = build_chunk_local_attention_bias(
        chunk_id, is_audio, attn_mask, dtype=torch.float32
    )
    allowed = bias[0, 0] == 0

    print("\n[test_sys_only_no_chunks]")
    _print_row_summary("row 0", batch_glyphs[0], chunk_id[0], pos[0])
    _print_mask("attention mask", batch_glyphs[0], allowed)

    assert torch.equal(chunk_id, torch.tensor([[-1, -1, -1, -1]]))
    assert torch.equal(pos, torch.tensor([[0, 1, 2, 3]]))


def test_inference_bias_matches_training_when_history_empty():
    """With L_past=0, inference helper must match the training helper exactly."""
    glyphs = _build_sample(
        sys_len=2,
        chunks=[
            dict(user_hdr=1, audio=2, asst_hdr=1, content=1, im_end=1),
            dict(user_hdr=1, audio=2, asst_hdr=1, content=1, im_end=1),
            dict(user_hdr=1, audio=2, asst_hdr=1, content=1, im_end=1),
        ],
    )
    is_audio, attn_mask, _ = _make_batch([glyphs])
    chunk_id = build_chunk_ids(is_audio, attn_mask)

    B, L = is_audio.shape
    empty_chunk_id = torch.empty((B, 0), dtype=torch.long)
    empty_is_audio = torch.empty((B, 0), dtype=torch.bool)
    empty_attn_mask = torch.empty((B, 0), dtype=attn_mask.dtype)

    for n in (1, 2, 3):
        train = build_chunk_local_attention_bias(
            chunk_id, is_audio, attn_mask, dtype=torch.float32, num_visible_audio_chunks=n
        )
        infer = build_chunk_local_inference_bias(
            empty_chunk_id,
            empty_is_audio,
            empty_attn_mask,
            chunk_id,
            is_audio,
            dtype=torch.float32,
            num_visible_audio_chunks=n,
        )
        assert train.shape == infer.shape, f"N={n} shape mismatch: {train.shape} vs {infer.shape}"
        assert torch.equal(train, infer), f"N={n}: training/inference bias differ"


def test_inference_bias_with_history_splits_equivalently():
    """Splitting (history + new) at any boundary must yield the same bias."""
    glyphs = _build_sample(
        sys_len=2,
        chunks=[
            dict(user_hdr=1, audio=2, asst_hdr=1, content=1, im_end=1),
            dict(user_hdr=1, audio=2, asst_hdr=1, content=1, im_end=1),
            dict(user_hdr=1, audio=2, asst_hdr=1, content=1, im_end=1),
        ],
    )
    is_audio, attn_mask, _ = _make_batch([glyphs])
    chunk_id = build_chunk_ids(is_audio, attn_mask)
    L = is_audio.shape[1]

    full = build_chunk_local_attention_bias(
        chunk_id, is_audio, attn_mask, dtype=torch.float32, num_visible_audio_chunks=2
    )  # (B, 1, L, L)

    for split in (1, 4, 7, L - 1):
        history_cid = chunk_id[:, :split]
        history_audio = is_audio[:, :split]
        history_mask = attn_mask[:, :split]
        new_cid = chunk_id[:, split:]
        new_audio = is_audio[:, split:]
        infer = build_chunk_local_inference_bias(
            history_cid,
            history_audio,
            history_mask,
            new_cid,
            new_audio,
            dtype=torch.float32,
            num_visible_audio_chunks=2,
        )  # (B, 1, L - split, L)
        # The lower-right block of `full` for query rows [split:] must match.
        assert torch.equal(full[:, :, split:, :], infer), f"split={split}: mismatch"


def test_inference_bias_validation():
    with pytest.raises(ValueError):
        build_chunk_local_inference_bias(
            chunk_id_history=torch.empty((1, 0), dtype=torch.long),
            is_audio_history=torch.empty((1, 0), dtype=torch.bool),
            attention_mask_history=torch.empty((1, 0), dtype=torch.long),
            chunk_id_new=torch.zeros((1, 2), dtype=torch.long),
            is_audio_new=torch.zeros((1, 2), dtype=torch.bool),
            dtype=torch.float32,
            num_visible_audio_chunks=0,
        )


def test_num_visible_audio_chunks_validation():
    """N < 1 should raise."""
    is_audio = torch.zeros((1, 4), dtype=torch.bool)
    attn_mask = torch.ones((1, 4), dtype=torch.long)
    chunk_id = build_chunk_ids(is_audio, attn_mask)
    with pytest.raises(ValueError):
        build_chunk_local_attention_bias(
            chunk_id, is_audio, attn_mask, dtype=torch.float32, num_visible_audio_chunks=0
        )


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_helpers_run_on_device(device):
    """Same shapes/values across CPU and CUDA; sanity check that nothing CPU-loops."""
    glyphs = _build_sample(
        sys_len=2,
        chunks=[
            dict(user_hdr=1, audio=2, asst_hdr=1, content=1, im_end=1),
            dict(user_hdr=1, audio=2, asst_hdr=1, content=1, im_end=1),
        ],
    )
    is_audio, attn_mask, _ = _make_batch([glyphs], device=device)
    chunk_id = build_chunk_ids(is_audio, attn_mask)
    pos = build_chunk_local_position_ids(is_audio, attn_mask)
    bias = build_chunk_local_attention_bias(
        chunk_id, is_audio, attn_mask, dtype=torch.float32, num_visible_audio_chunks=2
    )
    assert chunk_id.device.type == device
    assert pos.device.type == device
    assert bias.device.type == device
    assert bias.shape == (1, 1, is_audio.shape[1], is_audio.shape[1])
