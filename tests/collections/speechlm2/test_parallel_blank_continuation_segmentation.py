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
"""Blank-continuation chunk segmentation: reconstruction + invariants.

The blank-continuation parallel scheme splits a chunk's *content* tokens into
variable-length K-slot blocks. Partial / randomly-cut blocks end with a single
``<blank>`` "continue" marker, and the chunk closes with a terminator block
``[<|im_end|>, IGNORE...]``. No matter how the random cuts fall, stripping the
synthetic markers (blank + terminator) and concatenating the real tokens across
blocks MUST reconstruct the original content exactly — that is the core
correctness property these tests assert, alongside the structural invariants the
decoder relies on (im_end only in slot 0 of the terminator, blank only at
non-first slots, full blocks carry no marker, anchors track cumulative real
tokens).
"""

import random

import pytest

from nemo.collections.speechlm2.data.streaming_stt_dataset import (
    IGNORE_INDEX,
    StreamingSTTDataset,
)

# Sentinel ids that cannot collide with content tokens (content uses 0..999).
EOS_ID = 100_001
BLANK_ID = 100_002

segment = StreamingSTTDataset._segment_chunk_blocks_blank_continuation


def _real_tokens(targets, K):
    """Strip terminator + blank markers; return the flat list of real tokens."""
    out = []
    for block_idx, block in enumerate(targets):
        is_terminator = block_idx == len(targets) - 1
        for slot, tid in enumerate(block):
            if tid == IGNORE_INDEX:
                continue
            if is_terminator:
                # Terminator block: slot 0 is eos, everything else IGNORE.
                continue
            if tid == BLANK_ID or tid == EOS_ID:
                continue
            out.append(tid)
    return out


def _assert_invariants(anchors, targets, anchor_base, content, K):
    N = len(content)
    assert len(anchors) == len(targets)
    assert len(targets) >= 1, "every chunk must have at least the terminator block"

    # Terminator block: last block, slot 0 == eos, rest IGNORE.
    term = targets[-1]
    assert len(term) == K
    assert term[0] == EOS_ID
    assert all(t == IGNORE_INDEX for t in term[1:])
    assert anchors[-1] == anchor_base + N

    cumulative_real = 0
    for block_idx, (anchor, block) in enumerate(zip(anchors, targets)):
        assert len(block) == K, "every block is padded to K slots"
        is_terminator = block_idx == len(targets) - 1
        if is_terminator:
            continue

        # <|im_end|> may NEVER appear in a content block.
        assert EOS_ID not in block, f"im_end leaked into content block {block_idx}: {block}"

        # Count the real tokens and find the (optional) blank marker.
        real = [t for t in block if t not in (IGNORE_INDEX, BLANK_ID)]
        n_real = len(real)
        assert n_real >= 1, f"content block {block_idx} must hold >=1 real token: {block}"

        blank_slots = [s for s, t in enumerate(block) if t == BLANK_ID]
        assert len(blank_slots) <= 1, f"at most one blank per block: {block}"
        if blank_slots:
            bs = blank_slots[0]
            assert bs >= 1, f"blank must never be in slot 0: {block}"
            # Real tokens occupy slots 0..bs-1; blank sits right after them.
            assert bs == n_real, f"blank must directly follow the real tokens: {block}"
        else:
            # No blank => a FULL block (fullness == continue).
            assert n_real == K, f"a block without a blank marker must be full (K): {block}"

        # Anchor tracks cumulative real tokens consumed so far.
        assert anchor == anchor_base + cumulative_real
        # First real target of this block is content[cumulative_real].
        assert real[0] == content[cumulative_real]
        cumulative_real += n_real

    # All content tokens are accounted for across the content blocks.
    assert cumulative_real == N


@pytest.mark.parametrize("K", [1, 2, 4, 8])
@pytest.mark.parametrize("N", [0, 1, 2, 3, 4, 5, 7, 16, 33])
def test_reconstruction_no_cuts(K, N):
    """Greedy packing (cut_prob=0): reconstruction + invariants."""
    content = list(range(N))  # distinct tokens 0..N-1
    anchors, targets = segment(
        anchor_base=7,
        n_content=N,
        content=content,
        K=K,
        eos_id=EOS_ID,
        blank_id=BLANK_ID,
        cut_prob=0.0,
        rng=None,
    )
    assert _real_tokens(targets, K) == content
    _assert_invariants(anchors, targets, anchor_base=7, content=content, K=K)


@pytest.mark.parametrize("K", [2, 3, 4, 8])
@pytest.mark.parametrize("cut_prob", [0.2, 0.5, 1.0])
@pytest.mark.parametrize("seed", list(range(25)))
def test_reconstruction_random_cuts(K, cut_prob, seed):
    """No matter how the random cuts fall, reconstruction must be exact."""
    rng = random.Random(seed)
    N = rng.randint(0, 40)
    # Use distinct tokens so reconstruction is unambiguous.
    content = [rng.randint(0, 999) for _ in range(N)]
    anchors, targets = segment(
        anchor_base=3,
        n_content=N,
        content=content,
        K=K,
        eos_id=EOS_ID,
        blank_id=BLANK_ID,
        cut_prob=cut_prob,
        rng=rng,
    )
    assert _real_tokens(targets, K) == content
    _assert_invariants(anchors, targets, anchor_base=3, content=content, K=K)


def test_empty_chunk_is_terminator_only():
    """An empty chunk produces exactly one terminator block at the write_id."""
    anchors, targets = segment(
        anchor_base=42,
        n_content=0,
        content=[],
        K=4,
        eos_id=EOS_ID,
        blank_id=BLANK_ID,
        cut_prob=1.0,  # cut prob is irrelevant with no content
        rng=random.Random(0),
    )
    assert len(targets) == 1
    assert anchors == [42]
    assert targets[0] == [EOS_ID, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]


def test_cut_prob_one_splits_maximally():
    """With cut_prob=1 every multi-token block cuts, so blocks shrink to the
    cut point. Reconstruction must still hold and every content block carries a
    blank (no full blocks survive when always cutting)."""
    K = 4
    N = 12
    content = list(range(N))
    rng = random.Random(123)
    anchors, targets = segment(
        anchor_base=0,
        n_content=N,
        content=content,
        K=K,
        eos_id=EOS_ID,
        blank_id=BLANK_ID,
        cut_prob=1.0,
        rng=rng,
    )
    assert _real_tokens(targets, K) == content
    _assert_invariants(anchors, targets, anchor_base=0, content=content, K=K)
    # Every content block should end with a blank when always cutting (a 1-token
    # tail block can't be cut but still gets a partial-block blank marker).
    for block in targets[:-1]:
        assert BLANK_ID in block


def test_full_blocks_have_no_marker():
    """N a multiple of K with no cuts → all full blocks, none carry a blank;
    the chunk still closes with a slot-0 im_end terminator."""
    K = 4
    content = list(range(8))
    anchors, targets = segment(
        anchor_base=0,
        n_content=8,
        content=content,
        K=K,
        eos_id=EOS_ID,
        blank_id=BLANK_ID,
        cut_prob=0.0,
        rng=None,
    )
    # 2 full content blocks + terminator.
    assert len(targets) == 3
    assert targets[0] == [0, 1, 2, 3]
    assert targets[1] == [4, 5, 6, 7]
    assert targets[2] == [EOS_ID, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
    assert anchors == [0, 4, 8]
