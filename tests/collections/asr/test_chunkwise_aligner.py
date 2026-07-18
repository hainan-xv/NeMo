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

"""Loss-level tests for the Chunkwise-Aligner single fixed-path loss.

This is the pure-loss subset of the original test suite -- it exercises only
``nemo/collections/asr/losses/chunked_aligner_pytorch.py`` (the single-path
``ChunkwiseAlignerLoss`` and its reference log-prob). The remaining tests from
the upstream file (frozen external CTC / word aligners, chunked-aligner decoding
and end-to-end ``EncDecRNNTModel`` wiring) depend on modules that are ported in
the model-wiring step and are intentionally not included here yet.
"""

import pytest
import torch

from nemo.collections.asr.losses.chunked_aligner_pytorch import (
    ChunkwiseAlignerLoss,
    chunked_aligner_loss_bruteforce,
    chunkwise_aligner_single_path_logprob,
)


def _counts_to_token_chunk_ids(counts_per_sample, U_max):
    """List[List[int]] per-chunk counts -> [B, U_max] token chunk ids (-1 pad)."""
    B = len(counts_per_sample)
    out = torch.full((B, U_max), -1, dtype=torch.long)
    for b, counts in enumerate(counts_per_sample):
        u = 0
        for c, k in enumerate(counts):
            for _ in range(k):
                out[b, u] = c
                u += 1
    return out


def _random_feasible_counts(T_b, U_b, C, rng):
    """Random per-chunk token counts with sum == U_b and counts[c] <= frames_in_chunk."""
    n_chunks = (T_b + C - 1) // C
    frames_here = [min(C, T_b - c * C) for c in range(n_chunks)]
    if sum(frames_here) < U_b:
        return None  # infeasible to host U_b tokens at all
    counts = [0] * n_chunks
    remaining = U_b
    # Greedy random fill respecting per-chunk capacity.
    order = list(range(n_chunks))
    rng.shuffle(order)
    for c in order:
        if remaining == 0:
            break
        cap = frames_here[c]
        take = rng.randint(0, min(cap, remaining))
        counts[c] = take
        remaining -= take
    # Place any leftover wherever capacity remains.
    c = 0
    while remaining > 0 and c < n_chunks:
        room = frames_here[c] - counts[c]
        add = min(room, remaining)
        counts[c] += add
        remaining -= add
        c += 1
    if remaining != 0:
        return None
    return counts


@pytest.mark.unit
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4])
def test_loss_matches_single_path_reference(chunk_size):
    """ChunkwiseAlignerLoss (reduction='none') == -reference single-path logprob."""
    import random

    rng = random.Random(1234 + chunk_size)
    torch.manual_seed(7 + chunk_size)

    B, T, V = 4, 10, 6
    blank = V - 1

    act_lens = torch.tensor([10, 8, 6, 5])[:B]
    label_lens = torch.tensor([3, 2, 2, 1])[:B]
    U_max = int(label_lens.max())

    acts = torch.randn(B, T, U_max + 1, V)
    labels = torch.randint(0, V - 1, (B, U_max))

    counts_per_sample = []
    n_chunks_max = (int(act_lens.max()) + chunk_size - 1) // chunk_size
    chunk_counts = torch.zeros(B, n_chunks_max, dtype=torch.long)
    for b in range(B):
        counts = _random_feasible_counts(int(act_lens[b]), int(label_lens[b]), chunk_size, rng)
        if counts is None:
            pytest.skip("randomly generated an infeasible segmentation")
        counts_per_sample.append(counts)
        for c, k in enumerate(counts):
            chunk_counts[b, c] = k

    token_chunk_ids = _counts_to_token_chunk_ids(counts_per_sample, U_max)

    ref = chunkwise_aligner_single_path_logprob(
        acts, labels, act_lens, label_lens, chunk_counts, blank=blank, chunk_size=chunk_size
    )

    loss = ChunkwiseAlignerLoss(blank=blank, chunk_size=chunk_size, reduction='none')
    per_sample = loss(acts, labels, act_lens, label_lens, token_chunk_ids)

    assert torch.allclose(per_sample, -ref, atol=1e-4), f"{per_sample} vs {-ref}"


@pytest.mark.unit
def test_single_path_is_a_term_of_full_sum():
    """The fixed-path logprob must be <= the full-sum logprob (it's one of its terms)."""
    torch.manual_seed(0)
    B, T, V = 2, 8, 5
    chunk_size = 3
    blank = V - 1
    act_lens = torch.tensor([8, 6])
    label_lens = torch.tensor([2, 2])
    U_max = 2
    acts = torch.randn(B, T, U_max + 1, V)
    labels = torch.randint(0, V - 1, (B, U_max))

    n_chunks_max = (int(act_lens.max()) + chunk_size - 1) // chunk_size
    # Assign both tokens to chunk 0 (a valid, simple segmentation).
    chunk_counts = torch.zeros(B, n_chunks_max, dtype=torch.long)
    chunk_counts[:, 0] = 2

    single = chunkwise_aligner_single_path_logprob(
        acts, labels, act_lens, label_lens, chunk_counts, blank=blank, chunk_size=chunk_size
    )
    full = chunked_aligner_loss_bruteforce(acts, labels, act_lens, label_lens, blank=blank, chunk_size=chunk_size)

    assert torch.all(single <= full + 1e-4)


@pytest.mark.unit
def test_infeasible_assignment_is_skipped():
    """An overflowing assignment (more tokens than frames in a chunk) is excluded."""
    torch.manual_seed(3)
    B, T, V = 2, 4, 5
    chunk_size = 1  # each chunk hosts at most 1 token
    blank = V - 1
    act_lens = torch.tensor([4, 4])
    label_lens = torch.tensor([2, 2])
    U_max = 2
    acts = torch.randn(B, T, U_max + 1, V)
    labels = torch.randint(0, V - 1, (B, U_max))

    # Sample 0: feasible (one token per chunk). Sample 1: both tokens in chunk 0 -> overflow.
    token_chunk_ids = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)

    loss = ChunkwiseAlignerLoss(blank=blank, chunk_size=chunk_size, reduction='none')
    per_sample = loss(acts, labels, act_lens, label_lens, token_chunk_ids)

    assert per_sample[1].item() == 0.0  # infeasible -> zero contribution
    assert per_sample[0].item() != 0.0  # feasible -> real loss

    # mean_volume must only divide by the valid sample's label count.
    mv = ChunkwiseAlignerLoss(blank=blank, chunk_size=chunk_size, reduction='mean_volume')
    val = mv(acts, labels, act_lens, label_lens, token_chunk_ids)
    expected = per_sample[0] / float(label_lens[0])
    assert torch.allclose(val, expected, atol=1e-5)


@pytest.mark.unit
def test_valid_mask_excludes_samples():
    torch.manual_seed(5)
    B, T, V = 2, 6, 5
    chunk_size = 2
    blank = V - 1
    act_lens = torch.tensor([6, 6])
    label_lens = torch.tensor([2, 2])
    acts = torch.randn(B, T, 3, V)
    labels = torch.randint(0, V - 1, (B, 2))
    token_chunk_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

    loss = ChunkwiseAlignerLoss(blank=blank, chunk_size=chunk_size, reduction='none')
    valid_mask = torch.tensor([True, False])
    per_sample = loss(acts, labels, act_lens, label_lens, token_chunk_ids, valid_mask=valid_mask)
    assert per_sample[1].item() == 0.0


@pytest.mark.unit
def test_loss_is_differentiable():
    torch.manual_seed(9)
    B, T, V = 2, 6, 5
    chunk_size = 2
    blank = V - 1
    act_lens = torch.tensor([6, 4])
    label_lens = torch.tensor([2, 1])
    acts = torch.randn(B, T, 3, V, requires_grad=True)
    labels = torch.randint(0, V - 1, (B, 2))
    token_chunk_ids = torch.tensor([[0, 1], [0, -1]], dtype=torch.long)

    loss = ChunkwiseAlignerLoss(blank=blank, chunk_size=chunk_size, reduction='mean_volume')
    val = loss(acts, labels, act_lens, label_lens, token_chunk_ids)
    val.backward()
    assert acts.grad is not None
    assert torch.isfinite(acts.grad).all()


if __name__ == "__main__":
    pytest.main([__file__])
