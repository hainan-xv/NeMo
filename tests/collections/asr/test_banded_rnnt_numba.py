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

"""The CUDA banded RNN-T loss must agree with the reference implementation.

The reference in ``losses/banded_rnnt.py`` is the ground truth: it is slow but
it is what every CHAT result to date was trained with. A kernel that is fast and
subtly wrong would silently change the objective, so these tests compare the
VALUE and the GRADIENT, not just that it runs.

The band-geometry test needs no GPU and is the one most worth having: the caller
evaluates the joint at ``(b, t, u)`` triples and the returned ``[N, V+1]`` is
positional, so any disagreement in node ORDER silently pairs log-probabilities
with the wrong lattice nodes -- which trains, and trains on nonsense.
"""

import numpy as np
import pytest
import torch

from nemo.collections.asr.losses.banded_rnnt import BandedLattice, banded_rnnt_loss, build_lattices
from nemo.collections.asr.parts.numba.banded_rnnt.banded_rnnt_numba import (
    build_band_index,
    banded_rnnt_loss_cuda,
    kernel_is_usable,
)


# Labels must be < VOCAB and must not collide with the blank, or the reference's
# own `log_probs.gather` indexes out of bounds -- which surfaces as a CUDA
# device assert attributed to whichever kernel happens to run next, not to the
# bad index.
VOCAB = 12
BLANK = VOCAB - 1


def _random_batch(rng, batch=3, max_chunks=6, max_tok=4):
    """Per-utterance chunk token-id lists, the shape ``_chunk_tokens`` returns."""
    out = []
    for _ in range(batch):
        n_chunks = int(rng.integers(2, max_chunks + 1))
        out.append(
            [[int(rng.integers(0, BLANK)) for _ in range(int(rng.integers(0, max_tok + 1)))] for _ in range(n_chunks)]
        )
    return out


@pytest.mark.unit
@pytest.mark.parametrize("band_side", ["both", "later", "earlier"])
@pytest.mark.parametrize("band", [0, 1, 2])
def test_band_index_matches_reference_node_order(band, band_side):
    """build_band_index must enumerate exactly the reference's nodes, in order."""
    rng = np.random.default_rng(0)
    for _ in range(8):
        chunks = _random_batch(rng)
        per_utt, num_chunks, target_lens = build_lattices(chunks, band, band_side)
        ref = BandedLattice(per_utt, num_chunks, target_lens)
        got = build_band_index(chunks, band, band_side)

        assert got.num_nodes == ref.num_nodes
        np.testing.assert_array_equal(got.b_idx, np.asarray(ref.b_idx))
        np.testing.assert_array_equal(got.t_idx, np.asarray(ref.t_idx))
        np.testing.assert_array_equal(got.u_idx, np.asarray(ref.u_idx))
        np.testing.assert_array_equal(got.target_lens, np.asarray(target_lens))


@pytest.mark.unit
def test_band_bounds_are_monotone():
    """The kernels' bounds check assumes the band is a true band-diagonal."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        chunks = _random_batch(rng)
        idx = build_band_index(chunks, band=1, band_side="both")
        for b in range(len(idx.num_chunks)):
            T = int(idx.num_chunks[b])
            lo, hi = idx.band_lo[b, :T], idx.band_hi[b, :T]
            assert np.all(np.diff(lo) >= 0), "band_lo must be non-decreasing in t"
            assert np.all(np.diff(hi) >= 0), "band_hi must be non-decreasing in t"
            assert np.all(hi >= lo)


def _make_inputs(chunks, band, band_side, device, vocab=VOCAB, seed=0):
    per_utt, num_chunks, target_lens = build_lattices(chunks, band, band_side)
    ref_lat = BandedLattice(per_utt, num_chunks, target_lens)
    band_idx = build_band_index(chunks, band, band_side)

    B = len(chunks)
    u_max = max(max(target_lens), 1)
    targets = torch.zeros((B, u_max), dtype=torch.long, device=device)
    for b, cs in enumerate(chunks):
        flat = [tok for c in cs for tok in c]
        if flat:
            targets[b, : len(flat)] = torch.tensor(flat, dtype=torch.long, device=device)

    g = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(ref_lat.num_nodes, vocab, generator=g).to(device)
    return ref_lat, band_idx, targets, logits


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels require a GPU")
@pytest.mark.parametrize("band_side", ["both", "later"])
@pytest.mark.parametrize("band", [0, 1, 2])
def test_cuda_loss_matches_reference(band, band_side):
    device = torch.device("cuda")
    rng = np.random.default_rng(3)
    blank = BLANK
    for _ in range(4):
        chunks = _random_batch(rng)
        ref_lat, band_idx, targets, logits = _make_inputs(chunks, band, band_side, device)
        ok, why = kernel_is_usable(band_idx, device)
        if not ok:
            pytest.skip(why)

        lp_ref = logits.clone().requires_grad_(True).log_softmax(-1)
        nll_ref = banded_rnnt_loss(lp_ref, ref_lat, targets, blank)

        lp_k = logits.clone().requires_grad_(True).log_softmax(-1)
        nll_k = banded_rnnt_loss_cuda(lp_k, band_idx, targets, blank)

        torch.testing.assert_close(nll_k, nll_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels require a GPU")
@pytest.mark.parametrize("band_side", ["both", "later"])
@pytest.mark.parametrize("band", [0, 1])
def test_cuda_gradient_matches_reference(band, band_side):
    """The analytic gradient must match what autograd produces on the reference."""
    device = torch.device("cuda")
    rng = np.random.default_rng(11)
    blank = BLANK
    for _ in range(4):
        chunks = _random_batch(rng)
        ref_lat, band_idx, targets, logits = _make_inputs(chunks, band, band_side, device)
        ok, why = kernel_is_usable(band_idx, device)
        if not ok:
            pytest.skip(why)

        a = logits.clone().requires_grad_(True)
        banded_rnnt_loss(a.log_softmax(-1), ref_lat, targets, blank).sum().backward()

        b = logits.clone().requires_grad_(True)
        banded_rnnt_loss_cuda(b.log_softmax(-1), band_idx, targets, blank).sum().backward()

        torch.testing.assert_close(b.grad, a.grad, rtol=1e-3, atol=1e-4)
