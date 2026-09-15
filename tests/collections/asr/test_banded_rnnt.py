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
"""The banded RNN-T loss, pinned at both ends of its range.

The band interpolates between two losses we already have, so it can be checked
against both rather than merely inspected:

    band = 0    exactly one path survives -> must equal the forced-alignment
                cross-entropy, to floating point
    band >= T   the whole lattice is inside the band -> must equal the full
                RNN-T loss, computed here by brute-force path enumeration

A loss that is wrong in between would have to be wrong while agreeing with both
endpoints, which is a much smaller space of bugs.
"""

import itertools

import pytest
import torch

from nemo.collections.asr.losses.banded_rnnt import BandedLattice, banded_rnnt_loss, build_lattices
from nemo.collections.asr.parts.utils.chat_alignment import band_nodes


def _log_probs(n_nodes, vocab, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n_nodes, vocab).log_softmax(-1)


def _brute_force_full_rnnt(node_lp, index, T, U, targets, blank):
    """Sum over EVERY monotonic path through the full T x U lattice.

    Enumerates the interleavings of U label emissions and T blanks directly --
    only tractable for tiny lattices, which is the point: it shares no code with
    the implementation under test.
    """
    total = None
    for blank_positions in itertools.combinations(range(T + U), T):
        t = u = 0
        lp = 0.0
        ok = True
        for step in range(T + U):
            node = index.get((t, u))
            if node is None:
                ok = False
                break
            if step in blank_positions:
                lp = lp + node_lp[node, blank]
                t += 1
            else:
                if u >= U:
                    ok = False
                    break
                lp = lp + node_lp[node, targets[u]]
                u += 1
        if not ok or t != T or u != U:
            continue
        # A path that consumed all T chunks ended by taking the blank out of
        # (T-1, U); that final blank is scored by the caller, not here.
        total = lp if total is None else torch.logaddexp(total, lp)
    return total


class TestBandedRNNT:
    @pytest.mark.unit
    def test_band_zero_equals_the_forced_cross_entropy(self):
        """One surviving path, so the sum degenerates to that path's likelihood."""
        chunks = [[5, 6], [], [7]]  # tokens per chunk
        vocab, blank = 10, 9
        per_utt, n_chunks, t_lens = build_lattices([chunks], band=0)
        lat = BandedLattice(per_utt, n_chunks, t_lens)
        lp = _log_probs(lat.num_nodes, vocab)
        targets = torch.tensor([[5, 6, 7]])

        banded = banded_rnnt_loss(lp, lat, targets, blank)[0]

        # The forced path, scored directly: each chunk emits its tokens then one
        # blank, and u advances only on a real token.
        index = {(t, u): i for i, (b, t, u) in enumerate(zip(lat.b_idx, lat.t_idx, lat.u_idx))}
        u, forced = 0, 0.0
        for t, toks in enumerate(chunks):
            for tok in toks:
                forced = forced + lp[index[(t, u)], tok]
                u += 1
            forced = forced + lp[index[(t, u)], blank]
        assert torch.allclose(banded, -forced, atol=1e-5), f"{banded} vs {-forced}"

    @pytest.mark.unit
    def test_wide_band_equals_full_rnnt(self):
        """Every path is inside the band, so it must match brute-force RNN-T."""
        chunks = [[3], [4], []]
        vocab, blank = 6, 5
        T, U = len(chunks), 2
        per_utt, n_chunks, t_lens = build_lattices([chunks], band=99)
        lat = BandedLattice(per_utt, n_chunks, t_lens)
        assert lat.num_nodes == T * (U + 1), "a wide band must cover the whole lattice"

        lp = _log_probs(lat.num_nodes, vocab, seed=1)
        targets = torch.tensor([[3, 4]])
        banded = banded_rnnt_loss(lp, lat, targets, blank)[0]

        index = {(t, u): i for i, (b, t, u) in enumerate(zip(lat.b_idx, lat.t_idx, lat.u_idx))}
        # Brute force counts the T-1 blanks that advance time, then the final
        # blank out of (T-1, U) is added separately -- same convention as the
        # implementation.
        ref = _brute_force_full_rnnt(lp, index, T - 1, U, [3, 4], blank)
        ref = ref + lp[index[(T - 1, U)], blank]
        assert torch.allclose(banded, -ref, atol=1e-5), f"{banded} vs {-ref}"

    @pytest.mark.unit
    def test_a_wider_band_never_increases_the_loss(self):
        """More admissible paths can only add probability mass."""
        chunks = [[1, 2], [3], [], [4]]
        vocab, blank = 8, 7
        targets = torch.tensor([[1, 2, 3, 4]])
        prev = None
        for band in (0, 1, 2, 5):
            per_utt, n_chunks, t_lens = build_lattices([chunks], band=band)
            lat = BandedLattice(per_utt, n_chunks, t_lens)
            torch.manual_seed(7)
            lp = torch.randn(lat.num_nodes, vocab).log_softmax(-1)
            # Same generator seed gives the same values per node only if the node
            # ORDER is stable, which it is: nodes are emitted grouped by t.
            loss = banded_rnnt_loss(lp, lat, targets, blank)[0].item()
            if prev is not None:
                assert loss <= prev + 1e-4, f"band {band} loss {loss} > narrower band {prev}"
            prev = loss

    @pytest.mark.unit
    def test_cost_grows_as_expected_with_band(self):
        """band 0 is exactly U + T; band 1 is ~2.2x that, not 3x.

        Each chunk contributes (tokens in the 2*band+1 chunks around it) + 1
        nodes. The trailing +1 -- the node a chunk's blank is taken from -- does
        not scale with the band, so on Granary's ~1.5 tokens per chunk the real
        multiplier is closer to 3U/(U+T) than to 3.
        """
        import random

        random.seed(0)
        counts = [random.choice([0, 1, 1, 2, 2, 3]) for _ in range(54)]
        u_total, t_total = sum(counts), len(counts)

        n0 = len(band_nodes(counts, 0))
        assert n0 == u_total + t_total, "band 0 must be exactly the forced path's node count"

        n1 = len(band_nodes(counts, 1))
        assert n1 == pytest.approx(3 * u_total + t_total, rel=0.05), "band 1 should be about 3U + T"
        assert 1.8 <= n1 / n0 <= 2.6, f"band 1 cost ratio {n1 / n0:.2f} outside the expected range"

        # And still far cheaper than marginalising over the whole lattice.
        assert t_total * (u_total + 1) / n1 > 10

    @pytest.mark.unit
    def test_gradients_flow_to_every_scored_node(self):
        chunks = [[1, 2], [3]]
        per_utt, n_chunks, t_lens = build_lattices([chunks], band=1)
        lat = BandedLattice(per_utt, n_chunks, t_lens)
        lp = _log_probs(lat.num_nodes, 6, seed=3).requires_grad_(True)
        banded_rnnt_loss(lp, lat, torch.tensor([[1, 2, 3]]), 5).sum().backward()
        assert lp.grad is not None and torch.isfinite(lp.grad).all()
        assert (lp.grad.abs().sum(dim=1) > 0).all(), "a scored node received no gradient"

    @pytest.mark.unit
    def test_batches_are_independent(self):
        a, b = [[1], [2]], [[3, 4], []]
        per_utt, n_chunks, t_lens = build_lattices([a, b], band=1)
        lat = BandedLattice(per_utt, n_chunks, t_lens)
        lp = _log_probs(lat.num_nodes, 6, seed=5)
        targets = torch.tensor([[1, 2, 0], [3, 4, 0]])
        both = banded_rnnt_loss(lp, lat, targets, 5)

        for i, chunks in enumerate((a, b)):
            p1, c1, l1 = build_lattices([chunks], band=1)
            lat1 = BandedLattice(p1, c1, l1)
            offset = 0 if i == 0 else lat.b_idx.index(1)
            alone = banded_rnnt_loss(lp[offset : offset + lat1.num_nodes], lat1, targets[i : i + 1], 5)
            assert torch.allclose(both[i], alone[0], atol=1e-5), f"utterance {i} depends on its neighbour"


@pytest.mark.unit
def test_band_side_later_only_allows_deferral():
    """'later' must widen the band DOWNWARD in u, never upward.

    u is the number of labels emitted BY chunk t, so a LOWER u means fewer labels
    emitted so far -- a word deferred to a later chunk. A HIGHER u means a word
    pulled forward, i.e. emitted before its audio has fully arrived, which is the
    thing num_delay_frames exists to prevent.

    Reads backwards, so it is pinned here: a sign flip would be invisible in
    training and would just look like slightly worse WER.
    """
    from nemo.collections.asr.parts.utils.chat_alignment import band_nodes

    counts = [2, 2, 2]  # forced path: chunk t owns u in [2t, 2t+2]

    forced = {t: (u, u) for t, u in []}  # placeholder, computed below
    by_t = {}
    for side in ("both", "later", "earlier"):
        nodes = band_nodes(counts, 1, side)
        by_t[side] = {
            t: (min(u for tt, u in nodes if tt == t), max(u for tt, u in nodes if tt == t)) for t in range(3)
        }

    for t in range(3):
        lo_b, hi_b = by_t["both"][t]
        lo_l, hi_l = by_t["later"][t]
        lo_e, hi_e = by_t["earlier"][t]
        # 'later' keeps the forced upper edge and widens below.
        assert hi_l == 2 * t + 2, f"'later' widened UPWARD at t={t}: {hi_l}"
        assert lo_l == lo_b, f"'later' should widen down as far as 'both' at t={t}"
        # 'earlier' is the mirror image.
        assert lo_e == 2 * t, f"'earlier' widened DOWNWARD at t={t}: {lo_e}"
        assert hi_e == hi_b, f"'earlier' should widen up as far as 'both' at t={t}"


@pytest.mark.unit
def test_one_sided_band_has_fewer_nodes_than_two_sided():
    """The point of the one-sided band: roughly half the lattice."""
    from nemo.collections.asr.parts.utils.chat_alignment import band_nodes

    counts = [3] * 8
    both = len(band_nodes(counts, 1, "both"))
    later = len(band_nodes(counts, 1, "later"))
    forced = len(band_nodes(counts, 0, "later"))
    assert forced < later < both, f"forced={forced} later={later} both={both}"


@pytest.mark.unit
@pytest.mark.parametrize("side", ["both", "later", "earlier"])
def test_band_zero_is_the_forced_path_whatever_the_side(side):
    from nemo.collections.asr.parts.utils.chat_alignment import band_nodes

    assert band_nodes([2, 2, 2], 0, side) == band_nodes([2, 2, 2], 0, "both")


@pytest.mark.unit
def test_band_side_is_validated():
    from nemo.collections.asr.parts.utils.chat_alignment import band_nodes

    with pytest.raises(ValueError, match="side must be"):
        band_nodes([1, 1], 1, "rightwards")


def test_padding_does_not_create_label_transitions_for_short_utterances():
    """A short utterance's loss must not depend on how long its BATCHMATES are.

    ``has_label`` masks on ``targets.shape[1]`` -- the BATCH-PADDED width, not
    each utterance's own target length -- so nodes at ``u == U_b`` for a short
    utterance are scored as if a label followed, gathering the PAD id. That
    looks like a latent correctness bug and is worth pinning, but it is benign
    for a specific structural reason: ``band_nodes`` never emits a node beyond
    ``u == U_b``, so nothing exists at ``u + 1`` to consume that transition, and
    the gathered value is discarded. Measured on this batch: 6 nodes sit at or
    past their own end and 0 of them feed a label transition.

    Two independent pins, because the cheap one alone would not catch a
    regression that made the mask matter:
      1. the same utterance batched against a long vs. a short neighbour must
         score identically;
      2. the loss must be invariant to the PAD VALUE itself -- the direct test
         that the gathered padding is never consumed.
    """
    torch.manual_seed(0)
    V, blank = 10, 10
    short = [[7], [8], [], []]

    losses = []
    for neighbour in ([[1, 2], [3], [4, 5], [6]], [[1], [2], [], []]):
        chunks = [short, neighbour]
        per_utt, num_chunks, target_lens = build_lattices(chunks, 1, "both")
        lat = BandedLattice(per_utt, num_chunks, target_lens)
        targets = torch.zeros((2, max(max(target_lens), 1)), dtype=torch.long)
        for b, cs in enumerate(chunks):
            flat = [t for c in cs for t in c]
            if flat:
                targets[b, : len(flat)] = torch.tensor(flat)
        # Score every node identically so the only possible difference is which
        # transitions the mask admits.
        lp = torch.full((lat.num_nodes, V + 1), -1.0).log_softmax(-1)
        losses.append(banded_rnnt_loss(lp, lat, targets, blank)[0].item())

    assert losses[0] == pytest.approx(
        losses[1], abs=1e-5
    ), f"short utterance's loss changed with its batchmate's length: {losses}"

    # Pin 2: the pad value must not reach the loss at all.
    chunks = [short, [[1, 2], [3], [4, 5], [6]]]
    per_utt, num_chunks, target_lens = build_lattices(chunks, 1, "both")
    lat = BandedLattice(per_utt, num_chunks, target_lens)
    lp = torch.randn(lat.num_nodes, V + 1).log_softmax(-1)
    by_pad = []
    for pad in (0, 3, V):
        targets = torch.full((2, max(target_lens)), pad, dtype=torch.long)
        for b, cs in enumerate(chunks):
            flat = [t for c in cs for t in c]
            if flat:
                targets[b, : len(flat)] = torch.tensor(flat)
        by_pad.append(banded_rnnt_loss(lp, lat, targets, blank).tolist())
    assert by_pad[0] == by_pad[1] == by_pad[2], f"loss depends on the pad value: {by_pad}"
