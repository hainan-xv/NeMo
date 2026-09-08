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
"""RNN-T forward algorithm restricted to a band around a forced alignment.

THE MIDDLE GROUND. The ordinary RNN-T loss sums over every alignment and needs a
``[B, T, U, V+1]`` tensor; conditioning on one forced alignment scores ``U + T``
positions but trusts the aligner absolutely. This sums over every valid path
that stays within ``band`` chunks of the aligner's -- keeping the alignment as a
prior rather than a constraint, at roughly ``2*band + 1`` times the single-path
cost.

Two properties make it checkable rather than merely plausible, and both are
asserted in the tests:

  * at ``band = 0`` exactly one path survives, so the loss must equal the
    forced-alignment cross-entropy to floating point;
  * at ``band >= T`` the whole lattice is inside the band, so the loss must
    equal the full RNN-T loss.

Time is indexed in CHUNKS, matching the rest of CHAT: the joint emits per chunk,
and the marginalised arm already passes chunk counts as ``input_lengths``.
"""

from typing import List, Sequence, Tuple

import torch

__all__ = ["BandedLattice", "banded_rnnt_loss"]

NEG_INF = -1e30


class BandedLattice:
    """Node bookkeeping for one batch: which ``(b, t, u)`` are scored, and how
    each is reached.

    Built once per batch on the CPU from integer structure alone -- it depends on
    the alignment and lengths, never on the model output -- then reused for the
    gather, the recursion and the final read-out.
    """

    def __init__(
        self, per_utt_nodes: Sequence[Sequence[Tuple[int, int]]], num_chunks: Sequence[int], target_lens: Sequence[int]
    ):
        self.b_idx: List[int] = []
        self.t_idx: List[int] = []
        self.u_idx: List[int] = []
        pred_blank: List[int] = []  # node reached from (t-1, u), or -1
        pred_label: List[int] = []  # node reached from (t, u-1), or -1
        diagonal: List[int] = []

        index = {}
        for b, nodes in enumerate(per_utt_nodes):
            for t, u in nodes:
                index[(b, t, u)] = len(self.b_idx)
                self.b_idx.append(b)
                self.t_idx.append(t)
                self.u_idx.append(u)
                diagonal.append(t + u)
        for i in range(len(self.b_idx)):
            b, t, u = self.b_idx[i], self.t_idx[i], self.u_idx[i]
            pred_blank.append(index.get((b, t - 1, u), -1))
            pred_label.append(index.get((b, t, u - 1), -1))

        self.pred_blank = pred_blank
        self.pred_label = pred_label
        # A path ends having consumed every chunk and emitted every label; the
        # standard RNN-T termination then takes one final blank from there.
        self.final = [index[(b, int(num_chunks[b]) - 1, int(target_lens[b]))] for b in range(len(per_utt_nodes))]
        # Anti-diagonals: alpha(t,u) depends only on (t-1,u) and (t,u-1), both of
        # which have a smaller t+u, so every node on a diagonal can be computed
        # at once. That makes the recursion T+U sequential steps instead of T*U.
        order: dict = {}
        for i, d in enumerate(diagonal):
            order.setdefault(d, []).append(i)
        self.diagonals = [order[d] for d in sorted(order)]
        self.num_nodes = len(self.b_idx)

    def index_tensors(self, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        as_t = lambda x: torch.tensor(x, dtype=torch.long, device=device)  # noqa: E731
        return as_t(self.b_idx), as_t(self.t_idx), as_t(self.u_idx)


def banded_rnnt_loss(
    log_probs: torch.Tensor,
    lattice: BandedLattice,
    targets: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    """Negative log-likelihood per utterance, summing paths inside the band.

    Args:
        log_probs: ``[N, V+1]`` LOG-probabilities at the lattice nodes, in the
            node order ``lattice`` defines.
        lattice: the node bookkeeping.
        targets: ``[B, U_max]`` label ids.
        blank_id: index of the blank.

    Returns:
        ``[B]`` negative log-likelihoods.
    """
    device = log_probs.device
    n = lattice.num_nodes

    b_t = torch.tensor(lattice.b_idx, dtype=torch.long, device=device)
    u_t = torch.tensor(lattice.u_idx, dtype=torch.long, device=device)

    blank_lp = log_probs[:, blank_id]
    # The label leaving node (t, u) is targets[b, u]; at u == U there is none, so
    # clamp the gather and mask the result rather than indexing out of bounds.
    u_clamped = u_t.clamp(max=targets.shape[1] - 1) if targets.shape[1] > 0 else torch.zeros_like(u_t)
    label_lp = log_probs.gather(1, targets[b_t, u_clamped].unsqueeze(1)).squeeze(1)
    has_label = u_t < torch.tensor([targets.shape[1]], device=device)
    label_lp = torch.where(has_label, label_lp, torch.full_like(label_lp, NEG_INF))

    pb = torch.tensor(lattice.pred_blank, dtype=torch.long, device=device)
    pl = torch.tensor(lattice.pred_label, dtype=torch.long, device=device)
    pb_ok, pl_ok = pb >= 0, pl >= 0
    pb_safe, pl_safe = pb.clamp(min=0), pl.clamp(min=0)

    alpha = torch.full((n,), NEG_INF, device=device, dtype=log_probs.dtype)
    for step, diag in enumerate(lattice.diagonals):
        idx = torch.tensor(diag, dtype=torch.long, device=device)
        if step == 0:
            # (t=0, u=0) for every utterance: a path starts there with prob 1.
            alpha = torch.index_put(alpha, (idx,), torch.zeros(len(diag), device=device, dtype=alpha.dtype))
            continue
        i_pb, i_pl = pb_safe[idx], pl_safe[idx]
        from_blank = torch.where(pb_ok[idx], alpha[i_pb] + blank_lp[i_pb], torch.full_like(alpha[i_pb], NEG_INF))
        from_label = torch.where(pl_ok[idx], alpha[i_pl] + label_lp[i_pl], torch.full_like(alpha[i_pl], NEG_INF))
        alpha = torch.index_put(alpha, (idx,), torch.logaddexp(from_blank, from_label))

    final = torch.tensor(lattice.final, dtype=torch.long, device=device)
    return -(alpha[final] + blank_lp[final])


def build_lattices(
    chunk_tokens_per_utt: Sequence[Sequence[Sequence[int]]], band: int
) -> Tuple[List[List[Tuple[int, int]]], List[int], List[int]]:
    """Nodes, chunk counts and label counts for a batch of forced alignments."""
    from nemo.collections.asr.parts.utils.chat_alignment import band_nodes

    per_utt, num_chunks, target_lens = [], [], []
    for chunks in chunk_tokens_per_utt:
        counts = [len(c) for c in chunks]
        per_utt.append(band_nodes(counts, band))
        num_chunks.append(len(counts))
        target_lens.append(sum(counts))
    return per_utt, num_chunks, target_lens
