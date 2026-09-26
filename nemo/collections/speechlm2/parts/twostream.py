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
"""Two-stream SCRIPT: text and audio meet only in the LLM's last layer.

WHY THIS EXISTS. In packed SCRIPT the band is expressed as SEQUENCE LENGTH: each
candidate cut becomes another branch segment carrying its own copy of the chunk's
audio, so widening the band multiplies the packed length and the (B, L, V) logit
tensor with it. Measured, that is what makes chunk_size=2 with band_words=2
unrunnable.

Here the band is expressed as a LATTICE instead. Because the text stream is
causal and never attends audio, the hidden state after text prefix p is
independent of the chunk and of which cut is being scored -- so it is computed
ONCE and every lattice cell indexes into it. The band then selects a SUBSET of
(chunk, text-position) cells rather than adding segments: widening it costs less
work, not more.

    layers 1..N-1:  [prompt][x_1 .. x_p]            pure text, causal, cached
    layer N:        [prompt][x_1 .. x_p][audio_t]   audio appended, still causal
                                         ^ read the distribution here

Cell (t, p) answers: "given the transcript up to p and chunk t's audio, what
comes next?" That is exactly the quantity the banded lattice consumes, so
span_scores() and banded_forward() are reused unchanged -- including the
band_words=0 == forced-loss equivalence, which is the correctness test for this
whole path.

STATUS: first implementation. The packing below is deliberately simple (one audio
block per cell) so it can be checked against the existing lattice before being
optimised; see the note on block sharing in build_joint_inputs().
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

NEG_INF = -1.0e9


@dataclass
class CellIndex:
    """Which (chunk, text-position) cells the band actually needs.

    Attributes:
        chunk: ``(n_cells,)`` chunk id of each cell.
        text_pos: ``(n_cells,)`` spine token index ``p``; the cell conditions on
            the transcript prefix ``[0, p)`` and predicts the token AT ``p``.
        lo: ``(T,)`` first text position needed per chunk.
        hi: ``(T,)`` last text position needed per chunk (inclusive).
        offset: ``(T,)`` index into the flat cell axis where chunk t's block starts.
    """

    chunk: Tensor
    text_pos: Tensor
    lo: Tensor
    hi: Tensor
    offset: Tensor

    @property
    def n_cells(self) -> int:
        return int(self.chunk.numel())


def plan_cells(cut: Tensor, cut_valid: Tensor, reach: Tensor) -> CellIndex:
    """Enumerate the cells the band needs, per chunk.

    For chunk ``t`` the lattice may start at any valid candidate cut and run to
    ``reach[t]``, so every text position in ``[min valid cut, reach]`` is needed
    exactly once -- shared across all candidates of that chunk. That sharing is
    the point: in packed SCRIPT each candidate re-materialises its own positions.

    Args:
        cut: ``(T, J)`` candidate start cuts.
        cut_valid: ``(T, J)`` which candidates are real (not padding).
        reach: ``(T,)`` furthest spine index chunk t may emit up to.
    """
    T = int(cut.shape[0])
    chunk_ids: List[int] = []
    positions: List[int] = []
    lo_l: List[int] = []
    hi_l: List[int] = []
    off_l: List[int] = []
    for t in range(T):
        valid = cut[t][cut_valid[t]]
        if valid.numel() == 0:
            # No real candidate: emit a degenerate single cell so the tensors stay
            # rectangular. cut_valid keeps it out of the DP anyway.
            lo = hi = int(reach[t].item())
        else:
            lo = int(valid.min().item())
            hi = int(reach[t].item())
            if hi < lo:
                hi = lo
        off_l.append(len(chunk_ids))
        for p in range(lo, hi + 1):
            chunk_ids.append(t)
            positions.append(p)
        lo_l.append(lo)
        hi_l.append(hi)
    dev = cut.device
    return CellIndex(
        chunk=torch.tensor(chunk_ids, dtype=torch.long, device=dev),
        text_pos=torch.tensor(positions, dtype=torch.long, device=dev),
        lo=torch.tensor(lo_l, dtype=torch.long, device=dev),
        hi=torch.tensor(hi_l, dtype=torch.long, device=dev),
        offset=torch.tensor(off_l, dtype=torch.long, device=dev),
    )


def build_joint_inputs(
    text_h: Tensor,
    audio_emb: Tensor,
    cells: CellIndex,
    prompt_len: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pack the joint layer's input: the text stream once, then one audio block per cell.

    Layout::

        [ text_h  (m + U positions) ][ audio block for cell 0 ][ cell 1 ] ...

    The mask gives audio block ``c`` (chunk ``t_c``, text position ``p_c``) access
    to text keys ``< m + p_c`` and to its own block causally, and nothing else.
    That is the SCRIPT rule, but applied to ONE layer and with no text tokens
    duplicated -- only the audio is repeated, and only as queries.

    NOTE (optimisation left for later): blocks of the same chunk carry identical
    audio content and differ only in their text cutoff. A masked formulation with
    T blocks and per-query cutoffs would avoid repeating them; this version keeps
    one block per cell because it is obviously correct and the joint is a single
    layer, so the cost is bounded.

    Args:
        text_h: ``(m + U, H)`` last hidden state BEFORE the joint layer.
        audio_emb: ``(T, w, H)`` per-chunk audio, already projected to the LLM width.
        cells: from :func:`plan_cells`.
        prompt_len: ``m``.

    Returns:
        seq: ``(L_j, H)`` joint-layer input.
        mask: ``(L_j, L_j)`` bool, True where attention is allowed.
        read_at: ``(n_cells,)`` index of each block's LAST position -- where the
            output distribution is read.
    """
    tu, hdim = text_h.shape
    T, w, h2 = audio_emb.shape
    if h2 != hdim:
        raise ValueError(f"audio width {h2} != text width {hdim}")
    n = cells.n_cells

    blocks = audio_emb[cells.chunk]  # (n_cells, w, H) -- gather, no copy of text
    seq = torch.cat([text_h, blocks.reshape(n * w, hdim)], dim=0)
    total = tu + n * w

    mask = torch.zeros((total, total), dtype=torch.bool, device=text_h.device)
    # Text queries: causal among text. Their outputs are unused (we only read audio
    # blocks), but a well-formed mask keeps the layer's arithmetic sane.
    tri = torch.ones((tu, tu), dtype=torch.bool, device=text_h.device).tril()
    mask[:tu, :tu] = tri

    ar = torch.arange(w, device=text_h.device)
    for c in range(n):
        s = tu + c * w
        cutoff = prompt_len + int(cells.text_pos[c].item())  # keys strictly before p
        mask[s : s + w, :cutoff] = True
        own = (ar.unsqueeze(1) >= ar.unsqueeze(0)).T  # causal within the block
        mask[s : s + w, s : s + w] = own

    read_at = torch.arange(n, device=text_h.device) * w + (tu + w - 1)
    return seq, mask, read_at


def cell_logprobs(
    logits: Tensor,
    cells: CellIndex,
    spine_ids: Tensor,
    eot_id: int,
) -> Tuple[Tensor, Tensor]:
    """Per-cell log P(next spine token) and log P(<eot>).

    Args:
        logits: ``(n_cells, V)`` read at each block's last position.
        cells: from :func:`plan_cells`.
        spine_ids: ``(U,)`` transcript token ids.
        eot_id: end-of-chunk id.

    Returns:
        tok_lp: ``(n_cells,)`` log P(spine[p]) -- NEG_INF where p == U (nothing left).
        stop_lp: ``(n_cells,)`` log P(<eot>).
    """
    lse = torch.logsumexp(logits.float(), dim=-1)
    stop_lp = logits[:, eot_id].float() - lse
    U = int(spine_ids.numel())
    p = cells.text_pos
    in_range = p < U
    tgt = spine_ids[p.clamp(max=max(U - 1, 0))]
    tok_lp = logits.gather(1, tgt.unsqueeze(1)).squeeze(1).float() - lse
    tok_lp = torch.where(in_range, tok_lp, torch.full_like(tok_lp, NEG_INF))
    return tok_lp, stop_lp


def gather_span_tensors(
    tok_lp: Tensor,
    stop_lp: Tensor,
    cells: CellIndex,
    cut: Tensor,
    K: int,
) -> Tuple[Tensor, Tensor]:
    """Rearrange the flat cell grid into the (T, J, K) form span_scores() wants.

    ``token_logprob[t, j, k]`` must be log P(spine[cut[t,j] + k]) under chunk t,
    and ``stop_logprob[t, j, k]`` log P(<eot>) after emitting k tokens from that
    cut. Both are lookups into the per-chunk cell block.
    """
    T, J = cut.shape
    dev = tok_lp.device
    ks = torch.arange(K + 1, device=dev)
    pos = cut.unsqueeze(-1) + ks  # (T, J, K+1) absolute text positions

    # flat cell index = offset[t] + (pos - lo[t]); clamp keeps out-of-band lookups
    # in bounds -- span_valid marks them invalid downstream, so the value is never used.
    rel = pos - cells.lo.view(T, 1, 1)
    span = (cells.hi - cells.lo + 1).view(T, 1, 1)
    rel = rel.clamp(min=0).minimum(span - 1)
    flat = (cells.offset.view(T, 1, 1) + rel).clamp(min=0, max=cells.n_cells - 1)

    stop_logprob = stop_lp[flat]  # (T, J, K+1)
    token_logprob = tok_lp[flat][..., :K]  # (T, J, K)
    return token_logprob, stop_logprob
