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
"""Tests for the two-stream sigma path.

The objective is unchanged from packed SCRIPT -- only how sigma is produced
changes -- so the tests that matter are the ones that pin the CONTRACT with the
existing lattice: cell enumeration, the joint mask, and the band_words=0 ==
forced-loss equivalence.
"""

import pytest
import torch

from nemo.collections.speechlm2.parts.script_banded import banded_forward, span_scores
from nemo.collections.speechlm2.parts.twostream import (
    NEG_INF,
    build_joint_inputs,
    cell_logprobs,
    gather_span_tensors,
    plan_cells,
)


@pytest.mark.unit
def test_plan_cells_on_the_two_word_example():
    """'one two', 2 chunks, band 1 both sides.

    cands = [[0,1], [0,1,2]], reach = [2, 2]. Chunk 0 needs text positions
    [0..2], chunk 1 needs [0..2]; each position appears ONCE per chunk, which is
    the sharing that packed SCRIPT does not get.
    """
    cut = torch.tensor([[0, 1, 1], [0, 1, 2]])
    cut_valid = torch.tensor([[True, True, False], [True, True, True]])
    reach = torch.tensor([2, 2])

    cells = plan_cells(cut, cut_valid, reach)
    assert cells.lo.tolist() == [0, 0]
    assert cells.hi.tolist() == [2, 2]
    assert cells.chunk.tolist() == [0, 0, 0, 1, 1, 1]
    assert cells.text_pos.tolist() == [0, 1, 2, 0, 1, 2]
    assert cells.n_cells == 6
    assert cells.offset.tolist() == [0, 3]


@pytest.mark.unit
def test_joint_mask_gives_each_cell_its_own_text_cutoff():
    """Audio block for cell (t, p) must see text keys < m+p, and no more.

    This is the whole two-stream premise: text representations are shared, and a
    cell is distinguished ONLY by how much of them it may read.
    """
    m, U, H, T, w = 2, 3, 4, 2, 2
    text_h = torch.randn(m + U, H)
    audio = torch.randn(T, w, H)
    cut = torch.tensor([[0, 1], [1, 2]])
    cut_valid = torch.ones(2, 2, dtype=torch.bool)
    cells = plan_cells(cut, cut_valid, torch.tensor([2, 3]))

    seq, mask, read_at = build_joint_inputs(text_h, audio, cells, prompt_len=m)
    tu = m + U
    assert seq.shape == (tu + cells.n_cells * w, H)

    for c in range(cells.n_cells):
        p = int(cells.text_pos[c])
        start = tu + c * w
        row = mask[start]  # first query of this block
        assert row[: m + p].all(), f"cell {c} cannot see its own text prefix"
        assert not row[m + p : tu].any(), f"cell {c} sees text at or beyond p={p}"
        # blocks never see each other
        for other in range(cells.n_cells):
            if other == c:
                continue
            o = tu + other * w
            assert not row[o : o + w].any(), "audio blocks must not see each other"

    assert read_at.tolist() == [tu + c * w + w - 1 for c in range(cells.n_cells)]


@pytest.mark.unit
def test_cell_logprobs_masks_positions_past_the_transcript():
    """A cell at p == U has no next token; its token score must be unusable."""
    U, V = 3, 7
    logits = torch.randn(4, V)
    cells = plan_cells(torch.tensor([[0]]), torch.ones(1, 1, dtype=torch.bool), torch.tensor([3]))
    spine = torch.tensor([1, 2, 3])
    tok_lp, stop_lp = cell_logprobs(logits, cells, spine, eot_id=0)
    assert cells.text_pos.tolist() == [0, 1, 2, 3]
    assert tok_lp[3].item() == pytest.approx(NEG_INF)
    assert torch.isfinite(tok_lp[:3]).all()
    assert torch.isfinite(stop_lp).all()


@pytest.mark.unit
def test_band_zero_reproduces_the_forced_loss():
    """THE CONTRACT. With a single candidate per chunk the lattice must collapse
    to plain cross-entropy over the aligner's assignment.

    This is the same equivalence packed SCRIPT satisfies, so it validates the new
    sigma pipeline against a reference that already works -- before any quality
    number depends on it.
    """
    torch.manual_seed(0)
    U, V, T = 4, 11, 2
    spine = torch.tensor([3, 5, 7, 9])
    eot = 0
    # Aligner: chunk 0 emits spine[0:2], chunk 1 emits spine[2:4].
    cut = torch.tensor([[0], [2]])
    cut_valid = torch.ones(T, 1, dtype=torch.bool)
    reach = torch.tensor([2, 4])
    cells = plan_cells(cut, cut_valid, reach)

    logits = torch.randn(cells.n_cells, V)
    tok_lp, stop_lp = cell_logprobs(logits, cells, spine, eot_id=eot)

    K = 2
    token_logprob, stop_logprob = gather_span_tensors(tok_lp, stop_lp, cells, cut, K)
    span_valid = torch.ones(T, 1, K + 1, dtype=torch.bool)
    for t in range(T):
        for k in range(K + 1):
            span_valid[t, 0, k] = (int(cut[t, 0]) + k) <= int(reach[t])

    sigma = span_scores(token_logprob.unsqueeze(0), stop_logprob.unsqueeze(0), span_valid.unsqueeze(0))
    nll = banded_forward(
        sigma,
        cut.unsqueeze(0),
        cut_valid.unsqueeze(0),
        torch.tensor([T]),
        torch.tensor([U]),
    )[0]

    # Reference: score exactly the aligner's path, by hand.
    def cell(t, p):
        return int(cells.offset[t]) + (p - int(cells.lo[t]))

    ref = 0.0
    for t in range(T):
        u = int(cut[t, 0])
        nxt = int(cut[t + 1, 0]) if t + 1 < T else U
        for p in range(u, nxt):
            ref = ref + tok_lp[cell(t, p)]
        ref = ref + stop_lp[cell(t, nxt)]
    assert torch.allclose(nll, -ref, atol=1e-4), f"banded(J=1)={nll.item():.6f} vs forced={-ref.item():.6f}"


# ---------------------------------------------------------------------------
# Model-level: exercise twostream_loss with a stub LLM, so the wiring is checked
# without building a 1.7B model.
# ---------------------------------------------------------------------------


class _StubLayer(torch.nn.Module):
    """A transformer-layer stand-in that RESPECTS the mask.

    A layer that ignored the mask would let every cell see the whole transcript
    and the tests below would pass while the design was broken -- so it applies
    real masked attention, just with identity projections.
    """

    def forward(self, h, attention_mask=None, position_ids=None, position_embeddings=None):
        scores = h @ h.transpose(-1, -2)
        if attention_mask is not None:
            scores = scores + attention_mask[0]
        w = torch.softmax(scores.float(), dim=-1).to(h.dtype)
        return (w @ h,)


class _StubCore(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layers = torch.nn.ModuleList([layer])
        self.rotary_emb = None


@pytest.mark.unit
def test_twostream_loss_runs_and_is_finite(monkeypatch):
    """End-to-end through twostream_loss with stubs: shapes line up, loss is finite."""
    from nemo.collections.speechlm2.models.twostream_model import TwoStreamSTTModel

    H, V, T, w, U, m = 8, 13, 2, 2, 4, 2
    spine = torch.tensor([3, 5, 7, 9])
    cut = torch.tensor([[0], [2]])
    cut_valid = torch.ones(T, 1, dtype=torch.bool)
    reach = torch.tensor([2, 4])
    K = 2
    span_valid = torch.ones(T, 1, K + 1, dtype=torch.bool)

    model = TwoStreamSTTModel.__new__(TwoStreamSTTModel)
    torch.nn.Module.__init__(model)

    class _Cfg:
        joint_layers = 1

    model.core_cfg = _Cfg()
    model._eot_id = 0
    core = _StubCore(_StubLayer())
    head = torch.nn.Linear(H, V, bias=False)

    text_h = torch.randn(m + U, H)
    monkeypatch.setattr(TwoStreamSTTModel, "_llm_core", lambda self: core)
    monkeypatch.setattr(TwoStreamSTTModel, "_lm_head_of", lambda self: head)
    monkeypatch.setattr(TwoStreamSTTModel, "_text_hidden", lambda self, ids: text_h.unsqueeze(0))

    nll = model.twostream_loss(
        text_ids=torch.zeros(m + U, dtype=torch.long),
        audio_emb=torch.randn(T, w, H),
        cut=cut,
        cut_valid=cut_valid,
        span_valid=span_valid,
        reach=reach,
        spine_ids=spine,
        prompt_len=m,
        n_tokens=U,
    )
    assert nll.shape == ()
    assert torch.isfinite(nll), "banded NLL is not finite"
    assert nll.item() > 0, "NLL should be positive"


@pytest.mark.unit
def test_cost_does_not_grow_with_band_width():
    """THE DESIGN CLAIM, made measurable.

    In packed SCRIPT the band multiplies packed length. Here the cells a chunk
    needs are [min valid cut, reach] -- widening the band moves min cut DOWN but
    adds no new work per candidate, and candidates SHARE cells. So the joint
    sequence must not scale with J.
    """
    H, T, w, m = 4, 2, 2, 1
    text_h = torch.randn(m + 4, H)
    audio = torch.randn(T, w, H)
    reach = torch.tensor([2, 4])

    def joint_len(cut, cut_valid):
        cells = plan_cells(cut, cut_valid, reach)
        seq, _, _ = build_joint_inputs(text_h, audio, cells, prompt_len=m)
        return seq.shape[0], cells.n_cells

    narrow = joint_len(torch.tensor([[0], [2]]), torch.ones(2, 1, dtype=torch.bool))
    wide = joint_len(torch.tensor([[0, 0], [1, 2]]), torch.ones(2, 2, dtype=torch.bool))

    assert wide[1] >= narrow[1], "wider band should need at least as many cells"
    # J doubled; cells must NOT double -- candidates share the per-chunk block.
    assert wide[1] < 2 * narrow[1], f"cells scaled with J: {narrow[1]} -> {wide[1]}"
