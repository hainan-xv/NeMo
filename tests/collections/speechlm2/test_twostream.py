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
        # Mirror the real dataclass's fields. twostream_loss reads
        # joint_text_context DIRECTLY rather than via getattr(..., default), on
        # purpose: a silent default is what let a mis-plumbed config train an arm
        # that was a duplicate of its own baseline. A stub missing a field should
        # break loudly here instead.
        joint_layers = 1
        audio_position_mode = "cut"
        joint_text_context = "full"

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


@pytest.mark.unit
def test_generate_is_overridden_not_inherited():
    """THE REGRESSION THIS EXISTS FOR.

    TwoStreamSTTModel originally defined only training methods, so generate,
    validation_step and _eval_step were all inherited from ScriptSTTModel -- which
    decodes through the PACKED layout (audio at layer 0, all N layers). Validation
    therefore reported the WER of a packed-SCRIPT reading of these weights. On a
    warm-started run that looks entirely plausible and measures nothing about the
    architecture being trained, which is why it went unnoticed while the training
    loss was visibly wrong.
    """
    from nemo.collections.speechlm2.models.script_model import ScriptSTTModel
    from nemo.collections.speechlm2.models.twostream_model import TwoStreamSTTModel

    assert "generate" in TwoStreamSTTModel.__dict__, (
        "generate is inherited from ScriptSTTModel -- validation would decode through the "
        "packed layout and report WER for a different architecture"
    )
    assert TwoStreamSTTModel.generate is not ScriptSTTModel.generate


@pytest.mark.unit
def test_generate_uses_the_two_stream_read_out():
    """generate must go through _project_to_vocab (norm THEN head), not lm_head raw.

    Applying lm_head without model.norm changed the argmax on 74% of positions on
    Qwen3-1.7B, so a decode path that skipped it would produce different text --
    the loss bug and the decode bug are the same bug in two places.
    """
    import inspect

    from nemo.collections.speechlm2.models.twostream_model import TwoStreamSTTModel

    src = inspect.getsource(TwoStreamSTTModel.generate)
    assert "_project_to_vocab" in src, "generate bypasses the normalised read-out"
    assert "_run_joint" in src, "generate does not use the joint layer"
    # The incremental text cache is the design's inference claim; assert it is
    # actually used rather than the prefix being recomputed per token.
    assert "past_key_values" in src, "generate does not reuse the text-stream cache"


# ----------------------------------------------------------------------
# Regressions from the first DFW launch of the two-stream arms. Both bugs
# were invisible locally because the probe's stubs were LOOSER than the
# real collaborators: a stub tokenizer that batch-encoded a list, and a
# stub _chunk_audio that ignored n_chunks.
# ----------------------------------------------------------------------
def test_chunk_audio_derives_the_count_when_caller_passes_zero():
    """Decoding cannot know the chunk count up front and passes 0.

    That used to hit ``embs[:0]`` and return an EMPTY tensor, so generate ran
    zero chunks and every hypothesis came back blank -- a silent total WER
    failure, not a crash.
    """
    import torch

    from nemo.collections.speechlm2.models.twostream_model import TwoStreamSTTModel

    class _Stub:
        # 40 frames of width-8 chunks -> 5 chunks
        def perception(self, input_signal, input_signal_length):
            return torch.zeros(1, 40, 6), None

    frames = TwoStreamSTTModel._chunk_audio(_Stub(), torch.zeros(1, 100), torch.tensor([100]), 8, 0)
    assert frames.shape[0] == 5, f"expected 5 derived chunks, got {frames.shape[0]}"
    assert frames.shape[1:] == (8, 6)

    # A partial trailing chunk must round UP, not truncate away real audio.
    class _Stub2:
        def perception(self, input_signal, input_signal_length):
            return torch.zeros(1, 41, 6), None

    assert TwoStreamSTTModel._chunk_audio(_Stub2(), torch.zeros(1, 100), torch.tensor([100]), 8, 0).shape[0] == 6

    # An explicit count from training still wins over the derivation.
    assert TwoStreamSTTModel._chunk_audio(_Stub(), torch.zeros(1, 100), torch.tensor([100]), 8, 3).shape[0] == 3


def test_generate_accepts_one_prompt_per_utterance():
    """``_validation_system_prompts`` returns a LIST when cuts carry prompts.

    Passing that list straight to ``text_to_ids`` raised "TextEncodeInput must
    be ..." and killed both arms ~2 min in. The signature must match the
    inherited contract, and the body must index per utterance.
    """
    import inspect
    import typing

    from nemo.collections.speechlm2.models.script_model import ScriptSTTModel
    from nemo.collections.speechlm2.models.twostream_model import TwoStreamSTTModel

    ann = inspect.signature(TwoStreamSTTModel.generate).parameters["system_prompt"].annotation
    base = inspect.signature(ScriptSTTModel.generate).parameters["system_prompt"].annotation
    assert ann == base, f"two-stream generate must accept what the base one does: {base!r} vs {ann!r}"
    assert typing.get_args(ann), "expected a Union[str, List[str]], not a bare str"

    src = inspect.getsource(TwoStreamSTTModel.generate)
    assert "isinstance(system_prompt, (list, tuple))" in src, "generate must select the per-utterance prompt"


def test_twostream_config_fields_actually_reach_core_cfg():
    """The subclass config must be a real dataclass AND be used to build core_cfg.

    Two independent failures conspired here, both silent:
      * TwoStreamSTTModelConfig lacked @dataclass, so its annotations were plain
        class attributes and dataclasses.fields() returned only the parent's;
      * ScriptSTTModel hardcodes to_dataclass(ScriptSTTModelConfig, cfg), so even
        a correct dataclass would not have been consulted.
    Every read site uses getattr(..., default), so ++model.joint_layers=4 was
    accepted, logged, and discarded -- and an arm launched with
    extra_joint_layer=true trained as an exact duplicate of the baseline.
    """
    import dataclasses
    import inspect

    from nemo.collections.speechlm2.models.script_model import ScriptSTTModelConfig
    from nemo.collections.speechlm2.models.twostream_model import TwoStreamSTTModel, TwoStreamSTTModelConfig

    own = {f.name for f in dataclasses.fields(TwoStreamSTTModelConfig)}
    parent = {f.name for f in dataclasses.fields(ScriptSTTModelConfig)}
    for field in ("joint_layers", "loss_reduction", "extra_joint_layer", "extra_joint_init_from_last"):
        assert field in own, f"{field} is not a dataclass field -- is @dataclass missing?"
        assert field not in parent, f"{field} unexpectedly on the parent; this test would not catch a regression"

    # ...and the model must rebuild core_cfg against it, not inherit the parent's.
    src = inspect.getsource(TwoStreamSTTModel.__init__)
    assert "to_dataclass(TwoStreamSTTModelConfig" in src, "core_cfg must be rebuilt against the subclass config"

    # ...and the startup log/validation must be DERIVED from the dataclass, not a
    # hardcoded name list that silently goes stale when a field is added.
    assert "fields(TwoStreamSTTModelConfig)" in src, "validated field list must come from the dataclass"
    for field in own - parent:
        assert f'"{field}"' not in src, f"{field!r} looks hardcoded in __init__; derive the list instead"
