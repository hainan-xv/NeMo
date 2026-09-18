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
"""_corrected_wer must judge chunks WITH the audio.

It builds its own prompt rather than reusing the training collation, and it
originally embedded token ids alone -- leaving the reserved audio positions at
``embed(0)``, a constant. The decision then carried no acoustic evidence and
collapsed to the same answer for every chunk. Because a rejected chunk emits its
exactly-aligned reference span, rejecting everything reconstructs the reference
and the metric reads 0.00000 -- a perfect score produced by deleting the input.

That is invisible in every other metric: training goes through _prepare, which
splices correctly, so loss and the decision stats stayed honest while
corrected_wer and both wer_delta panels were meaningless.
"""

import types

import pytest
import torch

from nemo.collections.speechlm2.models.script_corrector_model import ScriptCorrectorModel
from nemo.collections.speechlm2.parts.script_corrector import CorrectorIds

IDS = CorrectorIds()
DIM = 4


class _Tok:
    def text_to_ids(self, text):
        return [900 + len(w) for w in text.split()] if text.strip() else []

    def ids_to_text(self, ids):
        return " ".join(f"w{int(i)}" for i in ids)


def _model(captured, accept: bool):
    """A stand-in exposing only what _corrected_wer touches."""

    def embed(ids_t):
        # Distinctive, and NOT what proj() returns, so a missing splice shows up.
        return torch.zeros(ids_t.shape[0], ids_t.shape[1], DIM)

    def llm(inputs_embeds=None, **kw):
        captured.append(inputs_embeds.clone())
        logits = torch.zeros(1, inputs_embeds.shape[1], 152000)
        logits[0, -1, IDS.accept if accept else 12345] = 10.0
        return types.SimpleNamespace(logits=logits)

    return types.SimpleNamespace(
        ids=IDS,
        tokenizer=_Tok(),
        llm=llm,
        _embed_tokens=embed,
        # proj marks audio rows with 7.0 so their presence is unambiguous.
        perception=types.SimpleNamespace(proj=lambda x: torch.full((x.shape[0], DIM), 7.0)),
        chat=types.SimpleNamespace(joint=types.SimpleNamespace(chunk_size=2)),
    )


def _state(enc_fill=1.0):
    enc = torch.full((1, 4, DIM), enc_fill)
    return {
        "device": torch.device("cpu"),
        "n_chunks": [2],
        "enc": enc,
        "enc_len": torch.tensor([4]),
        "hyp_i": [[[11], [12]]],
        "tgt_i": [[[21], [22]]],
        "ref_w": [[["w21"], ["w22"]]],
        "instr": [1, 2],
    }


@pytest.mark.unit
def test_audio_frames_are_spliced_into_the_decision_prompt():
    captured = []
    ScriptCorrectorModel._corrected_wer(_model(captured, accept=True), _state())
    assert captured, "the LLM was never called"
    marked = [int((e == 7.0).all(-1).sum()) for e in captured]
    assert sum(marked) > 0, "no audio rows in the prompt -- the decision is taken on a blanked input"


@pytest.mark.unit
def test_every_chunk_gets_its_own_audio():
    captured = []
    st = _state()
    ScriptCorrectorModel._corrected_wer(_model(captured, accept=True), st)
    assert len(captured) == st["n_chunks"][0]
    for i, e in enumerate(captured):
        assert int((e == 7.0).all(-1).sum()) > 0, f"chunk {i} had no audio spliced in"


@pytest.mark.unit
def test_rejecting_everything_emits_the_reference_spans():
    """The mechanism that turns a blanked decision into a perfect 0.00000."""
    st = _state()
    e, n = ScriptCorrectorModel._corrected_wer(_model([], accept=False), st, return_counts=True)
    assert (e, n) == (0, 2), "rejection emits tgt_i, which by construction IS the reference"


@pytest.mark.unit
def test_accepting_a_wrong_hypothesis_is_not_free():
    """The counterpart: if accepts were being scored against the reference the
    metric would not be able to read 0 while chat_wer_tf was 0.0876."""
    e, n = ScriptCorrectorModel._corrected_wer(_model([], accept=True), _state(), return_counts=True)
    assert e > 0, "accepting a hypothesis that differs from the reference must cost edits"


# --------------------------------------------------------------------------
# The dump's decision must BE the metric's decision, not a second opinion.
# --------------------------------------------------------------------------


def _gen_state(n=3):
    st = _state()
    st["examples"] = [types.SimpleNamespace(prompt_len=2) for _ in range(n)]
    st["embeds"] = torch.zeros(n, 4, DIM)
    return st


@pytest.mark.unit
def test_accepted_chunks_generate_nothing():
    """Generation is the expensive part; an accepted chunk has no correction to
    show, so it must not run at all."""
    captured = []
    gens = ScriptCorrectorModel._sample_predictions(
        _model(captured, accept=False), _gen_state(3), 3, 8, preds=[True, True, True]
    )
    assert gens == [None, None, None]
    assert captured == [], "no forward should run for accepted chunks"


@pytest.mark.unit
def test_rejected_chunks_generate_text_and_respect_the_cap():
    captured = []
    cap = 4
    gens = ScriptCorrectorModel._sample_predictions(
        _model(captured, accept=False), _gen_state(1), 1, cap, preds=[False]
    )
    assert gens[0] is not None
    assert gens[0].endswith("..."), "hitting the cap must be marked, not silently truncated"
    assert len(captured) <= cap, "generation must stop at the cap"


@pytest.mark.unit
def test_decisions_are_not_recomputed():
    """Passing preds in is the whole point: the printed accept/reject has to be
    the same number reject_recall was computed from. A dump reporting every
    chunk rejected while reject_recall read 0.375 is only possible if the two
    disagreed, and the dump was the wrong one."""
    import inspect

    sig = inspect.signature(ScriptCorrectorModel._sample_predictions)
    assert "preds" in sig.parameters, "the decision must be supplied by the caller"
    # It returns generations only -- no second opinion on accept/reject.
    gens = ScriptCorrectorModel._sample_predictions(_model([], accept=True), _gen_state(2), 2, 4, preds=[True, False])
    assert isinstance(gens, list) and gens[0] is None
