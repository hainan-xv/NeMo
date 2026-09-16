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
"""The CHAT/SCRIPT -> ChunkScorer translation layer.

This is where a wrong answer is SILENT. The fusion adds two vectors
elementwise, so if slot i means a different token in each, decoding still runs
and still produces text -- just worse text, with no error anywhere. These tests
pin the meaning of every slot rather than the plumbing around it.

Real models are not needed and would hide the bug: what matters is the index
arithmetic, which is identical at any scale.
"""

import math

import pytest
import torch

from nemo.collections.speechlm2.parts.joint_decode_adapters import (
    NEG_INF,
    ScriptChunkScorer,
    remap_script_logprobs,
)

V = 10  # text vocabulary
EOT = 7  # inside the text range, like the real <|im_end|> (151645 < 151669)
VS, VE = 8, 9  # audio delimiters, also in-vocab


def _lp(vals):
    """Turn raw scores into a normalised log-prob vector."""
    return torch.tensor(vals, dtype=torch.float32).log_softmax(-1)


def test_end_is_moved_from_eot_not_copied():
    """END must appear in slot V *and* be gone from its own index.

    Leaving it in place would let the same event be scored twice -- once as text
    and once as chunk-end -- which biases every chunk boundary.
    """
    lp = _lp([1.0] * V)
    out = remap_script_logprobs(lp, V, EOT, (VS, VE))
    assert out[V].item() == pytest.approx(lp[EOT].item(), abs=1e-5), "eot must reach slot V"
    assert out[EOT].item() <= NEG_INF, "eot must be vetoed at its own index"


def test_audio_delimiters_can_never_be_emitted_as_text():
    lp = _lp([0.0] * V)
    out = remap_script_logprobs(lp, V, EOT, (VS, VE))
    assert out[VS].item() <= NEG_INF
    assert out[VE].item() <= NEG_INF


def test_ordinary_text_tokens_pass_through_unchanged():
    lp = _lp([float(i) for i in range(V)])
    out = remap_script_logprobs(lp, V, EOT, (VS, VE))
    for i in range(V):
        if i in (EOT, VS, VE):
            continue
        assert out[i].item() == pytest.approx(lp[i].item(), abs=1e-6)


def test_llm_vocab_wider_than_text_vocab_is_truncated():
    """Qwen3 pads its embedding rows past the tokenizer; those rows are untrained.

    If they survived the remap they would compete with real tokens on equal
    terms, and the fusion has no way to tell them apart.
    """
    wide = _lp([0.0] * (V + 6))
    out = remap_script_logprobs(wide, V, EOT, (VS, VE))
    assert out.numel() == V + 1, "output must be exactly V+1 regardless of LM width"


def test_rejects_an_lm_narrower_than_the_text_vocab():
    with pytest.raises(ValueError, match="smaller than text vocab"):
        remap_script_logprobs(_lp([0.0] * (V - 2)), V, EOT, ())


def test_rejects_a_batched_distribution():
    """A [1, V] slice is the easy mistake; it must fail loudly, not broadcast."""
    with pytest.raises(ValueError, match="1-D"):
        remap_script_logprobs(torch.zeros(1, V), V, EOT, ())


# --------------------------------------------------------------------------
# ScriptChunkScorer state machine, with a fake model.
# --------------------------------------------------------------------------


class _FakeLLM:
    def __init__(self, vocab):
        self.vocab = vocab
        self.seen_lengths = []

    def __call__(self, inputs_embeds=None, **kw):
        self.seen_lengths.append(inputs_embeds.shape[1])
        b, t, _ = inputs_embeds.shape

        class _Out:
            pass

        o = _Out()
        o.logits = torch.zeros(b, t, self.vocab)
        return o


class _FakeScript:
    def __init__(self):
        self.llm = _FakeLLM(V + 4)
        self._eot_id, self._vision_start_id, self._vision_end_id = EOT, VS, VE
        self.d = 3

    def _embed_tokens(self, ids):
        return torch.zeros(ids.shape[0], ids.shape[1], self.d)


def _scorer(frames_t=28, chunk=14):
    m = _FakeScript()
    return ScriptChunkScorer(m, torch.zeros(frames_t, m.d), [1, 2, 3], chunk, V), m


def test_history_is_committed_only_at_the_chunk_boundary():
    """Within a chunk the tokens are provisional; a beam that dies must not
    have polluted the history the next chunk conditions on."""
    s, _ = _scorer()
    st = s.init_state()
    st = s.advance(st, 4)
    st = s.advance(st, 5)
    assert st == ((), (4, 5)), "tokens stay in the chunk buffer until close"
    st = s.close_chunk(st, 0)
    assert st == ((4, 5), ()), "close_chunk moves them into history and clears the buffer"


def test_prompt_grows_by_exactly_one_position_per_emitted_token():
    """instruction + history + <vs> + audio + <ve> + emitted."""
    s, m = _scorer(frames_t=28, chunk=14)
    st = s.init_state()
    s.logprobs(st, 0)
    base = m.llm.seen_lengths[-1]
    assert base == 3 + 0 + 1 + 14 + 1, f"unexpected prompt layout: {base}"
    s.logprobs(s.advance(st, 4), 0)
    assert m.llm.seen_lengths[-1] == base + 1


def test_last_chunk_may_be_short_and_is_not_padded():
    """Audio rarely divides evenly; the tail chunk must shrink, not run past."""
    s, m = _scorer(frames_t=20, chunk=14)  # chunk 1 holds only 6 frames
    s.logprobs(s.init_state(), 1)
    assert m.llm.seen_lengths[-1] == 3 + 1 + 6 + 1


def test_scorer_output_is_the_canonical_width():
    s, _ = _scorer()
    out = s.logprobs(s.init_state(), 0)
    assert out.shape == (V + 1,)
    assert out[VS].item() <= NEG_INF and out[VE].item() <= NEG_INF
