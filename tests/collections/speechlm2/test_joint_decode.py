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
"""Chunk-synchronous joint decoding, pinned at the ends of its range.

The fusion is only meaningful if it DEGENERATES correctly: at lam=1 it must
reproduce CHAT decoding exactly and at lam=0 it must reproduce SCRIPT decoding
exactly. A fusion that is wrong in between would have to be wrong while
agreeing with both endpoints, which the scripted scorers below make hard.

The scorers are stubs on purpose. What can actually break here is the fusion
arithmetic, the END translation and the skew bookkeeping -- none of which needs
a real model, and all of which would be invisible inside a GPU integration test.
"""

import math

import pytest

from nemo.collections.speechlm2.parts.joint_decode import NEG_INF, chunk_sync_joint_decode

V = 5  # tiny vocabulary; index V is END


class ScriptedScorer:
    """Returns a per-(chunk, position) distribution written out by the test.

    State is ``(chunk_offset, tokens_emitted_in_chunk)``. ``table[chunk][pos]``
    maps token -> probability; anything unlisted gets NEG_INF, which also
    exercises the veto path in the fusion.
    """

    def __init__(self, table, name="stub"):
        self.table = table
        self.name = name
        self.calls = []

    def init_state(self):
        return (0, 0)

    def logprobs(self, state, chunk_idx):
        _, pos = state
        self.calls.append((chunk_idx, pos))
        row = self.table.get(chunk_idx, {}).get(pos, {})
        out = [NEG_INF] * (V + 1)
        for tok, p in row.items():
            out[tok] = math.log(p)
        return out

    def advance(self, state, token):
        c, pos = state
        return (c, pos + 1)

    def close_chunk(self, state, chunk_idx):
        return (chunk_idx + 1, 0)


def _only(tok):
    """A distribution that can only produce ``tok``."""
    return {tok: 1.0}


def test_lam_one_reproduces_chat_alone():
    # CHAT wants [1, 2]; SCRIPT wants [3, 4]. At lam=1 SCRIPT must not matter.
    chat = ScriptedScorer({0: {0: _only(1), 1: _only(2), 2: _only(V)}})
    script = ScriptedScorer({0: {0: _only(3), 1: _only(4), 2: _only(V)}})
    out = chunk_sync_joint_decode(chat, script, num_chunks=1, vocab_size=V, lam=1.0, allow_skew=False)
    assert out == [1, 2]


def test_lam_zero_reproduces_script_alone():
    chat = ScriptedScorer({0: {0: _only(1), 1: _only(2), 2: _only(V)}})
    script = ScriptedScorer({0: {0: _only(3), 1: _only(4), 2: _only(V)}})
    out = chunk_sync_joint_decode(chat, script, num_chunks=1, vocab_size=V, lam=0.0, allow_skew=False)
    assert out == [3, 4]


def test_fusion_picks_the_token_both_models_like():
    """The point of the ensemble: a shared second choice beats two disagreeing firsts."""
    chat = ScriptedScorer({0: {0: {1: 0.6, 2: 0.4}, 1: _only(V)}})
    script = ScriptedScorer({0: {0: {3: 0.6, 2: 0.4}, 1: _only(V)}})
    out = chunk_sync_joint_decode(chat, script, num_chunks=1, vocab_size=V, lam=0.5, allow_skew=False)
    # 2 is neither model's argmax but is the only token both admit with mass.
    assert out == [2]


def test_a_token_vetoed_by_one_model_cannot_be_emitted():
    """Structural impossibilities must survive fusion, not be outvoted."""
    chat = ScriptedScorer({0: {0: {1: 1.0}, 1: _only(V)}})  # only 1
    script = ScriptedScorer({0: {0: {2: 1.0}, 1: _only(V)}})  # only 2
    # Neither token is admitted by both, so the only survivable path is END.
    out = chunk_sync_joint_decode(chat, script, num_chunks=1, vocab_size=V, lam=0.5, allow_skew=False)
    assert out == []


def test_end_is_translated_not_taken_literally():
    """END is a slot, not a shared token id.

    Each scorer reports its own terminator in slot V; the search must treat that
    slot as 'chunk over' rather than as vocabulary. If END leaked into the output
    the hypothesis would contain V.
    """
    chat = ScriptedScorer({0: {0: _only(1), 1: _only(V)}, 1: {0: _only(2), 1: _only(V)}})
    script = ScriptedScorer({0: {0: _only(1), 1: _only(V)}, 1: {0: _only(2), 1: _only(V)}})
    out = chunk_sync_joint_decode(chat, script, num_chunks=2, vocab_size=V, lam=0.5, allow_skew=False)
    assert out == [1, 2]
    assert V not in out


def test_chunks_are_decoded_in_order_and_concatenated():
    chat = ScriptedScorer({c: {0: _only(c + 1), 1: _only(V)} for c in range(3)})
    script = ScriptedScorer({c: {0: _only(c + 1), 1: _only(V)} for c in range(3)})
    out = chunk_sync_joint_decode(chat, script, num_chunks=3, vocab_size=V, lam=0.5, allow_skew=False)
    assert out == [1, 2, 3]


def test_skew_recovers_a_word_script_places_one_chunk_earlier():
    """The boundary case the whole skew mechanism exists for.

    CHAT puts the word in chunk 1 (its rule rounds a boundary word later).
    SCRIPT puts it in chunk 0. Without skew the two never agree on any chunk and
    fusion vetoes the word entirely; with skew, SCRIPT is read one chunk BEHIND
    (t-1) so the pair line up on the same word.
    """
    chat = ScriptedScorer({0: {0: _only(V)}, 1: {0: _only(7 % V), 1: _only(V)}})
    script = ScriptedScorer({0: {0: _only(7 % V), 1: _only(V)}, 1: {0: _only(V)}})

    without = chunk_sync_joint_decode(chat, script, num_chunks=2, vocab_size=V, lam=0.5, allow_skew=False)
    assert without == [], "without skew the disagreement should veto the word"

    with_skew = chunk_sync_joint_decode(chat, script, num_chunks=2, vocab_size=V, lam=0.5, allow_skew=True)
    assert with_skew == [7 % V], "skew should let the models agree on the boundary word"


def test_skew_penalty_biases_toward_the_aligned_reading():
    """When both readings are viable, the penalty should decide."""
    tbl = {c: {0: _only(1), 1: _only(V)} for c in range(3)}
    chat, script = ScriptedScorer(tbl), ScriptedScorer(dict(tbl))
    # A large penalty must not break decoding, just keep skew at 0.
    out = chunk_sync_joint_decode(
        chat, script, num_chunks=2, vocab_size=V, lam=0.5, allow_skew=True, skew_penalty=100.0
    )
    assert out == [1, 1]


def test_max_tokens_per_chunk_stops_a_model_that_never_ends():
    """A degenerate hypothesis must terminate the chunk rather than hang."""
    never_ends = ScriptedScorer({0: {p: _only(1) for p in range(50)}})
    out = chunk_sync_joint_decode(
        never_ends,
        ScriptedScorer({0: {p: _only(1) for p in range(50)}}),
        num_chunks=1,
        vocab_size=V,
        lam=0.5,
        max_tokens_per_chunk=4,
        allow_skew=False,
    )
    assert out == [1, 1, 1, 1]


def test_rejects_scorers_that_disagree_on_vocab_size():
    class Wrong(ScriptedScorer):
        def logprobs(self, state, chunk_idx):
            return [0.0] * (V + 2)

    with pytest.raises(ValueError, match="vocab mismatch"):
        chunk_sync_joint_decode(ScriptedScorer({0: {0: _only(V)}}), Wrong({}), num_chunks=1, vocab_size=V, lam=0.5)


def test_lam_outside_unit_interval_is_rejected():
    s = ScriptedScorer({0: {0: _only(V)}})
    with pytest.raises(ValueError, match="lam must be"):
        chunk_sync_joint_decode(s, s, num_chunks=1, vocab_size=V, lam=1.5)


def test_zero_chunks_is_empty_not_an_error():
    s = ScriptedScorer({})
    assert chunk_sync_joint_decode(s, s, num_chunks=0, vocab_size=V, lam=0.5) == []
