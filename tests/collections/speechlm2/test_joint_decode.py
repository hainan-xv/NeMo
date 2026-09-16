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
reproduce CHAT decoding exactly, at lam=0 SCRIPT decoding exactly. A fusion
wrong in between would have to be wrong while agreeing with both endpoints,
which the scripted scorers below make hard.

The scorers are stubs on purpose. What can break here is the fusion arithmetic
and the END handling -- neither needs a real model, and both would be invisible
inside a GPU integration test.
"""

import math

import pytest

from nemo.collections.speechlm2.parts.joint_decode import chunk_sync_joint_decode

V = 5  # tiny vocabulary; index V is END


class ScriptedScorer:
    """Returns a per-(chunk, position) distribution written out by the test.

    State is ``(chunk, position)``. ``table[chunk][pos]`` maps token -> weight;
    unlisted tokens get a small floor rather than -inf, matching real models,
    which never hard-zero a token.
    """

    FLOOR = 1e-6

    def __init__(self, table):
        self.table = table
        self.calls = []

    def init_state(self):
        return (0, 0)

    def logprobs(self, state, chunk_idx):
        _, pos = state
        self.calls.append((chunk_idx, pos))
        row = self.table.get(chunk_idx, {}).get(pos, {})
        return [math.log(row.get(i, self.FLOOR)) for i in range(V + 1)]

    def advance(self, state, token):
        c, pos = state
        return (c, pos + 1)

    def close_chunk(self, state, chunk_idx):
        return (chunk_idx + 1, 0)


def _only(tok):
    return {tok: 1.0}


def test_lam_one_reproduces_chat_alone():
    chat = ScriptedScorer({0: {0: _only(1), 1: _only(2), 2: _only(V)}})
    script = ScriptedScorer({0: {0: _only(3), 1: _only(4), 2: _only(V)}})
    assert chunk_sync_joint_decode(chat, script, 1, V, lam=1.0) == [1, 2]


def test_lam_zero_reproduces_script_alone():
    chat = ScriptedScorer({0: {0: _only(1), 1: _only(2), 2: _only(V)}})
    script = ScriptedScorer({0: {0: _only(3), 1: _only(4), 2: _only(V)}})
    assert chunk_sync_joint_decode(chat, script, 1, V, lam=0.0) == [3, 4]


def test_fusion_picks_the_token_both_models_like():
    """The point of the ensemble: a shared second choice beats two disagreeing firsts."""
    chat = ScriptedScorer({0: {0: {1: 0.6, 2: 0.4}, 1: _only(V)}})
    script = ScriptedScorer({0: {0: {3: 0.6, 2: 0.4}, 1: _only(V)}})
    assert chunk_sync_joint_decode(chat, script, 1, V, lam=0.5) == [2]


def test_both_models_vote_on_where_the_chunk_ends():
    """END is scored like any other symbol, so SCRIPT can end a chunk CHAT would
    have continued. This is the behaviour the chunk-boundary convention mismatch
    actually surfaces as -- a soft disagreement, resolved by the weighted sum."""
    # CHAT mildly prefers another token; SCRIPT strongly wants to stop.
    chat = ScriptedScorer({0: {0: {1: 0.6, V: 0.4}}, 1: {0: _only(V)}})
    script = ScriptedScorer({0: {0: {1: 0.05, V: 0.95}}, 1: {0: _only(V)}})
    assert chunk_sync_joint_decode(chat, script, 2, V, lam=0.5) == []
    # ...and with CHAT dominant the same step emits the token instead.
    chat2 = ScriptedScorer({0: {0: {1: 0.6, V: 0.4}, 1: _only(V)}, 1: {0: _only(V)}})
    script2 = ScriptedScorer({0: {0: {1: 0.05, V: 0.95}, 1: _only(V)}, 1: {0: _only(V)}})
    assert chunk_sync_joint_decode(chat2, script2, 2, V, lam=1.0) == [1]


def test_end_is_a_slot_not_a_token():
    """If END leaked into the output the hypothesis would contain V."""
    tbl = {0: {0: _only(1), 1: _only(V)}, 1: {0: _only(2), 1: _only(V)}}
    out = chunk_sync_joint_decode(ScriptedScorer(tbl), ScriptedScorer(dict(tbl)), 2, V, lam=0.5)
    assert out == [1, 2] and V not in out


def test_chunks_are_decoded_in_order_and_concatenated():
    tbl = {c: {0: _only(c + 1), 1: _only(V)} for c in range(3)}
    out = chunk_sync_joint_decode(ScriptedScorer(tbl), ScriptedScorer(dict(tbl)), 3, V, lam=0.5)
    assert out == [1, 2, 3]


def test_both_models_are_asked_about_the_same_chunk_and_history():
    """The premise of the whole method: identical (t, h) on both sides."""
    tbl = {c: {0: _only(1), 1: _only(V)} for c in range(3)}
    chat, script = ScriptedScorer(tbl), ScriptedScorer(dict(tbl))
    chunk_sync_joint_decode(chat, script, 3, V, lam=0.5)
    assert chat.calls == script.calls, "models were queried at different (chunk, position)"


def test_max_tokens_per_chunk_stops_a_chunk_that_never_ends():
    never = {0: {p: _only(1) for p in range(50)}}
    out = chunk_sync_joint_decode(
        ScriptedScorer(never), ScriptedScorer(dict(never)), 1, V, lam=0.5, max_tokens_per_chunk=4
    )
    assert out == [1, 1, 1, 1]


def test_rejects_scorers_that_disagree_on_vocab_size():
    class Wrong(ScriptedScorer):
        def logprobs(self, state, chunk_idx):
            return [0.0] * (V + 2)

    with pytest.raises(ValueError, match="vocab mismatch"):
        chunk_sync_joint_decode(ScriptedScorer({0: {0: _only(V)}}), Wrong({}), 1, V, lam=0.5)


def test_lam_outside_unit_interval_is_rejected():
    s = ScriptedScorer({0: {0: _only(V)}})
    with pytest.raises(ValueError, match="lam must be"):
        chunk_sync_joint_decode(s, s, 1, V, lam=1.5)


def test_zero_chunks_is_empty_not_an_error():
    s = ScriptedScorer({})
    assert chunk_sync_joint_decode(s, s, 0, V, lam=0.5) == []
