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
"""Decoding a non-causal encoder must not borrow the streaming context.

A cache-aware arm decodes at ``att_context_size = [left, chunk-1]``. A
full-context arm supports exactly ``[-1, -1]`` -- it attends to the whole
utterance, so there is no look-ahead to pick. Forcing the streaming value on it
asks for a context it never trained with, which is why job 18855869 refused.

The refusal was RIGHT; what was missing was a way to say "decode this one
offline". These tests pin both directions so neither silently regresses.
"""

import importlib.util
import pathlib
import types

import pytest

_SRC = pathlib.Path(__file__).parents[3] / "scripts" / "nemotron_leaderboard_eval.py"
_spec = importlib.util.spec_from_file_location("nemotron_leaderboard_eval", _SRC)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _model(supported, default_left=70):
    enc = types.SimpleNamespace(att_context_size_all=supported, att_context_size=[default_left, 13])
    return types.SimpleNamespace(encoder=enc)


@pytest.mark.unit
def test_streaming_model_resolves_the_chunk_context():
    m = _model([[70, 13], [70, 6]])
    assert _mod.resolve_att_context(m, chunk_size=14) == [70, 13]


@pytest.mark.unit
def test_full_context_model_without_the_flag_explains_itself():
    """The old message said only 'not one this model was trained for', which does
    not tell you the model is non-causal or what to do about it."""
    m = _model([[-1, -1]], default_left=-1)
    with pytest.raises(ValueError, match="FULL-CONTEXT"):
        _mod.resolve_att_context(m, chunk_size=14)
    with pytest.raises(ValueError, match="--full_context"):
        _mod.resolve_att_context(m, chunk_size=14)


@pytest.mark.unit
def test_full_context_flag_keeps_the_whole_utterance():
    m = _model([[-1, -1]], default_left=-1)
    assert _mod.resolve_att_context(m, chunk_size=14, full_context=True) == [-1, -1]


@pytest.mark.unit
def test_full_context_flag_is_refused_on_a_streaming_model():
    """Otherwise it would report an offline number for a streaming arm and look
    like a large unexplained win."""
    m = _model([[70, 13], [70, 6]])
    with pytest.raises(ValueError, match="cache-aware streaming"):
        _mod.resolve_att_context(m, chunk_size=14, full_context=True)


@pytest.mark.unit
def test_results_are_labelled_offline_on_disk():
    """A full-context number must never be filed where a streaming one would be."""
    sh = (pathlib.Path(__file__).parents[3] / "launch" / "eval_nemotron.sh").read_text()
    assert "_fullctx" in sh, "the results directory must record that this was offline"
    assert "--full_context" in sh


@pytest.mark.unit
def test_the_fullctx_launcher_requests_it():
    sh = (pathlib.Path(__file__).parents[3] / "launch" / "dfw_eval_chat_fullctx_parakeet.sh").read_text()
    assert "FULL_CONTEXT=1" in sh
