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
"""Transcripts must not tokenize to an ORPHAN word-start piece.

The expectations below are not guesses -- each was measured on Qwen3-1.7B's
tokenizer, which is what these models actually use::

    'a  b'   -> ['a', 'G', 'Gb']      orphan
    'a b '   -> ['a', 'Gb', 'G']      orphan
    'a   b'  -> ['a', 'GG', 'Gb']     orphan
    'a b'    -> ['a', 'Gb']           fine
    ' a b'   -> ['Ga', 'Gb']          fine
    'a\\tb'   -> ['a', 'tab+b']        fine
    'a \\n b' -> ['a', 'Gnl', 'Gb']    fine

So the check must fire on repeated SPACES and on trailing whitespace, and must
NOT fire on a leading space, a tab or a newline -- those merge into the
following piece and cost nothing.
"""

import pytest

from nemo.collections.asr.parts.utils.chat_alignment import assert_clean_transcript


@pytest.mark.unit
@pytest.mark.parametrize("text", ["a  b", "At conception,  members deeply researched", "a   b", "x  y  z"])
def test_repeated_spaces_are_rejected(text):
    with pytest.raises(ValueError, match="consecutive spaces"):
        assert_clean_transcript(text)


@pytest.mark.unit
@pytest.mark.parametrize("text", ["a b ", "hello world\t", "done.\n"])
def test_trailing_whitespace_is_rejected(text):
    with pytest.raises(ValueError, match="trailing whitespace"):
        assert_clean_transcript(text)


@pytest.mark.unit
@pytest.mark.parametrize("text", ["a b", " a b", "a\tb", "a \n b", "At conception, members deeply", ""])
def test_harmless_whitespace_is_accepted(text):
    """A leading space, a tab and a newline all merge into the next piece. Being
    stricter than the tokenizer would reject clean data."""
    assert_clean_transcript(text)


@pytest.mark.unit
def test_the_message_locates_the_problem():
    """A bare 'bad transcript' error is useless against a 61,000-shard corpus."""
    with pytest.raises(ValueError) as e:
        assert_clean_transcript("At conception,  members deeply researched", source="cut-42")
    msg = str(e.value)
    assert "cut-42" in msg, "must name the cut so the offending utterance can be found"
    assert "offset" in msg
    assert "conception" in msg, "must quote the surrounding text"


@pytest.mark.unit
def test_it_is_wired_into_the_training_target_path():
    """The check is worthless if it is not on the path that builds targets.
    _chunk_tokens is shared by CHAT training and the corrector's reference
    chunks, so guarding it covers both."""
    import inspect

    from nemo.collections.asr.models.chat_bpe_models import EncDecCHATBPEModel

    src = inspect.getsource(EncDecCHATBPEModel._chunk_tokens)
    assert "clean_transcript" in src


# --------------------------------------------------------------------------
# REPAIR, not raise: the same path serves validation, whose manifest has this.
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,want",
    [
        ("At conception,  members deeply", "At conception, members deeply"),
        ("a  b", "a b"),
        ("a   b", "a b"),
        ("a b ", "a b"),
        ("Six laps later,  Laffite also retired", "Six laps later, Laffite also retired"),
    ],
)
def test_orphaning_whitespace_is_repaired(raw, want):
    from nemo.collections.asr.parts.utils.chat_alignment import clean_transcript

    got, n = clean_transcript(raw)
    assert got == want
    assert n == 1, "the repair must be counted, or a data defect becomes invisible"


@pytest.mark.unit
@pytest.mark.parametrize("text", ["a b", "At conception, members deeply", ""])
def test_clean_text_is_untouched_and_uncounted(text):
    from nemo.collections.asr.parts.utils.chat_alignment import clean_transcript

    got, n = clean_transcript(text)
    assert got == text and n == 0


@pytest.mark.unit
def test_the_training_path_repairs_rather_than_raises():
    """A raise here killed job 18848949 at its first validation epoch: the mcv11
    dev manifest has a double space in 25% of utterances."""
    import inspect

    from nemo.collections.asr.models.chat_bpe_models import EncDecCHATBPEModel

    src = inspect.getsource(EncDecCHATBPEModel._chunk_tokens)
    assert "clean_transcript" in src
    assert "assert_clean_transcript(" not in src, "must not raise on the shared train/val path"
