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
"""Chunk-level ACCEPT/CORRECT labels for training a verifier.

Two properties matter most, and both are easy to get wrong:

1. An ASR that emits the RIGHT WORDS on DIFFERENT chunk boundaries must be
   labelled entirely correct. Per-chunk string comparison fails that, and a
   corrector trained on those labels would fight emission timing -- which the
   banded losses in this project deliberately leave loose.

2. Labels must be indexed on the HYPOTHESIS partition, because that is the span
   the corrector is handed at inference. Indexing on the reference looks
   equivalent and is not; see the regression test below, taken from a real
   training dump.
"""

import pytest

from nemo.collections.asr.parts.utils.chunk_error_labels import align_words, label_chunks, simple_normalize


def _stitch(hyp_chunks, labels, owned):
    """What the corrector would emit: hypothesis where accepted, reference where not."""
    out = []
    for k in range(len(hyp_chunks)):
        out += list(hyp_chunks[k]) if labels[k] is None else list(owned[k])
    return out


@pytest.mark.unit
def test_perfect_transcript_is_all_accept():
    labels, n, owned = label_chunks([["a", "b"], ["c", "d"]], [["a", "b"], ["c", "d"]])
    assert labels == [None, None] and n == 0
    assert owned == [["a", "b"], ["c", "d"]]


@pytest.mark.unit
def test_same_words_different_chunk_boundaries_is_all_accept():
    """The headline case: the ASR was early or late, not wrong."""
    labels, n, _ = label_chunks([["a"], ["b", "c", "d"]], [["a", "b"], ["c", "d"]])
    assert labels == [None, None] and n == 0
    labels, n, _ = label_chunks([["a", "b", "c"], ["d"]], [["a", "b"], ["c", "d"]])
    assert labels == [None, None] and n == 0


@pytest.mark.unit
def test_substitution_marks_the_hypothesis_chunk_holding_it():
    labels, n, owned = label_chunks([["a", "X"], ["c", "d"]], [["a", "b"], ["c", "d"]])
    assert n == 1
    assert labels[0] == "a b", "the wrong chunk carries the REFERENCE text as its target"
    assert labels[1] is None
    assert _stitch([["a", "X"], ["c", "d"]], labels, owned) == ["a", "b", "c", "d"]


@pytest.mark.unit
def test_lagging_emission_marks_the_chunk_that_actually_holds_the_error():
    """REGRESSION, from a real dump (2026-09-17 11:00:16).

    CHAT emitted 'real time' for the reference's 'real-time', one word late
    throughout. Labelling on the reference partition put <incorrect> on the chunk
    holding ' kernel to provide' -- correct text -- and marked the chunk holding
    ' real time filtering and' ACCEPT. Applying that target deleted 'kernel' and
    duplicated 'real time', so the correction RAISED WER.
    """
    ref = [["operating", "system", "kernel"], ["to", "provide", "real-time"], ["filtering", "and"]]
    hyp = [["operating", "system"], ["kernel", "to", "provide"], ["real", "time", "filtering", "and"]]
    labels, n, owned = label_chunks(hyp, ref, normalize=simple_normalize)

    assert labels[1] is None, "the chunk holding 'kernel to provide' is CORRECT and must be accepted"
    assert labels[2] == "real-time filtering and", "the chunk actually holding 'real time' carries the fix"
    assert n == 1

    # The whole point: stitching loses nothing and duplicates nothing.
    assert _stitch(hyp, labels, owned) == [
        "operating",
        "system",
        "kernel",
        "to",
        "provide",
        "real-time",
        "filtering",
        "and",
    ]


@pytest.mark.unit
def test_deletion_lands_on_the_chunk_it_would_have_been_read_into():
    """A word the ASR never emitted has no hypothesis position, so it is charged
    to the chunk owning the next aligned hypothesis word."""
    labels, n, owned = label_chunks([["a"], ["c", "d"]], [["a", "b"], ["c", "d"]])
    assert n == 1
    assert labels[0] is None, "chunk 0 emitted 'a' correctly"
    assert labels[1] == "b c d", "'b' is charged to the chunk that was about to be read"
    assert _stitch([["a"], ["c", "d"]], labels, owned) == ["a", "b", "c", "d"]


@pytest.mark.unit
def test_trailing_deletion_lands_on_the_final_chunk():
    labels, n, owned = label_chunks([["a", "b"], ["c"]], [["a", "b"], ["c", "d"]])
    assert n == 1 and labels[1] == "c d"
    assert _stitch([["a", "b"], ["c"]], labels, owned) == ["a", "b", "c", "d"]


@pytest.mark.unit
def test_insertion_marks_the_chunk_that_emitted_it():
    labels, n, owned = label_chunks([["a", "b", "X"], ["c", "d"]], [["a", "b"], ["c", "d"]])
    assert n == 1 and labels[0] == "a b"
    assert _stitch([["a", "b", "X"], ["c", "d"]], labels, owned) == ["a", "b", "c", "d"]


@pytest.mark.unit
def test_a_chunk_of_pure_insertion_is_corrected_to_nothing():
    """A chunk that owns no reference word must be rewritten to the empty
    string; leaving it ACCEPT would keep the spurious words."""
    labels, n, owned = label_chunks([["a", "b"], ["X"], ["c", "d"]], [["a", "b"], ["c", "d"]])
    assert labels[1] == ""
    assert owned[1] == []
    assert _stitch([["a", "b"], ["X"], ["c", "d"]], labels, owned) == ["a", "b", "c", "d"]


@pytest.mark.unit
def test_errors_in_several_chunks_are_all_marked():
    labels, n, _ = label_chunks([["X", "b"], ["c", "Y"]], [["a", "b"], ["c", "d"]])
    assert n == 2 and labels[0] == "a b" and labels[1] == "c d"


@pytest.mark.unit
def test_empty_hypothesis_marks_the_final_chunk_with_everything():
    labels, n, owned = label_chunks([[], []], [["a", "b"], ["c", "d"]])
    assert n == 1
    assert labels[-1] == "a b c d", "with no hypothesis position anywhere, all deletions are trailing"
    assert _stitch([[], []], labels, owned) == ["a", "b", "c", "d"]


@pytest.mark.unit
def test_casing_and_punctuation_alone_do_not_mark_a_chunk_wrong():
    labels, n, _ = label_chunks(
        [["Hello,", "world."], ["how", "are", "you?"]],
        [["hello", "world"], ["how", "are", "you"]],
        normalize=simple_normalize,
    )
    assert labels == [None, None] and n == 0


@pytest.mark.unit
def test_without_a_normalizer_the_same_input_looks_entirely_wrong():
    labels, n, _ = label_chunks(
        [["Hello,", "world."], ["how", "are", "you?"]],
        [["hello", "world"], ["how", "are", "you"]],
    )
    assert n > 0, "this is why simple_normalize exists"


@pytest.mark.unit
def test_a_real_error_still_shows_through_normalisation():
    labels, n, _ = label_chunks(
        [["Hello,", "planet."], ["how", "are", "you?"]],
        [["hello", "world"], ["how", "are", "you"]],
        normalize=simple_normalize,
    )
    assert n == 1 and labels[0] == "hello world"


@pytest.mark.unit
def test_correction_target_keeps_the_original_reference_text():
    """The DECISION is normalised; the TARGET is the true reference, casing and
    punctuation included, because that is what the model must emit."""
    labels, _, _ = label_chunks([["wrong", "words"], ["c"]], [["Hello,", "World!"], ["c"]], normalize=simple_normalize)
    assert labels[0] == "Hello, World!"


@pytest.mark.unit
def test_standalone_punctuation_in_the_reference_is_not_an_error():
    """CHAT is trained on punctuated text, so a reference chunk can begin with a
    bare ',' that normalises to nothing. Counting it as an unmatched word pushed
    the accept rate from ~0.93 to ~0.77."""
    labels, n, _ = label_chunks(
        [["stack"], ["or", "in", "the", "case"]],
        [["stack"], [",", "or", "in", "the", "case"]],
        normalize=simple_normalize,
    )
    assert labels == [None, None] and n == 0


@pytest.mark.unit
def test_alignment_ops_are_the_expected_four():
    ops = {op for op, _, _ in align_words(["a", "X", "c"], ["a", "b", "c"])}
    assert ops <= {"equal", "sub", "ins", "del"}
