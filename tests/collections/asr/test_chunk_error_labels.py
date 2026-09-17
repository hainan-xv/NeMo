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

The property that matters most is the one that is easiest to get wrong: an ASR
that emits the RIGHT WORDS on DIFFERENT chunk boundaries must be labelled
entirely correct. Per-chunk string comparison fails that, and a corrector
trained on those labels would learn to fight emission timing -- which the banded
losses in this project deliberately leave loose.
"""

from nemo.collections.asr.parts.utils.chunk_error_labels import align_words, label_chunks


def test_perfect_transcript_is_all_accept():
    labels, n = label_chunks(["a", "b", "c", "d"], [["a", "b"], ["c", "d"]])
    assert labels == [None, None] and n == 0


def test_same_words_different_chunk_boundaries_is_all_accept():
    """The headline case: the ASR was early or late, not wrong."""
    # reference splits 1|3, the hypothesis would have split 3|1 -- but the
    # flattened words are identical, so nothing is wrong.
    labels, n = label_chunks(["a", "b", "c", "d"], [["a"], ["b", "c", "d"]])
    assert labels == [None, None] and n == 0
    labels, n = label_chunks(["a", "b", "c", "d"], [["a", "b", "c"], ["d"]])
    assert labels == [None, None] and n == 0


def test_substitution_marks_only_its_own_chunk():
    labels, n = label_chunks(["a", "X", "c", "d"], [["a", "b"], ["c", "d"]])
    assert n == 1
    assert labels[0] == "a b", "the wrong chunk carries the REFERENCE text as its target"
    assert labels[1] is None


def test_deletion_is_attributed_via_the_reference_chunk():
    """A word the ASR never emitted has no hypothesis position -- only a
    reference one. Indexing on the reference is what makes this expressible."""
    labels, n = label_chunks(["a", "c", "d"], [["a", "b"], ["c", "d"]])
    assert n == 1 and labels[0] == "a b" and labels[1] is None


def test_insertion_blames_the_chunk_it_precedes_not_the_one_before():
    """A spurious word before chunk 1's first reference word belongs to chunk 1;
    blaming chunk 0 would mark a chunk the reference says was fine."""
    labels, n = label_chunks(["a", "b", "X", "c", "d"], [["a", "b"], ["c", "d"]])
    assert labels[1] == "c d"
    assert n == 1


def test_trailing_insertion_lands_on_the_final_chunk():
    labels, n = label_chunks(["a", "b", "c", "d", "X"], [["a", "b"], ["c", "d"]])
    assert labels[-1] == "c d" and n == 1


def test_errors_in_several_chunks_are_all_marked():
    labels, n = label_chunks(["X", "b", "c", "Y"], [["a", "b"], ["c", "d"]])
    assert n == 2 and labels == ["a b", "c d"]


def test_empty_hypothesis_marks_every_chunk():
    labels, n = label_chunks([], [["a", "b"], ["c", "d"]])
    assert n == 2 and all(l is not None for l in labels)


def test_alignment_ops_are_what_they_claim():
    ops = [o for o, _, _ in align_words(["a", "X", "c"], ["a", "b", "c"])]
    assert ops == ["equal", "sub", "equal"]
    ops = [o for o, _, _ in align_words(["a", "c"], ["a", "b", "c"])]
    assert ops == ["equal", "del", "equal"]
    ops = [o for o, _, _ in align_words(["a", "b", "c"], ["a", "c"])]
    assert ops == ["equal", "ins", "equal"]


# --------------------------------------------------------------------------
# Normalisation for the label decision.
# --------------------------------------------------------------------------

from nemo.collections.asr.parts.utils.chunk_error_labels import simple_normalize  # noqa: E402


def test_casing_and_punctuation_alone_do_not_mark_a_chunk_wrong():
    """The ASR writes 'Hello, world.'; the aligner has 'hello world'. Compared
    literally that is two errors -- measured, this inflated the wrong-chunk rate
    from 7% to 75%."""
    labels, n = label_chunks(
        ["Hello,", "world."], [["hello"], ["world"]], normalize=simple_normalize
    )
    assert n == 0 and labels == [None, None]


def test_without_a_normalizer_the_same_input_looks_entirely_wrong():
    """Pins the default as exact-match, so the normalisation is a visible choice."""
    _, n = label_chunks(["Hello,", "world."], [["hello"], ["world"]])
    assert n == 2


def test_a_real_error_still_shows_through_normalisation():
    labels, n = label_chunks(
        ["Hello,", "word."], [["hello"], ["world"]], normalize=simple_normalize
    )
    assert n == 1 and labels[1] == "world"


def test_correction_target_keeps_the_original_reference_text():
    """Normalisation decides IF a chunk is wrong; the target must stay the true
    reference, casing and punctuation included, since that is what to emit."""
    labels, _ = label_chunks(["X"], [["Hello,"]], normalize=simple_normalize)
    assert labels[0] == "Hello,", "target must not be normalised"


def test_simple_normalize_strips_edge_punctuation_only():
    assert simple_normalize("Hello,") == "hello"
    assert simple_normalize('"world."') == "world"
    assert simple_normalize("don't") == "don't", "internal apostrophes must survive"


def test_standalone_punctuation_in_the_reference_is_not_an_error():
    """CHAT trains on the punctuated transcript, so a chunk can legitimately
    start with a bare '.'. split() makes it a word and simple_normalize maps it
    to '' -- which matches nothing, so the chunk scored wrong even when the
    hypothesis was perfect. Measured: accept rate 0.93 -> 0.77."""
    labels, n = label_chunks(
        ["And", "my", "cousin"], [[".", "And", "my", "cousin"]], normalize=simple_normalize
    )
    assert n == 0 and labels == [None]


def test_punctuation_only_chunk_is_accepted_when_hypothesis_omits_it():
    labels, n = label_chunks(["a", "b"], [["a"], [","], ["b"]], normalize=simple_normalize)
    assert n == 0, "a chunk that is pure punctuation cannot be got wrong"


def test_real_errors_still_detected_alongside_punctuation():
    labels, n = label_chunks(
        ["And", "my", "COUSIN_X"], [[".", "And", "my", "cousin"]], normalize=simple_normalize
    )
    assert n == 1
