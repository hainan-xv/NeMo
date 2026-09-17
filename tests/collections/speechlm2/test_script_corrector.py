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
"""Training examples for the verifier/corrector.

The failure that would be invisible: leaking CHAT's hypothesis into the LOSS.
The hypothesis sits in the prompt, so if it is not masked the model is trained to
predict the ASR's output -- i.e. to imitate the thing it exists to check. Nothing
would error; the model would just quietly learn the wrong task. Most of these
tests exist for that.
"""

import pytest

from nemo.collections.speechlm2.parts.script_corrector import (
    CorrectorIds,
    build_corrector_example,
)

IDS = CorrectorIds()
INSTR = [1, 2, 3]
HIST = [10, 11]
HYP = [20, 21]
IGN = -100


def test_accept_target_is_a_single_token():
    """The efficiency claim rests on this: agreeing costs ONE token."""
    ex = build_corrector_example(INSTR, HIST, 4, HYP, None)
    assert ex.is_accept
    assert ex.input_ids[ex.prompt_len :] == [IDS.accept]
    assert ex.labels[ex.prompt_len :] == [IDS.accept]


def test_correction_target_is_text_then_eot():
    ex = build_corrector_example(INSTR, HIST, 4, HYP, [30, 31])
    assert not ex.is_accept
    assert ex.input_ids[ex.prompt_len :] == [30, 31, IDS.eot]


def test_prompt_is_masked_so_the_hypothesis_never_enters_the_loss():
    """If the hypothesis were supervised, the model would learn to reproduce
    CHAT's output rather than judge it."""
    ex = build_corrector_example(INSTR, HIST, 4, HYP, [30])
    assert all(l == IGN for l in ex.labels[: ex.prompt_len])
    # and specifically: no hypothesis token is ever a label
    assert not any(l in HYP for l in ex.labels), "hypothesis tokens leaked into the loss"


def test_history_is_also_masked():
    ex = build_corrector_example(INSTR, HIST, 4, HYP, None)
    assert not any(l in HIST for l in ex.labels)


def test_prompt_layout_is_exactly_the_documented_order():
    ex = build_corrector_example(INSTR, HIST, 3, HYP, None)
    expected = INSTR + HIST + [IDS.vision_start, 0, 0, 0, IDS.vision_end] + [IDS.hyp_start] + HYP + [IDS.hyp_end]
    assert ex.input_ids[: ex.prompt_len] == expected


def test_audio_window_reserves_exactly_audio_len_positions():
    for n in (0, 1, 14):
        ex = build_corrector_example(INSTR, HIST, n, HYP, None)
        seg = ex.input_ids[: ex.prompt_len]
        lo = seg.index(IDS.vision_start)
        hi = seg.index(IDS.vision_end)
        assert hi - lo - 1 == n, f"audio window wrong for len {n}"


def test_empty_hypothesis_is_representable():
    """A chunk where the ASR emitted nothing is how a DELETION presents, and it
    still has to be checkable."""
    ex = build_corrector_example(INSTR, HIST, 4, [], [30, 31])
    seg = ex.input_ids[: ex.prompt_len]
    assert seg[seg.index(IDS.hyp_start) + 1] == IDS.hyp_end
    assert ex.input_ids[ex.prompt_len :] == [30, 31, IDS.eot]


def test_empty_history_is_representable():
    ex = build_corrector_example(INSTR, [], 4, HYP, None)
    assert ex.input_ids[: len(INSTR)] == INSTR


def test_labels_and_inputs_are_the_same_length():
    for tgt in (None, [30, 31]):
        ex = build_corrector_example(INSTR, HIST, 4, HYP, tgt)
        assert len(ex.labels) == len(ex.input_ids)


def test_accept_token_is_distinct_from_eot():
    """ACCEPT must not collide with the correction terminator, or 'agree' and
    'emit nothing' become the same string and the decision is unrecoverable."""
    assert IDS.accept != IDS.eot


def test_duplicate_special_ids_are_rejected():
    bad = CorrectorIds(accept=151645)  # same as eot
    with pytest.raises(ValueError, match="distinct"):
        bad.validate(vocab_size=151669)


def test_special_ids_outside_the_vocabulary_are_rejected():
    with pytest.raises(ValueError, match="outside vocabulary"):
        CorrectorIds().validate(vocab_size=100)


def test_defaults_are_in_qwen3_range_and_do_not_clash_with_script():
    """These are repurposed EXISTING tokens; colliding with SCRIPT's audio
    delimiters would corrupt the audio window instead of erroring."""
    i = IDS
    assert i.accept not in (i.vision_start, i.vision_end, i.eot)
    assert i.hyp_start not in (i.vision_start, i.vision_end, i.eot, i.accept)
    assert i.hyp_end not in (i.vision_start, i.vision_end, i.eot, i.accept, i.hyp_start)
    i.validate(vocab_size=151669)


# --------------------------------------------------------------------------
# Per-utterance assembly: labels, history advance, timing tolerance.
# --------------------------------------------------------------------------

from nemo.collections.speechlm2.parts.script_corrector import corrector_examples_for_utterance  # noqa: E402


def _utt(hyp_words, hyp_ids, ref_words=None, ref_ids=None):
    ref_words = ref_words or [["a", "b"], ["c", "d"]]
    ref_ids = ref_ids or [[41, 42], [43, 44]]
    return corrector_examples_for_utterance(INSTR, ref_words, ref_ids, hyp_ids, hyp_words, [14, 14])


def test_perfect_hypothesis_yields_all_accept():
    ex = _utt([["a", "b"], ["c", "d"]], [[41, 42], [43, 44]])
    assert [e.is_accept for e in ex] == [True, True]


def test_right_words_wrong_boundaries_still_all_accept():
    """The timing rule must survive into the assembled examples, not just the
    labeller: the ASR was early, and no chunk should be labelled wrong."""
    ex = _utt([["a"], ["b", "c", "d"]], [[41], [42, 43, 44]])
    assert [e.is_accept for e in ex] == [True, True]


def test_a_real_error_produces_a_correction_carrying_reference_ids():
    ex = _utt([["a", "X"], ["c", "d"]], [[41, 99], [43, 44]])
    assert ex[0].is_accept is False and ex[1].is_accept is True
    assert ex[0].input_ids[ex[0].prompt_len :] == [41, 42, IDS.eot], "target must be the REFERENCE ids"


def test_history_advances_on_the_reference_not_the_hypothesis():
    """Chunk 1's prompt must contain the reference for chunk 0, even though the
    ASR got chunk 0 wrong -- that is what inference sees after correction."""
    ex = _utt([["a", "X"], ["c", "d"]], [[41, 99], [43, 44]])
    p1 = ex[1].input_ids[: ex[1].prompt_len]
    assert p1[len(INSTR) : len(INSTR) + 2] == [41, 42], "history should be reference ids 41,42"
    assert 99 not in p1, "the hypothesis's wrong token leaked into the next chunk's history"


def test_first_chunk_has_empty_history():
    ex = _utt([["a", "b"], ["c", "d"]], [[41, 42], [43, 44]])
    p0 = ex[0].input_ids[: ex[0].prompt_len]
    assert p0[: len(INSTR)] == INSTR
    assert p0[len(INSTR)] == IDS.vision_start, "no history should precede the audio on chunk 0"


def test_mismatched_per_chunk_input_lengths_are_rejected():
    with pytest.raises(ValueError, match="disagree on length"):
        corrector_examples_for_utterance(INSTR, [["a"]], [[41]], [[41], [42]], [["a"], ["b"]], [14])


# --------------------------------------------------------------------------
# Collation.
# --------------------------------------------------------------------------

from nemo.collections.speechlm2.parts.script_corrector import (  # noqa: E402
    collate_corrector_examples,
)

PAD = 151643


def test_padding_is_masked_in_labels_not_just_attention():
    """Masking only the attention still trains the model to emit pad -- which
    would surface as an inexplicably high ACCEPT rate, since pad and ACCEPT are
    both 'short output'."""
    short = build_corrector_example(INSTR, HIST, 2, [20], None)
    long = build_corrector_example(INSTR, HIST, 9, [20, 21, 22], [30, 31, 32])
    b = collate_corrector_examples([short, long], PAD)
    w = len(b.input_ids[0])
    assert len(b.input_ids[1]) == w
    npad = w - len(short.input_ids)
    assert b.attention_mask[0][-npad:] == [0] * npad
    assert b.labels[0][-npad:] == [IGN] * npad


def test_audio_slots_point_between_the_delimiters():
    ex = build_corrector_example(INSTR, HIST, 3, HYP, None)
    b = collate_corrector_examples([ex], PAD)
    seq = b.input_ids[0]
    lo, hi = seq.index(IDS.vision_start), seq.index(IDS.vision_end)
    positions = [p for (_r, p, _k) in b.audio_slots]
    assert positions == list(range(lo + 1, hi))
    assert [k for (_r, _p, k) in b.audio_slots] == [0, 1, 2], "frame index must be 0-based per row"


def test_audio_slots_are_per_row():
    a = build_corrector_example(INSTR, HIST, 2, HYP, None)
    c = build_corrector_example(INSTR, HIST, 3, HYP, None)
    b = collate_corrector_examples([a, c], PAD)
    rows = {}
    for r, _p, k in b.audio_slots:
        rows.setdefault(r, []).append(k)
    assert rows[0] == [0, 1] and rows[1] == [0, 1, 2]


def test_zero_length_audio_produces_no_slots():
    ex = build_corrector_example(INSTR, HIST, 0, HYP, None)
    b = collate_corrector_examples([ex], PAD)
    assert b.audio_slots == []


def test_empty_batch_is_rejected():
    with pytest.raises(ValueError, match="no examples"):
        collate_corrector_examples([], PAD)


def test_accept_flags_survive_collation():
    b = collate_corrector_examples(
        [
            build_corrector_example(INSTR, HIST, 2, HYP, None),
            build_corrector_example(INSTR, HIST, 2, HYP, [30]),
        ],
        PAD,
    )
    assert b.is_accept == [True, False]
