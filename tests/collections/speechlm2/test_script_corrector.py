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


# The assembler tokenizes correction targets itself, because a target is now an
# arbitrary aligned span rather than a whole reference chunk.
_VOCAB = {"a": 41, "b": 42, "c": 43, "d": 44, "X": 99, "hello": 45, "world": 46}


def _tok(text):
    return [_VOCAB[w] for w in text.split()]


def _utt(hyp_chunk_words, hyp_ids, ref_words=None):
    """Examples only. The hypothesis arrives GROUPED BY ITS OWN CHUNK, which is
    the index space the labels live in."""
    ref_words = ref_words or [["a", "b"], ["c", "d"]]
    return corrector_examples_for_utterance(INSTR, ref_words, hyp_chunk_words, hyp_ids, [14, 14], _tok)[0]


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
        corrector_examples_for_utterance(INSTR, [["a"]], [["a"], ["b"]], [[41], [42]], [14], _tok)


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


# --------------------------------------------------------------------------
# Metrics.
# --------------------------------------------------------------------------

from nemo.collections.speechlm2.parts.script_corrector import (  # noqa: E402
    decision_stats,
    word_errors,
)


def test_always_accept_scores_zero_reject_recall():
    """The collapse this 93%-ACCEPT corpus invites. Plain accuracy would call it
    93% correct; reject_recall correctly calls it useless."""
    labels = [True] * 9 + [False]
    st = decision_stats([True] * 10, labels)
    assert st["pred_accept_frac"] == 1.0
    assert st["reject_recall"] == 0.0, "it never caught the one real error"


def test_perfect_decisions_score_one():
    labels = [True, False, True, False]
    st = decision_stats(labels, labels)
    assert st["reject_recall"] == 1.0 and st["reject_precision"] == 1.0


def test_over_rejecting_shows_up_as_low_precision():
    st = decision_stats([False] * 4, [True, True, True, False])
    assert st["reject_recall"] == 1.0
    assert st["reject_precision"] == 0.25


def test_decision_stats_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length mismatch"):
        decision_stats([True], [True, False])


def test_word_errors_returns_edits_and_reference_length():
    assert word_errors(["a", "b"], ["a", "b"]) == (0, 2)
    assert word_errors(["a", "X"], ["a", "b"]) == (1, 2)
    assert word_errors(["a"], ["a", "b"]) == (1, 2)
    assert word_errors(["a", "b", "c"], ["a", "b"]) == (1, 2)


def test_wer_accumulates_additively_not_as_a_mean_of_rates():
    """A 1-word utterance fully wrong and a 9-word utterance fully right is 10%
    corpus WER, not the 50% a mean of per-utterance rates would give."""
    e1, n1 = word_errors(["X"], ["a"])
    e2, n2 = word_errors(list("bcdefghij"), list("bcdefghij"))
    assert (e1 + e2) / (n1 + n2) == pytest.approx(0.1)


def test_a_word_straddling_a_chunk_boundary_is_not_two_errors():
    """The bug that measured 41% WER against a true ~8%.

    CHAT emits BPE pieces and may split a word across chunks, so the grouping of
    hypothesis words into chunks cannot come from detokenizing each chunk alone.
    _hyp_chunk_words slices ONE detokenization by prefix word-counts, which keeps
    a straddling word whole and owned by the chunk it started in.
    """
    import types

    from nemo.collections.speechlm2.models.script_corrector_model import ScriptCorrectorModel

    class _Tok:
        """Piece 7 is 'hel', 8 completes it into 'hello' and adds 'world'."""

        def ids_to_text(self, ids):
            return {(7,): "hel", (7, 8): "hello world"}[tuple(ids)]

    per_chunk, flat = ScriptCorrectorModel._hyp_chunk_words(types.SimpleNamespace(tokenizer=_Tok()), [[7], [8]])
    assert flat == ["hello", "world"]
    assert per_chunk == [["hello"], ["world"]], "the straddling word must stay whole, not become 'hel'"
    # The invariant the labeller depends on.
    assert [w for c in per_chunk for w in c] == flat


def test_straddling_words_still_label_as_accept_end_to_end():
    ex = corrector_examples_for_utterance(
        INSTR, [["hello"], ["world"]], [["hello"], ["world"]], [[7], [8]], [14, 14], _tok
    )[0]
    assert [e.is_accept for e in ex] == [True, True]


# --------------------------------------------------------------------------
# Sample printing.
# --------------------------------------------------------------------------

from nemo.collections.speechlm2.parts.script_corrector import format_sample  # noqa: E402


def test_sample_shows_chunk_boundaries_on_both_sides():
    out = format_sample(["the cat", "sat down"], ["the", "cat sat down"], [None, None], step=10)
    assert "ref : the cat | sat down" in out
    assert "hyp : the | cat sat down" in out, "hypothesis boundaries must be visible"
    assert "step 10" in out


def test_sample_marks_correct_and_incorrect_chunks():
    out = format_sample(["a b", "c d"], ["a X", "c d"], ["a b", None])
    assert "<incorrect>" in out and "-> 'a b'" in out
    assert "<correct>" in out


def test_sample_survives_fewer_hypothesis_chunks_than_labels():
    """A hypothesis can end early -- that is how a trailing deletion presents --
    and the dump must still render rather than crash the training run."""
    out = format_sample(["a", "b"], ["a"], [None, "b"])
    assert "chunk 1: <incorrect>" in out


# --------------------------------------------------------------------------
# Rank-uniform logging. A metric that some ranks skip hangs the whole job.
# --------------------------------------------------------------------------


def test_every_decision_stat_is_in_the_fixed_metric_key_list():
    """decision_stats returns {} for an empty batch, and self.log(sync_dist=True)
    is a collective -- so a metric logged on some ranks and not others desyncs
    the group and the job dies ten minutes later in an ALLREDUCE watchdog
    timeout, with healthy step timings and no traceback. That killed
    dfw_corrector_v1 at step ~6654.

    Reading _METRIC_KEYS from source keeps this test free of a GPU/torch import
    chain while still failing if someone adds a stat and forgets the key.
    """
    import pathlib
    import re

    src = pathlib.Path(__file__).parents[3] / "nemo/collections/speechlm2/models/script_corrector_model.py"
    text = src.read_text()
    block = re.search(r"_METRIC_KEYS = \((.*?)\)", text, re.S).group(1)
    keys = set(re.findall(r'"([^"]+)"', block))

    produced = {f"train_{k}" for k in decision_stats([True, False], [True, True])}
    missing = produced - keys
    assert not missing, f"decision_stats emits {missing}, absent from _METRIC_KEYS"


def test_metric_keys_are_unique():
    import pathlib
    import re

    src = pathlib.Path(__file__).parents[3] / "nemo/collections/speechlm2/models/script_corrector_model.py"
    block = re.search(r"_METRIC_KEYS = \((.*?)\)", src.read_text(), re.S).group(1)
    keys = re.findall(r'"([^"]+)"', block)
    assert len(keys) == len(set(keys)), "a duplicated key logs twice on every rank"


def test_corrector_model_defines_the_hooks_it_must_override():
    """Guards against an edit silently deleting a method.

    A span-based rewrite of _chat_hypotheses once removed validation_step and
    both validation hooks along with it. Nothing failed at import: the PARENT's
    validation_step ran instead and died 4 minutes into the job with "'list'
    object has no attribute 'text'", because it expects SCRIPT's batch type.

    Parsed from source so the test needs no torch/GPU import chain.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).parents[3] / "nemo/collections/speechlm2/models/script_corrector_model.py"
    tree = ast.parse(src.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScriptCorrectorModel")
    defined = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    required = {
        "training_step",
        "validation_step",
        "on_validation_epoch_start",
        "on_validation_epoch_end",
        "on_train_batch_end",
        "_prepare",
        "_log_all",
    }
    missing = required - defined
    assert not missing, f"ScriptCorrectorModel is missing {missing}"


# --------------------------------------------------------------------------
# Structural guard.
# --------------------------------------------------------------------------


def test_module_still_defines_every_public_symbol():
    """A span rewrite that replaces one function by matching to the NEXT 'def'
    silently swallows any class sitting between them. That deleted
    CorrectorBatch once, and the only symptom was five unrelated collation tests
    failing. Pin the surface so the next one fails here, loudly.
    """
    import ast as _ast
    import pathlib as _pathlib

    import nemo.collections.speechlm2.parts.script_corrector as _mod

    src = _pathlib.Path(_mod.__file__).read_text()
    top = {n.name for n in _ast.parse(src).body if isinstance(n, (_ast.FunctionDef, _ast.ClassDef))}
    expected = {
        "CorrectorIds",
        "CorrectorExample",
        "CorrectorBatch",
        "build_corrector_example",
        "corrector_examples_for_utterance",
        "collate_corrector_examples",
        "decision_stats",
        "word_errors",
        "format_sample",
    }
    assert expected <= top, f"missing from the module: {sorted(expected - top)}"
