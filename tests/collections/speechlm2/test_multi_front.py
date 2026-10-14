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
"""Multi-front (self-correcting) decode: serialisation and decode-loop contract."""

import pytest

from nemo.collections.speechlm2.parts.multi_front import (
    TEMP_END,
    TEMP_START,
    Block,
    final_text,
    multi_front_decode,
    parse,
    provisional_text,
    revision_stats,
    serialize,
)

# The worked example: 14 frames = 7 fast chunks + 1 slow chunk, fast says "fax".
ONE_BLOCK = "<temp_start>the quick brown fax<temp_end>the quick brown fox"
THREE_BLOCKS = (
    "<temp_start>the quick brown fax<temp_end>the quick brown fox"
    "<temp_start>jumps ovr<temp_end>jumps over"
    "<temp_start>the lazy dog<temp_end>the lazy dog"
)


# ---------------------------------------------------------------- serialisation


def test_serialize_matches_the_worked_example():
    assert serialize([Block("the quick brown fax", "the quick brown fox")]) == ONE_BLOCK


@pytest.mark.parametrize("s", [ONE_BLOCK, THREE_BLOCKS, ""])
def test_round_trip(s):
    assert serialize(parse(s)) == s


def test_parse_recovers_both_spans():
    b = parse(ONE_BLOCK)
    assert len(b) == 1
    assert b[0].provisional == "the quick brown fax"
    assert b[0].final == "the quick brown fox"


def test_final_text_is_what_the_user_is_left_with():
    assert final_text(THREE_BLOCKS) == "the quick brown fox jumps over the lazy dog"


def test_provisional_text_is_what_was_shown_first():
    """Scoring this against the reference gives the quality of the FIRST paint."""
    assert provisional_text(THREE_BLOCKS) == "the quick brown fax jumps ovr the lazy dog"


def test_empty_provisional_span_is_legal():
    """A block where the fast front emitted nothing still carries its correction."""
    s = f"{TEMP_START}{TEMP_END}hello there"
    assert parse(s)[0].provisional == ""
    assert final_text(s) == "hello there"


def test_empty_final_span_is_legal():
    """Silence: the slow front confirms that nothing was said."""
    s = f"{TEMP_START}uh{TEMP_END}"
    assert parse(s)[0].final == ""
    assert final_text(s) == ""


@pytest.mark.parametrize(
    "bad",
    [
        "the quick brown fox",  # no tags at all
        "<temp_start>unterminated",  # missing <temp_end>
        "<temp_end>backwards<temp_start>",  # wrong order
    ],
)
def test_malformed_strings_raise(bad):
    """A truncated trace must not parse to 'fewer blocks' and understate metrics."""
    with pytest.raises(ValueError):
        parse(bad)


# ---------------------------------------------------------------------- metrics


def test_corrected_flag_ignores_whitespace_only_differences():
    assert not Block("the  fox", "the fox").corrected
    assert Block("the fax", "the fox").corrected


def test_revision_stats_on_the_three_block_example():
    st = revision_stats(THREE_BLOCKS)
    assert st.blocks == 3
    assert st.corrected_blocks == 2  # the third block agreed
    assert st.block_correction_rate == pytest.approx(2 / 3)


def test_word_revision_rate_counts_from_the_first_word_that_moves():
    """A reader perceives a change at the first moved word; the tail re-renders."""
    # "the quick brown fax" -> "the quick brown fox": 3 words survive, 1 revised.
    st = revision_stats(ONE_BLOCK)
    assert st.provisional_words == 4
    assert st.revised_words == 1
    assert st.word_revision_rate == pytest.approx(0.25)


def test_a_fully_agreeing_stream_has_zero_revisions():
    s = f"{TEMP_START}hello world{TEMP_END}hello world"
    st = revision_stats(s)
    assert st.corrected_blocks == 0
    assert st.revised_words == 0


# ------------------------------------------------------------------ decode loop


class FakeModel:
    """Records what each front was conditioned on, and how it was windowed."""

    def __init__(self, n_frames=28, prov=None, auth=None):
        self._n = n_frames
        self.calls = []
        self._prov = prov or ["the quick brown fax", "jumps ovr"]
        self._auth = auth or ["the quick brown fox", "jumps over"]

    def encode_frames(self, audios, audio_lens, chunk):
        import torch

        return [torch.zeros(self._n, 4)]

    def generate(self, audios, audio_lens, system_prompt, chunk_size_override, history_text, frame_range, **kw):
        self.calls.append({"chunk": chunk_size_override, "history": list(history_text), "range": tuple(frame_range)})
        block = frame_range[0] // 14
        src = self._auth if chunk_size_override == 14 else self._prov
        return [src[block] if block < len(src) else ""]


def test_slow_chunk_must_be_a_multiple_of_fast():
    with pytest.raises(ValueError, match="exact multiple"):
        multi_front_decode(FakeModel(), None, None, fast_chunk=4, slow_chunk=14)


def test_identical_fronts_are_rejected():
    with pytest.raises(ValueError, match="must differ"):
        multi_front_decode(FakeModel(), None, None, fast_chunk=14, slow_chunk=14)


def test_decode_produces_the_expected_string():
    out = multi_front_decode(FakeModel(n_frames=28), None, None)
    assert out == [
        "<temp_start>the quick brown fax<temp_end>the quick brown fox" "<temp_start>jumps ovr<temp_end>jumps over"
    ]


def test_each_block_decodes_its_own_frame_window():
    """The encoder sees the whole utterance; only the DECODE window moves."""
    m = FakeModel(n_frames=28)
    multi_front_decode(m, None, None)
    assert [c["range"] for c in m.calls] == [(0, 14), (0, 14), (14, 28), (14, 28)]


def test_resync_conditions_the_fast_front_on_committed_text():
    """Block 2's fast front must see the slow front's correction, not its own 'fax'."""
    m = FakeModel(n_frames=28)
    multi_front_decode(m, None, None, resync=True)
    fast_block2 = [c for c in m.calls if c["chunk"] == 2 and c["range"] == (14, 28)][0]
    assert fast_block2["history"] == ["the quick brown fox"]  # corrected, not "fax"


def test_without_resync_the_fast_front_keeps_its_own_error():
    m = FakeModel(n_frames=28)
    multi_front_decode(m, None, None, resync=False)
    fast_block2 = [c for c in m.calls if c["chunk"] == 2 and c["range"] == (14, 28)][0]
    assert fast_block2["history"] == ["the quick brown fax"]  # drifted


def test_slow_front_is_always_conditioned_on_committed_text():
    """The authoritative front must never see provisional text, resync or not."""
    for resync in (True, False):
        m = FakeModel(n_frames=28)
        multi_front_decode(m, None, None, resync=resync)
        slow_block2 = [c for c in m.calls if c["chunk"] == 14 and c["range"] == (14, 28)][0]
        assert slow_block2["history"] == ["the quick brown fox"]


def test_partial_tail_block_still_gets_a_correction():
    """30 frames = two full blocks + 2: the remainder must not stay provisional."""
    m = FakeModel(n_frames=30, prov=["a", "b", "c"], auth=["A", "B", "C"])
    out = multi_front_decode(m, None, None)
    assert len(parse(out[0])) == 3
    assert final_text(out[0]) == "A B C"


# --------------------------------------------------------------------- display


def test_display_frames_paints_then_corrects():
    """Two frames per block: provisional painted, then replaced."""
    from nemo.collections.speechlm2.parts.multi_front import display_frames

    assert display_frames(ONE_BLOCK) == [
        ("", "the quick brown fax"),  # provisional on screen
        ("the quick brown fox", ""),  # slow front replaced it
    ]


def test_display_frames_accumulate_committed_text():
    from nemo.collections.speechlm2.parts.multi_front import display_frames

    frames = display_frames(THREE_BLOCKS)
    assert len(frames) == 6
    # Block 2 is painted on top of block 1's CORRECTED text, not its provisional.
    assert frames[2] == ("the quick brown fox", "jumps ovr")
    assert frames[-1] == ("the quick brown fox jumps over the lazy dog", "")


def test_stream_render_erases_exactly_what_it_wrote():
    """The provisional tail must be rubbed out, not left behind under the fix."""
    import io

    from nemo.collections.speechlm2.parts.multi_front import render_stream

    buf = io.StringIO()
    render_stream(ONE_BLOCK, out=buf)
    written = buf.getvalue()
    assert "fax" in written  # it WAS shown
    assert written.count("\b") >= len("the quick brown fax")  # and then erased
    # What survives once backspaces are applied is the corrected text.
    assert _apply_backspaces(written).strip() == "the quick brown fox"


def test_stream_render_final_screen_equals_final_text():
    import io

    from nemo.collections.speechlm2.parts.multi_front import render_stream

    buf = io.StringIO()
    render_stream(THREE_BLOCKS, out=buf)
    assert _apply_backspaces(buf.getvalue()).split() == final_text(THREE_BLOCKS).split()


def _apply_backspaces(s: str) -> str:
    """Replay a terminal's cursor so we can assert on what is actually visible."""
    out, col = [], 0
    for ch in s:
        if ch == "\b":
            col = max(0, col - 1)
        elif ch == "\n":
            out.append("\n")
            col = len(out)
        else:
            if col < len(out):
                out[col] = ch
            else:
                out.append(ch)
            col += 1
    return "".join(out)


def test_dual_render_puts_fast_above_slow():
    from nemo.collections.speechlm2.parts.multi_front import render_dual

    txt = render_dual(ONE_BLOCK, width=100)
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    assert lines[0].startswith("  fast") and "fax" in lines[0]
    assert lines[1].startswith("  slow") and "fox" in lines[1]


def test_dual_render_marks_only_corrected_blocks():
    from nemo.collections.speechlm2.parts.multi_front import render_dual

    txt = render_dual(THREE_BLOCKS, width=200)
    # Blocks 1 and 2 were corrected, block 3 agreed -> carets under the first two only.
    marker = [ln for ln in txt.splitlines() if ln.strip().startswith("^")]
    assert marker, "corrected blocks must be marked"
    assert "^" in marker[0]


def test_dual_render_columns_align():
    """The two fronts must line up per block, or they cannot be read against each other."""
    from nemo.collections.speechlm2.parts.multi_front import render_dual

    txt = render_dual(ONE_BLOCK, width=100)
    fast, slow = txt.splitlines()[0], txt.splitlines()[1]
    assert fast.index("the") == slow.index("the")


def test_dual_render_wraps_to_width():
    from nemo.collections.speechlm2.parts.multi_front import render_dual

    many = "".join(f"{TEMP_START}word{i}{TEMP_END}word{i}" for i in range(40))
    txt = render_dual(many, width=60)
    assert all(len(ln) <= 70 for ln in txt.splitlines())  # 60 + the "  fast  " gutter
    assert txt.count("  fast  ") > 1  # actually wrapped onto several row groups


def test_dual_render_empty_is_empty():
    from nemo.collections.speechlm2.parts.multi_front import render_dual

    assert render_dual("") == ""
