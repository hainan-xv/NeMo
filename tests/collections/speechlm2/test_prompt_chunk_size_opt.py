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
"""``render_control_prompt(include_chunk_size=False)``.

The point of the option is multi-front prefix sharing: two fronts decoding the
same audio at different chunk sizes must produce the SAME instruction, so
``[instruction][history]`` is one prefix and one prefill serves both. With the
chunk size named, the two prompts diverge at the chunk-size digit -- which sits
before the history -- so no history KV can be shared.
"""
import pytest

from nemo.collections.speechlm2.parts.script_prompt import ScriptControls, render_control_prompt

BASE = "Transcribe."


@pytest.mark.unit
def test_prompts_match_across_chunk_sizes_when_omitted():
    """THE property: same delay/style, different chunk size -> identical prompt."""
    fast = render_control_prompt(BASE, ScriptControls(2, 3, True, True), include_chunk_size=False)
    slow = render_control_prompt(BASE, ScriptControls(28, 3, True, True), include_chunk_size=False)
    assert fast == slow
    assert "2 frames" not in fast and "28 frames" not in fast


@pytest.mark.unit
def test_prompts_differ_across_chunk_sizes_when_included():
    """The status quo, kept as the contrast case."""
    fast = render_control_prompt(BASE, ScriptControls(2, 3, True, True), include_chunk_size=True)
    slow = render_control_prompt(BASE, ScriptControls(28, 3, True, True), include_chunk_size=True)
    assert fast != slow


@pytest.mark.unit
def test_default_is_unchanged():
    """Existing checkpoints must keep their exact training wording."""
    c = ScriptControls(14, 3, True, False)
    assert render_control_prompt(BASE, c) == render_control_prompt(BASE, c, include_chunk_size=True)
    assert "chunked every 14 frames with an emission delay of 3 frames." in render_control_prompt(BASE, c)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("delay", "expected"),
    [(0, "The audio is chunked with no emission delay."), (1, "an emission delay of 1 frame."), (3, "of 3 frames.")],
)
def test_delay_is_still_stated(delay, expected):
    """Delay is NOT observable from the window, so it must survive the change."""
    p = render_control_prompt(BASE, ScriptControls(14, delay, True, True), include_chunk_size=False)
    assert expected in p


@pytest.mark.unit
@pytest.mark.parametrize("cap", [True, False])
@pytest.mark.parametrize("pun", [True, False])
def test_style_clauses_survive(cap, pun):
    p = render_control_prompt(BASE, ScriptControls(7, 2, cap, pun), include_chunk_size=False)
    assert ("Use capitalization." if cap else "Do not use capitalization.") in p
    assert ("Use punctuation." if pun else "Do not use punctuation.") in p


@pytest.mark.unit
def test_wording_is_a_minimal_edit_from_the_included_form():
    """Keeps the no-chunk form close to the warm-start checkpoint's distribution."""
    c = ScriptControls(14, 3, True, True)
    with_cs = render_control_prompt(BASE, c, include_chunk_size=True)
    without = render_control_prompt(BASE, c, include_chunk_size=False)
    assert with_cs.replace(" every 14 frames", "") == without
