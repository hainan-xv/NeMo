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
"""Unit coverage for ``_set_encoder_att_context``'s two chunk grids.

``first_chunk_size`` is drawn by the dataset against the LLM chunk, while
``encoder_chunk_size`` pins the ENCODER's look-ahead to a different (smaller)
value. Conflating the two made every ``encoder_chunk_size=1`` run die on
``first_chunk_size must be in [1, 1], got 42`` at the first training step.
"""
from types import SimpleNamespace

import pytest

from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel

_set_ctx = StreamingSTTModel._set_encoder_att_context


def _fake(encoder_chunk_size=None, left=70):
    """A stand-in with just the attributes the setter touches."""
    encoder = SimpleNamespace(att_context_size=[left, 0], att_context_style="chunked_limited")
    core_cfg = SimpleNamespace(att_context_size=[left, 0], encoder_chunk_size=encoder_chunk_size)
    return SimpleNamespace(core_cfg=core_cfg, perception=SimpleNamespace(encoder=encoder))


@pytest.mark.unit
def test_encoder_chunk_size_does_not_bound_first_chunk_size():
    """Regression: first_chunk_size is in LLM-chunk units, not encoder units."""
    m = _fake(encoder_chunk_size=1)
    # chunk_size=42 is the LLM chunk; the dataset passes first_chunk_size=42 for
    # every batch when random_first_chunk is off. This used to raise.
    _set_ctx(m, 42, first_chunk_size=42)
    enc = m.perception.encoder
    # A chunk of width 1 has no grid phase, so the shift collapses to 0...
    assert enc.att_chunk_offset == 0
    # ...and the look-ahead stays pinned to the encoder chunk (right = 1 - 1).
    assert enc.att_context_size == [70, 0]


@pytest.mark.unit
@pytest.mark.parametrize("fcs", [1, 7, 21, 42])
def test_ragged_first_chunk_accepted_under_pinned_encoder(fcs):
    m = _fake(encoder_chunk_size=1)
    _set_ctx(m, 42, first_chunk_size=fcs)
    assert m.perception.encoder.att_chunk_offset == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("chunk_size", "fcs", "expected_offset"),
    [(14, 14, 0), (14, None, 0), (14, 7, 7), (14, 1, 13), (42, 28, 14)],
)
def test_unpinned_encoder_offset_unchanged(chunk_size, fcs, expected_offset):
    """With encoder_chunk_size unset the two grids coincide — behaviour is as before."""
    m = _fake(encoder_chunk_size=None)
    _set_ctx(m, chunk_size, first_chunk_size=fcs)
    assert m.perception.encoder.att_chunk_offset == expected_offset
    assert m.perception.encoder.att_context_size == [70, chunk_size - 1]


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1, 43])
def test_out_of_range_first_chunk_size_still_rejected(bad):
    m = _fake(encoder_chunk_size=None)
    with pytest.raises(ValueError, match=r"first_chunk_size must be in \[1, 42\]"):
        _set_ctx(m, 42, first_chunk_size=bad)
