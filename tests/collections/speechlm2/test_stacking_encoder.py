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
"""StackingEncoder: the encoder-free audio front-end.

The point of these tests is that the frame GRID is unchanged. Every latency
quantity in SCRIPT is counted in 0.08 s encoder frames, so a stacked front-end
is only a drop-in replacement if it produces exactly the frame count the
Conformer did -- an off-by-one would silently shift every chunk boundary.
"""
import pytest
import torch

from nemo.collections.speechlm2.modules.stacking_encoder import StackingEncoder


@pytest.mark.unit
def test_holds_no_parameters():
    """All acoustic modelling moves into the LLM; this module is a reshape."""
    assert sum(p.numel() for p in StackingEncoder(128, 8).parameters()) == 0


@pytest.mark.unit
def test_dimensions_line_up_with_the_conformer_it_replaces():
    """128 mel x 8 frames == 1024 == the Conformer's d_model, so proj is reusable."""
    m = StackingEncoder(feat_in=128, subsampling_factor=8)
    assert m.d_model == 1024
    assert m.out_proj is None, "no projection should be needed when the widths already match"


@pytest.mark.unit
@pytest.mark.parametrize(("t_in", "expected"), [(112, 14), (224, 28), (300, 38), (1120, 140), (8, 1), (1, 1)])
def test_output_length_is_ceil_div(t_in, expected):
    m = StackingEncoder(128, 8, match_conv_subsampling=False)
    x = torch.randn(1, 128, t_in)
    y, yl = m(x, torch.tensor([t_in]))
    assert int(yl) == expected
    assert y.shape[-1] >= expected


@pytest.mark.unit
def test_tail_is_padded_not_dropped():
    """A partial final frame must still be emitted, or the last <=70ms vanishes."""
    m = StackingEncoder(128, 8, match_conv_subsampling=False)
    x = torch.ones(1, 128, 9)
    y, yl = m(x, torch.tensor([9]))
    assert int(yl) == 2, "9 frames must yield 2 output frames, not 1"
    assert y.shape[-1] == 2
    # Frame 0 is fully populated; frame 1 holds one real sub-frame then zeros.
    assert torch.all(y[0, :, 0] == 1.0)
    assert torch.all(y[0, :128, 1] == 1.0)
    assert torch.all(y[0, 128:, 1] == 0.0)


@pytest.mark.unit
def test_channel_layout_is_eight_frames_concatenated():
    """Channel block i*F:(i+1)*F must be the i-th sub-frame, verbatim."""
    f, s = 128, 8
    m = StackingEncoder(f, s, match_conv_subsampling=False)
    # Make every mel frame identifiable by its time index.
    x = torch.arange(s, dtype=torch.float32).view(1, 1, s).expand(1, f, s).contiguous()
    y, _ = m(x, torch.tensor([s]))
    for i in range(s):
        assert torch.all(y[0, i * f : (i + 1) * f, 0] == float(i)), f"sub-frame {i} misplaced"


@pytest.mark.unit
def test_optional_projection_when_widths_do_not_match():
    m = StackingEncoder(feat_in=80, subsampling_factor=4, d_model=1024, match_conv_subsampling=False)
    assert m.out_proj is not None and m.stacked_dim == 320
    y, _ = m(torch.randn(2, 80, 40), torch.tensor([40, 20]))
    assert y.shape == (2, 1024, 10)


@pytest.mark.unit
def test_lengths_never_exceed_the_emitted_frames():
    """A padded batch must not report more frames than the tensor holds."""
    m = StackingEncoder(128, 8, match_conv_subsampling=False)
    y, yl = m(torch.randn(3, 128, 100), torch.tensor([100, 50, 1]))
    assert int(yl.max()) <= y.shape[-1]
    assert yl.tolist() == [13, 7, 1]


@pytest.mark.unit
@pytest.mark.parametrize("bad", [torch.randn(2, 64, 40), torch.randn(2, 40)])
def test_rejects_wrong_shapes(bad):
    with pytest.raises(ValueError):
        StackingEncoder(128, 8)(bad, torch.tensor([40, 40]))


@pytest.mark.unit
@pytest.mark.parametrize(("t_in", "conformer_len"), [(50, 8), (80, 11), (100, 14), (112, 15), (1120, 141)])
def test_default_frame_count_matches_conv_subsampling(t_in, conformer_len):
    """The SCRIPT dataset's audio_frame_index is built against the Conformer.

    Emitting one fewer frame makes the final index fall past the end: during
    TRAINING it is silently zero-padded, but encode_frames at inference returns
    only the real frames, so the last chunk is a different width in training
    than at decode time. Measured against the real ConformerEncoder, which
    produces ceil(T / 8) + 1.
    """
    m = StackingEncoder(128, 8)  # default: match_conv_subsampling=True
    y, yl = m(torch.randn(1, 128, t_in), torch.tensor([t_in]))
    assert int(yl) == conformer_len
    assert y.shape[-1] >= conformer_len


@pytest.mark.unit
def test_the_extra_frame_is_zeros():
    """It is padding, not invented content: the real audio ends where it ends."""
    m = StackingEncoder(128, 8)
    y, yl = m(torch.ones(1, 128, 16), torch.tensor([16]))
    assert int(yl) == 3  # ceil(16/8) + 1
    assert torch.all(y[0, :, :2] == 1.0)
    assert torch.all(y[0, :, 2] == 0.0)
