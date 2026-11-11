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

"""Chunk-local mixing must be bidirectional INSIDE a chunk and sealed ACROSS chunks.

The whole point is to match a chunked_limited Conformer's information access. If
the mixer leaked across a boundary the arm would be non-streamable and the
latency claim false; if it did not actually mix within the chunk it would be a
no-op and the lookahead confound would remain. Both directions are tested.
"""

import pytest
import torch

from nemo.collections.asr.parts.submodules.chunk_bidir_conv import ChunkBidirMixer

C = 14
D = 8


def _mixer(chunk_size=C, channels=D):
    m = ChunkBidirMixer(channels=channels, chunk_size=chunk_size).eval()
    # Undo the identity init so the block actually computes something.
    torch.nn.init.normal_(m.pointwise.weight, std=0.5)
    torch.nn.init.normal_(m.pointwise.bias, std=0.5)
    return m


def _sensitivity(m, T, t_perturb):
    """Which output frames change when input frame t_perturb changes?"""
    torch.manual_seed(0)
    x = torch.randn(1, D, T)
    y0 = m.mix(x)
    x2 = x.clone()
    x2[0, :, t_perturb] += 10.0
    y1 = m.mix(x2)
    return (y1 - y0).abs().sum(dim=1).squeeze(0)  # (T,) per-frame change


def test_no_leakage_across_chunk_boundary():
    """Perturbing a frame must not move ANY output outside its own chunk."""
    m = _mixer()
    T = C * 4
    for t in (0, 5, C - 1, C, C + 7, 2 * C, 3 * C + 3):
        delta = _sensitivity(m, T, t)
        chunk = t // C
        lo, hi = chunk * C, (chunk + 1) * C
        outside = torch.cat([delta[:lo], delta[hi:]])
        assert outside.max().item() == 0.0, f"leak from frame {t} (chunk {chunk}) to outside [{lo},{hi})"


def test_mixing_is_bidirectional_within_chunk():
    """A frame must influence BOTH earlier and later frames in its own chunk."""
    m = _mixer()
    T = C * 3
    t = C + 7  # middle of chunk 1
    delta = _sensitivity(m, T, t)
    lo = C
    assert delta[lo : lo + 7].sum().item() > 0, "no backward influence within chunk"
    assert delta[lo + 8 : lo + C].sum().item() > 0, "no forward influence within chunk"


def test_every_position_reaches_every_other_in_chunk():
    """Default kernel 2C-1 should make the chunk fully connected."""
    m = _mixer()
    delta = _sensitivity(m, C, 0)
    assert (delta > 0).all(), "frame 0 does not reach the whole chunk"
    delta = _sensitivity(m, C, C - 1)
    assert (delta > 0).all(), "last frame does not reach the whole chunk"


def test_identity_at_init_preserves_warm_start():
    m = ChunkBidirMixer(channels=D, chunk_size=C).eval()
    x = torch.randn(2, D, C * 3)
    out, lens = m(([x], torch.tensor([C * 3, C * 2])))
    torch.testing.assert_close(out[-1], x)


def test_handles_length_not_divisible_by_chunk():
    m = _mixer()
    for T in (1, C - 1, C + 1, C * 3 + 5):
        y = m.mix(torch.randn(1, D, T))
        assert y.shape == (1, D, T), f"shape changed for T={T}"


def test_ragged_batch_still_sealed_per_chunk():
    """Padding from a shorter utterance must not pull signal across boundaries."""
    m = _mixer()
    T = C * 3
    delta = _sensitivity(m, T, 2 * C + 1)
    assert delta[: 2 * C].max().item() == 0.0


@pytest.mark.parametrize("bad", [0, -3])
def test_rejects_invalid_chunk_size(bad):
    with pytest.raises(ValueError):
        ChunkBidirMixer(channels=D, chunk_size=bad)


def test_rejects_even_kernel():
    with pytest.raises(ValueError, match="odd"):
        ChunkBidirMixer(channels=D, chunk_size=C, kernel_size=8)


# ---------------------------------------------------------------------------
# Whole-encoder behaviour: causal ACROSS chunks, bidirectional WITHIN one.
# ---------------------------------------------------------------------------


def _small_encoder(chunk_size=4, mixer_positions=None):
    """A tiny causal jasper stack (one stride-2 block) with mixers inserted."""
    from nemo.collections.asr.modules.chunk_bidir_conv_asr import ChunkBidirConvASREncoder

    def blk(filters, kernel, stride=1):
        return {
            "filters": filters,
            "repeat": 1,
            "kernel": [kernel],
            "stride": [stride],
            "dilation": [1],
            "dropout": 0.0,
            "residual": False,
            "separable": True,
            "se": False,
            "future_context": 0,
        }

    return ChunkBidirConvASREncoder(
        jasper=[blk(16, 3), blk(16, 3, stride=2), blk(16, 3), blk(16, 3)],
        activation="relu",
        feat_in=8,
        chunk_size=chunk_size,
        mixer_positions=mixer_positions,
    ).eval()


def test_mixers_land_after_the_last_stride_block():
    enc = _small_encoder()
    # Block index 1 carries stride 2, so every mixer must sit at index >= 1.
    assert min(enc.mixer_positions) >= 1


def test_mixer_before_stride_is_rejected():
    with pytest.raises(ValueError, match="precedes the last stride"):
        _small_encoder(mixer_positions=[0])


def test_encoder_is_causal_across_chunks_and_bidirectional_within():
    """The defining property: output frame t depends on inputs up to its chunk end, never beyond."""
    torch.manual_seed(0)
    CS = 4
    enc = _small_encoder(chunk_size=CS, mixer_positions=[2, 3])
    for m in enc.modules():
        if isinstance(m, ChunkBidirMixer):
            torch.nn.init.normal_(m.pointwise.weight, std=0.5)

    T_in = 64
    x = torch.randn(1, 8, T_in)
    lens = torch.tensor([T_in])
    with torch.no_grad():
        y0, _ = enc(audio_signal=x, length=lens)

    T_out = y0.shape[-1]
    # Perturb an input frame belonging to OUTPUT chunk 1 (stride 2 -> out = in//2).
    t_out = CS + 1
    t_in = t_out * 2
    x2 = x.clone()
    x2[0, :, t_in:] += 10.0  # everything from that point on
    with torch.no_grad():
        y1, _ = enc(audio_signal=x2, length=lens)

    delta = (y1 - y0).abs().sum(dim=1).squeeze(0)
    chunk_start = (t_out // CS) * CS
    # Outputs in strictly EARLIER chunks must be untouched.
    assert delta[:chunk_start].max().item() == pytest.approx(
        0.0, abs=1e-6
    ), "future input leaked backwards past a chunk boundary"
    # And the current chunk must react (otherwise the mixer is doing nothing).
    assert delta[chunk_start : chunk_start + CS].sum().item() > 0
