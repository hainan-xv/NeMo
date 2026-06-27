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

import pytest
import torch

from nemo.collections.asr.modules.chunk_alternating_conformer_encoder import (
    ChannelAxisConformerLayer,
    ChunkAlternatingConformerEncoder,
)
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder


def _encoder_kwargs(d_model=64, n_layers=4, n_heads=4, **overrides):
    """Small, deterministic ConformerEncoder config shared by the tests."""
    kwargs = dict(
        feat_in=80,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        subsampling="striding",
        subsampling_factor=4,
        subsampling_conv_channels=32,
        ff_expansion_factor=4,
        self_attention_model="rel_pos",
        conv_kernel_size=9,
        conv_norm_type="batch_norm",
        dropout=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
        dropout_att=0.0,
        pos_emb_max_len=5000,
    )
    kwargs.update(overrides)
    return kwargs


def _build_encoder(chunk_size=4, chunk_tokens=4, channel_conv_kernel_size=3, channel_layers="odd", **overrides):
    return ChunkAlternatingConformerEncoder(
        chunk_size=chunk_size,
        chunk_tokens=chunk_tokens,
        channel_conv_kernel_size=channel_conv_kernel_size,
        channel_layers=channel_layers,
        **_encoder_kwargs(**overrides),
    )


class TestChannelAxisReshape:
    """The per-chunk reshape used by the channel layer must be a lossless bijection."""

    @pytest.mark.unit
    @pytest.mark.parametrize("chunk_size,chunk_tokens", [(4, 4), (8, 8), (6, 4), (8, 16)])
    def test_reshape_round_trip_is_identity(self, chunk_size, chunk_tokens):
        torch.manual_seed(0)
        B, T, D = 2, 24, 32
        layer = ChannelAxisConformerLayer(d_model=D, chunk_size=chunk_size, chunk_tokens=chunk_tokens)
        x = torch.randn(B, T, D)

        C, M, new_d = chunk_size, chunk_tokens, layer.new_d_model
        n_chunks = (T + C - 1) // C
        T_pad = n_chunks * C
        xp = torch.nn.functional.pad(x, (0, 0, 0, T_pad - T)) if T_pad > T else x

        # forward reshape: [B, T_pad, D] -> [B*n_chunks, M, new_d]
        y = xp.reshape(B, n_chunks, C * D).reshape(B, n_chunks, new_d, M).transpose(-1, -2).contiguous()
        y = y.reshape(B * n_chunks, M, new_d)
        # inverse reshape back to [B, T_pad, D]
        z = y.reshape(B, n_chunks, M, new_d).transpose(-1, -2).contiguous()
        z = z.reshape(B, n_chunks, C * D).reshape(B, T_pad, D)

        assert torch.allclose(z[:, :T], x, atol=1e-6)


class TestChannelAxisConformerLayer:
    @pytest.mark.unit
    def test_preserves_shape(self):
        torch.manual_seed(0)
        layer = ChannelAxisConformerLayer(d_model=32, chunk_size=4, chunk_tokens=4).eval()
        x = torch.randn(3, 17, 32)  # T deliberately not a multiple of chunk_size
        with torch.no_grad():
            y = layer(x=x)
        assert y.shape == x.shape

    @pytest.mark.unit
    def test_heads_are_coerced_to_divisor_of_new_d_model(self):
        # new_d_model = 6*16/4 = 24; 24 % 5 != 0 -> heads must be coerced down to a divisor.
        layer = ChannelAxisConformerLayer(d_model=16, chunk_size=6, chunk_tokens=4, n_heads=5)
        assert layer.new_d_model == 24
        assert layer.num_heads <= 5
        assert layer.new_d_model % layer.num_heads == 0

    @pytest.mark.unit
    def test_invalid_geometry_raises(self):
        # chunk_size*d_model must be divisible by chunk_tokens.
        with pytest.raises(ValueError):
            ChannelAxisConformerLayer(d_model=10, chunk_size=3, chunk_tokens=4)

    @pytest.mark.unit
    def test_even_conv_kernel_raises(self):
        with pytest.raises(ValueError):
            ChannelAxisConformerLayer(d_model=16, chunk_size=4, chunk_tokens=4, conv_kernel_size=4)

    @pytest.mark.unit
    def test_cache_aware_streaming_not_supported(self):
        layer = ChannelAxisConformerLayer(d_model=16, chunk_size=4, chunk_tokens=4)
        x = torch.randn(1, 8, 16)
        with pytest.raises(NotImplementedError):
            layer(x=x, cache_last_channel=torch.zeros(1, 1, 16))


class TestChunkAlternatingConformerEncoder:
    @pytest.mark.unit
    @pytest.mark.parametrize("channel_layers,expected", [("odd", {1, 3}), ("even", {0, 2})])
    def test_alternating_layers_are_channel_axis(self, channel_layers, expected):
        enc = _build_encoder(channel_layers=channel_layers, n_layers=4)
        channel_idx = {i for i, l in enumerate(enc.layers) if isinstance(l, ChannelAxisConformerLayer)}
        assert channel_idx == expected

    @pytest.mark.unit
    def test_forward_runs_and_shapes_match_standard_encoder(self):
        torch.manual_seed(0)
        std = ConformerEncoder(**_encoder_kwargs()).eval()
        chan = _build_encoder().eval()
        feats = torch.randn(2, 80, 240)
        length = torch.tensor([240, 160])
        with torch.no_grad():
            o_std, l_std = std(audio_signal=feats, length=length)
            o_chan, l_chan = chan(audio_signal=feats, length=length)
        assert o_chan.shape == o_std.shape
        assert torch.equal(l_chan, l_std)

    @pytest.mark.unit
    def test_gradients_flow(self):
        torch.manual_seed(0)
        enc = _build_encoder().train()
        feats = torch.randn(2, 80, 240, requires_grad=True)
        out, _ = enc(audio_signal=feats, length=torch.tensor([240, 160]))
        out.sum().backward()
        assert feats.grad is not None and torch.isfinite(feats.grad).all()
        for p in enc.parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all()

    @pytest.mark.unit
    @pytest.mark.parametrize("extra_pad", [4, 40, 120])
    def test_output_is_padding_invariant(self, extra_pad):
        """Regression test for the padding-leak bug.

        The channel-axis layers must mask padded time steps before the per-chunk
        reshape; otherwise padded frames leak into the valid frames of the boundary
        chunk and (via the next full-context time layer) across the whole sequence,
        making the output depend on how much padding the batch carries.
        """
        torch.manual_seed(0)
        enc = _build_encoder().eval()
        torch.manual_seed(1)
        base = torch.randn(1, 80, 200)

        def run(feats):
            with torch.no_grad():
                out, olen = enc(audio_signal=feats, length=torch.tensor([200]))
            return out[..., : olen.item()]

        ref = run(base)
        padded = torch.cat([base, torch.randn(1, 80, extra_pad)], dim=2)
        got = run(padded)
        assert ref.shape == got.shape
        # Valid-frame outputs must not change when extra padding is appended.
        assert (ref - got).abs().max().item() < 1e-4

    @pytest.mark.unit
    def test_padding_invariance_matches_standard_encoder(self):
        """The channel encoder should be as padding-invariant as the standard one."""
        torch.manual_seed(0)
        std = ConformerEncoder(**_encoder_kwargs()).eval()
        chan = _build_encoder().eval()
        torch.manual_seed(1)
        base = torch.randn(1, 80, 200)
        padded = torch.cat([base, torch.randn(1, 80, 80)], dim=2)

        def delta(model):
            with torch.no_grad():
                o1, l1 = model(audio_signal=base, length=torch.tensor([200]))
                o2, _ = model(audio_signal=padded, length=torch.tensor([200]))
            tv = l1.item()
            return (o1[..., :tv] - o2[..., :tv]).abs().max().item()

        # Both should be invariant to trailing padding to within fp tolerance.
        assert delta(chan) < 1e-4
        assert delta(std) < 1e-4
