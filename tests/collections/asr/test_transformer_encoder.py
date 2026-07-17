# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import pytest
import torch

from nemo.collections.asr.modules.transformer_encoder import (
    FeatureStacking,
    StreamingTransformerEncoder,
    TransformerEncoder,
    TransformerEncoderConfig,
    _make_sliding_window_mod,
)
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.submodules.multi_head_attention import RotaryPositionalEncoding


class TestTransformerEncoderConfig:
    @pytest.mark.unit
    def test_default_config(self):
        cfg = TransformerEncoderConfig()
        assert cfg.feat_in == 128
        assert cfg.d_model == 512
        assert cfg.n_heads == 8
        assert cfg.n_layers == 17
        assert cfg.drop_rate == 0.1
        assert cfg.qkv_bias is False
        assert cfg.qk_norm is False
        assert cfg.ff_expansion == 4.0
        assert cfg.pre_block_norm is True
        assert cfg.subsampling_factor == 4
        assert cfg.attn_mode == "full"
        assert cfg.self_attention_model == "rel_pos"
        assert cfg.rope_base == 10000.0
        assert cfg.rotary_fraction == 1.0

    @pytest.mark.unit
    def test_custom_config(self):
        cfg = TransformerEncoderConfig(
            feat_in=128, d_model=1280, n_heads=16, n_layers=32, qk_norm=True, self_attention_model="abs_pos"
        )
        assert cfg.feat_in == 128
        assert cfg.d_model == 1280
        assert cfg.n_heads == 16
        assert cfg.n_layers == 32
        assert cfg.qk_norm is True
        assert cfg.self_attention_model == "abs_pos"


class TestFeatureStacking:
    @pytest.mark.unit
    @pytest.mark.parametrize("subsampling_factor", [2, 4, 8])
    def test_output_shape(self, subsampling_factor):
        B, C, T = 2, 80, 400
        stacking = FeatureStacking(subsampling_factor=subsampling_factor, feat_in=C, feat_out=256)
        x = torch.randn(B, C, T)
        lengths = torch.tensor([400, 300])

        out, out_lengths = stacking(x, lengths)
        expected_t = stacking.compute_num_out_frames(T)
        assert out.shape == (B, expected_t, 256)
        assert out_lengths[0].item() == expected_t

    @pytest.mark.unit
    def test_padding_when_not_divisible(self):
        B, C, T = 1, 80, 401
        subsampling_factor = 4
        stacking = FeatureStacking(subsampling_factor=subsampling_factor, feat_in=C, feat_out=256)
        x = torch.randn(B, C, T)
        lengths = torch.tensor([401])

        out, out_lengths = stacking(x, lengths)
        expected_t = stacking.compute_num_out_frames(T)
        assert out.shape == (B, expected_t, 256)
        assert out_lengths[0].item() == expected_t

    @pytest.mark.unit
    def test_length_shorter_than_batch(self):
        """Output length must be ceil(sample_length / factor), not dependent on batch T."""
        B, C, T = 2, 80, 403
        subsampling_factor = 4
        stacking = FeatureStacking(subsampling_factor=subsampling_factor, feat_in=C, feat_out=256)
        x = torch.randn(B, C, T)
        lengths = torch.tensor([401, 397])

        _, out_lengths = stacking(x, lengths)
        assert out_lengths[0].item() == stacking.compute_num_out_frames(401)
        assert out_lengths[1].item() == stacking.compute_num_out_frames(397)

    @pytest.mark.unit
    def test_no_padding_when_divisible(self):
        B, C, T = 1, 80, 400
        stacking = FeatureStacking(subsampling_factor=4, feat_in=C, feat_out=256)
        x = torch.randn(B, C, T)
        lengths = torch.tensor([400])

        out, out_lengths = stacking(x, lengths)
        assert out.shape == (B, stacking.compute_num_out_frames(T), 256)
        assert out_lengths[0].item() == stacking.compute_num_out_frames(T)


class TestBypassPreEncode:
    """Testing bypass pre-encode functionality."""

    def test_bypass_pre_encode_forward(self):
        """Testing that forward works with "bypass pre-encode" mode.

        Forwards are wrapped in ``torch.no_grad()`` so the test runs on CPU as well as GPU:
        FlexAttention's CPU path refuses to run when any input requires gradients (parameters
        of an ``nn.Module`` do by default), and we are only checking output shapes here, never
        calling ``.backward()``.
        """
        # For pre-encoded embeddings, the shape is (batch_size, n_frames, emb_dim)
        batch_size = 2
        n_frames, emb_dim, feat_out = 17, 64, 8  # emb_dim=64 with n_heads=4 -> head_dim=16 (>= 16)
        random_input = torch.rand((batch_size, n_frames, emb_dim))
        random_length = torch.tensor([n_frames] * batch_size, dtype=torch.int64)

        model = TransformerEncoder(
            feat_in=10,
            n_layers=3,
            d_model=emb_dim,
            n_heads=4,
            feat_out=feat_out,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
        )
        model.train()
        with torch.no_grad():
            fwd_outputs = model(audio_signal=random_input, length=random_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

        model.eval()
        with torch.no_grad():
            fwd_outputs = model(audio_signal=random_input, length=random_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

    def test_error_shape_invalid_bypass_pre_encode_forward(self):
        """
        Testing that error messages are correctly triggered regarding "bypass pre-encode" mode.
        Both correct samples and wrongs samples are tested.

        (1) bypass_pre_encode = False (default):
            `audio_signal` must be a tensor containing audio features.
            Shape: (batch, self._feat_in, n_frames)
        (2) bypass_pre_encode = True:
            `audio_signal` must be a tensor containing pre-encoded embeddings.
            Shape: (batch, n_frame, self.d_model)
        """
        batch_size = 2
        n_frames, emb_dim, feat_in, feat_out = 17, 64, 10, 8  # emb_dim=64 with n_heads=4 -> head_dim=16 (>= 16)

        pre_encode_input = torch.rand((batch_size, n_frames, emb_dim))
        feat_input = torch.rand((batch_size, feat_in, n_frames))
        input_length = torch.tensor([n_frames] * batch_size, dtype=torch.int64)

        model = TransformerEncoder(
            feat_in=feat_in,
            n_layers=3,
            d_model=emb_dim,
            n_heads=4,
            feat_out=feat_out,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
        )
        sub_sampled_n_frames = np.ceil(n_frames / model.subsampling_factor)

        # Test with bypass_pre_encode = True, should be pre_encode_input but given feat_input.
        model.train()
        with pytest.raises(ValueError):
            model(audio_signal=feat_input, length=input_length, bypass_pre_encode=True)

        model.eval()
        with pytest.raises(ValueError):
            model(audio_signal=feat_input, length=input_length, bypass_pre_encode=True)

        # Test with bypass_pre_encode = True, given the correct input pre_encode_input.
        # NB: forwards that actually reach FlexAttention are wrapped in ``torch.no_grad()`` so
        # the test passes on CPU (FlexAttention's CPU path refuses inputs that require grad).
        # The ``pytest.raises(ValueError)`` blocks above/below intentionally do *not* need this
        # wrapper because the shape check in ``TransformerEncoder.forward()`` raises before any
        # attention computation.
        model.train()
        with torch.no_grad():
            fwd_outputs = model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

        model.eval()
        with torch.no_grad():
            fwd_outputs = model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

        # Test with bypass_pre_encode = False, should be feat_input but given pre_encode_input.
        model.train()
        with pytest.raises(ValueError):
            model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=False)

        model.eval()
        with pytest.raises(ValueError):
            model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=False)

        # Test with bypass_pre_encode = False, given the correct input feat_input.
        model.train()
        with torch.no_grad():
            fwd_outputs = model(audio_signal=feat_input, length=input_length, bypass_pre_encode=False)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, sub_sampled_n_frames)

        model.eval()
        with torch.no_grad():
            fwd_outputs = model(audio_signal=feat_input, length=input_length, bypass_pre_encode=False)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, sub_sampled_n_frames)

    @pytest.mark.unit
    def test_bypass_pre_encode_matches_manual_pre_encode(self):
        """``bypass_pre_encode=True`` must skip *only* the pre-encoder.

        Running the pre-encoder by hand and feeding its output back in with
        ``bypass_pre_encode=True`` should reproduce the full forward
        (``bypass_pre_encode=False``) exactly, because the positional-encoding, norm and
        Transformer-block stack downstream of the pre-encoder is identical on both paths.
        """
        B, feat_in, T, d_model, feat_out = 2, 32, 64, 64, 8  # d_model=64 with n_heads=4 -> head_dim=16 (>= 16)
        model = TransformerEncoder(
            feat_in=feat_in,
            d_model=d_model,
            n_heads=4,
            n_layers=2,
            feat_out=feat_out,
            subsampling_factor=4,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
        )
        model.eval()

        mel = torch.randn(B, feat_in, T)
        lengths = torch.tensor([T, T - 8], dtype=torch.int64)

        with torch.no_grad():
            out_full, len_full = model(audio_signal=mel, length=lengths, bypass_pre_encode=False)

            # Reproduce just the pre-encoder, then bypass it on the next call.
            pre_x, pre_len = model.pre_encode(mel, lengths)
            out_bypass, len_bypass = model(audio_signal=pre_x, length=pre_len, bypass_pre_encode=True)

        assert out_full.shape == out_bypass.shape == (B, feat_out, pre_x.shape[1])
        assert torch.equal(len_full, len_bypass)
        assert torch.allclose(out_full, out_bypass, atol=1e-5)


class TestTransformerEncoder:
    @pytest.mark.unit
    def test_model_creation(self):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        assert len(model.layers) == 2

    @pytest.mark.unit
    def test_model_creation_with_qk_norm(self):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, qk_norm=True)
        attn = model.layers[0].attn
        assert hasattr(attn, 'q_norm')
        assert hasattr(attn, 'k_norm')

    @pytest.mark.unit
    def test_model_creation_without_qk_norm(self):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, qk_norm=False)
        attn = model.layers[0].attn
        assert not hasattr(attn, 'q_norm')
        assert not hasattr(attn, 'k_norm')

    @pytest.mark.unit
    def test_invalid_attn_mode(self):
        with pytest.raises(ValueError, match="not yet supported"):
            TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, attn_mode="sliding_window")

    @pytest.mark.unit
    def test_head_dim_below_16_raises(self):
        """head_dim = d_model // n_heads must be >= 16 (PyTorch FlexAttention CUDA requirement).

        The check happens at construction time, so an unsupported (d_model, n_heads) pair raises
        before any forward pass.
        """
        # d_model=32, n_heads=4 -> head_dim=8 (< 16).
        with pytest.raises(ValueError, match="per-head embedding dimension >= 16"):
            TransformerEncoder(feat_in=128, d_model=32, n_heads=4, n_layers=2)

    @pytest.mark.unit
    def test_causal_forward_cpu(self):
        model = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, attn_mode="causal")
        model.eval()

        x = torch.randn(2, 80, 400)
        lengths = torch.tensor([400, 300])

        with torch.no_grad():
            out, out_lengths = model(x, lengths)

        assert out.shape == (2, 64, 100)
        assert out_lengths.tolist() == [100, 75]
        assert not torch.isnan(out).any()

    @pytest.mark.unit
    def test_causal_future_does_not_affect_past(self):
        """Output at position t must be invariant to changes at positions > t."""
        model = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, attn_mode="causal")
        model.eval()

        B, C, T = 1, 80, 400
        x_a = torch.randn(B, C, T)
        x_b = x_a.clone()
        # Perturb only the second half of frames.
        x_b[:, :, T // 2 :] = torch.randn(B, C, T - T // 2)
        lengths = torch.tensor([T])

        with torch.no_grad():
            out_a, _ = model(x_a, lengths)
            out_b, _ = model(x_b, lengths)

        # Output frames covering only past + present should be identical.
        # First half of *output* frames corresponds to first half of input frames after subsampling.
        safe_t = (T // 2) // model.pre_encode.subsampling_factor
        assert torch.allclose(out_a[:, :, :safe_t], out_b[:, :, :safe_t], atol=1e-5)

    @pytest.mark.unit
    def test_freeze_unfreeze_partial_restores_prior_state(self):
        model = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2)
        for p in model.final_norm.parameters():
            p.requires_grad = False
        prior = {n: p.requires_grad for n, p in model.named_parameters()}

        model.freeze()
        assert all(not p.requires_grad for p in model.parameters())
        assert not model.training

        model.unfreeze(partial=True)
        assert {n: p.requires_grad for n, p in model.named_parameters()} == prior
        assert model.training

    @pytest.mark.unit
    def test_forward_cpu(self):
        """Forward pass on CPU uses unfused FlexAttention fallback."""
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, subsampling_factor=4)
        model.eval()

        B, C, T = 2, 128, 400
        x = torch.randn(B, C, T)
        lengths = torch.tensor([400, 300])

        with torch.no_grad():
            out, out_lengths = model(audio_signal=x, length=lengths)

        assert out.shape == (B, 64, T // 4)
        assert out_lengths[0].item() == T // 4
        assert out_lengths[1].item() == 300 // 4
        assert not torch.isnan(out).any()

    @pytest.mark.unit
    def test_forward_cpu_with_qk_norm(self):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, qk_norm=True)
        model.eval()

        x = torch.randn(1, 128, 200)
        lengths = torch.tensor([200])

        with torch.no_grad():
            out, _ = model(audio_signal=x, length=lengths)

        assert out.shape == (1, 64, 50)
        assert not torch.isnan(out).any()

    @pytest.mark.run_only_on('GPU')
    def test_forward_basic(self):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, subsampling_factor=4)
        model = model.cuda().to(torch.bfloat16)

        B, C, T = 2, 128, 400
        x = torch.randn(B, C, T, device='cuda', dtype=torch.bfloat16)
        lengths = torch.tensor([400, 300], device='cuda')

        model.eval()
        with torch.no_grad():
            out, out_lengths = model(audio_signal=x, length=lengths)

        assert out.shape == (B, 64, T // 4)
        assert out_lengths[0].item() == T // 4
        assert out_lengths[1].item() == 300 // 4
        assert not torch.isnan(out).any()

    @pytest.mark.run_only_on('GPU')
    def test_forward_with_qk_norm(self):
        model = TransformerEncoder(
            feat_in=128, d_model=128, n_heads=8, n_layers=2, drop_rate=0.0, qk_norm=True, subsampling_factor=8
        )
        model = model.cuda().to(torch.bfloat16)

        B, C, T = 2, 128, 800
        x = torch.randn(B, C, T, device='cuda', dtype=torch.bfloat16)
        lengths = torch.tensor([800, 640], device='cuda')

        model.eval()
        with torch.no_grad():
            out, out_lengths = model(audio_signal=x, length=lengths)

        assert out.shape == (B, 128, T // 8)
        assert out_lengths[1].item() == 640 // 8
        assert not torch.isnan(out).any()

    @pytest.mark.run_only_on('GPU')
    def test_forward_output_channels_first(self):
        """Verify output is (B, D, T) channels-first as expected by downstream decoders."""
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=1, drop_rate=0.0)
        model = model.cuda().to(torch.bfloat16)

        x = torch.randn(1, 128, 200, device='cuda', dtype=torch.bfloat16)
        lengths = torch.tensor([200], device='cuda')

        model.eval()
        with torch.no_grad():
            out, _ = model(audio_signal=x, length=lengths)

        assert out.shape[1] == 64  # D dimension
        assert out.shape[2] == 200 // 4  # T dimension

    @pytest.mark.run_only_on('GPU')
    def test_eval_deterministic(self):
        """In eval mode with no dropout, repeated forward passes should produce identical output."""
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0)
        model = model.cuda().to(torch.bfloat16).eval()

        x = torch.randn(1, 128, 200, device='cuda', dtype=torch.bfloat16)
        lengths = torch.tensor([200], device='cuda')

        with torch.no_grad():
            out1, _ = model(audio_signal=x, length=lengths)
            out2, _ = model(audio_signal=x, length=lengths)

        assert torch.allclose(out1, out2, atol=1e-6)

    @pytest.mark.run_only_on('GPU')
    def test_padding_does_not_affect_valid_output(self):
        """Padding frames should not change the encoded output at valid positions."""
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0)
        model = model.cuda().to(torch.bfloat16).eval()

        T_valid = 200
        x_short = torch.randn(1, 128, T_valid, device='cuda', dtype=torch.bfloat16)
        lengths_short = torch.tensor([T_valid], device='cuda')

        T_padded = 400
        x_long = torch.zeros(1, 128, T_padded, device='cuda', dtype=torch.bfloat16)
        x_long[:, :, :T_valid] = x_short
        lengths_long = torch.tensor([T_valid], device='cuda')

        with torch.no_grad():
            out_short, len_short = model(audio_signal=x_short, length=lengths_short)
            out_long, len_long = model(audio_signal=x_long, length=lengths_long)

        assert len_short[0].item() == len_long[0].item()
        valid_t = len_short[0].item()
        # bf16 + different block mask shapes cause small numerical differences in Triton kernels
        assert torch.allclose(out_short[:, :, :valid_t], out_long[:, :, :valid_t], atol=5e-2)

    @pytest.mark.run_only_on('GPU')
    def test_backward_pass(self):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0)
        model = model.cuda().to(torch.bfloat16).train()

        x = torch.randn(2, 128, 200, device='cuda', dtype=torch.bfloat16)
        lengths = torch.tensor([200, 160], device='cuda')

        out, _ = model(audio_signal=x, length=lengths)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestSelfAttentionModel:
    """Tests for the ``self_attention_model`` positional encoding option."""

    @pytest.mark.unit
    def test_default_is_rel_pos(self):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2)
        assert model.self_attention_model == "rel_pos"

    @pytest.mark.unit
    @pytest.mark.parametrize("mode", ["abs_pos", "rel_pos", "no_pos", "rope"])
    def test_valid_modes_are_accepted(self, mode):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, self_attention_model=mode)
        assert model.self_attention_model == mode

    @pytest.mark.unit
    def test_none_aliases_no_pos(self):
        """Passing ``self_attention_model=None`` must be equivalent to ``"no_pos"``."""
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, self_attention_model=None)
        assert model.self_attention_model == "no_pos"
        assert model.pos_enc is None

    @pytest.mark.unit
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            TransformerEncoder(
                feat_in=128, d_model=64, n_heads=4, n_layers=2, self_attention_model="rel_pos_local_attn"
            )

    @pytest.mark.unit
    def test_rel_pos_attention_params_allocated(self):
        """rel_pos mode allocates the Transformer-XL bias parameters per attention layer."""
        d_model, n_heads, n_layers = 64, 4, 2
        model = TransformerEncoder(
            feat_in=128, d_model=d_model, n_heads=n_heads, n_layers=n_layers, self_attention_model="rel_pos"
        )
        head_dim = d_model // n_heads
        assert model.pos_enc is not None
        for layer in model.layers:
            attn = layer.attn
            assert attn.linear_pos is not None
            assert attn.pos_bias_u is not None
            assert attn.pos_bias_v is not None
            assert attn.pos_bias_u.shape == (n_heads, head_dim)
            assert attn.pos_bias_v.shape == (n_heads, head_dim)

    @pytest.mark.unit
    @pytest.mark.parametrize("mode", ["abs_pos", "no_pos", "rope"])
    def test_non_rel_pos_modes_have_no_rel_params(self, mode):
        """abs_pos, no_pos and rope modes must not allocate the rel-pos parameters."""
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, self_attention_model=mode)
        for layer in model.layers:
            attn = layer.attn
            assert attn.linear_pos is None
            assert attn.pos_bias_u is None
            assert attn.pos_bias_v is None

    @pytest.mark.unit
    def test_no_pos_has_no_positional_encoding_module(self):
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=2, self_attention_model="no_pos")
        assert model.pos_enc is None
        # set_max_audio_length is invoked in __init__; it must not crash for no_pos and must
        # still record the requested max length so update_max_seq_length works normally.
        assert model.max_audio_length == model.pos_emb_max_len

    @pytest.mark.unit
    @pytest.mark.parametrize("mode", ["abs_pos", "rel_pos", "no_pos", "rope", None])
    def test_forward_each_mode_cpu(self, mode):
        """Each ``self_attention_model`` choice (including ``None``) must produce a valid forward."""
        model = TransformerEncoder(
            feat_in=128,
            d_model=64,
            n_heads=4,
            n_layers=2,
            drop_rate=0.0,
            subsampling_factor=4,
            self_attention_model=mode,
        )
        model.eval()

        B, C, T = 2, 128, 200
        x = torch.randn(B, C, T)
        lengths = torch.tensor([T, 160])

        with torch.no_grad():
            out, out_lengths = model(audio_signal=x, length=lengths)

        assert out.shape == (B, 64, T // 4)
        assert out_lengths[0].item() == T // 4
        assert out_lengths[1].item() == 160 // 4
        assert not torch.isnan(out).any()

    @pytest.mark.unit
    def test_base_build_mask_mod_unchanged(self):
        """The refactored ``_build_mask_mod`` hook must preserve the base full/causal masks."""
        full = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=1, attn_mode="full")
        causal = TransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=1, attn_mode="causal")
        length = torch.tensor([4, 4], dtype=torch.int64)

        def allowed(mod, q, kv):
            return bool(mod(torch.tensor(0), torch.tensor(0), torch.tensor(q), torch.tensor(kv)))

        full_mod = full._build_mask_mod(length)
        causal_mod = causal._build_mask_mod(length)
        # Full: any valid (non-pad) key is attendable, including future keys.
        assert allowed(full_mod, 1, 3)
        assert allowed(full_mod, 3, 1)
        # Padding (kv >= length) is masked in both.
        assert not allowed(full_mod, 1, 4)
        # Causal: future keys are masked, past/present allowed.
        assert allowed(causal_mod, 3, 1)
        assert not allowed(causal_mod, 1, 3)

    @pytest.mark.unit
    def test_rel_pos_broadcasts_when_T_differs_from_n_heads(self):
        """Regression test for the Transformer-XL bias broadcasting.

        ``pos_bias_{u,v}`` has shape ``(H, D)`` and must broadcast against the head axis of
        ``q`` which has shape ``(B, H, T, D)``. A naive add would right-align ``H`` against
        ``T`` and either crash (``T != H``) or silently apply the bias on the wrong axis
        (``T == H``). This test exercises a configuration where ``T_attn != n_heads`` so the
        broken broadcast would surface as an error.
        """
        # 200 input frames / subsampling_factor=4 -> 50 attention frames; n_heads=4 -> T != H.
        model = TransformerEncoder(
            feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, self_attention_model="rel_pos"
        )
        model.eval()

        B, C, T = 2, 128, 200
        x = torch.randn(B, C, T)
        lengths = torch.tensor([T, 160])

        with torch.no_grad():
            out, _ = model(audio_signal=x, length=lengths)

        assert out.shape == (B, 64, T // 4)
        assert not torch.isnan(out).any()

    @pytest.mark.unit
    def test_rope_uses_shared_rotary_pos_enc(self):
        """rope mode builds a single ``RotaryPositionalEncoding`` reused by every attention layer.

        The cos/sin buffers are computed once on the shared module (see ``TransformerEncoder``),
        so each layer's ``attn.rope`` must be the *same* object as ``model.pos_enc``.
        """
        model = TransformerEncoder(feat_in=128, d_model=64, n_heads=4, n_layers=3, self_attention_model="rope")
        assert isinstance(model.pos_enc, RotaryPositionalEncoding)
        for layer in model.layers:
            attn = layer.attn
            assert attn._uses_rope is True
            assert attn.rope is model.pos_enc

    @pytest.mark.unit
    def test_rope_partial_rotation_forward_cpu(self):
        """``rotary_fraction`` < 1.0 rotates only part of each head dim (exercises the pass-through split)."""
        model = TransformerEncoder(
            feat_in=128,
            d_model=64,
            n_heads=4,
            n_layers=2,
            drop_rate=0.0,
            subsampling_factor=4,
            self_attention_model="rope",
            rotary_fraction=0.5,
        )
        model.eval()

        B, C, T = 2, 128, 200
        x = torch.randn(B, C, T)
        lengths = torch.tensor([T, 160])

        with torch.no_grad():
            out, _ = model(audio_signal=x, length=lengths)

        assert out.shape == (B, 64, T // 4)
        assert not torch.isnan(out).any()


class TestSlidingWindowMod:
    """Unit tests for the ``_make_sliding_window_mod`` FlexAttention mask factory."""

    @staticmethod
    def _allowed(mod, q, kv):
        return bool(mod(torch.tensor(0), torch.tensor(0), torch.tensor(q), torch.tensor(kv)))

    @pytest.mark.unit
    def test_bounded_window(self):
        """left=2, right=1 -> query q attends to kv in [q - 2, q + 1]."""
        mod = _make_sliding_window_mod(2, 1)
        assert self._allowed(mod, 5, 3)  # lower edge
        assert self._allowed(mod, 5, 5)  # self
        assert self._allowed(mod, 5, 6)  # upper edge (look-ahead)
        assert not self._allowed(mod, 5, 2)  # beyond left context
        assert not self._allowed(mod, 5, 7)  # beyond right context

    @pytest.mark.unit
    def test_unlimited_left_is_causal(self):
        """left<0, right=0 -> unlimited past, no look-ahead (causal)."""
        mod = _make_sliding_window_mod(-1, 0)
        assert self._allowed(mod, 5, 0)  # far past allowed
        assert self._allowed(mod, 5, 5)  # self allowed
        assert not self._allowed(mod, 5, 6)  # future masked

    @pytest.mark.unit
    def test_unlimited_right_only_left_bound(self):
        """right<0, left=1 -> unlimited future, only one frame of past."""
        mod = _make_sliding_window_mod(1, -1)
        assert self._allowed(mod, 5, 4)  # one frame back allowed
        assert self._allowed(mod, 5, 99)  # far future allowed
        assert not self._allowed(mod, 5, 3)  # two frames back masked


class TestStreamingTransformerEncoder:
    """Tests for the sliding-window streaming encoder and its cache-aware interface."""

    @pytest.mark.unit
    def test_satisfies_streaming_encoder_interface(self):
        enc = StreamingTransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2)
        # Must pass the isinstance(encoder, StreamingEncoder) gate in AudioPerceptionModule.
        assert isinstance(enc, StreamingEncoder)
        assert isinstance(enc, TransformerEncoder)
        for method in ("cache_aware_stream_step", "get_initial_cache_state", "setup_streaming_params"):
            assert callable(getattr(enc, method))

    @pytest.mark.unit
    def test_streaming_cfg_tracks_att_context(self):
        """``streaming_cfg`` is (re)built from ``att_context_size`` — the left context sizes the
        rolling cache; FeatureStacking needs no pre-encode overlap."""
        enc = StreamingTransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, att_context_size=[7, 0])
        assert enc.streaming_cfg.pre_encode_cache_size == 0
        assert enc.streaming_cfg.last_channel_cache_size == 7
        # Retuning the window rebuilds the cfg.
        enc.set_default_att_context_size([12, 0])
        assert enc.streaming_cfg.last_channel_cache_size == 12

    @pytest.mark.unit
    def test_initial_cache_state_shapes(self):
        """A rolling cache pre-allocates ``left`` frames per layer (padded); ``cache_last_time`` is
        a zero-width placeholder (no conv) and valid length starts at 0."""
        d_model, n_layers, left, B = 64, 3, 7, 2
        enc = StreamingTransformerEncoder(
            feat_in=80, d_model=d_model, n_heads=4, n_layers=n_layers, att_context_size=[left, 0]
        )
        clc, clt, clcl = enc.get_initial_cache_state(batch_size=B)
        assert clc.shape == (n_layers, B, left, d_model)
        assert clt.shape == (n_layers, B, 0)
        assert clcl.shape == (B,)
        assert clcl.sum().item() == 0
        # A full cache (left < 0) starts empty and grows.
        enc.set_default_att_context_size([-1, 0])
        clc_full, _, _ = enc.get_initial_cache_state(batch_size=B)
        assert clc_full.shape == (n_layers, B, 0, d_model)

    @pytest.mark.unit
    def test_default_att_context_size_is_full(self):
        enc = StreamingTransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2)
        assert enc.att_context_size == [-1, -1]

    @pytest.mark.unit
    def test_set_default_att_context_size(self):
        enc = StreamingTransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, att_context_size=[70, 1])
        assert enc.att_context_size == [70, 1]
        # StreamingSTTModel retunes the look-ahead per chunk by reassigning this attribute.
        enc.set_default_att_context_size([70, 5])
        assert enc.att_context_size == [70, 5]

    @pytest.mark.unit
    def test_invalid_att_context_size_raises(self):
        with pytest.raises(ValueError, match=r"\[left, right\] pair"):
            StreamingTransformerEncoder(feat_in=80, d_model=64, n_heads=4, n_layers=2, att_context_size=[1, 2, 3])

    @pytest.mark.unit
    def test_attn_mode_kwarg_is_ignored(self):
        """Unlike the base (which rejects it), the streaming encoder swallows ``attn_mode``."""
        enc = StreamingTransformerEncoder(
            feat_in=80, d_model=64, n_heads=4, n_layers=2, attn_mode="sliding_window", att_context_size=[3, 0]
        )
        assert enc.attn_mode == "sliding_window"
        assert enc.att_context_size == [3, 0]

    @pytest.mark.unit
    def test_forward_cpu_shape(self):
        enc = StreamingTransformerEncoder(
            feat_in=128, d_model=64, n_heads=4, n_layers=2, drop_rate=0.0, att_context_size=[10, 1]
        )
        enc.eval()

        B, C, T = 2, 128, 200
        x = torch.randn(B, C, T)
        lengths = torch.tensor([T, 160])

        with torch.no_grad():
            out, out_lengths = enc(audio_signal=x, length=lengths)

        assert out.shape == (B, 64, T // 4)
        assert out_lengths.tolist() == [T // 4, 160 // 4]
        assert not torch.isnan(out).any()

    @pytest.mark.unit
    def test_sliding_window_limits_receptive_field(self):
        """With a single layer and window [1, 1], output at frame i must be invariant to
        input changes outside [i - 1, i + 1]. Uses ``bypass_pre_encode`` + ``no_pos`` so
        frame indices map 1:1 and no positional mixing obscures the receptive field."""
        B, T, d_model = 1, 8, 64
        enc = StreamingTransformerEncoder(
            feat_in=d_model,
            d_model=d_model,
            n_heads=4,
            n_layers=1,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            self_attention_model="no_pos",
            att_context_size=[1, 1],
        )
        enc.eval()

        x = torch.randn(B, T, d_model)
        length = torch.tensor([T], dtype=torch.int64)
        x_perturbed = x.clone()
        x_perturbed[:, 5, :] = torch.randn(d_model)  # change only frame 5

        with torch.no_grad():
            out, _ = enc(audio_signal=x, length=length, bypass_pre_encode=True)
            out_perturbed, _ = enc(audio_signal=x_perturbed, length=length, bypass_pre_encode=True)

        # out is (B, d_model, T). Frame 2's window {1, 2, 3} excludes frame 5 -> unchanged.
        assert torch.allclose(out[:, :, 2], out_perturbed[:, :, 2], atol=1e-6)
        # Frame 4's window {3, 4, 5} includes frame 5 -> must change.
        assert not torch.allclose(out[:, :, 4], out_perturbed[:, :, 4], atol=1e-6)

    @pytest.mark.unit
    def test_full_window_attends_everywhere(self):
        """Contrast to the windowed case: att_context_size [-1, -1] is full attention, so a
        distant perturbation *does* reach every output frame."""
        B, T, d_model = 1, 8, 64
        enc = StreamingTransformerEncoder(
            feat_in=d_model,
            d_model=d_model,
            n_heads=4,
            n_layers=1,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            self_attention_model="no_pos",
            att_context_size=[-1, -1],
        )
        enc.eval()

        x = torch.randn(B, T, d_model)
        length = torch.tensor([T], dtype=torch.int64)
        x_perturbed = x.clone()
        x_perturbed[:, 5, :] = torch.randn(d_model)

        with torch.no_grad():
            out, _ = enc(audio_signal=x, length=length, bypass_pre_encode=True)
            out_perturbed, _ = enc(audio_signal=x_perturbed, length=length, bypass_pre_encode=True)

        assert not torch.allclose(out[:, :, 2], out_perturbed[:, :, 2], atol=1e-6)

    @staticmethod
    def _stream_sequence(enc, x, chunk_len, n_chunks, batch_size, bypass_pre_encode):
        """Feed ``x`` through the cache-aware streaming path chunk by chunk and return the
        concatenated encoder output ``(B, D, T')``. ``x`` is ``(B, T, d_model)`` when
        ``bypass_pre_encode`` else ``(B, feat_in, T_in)``; ``chunk_len`` is in input frames."""
        clc, clt, clcl = enc.get_initial_cache_state(batch_size=batch_size, dtype=x.dtype, device=x.device)
        outs = []
        for c in range(n_chunks):
            if bypass_pre_encode:
                chunk = x[:, c * chunk_len : (c + 1) * chunk_len, :]
            else:
                chunk = x[:, :, c * chunk_len : (c + 1) * chunk_len]
            chunk_frames = chunk.shape[1] if bypass_pre_encode else chunk.shape[2]
            clen = torch.tensor([chunk_frames] * batch_size, dtype=torch.int64)
            enc_out, _, clc, clt, clcl = enc.cache_aware_stream_step(
                processed_signal=chunk,
                processed_signal_length=clen,
                cache_last_channel=clc,
                cache_last_time=clt,
                cache_last_channel_len=clcl,
                bypass_pre_encode=bypass_pre_encode,
            )
            outs.append(enc_out)
        return torch.cat(outs, dim=2), clcl

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "sam, left, chunk, n_chunks, batch, bypass",
        [
            ("rel_pos", 3, 2, 4, 1, True),  # warm-up: cache fills over several chunks (chunk < left)
            ("rel_pos", 2, 4, 4, 1, True),  # rolling: chunk > left
            ("rel_pos", 3, 2, 4, 2, True),  # batched
            ("no_pos", 3, 2, 4, 1, True),  # no positional encoding
            ("rel_pos", -1, 2, 4, 1, True),  # full (unbounded) cache
            ("rel_pos", 3, 2, 4, 1, False),  # with FeatureStacking subsampling (aligned chunks)
        ],
    )
    def test_streaming_matches_full_forward(self, sam, left, chunk, n_chunks, batch, bypass):
        """The defining guarantee: causal chunk-by-chunk streaming with the KV cache must equal the
        full-sequence causal forward. Requires ``right == 0`` (a frame never needs a future frame)."""
        torch.manual_seed(0)
        d_model, sub, feat_in = 64, 4, 32
        enc = StreamingTransformerEncoder(
            feat_in=feat_in if not bypass else d_model,
            d_model=d_model,
            n_heads=4,
            n_layers=2,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            self_attention_model=sam,
            subsampling_factor=sub,
            att_context_size=[left, 0],
        )
        enc.eval()

        if bypass:
            x = torch.randn(batch, chunk * n_chunks, d_model)
            chunk_len = chunk
            in_len = x.shape[1]
        else:
            # ``chunk`` encoder frames == ``chunk * sub`` input frames; keep chunks aligned to the
            # subsampling factor so each chunk subsamples independently (matches the full forward).
            x = torch.randn(batch, feat_in, chunk * sub * n_chunks)
            chunk_len = chunk * sub
            in_len = x.shape[2]
        lengths = torch.tensor([in_len] * batch, dtype=torch.int64)

        with torch.no_grad():
            full_out, _ = enc(audio_signal=x, length=lengths, bypass_pre_encode=bypass)
            stream_out, final_valid_len = self._stream_sequence(enc, x, chunk_len, n_chunks, batch, bypass)

        assert stream_out.shape == full_out.shape
        assert torch.allclose(full_out, stream_out, atol=1e-5)
        # Valid cache length saturates at ``left`` for a rolling cache (whole utterance for full).
        expected = chunk * n_chunks if left < 0 else min(chunk * n_chunks, left)
        assert final_valid_len.tolist() == [expected] * batch

    @pytest.mark.unit
    def test_streaming_abs_pos_not_implemented(self):
        """``abs_pos`` streaming is intentionally unsupported (no position offset in the cache)."""
        enc = StreamingTransformerEncoder(
            feat_in=64, d_model=64, n_heads=4, n_layers=1, self_attention_model="abs_pos", att_context_size=[3, 0]
        )
        enc.eval()
        clc, clt, clcl = enc.get_initial_cache_state(batch_size=1)
        with torch.no_grad(), pytest.raises(NotImplementedError, match="Cache-aware streaming"):
            enc.cache_aware_stream_step(
                processed_signal=torch.randn(1, 2, 64),
                processed_signal_length=torch.tensor([2]),
                cache_last_channel=clc,
                cache_last_time=clt,
                cache_last_channel_len=clcl,
                bypass_pre_encode=True,
            )
