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

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer, UnitaryResidual


def _encoder_kwargs(d_model=16, n_layers=2, **overrides):
    """Small, deterministic ConformerEncoder config for fast unit tests."""
    kwargs = dict(
        feat_in=16,
        n_layers=n_layers,
        d_model=d_model,
        feat_out=8,
        dropout=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
        dropout_att=0.0,
        conv_norm_type="layer_norm",
        conv_kernel_size=3,
    )
    kwargs.update(overrides)
    return kwargs


def _all_unitary_modules(model):
    return [m for m in model.modules() if isinstance(m, UnitaryResidual)]


def _assert_orthogonal(q, atol=1e-5):
    assert q.shape[-1] == q.shape[-2]
    d = q.shape[-1]
    identity = torch.eye(d, dtype=q.dtype, device=q.device)
    # Columns (and rows) are orthonormal: Q^T Q = Q Q^T = I.
    assert torch.allclose(q.transpose(-1, -2) @ q, identity, atol=atol), "Q^T Q != I"
    assert torch.allclose(q @ q.transpose(-1, -2), identity, atol=atol), "Q Q^T != I"
    # expm of a skew-symmetric matrix lies in SO(d): det == +1. The determinant is far more
    # numerically sensitive than the orthogonality residual (it multiplies d eigenvalues), so it
    # gets a looser float32 tolerance.
    assert torch.allclose(torch.linalg.det(q), torch.tensor(1.0, dtype=q.dtype), atol=1e-3)


class TestUnitaryResidualModule:
    """Unit tests for the UnitaryResidual building block, focused on the unitary property."""

    @pytest.mark.parametrize("d_model", [2, 4, 8, 16])
    def test_identity_at_initialization(self, d_model):
        """Zero-init weight => Q = expm(0) = I, so the module is the identity map at init."""
        module = UnitaryResidual(d_model)
        q = module.get_orthogonal_matrix()
        assert torch.allclose(q, torch.eye(d_model), atol=1e-6)

        x = torch.randn(3, 5, d_model)
        assert torch.allclose(module(x), x, atol=1e-6)

    @pytest.mark.parametrize("d_model", [2, 4, 8, 16])
    def test_orthogonal_at_init_and_after_perturbation(self, d_model):
        """Q must be orthogonal for ANY value of the underlying weight, not just at init."""
        module = UnitaryResidual(d_model)
        _assert_orthogonal(module.get_orthogonal_matrix())

        # Arbitrary (non-skew, non-zero) weights must still yield an orthogonal Q.
        for seed in range(5):
            torch.manual_seed(seed)
            with torch.no_grad():
                module.weight.copy_(torch.randn(d_model, d_model) * 3.0)
            _assert_orthogonal(module.get_orthogonal_matrix())

    @pytest.mark.parametrize("d_model", [4, 8, 16])
    def test_norm_preservation(self, d_model):
        """An orthogonal map preserves the L2 norm of every vector it acts on."""
        module = UnitaryResidual(d_model)
        with torch.no_grad():
            module.weight.copy_(torch.randn(d_model, d_model))

        x = torch.randn(4, 7, d_model)
        y = module(x)
        assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)

    def test_forward_matches_matrix_multiply(self):
        """forward(x) must equal applying Q to each (column) vector: y = Q x."""
        d_model = 8
        module = UnitaryResidual(d_model)
        with torch.no_grad():
            module.weight.copy_(torch.randn(d_model, d_model))
        q = module.get_orthogonal_matrix()

        v = torch.randn(d_model)
        assert torch.allclose(module(v), q @ v, atol=1e-5)

        batch = torch.randn(2, 3, d_model)
        expected = torch.matmul(batch, q.transpose(-1, -2))
        assert torch.allclose(module(batch), expected, atol=1e-5)

    def test_gradients_flow_and_orthogonality_survives_optimizer_step(self):
        """The weight must be trainable, and Q must remain orthogonal after an update."""
        d_model = 8
        module = UnitaryResidual(d_model)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.5)

        target = torch.randn(4, d_model)
        x = torch.randn(4, d_model)
        for _ in range(3):
            optimizer.zero_grad()
            loss = (module(x) - target).pow(2).mean()
            loss.backward()
            assert module.weight.grad is not None
            assert torch.isfinite(module.weight.grad).all()
            assert module.weight.grad.abs().sum() > 0
            optimizer.step()
            # Orthogonality is guaranteed by construction at every step.
            _assert_orthogonal(module.get_orthogonal_matrix(), atol=1e-4)

    def test_dtype_follows_input(self):
        module = UnitaryResidual(4)
        x64 = torch.randn(2, 4, dtype=torch.float64)
        assert module(x64).dtype == torch.float64
        x32 = torch.randn(2, 4, dtype=torch.float32)
        assert module(x32).dtype == torch.float32


class TestConformerLayerUnitaryResidual:
    """Tests at the ConformerLayer level."""

    def test_layer_creates_four_unitary_transforms_when_enabled(self):
        layer = ConformerLayer(d_model=8, d_ff=16, self_attention_model="abs_pos", use_unitary_residual=True)
        assert layer.use_unitary_residual is True
        for name in ["unitary_feed_forward1", "unitary_self_att", "unitary_conv", "unitary_feed_forward2"]:
            assert isinstance(getattr(layer, name), UnitaryResidual)

    def test_layer_has_no_unitary_transforms_by_default(self):
        layer = ConformerLayer(d_model=8, d_ff=16, self_attention_model="abs_pos")
        assert layer.use_unitary_residual is False
        assert not any(isinstance(m, UnitaryResidual) for m in layer.modules())

    def test_layer_forward_identity_at_init_matches_plain_layer(self):
        """At init the unitary transforms are identity, so a flagged layer == a plain layer."""
        torch.manual_seed(0)
        layer_u = ConformerLayer(d_model=8, d_ff=16, self_attention_model="abs_pos", dropout=0.0, use_unitary_residual=True)
        torch.manual_seed(0)
        layer_p = ConformerLayer(d_model=8, d_ff=16, self_attention_model="abs_pos", dropout=0.0)
        # Share all common weights; the unitary_* keys are simply extra and ignored.
        layer_p.load_state_dict(layer_u.state_dict(), strict=False)
        layer_u.eval()
        layer_p.eval()

        x = torch.randn(2, 6, 8)
        out_u = layer_u(x, att_mask=None, pos_emb=None, pad_mask=None)
        out_p = layer_p(x, att_mask=None, pos_emb=None, pad_mask=None)
        assert torch.allclose(out_u, out_p, atol=1e-5)


class TestConformerEncoderUnitaryResidual:
    """End-to-end tests through ConformerEncoder."""

    def test_backward_compatible_by_default(self):
        model = ConformerEncoder(**_encoder_kwargs())
        assert model.use_unitary_residual is False
        assert not any("unitary" in name for name, _ in model.named_parameters())

    def test_enabled_adds_expected_number_of_unitary_params(self):
        n_layers = 3
        model = ConformerEncoder(**_encoder_kwargs(n_layers=n_layers, use_unitary_residual=True))
        unitary_modules = _all_unitary_modules(model)
        # 4 residual sites per ConformerLayer.
        assert len(unitary_modules) == 4 * n_layers

    @pytest.mark.parametrize("self_attention_model", ["rel_pos", "abs_pos"])
    def test_forward_runs_and_shapes_match(self, self_attention_model):
        batch_size, n_frames, d_model, feat_out = 2, 17, 16, 8
        model = ConformerEncoder(
            **_encoder_kwargs(d_model=d_model, feat_out=feat_out, self_attention_model=self_attention_model, use_unitary_residual=True)
        )
        x = torch.rand(batch_size, n_frames, d_model)
        length = torch.tensor([n_frames, n_frames], dtype=torch.int64)

        model.train()
        out = model(audio_signal=x, length=length, bypass_pre_encode=True)[0]
        assert out.shape == (batch_size, feat_out, n_frames)
        assert torch.isfinite(out).all()

        model.eval()
        out = model(audio_signal=x, length=length, bypass_pre_encode=True)[0]
        assert out.shape == (batch_size, feat_out, n_frames)

    def test_identity_at_init_matches_standard_encoder(self):
        """A flagged encoder at init must reproduce a standard encoder sharing the same weights."""
        torch.manual_seed(123)
        enc_u = ConformerEncoder(**_encoder_kwargs(use_unitary_residual=True))
        enc_p = ConformerEncoder(**_encoder_kwargs(use_unitary_residual=False))
        enc_p.load_state_dict(enc_u.state_dict(), strict=False)
        enc_u.eval()
        enc_p.eval()

        x = torch.rand(2, 17, 16)
        length = torch.tensor([17, 17], dtype=torch.int64)
        out_u = enc_u(audio_signal=x, length=length, bypass_pre_encode=True)[0]
        out_p = enc_p(audio_signal=x, length=length, bypass_pre_encode=True)[0]
        assert torch.allclose(out_u, out_p, atol=1e-5)

        # Once the unitary weights become non-trivial, the output must change.
        with torch.no_grad():
            for name, param in enc_u.named_parameters():
                if "unitary" in name:
                    param.normal_(0.0, 0.2)
        out_u_rot = enc_u(audio_signal=x, length=length, bypass_pre_encode=True)[0]
        assert not torch.allclose(out_u_rot, out_p, atol=1e-4)

    def test_unitary_modules_stay_orthogonal_after_training_step(self):
        model = ConformerEncoder(**_encoder_kwargs(use_unitary_residual=True))
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        x = torch.rand(2, 17, 16)
        length = torch.tensor([17, 17], dtype=torch.int64)
        out = model(audio_signal=x, length=length, bypass_pre_encode=True)[0]
        loss = out.pow(2).mean()
        loss.backward()

        for m in _all_unitary_modules(model):
            assert m.weight.grad is not None

        optimizer.step()

        for m in _all_unitary_modules(model):
            _assert_orthogonal(m.get_orthogonal_matrix(), atol=1e-4)
