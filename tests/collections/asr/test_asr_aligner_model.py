# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.models import EncDecAlignerModel
from nemo.collections.asr.modules import AlignerCTCHead, AlignerJoint

# fmt: off
LABELS = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
          'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'"]
# fmt: on


def _build_model(aligner_type: str, aux_nonar_loss_weight: float = 0.0) -> EncDecAlignerModel:
    model_defaults = {'enc_hidden': 128, 'pred_hidden': 64, 'joint_hidden': 64}

    preprocessor = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'features': 64}

    encoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASREncoder',
        'feat_in': 64,
        'activation': 'relu',
        'conv_mask': True,
        'jasper': [
            {
                'filters': model_defaults['enc_hidden'],
                'repeat': 1,
                'kernel': [1],
                'stride': [1],
                'dilation': [1],
                'dropout': 0.0,
                'residual': False,
                'separable': True,
                'se': True,
                'se_context_size': -1,
            }
        ],
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
        'prednet': {'pred_hidden': model_defaults['pred_hidden'], 'pred_rnn_layers': 1},
    }

    joint = {
        '_target_': 'nemo.collections.asr.modules.AlignerJoint',
        'jointnet': {'joint_hidden': model_defaults['joint_hidden'], 'activation': 'relu'},
    }

    decoding = {'aligner_type': aligner_type, 'max_symbols': None}

    cfg = DictConfig(
        {
            'labels': ListConfig(LABELS),
            'aligner_type': aligner_type,
            'label_smoothing': 0.1,
            'aux_nonar_loss_weight': aux_nonar_loss_weight,
            'compute_eval_loss': True,
            'preprocessor': DictConfig(preprocessor),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'joint': DictConfig(joint),
            'ctc_head': DictConfig({'hidden': None, 'activation': 'relu', 'dropout': 0.0}),
            'decoding': DictConfig(decoding),
        }
    )
    return EncDecAlignerModel(cfg=cfg)


class TestEncDecAlignerModel:
    @pytest.mark.unit
    def test_constructor_ar(self):
        model = _build_model('ar')
        assert model.eos_id == len(LABELS)
        assert isinstance(model.joint, AlignerJoint)
        assert model.joint.num_classes == len(LABELS) + 1
        # AR model does not need a per-frame head unless aux loss is enabled.
        assert model.ctc_head is None

    @pytest.mark.unit
    def test_constructor_nonar(self):
        model = _build_model('nonar')
        assert isinstance(model.ctc_head, AlignerCTCHead)
        assert model.ctc_head.num_classes == len(LABELS) + 1

    @pytest.mark.unit
    def test_append_eos(self):
        model = _build_model('ar')
        transcript = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        transcript_len = torch.tensor([3, 2])
        targets, target_len = model._append_eos(transcript, transcript_len)
        assert torch.equal(target_len, torch.tensor([4, 3]))
        assert targets[0, 3].item() == model.eos_id
        assert targets[1, 2].item() == model.eos_id

    @pytest.mark.unit
    @pytest.mark.parametrize("aligner_type", ['ar', 'nonar'])
    def test_train_loss_is_finite(self, aligner_type):
        model = _build_model(aligner_type)
        model.train()
        batch_size, audio_len = 4, 4000
        signal = torch.randn(batch_size, audio_len)
        signal_len = torch.full((batch_size,), audio_len, dtype=torch.long)
        transcript = torch.randint(0, len(LABELS), (batch_size, 8))
        transcript_len = torch.tensor([8, 7, 6, 5])

        encoded, encoded_len = model.forward(input_signal=signal, input_signal_length=signal_len)
        loss, logs = model._aligner_loss(encoded, encoded_len, transcript, transcript_len)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0
        loss.backward()
        # Encoder must receive gradients from the (cross-entropy) loss.
        grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() for g in grads)

    @pytest.mark.unit
    @pytest.mark.parametrize("aligner_type", ['ar', 'nonar'])
    def test_greedy_decode_shapes(self, aligner_type):
        model = _build_model(aligner_type)
        model.eval()
        batch_size, audio_len = 2, 4000
        signal = torch.randn(batch_size, audio_len)
        signal_len = torch.full((batch_size,), audio_len, dtype=torch.long)

        with torch.no_grad():
            encoded, encoded_len = model.forward(input_signal=signal, input_signal_length=signal_len)
            texts, token_ids = model.decoding.decode_encoder_output(encoded, encoded_len)

        assert len(texts) == batch_size
        assert len(token_ids) == batch_size
        for ids in token_ids:
            # EOS must never appear inside the emitted hypothesis.
            assert model.eos_id not in ids
            # One token per frame at most.
            assert len(ids) <= int(encoded_len.max().item())
        assert all(isinstance(t, str) for t in texts)

    @pytest.mark.unit
    def test_aux_nonar_loss_builds_head(self):
        model = _build_model('ar', aux_nonar_loss_weight=0.3)
        assert isinstance(model.ctc_head, AlignerCTCHead)
        signal = torch.randn(2, 4000)
        signal_len = torch.full((2,), 4000, dtype=torch.long)
        transcript = torch.randint(0, len(LABELS), (2, 6))
        transcript_len = torch.tensor([6, 5])
        encoded, encoded_len = model.forward(input_signal=signal, input_signal_length=signal_len)
        loss, logs = model._aligner_loss(encoded, encoded_len, transcript, transcript_len)
        assert 'ar_loss' in logs and 'nonar_loss' in logs
        assert torch.isfinite(loss)
