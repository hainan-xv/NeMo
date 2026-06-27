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
from omegaconf import DictConfig

from nemo.collections.asr.modules import RNNTDecoder, RNNTJoint
from nemo.collections.asr.parts.submodules.chunked_aligner_decoding import ChunkedAlignerDecoding


def _build_decoding(vocab_size: int, chunk_size: int, decoding_cfg=None, enc_hidden=32, pred_hidden=16, joint_hidden=16):
    """Build a ChunkedAlignerDecoding backed by small real RNN-T decoder + joint modules."""
    decoder = RNNTDecoder(
        prednet={'pred_hidden': pred_hidden, 'pred_rnn_layers': 1, 'dropout': 0.0},
        vocab_size=vocab_size,
    )
    joint = RNNTJoint(
        jointnet={
            'encoder_hidden': enc_hidden,
            'pred_hidden': pred_hidden,
            'joint_hidden': joint_hidden,
            'activation': 'relu',
            'dropout': 0.0,
        },
        num_classes=vocab_size,
        vocabulary=[str(i) for i in range(vocab_size)],
    )
    decoder.eval()
    joint.eval()

    blank_id = joint.num_classes_with_blank - 1  # == vocab_size
    if decoding_cfg is not None:
        decoding_cfg = DictConfig(decoding_cfg)
    return ChunkedAlignerDecoding(
        decoding_cfg=decoding_cfg,
        decoder=decoder,
        joint=joint,
        blank_id=blank_id,
        chunk_size=chunk_size,
        vocabulary=[str(i) for i in range(vocab_size)],
    ), enc_hidden


class TestChunkedAlignerDecoding:
    @pytest.mark.unit
    @pytest.mark.parametrize('chunk_size', [1, 2, 3, 4, 12])
    @pytest.mark.parametrize('seed', [0, 1, 2])
    def test_batched_matches_per_utterance(self, chunk_size, seed):
        """The batched label-looping walk must be token-identical to the reference walk."""
        torch.manual_seed(seed)
        vocab_size = 6
        B, T = 5, 17

        dec, enc_hidden = _build_decoding(vocab_size, chunk_size)

        encoder_output = torch.randn(B, enc_hidden, T)
        # Varied per-utterance lengths, including a zero-length and full-length sample.
        encoded_lengths = torch.tensor([T, T - 1, T - 5, 0, 3][:B], dtype=torch.long)

        with torch.no_grad():
            ref = dec._chunked_greedy(encoder_output, encoded_lengths)
            bat = dec._chunked_greedy_batched(encoder_output, encoded_lengths)

        assert ref == bat, f"batched vs per-utterance hypotheses differ (chunk_size={chunk_size}): {ref} != {bat}"

    @pytest.mark.unit
    def test_batched_matches_per_utterance_uniform_lengths(self):
        torch.manual_seed(7)
        vocab_size = 8
        B, T, chunk_size = 4, 24, 5

        dec, enc_hidden = _build_decoding(vocab_size, chunk_size)
        encoder_output = torch.randn(B, enc_hidden, T)
        encoded_lengths = torch.full((B,), T, dtype=torch.long)

        with torch.no_grad():
            ref = dec._chunked_greedy(encoder_output, encoded_lengths)
            bat = dec._chunked_greedy_batched(encoder_output, encoded_lengths)
        assert ref == bat

    @pytest.mark.unit
    def test_max_symbols_cap_respected(self):
        torch.manual_seed(3)
        vocab_size = 6
        B, T, chunk_size = 3, 20, 2
        max_symbols = 4

        dec, enc_hidden = _build_decoding(
            vocab_size, chunk_size, decoding_cfg={'max_symbols': max_symbols, 'loop_labels': True}
        )
        encoder_output = torch.randn(B, enc_hidden, T)
        encoded_lengths = torch.full((B,), T, dtype=torch.long)

        with torch.no_grad():
            ref = dec._chunked_greedy(encoder_output, encoded_lengths)
            bat = dec._chunked_greedy_batched(encoder_output, encoded_lengths)
        assert ref == bat
        for ids in bat:
            assert len(ids) <= max_symbols

    @pytest.mark.unit
    def test_decode_entrypoint_uses_loop_labels(self):
        """decode_encoder_output should route through the batched walk by default."""
        torch.manual_seed(0)
        vocab_size = 6
        B, T, chunk_size = 3, 15, 3

        dec, enc_hidden = _build_decoding(vocab_size, chunk_size, decoding_cfg={'max_symbols': None})
        assert dec.loop_labels is True
        encoder_output = torch.randn(B, enc_hidden, T)
        encoded_lengths = torch.tensor([T, T - 2, 6], dtype=torch.long)

        with torch.no_grad():
            texts, token_ids = dec.decode_encoder_output(encoder_output, encoded_lengths)
            ref = dec._chunked_greedy(encoder_output, encoded_lengths)
        assert token_ids == ref
        assert len(texts) == B

    @pytest.mark.unit
    @pytest.mark.parametrize(
        'cfg',
        [
            {'use_cuda_graph_decoder': True},
            {'greedy': {'use_cuda_graph_decoder': True}},
        ],
    )
    def test_cuda_graph_request_warns_and_disables(self, monkeypatch, cfg):
        """Requesting CUDA-graph decoding (top-level or under greedy) logs a warning."""
        from nemo.collections.asr.parts.submodules import chunked_aligner_decoding as cad

        warnings = []
        monkeypatch.setattr(cad.logging, 'warning', lambda msg, *a, **k: warnings.append(str(msg)))
        _build_decoding(6, 3, decoding_cfg=cfg)
        assert any('CUDA-graph' in w for w in warnings)

    @pytest.mark.unit
    def test_no_cuda_graph_request_no_warning(self, monkeypatch):
        from nemo.collections.asr.parts.submodules import chunked_aligner_decoding as cad

        warnings = []
        monkeypatch.setattr(cad.logging, 'warning', lambda msg, *a, **k: warnings.append(str(msg)))
        _build_decoding(6, 3, decoding_cfg={'max_symbols': None})
        assert not any('CUDA-graph' in w for w in warnings)
