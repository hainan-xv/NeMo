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

"""Neural modules for the Aligner-Encoder ASR architecture.

Reference: Stooke et al., "Aligner-Encoders: Self-Attention Transformers Can Be
Self-Transducers" (https://arxiv.org/abs/2502.05232).

The Aligner-Encoder reuses the encoder and prediction network of an RNN-T, but
the encoder is expected to perform the audio-to-text alignment internally during
its forward pass ("self-transduction"). As a result:

* The joint network combines the acoustic embedding ``h_i`` with the text
  embedding ``g_i`` in a strict **one-to-one** fashion (the main diagonal of the
  RNN-T lattice) instead of forming the full ``(B, T, U, V)`` outer product.
* There is **no blank token** -- the vocabulary contains the real tokens plus a
  single end-of-sentence (EOS) token, and the model is trained with a frame-wise
  cross-entropy loss.

This module provides:

* :class:`AlignerJoint` -- the one-to-one joint used by the autoregressive model.
* :class:`AlignerCTCHead` -- the per-frame head ``f_ind(h_i)`` used by the
  non-autoregressive (CTC-like) variant.
"""

from typing import List, Optional

import torch
from omegaconf import DictConfig

from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import AcousticEncodedRepresentation, EmbeddedTextType, LogprobsType, NeuralType

__all__ = ['AlignerJoint', 'AlignerCTCHead']


def _resolve_activation(activation: str) -> torch.nn.Module:
    activation = activation.lower()
    if activation == 'relu':
        return torch.nn.ReLU(inplace=True)
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    raise ValueError(f"Unsupported activation '{activation}'. Choose one of [relu, sigmoid, tanh].")


class AlignerJoint(NeuralModule):
    """One-to-one joint network for the Aligner-Encoder.

    Given the encoder output ``f`` of shape ``(B, H_enc, T)`` and the prediction
    network output ``g`` of shape ``(B, H_pred, U)``, this module pairs frame
    ``i`` of the encoder with step ``i`` of the prediction network and produces
    per-position logits of shape ``(B, U, V)``, where ``V`` is the number of
    output classes (real tokens + EOS, **no** blank).

    Unlike :class:`~nemo.collections.asr.modules.RNNTJoint`, the time and label
    axes are collapsed into a single axis -- only the diagonal of the lattice is
    realized. When ``T >= U`` the encoder output is truncated to its first ``U``
    frames; when ``T < U`` it is right-padded with zeros (the corresponding
    positions are expected to be masked out by the loss).

    Args:
        jointnet: Dict-like config with the keys ``encoder_hidden``,
            ``pred_hidden``, ``joint_hidden``, and optionally ``activation``
            (default ``relu``) and ``dropout`` (default ``0.0``).
        num_classes: Number of output classes including EOS but excluding blank.
        vocabulary: Optional list of tokens used by downstream decoding/WER.
        log_softmax: If ``None`` (default) log-softmax is applied automatically
            only on CPU tensors (mirrors RNNTJoint behavior). If ``True``/``False``
            it forces/disables log-softmax on all devices.
    """

    @property
    def input_types(self):
        return {
            "encoder_outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "decoder_outputs": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
        }

    @property
    def output_types(self):
        return {"outputs": NeuralType(('B', 'T', 'D'), LogprobsType())}

    def __init__(
        self,
        jointnet: DictConfig,
        num_classes: int,
        vocabulary: Optional[List] = None,
        log_softmax: Optional[bool] = None,
    ):
        super().__init__()

        self.vocabulary = vocabulary
        self._num_classes = num_classes
        self.log_softmax = log_softmax

        enc_hidden = jointnet['encoder_hidden']
        pred_hidden = jointnet['pred_hidden']
        joint_hidden = jointnet['joint_hidden']
        activation = jointnet.get('activation', 'relu')
        dropout = jointnet.get('dropout', 0.0)

        self.enc = torch.nn.Linear(enc_hidden, joint_hidden)
        self.pred = torch.nn.Linear(pred_hidden, joint_hidden)

        layers = (
            [_resolve_activation(activation)]
            + ([torch.nn.Dropout(p=dropout)] if dropout else [])
            + [torch.nn.Linear(joint_hidden, num_classes)]
        )
        self.joint_net = torch.nn.Sequential(*layers)

    @typecheck()
    def forward(self, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor) -> torch.Tensor:
        # encoder_outputs: (B, H_enc, T) -> (B, T, H_enc); decoder_outputs: (B, H_pred, U) -> (B, U, H_pred)
        f = self.enc(encoder_outputs.transpose(1, 2))
        g = self.pred(decoder_outputs.transpose(1, 2))
        return self.joint_after_projection(f, g)

    def joint_after_projection(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Combine projected encoder/prediction tensors one-to-one.

        Args:
            f: Projected encoder output of shape ``(B, T, H)``.
            g: Projected prediction output of shape ``(B, U, H)``.

        Returns:
            Logits (or log-probs) of shape ``(B, U, V)``.
        """
        u = g.size(1)
        t = f.size(1)

        if t < u:
            pad = f.new_zeros(f.size(0), u - t, f.size(2))
            f = torch.cat([f, pad], dim=1)
        f = f[:, :u, :]

        inp = f + g
        res = self.joint_net(inp)

        if self.log_softmax is None:
            if not res.is_cuda:
                res = res.log_softmax(dim=-1)
        elif self.log_softmax:
            res = res.log_softmax(dim=-1)

        return res

    @property
    def num_classes_with_blank(self):
        # The Aligner-Encoder has no blank token; this property exists for API
        # compatibility with code that inspects RNN-T-style joints.
        return self._num_classes

    @property
    def num_classes(self):
        return self._num_classes


class AlignerCTCHead(NeuralModule):
    """Per-frame classification head for the non-autoregressive Aligner.

    Implements ``f_ind(h_i)`` from the paper: it maps each encoder frame to a
    distribution over the output vocabulary independently, as in CTC but without
    a blank token. The model emits one token per frame and discards everything
    after the first EOS at inference time.

    Args:
        feat_in: Encoder output dimension.
        num_classes: Number of output classes including EOS, excluding blank.
        hidden: Optional hidden size. If provided, a 2-layer MLP is used,
            otherwise a single linear projection.
        activation: Activation for the optional hidden layer.
        dropout: Dropout applied before the final projection.
    """

    @property
    def input_types(self):
        return {"encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation())}

    @property
    def output_types(self):
        return {"logits": NeuralType(('B', 'T', 'D'), LogprobsType())}

    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        hidden: Optional[int] = None,
        activation: str = 'relu',
        dropout: float = 0.0,
    ):
        super().__init__()
        self._num_classes = num_classes

        if hidden is not None and hidden > 0:
            layers = [torch.nn.Linear(feat_in, hidden), _resolve_activation(activation)]
            if dropout:
                layers.append(torch.nn.Dropout(p=dropout))
            layers.append(torch.nn.Linear(hidden, num_classes))
        else:
            layers = ([torch.nn.Dropout(p=dropout)] if dropout else []) + [torch.nn.Linear(feat_in, num_classes)]

        self.proj = torch.nn.Sequential(*layers)

    @typecheck()
    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        # (B, D, T) -> (B, T, D) -> (B, T, V)
        return self.proj(encoder_output.transpose(1, 2))

    @property
    def num_classes_with_blank(self):
        return self._num_classes

    @property
    def num_classes(self):
        return self._num_classes
