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

"""Frame-wise cross-entropy loss for the Aligner-Encoder ASR model.

The Aligner-Encoder is trained with the simple frame-wise cross-entropy loss of
an attention-based encoder-decoder (AED) instead of the dynamic-programming
marginalization of RNN-T. Output position ``i`` (paired one-to-one with encoder
frame ``i``) is trained to predict target token ``y_i``; positions beyond the
target length are ignored.

Reference: Stooke et al., "Aligner-Encoders: Self-Attention Transformers Can Be
Self-Transducers" (https://arxiv.org/abs/2502.05232).
"""

import torch
import torch.nn.functional as F

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType

__all__ = ['AlignerCrossEntropyLoss']

IGNORE_INDEX = -100


class AlignerCrossEntropyLoss(Loss):
    """Length-masked, label-smoothed cross-entropy over per-position logits.

    Args:
        num_classes: Number of output classes (real tokens + EOS, no blank).
        label_smoothing: Label smoothing weight (the paper uses ``0.1``).
        reduction: One of ``mean`` (average over valid positions), ``sum`` or
            ``none``.
    """

    @property
    def input_types(self):
        return {
            "log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "target_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_classes: int, label_smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction must be one of ['mean', 'sum', 'none'], got '{reduction}'.")
        self.num_classes = num_classes
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    @typecheck()
    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """Compute the masked cross-entropy loss.

        Args:
            log_probs: Per-position logits (un-normalized scores are fine) of
                shape ``(B, N, V)``. The first ``target_lengths[b]`` positions of
                sample ``b`` are supervised.
            targets: Target token ids of shape ``(B, N)`` (must already include the
                trailing EOS token).
            target_lengths: Number of valid target positions per sample, ``(B,)``.

        Returns:
            Scalar loss tensor.
        """
        batch_size, max_len, num_classes = log_probs.shape

        # Guard fully-degenerate inputs (e.g. an empty batch after fault-tolerant
        # audio loading dropped every cut): return a finite, differentiable zero
        # instead of letting the mean over zero elements become NaN.
        if batch_size == 0 or max_len == 0:
            return log_probs.sum() * 0.0

        # Targets and logits must share the same length axis; clip to the shorter.
        target_len_axis = targets.size(1)
        common = min(max_len, target_len_axis)
        logits = log_probs[:, :common, :]
        tgt = targets[:, :common].clone().long()

        # Mask out positions beyond each sample's (EOS-inclusive) target length.
        position = torch.arange(common, device=tgt.device).unsqueeze(0)
        valid = position < target_lengths.unsqueeze(1).to(tgt.device)
        tgt[~valid] = IGNORE_INDEX

        # Upcast to fp32 for numerical stability under bf16/fp16 autocast, and use
        # the functional API with reduction='none' so we can divide by a safe
        # (non-zero) count of supervised positions. ``F.cross_entropy`` returns 0
        # for ignored positions, so a fully-masked batch yields a finite 0 loss.
        per_position = F.cross_entropy(
            logits.reshape(-1, num_classes).float(),
            tgt.reshape(-1),
            ignore_index=IGNORE_INDEX,
            label_smoothing=self.label_smoothing,
            reduction='none',
        )

        if self.reduction == 'none':
            return per_position.view(batch_size, common)
        if self.reduction == 'sum':
            return per_position.sum()

        num_valid = (tgt.reshape(-1) != IGNORE_INDEX).sum().clamp(min=1).to(per_position.dtype)
        return per_position.sum() / num_valid
