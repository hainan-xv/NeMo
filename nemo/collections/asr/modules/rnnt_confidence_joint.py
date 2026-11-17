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

"""RNNTAttJoint with a second, binary "confidence" head.

Both heads read the SAME joint representation -- the output of the joint's cross
attention -- so the confidence prediction is conditioned on exactly the
information the token prediction sees, and nothing more. That matters for
interpreting the experiment: if the two heads had different context, any
behavioural difference could be attributed to the context rather than to the
confidence mechanism.

Subclassed rather than bolted onto RNNTAttJoint so that every existing arm is
bit-identical: the confidence head is not constructed at all unless this class
is selected in the config.

See docs/reports/confidence_rnnt.tex for the lattice and the known
identifiability caveat.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from nemo.collections.asr.modules.rnnt import RNNTAttJoint, chunk_concat_audio

__all__ = ["RNNTAttConfidenceJoint"]


class RNNTAttConfidenceJoint(RNNTAttJoint):
    """``RNNTAttJoint`` plus a binary head over {not confident, confident}.

    Args:
        confidence_hidden: width of the head's hidden layer; 0 makes it a single
            linear map from the joint representation to 2 logits.
        confidence_init_bias: bias added to the "confident" logit at init. A
            positive value starts the model confident, so early training behaves
            like ordinary RNN-T and the defer branch has to earn its mass.
    """

    def __init__(self, *args, confidence_hidden: int = 0, confidence_init_bias: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)

        d = self.joint_hidden
        if confidence_hidden and confidence_hidden > 0:
            self.confidence_net = nn.Sequential(
                nn.Linear(d, confidence_hidden),
                nn.ReLU(),
                nn.Linear(confidence_hidden, 2),
            )
            last = self.confidence_net[-1]
        else:
            self.confidence_net = nn.Linear(d, 2)
            last = self.confidence_net

        # Start confident: P_c(1) ~ sigmoid(confidence_init_bias). With the
        # default this is ~0.88, so the lattice starts close to plain RNN-T
        # rather than at an arbitrary split.
        with torch.no_grad():
            last.bias.zero_()
            last.bias[1] = float(confidence_init_bias)

        self.confidence_hidden = confidence_hidden
        self.confidence_init_bias = confidence_init_bias

    def joint_with_confidence(
        self, f: torch.Tensor, g: torch.Tensor, f_len: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(log_py, log_pc)`` from one pass over the joint.

        ``log_py`` is (B, T, U+1, V+1) log-softmaxed over the vocabulary;
        ``log_pc`` is (B, T, U+1, 2) log-softmaxed over {defer, commit}.
        """
        # Mirror RNNTAttJoint.joint() exactly: chunk FIRST (which also records
        # num_chunks_per_utterance, which callers rely on), apply the history
        # window, and only then project. Projecting before chunking silently
        # produces the wrong shapes.
        if f_len is not None and f_len.dim() == 1 and self.chunk_size > 0:
            chunked, chunk_lengths = chunk_concat_audio(f, f_len, self.chunk_size)
            self.num_chunks_per_utterance = (chunk_lengths != 0).sum(dim=1)
            chunked, chunk_lengths = self._apply_history_window(chunked, chunk_lengths)
            f_proj, g_proj, lens = self.project_encoder(chunked), self.project_prednet(g), chunk_lengths
        else:
            self.num_chunks_per_utterance = None
            f_proj, g_proj, lens = self.project_encoder(f), self.project_prednet(g), f_len

        B, T, _ = f_proj.shape
        f_resh = torch.reshape(f_proj, [B, T, -1, self.joint_hidden])
        inp = self.cross_attention(f_resh, g_proj, lens)

        log_py = self.joint_net(inp).log_softmax(dim=-1)
        log_pc = self.confidence_net(inp).log_softmax(dim=-1)
        return log_py, log_pc
