# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.modules import rnnt_abstract
#from nemo.collections.asr.parts.submodules.rnnt_loop_labels_computer import GreedyBatchedRNNTLoopLabelsComputer
#from nemo.collections.asr.parts.submodules.tdt_loop_labels_computer import GreedyBatchedTDTLoopLabelsComputer
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodConfig, ConfidenceMethodMixin
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.collections.common.parts.rnn import label_collate
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import AcousticEncodedRepresentation, HypothesisType, LengthsType, NeuralType
from nemo.utils import logging


def pack_hypotheses(hypotheses: List[rnnt_utils.Hypothesis], logitlen: torch.Tensor,) -> List[rnnt_utils.Hypothesis]:

    if hasattr(logitlen, 'cpu'):
        logitlen_cpu = logitlen.to('cpu')
    else:
        logitlen_cpu = logitlen

    for idx, hyp in enumerate(hypotheses):  # type: rnnt_utils.Hypothesis
        hyp.y_sequence = (
            hyp.y_sequence.to(torch.long)
            if isinstance(hyp.y_sequence, torch.Tensor)
            else torch.tensor(hyp.y_sequence, dtype=torch.long)
        )
        hyp.length = logitlen_cpu[idx]

        if hyp.dec_state is not None:
            hyp.dec_state = _states_to_device(hyp.dec_state)

    return hypotheses


def _states_to_device(dec_state, device='cpu'):
    if torch.is_tensor(dec_state):
        dec_state = dec_state.to(device)

    elif isinstance(dec_state, (list, tuple)):
        dec_state = tuple(_states_to_device(dec_i, device) for dec_i in dec_state)

    return dec_state


class _GreedyNARTDTInfer(Typing, ConfidenceMethodMixin):
    """A greedy transducer decoder.

    Provides a common abstraction for sample level and batch level greedy decoding.

    Args:
        joint_model: rnnt_utils.AbstractRNNTJoint implementation.
        blank_index: int index of the blank token. Can be 0 or len(vocabulary).
        max_symbols_per_step: Optional int. The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        preserve_alignments: Bool flag which preserves the history of alignments generated during
            greedy decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `alignments` in it. Here, `alignments` is a List of List of
            Tuple(Tensor (of length V + 1), Tensor(scalar, label after argmax)).

            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
            U is the number of target tokens for the current timestep Ti.
        preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores generated
            during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `frame_confidence` in it. Here, `frame_confidence` is a List of List of floats.

            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more confidence scores.
            U is the number of target tokens for the current timestep Ti.
        confidence_method_cfg: A dict-like object which contains the method name and settings to compute per-frame
            confidence scores.

            name: The method name (str).
                Supported values:
                    - 'max_prob' for using the maximum token probability as a confidence.
                    - 'entropy' for using a normalized entropy of a log-likelihood vector.

            entropy_type: Which type of entropy to use (str). Used if confidence_method_cfg.name is set to `entropy`.
                Supported values:
                    - 'gibbs' for the (standard) Gibbs entropy. If the alpha (α) is provided,
                        the formula is the following: H_α = -sum_i((p^α_i)*log(p^α_i)).
                        Note that for this entropy, the alpha should comply the following inequality:
                        (log(V)+2-sqrt(log^2(V)+4))/(2*log(V)) <= α <= (1+log(V-1))/log(V-1)
                        where V is the model vocabulary size.
                    - 'tsallis' for the Tsallis entropy with the Boltzmann constant one.
                        Tsallis entropy formula is the following: H_α = 1/(α-1)*(1-sum_i(p^α_i)),
                        where α is a parameter. When α == 1, it works like the Gibbs entropy.
                        More: https://en.wikipedia.org/wiki/Tsallis_entropy
                    - 'renyi' for the Rényi entropy.
                        Rényi entropy formula is the following: H_α = 1/(1-α)*log_2(sum_i(p^α_i)),
                        where α is a parameter. When α == 1, it works like the Gibbs entropy.
                        More: https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy

            alpha: Power scale for logsoftmax (α for entropies). Here we restrict it to be > 0.
                When the alpha equals one, scaling is not applied to 'max_prob',
                and any entropy type behaves like the Shannon entropy: H = -sum_i(p_i*log(p_i))

            entropy_norm: A mapping of the entropy value to the interval [0,1].
                Supported values:
                    - 'lin' for using the linear mapping.
                    - 'exp' for using exponential mapping with linear shift.
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "partial_hypotheses": [NeuralType(elements_type=HypothesisType(), optional=True)],  # must always be last
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"predictions": [NeuralType(elements_type=HypothesisType())]}

    def __init__(
        self,
        joint_model: rnnt_abstract.AbstractRNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments: bool = False,
        preserve_frame_confidence: bool = False,
        confidence_method_cfg: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.joint = joint_model

        self._blank_index = blank_index
        self._SOS = blank_index  # Start of single index

        if max_symbols_per_step is not None and max_symbols_per_step <= 0:
            raise ValueError(f"Expected max_symbols_per_step > 0 (or None), got {max_symbols_per_step}")
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence

        # set confidence calculation method
        self._init_confidence_method(confidence_method_cfg)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _joint_step(self, enc, log_normalize: Optional[bool] = None):
        """
        Common joint step based on AbstractRNNTJoint implementation.

        Args:
            enc: Output of the Encoder model. A torch.Tensor of shape [B, 1, H1]
            pred: Output of the Decoder model. A torch.Tensor of shape [B, 1, H2]
            log_normalize: Whether to log normalize or not. None will log normalize only for CPU.

        Returns:
             logits of shape (B, T=1, U=1, V + 1)
        """
        with torch.no_grad():
            logits = self.joint.joint(enc)

            if log_normalize is None:
                if not logits.is_cuda:  # Use log softmax only if on CPU
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)
            else:
                if log_normalize:
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)

        return logits

    def _joint_step_after_projection(self, enc, log_normalize: Optional[bool] = None) -> torch.Tensor:
        """
        Common joint step based on AbstractRNNTJoint implementation.

        Args:
            enc: Output of the Encoder model after projection. A torch.Tensor of shape [B, 1, H]
            pred: Output of the Decoder model after projection. A torch.Tensor of shape [B, 1, H]
            log_normalize: Whether to log normalize or not. None will log normalize only for CPU.

        Returns:
             logits of shape (B, T=1, U=1, V + 1)
        """
        with torch.no_grad():
            logits = self.joint.joint_after_projection(enc)

            if log_normalize is None:
                if not logits.is_cuda:  # Use log softmax only if on CPU
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)
            else:
                if log_normalize:
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)

        return logits




class GreedyNARTDTInfer(_GreedyNARTDTInfer):
    """A greedy TDT decoder.

    Sequence level greedy decoding, performed auto-regressively.

    Args:
        joint_model: rnnt_utils.AbstractRNNTJoint implementation.
        blank_index: int index of the blank token. Must be len(vocabulary) for TDT models.
        durations: a list containing durations for TDT.
        max_symbols_per_step: Optional int. The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        preserve_alignments: Bool flag which preserves the history of alignments generated during
            greedy decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `alignments` in it. Here, `alignments` is a List of List of
            Tuple(Tensor (of length V + 1 + num-big-blanks), Tensor(scalar, label after argmax)).
            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
            U is the number of target tokens for the current timestep Ti.
        preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores generated
            during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `frame_confidence` in it. Here, `frame_confidence` is a List of List of floats.
            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more confidence scores.
            U is the number of target tokens for the current timestep Ti.
        include_duration_confidence: Bool flag indicating that the duration confidence scores are to be calculated and
            attached to the regular frame confidence,
            making TDT frame confidence element a pair: (`prediction_confidence`, `duration_confidence`).
        confidence_method_cfg: A dict-like object which contains the method name and settings to compute per-frame
            confidence scores.

            name: The method name (str).
                Supported values:
                    - 'max_prob' for using the maximum token probability as a confidence.
                    - 'entropy' for using a normalized entropy of a log-likelihood vector.

            entropy_type: Which type of entropy to use (str). Used if confidence_method_cfg.name is set to `entropy`.
                Supported values:
                    - 'gibbs' for the (standard) Gibbs entropy. If the alpha (α) is provided,
                        the formula is the following: H_α = -sum_i((p^α_i)*log(p^α_i)).
                        Note that for this entropy, the alpha should comply the following inequality:
                        (log(V)+2-sqrt(log^2(V)+4))/(2*log(V)) <= α <= (1+log(V-1))/log(V-1)
                        where V is the model vocabulary size.
                    - 'tsallis' for the Tsallis entropy with the Boltzmann constant one.
                        Tsallis entropy formula is the following: H_α = 1/(α-1)*(1-sum_i(p^α_i)),
                        where α is a parameter. When α == 1, it works like the Gibbs entropy.
                        More: https://en.wikipedia.org/wiki/Tsallis_entropy
                    - 'renyi' for the Rényi entropy.
                        Rényi entropy formula is the following: H_α = 1/(1-α)*log_2(sum_i(p^α_i)),
                        where α is a parameter. When α == 1, it works like the Gibbs entropy.
                        More: https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy

            alpha: Power scale for logsoftmax (α for entropies). Here we restrict it to be > 0.
                When the alpha equals one, scaling is not applied to 'max_prob',
                and any entropy type behaves like the Shannon entropy: H = -sum_i(p_i*log(p_i))

            entropy_norm: A mapping of the entropy value to the interval [0,1].
                Supported values:
                    - 'lin' for using the linear mapping.
                    - 'exp' for using exponential mapping with linear shift.
    """

    def __init__(
        self,
        joint_model: rnnt_abstract.AbstractRNNTJoint,
        blank_index: int,
        durations: list,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments: bool = False,
        preserve_frame_confidence: bool = False,
        include_duration_confidence: bool = False,
        confidence_method_cfg: Optional[DictConfig] = None,
    ):
        super().__init__(
            joint_model=joint_model,
            blank_index=blank_index,
            max_symbols_per_step=max_symbols_per_step,
            preserve_alignments=preserve_alignments,
            preserve_frame_confidence=preserve_frame_confidence,
            confidence_method_cfg=confidence_method_cfg,
        )
        self.durations = durations
        self.include_duration_confidence = include_duration_confidence

    @typecheck()
    def forward(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        partial_hypotheses: Optional[List[rnnt_utils.Hypothesis]] = None,
    ):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-regressively.
        Args:
            encoder_output: A tensor of size (batch, features, timesteps).
            encoded_lengths: list of int representing the length of each sequence
                output sequence.
        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        # Preserve decoder and joint training state
        joint_training_state = self.joint.training

        encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)
        encoder_output = self._joint_step(encoder_output, log_normalize=False)

        with torch.inference_mode():
            # Apply optional preprocessing

            self.joint.eval()

            hypotheses = []
            # Process each sequence independently
            with self.joint.as_frozen():
                for batch_idx in range(encoder_output.size(0)):
                    inseq = encoder_output[batch_idx, :, :, :].unsqueeze(1)  # [T, 1, D]
                    logitlen = encoded_lengths[batch_idx]

                    partial_hypothesis = partial_hypotheses[batch_idx] if partial_hypotheses is not None else None
                    hypothesis = self._greedy_decode(inseq, logitlen, partial_hypotheses=partial_hypothesis)
                    hypotheses.append(hypothesis)

            # Pack results into Hypotheses
            packed_result = pack_hypotheses(hypotheses, encoded_lengths)

        self.joint.train(joint_training_state)

        return (packed_result,)

    @torch.no_grad()
    def _greedy_decode(
        self, logits: torch.Tensor, out_len: torch.Tensor, partial_hypotheses: Optional[rnnt_utils.Hypothesis] = None
    ):
        # Initialize blank state and empty label set in Hypothesis
        hypothesis = rnnt_utils.Hypothesis(score=0.0, y_sequence=[], dec_state=None, timestep=[], last_token=None)

        if partial_hypotheses is not None:
            hypothesis.last_token = partial_hypotheses.last_token
            hypothesis.y_sequence = (
                partial_hypotheses.y_sequence.cpu().tolist()
                if isinstance(partial_hypotheses.y_sequence, torch.Tensor)
                else partial_hypotheses.y_sequence
            )

        if self.preserve_alignments:
            # Alignments is a 2-dimensional dangling list representing T x U
            hypothesis.alignments = [[]]

        if self.preserve_frame_confidence:
            hypothesis.frame_confidence = [[]]

        logp = logits[:, 0, 0, : -len(self.durations)]

        duration_logp = logits[:, 0, 0, -len(self.durations) :]

        v, k = torch.max(logp, dim=-1)
        k = k.tolist()

        d_v, d_k = torch.max(duration_logp, dim=-1)
        d_k = d_k.tolist()

        time_idx = 0
        while time_idx < out_len:
            token_idx = k[time_idx]
            if token_idx != self._blank_index:
                hypothesis.y_sequence.append(token_idx)
            skip = self.durations[d_k[time_idx]]
            time_idx += skip

        return hypothesis

