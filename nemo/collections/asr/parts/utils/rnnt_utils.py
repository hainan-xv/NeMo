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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms.

    score: A float score obtained from an AbstractRNNTDecoder module's score_hypothesis method.

    y_sequence: Either a sequence of integer ids pointing to some vocabulary, or a packed torch.Tensor
        behaving in the same manner. dtype must be torch.Long in the latter case.

    dec_state: A list (or list of list) of LSTM-RNN decoder states. Can be None.

    text: (Optional) A decoded string after processing via CTC / RNN-T decoding (removing the CTC/RNNT
        `blank` tokens, and optionally merging word-pieces). Should be used as decoded string for
        Word Error Rate calculation.

    timestep: (Optional) A list of integer indices representing at which index in the decoding
        process did the token appear. Should be of same length as the number of non-blank tokens.

    alignments: (Optional) Represents the CTC / RNNT token alignments as integer tokens along an axis of
        time T (for CTC) or Time x Target (TxU).
        For CTC, represented as a single list of integer indices.
        For RNNT, represented as a dangling list of list of integer indices.
        Outer list represents Time dimension (T), inner list represents Target dimension (U).
        The set of valid indices **includes** the CTC / RNNT blank token in order to represent alignments.

    frame_confidence: (Optional) Represents the CTC / RNNT per-frame confidence scores as token probabilities
        along an axis of time T (for CTC) or Time x Target (TxU).
        For CTC, represented as a single list of float indices.
        For RNNT, represented as a dangling list of list of float indices.
        Outer list represents Time dimension (T), inner list represents Target dimension (U).

    token_confidence: (Optional) Represents the CTC / RNNT per-token confidence scores as token probabilities
        along an axis of Target U.
        Represented as a single list of float indices.

    word_confidence: (Optional) Represents the CTC / RNNT per-word confidence scores as token probabilities
        along an axis of Target U.
        Represented as a single list of float indices.

    length: Represents the length of the sequence (the original length without padding), otherwise
        defaults to 0.

    y: (Unused) A list of torch.Tensors representing the list of hypotheses.

    lm_state: (Unused) A dictionary state cache used by an external Language Model.

    lm_scores: (Unused) Score of the external Language Model.

    ngram_lm_state: (Optional) State of the external n-gram Language Model.

    tokens: (Optional) A list of decoded tokens (can be characters or word-pieces.

    last_token (Optional): A token or batch of tokens which was predicted in the last step.
    """

    score: float
    y_sequence: Union[List[int], torch.Tensor]
    text: Optional[str] = None
    dec_out: Optional[List[torch.Tensor]] = None
    dec_state: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor]]] = None
    timestep: Union[List[int], torch.Tensor] = field(default_factory=list)
    alignments: Optional[Union[List[int], List[List[int]]]] = None
    frame_confidence: Optional[Union[List[float], List[List[float]]]] = None
    token_confidence: Optional[List[float]] = None
    word_confidence: Optional[List[float]] = None
    length: Union[int, torch.Tensor] = 0
    y: List[torch.tensor] = None
    lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
    lm_scores: Optional[torch.Tensor] = None
    ngram_lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
    tokens: Optional[Union[List[int], torch.Tensor]] = None
    last_token: Optional[torch.Tensor] = None
    next_t: Optional[int] = None
    batch_id: Optional[int] = None

    @property
    def non_blank_frame_confidence(self) -> List[float]:
        """Get per-frame confidence for non-blank tokens according to self.timestep

        Returns:
            List with confidence scores. The length of the list is the same as `timestep`.
        """
        non_blank_frame_confidence = []
        # self.timestep can be a dict for RNNT
        timestep = self.timestep['timestep'] if isinstance(self.timestep, dict) else self.timestep
        if len(self.timestep) != 0 and self.frame_confidence is not None:
            if any(isinstance(i, list) for i in self.frame_confidence):  # rnnt
                t_prev = -1
                offset = 0
                for t in timestep:
                    if t != t_prev:
                        t_prev = t
                        offset = 0
                    else:
                        offset += 1
                    non_blank_frame_confidence.append(self.frame_confidence[t][offset])
            else:  # ctc
                non_blank_frame_confidence = [self.frame_confidence[t] for t in timestep]
        return non_blank_frame_confidence

    @property
    def words(self) -> List[str]:
        """Get words from self.text

        Returns:
            List with words (str).
        """
        return [] if self.text is None else self.text.split()


# highly optimized Cuda structure representing hypotheses
class CudaHypothesesStatelessTransducer:
    scores: torch.FloatTensor
    ys: torch.LongTensor
    dec_states: torch.LongTensor  # since this is for stateless decoder, the "states" stores token id's of last words in the history context.

    def __init__(
        self, n, m, d, device
    ):  # max number of utterance, max_length_per_utterance, dimension of decoder states
        self.scores = torch.zeros([n], dtype=torch.float, device=device)
        #        self.next_ts = torch.zeros([n], dtype=torch.long, device=device)
        #        self.batch_ids = torch.zeros([n], dtype=torch.long, device=device)
        self.ys = torch.zeros([n * m], dtype=torch.long, device=device)
        self.dec_states = torch.zeros([n, d], dtype=torch.long, device=device)
        self.batchsize = n
        self.m = m

    def get_hyps(self):
        ret = []
        m = self.m
        for i in range(self.batchsize):
            hyp = Hypothesis(score=self.scores[i], y_sequence=self.ys[i * m + 1 : i * m + 1 + self.ys[i * m]],)
            ret.append(hyp)
        return ret


# highly optimized Cuda structure representing hypotheses
class CudaBeamSearchHypothesesStatelessTransducer:
    def __init__(
        self, beam, max_length, num_hyps, context_size, device, blank, encoded_lengths
    ):  # max number of utterance, max_length_per_utterance, dimension of decoder states

        self.scores = torch.zeros([num_hyps], dtype=torch.float, device=device)
        self.ys = torch.zeros([num_hyps * max_length], dtype=torch.long, device=device)
        self.dec_states = torch.zeros([num_hyps, context_size], dtype=torch.long, device=device) + blank
        self.last_label = torch.full([num_hyps, 1], fill_value=blank, dtype=torch.long, device=device)
        self.encoded_lengths = encoded_lengths

        self.context_size = context_size

        self.max_length = max_length
        self.B = num_hyps  # number of utterances to decode
        self.num_hyps = num_hyps  # self.num_hyp will change.
        self.beam = beam

        self.hyp2t = torch.zeros([num_hyps], dtype=torch.long, device=device)
        self.hyp2done = torch.zeros([num_hyps], dtype=torch.long, device=device)
        self.hyp2b = torch.zeros([num_hyps], dtype=torch.long, device=device)

        for i in range(num_hyps):
            self.hyp2b[i] = i

        self.begin = True
        self.b2done = [False for i in range(num_hyps)]
        self.num_b_not_done = self.B
        self.b2done_hyps = [[] for i in range(num_hyps)]

        self.beam_beam_encoded_lengths = torch.reshape(self.encoded_lengths.repeat_interleave(self.beam * self.beam), [-1])
        self.beam_encoded_lengths = torch.reshape(self.encoded_lengths.repeat_interleave(self.beam), [-1])

        self.shift = torch.reshape(torch.tensor(range(self.B), device=self.hyp2t.device) * self.beam * self.beam, [self.B, 1])
  
        self.beam_hyp2b = self.hyp2b.repeat_interleave(self.beam)
#        self.beam_beam_hyp2b = self.hyp2b.repeat_interleave(self.beam * self.beam)

    def expand(self):
        # copy each hypothesis beam times, in the expanded_* variables so that
        # we could grow those hyps in search.

        self.expanded_scores = self.scores.repeat_interleave(self.beam)

        self.expanded_ys = torch.reshape(self.ys, [-1, self.max_length]).repeat(1, self.beam)
        self.expanded_ys = torch.reshape(self.expanded_ys, [-1])

        self.expanded_dec_states = torch.reshape(self.dec_states.repeat_interleave(self.beam), [-1, self.context_size])

        self.expanded_hyp2t = self.hyp2t.repeat_interleave(self.beam)
        self.expanded_hyp2done = self.hyp2done.repeat_interleave(self.beam)
        self.expanded_last_label = torch.reshape(self.last_label.repeat_interleave(self.beam), [-1, 1])

    def compress(self):
        if self.begin:
            self.begin = False
            self.scores = self.expanded_scores
            self.ys = self.expanded_ys
            self.dec_states = self.expanded_dec_states
            self.hyp2t = self.expanded_hyp2t
            self.hyp2done = self.expanded_hyp2done
#            self.hyp2b = self.beam_hyp2b
            self.last_label = self.expanded_last_label
            self.num_hyps = self.num_hyps * self.beam
#            self.encoded_lengths = self.expanded_encoded_lengths
            return

        v, k = torch.reshape(self.expanded_scores, [self.B, -1]).topk(self.beam, dim=-1)

        k = torch.reshape(k + self.shift, [-1])

        self.ys = torch.reshape(self.expanded_ys, [-1, self.max_length])[k]
        self.ys = torch.reshape(self.ys, [-1])

        self.scores = self.expanded_scores[k]
        self.dec_states = self.expanded_dec_states[k]
        self.hyp2t = self.expanded_hyp2t[k]
        self.hyp2done = self.expanded_hyp2done[k]
#        self.hyp2b = self.expanded_hyp2b[k]
        self.last_label = self.expanded_last_label[k]

        self.dedup()

    def dedup(self):
        ys = torch.reshape(self.ys, [self.B * self.beam, self.max_length])
        for i in range(self.B):
            ys_i = ys[i * self.beam : (i + 1) * self.beam, : ys[i * self.beam, 0] + 1]
            #            print("to unique", ys_i)
            #            print(torch.unique(ys_i, dim=0).shape)
            if torch.unique(ys_i, dim=0).shape[0] == 1:
                self.scores[i * self.beam : i * self.beam + self.beam - 1] -= 99999999.0

    #        print(self.expanded_scores)
    #        print(k)
    #        print(self.scores)

    #        print("new ys is", self.ys)
    #        print('k is', k)
    #        print("old ys is", self.expanded_ys)

    #        print("v is", v)

    #        v, k = torch.reshape(v, [-1]), torch.reshape(k, [-1])

    #        print("v, k are", v, k)

    #        assert(0)

    def clean_up(self):
        not_done_idx = torch.where(self.hyp2done != True)[0]
        done_idx = torch.where(self.hyp2done == True)[0]
        self.hyp2t *= torch.logical_not(self.hyp2done)
        if not_done_idx.shape[-1] != self.hyp2t.shape[-1]:
            #            print('adjust')
            #            self.hyp2t = self.hyp2t[not_done_idx]
            #            self.hyp2b = self.hyp2b[not_done_idx]
            #            B = not_done_idx.shape[-1]
            #            self.dec_states = decoder.batch_subset_states(self.dec_states, self.dec_states, not_done_idx)
            #            self.last_label = self.last_label[not_done_idx]
            #
            #            self.encoded_lengths = self.encoded_lengths[not_done_idx]
            #            self.hyp2done = self.hyp2done[not_done_idx]
            self.scores[
                done_idx
            ] = -9999999999.9  # make the score very bad so this will never show up in the search for next iteration

            #            print("done idx is", done_idx)
            m = self.max_length
            for i in done_idx.tolist():
                #                print(i)
                b = self.beam_hyp2b[i].item()
                #                print('b is', b)
                if not self.b2done[b] and len(self.b2done_hyps[b]) < self.beam:
                    hyp = Hypothesis(score=self.scores[i], y_sequence=self.ys[i * m + 1 : i * m + 1 + self.ys[i * m]],)
                    self.b2done_hyps[b].append(hyp)

            for b in range(len(self.b2done_hyps)):
                hyps = self.b2done_hyps[b]
                if len(hyps) >= self.beam and not self.b2done[b]:
                    self.b2done[b] = True
                    self.hyp2done[b * self.beam : b * self.beam + self.beam] = True
                    self.num_b_not_done -= 1

    def print_expanded(self):
        for i in range(self.num_hyps * self.beam):
            this_len = self.expanded_ys[i * self.max_length].item()
            to_print = ''
            for j in range(this_len):
                to_print += str(self.expanded_ys[i * self.max_length + 1 + j].item()) + ' '
            print(i, this_len, to_print)

        print()

    def print(self):
        for i in range(self.num_hyps):
            this_len = self.ys[i * self.max_length].item()
            to_print = ''
            for j in range(this_len):
                to_print += str(self.ys[i * self.max_length + 1 + j].item()) + ' '
            print(i, this_len, to_print)

        print()

    #    def get_beam_state(self):
    #        return [self.dec_states[:self.num_hyps,...]]

    def get_hyps(self):
        #        ret = [[] for i in range(self.num_hyps)]
        #        m = self.max_length
        #        for i in range(self.num_hyps):
        #            hyp = Hypothesis(score=self.scores[i], y_sequence=self.ys[i * m + 1 : i * m + 1 + self.ys[i * m]],)
        #            utt_id = i
        #            ret[utt_id].append(hyp)
        #        return ret

        return self.b2done_hyps


@dataclass
class NBestHypotheses:
    """List of N best hypotheses"""

    n_best_hypotheses: Optional[List[Hypothesis]]


@dataclass
class HATJointOutput:
    """HATJoint outputs for beam search decoding

    hat_logprobs: standard HATJoint outputs as for RNNTJoint

    ilm_logprobs: internal language model probabilities (for ILM subtraction)
    """

    hat_logprobs: Optional[torch.Tensor] = None
    ilm_logprobs: Optional[torch.Tensor] = None


def is_prefix(x: List[int], pref: List[int]) -> bool:
    """
    Obtained from https://github.com/espnet/espnet.

    Check if pref is a prefix of x.

    Args:
        x: Label ID sequence.
        pref: Prefix label ID sequence.

    Returns:
        : Whether pref is a prefix of x.
    """
    if len(pref) >= len(x):
        return False

    for i in range(len(pref)):
        if pref[i] != x[i]:
            return False

    return True


def select_k_expansions(
    hyps: List[Hypothesis], topk_idxs: torch.Tensor, topk_logps: torch.Tensor, gamma: float, beta: int,
) -> List[Tuple[int, Hypothesis]]:
    """
    Obtained from https://github.com/espnet/espnet

    Return K hypotheses candidates for expansion from a list of hypothesis.
    K candidates are selected according to the extended hypotheses probabilities
    and a prune-by-value method. Where K is equal to beam_size + beta.

    Args:
        hyps: Hypotheses.
        topk_idxs: Indices of candidates hypothesis. Shape = [B, num_candidates]
        topk_logps: Log-probabilities for hypotheses expansions. Shape = [B, V + 1]
        gamma: Allowed logp difference for prune-by-value method.
        beta: Number of additional candidates to store.

    Return:
        k_expansions: Best K expansion hypotheses candidates.
    """
    k_expansions = []

    for i, hyp in enumerate(hyps):
        hyp_i = [(int(k), hyp.score + float(v)) for k, v in zip(topk_idxs[i], topk_logps[i])]
        k_best_exp_val = max(hyp_i, key=lambda x: x[1])

        k_best_exp_idx = k_best_exp_val[0]
        k_best_exp = k_best_exp_val[1]

        expansions = sorted(filter(lambda x: (k_best_exp - gamma) <= x[1], hyp_i), key=lambda x: x[1],)

        if len(expansions) > 0:
            k_expansions.append(expansions)
        else:
            k_expansions.append([(k_best_exp_idx, k_best_exp)])

    return k_expansions
