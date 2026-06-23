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

"""Prediction network + joint that fuse a GPT-style LM into a TDT transducer.

The fusion is entirely encapsulated in these two modules so the generic (training and
greedy-decoding) transducer machinery needs no changes:

* :class:`TDTGPTFusionDecoder` owns the GPT LM. Its output carries the LM's per-step log-probs
  *concatenated onto* the prediction-network hidden state, and the LM history is threaded inside the
  decoder state. Because the LM ride-along lives inside ``decoder_output`` / ``state``, the existing
  batched state-management (masked ``torch.where`` etc.) carries and masks it for free.
* :class:`TDTGPTFusionJoint` splits the ride-along back off in ``project_prednet`` /
  ``joint_after_projection`` and log-linearly adds ``alpha * log P_LM`` to the non-blank token logits
  only (blank and the TDT duration logits are left untouched).

During training the LM log-probs added to the joint are ``detach()``-ed, so the transducer loss never
updates the LM parameters; the LM is trained only by its own next-token CE loss (computed in the
model). During greedy decoding (which must run with ``use_cuda_graph_decoder=false``) the same joint
adds the LM term, so the fusion is identical in train and decode.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from nemo.collections.asr.modules.gpt_lm import GPTLanguageModel
from nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTJoint
from nemo.collections.common.parts import rnn

# A decoder state is a 4-tuple: (lstm_h, lstm_c, lm_history, lm_history_len)
#   lstm_h, lstm_c   : [L, B, H]  (standard RNN-T LSTM state)
#   lm_history       : [B, max_ctx] long  (committed token ids; index 0 holds the BOS token)
#   lm_history_len   : [B] long           (number of valid tokens in lm_history)
FusionState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class TDTGPTFusionDecoder(RNNTDecoder):
    """RNN-T/TDT prediction network that additionally owns and advances a GPT LM.

    Adds the GPT LM's per-step log-probabilities (over the real vocabulary) as a tail concatenated
    onto the prediction-network output, so a paired :class:`TDTGPTFusionJoint` can fuse them.
    """

    def __init__(
        self,
        prednet: Dict[str, Any],
        vocab_size: int,
        gpt_lm: Dict[str, Any],
        normalization_mode: Optional[str] = None,
        random_state_sampling: bool = False,
        blank_as_pad: bool = True,
    ):
        super().__init__(
            prednet=prednet,
            vocab_size=vocab_size,
            normalization_mode=normalization_mode,
            random_state_sampling=random_state_sampling,
            blank_as_pad=blank_as_pad,
        )
        gpt_lm = dict(gpt_lm) if gpt_lm is not None else {}
        self.lm = GPTLanguageModel(vocab_size=vocab_size, **gpt_lm)
        self.max_ctx = self.lm.max_ctx
        # Stash of the most recent (non-detached) LM logits from a training forward, read by the
        # model to compute the LM cross-entropy loss.
        self._last_lm_logits: Optional[torch.Tensor] = None

    # -- LM helpers ---------------------------------------------------------------------------

    def get_last_lm_logits(self) -> Optional[torch.Tensor]:
        """Return the LM logits ``[B, U+1, V]`` from the most recent training ``forward`` (or None)."""
        return self._last_lm_logits

    def _lm_logprob_last(self, history: torch.Tensor, history_len: torch.Tensor) -> torch.Tensor:
        """Log-probs over the vocab for the next token, given a right-padded token history.

        Args:
            history: ``[B, max_ctx]`` long token ids.
            history_len: ``[B]`` number of valid tokens per row (>= 1).

        Returns:
            ``[B, V]`` log-softmax over the real vocabulary at the last valid position.
        """
        device = history.device
        max_len = int(history_len.max().item()) if history_len.numel() > 0 else 1
        max_len = max(max_len, 1)
        ids = history[:, :max_len]
        attn = (torch.arange(max_len, device=device)[None, :] < history_len[:, None]).long()
        logits = self.lm(ids, attention_mask=attn)  # [B, max_len, V]
        last_idx = (history_len - 1).clamp(min=0)
        batch_idx = torch.arange(ids.size(0), device=device)
        last_logits = logits[batch_idx, last_idx]  # [B, V]
        return F.log_softmax(last_logits.float(), dim=-1)

    # -- training forward (whole sequence) ----------------------------------------------------

    def forward(self, targets, target_length, states=None):
        """Training forward: returns ``g`` of shape ``[B, D + V, U + 1]`` (LM log-probs in the tail).

        The LM is run once over ``[BOS, y_0, ..., y_{U-1}]`` (causal), aligned position-by-position
        with the prediction network (which is fed ``[SOS, y_0, ..., y_{U-1}]``). The non-detached LM
        logits are stashed for the LM loss; the tail concatenated onto ``g`` is detached so the
        transducer loss cannot update the LM.
        """
        y = rnn.label_collate(targets)  # [B, U]

        # Standard LSTM prediction network (LM-free), with SOS prepended -> [B, U+1, D]
        g, hid = RNNTDecoder.predict(self, y, state=states, add_sos=True)

        batch = y.size(0)
        device = y.device
        bos = torch.full((batch, 1), self.lm.bos_id, dtype=torch.long, device=device)
        lm_input = torch.cat([bos, y.to(device).long()], dim=1)  # [B, U+1]
        seq_len = lm_input.size(1)
        attn = (torch.arange(seq_len, device=device)[None, :] < (target_length.to(device) + 1)[:, None]).long()
        lm_logits = self.lm(lm_input, attention_mask=attn)  # [B, U+1, V]
        self._last_lm_logits = lm_logits

        lm_logprob = F.log_softmax(lm_logits.float(), dim=-1).detach().to(g.dtype)  # [B, U+1, V]
        g = torch.cat([g, lm_logprob], dim=-1)  # [B, U+1, D + V]
        g = g.transpose(1, 2)  # [B, D + V, U + 1]
        return g, target_length, hid

    # -- decoding step (single label) ---------------------------------------------------------

    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[FusionState] = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, FusionState]:
        """Single-step decode: advance LSTM + LM, return ``[B, 1, D + V]`` and the new fusion state."""
        lstm_state = (state[0], state[1]) if state is not None else None
        g, hid = RNNTDecoder.predict(self, y, state=lstm_state, add_sos=add_sos, batch_size=batch_size)

        batch = g.size(0)
        device = g.device
        if state is not None and len(state) >= 4:
            history, history_len = state[2], state[3]
        else:
            history = torch.full((batch, self.max_ctx), self.lm.bos_id, dtype=torch.long, device=device)
            history_len = torch.zeros(batch, dtype=torch.long, device=device)

        if y is None:
            labels = torch.full((batch,), self.lm.bos_id, dtype=torch.long, device=device)
        else:
            labels = y.to(device).long().view(batch, -1)[:, -1]

        new_history = history.clone()
        write_idx = history_len.clamp(max=self.max_ctx - 1)
        batch_idx = torch.arange(batch, device=device)
        new_history[batch_idx, write_idx] = labels
        new_history_len = (history_len + 1).clamp(max=self.max_ctx)

        lm_logprob = self._lm_logprob_last(new_history, new_history_len)  # [B, V]
        g = torch.cat([g, lm_logprob.unsqueeze(1).to(g.dtype)], dim=-1)  # [B, 1, D + V]
        new_state: FusionState = (hid[0], hid[1], new_history, new_history_len)
        return g, new_state

    # -- state management (extends the LSTM-only ops to also carry the LM history) -------------

    def initialize_state(self, y: torch.Tensor) -> FusionState:
        h, c = RNNTDecoder.initialize_state(self, y)
        batch = y.size(0)
        history = torch.full((batch, self.max_ctx), self.lm.bos_id, dtype=torch.long, device=y.device)
        history_len = torch.zeros(batch, dtype=torch.long, device=y.device)
        return (h, c, history, history_len)

    @classmethod
    def batch_replace_states_mask(
        cls,
        src_states: FusionState,
        dst_states: FusionState,
        mask: torch.Tensor,
        other_src_states: Optional[FusionState] = None,
    ):
        RNNTDecoder.batch_replace_states_mask(
            src_states=src_states, dst_states=dst_states, mask=mask, other_src_states=other_src_states
        )
        other = other_src_states if other_src_states is not None else dst_states
        torch.where(mask.unsqueeze(-1), src_states[2], other[2], out=dst_states[2])
        torch.where(mask, src_states[3], other[3], out=dst_states[3])

    @classmethod
    def batch_replace_states_all(
        cls,
        src_states: FusionState,
        dst_states: FusionState,
        batch_size: Optional[int] = None,
    ):
        RNNTDecoder.batch_replace_states_all(src_states=src_states, dst_states=dst_states, batch_size=batch_size)
        if batch_size is None:
            dst_states[2].copy_(src_states[2])
            dst_states[3].copy_(src_states[3])
        else:
            dst_states[2][:batch_size].copy_(src_states[2][:batch_size])
            dst_states[3][:batch_size].copy_(src_states[3][:batch_size])

    @classmethod
    def clone_state(cls, state: FusionState) -> FusionState:
        return (state[0].clone(), state[1].clone(), state[2].clone(), state[3].clone())

    @classmethod
    def batch_split_states(cls, batch_states: FusionState) -> List[FusionState]:
        h, c, history, history_len = batch_states
        return [
            (h[:, i], c[:, i], history[i], history_len[i])
            for i in range(h.size(1))
        ]

    @classmethod
    def batch_unsplit_states(cls, batch_states: List[FusionState], device=None, dtype=None) -> FusionState:
        h = torch.stack([s[0] for s in batch_states], dim=1).to(device=device, dtype=dtype)
        c = torch.stack([s[1] for s in batch_states], dim=1).to(device=device, dtype=dtype)
        history = torch.stack([s[2] for s in batch_states], dim=0).to(device=device)
        history_len = torch.stack([s[3] for s in batch_states], dim=0).to(device=device)
        return (h, c, history, history_len)

    def mask_select_states(self, states: FusionState, mask: torch.Tensor) -> FusionState:
        return (states[0][:, mask], states[1][:, mask], states[2][mask], states[3][mask])


class TDTGPTFusionJoint(RNNTJoint):
    """TDT joint that log-linearly fuses GPT-LM log-probs into the non-blank token logits.

    The prediction-network output handed to this joint has the LM log-probs concatenated onto its
    feature dimension (see :class:`TDTGPTFusionDecoder`). This joint projects only the real
    prediction-network part, then adds ``lm_fusion_alpha * log P_LM`` to the non-blank token logits.
    """

    def __init__(
        self,
        jointnet: Dict[str, Any],
        num_classes: int,
        num_extra_outputs: int = 0,
        lm_fusion_alpha: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            jointnet=jointnet, num_classes=num_classes, num_extra_outputs=num_extra_outputs, **kwargs
        )
        self.lm_fusion_alpha = float(lm_fusion_alpha)
        # Number of real (non-blank, non-duration) token logits == LM vocab size.
        self._lm_vocab_size = num_classes

    def project_prednet(self, prednet_output: torch.Tensor) -> torch.Tensor:
        """Project the prediction-network hidden part; pass the LM log-prob tail through unchanged."""
        hidden = prednet_output[..., : self.pred_hidden]
        lm_tail = prednet_output[..., self.pred_hidden :]
        projected = self.pred(hidden)  # [..., joint_hidden]
        return torch.cat([projected, lm_tail], dim=-1)

    def joint_after_projection(
        self, f: torch.Tensor, g: torch.Tensor, f_len: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run the standard joint, then add ``alpha * log P_LM`` to the non-blank token logits."""
        g_hidden = g[..., : self.joint_hidden]
        lm_logprob = g[..., self.joint_hidden :]  # [B, U, V]

        res = super().joint_after_projection(f, g_hidden, f_len)  # [B, T, U, V + 1 + num_durations]

        vocab = self._lm_vocab_size
        token_logits = res[..., :vocab] + self.lm_fusion_alpha * lm_logprob.unsqueeze(1).to(res.dtype)
        rest = res[..., vocab:]
        return torch.cat([token_logits, rest], dim=-1)
