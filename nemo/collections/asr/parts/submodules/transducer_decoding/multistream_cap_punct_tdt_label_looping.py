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

"""Batched greedy Label-Looping decoding for the 3-stream (spelling + cap + punct) TDT model.

This mirrors :class:`GreedyBatchedMultiStreamTDTLabelLoopingComputer`, but the label part of the
joint output is laid out as ``[ punct(num_punct) | cap(num_cap) | spell(num_spell) | blank ]``
(followed by the duration logits). At each Joint evaluation we pick, by argmax over each slice:

* the spelling token (incl. blank, the last index of the spelling slice),
* the capitalization class, and
* the (word-ending) punctuation class.

The *product* id ``(punct * num_cap + cap) * num_spell + spell`` is stored in the hypotheses,
while only the *spelling* id is fed back to the prediction network (SOS/pad index ``num_spell``).

CUDA graphs are NOT supported yet: the computer always runs the pure-PyTorch :meth:`torch_impl`.
"""

from typing import Optional

import torch
from omegaconf import ListConfig

from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import (
    BatchedLabelLoopingState,
    GreedyBatchedLabelLoopingComputerBase,
    LabelLoopingStateItem,
)
from nemo.collections.asr.parts.utils import rnnt_utils


class GreedyBatchedMultiStreamCapPunctTDTLabelLoopingComputer(GreedyBatchedLabelLoopingComputerBase):
    """Batched greedy Label-Looping decoder for spelling + capitalization + punctuation TDT.

    Args:
        decoder: prediction network (operates on the spelling vocabulary).
        joint: joint network with the layout ``[punct | cap | spell | blank | durations]``.
        blank_index: absolute joint blank index (``num_punct + num_cap + num_spell``).
        durations: list of TDT durations, e.g. ``[0, 1, 2, 3, 4]``.
        num_punct: number of punctuation classes (incl. PUNCT_NONE).
        num_cap: number of capitalization classes.
        num_spell: spelling vocabulary size (the spelling blank/SOS index).
        max_symbols_per_step: max symbols to emit per frame (to avoid infinite looping).
        include_duration: whether to store predicted token durations in the hypotheses.
    """

    def __init__(
        self,
        decoder,
        joint,
        blank_index: int,
        durations: list[int] | ListConfig,
        num_punct: int,
        num_cap: int,
        num_spell: int,
        max_symbols_per_step: Optional[int] = 10,
        include_duration: bool = False,
    ):
        super().__init__()
        self.decoder = decoder
        self.joint = joint
        # keep durations on CPU to avoid side effects in multi-gpu environment
        self.durations = torch.tensor(list(durations), device="cpu").to(torch.long)
        self.num_punct = int(num_punct)
        self.num_cap = int(num_cap)
        self.num_spell = int(num_spell)
        self._blank_index = int(blank_index)
        # The prediction network operates on the spelling vocabulary (SOS/blank index = num_spell).
        self._SOS = self.num_spell
        self.spell_blank = self.num_spell  # blank index *within* the spelling slice
        self.max_symbols = max_symbols_per_step
        self.include_duration = include_duration
        self.preserve_alignments = False
        self.preserve_frame_confidence = False

        # CUDA graphs are not supported (yet) for the multistream decoder.
        self.allow_cuda_graphs = False
        self.cuda_graphs_mode = None
        self.cuda_graphs_allow_fallback = False
        self.state = None

        # no fusion / biasing models
        self.fusion_models = []
        self.fusion_models_alpha = []
        self.biasing_multi_model = None

    def maybe_enable_cuda_graphs(self) -> bool:
        return False

    def disable_cuda_graphs(self) -> bool:
        return False

    def reset_cuda_graphs_state(self):
        self.state = None

    def cuda_graphs_impl(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
        prev_batched_state: Optional[BatchedLabelLoopingState] = None,
        multi_biasing_ids: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError(
            "CUDA graphs are not supported for the multistream cap+punct TDT batched decoder. Use `torch_impl`."
        )

    def torch_impl(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
        prev_batched_state: Optional[BatchedLabelLoopingState] = None,
        multi_biasing_ids: Optional[torch.Tensor] = None,
    ) -> tuple[rnnt_utils.BatchedHyps, None, BatchedLabelLoopingState]:
        """Pure-PyTorch batched greedy label-looping decoding (3-stream cap+punct TDT)."""
        batch_size, max_time, _unused = encoder_output.shape
        device = encoder_output.device

        encoder_output_projected = self.joint.project_encoder(encoder_output)
        float_dtype = encoder_output_projected.dtype

        num_punct = self.num_punct
        num_cap = self.num_cap
        num_spell = self.num_spell
        spell_blank = self.spell_blank
        cap_start = num_punct
        spell_start = num_punct + num_cap

        batched_hyps = rnnt_utils.BatchedHyps(
            batch_size=batch_size,
            init_length=max_time * self.max_symbols if self.max_symbols is not None else max_time,
            device=device,
            float_dtype=float_dtype,
            is_with_durations=self.include_duration,
        )

        model_durations = self.durations.to(device, non_blocking=True)
        num_durations = model_durations.shape[0]

        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
        last_timesteps = torch.maximum(encoder_output_length - 1, torch.zeros_like(encoder_output_length))
        time_indices = (
            torch.zeros_like(batch_indices) if prev_batched_state is None else prev_batched_state.time_jumps.clone()
        )
        safe_time_indices = torch.minimum(time_indices, last_timesteps)
        time_indices_current_labels = torch.zeros_like(time_indices)

        active_mask = time_indices < encoder_output_length
        active_mask_prev = active_mask.clone()

        if prev_batched_state is None:
            state = self.decoder.initialize_state(encoder_output_projected)
            spell_labels = torch.full_like(batch_indices, fill_value=self._SOS)
            decoder_output, state, *_ = self.decoder.predict(
                spell_labels.unsqueeze(1), state, add_sos=False, batch_size=batch_size
            )
            decoder_output = self.joint.project_prednet(decoder_output)
        else:
            decoder_output = prev_batched_state.predictor_outputs
            state = prev_batched_state.predictor_states
            spell_labels = torch.full_like(batch_indices, fill_value=self._SOS)

        cap_labels = torch.zeros_like(batch_indices)
        punct_labels = torch.zeros_like(batch_indices)
        scores = torch.zeros(batch_size, device=device, dtype=float_dtype)
        durations = torch.zeros_like(batch_indices)

        def split_label_logits(logits):
            """Return (spell_scores, spell_k, cap_k, punct_k) from the label part of `logits`."""
            label_logits = logits[:, :-num_durations]
            sp_scores, sp_k = label_logits[:, spell_start:].max(dim=-1)  # spell incl. blank
            _, cp_k = label_logits[:, cap_start:spell_start].max(dim=-1)
            _, pn_k = label_logits[:, :cap_start].max(dim=-1)
            return sp_scores, sp_k, cp_k, pn_k

        while active_mask.any():
            active_mask_prev.copy_(active_mask)

            logits = (
                self.joint.joint_after_projection(
                    encoder_output_projected[batch_indices, safe_time_indices].unsqueeze(1),
                    decoder_output,
                )
                .squeeze(1)
                .squeeze(1)
            )
            scores, spell_labels, cap_labels, punct_labels = split_label_logits(logits)
            durations = model_durations[logits[:, -num_durations:].argmax(dim=-1)]

            blank_mask = spell_labels == spell_blank
            durations.masked_fill_(torch.logical_and(durations == 0, blank_mask), 1)
            time_indices_current_labels.copy_(time_indices)

            time_indices = time_indices + durations * active_mask
            torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
            active_mask = time_indices < encoder_output_length
            advance_mask = torch.logical_and(active_mask, blank_mask)

            while advance_mask.any():
                torch.where(advance_mask, time_indices, time_indices_current_labels, out=time_indices_current_labels)
                logits = (
                    self.joint.joint_after_projection(
                        encoder_output_projected[batch_indices, safe_time_indices].unsqueeze(1),
                        decoder_output,
                    )
                    .squeeze(1)
                    .squeeze(1)
                )
                more_scores, more_spell, more_cap, more_punct = split_label_logits(logits)
                torch.where(advance_mask, more_spell, spell_labels, out=spell_labels)
                torch.where(advance_mask, more_cap, cap_labels, out=cap_labels)
                torch.where(advance_mask, more_punct, punct_labels, out=punct_labels)
                torch.where(advance_mask, more_scores, scores, out=scores)
                more_durations = model_durations[logits[:, -num_durations:].argmax(dim=-1)]

                blank_mask = spell_labels == spell_blank
                more_durations.masked_fill_(torch.logical_and(more_durations == 0, blank_mask), 1)
                torch.where(advance_mask, time_indices + more_durations, time_indices, out=time_indices)
                torch.where(advance_mask, more_durations, durations, out=durations)
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                active_mask = time_indices < encoder_output_length
                advance_mask = torch.logical_and(active_mask, blank_mask)

            found_labels_mask = torch.logical_and(active_mask_prev, spell_labels != spell_blank)
            # product id: (punct * num_cap + cap) * num_spell + spell ; only committed where found
            combined_labels = (punct_labels * num_cap + cap_labels) * num_spell + spell_labels
            if self.max_symbols is not None:
                batched_hyps.add_results_masked_no_checks_(
                    active_mask=found_labels_mask,
                    labels=combined_labels,
                    time_indices=time_indices_current_labels,
                    scores=scores,
                    token_durations=durations if self.include_duration else None,
                )
            else:
                batched_hyps.add_results_masked_(
                    active_mask=found_labels_mask,
                    labels=combined_labels,
                    time_indices=time_indices_current_labels,
                    scores=scores,
                    token_durations=durations if self.include_duration else None,
                )

            # prediction network step using the *spelling* labels
            prev_state = state
            prev_decoder_output = decoder_output
            decoder_output, state, *_ = self.decoder.predict(
                spell_labels.unsqueeze(1), state, add_sos=False, batch_size=batch_size
            )
            decoder_output = self.joint.project_prednet(decoder_output)
            self.decoder.batch_replace_states_mask(
                src_states=prev_state, dst_states=state, mask=~found_labels_mask
            )
            torch.where(
                found_labels_mask.unsqueeze(-1).unsqueeze(-1), decoder_output, prev_decoder_output, out=decoder_output
            )

            if self.max_symbols is not None:
                force_blank_mask = torch.logical_and(
                    active_mask,
                    torch.logical_and(
                        torch.logical_and(
                            spell_labels != spell_blank,
                            batched_hyps.last_timestamp_lasts >= self.max_symbols,
                        ),
                        batched_hyps.last_timestamp == time_indices,
                    ),
                )
                time_indices = time_indices + force_blank_mask
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                active_mask = time_indices < encoder_output_length

        if prev_batched_state is not None:
            batched_hyps.timestamps += prev_batched_state.decoded_lengths.unsqueeze(1)

        # last spelling label for state feedback (num_spell, the spelling SOS/pad, is a valid product
        # id, so detect "nothing decoded" via current_lengths rather than a sentinel id).
        decoded_any = batched_hyps.current_lengths > 0
        last_product = batched_hyps.get_last_labels(pad_id=0)
        last_spell = torch.where(
            decoded_any, last_product % num_spell, torch.full_like(last_product, self._SOS)
        )
        decoding_state = BatchedLabelLoopingState(
            predictor_states=state,
            predictor_outputs=decoder_output,
            labels=(
                torch.where(decoded_any, last_spell, prev_batched_state.labels)
                if prev_batched_state is not None
                else last_spell
            ),
            decoded_lengths=(
                encoder_output_length.clone()
                if prev_batched_state is None
                else encoder_output_length + prev_batched_state.decoded_lengths
            ),
            fusion_states_list=[],
            time_jumps=time_indices - encoder_output_length,
        )
        return batched_hyps, None, decoding_state

    def split_batched_state(self, state: BatchedLabelLoopingState) -> list[LabelLoopingStateItem]:
        state_items: list[LabelLoopingStateItem] = []
        for i, predictor_state in enumerate(self.decoder.batch_split_states(state.predictor_states)):
            state_items.append(
                LabelLoopingStateItem(
                    predictor_state=predictor_state,
                    predictor_output=state.predictor_outputs[i],
                    label=state.labels[i],
                    decoded_length=state.decoded_lengths[i],
                    fusion_state_list=[],
                    time_jump=state.time_jumps[i],
                )
            )
        return state_items

    def merge_to_batched_state(self, state_items: list[LabelLoopingStateItem | None]) -> BatchedLabelLoopingState:
        if any(item is None for item in state_items):
            not_none_item = next(item for item in state_items if item is not None)
            assert not_none_item is not None
            device = not_none_item.predictor_output.device
            labels = torch.full([1], fill_value=self._SOS, dtype=torch.long, device=device)
            decoder_output, predictor_state, *_ = self.decoder.predict(
                labels.unsqueeze(1), None, add_sos=False, batch_size=1
            )
            decoder_output = self.joint.project_prednet(decoder_output)
            start_item = LabelLoopingStateItem(
                predictor_state=self.decoder.batch_split_states(predictor_state)[0],
                predictor_output=decoder_output[0],
                label=labels[0],
                decoded_length=torch.zeros([], dtype=torch.long, device=device),
                fusion_state_list=[],
                time_jump=torch.zeros([], dtype=torch.long, device=device),
            )
            state_items = [item if item is not None else start_item for item in state_items]

        return BatchedLabelLoopingState(
            predictor_states=self.decoder.batch_unsplit_states([item.predictor_state for item in state_items]),
            predictor_outputs=torch.stack([item.predictor_output for item in state_items]),
            labels=torch.stack([item.label for item in state_items]),
            decoded_lengths=torch.stack([item.decoded_length for item in state_items]),
            fusion_states_list=[],
            time_jumps=torch.stack([item.time_jump for item in state_items]),
        )
