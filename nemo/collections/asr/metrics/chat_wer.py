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
"""WER reported both raw and Whisper-normalised, from a single decode.

WHY BOTH. The ASR collection's ``WER`` compares decoded text to the reference
verbatim: on a PnC model ``Media.`` against ``media`` is a substitution. The
speechlm2 metric these CHAT runs are compared against instead pushed both sides
through Whisper's ``EnglishTextNormalizer`` first, which folds case, punctuation,
contractions and number formatting. The two therefore measure different things,
and a run logging one cannot be read against a run logging the other -- which is
exactly the trap that made a ported model look like it had stopped converging.

Reporting both from the same decode makes the comparison direct and costs
nothing: the decode dominates, and the normaliser runs on strings that have
already been produced.
"""

from typing import Optional, Tuple

import torch

from kaldialign import edit_distance

from nemo.collections.asr.metrics.wer import WER, move_dimension_to_the_front
from nemo.utils import logging

__all__ = ["ChatWER"]


def _load_normalizer():
    """Whisper's English normaliser, or None if it is not installed.

    Missing it must not take down training -- the raw number is still correct
    and is what the checkpoint callback monitors.
    """
    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        return EnglishTextNormalizer()
    except Exception as e:  # noqa: BLE001
        logging.warning(f"whisper_normalizer unavailable ({e!r}); val_wer_norm will not be reported")
        return None


class ChatWER(WER):
    """``WER`` that also accumulates normalised edit distances.

    The raw path is deliberately identical to the parent's, so ``val_wer`` keeps
    meaning exactly what it means for every other model in the collection and
    the checkpoint callback's monitor is unaffected.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._normalizer = _load_normalizer()
        # Plain attributes, not metric states: they must survive the reset() that
        # validation_pass performs right after compute().
        self.last_norm_scores = torch.tensor(0.0)
        self.last_norm_words = torch.tensor(0.0)  # replaced on the right device by compute()
        self._norm_scores = 0.0
        self._norm_words = 0.0

    def _texts(self, predictions, predictions_lengths, targets, targets_lengths, predictions_mask, input_ids):
        """Decode once; return (references, hypotheses) as strings."""
        references = []
        with torch.no_grad():
            tgt_lens = targets_lengths.long().cpu()
            tgt = targets.long().cpu()
            if self.batch_dim_index != 0:
                tgt = move_dimension_to_the_front(tgt, self.batch_dim_index)
            for i in range(tgt.shape[0]):
                references.append(self.decoding.decode_ids_to_str(tgt[i][: tgt_lens[i].item()].numpy().tolist()))
            hyps = (
                self.decode(predictions, predictions_lengths, predictions_mask, input_ids)
                if predictions.numel() > 0
                else []
            )
        texts = []
        for h in hyps:
            if isinstance(h, list):
                h = h[0]
            texts.append(h.text)
        return references, texts

    def update(
        self,
        predictions,
        predictions_lengths,
        targets,
        targets_lengths,
        predictions_mask=None,
        input_ids=None,
        **kwargs,
    ):  # noqa: E501
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

        refs, hyps = self._texts(
            predictions, predictions_lengths, targets, targets_lengths, predictions_mask, input_ids
        )

        if self._normalizer is not None:
            for h, r in zip(hyps, refs):
                r_list = self._normalizer(r).split()
                h_list = self._normalizer(h).split()
                self._norm_words += len(r_list)
                self._norm_scores += edit_distance(r_list, h_list)['total']

        # The raw numbers come from the parent, unchanged -- but feed it the
        # hypotheses we already decoded rather than paying for a second decode.
        saved, self.decode = self.decode, lambda *a, **k: [Hypothesis(score=0.0, y_sequence=[], text=t) for t in hyps]
        try:
            super().update(
                predictions=predictions,
                predictions_lengths=predictions_lengths,
                targets=targets,
                targets_lengths=targets_lengths,
                predictions_mask=predictions_mask,
                input_ids=input_ids,
                **kwargs,
            )
        finally:
            self.decode = saved

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # On the metric's OWN device, not the default one. These end up in the
        # dict validation_pass returns, and Lightning all-reduces every logged
        # value over the process group -- which is NCCL, and NCCL cannot reduce
        # a CPU tensor ("No backend type associated with device type cpu"). It
        # costs nothing on one GPU and kills an 8-node run at the first
        # validation.
        dev = self.scores.device
        self.last_norm_scores = torch.tensor(float(self._norm_scores), device=dev)
        self.last_norm_words = torch.tensor(float(self._norm_words), device=dev)
        return super().compute()

    def reset(self):
        self._norm_scores = 0.0
        self._norm_words = 0.0
        return super().reset()

    def normalized(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """(errors, words) under the Whisper normaliser, or None if unavailable."""
        if self._normalizer is None:
            return None
        return self.last_norm_scores, self.last_norm_words
