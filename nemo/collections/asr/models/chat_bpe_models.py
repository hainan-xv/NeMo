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
"""CHAT transducer with a selectable training objective.

ONE model, two losses. ``EncDecCHATBPEModel`` is an ``EncDecRNNTBPEModel`` --
same ``RNNTAttJoint``, same ``RNNTDecoder``, same 1,024-piece SentencePiece
vocabulary plus blank, same greedy chunk-synchronous decoding -- with a single
switch::

    model.loss_type: rnnt              # marginalise over every alignment
    model.loss_type: forced_alignment  # condition on ONE alignment

This replaces a separate ChatSTTModel class. Two classes for two losses meant
two vocabularies (the old one carried three unused SCRIPT delimiters, giving
1,027 text classes against the standard 1,024), two decode paths to keep in
sync, and a checkpoint from one that could not be loaded by the other. Folding
the objective into a config field makes the two arms differ in exactly the thing
under study and nothing else, and lets either arm initialise from the other.

WHY A FORCED ALIGNMENT AT ALL. The RNN-T loss sums over every way of
interleaving labels and blanks, which needs a ``[B, T, U, V+1]`` tensor. Scoring
one path instead needs only ``[U+T, V+1]``: the cost stops scaling with the
lattice, which is what makes a large vocabulary trainable. The bet is that a
good alignment is nearly as good a target as the marginal -- that is what the
two ``loss_type`` settings measure against each other.

The forced path is built from the word timings the Granary cuts already carry in
``cut.custom["alignments"]``, so it needs no new dataset class -- only
``return_cuts=True`` on the ordinary Lhotse BPE dataset.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from lhotse.dataset.collation import collate_vectors
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.utils.chat_alignment import assign_words_to_chunks, build_forced_path
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

__all__ = ["EncDecCHATBPEModel"]

LOSS_TYPES = ("rnnt", "forced_alignment")


class EncDecCHATBPEModel(EncDecRNNTBPEModel):
    """CHAT transducer trainable with either the marginalised or a forced loss."""

    def __init__(self, cfg: DictConfig, trainer=None):
        # EVERY attribute the data-setup path reads must be assigned BEFORE
        # super().__init__(): ModelPT.__init__ calls setup_training_data() from
        # inside it, so anything set afterwards does not exist yet and the model
        # dies with AttributeError at construction.
        self.loss_type = str(cfg.get("loss_type", "rnnt"))
        if self.loss_type not in LOSS_TYPES:
            raise ValueError(f"model.loss_type must be one of {LOSS_TYPES}, got {self.loss_type!r}")

        fa = cfg.get("forced_alignment", {}) or {}
        self.num_delay_frames = int(fa.get("num_delay_frames", 0))
        self.recover_history_words = int(fa.get("recover_history_words", 0))
        # Seconds of audio per ENCODER frame: 10 ms hop x 8x subsampling. Only
        # used to turn word end times into frame indices; a wrong value silently
        # shifts every word to the wrong chunk, so it is derived from the
        # configured preprocessor rather than hard-coded.
        self.frame_length_in_secs = float(
            fa.get("frame_length_in_secs", None) or cfg.preprocessor.window_stride * cfg.encoder.subsampling_factor
        )

        self._ws_ids: Optional[frozenset] = None
        # Set only while the TRAINING loader is being built: the forced loss
        # needs the cuts (for their alignments), validation does not and is
        # scored by the ordinary WER path.
        self._want_cuts = False

        super().__init__(cfg=cfg, trainer=trainer)

        if self.loss_type == "forced_alignment":
            logging.info(
                f"CHAT forced-alignment loss: delay={self.num_delay_frames} frames, "
                f"recover_history_words={self.recover_history_words}, "
                f"frame_length={self.frame_length_in_secs:.4f}s, chunk_size={self.joint.chunk_size}"
            )

    # ------------------------------------------------------------------ data

    def setup_training_data(self, train_data_config):
        self._want_cuts = self.loss_type == "forced_alignment"
        try:
            super().setup_training_data(train_data_config)
        finally:
            self._want_cuts = False

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if self._want_cuts and config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextBpeDataset(tokenizer=self.tokenizer, return_cuts=True),
                tokenizer=self.tokenizer,
            )
        return super()._setup_dataloader_from_config(config)

    # -------------------------------------------------------- forced path

    def _word_start_ids(self) -> frozenset:
        """Token ids that begin a word (U+2581 SentencePiece, U+0120 byte BPE).

        Training extends a chunk backward over whole words and retract-style
        decoding rolls back over whole words; both must agree on where a word
        starts, or the model would be asked to recover from states it never saw.
        """
        if self._ws_ids is None:
            ids = set()
            for i in range(self.tokenizer.vocab_size):
                try:
                    got = self.tokenizer.ids_to_tokens([i])
                except Exception:  # noqa: BLE001 -- a gap in the vocab is not fatal
                    continue
                piece = got[0] if got else None
                if isinstance(piece, str) and (piece.startswith("▁") or piece.startswith("Ġ")):
                    ids.add(i)
            self._ws_ids = frozenset(ids)
        return self._ws_ids

    def _chunk_tokens(self, cut, n_chunks: int) -> List[List[int]]:
        """Tokens each chunk is responsible for, one list per chunk.

        A chunk's words are joined and tokenized TOGETHER rather than one word at
        a time, so the pieces match what the tokenizer would produce for running
        text.
        """
        words = (cut.custom or {}).get("alignments", []) or []
        groups = assign_words_to_chunks(
            [w["end_time"] for w in words],
            n_chunks,
            self.joint.chunk_size,
            self.frame_length_in_secs,
            self.num_delay_frames,
        )
        out = []
        for gi, idxs in enumerate(groups):
            if not idxs:
                out.append([])
                continue
            text = " ".join(words[i]["text"] for i in idxs)
            # Every chunk but the first continues mid-utterance, so its opening
            # token must carry the word-boundary marker; without the leading
            # space SentencePiece would emit a continuation piece instead.
            out.append(self.tokenizer.text_to_ids(text if gi == 0 else " " + text))
        return out

    def _build_batch_path(self, cuts, n_chunks: torch.Tensor, device):
        """Assemble (b, t, u, labels) and the prediction-network input."""
        blank = self.joint.num_classes_with_blank - 1
        ws = self._word_start_ids() if self.recover_history_words > 0 else None

        b_all, t_all, u_all, lab_all, preds = [], [], [], [], []
        for b, cut in enumerate(cuts):
            chunks = self._chunk_tokens(cut, int(n_chunks[b]))
            starts = [[i for i, t in enumerate(c) if i == 0 or t in ws] for c in chunks] if ws else None
            t_idx, u_idx, labels = build_forced_path(chunks, blank, self.recover_history_words, starts)
            b_all += [b] * len(t_idx)
            t_all += t_idx
            u_all += u_idx
            lab_all += labels
            preds.append(torch.tensor([tok for c in chunks for tok in c], dtype=torch.long))

        pred_lens = torch.tensor([p.numel() for p in preds], dtype=torch.long)
        pred_input = collate_vectors(preds, padding_value=0).to(device)
        as_t = lambda x: torch.tensor(x, dtype=torch.long, device=device)  # noqa: E731
        return as_t(b_all), as_t(t_all), as_t(u_all), as_t(lab_all), pred_input, pred_lens.to(device)

    def _forced_alignment_loss(self, encoded, encoded_len, cuts) -> torch.Tensor:
        # Chunk counts come from the ACTUAL encoder output, not from the
        # duration, so the two sides cannot disagree about the tail chunk.
        chunk_size = self.joint.chunk_size
        n_chunks = torch.div(encoded_len + chunk_size - 1, chunk_size, rounding_mode="floor")

        b_idx, t_idx, u_idx, labels, pred_input, pred_lens = self._build_batch_path(
            cuts, n_chunks.cpu(), encoded.device
        )

        if labels.numel() == 0:
            # cross_entropy over an empty path is nan, and that nan reaches the
            # weights. A batch can legitimately have no alignable words; return a
            # graph-connected zero so backward still runs on every rank and DDP
            # stays in lockstep.
            logging.warning(f"empty forced-alignment path at step {self.global_step}; contributing zero loss")
            return encoded.sum() * 0.0

        g, _, _ = self.decoder(targets=pred_input, target_length=pred_lens)
        g = g.transpose(1, 2)  # (B, D, U+1) -> (B, U+1, D); the decoder prepends its own SOS

        # forward() hands back [B, D, T], but joint_on_path chunks along the TIME
        # axis and would otherwise slice the feature axis into "chunks" -- which
        # reshapes to a plausible-looking tensor rather than failing loudly.
        logits = self.joint.joint_on_path(encoded.transpose(1, 2), g, b_idx, t_idx, u_idx, encoded_len)
        return F.cross_entropy(logits.float(), labels)

    # ------------------------------------------------------------- training

    def training_step(self, batch, batch_nb):
        if self.loss_type != "forced_alignment":
            return super().training_step(batch, batch_nb)

        signal, signal_len, transcript, transcript_len, cuts = batch
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        loss_value = self.add_auxiliary_losses(self._forced_alignment_loss(encoded, encoded_len, cuts))

        # train_loss on the progress bar, not just in wandb: a forced-alignment
        # loss that collapses to ~0 in the first few steps means the targets are
        # empty, and that has happened here before. It should be visible in the
        # job's own log without opening a browser.
        self.log('train_loss', loss_value, prog_bar=True)
        self.log_dict(
            {
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }
        )
        return loss_value

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []
