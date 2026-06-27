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

"""Frozen external CTC model used to generate label->chunk alignments on the fly.

This powers the *Chunkwise-Aligner* baseline (arXiv:2605.11422): instead of the
alignment-free full-sum objective, an external model fixes which chunk each label
is emitted in, and the trainee maximizes the probability of that single path.

The aligner loads a pretrained, frozen CTC model (kept out of the trainee's
parameter set / state dict), CTC-force-aligns the trainee's target token sequence
to the audio with :func:`viterbi_decoding`, reads off each token's start frame,
and buckets it into a fixed-size encoder chunk. The external model MUST share the
trainee's tokenizer/vocabulary so that its forced alignment is indexed by the
trainee's token ids. Frame-rate mismatches between the two encoders are handled
by mapping the token start frame *proportionally* into the trainee's encoder
frame axis, so no stride bookkeeping is required.

Utterances whose resulting assignment cannot be left-packed into the chunk
lattice (a chunk receives more tokens than it has frames, the audio has fewer
frames than tokens ``T < U``, or a token gets no alignment) are flagged invalid
so the caller can skip them and report the discard ratio.
"""

from typing import Optional, Tuple

import torch

from nemo.collections.asr.parts.utils.aligner_utils import viterbi_decoding
from nemo.utils import logging

__all__ = ['ExternalCTCForcedAligner']


class ExternalCTCForcedAligner:
    """Load a frozen CTC model and emit per-token chunk assignments via forced alignment.

    Args:
        model_path: path to a local ``.nemo`` CTC checkpoint.
        pretrained_name: NGC/HF pretrained name (used if ``model_path`` is None).
        expected_vocab_size: trainee vocabulary size (excluding blank). If given,
            a mismatch with the external model's vocab is logged as a warning --
            the alignment is only meaningful when the two share a tokenizer.
        viterbi_device: device for the Viterbi pass (default: the audio's device).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        pretrained_name: Optional[str] = None,
        expected_vocab_size: Optional[int] = None,
        viterbi_device: Optional[str] = None,
    ):
        if not model_path and not pretrained_name:
            raise ValueError(
                "ExternalCTCForcedAligner requires either `model_path` (a local .nemo) "
                "or `pretrained_name` (a pretrained model name)."
            )

        # Import lazily to avoid a heavy import at module load time.
        from nemo.collections.asr.models import ASRModel

        if model_path:
            logging.info(f"[chunkwise-aligner] Loading frozen external CTC aligner from file: {model_path}")
            model = ASRModel.restore_from(restore_path=model_path, map_location="cpu")
        else:
            logging.info(f"[chunkwise-aligner] Loading frozen external CTC aligner: {pretrained_name}")
            model = ASRModel.from_pretrained(model_name=pretrained_name, map_location="cpu")

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # Number of output columns of the CTC head, blank is the last index.
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            ext_vocab = model.tokenizer.vocab_size
        elif hasattr(model, 'decoder') and hasattr(model.decoder, 'vocabulary'):
            ext_vocab = len(model.decoder.vocabulary)
        else:
            ext_vocab = None

        if expected_vocab_size is not None and ext_vocab is not None and ext_vocab != expected_vocab_size:
            logging.warning(
                "[chunkwise-aligner] External CTC aligner vocab size "
                f"({ext_vocab}) != trainee vocab size ({expected_vocab_size}). The forced alignment is only "
                "meaningful if the external model shares the trainee's tokenizer; mismatched ids will produce "
                "garbage chunk assignments."
            )

        self._model = model
        self._device = torch.device("cpu")
        self.viterbi_device = viterbi_device

    def to(self, device, dtype=None):
        self._model = self._model.to(device)
        self._device = torch.device(device)
        return self

    @torch.no_grad()
    def _ctc_log_probs(self, input_signal: torch.Tensor, input_signal_length: torch.Tensor):
        """Run the frozen CTC model -> (log_probs [B, T_ext, V], enc_len_ext [B])."""
        if self._device != input_signal.device:
            self.to(input_signal.device)

        # The external model is fp32 and frozen; disable autocast so its preprocessor
        # / encoder run in full precision regardless of the trainee's AMP settings.
        autocast = torch.cuda.amp.autocast(enabled=False) if input_signal.is_cuda else torch.no_grad()
        with autocast:
            outputs = self._model.forward(
                input_signal=input_signal.float(),
                input_signal_length=input_signal_length,
            )
        log_probs, enc_len_ext = outputs[0], outputs[1]
        return log_probs.float(), enc_len_ext

    @torch.no_grad()
    def align_to_chunks(
        self,
        input_signal: torch.Tensor,
        input_signal_length: torch.Tensor,
        labels: torch.Tensor,
        label_lens: torch.Tensor,
        target_frame_lengths: torch.Tensor,
        chunk_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Force-align labels and bucket each token into a trainee encoder chunk.

        Args:
            input_signal: raw audio ``[B, T_audio]``.
            input_signal_length: audio lengths ``[B]``.
            labels: trainee target token ids ``[B, U]`` (no blanks/EOS).
            label_lens: number of valid labels per sample ``[B]``.
            target_frame_lengths: number of trainee encoder frames per sample
                ``[B]`` (post token-extraction; this is the axis the loss indexes).
            chunk_size: number of trainee encoder frames per chunk ``C``.

        Returns:
            token_chunk_ids: ``[B, U]`` long; chunk index per token, ``-1`` padded
                beyond ``label_lens`` and for invalid samples.
            valid_mask: ``[B]`` bool; ``False`` for utterances to skip.
        """
        B = labels.shape[0]
        U_max = int(labels.shape[1])
        device = labels.device

        log_probs, enc_len_ext = self._ctc_log_probs(input_signal, input_signal_length)
        V_ext = int(log_probs.shape[-1])
        blank_ext = V_ext - 1
        T_ext_max = int(log_probs.shape[1])

        label_lens = label_lens.to(torch.long).cpu()
        target_frame_lengths = target_frame_lengths.to(torch.long).cpu()
        labels_cpu = labels.to(torch.long).cpu()
        enc_len_ext = enc_len_ext.to(torch.long).cpu()

        # Build the CTC-extended targets y_ext = [blank, y0, blank, y1, ..., blank]
        # of length 2U+1, padded with V_ext (the index of the column viterbi_decoding
        # appends for padding). U_ext_max covers the longest sample.
        U_ext_lens = (2 * label_lens + 1).clamp(min=1)
        U_ext_max = int(U_ext_lens.max().item())
        y_ext = torch.full((B, U_ext_max), V_ext, dtype=torch.long)
        for b in range(B):
            U_b = int(label_lens[b])
            seq = [blank_ext]
            for u in range(U_b):
                seq.append(int(labels_cpu[b, u]))
                seq.append(blank_ext)
            y_ext[b, : len(seq)] = torch.tensor(seq, dtype=torch.long)

        viterbi_device = self.viterbi_device or ("cuda" if log_probs.is_cuda else "cpu")
        alignments = viterbi_decoding(
            log_probs_batch=log_probs,
            y_batch=y_ext,
            T_batch=enc_len_ext.clamp(max=T_ext_max),
            U_batch=U_ext_lens,
            viterbi_device=viterbi_device,
        )

        token_chunk_ids = torch.full((B, U_max), -1, dtype=torch.long)
        valid_mask = torch.ones(B, dtype=torch.bool)

        for b in range(B):
            U_b = int(label_lens[b])
            T_tr = int(target_frame_lengths[b])
            if U_b == 0 or T_tr < U_b:
                valid_mask[b] = False
                continue

            n_chunks = (T_tr + chunk_size - 1) // chunk_size
            align_b = alignments[b]
            T_ext_b = max(len(align_b), 1)

            # First frame at which each real token's extended index (2i+1) appears.
            first_seen = {}
            for t, idx in enumerate(align_b):
                if idx % 2 == 1 and idx not in first_seen:
                    first_seen[idx] = t

            ok = True
            counts = [0] * n_chunks
            prev_chunk = 0
            for i in range(U_b):
                ext_idx = 2 * i + 1
                if ext_idx not in first_seen:
                    ok = False
                    break
                start_ext = first_seen[ext_idx]
                # Map the start frame proportionally into the trainee frame axis,
                # then bucket into a chunk. Proportional mapping is robust to any
                # subsampling/frame-rate mismatch between the two encoders.
                tr_frame = int((start_ext / T_ext_b) * T_tr)
                if tr_frame >= T_tr:
                    tr_frame = T_tr - 1
                chunk = tr_frame // chunk_size
                if chunk >= n_chunks:
                    chunk = n_chunks - 1
                # Forced alignment is monotonic, but proportional rounding could in
                # principle tie/decrease; clamp to keep the assignment non-decreasing.
                if chunk < prev_chunk:
                    chunk = prev_chunk
                prev_chunk = chunk
                token_chunk_ids[b, i] = chunk
                counts[chunk] += 1

            if not ok:
                valid_mask[b] = False
                token_chunk_ids[b, :] = -1
                continue

            # Left-packing feasibility: a chunk cannot host more tokens than frames.
            for c in range(n_chunks):
                frames_here = min(chunk_size, T_tr - c * chunk_size)
                if counts[c] > frames_here:
                    ok = False
                    break
            if not ok:
                valid_mask[b] = False
                token_chunk_ids[b, :] = -1

        return token_chunk_ids.to(device), valid_mask.to(device)
