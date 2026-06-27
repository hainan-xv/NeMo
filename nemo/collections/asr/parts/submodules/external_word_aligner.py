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

"""Frozen external **word-level** forced aligner for the Chunkwise-Aligner baseline.

The original :class:`ExternalCTCForcedAligner` force-aligns the trainee's *token
ids* and therefore requires the external model to share the trainee's tokenizer.
That is impractical when the alignment is produced by a strong off-the-shelf
model with a completely different vocabulary (e.g. ``Qwen/Qwen3-ForcedAligner-0.6B``).

This module relaxes that constraint by aligning at the **word** level:

  1. The external aligner returns per-word start/end times (in seconds) for the
     reference transcript -- this is tokenizer-agnostic.
  2. Each word's start time is mapped *proportionally* into the trainee encoder
     frame axis and bucketed into a fixed-size chunk.
  3. The trainee's own sub-word tokens are grouped back into words (using the
     trainee tokenizer's word-boundary marker), and **every sub-word of a word is
     assigned the chunk of that word**.

So the external and trainee tokenizers never need to match -- only the *word
sequence* must agree (both come from the same normalized transcript).

Utterances whose word counts disagree, that cannot be left-packed into the chunk
lattice (a chunk would host more tokens than it has frames, or ``T < U``), or for
which the external aligner errors out, are flagged invalid so the caller can skip
them and report the discard ratio (same contract as the CTC aligner).
"""

from typing import List, Optional, Sequence, Tuple

import torch

from nemo.utils import logging

__all__ = ['QwenWordForcedAligner']


# SentencePiece / byte-level-BPE word-start markers and the WordPiece continuation
# marker. SentencePiece (``type: bpe`` in NeMo) prefixes the first sub-word of each
# word with U+2581 ("▁"); byte-level BPE (GPT-2 style) uses "Ġ"; WordPiece marks
# *continuation* pieces with "##".
_SPE_WORD_START = "\u2581"
_BBPE_WORD_START = "\u0120"  # 'Ġ'
_WORDPIECE_CONT = "##"


def _to_torch_dtype(dtype) -> Optional[torch.dtype]:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).lower().replace("torch.", "")
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }.get(name, None)


def group_token_ids_into_words(pieces: Sequence[str]) -> List[List[int]]:
    """Group a sequence of sub-word *pieces* into word index-groups.

    Returns a list of words, each a list of indices into ``pieces`` that belong to
    that word. Handles SentencePiece / byte-level-BPE (word-start markers) and
    WordPiece (``##`` continuation) conventions. The first piece always starts a
    word.
    """
    is_wordpiece = any(p.startswith(_WORDPIECE_CONT) for p in pieces)
    groups: List[List[int]] = []
    for i, p in enumerate(pieces):
        if i == 0:
            new_word = True
        elif is_wordpiece:
            new_word = not p.startswith(_WORDPIECE_CONT)
        else:
            new_word = p.startswith(_SPE_WORD_START) or p.startswith(_BBPE_WORD_START)
        if new_word or not groups:
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


class QwenWordForcedAligner:
    """Word-level forced aligner backed by ``Qwen3-ForcedAligner`` (or compatible).

    Args:
        model_name_or_path: HF repo id or local dir of the Qwen forced aligner
            (default ``Qwen/Qwen3-ForcedAligner-0.6B``).
        tokenizer: the *trainee* tokenizer (``TokenizerSpec``). Used only to (a)
            detokenize the trainee labels into reference words for the aligner and
            (b) group trainee sub-words back into words. The aligner's own
            vocabulary is irrelevant.
        language: language name passed to the aligner (e.g. ``"English"``).
        dtype: aligner compute dtype (string or ``torch.dtype``).
        device: device to load the aligner on. ``None`` -> inferred from the audio
            on first use.
        sample_rate: sample rate of ``input_signal`` (Hz). Word times are mapped
            into trainee frames using the audio duration, so this must match the
            audio fed to the trainee.
    """

    def __init__(
        self,
        tokenizer,
        model_name_or_path: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        language: str = "English",
        dtype="bfloat16",
        device: Optional[str] = None,
        sample_rate: int = 16000,
    ):
        if tokenizer is None:
            raise ValueError(
                "QwenWordForcedAligner needs the trainee tokenizer to group sub-words into words. "
                "The word-level (Qwen) backend currently requires a sub-word tokenizer model "
                "(loss_type='chunkwise_aligner' with backend='qwen')."
            )
        self.tokenizer = tokenizer
        self.model_name_or_path = model_name_or_path
        self.language = language
        self.dtype = _to_torch_dtype(dtype)
        self.sample_rate = int(sample_rate)
        self._device = torch.device(device) if device is not None else None
        self._aligner = None  # lazily constructed on first use

    # ------------------------------------------------------------------ loading
    def _ensure_loaded(self, device: torch.device):
        if self._aligner is not None:
            return
        if self._device is None:
            self._device = device
        try:
            from qwen_asr import Qwen3ForcedAligner
        except Exception as e:  # pragma: no cover - depends on optional package
            raise ImportError(
                "The Qwen word-level aligner backend requires the `qwen-asr` package "
                "(pip install -U qwen-asr) which provides `Qwen3ForcedAligner`. "
                f"Import failed: {e!r}"
            )
        device_map = str(self._device) if self._device.type == "cuda" else "cpu"
        logging.info(
            f"[chunkwise-aligner] Loading frozen word-level aligner '{self.model_name_or_path}' "
            f"on {device_map} (dtype={self.dtype})."
        )
        kwargs = {"device_map": device_map}
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype
        self._aligner = Qwen3ForcedAligner.from_pretrained(self.model_name_or_path, **kwargs)

    def to(self, device, dtype=None):
        # The HF aligner is placed via device_map at load time; just record intent.
        self._device = torch.device(device)
        if dtype is not None:
            self.dtype = _to_torch_dtype(dtype)
        return self

    # --------------------------------------------------------------- tokenizing
    def _pieces(self, ids: List[int]) -> List[str]:
        return [str(p) for p in self.tokenizer.ids_to_tokens(ids)]

    def _word_text(self, ids: List[int], group: List[int]) -> str:
        return self.tokenizer.ids_to_text([ids[j] for j in group]).strip()

    # ------------------------------------------------------------------ qwen call
    def _run_aligner(self, audios, texts) -> List[Sequence]:
        """Call the external aligner. Returns one result per sample; each result is
        a sequence of items exposing ``.start_time`` (seconds). Isolated so tests
        can stub it without the heavy model / package."""
        self._ensure_loaded(self._device or torch.device("cpu"))
        languages = [self.language] * len(audios)
        return self._aligner.align(audio=audios, text=texts, language=languages)

    # --------------------------------------------------------------------- align
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
        """Word-align the batch and bucket each token into a trainee encoder chunk.

        Mirrors :meth:`ExternalCTCForcedAligner.align_to_chunks`:

        Returns:
            token_chunk_ids: ``[B, U]`` long; chunk index per token, ``-1`` padded
                beyond ``label_lens`` and for invalid samples.
            valid_mask: ``[B]`` bool; ``False`` for utterances to skip.
        """
        B = int(labels.shape[0])
        U_max = int(labels.shape[1])
        out_device = labels.device

        # Load the (heavy) aligner on the same accelerator as the audio on first use.
        if self._device is None and input_signal.is_cuda:
            self._device = input_signal.device

        label_lens_cpu = label_lens.to(torch.long).cpu()
        target_frames_cpu = target_frame_lengths.to(torch.long).cpu()
        sig_len_cpu = input_signal_length.to(torch.long).cpu()
        labels_cpu = labels.to(torch.long).cpu()

        token_chunk_ids = torch.full((B, U_max), -1, dtype=torch.long)
        valid_mask = torch.ones(B, dtype=torch.bool)

        # 1) Group each sample's sub-words into words; build reference text + audio.
        word_groups: List[List[List[int]]] = [[] for _ in range(B)]
        word_texts: List[List[str]] = [[] for _ in range(B)]
        audios = []
        align_indices: List[int] = []
        for b in range(B):
            U_b = int(label_lens_cpu[b])
            if U_b <= 0:
                valid_mask[b] = False
                continue
            ids_b = labels_cpu[b, :U_b].tolist()
            groups = group_token_ids_into_words(self._pieces(ids_b))
            word_groups[b] = groups
            texts_b = [self._word_text(ids_b, g) for g in groups]
            # Drop words that detokenize to empty (cannot be aligned / matched).
            if any(len(t) == 0 for t in texts_b) or len(groups) == 0:
                valid_mask[b] = False
                continue
            word_texts[b] = texts_b

            n = max(int(sig_len_cpu[b]), 1)
            wav = input_signal[b, :n].detach().to('cpu', dtype=torch.float32).contiguous().numpy()
            audios.append((wav, self.sample_rate))
            align_indices.append(b)

        if not align_indices:
            return token_chunk_ids.to(out_device), valid_mask.to(out_device)

        # 2) Run the external word aligner on the valid subset (best-effort: any
        #    failure discards that batch's samples but keeps training alive).
        texts_for_aligner = [' '.join(word_texts[b]) for b in align_indices]
        try:
            results = self._run_aligner(audios, texts_for_aligner)
        except Exception as e:
            logging.warning(
                f"[chunkwise-aligner] External word aligner failed on a batch of "
                f"{len(align_indices)} samples; discarding them this step. Cause: {e!r}"
            )
            for b in align_indices:
                valid_mask[b] = False
            return token_chunk_ids.to(out_device), valid_mask.to(out_device)

        if len(results) != len(align_indices):
            logging.warning(
                f"[chunkwise-aligner] External aligner returned {len(results)} results for "
                f"{len(align_indices)} inputs; discarding this batch."
            )
            for b in align_indices:
                valid_mask[b] = False
            return token_chunk_ids.to(out_device), valid_mask.to(out_device)

        # 3) Map each word's start time -> trainee frame -> chunk; expand to sub-words.
        for result, b in zip(results, align_indices):
            U_b = int(label_lens_cpu[b])
            T_tr = int(target_frames_cpu[b])
            groups = word_groups[b]
            n_words = len(groups)

            # Word counts must agree between the trainee transcript and the aligner.
            if len(result) != n_words or T_tr < U_b:
                valid_mask[b] = False
                continue

            audio_dur = max(int(sig_len_cpu[b]) / float(self.sample_rate), 1e-6)
            n_chunks = (T_tr + chunk_size - 1) // chunk_size
            counts = [0] * n_chunks
            prev_chunk = 0
            ok = True
            for wi, group in enumerate(groups):
                start_sec = float(getattr(result[wi], 'start_time', 0.0) or 0.0)
                tr_frame = int((start_sec / audio_dur) * T_tr)
                tr_frame = min(max(tr_frame, 0), T_tr - 1)
                chunk = tr_frame // chunk_size
                if chunk >= n_chunks:
                    chunk = n_chunks - 1
                # Word timestamps are monotonic, but proportional rounding could tie
                # / regress; clamp so the per-token assignment is non-decreasing.
                if chunk < prev_chunk:
                    chunk = prev_chunk
                prev_chunk = chunk
                for ti in group:
                    token_chunk_ids[b, ti] = chunk
                    counts[chunk] += 1

            # Left-packing feasibility: a chunk cannot host more tokens than frames.
            for c in range(n_chunks):
                frames_here = min(chunk_size, T_tr - c * chunk_size)
                if counts[c] > frames_here:
                    ok = False
                    break
            if not ok:
                valid_mask[b] = False
                token_chunk_ids[b, :] = -1

        return token_chunk_ids.to(out_device), valid_mask.to(out_device)
