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

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from nemo.utils import logging

__all__ = ['QwenWordForcedAligner', 'PrecomputedWordForcedAligner']


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


def map_word_starts_to_token_chunks(
    word_groups: List[List[int]],
    word_starts_sec: Sequence[float],
    audio_dur: float,
    T_tr: int,
    U_b: int,
    chunk_size: int,
    enforce_left_packing: bool = True,
    word_ends_sec: Optional[Sequence[float]] = None,
    anchor: str = 'start',
) -> Optional[dict]:
    """Map per-word timestamps (seconds) to a per-token chunk assignment.

    Shared by the live (:class:`QwenWordForcedAligner`) and the offline
    (:class:`PrecomputedWordForcedAligner`) backends so both produce identical
    chunk assignments.

    Args:
        word_groups: list of words, each a list of token indices (into the
            trainee label sequence) that belong to that word.
        word_starts_sec: per-word start time in seconds (one per group).
        audio_dur: utterance duration in seconds.
        T_tr: number of trainee encoder frames for this utterance.
        U_b: number of (real) trainee tokens.
        chunk_size: encoder frames per chunk.
        enforce_left_packing: when ``True`` (additive-joint Chunkwise-Aligner
            baseline) a chunk can host at most ``chunk_size`` tokens (tokens are
            left-packed onto frames) and ``T >= U`` is required. When ``False``
            (CHAT cross-attention baseline) a chunk pools all its frames into one
            representation and may host arbitrarily many tokens, so neither
            constraint applies -- only monotonicity / in-range bucketing.
        word_ends_sec: per-word END time in seconds (one per group). Required when
            ``anchor='end'``; ignored otherwise.
        anchor: which word timestamp anchors a word's tokens to a chunk.
            ``'start'`` uses the word onset; ``'end'`` uses the end of the word's
            last sub-word (so a token is only emitted once its audio has been
            heard -- a better match for an RNN-T/CHAT model's natural delayed
            emission). Falls back to ``'start'`` if end times are unavailable.

    Returns:
        ``{token_index: chunk}`` of length ``U_b``, or ``None`` if the assignment
        is infeasible (word-count mismatch; or, when ``enforce_left_packing``,
        ``T < U`` or a chunk would host more tokens than it has frames).
    """
    if len(word_starts_sec) != len(word_groups):
        return None
    use_end = str(anchor).lower() == 'end' and word_ends_sec is not None
    if use_end and len(word_ends_sec) != len(word_groups):
        return None
    anchor_times = word_ends_sec if use_end else word_starts_sec
    if enforce_left_packing and T_tr < U_b:
        return None
    audio_dur = max(float(audio_dur), 1e-6)
    n_chunks = (T_tr + chunk_size - 1) // chunk_size
    counts = [0] * n_chunks
    assignment: dict = {}
    prev_chunk = 0
    for wi, group in enumerate(word_groups):
        anchor_sec = float(anchor_times[wi] or 0.0)
        tr_frame = int((anchor_sec / audio_dur) * T_tr)
        tr_frame = min(max(tr_frame, 0), T_tr - 1)
        chunk = tr_frame // chunk_size
        if chunk >= n_chunks:
            chunk = n_chunks - 1
        # Word timestamps are monotonic, but proportional rounding could tie /
        # regress; clamp so the per-token assignment is non-decreasing.
        if chunk < prev_chunk:
            chunk = prev_chunk
        prev_chunk = chunk
        for ti in group:
            assignment[ti] = chunk
            counts[chunk] += 1
    # Left-packing feasibility: a chunk cannot host more tokens than frames
    # (additive-joint baseline only; CHAT cross-attention pools the whole chunk).
    if enforce_left_packing:
        for c in range(n_chunks):
            frames_here = min(chunk_size, T_tr - c * chunk_size)
            if counts[c] > frames_here:
                return None
    return assignment


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
        anchor: str = "end",
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
        self.anchor = str(anchor).lower()
        self._device = torch.device(device) if device is not None else None
        self._aligner = None  # lazily constructed on first use

    # ------------------------------------------------------------------ loading
    def _ensure_loaded(self, device: torch.device):
        if self._aligner is not None:
            return
        if self._device is None:
            self._device = device
        # Reuse the battle-tested speechlm2 wrapper rather than re-implementing the
        # Qwen call: it owns the correct ``from_pretrained`` kwargs, the
        # ``set_default_dtype(float32)`` safeguard (qwen_asr builds bf16 scalar
        # tensors numpy cannot convert), and a fast pre-resampled numpy path.
        try:
            from nemo.collections.speechlm2.modules.qwen_forced_aligner import QwenForcedAligner
        except Exception as e:  # pragma: no cover - depends on optional package
            raise ImportError(
                "The Qwen word-level aligner backend requires the `qwen-asr` package "
                "(provides `Qwen3ForcedAligner`) and `nemo.collections.speechlm2`. "
                "Install `qwen_asr` into the container image (see docker/Dockerfile.speech: "
                "`RUN pip install qwen_asr`) rather than at launch time. "
                f"Import failed: {e!r}"
            )
        device_map = str(self._device) if self._device.type == "cuda" else "cpu"
        dtype = self.dtype if self.dtype is not None else torch.bfloat16
        logging.info(
            f"[chunkwise-aligner] Loading frozen word-level aligner '{self.model_name_or_path}' "
            f"on {device_map} (dtype={dtype})."
        )
        self._aligner = QwenForcedAligner(
            pretrained_model=self.model_name_or_path,
            language=self.language,
            device=device_map,
            dtype=dtype,
        )

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
        """Call the external aligner. ``audios`` is a list of mono float32 numpy
        arrays (one per sample). Returns one result per sample; each result is a
        sequence of :class:`WordAlignment`-like items exposing ``.start_time``
        (seconds). Isolated so tests can stub it without the heavy model / package.
        """
        self._ensure_loaded(self._device or torch.device("cpu"))
        sr_target = getattr(self._aligner, "SAMPLE_RATE", 16000)
        if self.sample_rate == sr_target:
            # Fast path: audio is already at the aligner's sample rate.
            return self._aligner.align_numpy(audios, texts)
        # Otherwise hand to the resampling tensor path of the wrapper.
        import numpy as np

        lens = torch.tensor([int(np.asarray(a).shape[0]) for a in audios], dtype=torch.long)
        maxlen = int(lens.max()) if len(audios) else 0
        batch = torch.zeros(len(audios), maxlen, dtype=torch.float32)
        for i, a in enumerate(audios):
            a = np.asarray(a, dtype=np.float32)
            batch[i, : a.shape[0]] = torch.from_numpy(a)
        return self._aligner.align(batch, lens, texts, source_sample_rate=self.sample_rate)

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
        enforce_left_packing: bool = True,
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
            audios.append(wav)
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
            audio_dur = int(sig_len_cpu[b]) / float(self.sample_rate)
            starts = [float(getattr(w, 'start_time', 0.0) or 0.0) for w in result]
            ends = [float(getattr(w, 'end_time', 0.0) or 0.0) for w in result]

            assignment = map_word_starts_to_token_chunks(
                groups,
                starts,
                audio_dur,
                T_tr,
                U_b,
                chunk_size,
                enforce_left_packing=enforce_left_packing,
                word_ends_sec=ends,
                anchor=self.anchor,
            )
            if assignment is None:
                valid_mask[b] = False
                continue
            for ti, chunk in assignment.items():
                token_chunk_ids[b, ti] = chunk

        return token_chunk_ids.to(out_device), valid_mask.to(out_device)


def _file_id(path_or_id: str) -> str:
    """Normalize a manifest audio path (or id) to a stable key: basename sans ext."""
    base = os.path.basename(str(path_or_id))
    stem, _ = os.path.splitext(base)
    return stem


def _load_word_start_alignments(path: str) -> Dict[str, Tuple[List[float], Optional[List[float]]]]:
    """Load offline word alignments into ``{file_id: (starts_sec, ends_sec|None)}``.

    Accepts either:
      * a JSON dict ``{file_id_or_path: value}``, or
      * a JSON-lines file with one record per line,
    where ``value`` / each record is one of:
      * a list of floats (word start times in seconds; no ends),
      * ``{"starts": [float, ...], "ends": [float, ...]}`` (preferred -- carries
        both onset and end-of-last-sub-word times),
      * ``{"word_starts": [...], "word_ends": [...]}``, or
      * ``{"words": [{"start"|"start_time": float, "end"|"end_time": float}, ...]}``.
    Records may carry ``file_id`` / ``audio_filepath`` / ``audio_file`` as the key.
    """

    def _times_from_value(v) -> Optional[Tuple[List[float], Optional[List[float]]]]:
        if isinstance(v, list):
            if all(isinstance(x, (int, float)) for x in v):
                return [float(x) for x in v], None
            # list of word dicts
            starts, ends = [], []
            for w in v:
                if isinstance(w, dict):
                    starts.append(float(w.get('start', w.get('start_time', 0.0)) or 0.0))
                    ends.append(float(w.get('end', w.get('end_time', 0.0)) or 0.0))
            return starts, (ends if len(ends) == len(starts) and any(ends) else None)
        if isinstance(v, dict):
            if 'starts' in v or 'word_starts' in v:
                starts = [float(x) for x in (v.get('starts') or v.get('word_starts'))]
                raw_ends = v.get('ends') if v.get('ends') is not None else v.get('word_ends')
                ends = [float(x) for x in raw_ends] if raw_ends is not None else None
                if ends is not None and len(ends) != len(starts):
                    ends = None
                return starts, ends
            if 'words' in v:
                starts = [float(w.get('start', w.get('start_time', 0.0)) or 0.0) for w in v['words']]
                ends = [float(w.get('end', w.get('end_time', 0.0)) or 0.0) for w in v['words']]
                return starts, (ends if any(ends) else None)
        return None

    # A directory -> merge every *.json / *.jsonl inside (handy for sharded runs).
    if os.path.isdir(path):
        merged: Dict[str, Tuple[List[float], Optional[List[float]]]] = {}
        for fn in sorted(os.listdir(path)):
            if fn.endswith('.json') or fn.endswith('.jsonl'):
                merged.update(_load_word_start_alignments(os.path.join(path, fn)))
        return merged

    alignments: Dict[str, Tuple[List[float], Optional[List[float]]]] = {}
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = rec.get('file_id') or rec.get('audio_filepath') or rec.get('audio_file')
                if key is None:
                    continue
                times = _times_from_value(rec)
                if times is not None:
                    alignments[_file_id(key)] = times
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for key, v in data.items():
            times = _times_from_value(v)
            if times is not None:
                alignments[_file_id(key)] = times
    return alignments


class PrecomputedWordForcedAligner:
    """Offline counterpart of :class:`QwenWordForcedAligner` (Option B).

    The heavy Qwen forward is run **once, offline** (see
    ``scripts/asr_aligner/generate_qwen_word_alignments.py``) and the resulting
    per-word start times are stored keyed by ``file_id`` (audio basename sans
    extension). At train time this class only does the cheap word->frame->chunk
    mapping -- no model, no ``qwen_asr``/``transformers`` in the training process.

    The model resolves each batch row's ``file_id`` (from the dataset's
    ``sample_id``) and passes it in, so matching is content-stable across tarred
    shards, shuffling and DDP sharding.

    Args:
        tokenizer: the trainee tokenizer, used to group sub-words into words.
        alignments_path: path to the offline word-start file (JSON or JSONL).
        sample_rate: unused for mapping (durations are passed in seconds) but kept
            for API parity with the live aligner.
    """

    def __init__(self, tokenizer, alignments_path: str, sample_rate: int = 16000, anchor: str = "end"):
        if tokenizer is None:
            raise ValueError(
                "PrecomputedWordForcedAligner needs the trainee tokenizer to group sub-words into words."
            )
        if not alignments_path or not os.path.exists(alignments_path):
            raise FileNotFoundError(
                f"external_aligner.alignments_path='{alignments_path}' not found. Generate it first with "
                "scripts/asr_aligner/generate_qwen_word_alignments.py."
            )
        self.tokenizer = tokenizer
        self.sample_rate = int(sample_rate)
        self.anchor = str(anchor).lower()
        self.alignments_path = alignments_path
        self._alignments = _load_word_start_alignments(alignments_path)
        self._missing_logged = 0
        self._warned_missing_ends = False
        n_with_ends = sum(1 for _, ends in self._alignments.values() if ends is not None)
        logging.info(
            f"[chunkwise-aligner] Loaded {len(self._alignments)} precomputed word alignments "
            f"from '{alignments_path}' ({n_with_ends} with end times; anchor='{self.anchor}')."
        )
        if self.anchor == 'end' and n_with_ends == 0:
            logging.warning(
                "[chunkwise-aligner] anchor='end' but the precomputed alignments carry NO end times "
                "(old format = start times only). Falling back to word-start anchoring. Regenerate the "
                "alignments with scripts/asr_aligner/generate_qwen_word_alignments.py to store end times."
            )

    def to(self, *args, **kwargs):  # for model.to(device) compatibility
        return self

    def _pieces(self, ids: List[int]) -> List[str]:
        return [str(p) for p in self.tokenizer.ids_to_tokens(ids)]

    @torch.no_grad()
    def align_to_chunks(
        self,
        file_ids: Sequence[Optional[str]],
        labels: torch.Tensor,
        label_lens: torch.Tensor,
        target_frame_lengths: torch.Tensor,
        audio_durations: torch.Tensor,
        chunk_size: int,
        enforce_left_packing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bucket each token into a chunk from precomputed word start times.

        Args:
            file_ids: per-sample ``file_id`` (basename sans ext) or ``None`` when
                it could not be resolved (-> sample discarded).
            labels: ``[B, U]`` trainee token ids.
            label_lens: ``[B]`` real token counts.
            target_frame_lengths: ``[B]`` trainee encoder frames per sample.
            audio_durations: ``[B]`` utterance durations in seconds.
            chunk_size: encoder frames per chunk.

        Returns:
            ``(token_chunk_ids [B, U], valid_mask [B])`` -- same contract as the
            live word aligner.
        """
        B = int(labels.shape[0])
        U_max = int(labels.shape[1])
        out_device = labels.device

        labels_cpu = labels.to(torch.long).cpu()
        label_lens_cpu = label_lens.to(torch.long).cpu()
        target_frames_cpu = target_frame_lengths.to(torch.long).cpu()
        durations_cpu = audio_durations.to(torch.float32).cpu()

        token_chunk_ids = torch.full((B, U_max), -1, dtype=torch.long)
        valid_mask = torch.ones(B, dtype=torch.bool)

        for b in range(B):
            U_b = int(label_lens_cpu[b])
            key = file_ids[b] if b < len(file_ids) else None
            if U_b <= 0 or key is None:
                valid_mask[b] = False
                continue
            entry = self._alignments.get(_file_id(key))
            if entry is None:
                valid_mask[b] = False
                if self._missing_logged < 20:
                    logging.warning(f"[chunkwise-aligner] No precomputed alignment for file_id='{key}'; skipping.")
                    self._missing_logged += 1
                continue
            starts, ends = entry

            ids_b = labels_cpu[b, :U_b].tolist()
            groups = group_token_ids_into_words(self._pieces(ids_b))
            T_tr = int(target_frames_cpu[b])
            audio_dur = float(durations_cpu[b])

            assignment = map_word_starts_to_token_chunks(
                groups,
                starts,
                audio_dur,
                T_tr,
                U_b,
                chunk_size,
                enforce_left_packing=enforce_left_packing,
                word_ends_sec=ends,
                anchor=self.anchor,
            )
            if assignment is None:
                valid_mask[b] = False
                continue
            for ti, chunk in assignment.items():
                token_chunk_ids[b, ti] = chunk

        return token_chunk_ids.to(out_device), valid_mask.to(out_device)
