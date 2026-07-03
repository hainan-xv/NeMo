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

import copy
import logging
import math
import random
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from lhotse import CutSet
from lhotse.dataset.collation import collate_audio
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors
from nemo.collections.speechlm2.parts.alignments import WordAlignment, get_word_alignments_for_batch
from nemo.collections.speechlm2.parts.utils import to_dataclass

AUDIO_TOKEN_IDX = -200
IGNORE_INDEX = -100

# Sentinel for get_batch_data(truncation_plan=...): the default means "sample a
# fresh flush-truncation plan internally". Passing None (or a list) explicitly
# overrides that — used by encoder_reuse_k>1 to share ONE plan across the K
# reused views so they truncate the audio identically.
_TRUNCATION_AUTO = object()


def right_collate_vectors(
    tensors: Iterable[Union[torch.Tensor, np.ndarray]],
    padding_value: Union[int, float] = CrossEntropyLoss().ignore_index,
) -> torch.Tensor:
    tensors = [torch.as_tensor(t) for t in tensors]
    assert all(len(t.shape) == 1 for t in tensors), "Expected only 1-D input tensors."
    try:
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value, padding_side="right")
    except TypeError:
        # Backward compatibility for torch builds where `padding_side`
        # is not supported in torch.nn.utils.rnn.pad_sequence.
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value)


def build_chunk_word_alignment_debug(
    messages: List[dict],
    audio_tag: str,
    blank_token: str,
    tokenizer: AutoTokenizer,
) -> List[str]:
    """Build readable chunk→word assignment lines from LLM messages.

    Returns one line per user audio chunk, in chronological order:
    ``chunk 000 (frames=14): hello world``.
    """
    chunk_frames: list[int] = []
    chunk_texts: list[Optional[str]] = []
    aligned_texts: list[Optional[str]] = []
    original_texts: list[Optional[str]] = []
    pending_chunk_ids: list[int] = []

    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content", ""))

        if role == "user":
            n_frames = content.count(audio_tag) if audio_tag else 0
            chunk_frames.append(n_frames)
            chunk_texts.append(None)
            aligned_texts.append(None)
            original_texts.append(None)
            pending_chunk_ids.append(len(chunk_frames) - 1)
        elif role == "assistant":
            if pending_chunk_ids:
                chunk_idx = pending_chunk_ids.pop(0)
                chunk_texts[chunk_idx] = content
                if "aligned_text" in msg:
                    aligned_texts[chunk_idx] = str(msg.get("aligned_text", ""))
                if "original_text" in msg:
                    original_texts[chunk_idx] = str(msg.get("original_text", ""))

    debug_lines = []
    for idx, (n_frames, text) in enumerate(zip(chunk_frames, chunk_texts)):
        if text is None:
            text_repr = "<missing-assistant-turn>"
        elif blank_token and text == blank_token:
            text_repr = "<blank>"
        elif text == "":
            text_repr = "<empty>"
        else:
            text_repr = text
        debug_lines.append(f"chunk {idx:03d} (frames={n_frames}): {text_repr}")

        aligned = aligned_texts[idx]
        original = original_texts[idx]
        if aligned is not None and original is not None:
            aligned_norm = re.sub(r"\s+", " ", aligned).strip()
            original_norm = re.sub(r"\s+", " ", original).strip()
            if aligned_norm != original_norm:
                token_ids = tokenizer.tokenizer.encode(original, add_special_tokens=False)
                token_pieces = tokenizer.tokenizer.convert_ids_to_tokens(token_ids)
                token_pieces_repr = ", ".join(f'"{piece}"' for piece in token_pieces)
                debug_lines.append("    chunk | aligned text | original text | token ids | token pieces")
                debug_lines.append(
                    f"    {idx:03d} | {aligned_norm} | {original_norm} | {token_ids} | {token_pieces_repr}"
                )

    return debug_lines


@dataclass
class StreamingSTTBatch:
    """
    A batch of data for StreamingSTTModel.

    Attributes:
        audios: (B, T) audio signals.
        audio_lens: (B,) lengths of the audio signals in samples.
        input_tokens: (B, L) input token IDs for the LLM. Audio positions are marked with AUDIO_TOKEN_IDX.
        input_token_lens: (B,) lengths of the input token sequences.
        target_tokens: (B, L) target token IDs for the LLM. Non-trainable positions are IGNORE_INDEX.
        target_token_lens: (B,) lengths of the target token sequences.
        text: list of ground-truth transcription strings.
        chunk_word_alignment: optional per-sample chunk/word debug lines.
        cuts: Optional[CutSet] containing the cuts for the batch.
        chunk_anchor_positions: (B, max_chunks) int64. Position in input_tokens of
            each chunk's anchor (the <write> token in compact-template mode). Padded
            with -1 for samples with fewer chunks. Only populated when the model
            requests parallel-chunk-head supervision via compact_template=True.
        chunk_target_tokens: (B, max_chunks, K) int64. Per-chunk K-slot target slate
            for the parallel heads. Slot k holds the token to predict at slot k of
            the K parallel heads anchored at chunk_anchor_positions. For a chunk
            emitting N<K tokens, slots 0..N-1 hold the N tokens, slot N holds
            <|im_end|> (the chunk-closer), and slots N+1..K-1 hold IGNORE_INDEX.
            For N==K, all slots hold the N tokens (no closer slot). For N>K,
            all slots hold IGNORE_INDEX (skipped from loss). Padded chunks
            (anchor==-1) also hold all IGNORE_INDEX.
    """

    audios: Optional[torch.Tensor] = None
    audio_lens: Optional[torch.Tensor] = None
    input_tokens: Optional[torch.Tensor] = None
    input_token_lens: Optional[torch.Tensor] = None
    target_tokens: Optional[torch.Tensor] = None
    target_token_lens: Optional[torch.Tensor] = None
    text: Optional[List[str]] = None
    chunk_word_alignment: Optional[List[List[str]]] = None
    cuts: Optional[CutSet] = None
    chunk_anchor_positions: Optional[torch.Tensor] = None
    chunk_target_tokens: Optional[torch.Tensor] = None
    # (B, L) float per-target-token CE weights, aligned with ``target_tokens``.
    # Only populated when position-weighting is enabled (delay_weight_gamma != 1.0).
    # ``None`` => every supervised token has weight 1.0 (legacy uniform CE).
    target_weights: Optional[torch.Tensor] = None


@dataclass
class StreamingSTTDataConfig:
    sample_rate: int
    frame_length_in_secs: float
    chunk_size: int
    num_delay_frames: int = 0
    # --- Delay-randomized, position-weighted alignment objective ---
    # Port of the ASR ChatExternalAligner "delay-randomized, position-weighted"
    # training objective (nemo/collections/asr/losses/chunked_aligner_pytorch.py)
    # to the streaming SpeechLLM. Two independent, opt-in knobs:
    #   * random_delay_max_frames > 0: each word's emission is pushed LATER by an
    #     independent ``Uniform{0..random_delay_max_frames}`` frames (on top of the
    #     fixed ``num_delay_frames``), made monotonic non-decreasing so word order
    #     and chunk assignment stay valid. This perturbs which chunk each word is
    #     emitted in. Sampled fresh on every ``get_batch_data`` call, so combined
    #     with the model's ``encoder_reuse_k`` (encode once, average K partitions)
    #     it realizes the "average over K randomly delayed alignments" objective.
    #     Applied in TRAINING only (eval stays deterministic). 0 => disabled.
    #   * delay_weight_gamma in (0, 1]: each supervised token's CE term is scaled
    #     by ``gamma ** p`` where ``p`` is the within-chunk frame position at which
    #     the word's audio finished (0 = chunk start). A word that finishes late in
    #     its emission chunk is down-weighted; a word bumped to a later chunk than
    #     its audio (by delay) is treated as available from frame 0 => weight 1.0.
    #     1.0 => uniform weights (exact legacy CE). Only the compact template
    #     emits per-token weights; the standard chat template stays uniform.
    # Currently effective only on the fixed-chunk, non-projected emission path.
    random_delay_max_frames: int = 0
    delay_weight_gamma: float = 1.0
    words_per_group: int = 1
    # Trainable encoder-output subsampling factor (see the model config). With
    # factor > 1 the LLM consumes ``chunk_size // factor`` audio placeholders per
    # chunk while the alignment/timeline math stays in ``chunk_size`` encoder
    # frames. Requires fixed chunking and chunk_size % factor == 0. Default 1 =
    # no subsampling (one audio token per encoder frame, unchanged behavior).
    encoder_subsampling_factor: int = 1
    audio_tag: str = "<audio>"
    blank_token: str = "<blank>"
    system_role: str = "system"
    system_prompt: str = "Transcribe the audio into text."
    prompt_field: str = "system_prompt"
    compact_template: bool = False
    write_token: str = "<|im_start|>"
    supervise_im_end_in_loss: bool = False
    project_unaligned_text_to_chunks: bool = False
    max_audio_chunks_per_turn: int = 1
    # Optional discrete set of chunks-per-turn group sizes for fixed-chunk
    # multi-chunk training. Comma-separated string (e.g. "1,2,7") or list of
    # ints. When set, each turn's group size is sampled uniformly from EXACTLY
    # these values instead of uniformly over [1, max_audio_chunks_per_turn].
    # None (default) keeps the legacy uniform-range behavior. Propagated from
    # the model's audio_chunks_per_turn_choices flag.
    audio_chunks_per_turn_choices: Optional[str] = None
    # K — only effective in dynamic chunking (chunk_size == 0). Each audio
    # segment is rounded UP to a multiple of K frames (and total audio is
    # padded to K-multiple). The model implicitly learns to emit only at
    # K-aligned positions; deploy-time K' (any multiple of K_train) is set via
    # dynamic_min_chunk_size / dynamic_max_chunk_size. Default 1 = no-op.
    chunk_step: int = 1
    # --- Parallel chunk heads (multi-token-per-chunk prediction) ---
    # When > 0, the dataset emits ``chunk_anchor_positions`` and
    # ``chunk_target_tokens`` in each batch so the model can train K parallel
    # next-token heads. K is the per-forward block size: a chunk's emit-stream
    # (content + <|im_end|>) is split into ceil(S/K) K-token blocks, each
    # anchored at write_id + g*K. Chunks longer than K are NOT skipped — they
    # are supervised across multiple blocks. Only supported with
    # compact_template=True (anchor = write_id).
    parallel_chunk_slots: int = 0
    # When True, audio chunks with no text emit a bare <|im_end|> (chunk-end)
    # instead of the explicit <blank> token: the per-chunk emit-stream becomes
    # ``write_id -> <eos>`` rather than ``write_id -> <blank> -> <eos>``. Only
    # effective with compact_template=True. Default False keeps the legacy
    # <blank> scheme. Propagated from the model's empty_chunk_eos_only flag.
    empty_chunk_eos_only: bool = False
    # --- Blank-continuation parallel scheme (strict <|im_end|> placement) ---
    # When True, ``_build_parallel_chunk_targets`` emits variable-length blocks:
    # <|im_end|> only ever appears in slot 0 of a terminator block, partial/cut
    # blocks end with a single <blank> "continue" marker, and anchors advance by
    # the number of *real* content tokens consumed (blanks are synthetic head
    # targets, not sequence positions). Propagated from the model's
    # parallel_blank_continuation flag.
    parallel_blank_continuation: bool = False
    # Per-block "cut" probability for the blank-continuation augmentation (see
    # the model config). 0.0 disables it. Propagated from the model.
    parallel_cut_prob: float = 0.0
    # --- Flush token (explicit end-of-audio "dump the tail" control signal) ---
    # When True, a ``flush_token`` user turn is ALWAYS appended after the last
    # audio chunk; words deferred (by delay) past the last real chunk are emitted
    # by the assistant AFTER the flush, otherwise the post-flush assistant turn is
    # empty and emits only the end-of-utterance <|im_end|>. Propagated from the
    # model's use_flush / flush_token. flush_truncate_prob > 0 adds a training-only
    # augmentation that truncates a cut at a random chunk boundary (so the model
    # learns to flush mid-utterance); the cutoff is shared per batch and bounded
    # by the shortest cut. Effective on the fixed-chunk path only (chunk_size > 0).
    use_flush: bool = False
    flush_token: str = "<flush>"
    flush_truncate_prob: float = 0.0


def decode_with_blank(
    ids: list[int],
    blank_token: str,
    tokenizer: AutoTokenizer,
    replace_blank: Optional[str] = None,
    strip_whitespace: bool = False,
    collapse_whitespace: bool = True,
    join_with: Optional[str] = " ",
    write_token: Optional[str] = None,
) -> str:
    """Decode token IDs, treating blank tokens as segment boundaries.

    Splits the token sequence at ``blank_token`` boundaries, decodes each
    segment separately (preserving BPE within each turn), then joins with
    spaces.

    Args:
        ids: Token IDs to decode.
        blank_token: The blank token string (e.g., ``"<blank>"``).
        tokenizer: NeMo AutoTokenizer.
        replace_blank: If provided, blank tokens are replaced with this string
            in the output instead of being skipped.  For example,
            ``replace_blank=""`` keeps the spacing, ``replace_blank="..."``
            inserts an ellipsis.
        strip_whitespace: If True, strip whitespace from the output.
        collapse_whitespace: If True, collapse multiple consecutive whitespace characters into a single space.
        join_with: If provided, join the segments divided by blank tokens with this string, else join with empty string.
    """
    if blank_token == "":
        # No blank token: use EOS (e.g. <|im_end|>) as chunk separator so
        # per-chunk outputs get joined with spaces instead of BPE-merged into one run.
        blank_id = tokenizer.tokenizer.eos_token_id
    else:
        blank_id = tokenizer.tokenizer.convert_tokens_to_ids(blank_token)
    write_id = None
    if write_token is not None:
        write_id = tokenizer.tokenizer.convert_tokens_to_ids(write_token)

    segments = []
    current = []
    for tid in ids:
        if tid == blank_id:
            if current:
                segments.append(tokenizer.ids_to_tokens(current))
                current = []
            if replace_blank is not None:
                segments.append(replace_blank)
        elif tid == write_id:
            continue
        else:
            current.append(tid)
    if current:
        segments.append(tokenizer.ids_to_tokens(current))

    text_segments = []
    # Some NeMo tokenizer wrappers (notably older HF AutoTokenizer wrappers)
    # don't accept ``remove_special_tokens`` on ``tokens_to_text``. Try the
    # kwarg path first, fall back to filtering special tokens manually so
    # this stays robust across NeMo versions installed on the eval side.
    for seg in segments:
        if isinstance(seg, str):
            text_segments.append(seg)
            continue
        try:
            text_segments.append(tokenizer.tokens_to_text(seg, remove_special_tokens=True))
        except TypeError:
            hf_tok = getattr(tokenizer, "tokenizer", tokenizer)
            special = set(getattr(hf_tok, "all_special_tokens", []) or [])
            seg_clean = [t for t in seg if t not in special] if special else seg
            text_segments.append(tokenizer.tokens_to_text(seg_clean))
    text = join_with.join(text_segments) if join_with else "".join(text_segments)

    if strip_whitespace:
        text = text.strip()
    if collapse_whitespace:
        text = re.sub(r'\s+', ' ', text)
    return text


def compute_word_spans(
    alignments: List[WordAlignment],
    transcript: str,
    preserve_trailing_whitespace: bool = False,
    preserve_leading_whitespace: bool = False,
) -> List[tuple[int, int]]:
    """Find (start, end) character positions for each alignment word in the transcript.

    Trailing punctuation (non-alphanumeric, non-whitespace characters) that
    immediately follows a word is always included in the span so that commas,
    periods, quotes, etc. are preserved.

    Args:
        alignments: Word-level alignment results.
        transcript: Original transcription string.
        preserve_trailing_whitespace: When True, each span extends through
            trailing whitespace up to (but not including) the next alphanumeric
            character.  This is useful when extracting multi-word spans so
            that ``transcript[first_span[0]:last_span[1]]`` includes the
            inter-word spaces.
        preserve_leading_whitespace: When True, each span extends backward
            through preceding whitespace (not crossing the previous word's
            span end).  This matches GPT-style BPE tokenization where a
            leading space is part of the word token (e.g. ``" world"`` vs
            ``"world"``).  For ``"hello world"`` this yields
            ``[(0,5), (5,11)]`` = ``"hello"``, ``" world"``.

    Returns a list parallel to *alignments*.  If a word cannot be located, its
    span is ``None``.
    """

    if preserve_trailing_whitespace and preserve_leading_whitespace:
        raise ValueError(
            "preserve_trailing_whitespace and preserve_leading_whitespace cannot be True at the same time"
        )
    spans: List[tuple[int, int] | None] = []
    search_pos = 0
    for word in alignments:
        idx = transcript.lower().find(word.text.lower(), search_pos)
        if idx == -1:
            spans.append(None)
            continue
        start = idx
        # Optionally extend start backward through leading whitespace,
        # clamped at the previous word's span end.
        if preserve_leading_whitespace:
            while start > search_pos and transcript[start - 1].isspace():
                start -= 1
        end = idx + len(word.text)
        # Include trailing punctuation (e.g., comma, period, quotes)
        while end < len(transcript) and not transcript[end].isalnum() and not transcript[end].isspace():
            end += 1
        # Optionally include trailing whitespace up to the next word
        if preserve_trailing_whitespace:
            while end < len(transcript) and transcript[end].isspace():
                end += 1
        spans.append((start, end))
        search_pos = end
    return spans


def compute_alignment_spans(
    alignments: List[WordAlignment],
    transcript: str,
) -> List[tuple[int, int] | None]:
    """Find exact alignment-word spans in the original transcript.

    Unlike ``compute_word_spans``, this returns only the matched alignment word
    span. Text between matched spans is handled separately as an unaligned gap.
    """
    spans: List[tuple[int, int] | None] = []
    search_pos = 0
    transcript_lower = transcript.lower()
    for word in alignments:
        needle = word.text
        if not needle:
            spans.append(None)
            continue
        idx = transcript_lower.find(needle.lower(), search_pos)
        if idx == -1:
            spans.append(None)
            continue
        end = idx + len(needle)
        spans.append((idx, end))
        search_pos = end
    return spans


def assign_fixed_chunk_ids(
    alignments: List[WordAlignment],
    num_chunks: int,
    chunk_size: int,
    frame_length_in_secs: float,
    num_delay_frames: int,
    words_per_group: int,
) -> List[Optional[int]]:
    """Return the fixed-chunk assistant turn index for each alignment word."""
    chunk_ids: List[Optional[int]] = [None] * len(alignments)
    if num_chunks <= 0:
        return chunk_ids

    word_idx = 0
    word_buffer: list[int] = []
    for chunk_i in range(num_chunks):
        chunk_end_frame = (chunk_i + 1) * chunk_size
        while word_idx < len(alignments):
            word = alignments[word_idx]
            word_end_frame = math.ceil(word.end_time / frame_length_in_secs)
            ready_frame = word_end_frame + num_delay_frames
            if ready_frame <= chunk_end_frame:
                word_buffer.append(word_idx)
                word_idx += 1
            else:
                break

        is_last_chunk = chunk_i == num_chunks - 1
        if word_buffer and (len(word_buffer) >= words_per_group or is_last_chunk):
            for idx in word_buffer:
                chunk_ids[idx] = chunk_i
            word_buffer = []

    # Preserve existing behavior: words delayed past the final chunk are emitted
    # in the final assistant turn.
    for idx in range(word_idx, len(alignments)):
        chunk_ids[idx] = num_chunks - 1

    return chunk_ids


def project_transcript_to_chunks(
    alignments: List[WordAlignment],
    transcript: str,
    alignment_chunk_ids: List[Optional[int]],
    num_chunks: int,
) -> tuple[List[str], List[str]]:
    """Project original transcript text onto alignment-derived chunk IDs.

    Alignment words provide timing. The original transcript provides training
    text. Any original-text gap between two matched alignment anchors is assigned
    to their shared chunk, or to the later chunk when the neighbors differ.
    """
    original_chunks = [""] * num_chunks
    aligned_chunks: list[list[str]] = [[] for _ in range(num_chunks)]
    if num_chunks <= 0:
        return original_chunks, []

    for word, chunk_id in zip(alignments, alignment_chunk_ids):
        if chunk_id is not None and 0 <= chunk_id < num_chunks:
            aligned_chunks[chunk_id].append(word.text)

    spans = compute_alignment_spans(alignments, transcript)
    matched = [
        (idx, span, alignment_chunk_ids[idx])
        for idx, span in enumerate(spans)
        if span is not None and alignment_chunk_ids[idx] is not None and 0 <= alignment_chunk_ids[idx] < num_chunks
    ]

    if not matched:
        return [" ".join(words) for words in aligned_chunks], [" ".join(words) for words in aligned_chunks]

    prev_end = 0
    prev_chunk: Optional[int] = None
    for _, (start, end), chunk_id in matched:
        gap = transcript[prev_end:start]
        if gap:
            gap_chunk = chunk_id if prev_chunk is None or prev_chunk != chunk_id else prev_chunk
            original_chunks[gap_chunk] += gap
        original_chunks[chunk_id] += transcript[start:end]
        prev_end = end
        prev_chunk = chunk_id

    if prev_chunk is not None and prev_end < len(transcript):
        original_chunks[prev_chunk] += transcript[prev_end:]

    # If matching failed for a chunk, keep its aligned text rather than dropping it.
    aligned_text_chunks = [" ".join(words) for words in aligned_chunks]
    for idx, (original, aligned) in enumerate(zip(original_chunks, aligned_text_chunks)):
        if not original.strip() and aligned.strip():
            original_chunks[idx] = aligned

    return original_chunks, aligned_text_chunks


def parse_chunk_group_choices(choices) -> Optional[List[int]]:
    """Parse a discrete chunk-group-size spec into a sorted list of unique ints.

    Accepts a comma-separated string (e.g. ``"1,2,7"``), an iterable of ints, or
    None/empty. Returns a de-duplicated, ascending list of positive ints, or
    None when nothing valid is given (caller then falls back to the uniform
    ``[1, max_chunks_per_turn]`` sampling).
    """
    if choices is None:
        return None
    if isinstance(choices, str):
        tokens = [tok.strip() for tok in choices.split(",")]
    else:
        # Iterable of values (list/tuple/OmegaConf ListConfig).
        tokens = [str(tok).strip() for tok in choices]
    values: List[int] = []
    for tok in tokens:
        if tok == "":
            continue
        v = int(tok)
        if v < 1:
            raise ValueError(f"chunk-group choice must be a positive int, got {v} (from {choices!r})")
        values.append(v)
    if not values:
        return None
    return sorted(set(values))


def sample_fixed_chunk_group_schedule(
    num_chunks: int,
    max_chunks_per_turn: int,
    allowed_group_sizes: Optional[List[int]] = None,
) -> List[int]:
    """Sample batch-shared per-position group sizes for fixed chunking.

    When ``allowed_group_sizes`` is given (a non-empty list of positive ints),
    each group size is drawn uniformly from exactly those discrete values
    (e.g. ``[1, 2, 7]`` => only 1-, 2-, or 7-chunk turns). Otherwise the legacy
    behavior applies: group sizes are drawn uniformly from ``[1, max_chunks_per_turn]``.
    """
    if num_chunks <= 0:
        return []

    if allowed_group_sizes:
        sizes = [s for s in allowed_group_sizes if s >= 1]
        if not sizes:
            return [1] * num_chunks
        if len(sizes) == 1:
            # Single allowed size: deterministic tiling (still randomized batch
            # to batch only in the trivial sense; nothing to sample).
            only = int(sizes[0])
            schedule: List[int] = []
            chunks_consumed = 0
            while chunks_consumed < num_chunks:
                schedule.append(only)
                chunks_consumed += only
            return schedule
        schedule = []
        chunks_consumed = 0
        while chunks_consumed < num_chunks:
            pick = int(torch.randint(0, len(sizes), (1,)).item())
            group_size = int(sizes[pick])
            schedule.append(group_size)
            chunks_consumed += group_size
        return schedule

    max_chunks_per_turn = max(int(max_chunks_per_turn), 1)
    if max_chunks_per_turn == 1:
        return [1] * num_chunks

    schedule = []
    chunks_consumed = 0
    while chunks_consumed < num_chunks:
        group_size = int(torch.randint(1, max_chunks_per_turn + 1, (1,)).item())
        schedule.append(group_size)
        chunks_consumed += group_size
    return schedule


def iter_fixed_chunk_groups(num_chunks: int, schedule: Optional[List[int]]) -> Iterable[tuple[int, int, int]]:
    """Yield ``(start_chunk, end_chunk, scheduled_group_size)`` for one sample."""
    start = 0
    schedule = schedule or [1] * num_chunks
    for group_size in schedule:
        if start >= num_chunks:
            break
        group_size = max(int(group_size), 1)
        end = min(start + group_size, num_chunks)
        yield start, end, group_size
        start = end

    # If the sample is longer than the batch schedule due to an edge case,
    # fall back to one chunk per turn rather than dropping audio.
    while start < num_chunks:
        end = start + 1
        yield start, end, 1
        start = end


def get_llm_messages_for_sample(
    system_role: str,
    system_prompt: str,
    audio_tag: str,
    blank_token: str,
    chunk_size: int,
    num_delay_frames: int,
    audio_duration_secs: float,
    frame_length_in_secs: float,
    alignments: Optional[List[WordAlignment]] = None,
    transcript: Optional[str] = None,
    words_per_group: int = 1,
    chunk_step: int = 1,
    project_unaligned_text_to_chunks: bool = False,
    fixed_chunk_group_schedule: Optional[List[int]] = None,
    subsampling_factor: int = 1,
    random_delay_max_frames: int = 0,
    delay_weight_gamma: float = 1.0,
    apply_random_delay: bool = True,
    use_flush: bool = False,
    flush_token: str = "<flush>",
) -> List[dict]:
    """
    Get the LLM messages for a sample, using the alignments to determine the turns for the audio and text.

    The conversation is structured as alternating user (audio chunks) and assistant (transcription or blank) turns.
    A word becomes "ready" at the chunk whose end frame >= word_end_frame + num_delay_frames.

    For example, if the alignments are:
    [
        WordAlignment(text="Hello", start_time=0.16, end_time=0.48),
        WordAlignment(text="World", start_time=0.60, end_time=0.80),
    ]
    And the audio duration is 1s, audio_tag is "<audio>", chunk_size is 2, frame_length_in_secs is 0.08s,
    num_delay_frames is 0, then the messages will be:
    [
        {"role": "system", "content": "Transcribe the audio into text."},
        {"role": "user", "content": "<audio><audio>"},  # frames 0-1, 0~0.16s
        {"role": "assistant", "content": "<blank>"},
        {"role": "user", "content": "<audio><audio>"},  # frames 2-3, 0.16~0.32s
        {"role": "assistant", "content": "<blank>"},
        {"role": "user", "content": "<audio><audio>"},  # frames 4-5, 0.32~0.48s
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "<audio><audio>"},  # frames 6-7, 0.48~0.64s
        {"role": "assistant", "content": "<blank>"},
        {"role": "user", "content": "<audio><audio>"},  # frames 8-9, 0.64~0.80s
        {"role": "assistant", "content": "World"},
        {"role": "user", "content": "<audio><audio>"},  # frames 10-11, 0.80~0.96s
        {"role": "assistant", "content": "<blank>"},
        {"role": "user", "content": "<audio><audio>"},  # frames 12-13, 0.96~1.12s
        {"role": "assistant", "content": "<blank>"},
    ]

    Note: the last chunk may extend beyond audio_duration_secs since num_frames is
    ceiled to a multiple of chunk_size. The model must pad the audio accordingly.

    Args:
        system_role: The role of the system.
        system_prompt: The prompt for the system.
        audio_tag: The tag for the audio placeholder.
        blank_token: The token for blank/no-emission.
        chunk_size: The number of frames per chunk. If -1, the whole audio is used as a single chunk.
        num_delay_frames: Number of frames to delay word emission after word end.
        audio_duration_secs: The duration of the audio in seconds.
        frame_length_in_secs: The length of a single frame in seconds.
        alignments: List of WordAlignment objects for the sample.
    """

    messages = [{"role": system_role, "content": system_prompt}]

    # Encoder subsampling: the model shrinks the encoder output by this factor,
    # so each chunk of ``F`` encoder frames is represented by ``F // S`` audio
    # placeholders in the LLM sequence. ``audio_tokens(F)`` converts a frame
    # count into the placeholder count. S==1 -> identity (unchanged behavior).
    S = max(int(subsampling_factor), 1)

    def audio_tokens(num_audio_frames: int) -> int:
        return num_audio_frames // S

    num_frames = math.ceil(audio_duration_secs / frame_length_in_secs)

    if chunk_size < 0 or chunk_size is None:
        # Offline mode: use the whole audio as a single chunk
        num_chunks = 1 if num_frames > 0 else 0
        chunk_size = num_frames
        offline_mode = True
        num_delay_frames = 0  # delay is not used in offline mode
    else:
        offline_mode = False

    if alignments is None:
        alignments = []

    if offline_mode and not alignments:
        messages.append({"role": "user", "content": audio_tag * audio_tokens(num_frames)})
        messages.append({"role": "assistant", "content": transcript if transcript is not None else blank_token})
        return messages

    # Default behavior preserves the original span-slicing path.  The
    # projection path below is opt-in via project_unaligned_text_to_chunks.
    word_spans = (
        compute_word_spans(alignments, transcript, preserve_leading_whitespace=True)
        if transcript and (chunk_size == 0 or not project_unaligned_text_to_chunks)
        else None
    )

    if chunk_size == 0:
        # Dynamic chunking: one user turn per word group, sized to word boundary.
        # The model learns to predict when to stop listening via audio-position targets.
        # When chunk_step > 1, each segment's frame count is rounded UP to a
        # multiple of K so the model only ever emits at K-aligned positions.
        K = max(int(chunk_step), 1)
        prev_end_frame = 0
        word_buffer: list[int] = []  # indices of buffered words

        for word_idx, word in enumerate(alignments):
            word_buffer.append(word_idx)

            # Emit when buffer reaches words_per_group or this is the last word
            if len(word_buffer) < words_per_group and word_idx < len(alignments) - 1:
                continue

            # Chunk boundary = end frame of the last word in this group, snapped
            # UP to the next multiple of K. num_frames here is already K-padded
            # (caller guarantees this), so the clamp keeps things K-aligned.
            last_word = alignments[word_buffer[-1]]
            group_end_frame = math.ceil(last_word.end_time / frame_length_in_secs) + num_delay_frames
            if K > 1:
                group_end_frame = ((group_end_frame + K - 1) // K) * K
            group_end_frame = min(group_end_frame, num_frames)
            n_frames_chunk = group_end_frame - prev_end_frame

            if n_frames_chunk > 0:
                messages.append({"role": "user", "content": audio_tag * audio_tokens(n_frames_chunk)})

            # Build assistant content from all buffered words
            if word_spans and transcript:
                first_span = word_spans[word_buffer[0]]
                last_span = word_spans[word_buffer[-1]]
                if first_span is not None and last_span is not None:
                    content = transcript[first_span[0] : last_span[1]]
                else:
                    content = " ".join(alignments[i].text for i in word_buffer)
            else:
                content = " ".join(alignments[i].text for i in word_buffer)

            if n_frames_chunk <= 0 and messages[-1]["role"] == "assistant":
                # Words at same boundary as previous group — append
                messages[-1]["content"] += " " + content
            else:
                messages.append({"role": "assistant", "content": content})

            prev_end_frame = group_end_frame
            word_buffer = []

        # Trailing silence frames (after last word) — user turn only, no assistant.
        if prev_end_frame < num_frames:
            messages.append({"role": "user", "content": audio_tag * audio_tokens(num_frames - prev_end_frame)})
    else:
        # Fixed chunking: split the audio into equal-sized chunks.
        num_chunks = math.ceil(num_frames / chunk_size) if num_frames > 0 else 0
        # Audio placeholders per chunk after encoder subsampling: ceil(chunk_size/S).
        # When chunk_size % S != 0 the model pads each chunk's tail (repeating its
        # last frame) up to a multiple of S so it still yields this many sub-frames;
        # the dataset and model MUST agree on this count to stay aligned.
        slots_per_chunk = (chunk_size + S - 1) // S
        if project_unaligned_text_to_chunks:
            alignment_chunk_ids = assign_fixed_chunk_ids(
                alignments=alignments,
                num_chunks=num_chunks,
                chunk_size=chunk_size,
                frame_length_in_secs=frame_length_in_secs,
                num_delay_frames=num_delay_frames,
                words_per_group=words_per_group,
            )
            if transcript:
                original_chunk_texts, aligned_chunk_texts = project_transcript_to_chunks(
                    alignments=alignments,
                    transcript=transcript,
                    alignment_chunk_ids=alignment_chunk_ids,
                    num_chunks=num_chunks,
                )
            else:
                aligned_by_chunk: list[list[str]] = [[] for _ in range(num_chunks)]
                for word, chunk_id in zip(alignments, alignment_chunk_ids):
                    if chunk_id is not None and 0 <= chunk_id < num_chunks:
                        aligned_by_chunk[chunk_id].append(word.text)
                aligned_chunk_texts = [" ".join(words) for words in aligned_by_chunk]
                original_chunk_texts = aligned_chunk_texts

            for start_chunk, end_chunk, group_size in iter_fixed_chunk_groups(
                num_chunks, fixed_chunk_group_schedule
            ):
                messages.append({"role": "user", "content": audio_tag * (group_size * slots_per_chunk)})
                if transcript:
                    content = "".join(original_chunk_texts[start_chunk:end_chunk])
                else:
                    content = " ".join(
                        text.strip() for text in original_chunk_texts[start_chunk:end_chunk] if text.strip()
                    )
                aligned_text = " ".join(
                    text.strip() for text in aligned_chunk_texts[start_chunk:end_chunk] if text.strip()
                )
                if content.strip():
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content,
                            "aligned_text": aligned_text,
                            "original_text": content,
                        }
                    )
                else:
                    messages.append({"role": "assistant", "content": blank_token})
            if use_flush:
                # Explicit end-of-audio flush turn. The projected path clamps all
                # words into valid chunks (no residual), so the post-flush
                # assistant turn is empty and emits only the EOU <|im_end|>.
                messages.append({"role": "user", "content": flush_token})
                messages.append({"role": "assistant", "content": ""})
        else:
            # Precompute each word's emission "ready" frame and its position
            # weight ONCE (see StreamingSTTDataConfig.random_delay_max_frames /
            # delay_weight_gamma).
            #   ready = word_end + num_delay_frames (+ Uniform{0..max} in training),
            #   made monotonic non-decreasing (running max) so word order and the
            #   chunk assignment below stay valid.
            #   weight = gamma ** p, where p is the LOOKAHEAD DEFICIT: chunk_size minus
            #   the number of REAL audio frames the model hears between the word's audio
            #   end (`wef`) and where it commits the token (end of the emission chunk),
            #   measured against the true audio length `num_frames` (NOT the padded chunk
            #   boundary). >=1 full chunk of real lookahead => p<=0 => weight 1.0; ~0 real
            #   lookahead => p=chunk_size-1 => max down-weight. Delaying a word into a
            #   later REAL chunk adds genuine lookahead; delaying it onto a padded tail /
            #   phantom chunk adds none, so tail words stay down-weighted.
            use_delay = bool(apply_random_delay) and int(random_delay_max_frames) > 0
            use_weight = float(delay_weight_gamma) != 1.0
            word_ready_frames: list[int] = []
            word_weights: list[float] = []
            running_ready = 0
            for w in alignments:
                wef = math.ceil(w.end_time / frame_length_in_secs)
                ready = wef + num_delay_frames
                if use_delay:
                    ready += random.randint(0, int(random_delay_max_frames))
                ready = max(ready, running_ready)  # monotonic non-decreasing
                running_ready = ready
                word_ready_frames.append(ready)
                if use_weight and chunk_size > 0:
                    emit_chunk_idx = max(0, math.ceil(ready / chunk_size) - 1)
                    # Effective right context = REAL audio frames between the word's audio
                    # end (`wef`) and where the token is committed (end of the emission
                    # chunk), bounded by the true audio length `num_frames`. Frames beyond
                    # `num_frames` are padding/flush and provide no real lookahead, so a
                    # word pushed onto a padded tail / phantom chunk gains nothing and stays
                    # down-weighted, while a word delayed into a later REAL chunk gets its
                    # genuine extra lookahead. (For interior chunks this reduces exactly to
                    # p = wef - emit_chunk_idx * chunk_size.)
                    emission_end_frame = (emit_chunk_idx + 1) * chunk_size
                    real_lookahead = min(emission_end_frame, num_frames) - wef
                    p = min(max(chunk_size - real_lookahead, 0), chunk_size - 1)
                    word_weights.append(float(delay_weight_gamma) ** p)
                else:
                    word_weights.append(1.0)

            word_idx = 0
            word_buffer: list[int] = []
            for start_chunk, end_chunk, group_size in iter_fixed_chunk_groups(
                num_chunks, fixed_chunk_group_schedule
            ):
                chunk_end_frame = end_chunk * chunk_size
                messages.append({"role": "user", "content": audio_tag * (group_size * slots_per_chunk)})

                while word_idx < len(alignments):
                    if word_ready_frames[word_idx] <= chunk_end_frame:
                        word_buffer.append(word_idx)
                        word_idx += 1
                    else:
                        break

                is_last_chunk = end_chunk == num_chunks
                if word_buffer and (len(word_buffer) >= words_per_group or is_last_chunk):
                    if word_spans and transcript:
                        first_span = word_spans[word_buffer[0]]
                        last_span = word_spans[word_buffer[-1]]
                        if first_span is not None and last_span is not None:
                            content = transcript[first_span[0] : last_span[1]]
                        else:
                            content = " ".join(alignments[i].text for i in word_buffer)
                    else:
                        content = " ".join(alignments[i].text for i in word_buffer)
                    asst_msg = {"role": "assistant", "content": content}
                    if use_weight:
                        # Per-turn weight: the LAST emitted word gates the chunk
                        # boundary (mirrors the ASR EOC arc taking the last token's
                        # weight). With words_per_group=1 this is exact.
                        asst_msg["weight"] = word_weights[word_buffer[-1]]
                    messages.append(asst_msg)
                    word_buffer = []
                else:
                    messages.append({"role": "assistant", "content": blank_token})

            residual_content: Optional[str] = None
            residual_weight = None
            if word_idx < len(alignments):
                residual_indices = list(range(word_idx, len(alignments)))
                if word_spans and transcript:
                    first_span = word_spans[residual_indices[0]]
                    last_span = word_spans[residual_indices[-1]]
                    if first_span is not None and last_span is not None:
                        residual_content = transcript[first_span[0] : last_span[1]]
                    else:
                        residual_content = " ".join(alignments[i].text for i in residual_indices)
                else:
                    residual_content = " ".join(alignments[i].text for i in residual_indices)
                residual_weight = word_weights[residual_indices[-1]] if use_weight else None

            if use_flush:
                # Explicit end-of-audio flush: feed <flush> as a final user turn,
                # then emit any residual (delayed past the last chunk) words. With
                # no residual the post-flush assistant turn is empty and emits only
                # the EOU <|im_end|>. This replaces the legacy "merge residual into
                # the last chunk" behavior so the tail is trained behind <flush>.
                messages.append({"role": "user", "content": flush_token})
                asst_msg = {"role": "assistant", "content": residual_content if residual_content is not None else ""}
                if residual_content is not None and residual_weight is not None:
                    asst_msg["weight"] = residual_weight
                messages.append(asst_msg)
            elif residual_content is not None:
                if messages[-1]["role"] == "assistant" and messages[-1]["content"] == blank_token:
                    messages[-1]["content"] = residual_content
                    if residual_weight is not None:
                        messages[-1]["weight"] = residual_weight
                elif messages[-1]["role"] == "assistant":
                    messages[-1]["content"] += " " + residual_content
                    if residual_weight is not None:
                        messages[-1]["weight"] = residual_weight
                else:
                    asst_msg = {"role": "assistant", "content": residual_content}
                    if residual_weight is not None:
                        asst_msg["weight"] = residual_weight
                    messages.append(asst_msg)

    return messages


def get_llm_messages_for_batch(
    system_role: str,
    system_prompt: List[str],
    audio_tag: str,
    blank_token: str,
    chunk_size: int,
    num_delay_frames: int,
    audio_durations_secs: List[float],
    frame_length_in_secs: float,
    alignments: Optional[List[List[WordAlignment]]] = None,
    transcripts: Optional[List[str]] = None,
    words_per_group: int = 1,
    chunk_step: int = 1,
    project_unaligned_text_to_chunks: bool = False,
    fixed_chunk_group_schedule: Optional[List[int]] = None,
    subsampling_factor: int = 1,
    random_delay_max_frames: int = 0,
    delay_weight_gamma: float = 1.0,
    apply_random_delay: bool = True,
    use_flush: bool = False,
    flush_token: str = "<flush>",
) -> List[List[dict]]:
    """
    Get the LLM messages for a batch of samples.

    Args:
        system_role: The role of the system.
        system_prompt: The list of prompts for each sample in the batch.
        audio_tag: The tag for the audio placeholder.
        blank_token: The token for blank/no-emission.
        chunk_size: The number of frames per chunk.
        num_delay_frames: Number of frames to delay word emission after word end.
        audio_durations_secs: List of audio durations in seconds, one per sample.
        frame_length_in_secs: The length of a single frame in seconds.
        alignments: List of lists of WordAlignment objects for the batch.
        transcripts: Original transcription strings, one per sample.  When provided,
            assistant turn content preserves punctuation and spacing from the transcript.
        words_per_group: Minimum number of words to buffer before emitting an
            assistant turn (default 1 = emit each word immediately).
    """
    if transcripts is None:
        transcripts = [None] * len(audio_durations_secs)
    batch_messages = []
    for sample_alignments, duration_secs, prompt, transcript in zip(
        alignments,
        audio_durations_secs,
        system_prompt,
        transcripts,
    ):
        batch_messages.append(
            get_llm_messages_for_sample(
                system_role=system_role,
                system_prompt=prompt,
                audio_tag=audio_tag,
                blank_token=blank_token,
                chunk_size=chunk_size,
                num_delay_frames=num_delay_frames,
                audio_duration_secs=duration_secs,
                frame_length_in_secs=frame_length_in_secs,
                alignments=sample_alignments,
                transcript=transcript,
                words_per_group=words_per_group,
                chunk_step=chunk_step,
                project_unaligned_text_to_chunks=project_unaligned_text_to_chunks,
                fixed_chunk_group_schedule=fixed_chunk_group_schedule,
                subsampling_factor=subsampling_factor,
                random_delay_max_frames=random_delay_max_frames,
                delay_weight_gamma=delay_weight_gamma,
                apply_random_delay=apply_random_delay,
                use_flush=use_flush,
                flush_token=flush_token,
            )
        )
    return batch_messages


def parse_chat_template_ids(hf_tok, last_turn: bool = False) -> tuple[list[int], list[int], list[int]]:
    """Discover turn-structure token IDs from a HuggingFace chat template.

    Extracts the structural token IDs that surround user and assistant content
    in the chat template.  Uses a 2-message sentinel conversation (1 user +
    1 assistant) to get the ``user_header``, ``asst_footer``, and the full
    ``user_footer_and_asst_header`` (which may include Qwen3-style
    ``<think>...</think>`` suppression tags).

    When ``last_turn=False``, a second 4-message sentinel is used to obtain
    the assistant header *without* thinking tags — Qwen3 only injects them on
    the last assistant turn, and in streaming each chunk is a non-final turn.

    When ``last_turn=True``, the 2-message result is returned as-is, since the
    assistant turn IS the last turn and must include thinking suppression tags
    to match training.

    Args:
        hf_tok: A HuggingFace tokenizer (``tokenizer.tokenizer``).
        last_turn: When True, the extracted assistant header corresponds to the
            last turn in the conversation, which may include thinking
            suppression tags (e.g. for single-turn offline inference).

    Returns:
        ``(user_header_ids, user_footer_and_asst_header_ids, asst_footer_ids)``

        - *user_header_ids*: tokens before user content, BOS stripped
          (e.g. ``[<|im_start|>, user, \\n]``).
        - *user_footer_and_asst_header_ids*: tokens between user content and
          assistant content.
        - *asst_footer_ids*: tokens after assistant content
          (e.g. ``[<|im_end|>, \\n]``).
    """
    _SENTINEL = "XSENTINELX"

    # --- 2-message template: correct footer, full assistant header ---
    convo_2msg = hf_tok.apply_chat_template(
        [
            {"role": "user", "content": _SENTINEL},
            {"role": "assistant", "content": _SENTINEL},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    parts = convo_2msg.split(_SENTINEL)
    assert len(parts) >= 3, f"Expected >=3 parts after splitting on sentinel, got {len(parts)}: {parts}"

    user_header_ids = hf_tok.encode(parts[0], add_special_tokens=False)
    asst_footer_ids = hf_tok.encode(parts[2], add_special_tokens=False) if parts[2].strip() else []

    # Strip leading BOS from user header — it is already in the KV cache
    # from the system prompt during inference.
    bos_id = getattr(hf_tok, "bos_token_id", None)
    if user_header_ids and bos_id is not None and user_header_ids[0] == bos_id:
        user_header_ids = user_header_ids[1:]

    if last_turn:
        # Last turn: use the 2-msg assistant header (includes thinking tags).
        user_footer_and_asst_header_ids = hf_tok.encode(parts[1], add_special_tokens=False)
    else:
        # Non-last turn: use the 4-msg assistant header (no thinking tags).
        # The 4-msg trick places the sentinel on the first assistant turn,
        # which is NOT the last turn → Qwen3 omits thinking tags.
        convo_4msg = hf_tok.apply_chat_template(
            [
                {"role": "user", "content": _SENTINEL},
                {"role": "assistant", "content": _SENTINEL},
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "x"},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        parts_4msg = convo_4msg.split(_SENTINEL)
        assert len(parts_4msg) >= 3
        user_footer_and_asst_header_ids = hf_tok.encode(parts_4msg[1], add_special_tokens=False)

    return user_header_ids, user_footer_and_asst_header_ids, asst_footer_ids


def build_compact_turn_markers(hf_tok, write_token: str) -> tuple[list[int], list[int], list[int]]:
    """Return the compact-format analogue of ``parse_chat_template_ids``.

    Compact format drops the user/assistant role delimiters: turns look like
    ``<audio>*N <write_token> TEXT <eos>`` with no header before audio and only
    the ``write_token`` marking the audio→text transition.  The turn-end is
    the tokenizer's native EOS.

    ``write_token`` should be an existing vocab token the LLM saw pretraining
    as a turn-boundary marker (e.g. ``"<|im_start|>"`` for Qwen3,
    ``"<start_of_turn>"`` for Gemma).
    """
    write_ids = hf_tok.encode(write_token, add_special_tokens=False)
    if len(write_ids) != 1:
        raise ValueError(
            f"write_token {write_token!r} must encode to exactly 1 token, got {write_ids}. "
            f"Pick a tokenizer-native turn-boundary token or override via config."
        )
    eos_id = getattr(hf_tok, "eos_token_id", None)
    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id is required for compact_template=True")
    return [], [write_ids[0]], [eos_id]


def _tokenize_compact_with_assistant_mask(
    messages: List[dict],
    tokenizer: AutoTokenizer,
    write_id: int,
    eos_id: int,
    supervise_im_end_in_loss: bool = False,
    empty_chunk_eos_only: bool = False,
    blank_token: Optional[str] = None,
    emit_weights: bool = False,
) -> tuple[list[int], list[int], Optional[list[float]]]:
    """Tokenize chat messages in compact format and return (input_ids, assistant_mask, assistant_weights).

    ``assistant_weights`` (aligned 1:1 with ``input_ids``) carries each token's CE
    weight from the assistant message ``weight`` field — applied to that turn's
    ``write_id`` and content tokens. The closing ``<eos>`` / ``<|im_end|>`` (when
    supervised) inherits the weight of the token PRIOR to it in the chunk (the
    last content token), or 1.0 when the chunk emitted no content tokens —
    mirroring the ASR EOC arc. Non-assistant tokens and empty chunks get weight
    1.0. Returns ``None`` for the weights when ``emit_weights`` is False (the
    common, unweighted case).

    Compact per-turn layout (no role wrapping between audio and text):
        [system_wrapped] [user_content, <write>, asst_content, <eos>]*K

    The system prompt IS still wrapped via ``apply_chat_template`` (Qwen3 system
    block), only the per-turn scaffolding is compacted.  Loss is applied on
    ``<write>`` and assistant content.  When ``supervise_im_end_in_loss=True``,
    the closing ``<eos>`` is also trainable.

    When ``empty_chunk_eos_only=True`` (and ``blank_token`` is given), an empty
    chunk — whose assistant content is exactly ``blank_token`` — is encoded as
    ``<write> <eos>`` with NO ``<blank>`` token in between, and its ``<eos>`` is
    always supervised (mask=1) since it is that chunk's sole target. This keeps
    train/inference consistent: the model never sees/predicts a <blank> for
    empty chunks, it predicts <eos> directly.
    """
    hf_tok = tokenizer.tokenizer

    input_ids: list[int] = []
    assistant_mask: list[int] = []
    assistant_weights: Optional[list[float]] = [] if emit_weights else None

    def _extend_w(n: int, value: float = 1.0) -> None:
        if assistant_weights is not None:
            assistant_weights.extend([value] * n)

    # --- System section: keep Qwen3-style wrapping ---
    system_msgs = [m for m in messages if m["role"] == "system"]
    if system_msgs:
        system_ids = hf_tok.apply_chat_template(
            system_msgs,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        input_ids.extend(list(system_ids))
        assistant_mask.extend([0] * len(system_ids))
        _extend_w(len(system_ids))

    # --- Per-turn compact encoding ---
    turn_msgs = [m for m in messages if m["role"] != "system"]
    # Pairs: (user, assistant). The final turn may be user-only (trailing silence).
    i = 0
    while i < len(turn_msgs):
        msg = turn_msgs[i]
        if msg["role"] == "user":
            user_ids = hf_tok.encode(msg["content"], add_special_tokens=False) if msg["content"] else []
            input_ids.extend(user_ids)
            assistant_mask.extend([0] * len(user_ids))
            _extend_w(len(user_ids))
            i += 1
            # Pair with following assistant turn if present.
            if i < len(turn_msgs) and turn_msgs[i]["role"] == "assistant":
                asst = turn_msgs[i]
                if empty_chunk_eos_only and blank_token is not None and asst["content"] == blank_token:
                    # Empty chunk → bare <eos> (no <blank>). <eos> is always
                    # supervised here: it is the chunk's only target. Weight 1.0.
                    input_ids.append(write_id)
                    assistant_mask.append(1)
                    input_ids.append(eos_id)
                    assistant_mask.append(1)
                    _extend_w(2)
                else:
                    asst_ids = hf_tok.encode(asst["content"], add_special_tokens=False) if asst["content"] else []
                    w = float(asst.get("weight", 1.0))
                    # write_id
                    input_ids.append(write_id)
                    assistant_mask.append(1)
                    # assistant content
                    input_ids.extend(asst_ids)
                    assistant_mask.extend([1] * len(asst_ids))
                    # write_id + content carry this turn's weight.
                    _extend_w(1 + len(asst_ids), w)
                    # eos / <|im_end|>: when supervised, it inherits the weight of
                    # the token PRIOR to it in this chunk (the last content token),
                    # or 1.0 when the chunk emitted no content tokens — mirroring
                    # the ASR EOC arc taking the last token's weight (1.0 if empty).
                    input_ids.append(eos_id)
                    assistant_mask.append(int(supervise_im_end_in_loss))
                    _extend_w(1, w if asst_ids else 1.0)
                i += 1
        else:
            # Orphan assistant (shouldn't normally occur) — treat as standalone asst segment.
            if empty_chunk_eos_only and blank_token is not None and msg["content"] == blank_token:
                input_ids.append(write_id)
                assistant_mask.append(1)
                input_ids.append(eos_id)
                assistant_mask.append(1)
                _extend_w(2)
            else:
                asst_ids = hf_tok.encode(msg["content"], add_special_tokens=False) if msg["content"] else []
                w = float(msg.get("weight", 1.0))
                input_ids.append(write_id)
                assistant_mask.append(1)
                input_ids.extend(asst_ids)
                assistant_mask.extend([1] * len(asst_ids))
                _extend_w(1 + len(asst_ids), w)
                # eos / <|im_end|>: weight of the prior content token (1.0 if none).
                input_ids.append(eos_id)
                assistant_mask.append(int(supervise_im_end_in_loss))
                _extend_w(1, w if asst_ids else 1.0)
            i += 1

    return input_ids, assistant_mask, assistant_weights


def _tokenize_with_assistant_mask(
    messages: List[dict],
    tokenizer: AutoTokenizer,
    supervise_im_end_in_loss: bool = False,
) -> tuple[list[int], list[int]]:
    """
    Tokenize chat messages and return (input_ids, assistant_mask).

    First tries HF's ``return_assistant_tokens_mask`` (requires ``{% generation %}``
    in the chat template).  If that returns an all-zero mask, falls back to a
    sequential-search strategy: tokenize each assistant turn's content separately
    and locate it in the full token sequence.

    Args:
        messages: list of ``{"role": ..., "content": ...}`` dicts.
        tokenizer: NeMo AutoTokenizer (``tokenizer.tokenizer`` is the HF tokenizer).

    Returns:
        (input_ids, assistant_mask) — both plain Python lists of ints.
    """
    hf_tok = tokenizer.tokenizer

    # --- primary path: use HF's built-in mask ---
    result = hf_tok.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        enable_thinking=False,
    )
    input_ids = list(result["input_ids"])
    assistant_mask = list(result["assistant_masks"])

    if any(assistant_mask):
        return input_ids, assistant_mask

    # --- fallback: diff-based content detection ---
    # Tokenize the same messages but with all assistant contents replaced by
    # a single-character sentinel.  The two-pointer walk then identifies
    # content tokens (present in full but replaced by the sentinel in the
    # reference).
    #
    # We use a sentinel instead of "" (empty string) to preserve BPE context
    # boundaries.  With "", template tokens adjacent to the content can merge
    # (e.g. "assistant\n" + content "\n" → token "\n\n" vs "assistant\n" + "" →
    # token "\n"), causing the two-pointer to desync.  A sentinel like "X"
    # tokenizes to exactly 1 token and prevents BPE merging with neighbors.
    _SENTINEL_CHAR = "X"
    assistant_mask = [0] * len(input_ids)

    msgs_sentinel = [{**m, "content": _SENTINEL_CHAR} if m["role"] == "assistant" else m for m in messages]
    ids_sentinel_result = hf_tok.apply_chat_template(
        msgs_sentinel,
        tokenize=True,
        enable_thinking=False,
    )
    ids_sentinel = list(
        ids_sentinel_result["input_ids"] if hasattr(ids_sentinel_result, "keys") else ids_sentinel_result
    )

    eos_id = getattr(hf_tok, 'eos_token_id', None)
    i, j = 0, 0  # pointers into input_ids and ids_sentinel
    while i < len(input_ids) and j < len(ids_sentinel):
        if input_ids[i] == ids_sentinel[j]:
            i += 1
            j += 1
        else:
            # Divergence: ids_sentinel has the sentinel (1 token) where
            # input_ids has the actual content (1+ tokens).
            j += 1  # skip the sentinel token
            while i < len(input_ids) and (j >= len(ids_sentinel) or input_ids[i] != ids_sentinel[j]):
                assistant_mask[i] = 1
                i += 1
            # Optionally include EOS token in the assistant footer.
            if supervise_im_end_in_loss and eos_id is not None and i < len(input_ids) and input_ids[i] == eos_id:
                assistant_mask[i] = 1

    # Any remaining tokens in input_ids are also content.
    while i < len(input_ids):
        assistant_mask[i] = 1
        i += 1

    return input_ids, assistant_mask


def _strip_blank_for_empty_chunks(
    input_ids: list[int],
    assistant_mask: list[int],
    blank_id: int,
    eos_id: Optional[int],
) -> tuple[list[int], list[int]]:
    """Drop the <blank> token from empty chunks (standard chat template).

    For ``empty_chunk_eos_only=True`` with the non-compact chat template, an
    empty chunk is tokenized as ``... assistant_header <blank> <eos> ...``. This
    removes every ``blank_id`` token so the empty chunk becomes
    ``... assistant_header <eos> ...`` (the model emits <eos>/<|im_end|> directly),
    and forces the immediately-following ``<eos>`` (the turn end) to be
    supervised (mask=1) since it is that chunk's sole target — mirroring the
    fact that ``<blank>`` was previously the supervised target.

    ``<blank>`` is only ever used as empty-chunk content, so removing all
    occurrences is safe. No-op when ``blank_id < 0`` (blank disabled).
    """
    if blank_id is None or blank_id < 0:
        return input_ids, assistant_mask
    new_ids: list[int] = []
    new_mask: list[int] = []
    supervise_next_eos = False
    for tid, m in zip(input_ids, assistant_mask):
        if tid == blank_id:
            # Drop the blank; the <eos> that follows closes this empty chunk and
            # must be supervised so the chunk-end is learned.
            supervise_next_eos = True
            continue
        if supervise_next_eos:
            if eos_id is not None and tid == eos_id:
                m = 1
            supervise_next_eos = False
        new_ids.append(tid)
        new_mask.append(m)
    return new_ids, new_mask


def _mark_assistant_footer_for_loss(
    input_ids: list[int],
    assistant_mask: list[int],
    assistant_footer_ids: list[int],
) -> list[int]:
    """Mark the first assistant-footer token after assistant content.

    This intentionally uses the assistant content mask plus the full footer
    sequence, not token ID alone, because the same token can also close user
    turns in the chat template.
    """
    if not assistant_footer_ids:
        return assistant_mask

    flen = len(assistant_footer_ids)
    for idx in range(1, len(input_ids) - flen + 1):
        # Only assistant-mask 1->0 transitions can be assistant footers.
        if not (assistant_mask[idx - 1] and not assistant_mask[idx]):
            continue
        matched = True
        for offset, footer_id in enumerate(assistant_footer_ids):
            if assistant_mask[idx + offset] or input_ids[idx + offset] != footer_id:
                matched = False
                break
        if matched:
            assistant_mask[idx] = 1
    return assistant_mask


def _replace_audio_chunks(
    token_ids: list[int],
    chunk_ids: list[int],
    chunk_size: int,
    mask: list | None = None,
    weights: list | None = None,
) -> list[int] | tuple[list[int], list] | tuple[list[int], list, list]:
    """Replace each occurrence of *chunk_ids* with *chunk_size* copies of ``AUDIO_TOKEN_IDX``.

    This handles multi-token audio tags where BPE merges tokens across adjacent
    tags (e.g., ``<audio><audio>`` tokenizes differently from ``encode("<audio>") * 2``).
    By matching the full chunk at once, we avoid the BPE boundary problem.

    When *mask* (and optionally *weights*) is provided it is adjusted in sync:
    each matched span is replaced with *chunk_size* copies of the first element
    of that span (typically 0/1.0 for user-turn content).

    Returns:
        new_token_ids                        when mask is None
        (new_token_ids, new_mask)            when mask is provided, weights is None
        (new_token_ids, new_mask, new_weights) when mask AND weights are provided
    """
    chunk_len = len(chunk_ids)
    new_ids: list[int] = []
    new_mask: list | None = [] if mask is not None else None
    new_weights: list | None = [] if weights is not None else None
    i = 0
    n = len(token_ids)
    while i < n:
        if token_ids[i : i + chunk_len] == chunk_ids:
            new_ids.extend([AUDIO_TOKEN_IDX] * chunk_size)
            if new_mask is not None:
                new_mask.extend([mask[i]] * chunk_size)
            if new_weights is not None:
                new_weights.extend([weights[i]] * chunk_size)
            i += chunk_len
        else:
            new_ids.append(token_ids[i])
            if new_mask is not None:
                new_mask.append(mask[i])
            if new_weights is not None:
                new_weights.append(weights[i])
            i += 1
    if mask is None:
        return new_ids
    if new_weights is not None:
        return new_ids, new_mask, new_weights
    return new_ids, new_mask


class StreamingSTTDataset(torch.utils.data.Dataset):
    """
    Dataset for StreamingSTTModel.
    Operates directly on Lhotse Cuts (no NeMoMultimodalConversation wrapper).
    """

    def __init__(self, cfg: DictConfig | dict, tokenizer: AutoTokenizer, defer_get_batch: bool = False):
        """
        Args:
            cfg: Configuration for the dataset.
            tokenizer: Tokenizer for the dataset.
            defer_get_batch: If True, defer the get_batch_data call to the __getitem__ method and let the model do it.
                This is used in online forced alignment mode.
        """
        self.defer_get_batch = defer_get_batch
        self.tokenizer = tokenizer
        self.randomize_fixed_chunk_groups = True
        self.cfg: StreamingSTTDataConfig = to_dataclass(StreamingSTTDataConfig, cfg)
        # Unescape Python escape sequences (e.g. "\\n" → "\n") because Hydra/OmegaConf
        # loads YAML strings literally without interpreting backslash escapes.
        self.cfg.blank_token = self.cfg.blank_token.encode().decode('unicode_escape')

        # Tokenize the full audio chunk string (audio_tag * chunk_size) to get
        # its token ID sequence.  We must encode the full chunk as a single string
        # because BPE may merge tokens across adjacent audio tags (e.g.,
        # "<audio><audio>" tokenizes differently from encode("<audio>") * 2).
        # When chunk_size=-1 (offline mode), audio_chunk_ids is computed per sample
        # in get_batch_data because num_frames varies per sample.
        # Encoder subsampling: the LLM consumes ``chunk_size // factor`` audio
        # placeholders per chunk (the model shrinks the encoder output by the
        # same factor). Timeline/alignment math stays in ``chunk_size`` frames.
        self.subsampling_factor = max(int(getattr(self.cfg, "encoder_subsampling_factor", 1) or 1), 1)
        if self.subsampling_factor > 1:
            if self.cfg.chunk_size <= 0:
                raise ValueError(
                    f"encoder_subsampling_factor={self.subsampling_factor} requires fixed chunking "
                    f"(chunk_size>0), got chunk_size={self.cfg.chunk_size}"
                )
            # chunk_size need NOT be divisible by the factor: when it isn't, the
            # model pads each chunk's tail (repeating its last frame) up to a
            # multiple of the factor, so each chunk yields ceil(chunk_size/factor)
            # audio tokens. We emit exactly that many placeholders below.
        # Number of audio placeholders the LLM sees per chunk (post-subsampling):
        # ceil(chunk_size / factor) (== chunk_size // factor when divisible).
        self.audio_tokens_per_chunk = (
            (self.cfg.chunk_size + self.subsampling_factor - 1) // self.subsampling_factor
            if self.cfg.chunk_size > 0
            else None
        )

        if self.cfg.chunk_size > 0:
            audio_chunk_str = self.cfg.audio_tag * self.audio_tokens_per_chunk
            self.audio_chunk_ids = self.tokenizer.tokenizer.encode(audio_chunk_str, add_special_tokens=False)
        else:
            self.audio_chunk_ids = None

        # blank_token is part of the LLM output vocabulary — it must be a single
        # special token, otherwise loss is dominated by multi-token blanks and
        # generation becomes unreliable.  The model's __init__ should have called
        # tokenizer.add_special_tokens() before passing the tokenizer here.
        # An empty blank_token ("") disables the explicit blank: chunks without
        # words get empty assistant turns, stop signal is <|im_end|> alone.
        if self.cfg.blank_token == "":
            if self.cfg.chunk_size == 0:
                raise ValueError(
                    "blank_token='' is not supported with dynamic chunking (chunk_size=0) — "
                    "dynamic chunking requires a token to predict at non-final audio positions."
                )
            self.blank_id = -1
            logging.info("blank_token is empty: blank token mechanism disabled (fixed chunking only)")
        else:
            blank_ids = self.tokenizer.tokenizer.encode(self.cfg.blank_token, add_special_tokens=False)
            logging.info(f"blank_token: {str(self.cfg.blank_token)}, blank_id: {blank_ids}")
            if len(blank_ids) != 1:
                raise ValueError(
                    f"blank_token '{self.cfg.blank_token}' tokenizes into {len(blank_ids)} tokens {blank_ids}. "
                    f"It must be a single special token. Make sure the model adds it via "
                    f"tokenizer.add_special_tokens() before constructing the dataset."
                )
            self.blank_id = blank_ids[0]

        # Compact template: cache write_id and eos_id. Skip the parse_chat_template_ids
        # call since we derive the markers directly from config.
        if self.cfg.compact_template:
            hf_tok = self.tokenizer.tokenizer
            _, ufah_ids, af_ids = build_compact_turn_markers(hf_tok, self.cfg.write_token)
            self._write_id = ufah_ids[0]
            self._compact_eos_id = af_ids[0]
            logging.info(
                f"compact_template enabled: write_token={self.cfg.write_token!r} "
                f"(id={self._write_id}), eos_id={self._compact_eos_id}"
            )
        else:
            self._write_id = None
            self._compact_eos_id = None

        # Flush token: must be a single vocab token (model.__init__ registers it).
        if getattr(self.cfg, "use_flush", False):
            flush_ids = self.tokenizer.tokenizer.encode(self.cfg.flush_token, add_special_tokens=False)
            if len(flush_ids) != 1:
                raise ValueError(
                    f"flush_token '{self.cfg.flush_token}' tokenizes into {len(flush_ids)} tokens "
                    f"{flush_ids}. It must be a single special token added via "
                    f"tokenizer.add_special_tokens() in the model __init__."
                )
            self._flush_id = flush_ids[0]
            logging.info(f"use_flush enabled: flush_token={self.cfg.flush_token!r} (id={self._flush_id})")
        else:
            self._flush_id = None

        if self.cfg.supervise_im_end_in_loss:
            hf_tok = self.tokenizer.tokenizer
            if self.cfg.compact_template:
                self._assistant_footer_ids = [self._compact_eos_id]
            else:
                _, _, self._assistant_footer_ids = parse_chat_template_ids(hf_tok)
            logging.info(f"Assistant footer supervision enabled: footer_ids={self._assistant_footer_ids}")
        else:
            self._assistant_footer_ids = []

        # For dynamic chunking (chunk_size=0): cache the first token of the
        # user footer sequence (e.g. <|im_end|>).  This is the target the model
        # predicts at the last audio frame of each chunk to signal "ready to transcribe".
        if self.cfg.chunk_size == 0:
            if self.cfg.compact_template:
                # Compact: boundary target is write_id (<|im_start|> in Qwen3).
                self._user_footer_first_id = self._write_id
            else:
                hf_tok = self.tokenizer.tokenizer
                _, user_footer_and_asst_header_ids, _ = parse_chat_template_ids(hf_tok)
                self._user_footer_first_id = user_footer_and_asst_header_ids[0]
        else:
            self._user_footer_first_id = None

    def clone_for_eval(self) -> "StreamingSTTDataset":
        dataset = copy.copy(self)
        dataset.randomize_fixed_chunk_groups = False
        return dataset

    def __getitem__(self, cuts: CutSet) -> StreamingSTTBatch | None:
        try:
            audios, audio_lens, cuts = collate_audio(cuts, fault_tolerant=True)
        except Exception as e:
            logging.warning(f"Error collating audio from cuts: {e}")
            return None
        if len(cuts) == 0:
            logging.warning("No cuts found in the batch")
            return None

        text = [cut.supervisions[0].text for cut in cuts]

        if self.defer_get_batch:
            return StreamingSTTBatch(
                cuts=cuts,
                audios=audios,
                audio_lens=audio_lens,
                text=text,
            )

        alignments = get_word_alignments_for_batch(cuts)

        return self.get_batch_data(cuts, audios, audio_lens, alignments, text)

    def sample_flush_truncation_plan(
        self, audio_lens: torch.Tensor, apply_random_delay: bool = True
    ) -> Optional[List[Optional[int]]]:
        """Sample a per-sample mid-utterance truncation plan for the flush aug.

        Returns ``None`` when the augmentation is inactive (feature off, non-fixed
        chunking, eval, or empty batch). Otherwise returns a list (len == batch)
        whose entries are the target chunk count ``C`` for cuts selected to be
        truncated (each having > C chunks), or ``None`` for cuts left full. A
        SINGLE shared cutoff ``C`` is drawn per batch from
        ``[1, n_chunks(shortest cut)]`` (so every cut can be truncated to it);
        each cut is then independently selected with probability
        ``flush_truncate_prob``.

        Sampling the plan ONCE and passing it to :meth:`get_batch_data` lets the
        ``encoder_reuse_k>1`` path truncate the audio identically across all K
        reused views (the encoder output is shared, so audio / audio-slot counts
        must match across views — only the delay randomization may differ).
        """
        if not (
            apply_random_delay
            and bool(getattr(self.cfg, "use_flush", False))
            and float(getattr(self.cfg, "flush_truncate_prob", 0.0)) > 0.0
            and self.cfg.chunk_size > 0
        ):
            return None
        durations = (audio_lens.float() / self.cfg.sample_rate).tolist()
        if not durations:
            return None
        chunk_size = self.cfg.chunk_size
        frame_len = self.cfg.frame_length_in_secs
        n_chunks_per = [max(1, math.ceil(math.ceil(d / frame_len) / chunk_size)) for d in durations]
        C = random.randint(1, min(n_chunks_per))
        prob = float(self.cfg.flush_truncate_prob)
        plan: List[Optional[int]] = [
            (C if (n > C and random.random() < prob) else None) for n in n_chunks_per
        ]
        return plan if any(p is not None for p in plan) else None

    def get_batch_data(
        self,
        cuts: CutSet,
        audios: torch.Tensor,
        audio_lens: torch.Tensor,
        alignments: List[List[WordAlignment]],
        text: List[str],
        randomize_fixed_chunk_groups: Optional[bool] = None,
        apply_random_delay: bool = True,
        truncation_plan=_TRUNCATION_AUTO,
    ) -> StreamingSTTBatch:
        audio_durations_secs = (audio_lens.float() / self.cfg.sample_rate).tolist()

        # K-step alignment (dynamic chunking only): pad each waveform up to a
        # multiple of K frames so the encoder produces exactly that many
        # embeddings, matching the K-snapped segment lengths the dataset will
        # construct below. K=1 → no-op.
        K = max(int(getattr(self.cfg, "chunk_step", 1)), 1)
        if K > 1 and self.cfg.chunk_size == 0:
            new_lens = []
            for dur in audio_durations_secs:
                num_frames = math.ceil(dur / self.cfg.frame_length_in_secs)
                num_frames_padded = math.ceil(num_frames / K) * K
                samples_padded = math.ceil(num_frames_padded * self.cfg.frame_length_in_secs * self.cfg.sample_rate)
                new_lens.append(samples_padded)
            max_samples = max(new_lens) if new_lens else int(audio_lens.max().item())
            if audios.shape[1] < max_samples:
                audios = F.pad(audios, (0, max_samples - audios.shape[1]))
            audio_lens = torch.tensor(new_lens, dtype=audio_lens.dtype, device=audio_lens.device)
            audio_durations_secs = (audio_lens.float() / self.cfg.sample_rate).tolist()

        # --- Flush truncation augmentation (training only) ---
        # Replace selected cuts with a copy truncated at a chunk boundary so the
        # model learns to flush MID-utterance (not just at the true end). The plan
        # (per-sample target chunk count C, or None) is normally sampled here, but
        # the encoder_reuse_k>1 training path passes a SHARED plan so all K reused
        # views truncate the audio IDENTICALLY — the encoder output is computed
        # once and reused, so the views may differ only in delay randomization,
        # never in the audio / audio-slot count. Truncating = shrink audio_lens to
        # C chunks and keep only words whose audio fully precedes the cutoff;
        # get_llm_messages then appends <flush> after chunk C with any
        # boundary-delayed residual. We copy the outer alignments/text lists so
        # encoder_reuse_k re-calls (which reuse the caller's lists) are not
        # corrupted.
        if truncation_plan is _TRUNCATION_AUTO:
            truncation_plan = self.sample_flush_truncation_plan(audio_lens, apply_random_delay=apply_random_delay)
        if truncation_plan:
            chunk_size = self.cfg.chunk_size
            frame_len = self.cfg.frame_length_in_secs
            sr = self.cfg.sample_rate
            new_audio_lens = audio_lens.clone()
            alignments = [list(a) for a in alignments]
            text = list(text)
            changed = False
            for b, C in enumerate(truncation_plan):
                if not C:
                    continue
                cutoff_secs = C * chunk_size * frame_len
                cutoff_samples = int(round(cutoff_secs * sr))
                kept = [w for w in alignments[b] if w.end_time <= cutoff_secs]
                if not kept:
                    continue  # no complete word in the prefix — skip truncation
                alignments[b] = kept
                new_audio_lens[b] = min(int(audio_lens[b].item()), cutoff_samples)
                text[b] = " ".join(w.text for w in kept)
                changed = True
            if changed:
                audio_lens = new_audio_lens
                audio_durations_secs = (audio_lens.float() / self.cfg.sample_rate).tolist()

        system_prompts = [cut.custom.get(self.cfg.prompt_field, self.cfg.system_prompt) for cut in cuts]
        if randomize_fixed_chunk_groups is None:
            randomize_fixed_chunk_groups = self.randomize_fixed_chunk_groups
        fixed_chunk_group_schedule: Optional[List[int]] = None
        max_audio_chunks_per_turn = max(int(getattr(self.cfg, "max_audio_chunks_per_turn", 1)), 1)
        chunk_group_choices = parse_chunk_group_choices(
            getattr(self.cfg, "audio_chunks_per_turn_choices", None)
        )
        # The largest group we can produce: from the discrete choices when given,
        # else the uniform-range upper bound. Randomization is only meaningful
        # when this exceeds 1 (otherwise every turn is a single chunk).
        effective_max_chunks = max(chunk_group_choices) if chunk_group_choices else max_audio_chunks_per_turn
        if randomize_fixed_chunk_groups and self.cfg.chunk_size > 0 and effective_max_chunks > 1:
            max_num_chunks = max(
                (
                    math.ceil(math.ceil(duration / self.cfg.frame_length_in_secs) / self.cfg.chunk_size)
                    for duration in audio_durations_secs
                ),
                default=0,
            )
            fixed_chunk_group_schedule = sample_fixed_chunk_group_schedule(
                max_num_chunks,
                max_audio_chunks_per_turn,
                allowed_group_sizes=chunk_group_choices,
            )

        batch_messages = get_llm_messages_for_batch(
            system_role=self.cfg.system_role,
            system_prompt=system_prompts,
            audio_tag=self.cfg.audio_tag,
            blank_token=self.cfg.blank_token,
            chunk_size=self.cfg.chunk_size,
            num_delay_frames=self.cfg.num_delay_frames,
            audio_durations_secs=audio_durations_secs,
            frame_length_in_secs=self.cfg.frame_length_in_secs,
            alignments=alignments,
            transcripts=text,
            words_per_group=self.cfg.words_per_group,
            chunk_step=K,
            project_unaligned_text_to_chunks=self.cfg.project_unaligned_text_to_chunks,
            fixed_chunk_group_schedule=fixed_chunk_group_schedule,
            subsampling_factor=self.subsampling_factor,
            random_delay_max_frames=int(getattr(self.cfg, "random_delay_max_frames", 0)),
            delay_weight_gamma=float(getattr(self.cfg, "delay_weight_gamma", 1.0)),
            apply_random_delay=apply_random_delay,
            use_flush=bool(getattr(self.cfg, "use_flush", False)),
            flush_token=str(getattr(self.cfg, "flush_token", "<flush>")),
        )

        # Per-token CE weights are only produced (and threaded through) when
        # position-weighting is active AND we use the compact template.
        emit_weights = (
            float(getattr(self.cfg, "delay_weight_gamma", 1.0)) != 1.0 and bool(self.cfg.compact_template)
        )

        all_input_ids = []
        all_target_ids = []
        all_target_weights = [] if emit_weights else None
        chunk_word_alignment = []

        for sample_idx, messages in enumerate(batch_messages):
            chunk_word_alignment.append(
                build_chunk_word_alignment_debug(
                    messages,
                    audio_tag=self.cfg.audio_tag,
                    blank_token=self.cfg.blank_token,
                    tokenizer=self.tokenizer,
                )
            )
            # Tokenize and compute assistant content mask (+ optional per-token weights).
            token_weights: Optional[list[float]] = None
            if self.cfg.compact_template:
                input_ids, assistant_mask, token_weights = _tokenize_compact_with_assistant_mask(
                    messages,
                    self.tokenizer,
                    self._write_id,
                    self._compact_eos_id,
                    supervise_im_end_in_loss=self.cfg.supervise_im_end_in_loss,
                    empty_chunk_eos_only=bool(getattr(self.cfg, "empty_chunk_eos_only", False)),
                    blank_token=self.cfg.blank_token,
                    emit_weights=emit_weights,
                )
            else:
                input_ids, assistant_mask = _tokenize_with_assistant_mask(
                    messages,
                    self.tokenizer,
                    supervise_im_end_in_loss=self.cfg.supervise_im_end_in_loss,
                )
                # Standard chat template: empty chunks are tokenized with a
                # <blank> content token. When empty_chunk_eos_only is set, strip
                # those blanks so empty chunks emit <|im_end|> directly.
                if bool(getattr(self.cfg, "empty_chunk_eos_only", False)):
                    input_ids, assistant_mask = _strip_blank_for_empty_chunks(
                        input_ids,
                        assistant_mask,
                        self.blank_id,
                        self.tokenizer.tokenizer.eos_token_id,
                    )

            # Replace audio tag sequences with AUDIO_TOKEN_IDX markers. In the
            # default fixed path we can use one cached pattern; grouped turns
            # need per-user matching because BPE can tokenize 14/28/42 adjacent
            # audio tags differently.
            if self.audio_chunk_ids is not None and fixed_chunk_group_schedule is None:
                # Fixed chunking: single pre-computed pattern. The replacement
                # count is the post-subsampling token count (chunk_size//factor).
                if token_weights is not None:
                    input_ids, assistant_mask, token_weights = _replace_audio_chunks(
                        input_ids,
                        self.audio_chunk_ids,
                        self.audio_tokens_per_chunk,
                        mask=assistant_mask,
                        weights=token_weights,
                    )
                else:
                    input_ids, assistant_mask = _replace_audio_chunks(
                        input_ids, self.audio_chunk_ids, self.audio_tokens_per_chunk, mask=assistant_mask
                    )
            else:
                # Offline (chunk_size=-1) or dynamic (chunk_size=0): variable audio tag
                # counts per user turn.  Replace each user turn's audio tags separately.
                hf_tok = self.tokenizer.tokenizer
                for msg in messages:
                    if msg["role"] != "user":
                        continue
                    n_tags = msg["content"].count(self.cfg.audio_tag)
                    if n_tags == 0:
                        continue
                    chunk_ids = hf_tok.encode(self.cfg.audio_tag * n_tags, add_special_tokens=False)
                    if token_weights is not None:
                        input_ids, assistant_mask, token_weights = _replace_audio_chunks(
                            input_ids, chunk_ids, n_tags, mask=assistant_mask, weights=token_weights
                        )
                    else:
                        input_ids, assistant_mask = _replace_audio_chunks(
                            input_ids, chunk_ids, n_tags, mask=assistant_mask
                        )

            # Build targets: next-token prediction with loss only on assistant content.
            # target[i] corresponds to input[i] and holds the token at position i+1.
            # Loss is applied only where assistant_mask[i+1] is True.
            if self._assistant_footer_ids:
                assistant_mask = _mark_assistant_footer_for_loss(
                    input_ids,
                    assistant_mask,
                    self._assistant_footer_ids,
                )
            target_ids = input_ids[1:] + [IGNORE_INDEX]
            target_mask = assistant_mask[1:] + [0]
            target_ids = [tid if m else IGNORE_INDEX for tid, m in zip(target_ids, target_mask)]

            # Shift per-token weights the same way as the mask so they align with
            # ``target_ids`` (weight of position i+1). Only meaningful where the
            # target is supervised; elsewhere it is ignored by the loss.
            if all_target_weights is not None:
                target_weights = (token_weights[1:] + [1.0]) if token_weights is not None else [1.0] * len(target_ids)

            # Dynamic chunking: train the model to predict at audio positions.
            # Non-final audio frames → target = blank_id ("need more audio")
            # Final audio frame (before user footer) → target = user_footer first token ("ready")
            if self.cfg.chunk_size == 0:
                user_footer_id = self._user_footer_first_id
                for i in range(len(input_ids)):
                    if input_ids[i] != AUDIO_TOKEN_IDX:
                        continue
                    next_is_audio = i + 1 < len(input_ids) and input_ids[i + 1] == AUDIO_TOKEN_IDX
                    target_ids[i] = self.blank_id if next_is_audio else user_footer_id

            all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            all_target_ids.append(torch.tensor(target_ids, dtype=torch.long))
            if all_target_weights is not None:
                all_target_weights.append(torch.tensor(target_weights, dtype=torch.float))

        if self.cfg.chunk_size >= 0:  # fixed chunking or dynamic chunking: right-pad
            input_tokens = right_collate_vectors(all_input_ids, padding_value=self.tokenizer.pad_id)
            target_tokens = right_collate_vectors(all_target_ids, padding_value=IGNORE_INDEX)
            input_token_lens = torch.tensor([len(ids) for ids in all_input_ids], dtype=torch.long)
            target_token_lens = torch.tensor([len(ids) for ids in all_target_ids], dtype=torch.long)
            target_weights = (
                right_collate_vectors(all_target_weights, padding_value=0.0)
                if all_target_weights is not None
                else None
            )
        else:  # offline mode: left-pad
            input_tokens = left_collate_vectors(all_input_ids, padding_value=self.tokenizer.pad_id)
            target_tokens = left_collate_vectors(all_target_ids, padding_value=IGNORE_INDEX)
            # length is the same size as input_tokens.shape[1] since they're left-padded
            input_token_lens = torch.tensor(
                [input_tokens.shape[1] for _ in range(len(all_input_ids))], dtype=torch.long
            )
            target_token_lens = torch.tensor(
                [target_tokens.shape[1] for _ in range(len(all_target_ids))], dtype=torch.long
            )
            target_weights = (
                left_collate_vectors(all_target_weights, padding_value=0.0)
                if all_target_weights is not None
                else None
            )

        # --- Parallel chunk heads: build anchor positions + K-slot target slates ---
        chunk_anchor_positions, chunk_target_tokens = self._build_parallel_chunk_targets(
            all_input_ids,
            cuts=cuts,
            transcripts=text,
        )

        return StreamingSTTBatch(
            audios=audios,
            audio_lens=audio_lens,
            input_tokens=input_tokens,
            input_token_lens=input_token_lens,
            target_tokens=target_tokens,
            target_token_lens=target_token_lens,
            text=text,
            chunk_word_alignment=chunk_word_alignment,
            chunk_anchor_positions=chunk_anchor_positions,
            chunk_target_tokens=chunk_target_tokens,
            target_weights=target_weights,
        )

    # ------------------------------------------------------------------
    # Parallel chunk heads (multi-token-per-chunk prediction)
    # ------------------------------------------------------------------

    def _build_parallel_chunk_targets(
        self,
        all_input_ids: List[torch.Tensor],
        cuts: Optional[CutSet] = None,
        transcripts: Optional[List[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build anchor positions and K-slot targets for the parallel heads.

        Returns (None, None) when parallel-chunk-head supervision is disabled
        (parallel_chunk_slots <= 0) or unsupported (non-compact template).

        Iterative K-block scheme (compact template only):
            K is the number of tokens emitted per parallel forward (a *block*),
            decoupled from the max tokens a chunk may contain. A chunk whose
            emit-stream is ``content + [<|im_end|>]`` (length S = N+1) is split
            into ``ceil(S / K)`` blocks of up to K tokens each. Block ``g`` is
            anchored at position ``write_id_pos + g*K`` — the hidden state that,
            under standard next-token semantics, predicts that block's first
            token — and supervises the next K stream tokens:

                stream = [tok_1, ..., tok_N, <|im_end|>]
                block g targets = stream[g*K : (g+1)*K]   (IGNORE-padded to K)

            Examples (K=4):
                N=2 → 1 block:  [t1, t2, im_end, IGN]
                N=4 → 2 blocks: [t1,t2,t3,t4], [im_end, IGN, IGN, IGN]
                N=7 → 2 blocks: [t1,t2,t3,t4], [t5,t6,t7,im_end]

            At inference the model emits one block per forward and iterates
            within a chunk until <|im_end|> appears (or a max-iters cap). There
            is therefore no N>K case to skip — long chunks simply use more
            blocks, so every chunk is fully supervised.

        Padded blocks across the batch (fewer blocks than the batch max) get
        anchor=-1 and all IGNORE_INDEX targets.

        Args:
            all_input_ids: per-sample input token id tensors.
            cuts: optional CutSet for per-utterance identifiers in diagnostics.
            transcripts: optional list of ground-truth transcripts (one per
                sample), retained for diagnostic parity with prior versions.
        """
        K = int(getattr(self.cfg, "parallel_chunk_slots", 0) or 0)
        if K <= 0:
            return None, None
        if not self.cfg.compact_template:
            # Non-compact mode requires a different anchor scheme (last header
            # token vs write_id). Skip silently for v1 — the model side will
            # warn on init if this combination is requested.
            return None, None
        write_id = self._write_id
        eos_id = self._compact_eos_id
        if write_id is None or eos_id is None:
            return None, None

        blank_continuation = bool(getattr(self.cfg, "parallel_blank_continuation", False))
        blank_id = int(getattr(self, "blank_id", -1) or -1)
        if blank_continuation and blank_id < 0:
            # The blank-continuation scheme needs a real <blank> id to mark
            # partial/cut blocks. Without one, fall back to the legacy scheme.
            logging.warning(
                "parallel_blank_continuation=True but blank token is disabled (blank_id<0); "
                "falling back to the legacy fixed-K parallel scheme."
            )
            blank_continuation = False
        cut_prob = float(getattr(self.cfg, "parallel_cut_prob", 0.0) or 0.0)
        # Fresh per-call RNG so cut augmentation is independent across workers /
        # processes without relying on global numpy/torch seeding.
        rng = random.Random() if (blank_continuation and cut_prob > 0.0) else None

        per_sample_anchors: list[list[int]] = []
        per_sample_targets: list[list[list[int]]] = []
        for input_ids_t in all_input_ids:
            ids = input_ids_t.tolist()
            anchors: list[int] = []
            targets: list[list[int]] = []
            i = 0
            n = len(ids)
            while i < n:
                if ids[i] != write_id:
                    i += 1
                    continue
                # Found a chunk anchor. Collect content until the matching eos.
                content: list[int] = []
                j = i + 1
                while j < n and ids[j] != eos_id:
                    content.append(ids[j])
                    j += 1

                if blank_continuation:
                    # Blank-continuation scheme: split the *content* tokens into
                    # variable-length blocks; <|im_end|> never appears inside a
                    # content block — the chunk always closes with a terminator
                    # block [<|im_end|>, IGNORE...]. Partial / randomly-cut
                    # blocks end with a single <blank> "continue" marker, which
                    # is a synthetic head target only (NOT a sequence position),
                    # so anchors advance by the number of *real* content tokens
                    # consumed. Anchor of a block whose first real target is
                    # content[m] is position i + m (i for m=0 → write_id).
                    chunk_anchors, chunk_targets = self._segment_chunk_blocks_blank_continuation(
                        anchor_base=i,
                        n_content=len(content),
                        content=content,
                        K=K,
                        eos_id=eos_id,
                        blank_id=blank_id,
                        cut_prob=cut_prob,
                        rng=rng,
                    )
                    anchors.extend(chunk_anchors)
                    targets.extend(chunk_targets)
                else:
                    # Legacy fixed-K scheme. The stream the heads emit is the
                    # content tokens followed by exactly one <|im_end|>, split
                    # into K-sized blocks; block g is anchored at position
                    # i + g*K (the hidden state that predicts the block's first
                    # token under standard next-token semantics). Every anchor
                    # i + g*K is a real position in [i, j-1] (write_id or a
                    # content token), so the gathered hidden states always exist.
                    stream = content + [eos_id]
                    S = len(stream)
                    num_blocks = (S + K - 1) // K  # ceil(S / K), always >= 1
                    for g in range(num_blocks):
                        block = stream[g * K : g * K + K]
                        if len(block) < K:
                            block = block + [IGNORE_INDEX] * (K - len(block))
                        anchors.append(i + g * K)
                        targets.append(block)

                # Skip past this turn so write_id collisions inside content
                # (defensive — write_id is a special token so this is rare)
                # don't double-count.
                i = j + 1
            per_sample_anchors.append(anchors)
            per_sample_targets.append(targets)

        max_blocks = max((len(a) for a in per_sample_anchors), default=0)
        B = len(all_input_ids)
        if max_blocks == 0:
            return (
                torch.zeros((B, 0), dtype=torch.long),
                torch.zeros((B, 0, K), dtype=torch.long),
            )
        anchor_tensor = torch.full((B, max_blocks), -1, dtype=torch.long)
        target_tensor = torch.full((B, max_blocks, K), IGNORE_INDEX, dtype=torch.long)
        for b, (anchors, targets) in enumerate(zip(per_sample_anchors, per_sample_targets)):
            if not anchors:
                continue
            anchor_tensor[b, : len(anchors)] = torch.tensor(anchors, dtype=torch.long)
            target_tensor[b, : len(targets), :] = torch.tensor(targets, dtype=torch.long)
        return anchor_tensor, target_tensor

    @staticmethod
    def _segment_chunk_blocks_blank_continuation(
        anchor_base: int,
        n_content: int,
        content: List[int],
        K: int,
        eos_id: int,
        blank_id: int,
        cut_prob: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> tuple[list[int], list[list[int]]]:
        """Segment one chunk's content into blank-continuation blocks.

        Builds (anchors, targets) for a single chunk under the strict-<|im_end|>
        blank-continuation scheme:

          * each content block holds up to K *real* tokens;
          * a *full* K-token block carries no marker (fullness == continue);
          * a *partial* block (fewer than K real tokens — the natural last
            block) ends with a single ``blank_id`` "continue" marker;
          * with probability ``cut_prob`` a block instead *cuts*: a cut point is
            drawn uniformly from the block's non-first real positions
            (``1..take-1``), a ``blank_id`` is placed there, and the remaining
            tokens are pushed to the next block (re-checked independently, so a
            chunk may be cut multiple times);
          * the chunk always closes with a terminator block
            ``[eos_id, IGNORE...]`` whose slot 0 is <|im_end|>.

        Anchors advance by *real* tokens consumed: a block whose first real
        target is ``content[m]`` is anchored at ``anchor_base + m`` (so the
        terminator is anchored at ``anchor_base + n_content`` — the last real
        content token, or write_id when the chunk is empty). The blank marker is
        a synthetic head target only and never consumes a sequence position.

        Returns ``(anchors, targets)`` with each target block length-K,
        IGNORE-padded. Stripping the blank markers / terminator and concatenating
        the real tokens across blocks reconstructs ``content`` exactly,
        regardless of how the random cuts fall.
        """

        def _pad(block: list[int]) -> list[int]:
            if len(block) < K:
                return block + [IGNORE_INDEX] * (K - len(block))
            return block

        anchors: list[int] = []
        targets: list[list[int]] = []
        N = int(n_content)
        pos = 0
        while pos < N:
            take = min(K, N - pos)
            cut_j = None
            if rng is not None and cut_prob > 0.0 and take >= 2 and rng.random() < cut_prob:
                # Cut point uniform among the block's non-first positions.
                cut_j = rng.randint(1, take - 1)
            if cut_j is not None:
                anchors.append(anchor_base + pos)
                targets.append(_pad(content[pos : pos + cut_j] + [blank_id]))
                pos += cut_j
            elif take < K:
                # Natural partial (last) block → blank continuation marker.
                anchors.append(anchor_base + pos)
                targets.append(_pad(content[pos : pos + take] + [blank_id]))
                pos += take
            else:
                # Full K-token block → no marker.
                anchors.append(anchor_base + pos)
                targets.append(content[pos : pos + K])
                pos += K
        # Terminator block: slot-0 <|im_end|>, anchored at anchor_base + N.
        anchors.append(anchor_base + N)
        targets.append(_pad([eos_id]))
        return anchors, targets
