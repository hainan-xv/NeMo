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
"""Turn word alignments into SCRIPT's per-chunk supervision.

This module owns exactly one decision: **which words does each audio chunk
reveal?** Everything downstream (the packed layout in
:mod:`nemo.collections.speechlm2.parts.script`) just consumes that assignment.

The rule is a single line — a word belongs to the first chunk whose boundary
covers the word's end plus the emission delay::

    ready_frame = ceil(word.end_time / frame_length) + num_delay_frames
    word k belongs to the first chunk with chunk_end_frame >= ready_frame

``num_delay_frames`` is the model's latency knob. A larger delay lets a word be
emitted with more right-context audio available, at the cost of emitting it
later. Note the interaction with ``audio_history_chunks``: a delayed word is
emitted from a LATER chunk, so unless the branch's audio window reaches back
over previous chunks, that word's acoustics are no longer visible at the moment
it must be predicted.

Output is the standard chat ``messages`` list of alternating user (audio) and
assistant (words) turns, which
:meth:`~nemo.collections.speechlm2.data.script_dataset.ScriptSTTDataset._messages_to_chunks`
parses into :class:`~nemo.collections.speechlm2.parts.script.ChunkSpec` objects.
"""

import math
import random
from typing import List, Optional, Sequence, Union

from nemo.collections.speechlm2.data.streaming_stt_dataset import compute_word_spans
from nemo.collections.speechlm2.parts.alignments import WordAlignment
from nemo.collections.speechlm2.parts.script_prompt import apply_text_style


def _per_sample(value, n: int, name: str) -> list:
    """Broadcast a scalar setting to ``n`` samples, or validate a per-sample list.

    Prompt-controlled training draws capitalization / punctuation / delay per
    example, so these arrive as lists; everything else passes a scalar.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{name} has {len(value)} entries but the batch has {n} samples")
        return list(value)
    return [value] * n


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
    capitalization: bool = True,
    punctuation: bool = True,
    word_delay_prob: float = 0.0,
    rng: Optional[random.Random] = None,
) -> List[dict]:
    """Build the alternating user/assistant turns for one utterance.

    The audio is split into ``ceil(num_frames / chunk_size)`` fixed chunks. Each
    chunk contributes a user turn holding ``chunk_size`` audio tags, followed by
    an assistant turn holding the words that became ready during that chunk (or
    ``blank_token`` when none did).

    For example, with ``chunk_size=2``, ``frame_length_in_secs=0.08``,
    ``num_delay_frames=0``, a 1 s clip and alignments
    ``[("Hello", 0.16, 0.48), ("World", 0.60, 0.80)]``::

        [{"role": "system",    "content": "<the system prompt>"},
         {"role": "user",      "content": "<audio><audio>"},   # frames 0-1
         {"role": "assistant", "content": "<blank>"},
         {"role": "user",      "content": "<audio><audio>"},   # frames 2-3
         {"role": "assistant", "content": "<blank>"},
         {"role": "user",      "content": "<audio><audio>"},   # frames 4-5, 0.32-0.48s
         {"role": "assistant", "content": "Hello"},            # "Hello" ends at 0.48s
         ...
         {"role": "user",      "content": "<audio><audio>"},   # frames 8-9, 0.64-0.80s
         {"role": "assistant", "content": "World"},
         ...]

    The final chunk may extend past ``audio_duration_secs`` because the frame
    count is ceiled to a whole number of chunks; the model zero-pads those
    trailing frames, and the decoder does the same so the two match.

    Args:
        system_role: role name for the instruction turn (usually ``"system"``).
        system_prompt: the instruction text.
        audio_tag: placeholder repeated once per audio frame in a user turn.
        blank_token: assistant content for a chunk that reveals no words. May be
            ``""`` (the "no-blank" setup), in which case a silent chunk simply
            has empty assistant content and the branch only predicts ``<eot>``.
        chunk_size: frames per chunk. Must be positive — SCRIPT is fixed-chunking
            only (no dynamic or offline mode).
        num_delay_frames: emission delay in encoder frames.
        audio_duration_secs: clip duration.
        frame_length_in_secs: seconds per encoder frame.
        alignments: word alignments for this utterance.
        transcript: the original transcript. When given, each chunk's text is
            sliced out of it by character span, which preserves the original
            punctuation and spacing rather than re-joining word strings.
        capitalization: keep the transcript's casing; lowercase when False.
        punctuation: keep sentence punctuation; strip it when False. A chunk
            whose text is punctuation only becomes a silent chunk.

    Returns:
        The messages list described above.
    """
    if chunk_size <= 0:
        raise ValueError(f"SCRIPT requires fixed chunking with chunk_size > 0, got {chunk_size}")

    messages = [{"role": system_role, "content": system_prompt}]
    num_frames = math.ceil(audio_duration_secs / frame_length_in_secs)
    if alignments is None:
        alignments = []

    # Character spans let us reproduce the transcript's own punctuation/spacing.
    word_spans = compute_word_spans(alignments, transcript, preserve_leading_whitespace=True) if transcript else None

    # Chunk texts must TILE the transcript, not be sliced from it independently.
    #
    # Slicing each chunk as transcript[first[0]:last[1]] silently drops every
    # character falling between one chunk's last span and the next chunk's first
    # span. Such gaps are routine: a word's span runs forward through TRAILING
    # punctuation but stops at whitespace, and the next word's span runs backward
    # through whitespace but stops at anything else -- so an OPENING quote,
    # bracket or dash sitting between two words belongs to no span at all.
    #
    # Seen in training (job 13090447, step 3000): the reference
    #   ... interrupted the corporal. "He might have made good, even
    # produced the target
    #   ... interrupted the corporal.He might have made good, even
    # losing both the space and the quote, because the aligner's word is `He`
    # while the transcript reads `"He`.
    #
    # The concatenated chunk texts ARE the transcript the model is trained to
    # emit, so a dropped character is a silent, permanent corruption of the
    # target that also teaches the model to fuse words across chunk boundaries.
    # Taking transcript[cursor:end] instead makes the chunks a partition: every
    # character appears exactly once and concatenation reproduces the transcript.
    text_cursor = 0

    def _content_for(indices: List[int]) -> str:
        nonlocal text_cursor
        text = None
        if word_spans and transcript:
            # The LAST located word, so a single unfound word does not discard
            # the whole group's punctuation by falling back to a bare join.
            end = None
            for i in reversed(indices):
                if word_spans[i] is not None:
                    end = word_spans[i][1]
                    break
            if end is not None and end >= text_cursor:
                text = transcript[text_cursor:end]
                text_cursor = end
        if text is None and not (word_spans and transcript):
            # No transcript at all: the aligner's words are the ONLY source of
            # text, so use them. Nothing can be duplicated here because there is
            # no transcript for a later chunk to slice from.
            text = " ".join(alignments[i].text for i in indices)
        elif text is None:
            # There IS a transcript, but this group's words were not located in
            # it. Emit nothing, and leave the cursor where it is.
            #
            # Falling back to the aligner's own word forms here DUPLICATES text.
            # The cursor does not advance, so the next group emits
            # transcript[cursor:...] spanning this same region -- which therefore
            # appears twice, once in the aligner's spelling and once in the
            # transcript's. Seen in training as
            #
            #   ref: ... his wait of forty-eight hours ...
            #   tgt: ... his wait offortyeight forty-eight hours ...
            #
            # because the aligner reports "fortyeight" for a hyphenated
            # "forty-eight", so find() fails; when that word lands alone in a
            # chunk there is no located neighbour to rescue the slice. The model
            # is then trained to emit the word twice, in two different spellings.
            #
            # Emitting nothing makes this chunk silent and lets the NEXT chunk's
            # slice cover the text, since the cursor did not move. Nothing is
            # lost -- those words are simply emitted one chunk later.
            return ""
        # Restyle AFTER slicing: the character spans index into the original
        # transcript, so stripping punctuation first would invalidate them.
        return apply_text_style(text, capitalization, punctuation)

    num_chunks = math.ceil(num_frames / chunk_size) if num_frames > 0 else 0

    # PASS 1: which words each chunk reveals under the frame-delay rule.
    chunk_words: List[List[int]] = []
    word_idx = 0
    chunk_end_frame = 0
    for _ in range(num_chunks):
        chunk_end_frame += chunk_size
        ready: List[int] = []
        while word_idx < len(alignments):
            word_end_frame = math.ceil(alignments[word_idx].end_time / frame_length_in_secs)
            if word_end_frame + num_delay_frames <= chunk_end_frame:
                ready.append(word_idx)
                word_idx += 1
            else:
                break
        chunk_words.append(ready)

    # PASS 2: stochastic WORD-level delay, applied on top of the frame delay.
    #
    # For each non-empty chunk, hold back its last k words for the next chunk,
    # with P(k >= j) = word_delay_prob^j. At the default 0.5 that leaves a chunk
    # alone half the time, holds back one word a quarter of the time, two an
    # eighth, and so on -- one word held back on average.
    #
    # WHY. The frame delay applies one fixed latency to every word, so the model
    # sees exactly one emission time per word and is free to depend on it.
    # Randomising the boundary makes emission timing a distribution instead of a
    # constant -- the transducer analogue of the alignment freedom that
    # marginalising over paths would provide, which single-path training gives up.
    #
    # A held-back word is FROZEN where it lands: only a chunk's OWN words are
    # eligible, never ones pushed in from the previous chunk. That caps the added
    # latency at exactly one chunk. Without it a word could slip arbitrarily far
    # and the tail of an utterance would drift.
    #
    # The final chunk has nowhere to push to, so its words are never delayed.
    if word_delay_prob > 0.0 and rng is not None and num_chunks > 1:
        pushed: List[List[int]] = [[] for _ in range(num_chunks)]
        for t in range(num_chunks - 1):
            own = chunk_words[t]
            if not own:
                continue
            k = 0
            while k < len(own) and rng.random() < word_delay_prob:
                k += 1
            if k:
                pushed[t + 1] = own[-k:]
                chunk_words[t] = own[:-k]
        # Prepended only AFTER the loop, so a pushed-in word is never itself
        # eligible. Pushed words precede the receiving chunk's own words in the
        # transcript, so every chunk's list stays in transcript order -- which is
        # what the cursor in _content_for relies on.
        for t in range(num_chunks):
            if pushed[t]:
                chunk_words[t] = pushed[t] + chunk_words[t]

    # PASS 3: emit the turns.
    for t in range(num_chunks):
        messages.append({"role": "user", "content": audio_tag * chunk_size})
        # Styling can empty a chunk whose text was punctuation only; that makes it
        # silent, exactly like a chunk that revealed no words at all.
        content = _content_for(chunk_words[t]) if chunk_words[t] else ""
        messages.append({"role": "assistant", "content": content if content.strip() else blank_token})

    # Any words the delay pushed past the last chunk boundary (or whose alignment
    # end_time exceeds the clip duration) would otherwise be silently dropped,
    # showing up as deletions at the end of every utterance. Fold them into the
    # final assistant turn so the supervision stays lossless.
    residual = _content_for(list(range(word_idx, len(alignments)))) if word_idx < len(alignments) else ""
    if residual.strip():
        if messages[-1]["role"] != "assistant":
            messages.append({"role": "assistant", "content": residual})
        elif messages[-1]["content"] == blank_token:
            messages[-1]["content"] = residual
        else:
            # No separator: residual starts at the cursor, so it already carries
            # whatever whitespace the transcript has there. Adding one would
            # double it.
            messages[-1]["content"] += residual

    # Anything after the last aligned word's span -- trailing punctuation that no
    # word's span reached, e.g. a dash separated from the final word by a space.
    # Without this the chunks still would not partition the transcript, and the
    # target would end short of the reference.
    if transcript and text_cursor < len(transcript):
        tail = apply_text_style(transcript[text_cursor:], capitalization, punctuation)
        if tail.strip():
            if messages and messages[-1]["role"] == "assistant" and messages[-1]["content"] != blank_token:
                messages[-1]["content"] += tail
            elif messages and messages[-1]["role"] == "assistant":
                messages[-1]["content"] = tail
            else:
                messages.append({"role": "assistant", "content": tail})

    return messages


def get_llm_messages_for_batch(
    system_role: str,
    system_prompt: List[str],
    audio_tag: str,
    blank_token: str,
    chunk_size: int,
    num_delay_frames: Union[int, Sequence[int]],
    audio_durations_secs: List[float],
    frame_length_in_secs: float,
    alignments: Optional[List[List[WordAlignment]]] = None,
    transcripts: Optional[List[str]] = None,
    capitalization: Union[bool, Sequence[bool]] = True,
    punctuation: Union[bool, Sequence[bool]] = True,
    word_delay_prob: float = 0.0,
    rng: Optional[random.Random] = None,
) -> List[List[dict]]:
    """Per-sample :func:`get_llm_messages_for_sample` over a batch.

    ``system_prompt`` is a per-sample list (cuts may carry their own prompt).

    ``num_delay_frames``, ``capitalization`` and ``punctuation`` accept either a
    scalar (the whole batch shares it) or a per-sample sequence -- prompt-controlled
    training draws them independently per example. ``chunk_size`` is deliberately
    scalar-only: the encoder's right context is ``chunk_size - 1``, so it is a
    property of the batch, not of an example.
    """
    n = len(audio_durations_secs)
    if len(system_prompt) != n:
        raise ValueError(f"system_prompt has {len(system_prompt)} entries but the batch has {n} samples")
    delays = _per_sample(num_delay_frames, n, "num_delay_frames")
    caps = _per_sample(capitalization, n, "capitalization")
    puncts = _per_sample(punctuation, n, "punctuation")
    return [
        get_llm_messages_for_sample(
            system_role=system_role,
            system_prompt=system_prompt[i],
            audio_tag=audio_tag,
            blank_token=blank_token,
            chunk_size=chunk_size,
            num_delay_frames=delays[i],
            audio_duration_secs=audio_durations_secs[i],
            frame_length_in_secs=frame_length_in_secs,
            alignments=alignments[i] if alignments is not None else None,
            transcript=transcripts[i] if transcripts is not None else None,
            capitalization=caps[i],
            punctuation=puncts[i],
            word_delay_prob=word_delay_prob,
            rng=rng,
        )
        for i in range(n)
    ]
