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
from typing import List, Optional

from nemo.collections.speechlm2.data.streaming_stt_dataset import compute_word_spans
from nemo.collections.speechlm2.parts.alignments import WordAlignment


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

    def _content_for(indices: List[int]) -> str:
        if word_spans and transcript:
            first, last = word_spans[indices[0]], word_spans[indices[-1]]
            if first is not None and last is not None:
                return transcript[first[0] : last[1]]
        return " ".join(alignments[i].text for i in indices)

    num_chunks = math.ceil(num_frames / chunk_size) if num_frames > 0 else 0

    word_idx = 0
    chunk_end_frame = 0
    for _ in range(num_chunks):
        chunk_end_frame += chunk_size
        messages.append({"role": "user", "content": audio_tag * chunk_size})

        # Collect every word that has become ready by this chunk's boundary.
        ready: List[int] = []
        while word_idx < len(alignments):
            word_end_frame = math.ceil(alignments[word_idx].end_time / frame_length_in_secs)
            if word_end_frame + num_delay_frames <= chunk_end_frame:
                ready.append(word_idx)
                word_idx += 1
            else:
                break

        messages.append({"role": "assistant", "content": _content_for(ready) if ready else blank_token})

    # Any words the delay pushed past the last chunk boundary (or whose alignment
    # end_time exceeds the clip duration) would otherwise be silently dropped,
    # showing up as deletions at the end of every utterance. Fold them into the
    # final assistant turn so the supervision stays lossless.
    if word_idx < len(alignments):
        residual = _content_for(list(range(word_idx, len(alignments))))
        if messages[-1]["role"] != "assistant":
            messages.append({"role": "assistant", "content": residual})
        elif messages[-1]["content"] == blank_token:
            messages[-1]["content"] = residual
        else:
            messages[-1]["content"] += " " + residual

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
) -> List[List[dict]]:
    """Per-sample :func:`get_llm_messages_for_sample` over a batch.

    ``system_prompt`` is a per-sample list (cuts may carry their own prompt).
    """
    n = len(audio_durations_secs)
    if len(system_prompt) != n:
        raise ValueError(f"system_prompt has {len(system_prompt)} entries but the batch has {n} samples")
    return [
        get_llm_messages_for_sample(
            system_role=system_role,
            system_prompt=system_prompt[i],
            audio_tag=audio_tag,
            blank_token=blank_token,
            chunk_size=chunk_size,
            num_delay_frames=num_delay_frames,
            audio_duration_secs=audio_durations_secs[i],
            frame_length_in_secs=frame_length_in_secs,
            alignments=alignments[i] if alignments is not None else None,
            transcript=transcripts[i] if transcripts is not None else None,
        )
        for i in range(n)
    ]
