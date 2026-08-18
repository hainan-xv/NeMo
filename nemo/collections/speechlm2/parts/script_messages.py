# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""SCRIPT-enhanced streaming-STT message builders.

The clean ``streaming_stt_dataset`` base ships a *vanilla* ``get_llm_messages_for_batch`` /
``get_llm_messages_for_sample`` that support neither the variable per-turn chunk
schedule (``chunk_sizes``) nor the word-length-dependent emission delay that SCRIPT
was developed against. Rather than modify the shared base, SCRIPT carries its own
copies of these builders (plus the delay helpers) here, so :class:`ScriptSTTDataset`
gets the full behaviour on top of an unmodified automodel base.

Only ``compute_word_spans`` and ``WordAlignment`` are reused from the base — both
exist unchanged there.
"""

import math
from typing import List, Optional

import numpy as np

from nemo.collections.speechlm2.data.streaming_stt_dataset import compute_word_spans
from nemo.collections.speechlm2.parts.alignments import WordAlignment


def word_length_delay_frames(text: str) -> int:
    """Word-length-dependent emission delay (in encoder frames).

    Longer words carry more acoustic evidence, so they can be emitted with less
    delay. Test-of-concept handwritten rule (edit the thresholds here as needed):
        < 4 letters -> 3 frames; == 4 -> 2; == 5 -> 1; >= 6 -> 0.
    "Letters" = alphabetic characters only (punctuation / whitespace ignored).
    """
    n_letters = sum(1 for c in text if c.isalpha())
    if n_letters < 4:
        return 3
    if n_letters < 5:
        return 2
    if n_letters < 6:
        return 1
    return 0


def sample_word_length_delay_frames(
    text: str,
    rng: np.random.Generator,
    max_delay: int = 3,
    midpoint: float = 4.5,
    slope: float = 1.0,
) -> int:
    """Stochastic word-length-dependent emission delay (in encoder frames).

    Instead of a fixed schedule, the delay of each word is sampled from
        delay ~ Binomial(n=max_delay, p),   p = sigmoid((midpoint - n_letters) / slope)
    where ``n_letters`` is the alphabetic-character count of the word. ``p`` is a
    per-word "keep-delaying" probability that decreases smoothly with word length,
    so the *expected* delay ``max_delay * p`` is high for short words and low for
    long ones, while the full range {0, ..., max_delay} keeps non-zero mass for
    augmentation diversity. With midpoint=4.5, slope=1.0, max_delay=3 the mean
    delay is roughly: 2 letters -> 2.8, 3 -> 2.5, 4 -> 1.9, 5 -> 1.1, 6 -> 0.6,
    7 -> 0.2 frames (compare the fixed table's 3/2/1/0).

    delay >= 0 always, so a word is never emitted before its acoustic end, and
    because words are consumed in index order (see ``get_llm_messages_for_sample``)
    output order — i.e. token monotonicity — is preserved regardless of the draw.
    """
    n_letters = sum(1 for c in text if c.isalpha())
    p = 1.0 / (1.0 + math.exp((n_letters - midpoint) / slope))
    return int(rng.binomial(int(max_delay), p))


def _resolve_word_delay(
    word_text: str,
    num_delay_frames: int,
    use_word_length_delay: bool,
    stochastic: bool,
    rng: Optional[np.random.Generator],
    max_delay: int,
    midpoint: float,
    slope: float,
) -> int:
    """Pick the emission delay (in frames) for one word.

    Priority: word-length delay (stochastic if enabled and an RNG is available,
    otherwise the fixed table) when ``use_word_length_delay`` is set, else the
    scalar ``num_delay_frames``.
    """
    if not use_word_length_delay:
        return num_delay_frames
    if stochastic and rng is not None:
        return sample_word_length_delay_frames(word_text, rng, max_delay, midpoint, slope)
    return word_length_delay_frames(word_text)


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
    chunk_sizes: Optional[List[int]] = None,
    use_word_length_delay: bool = False,
    word_length_delay_stochastic: bool = False,
    word_delay_max: int = 3,
    word_delay_midpoint: float = 4.5,
    word_delay_slope: float = 1.0,
    delay_rng: Optional[np.random.Generator] = None,
    enable_flush: bool = False,
    flush_prob: float = 0.0,
    flush_rng: Optional[np.random.Generator] = None,
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
        messages.append({"role": "user", "content": audio_tag * num_frames})
        messages.append({"role": "assistant", "content": transcript if transcript is not None else blank_token})
        return messages

    # Pre-compute word character spans if transcript is provided.
    word_spans = compute_word_spans(alignments, transcript, preserve_leading_whitespace=True) if transcript else None

    if chunk_sizes is not None and chunk_size <= 0:
        raise ValueError("chunk_sizes can only be used with fixed chunking (chunk_size > 0)")
    if chunk_sizes is not None and any(size <= 0 for size in chunk_sizes):
        raise ValueError("all chunk_sizes must be positive")

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
            _delay = _resolve_word_delay(
                last_word.text,
                num_delay_frames,
                use_word_length_delay,
                word_length_delay_stochastic,
                delay_rng,
                word_delay_max,
                word_delay_midpoint,
                word_delay_slope,
            )
            group_end_frame = math.ceil(last_word.end_time / frame_length_in_secs) + _delay
            if K > 1:
                group_end_frame = ((group_end_frame + K - 1) // K) * K
            group_end_frame = min(group_end_frame, num_frames)
            n_frames_chunk = group_end_frame - prev_end_frame

            if n_frames_chunk > 0:
                messages.append({"role": "user", "content": audio_tag * n_frames_chunk})

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
            messages.append({"role": "user", "content": audio_tag * (num_frames - prev_end_frame)})
    else:
        # Fixed chunking, optionally with a variable per-turn schedule.
        if chunk_sizes is None:
            num_chunks = math.ceil(num_frames / chunk_size) if num_frames > 0 else 0
            sample_chunk_sizes = [chunk_size] * num_chunks
        else:
            sample_chunk_sizes = []
            covered_frames = 0
            for size in chunk_sizes:
                if covered_frames >= num_frames:
                    break
                sample_chunk_sizes.append(size)
                covered_frames += size
            if covered_frames < num_frames:
                raise ValueError(
                    f"chunk_sizes cover {covered_frames} frames, but the sample requires {num_frames}"
                )

        # Decide FLUSH chunks up front (no-op unless enable_flush). A flush chunk
        # emits every not-yet-emitted word whose AUDIO has ended by its boundary,
        # DISREGARDING the emission delay -- teaching the model to dump held words on
        # demand. The FINAL chunk always flushes (the end-of-audio drain that fixes
        # tail-drop); each non-final chunk flushes independently with prob flush_prob.
        # Empty flushes are allowed (they teach "flush with nothing pending -> emit
        # only <eot>"), which matches the always-on final-chunk flush at inference.
        n_sched_chunks = len(sample_chunk_sizes)
        flush_flags = [False] * n_sched_chunks
        if enable_flush and n_sched_chunks > 0:
            flush_flags[-1] = True
            if flush_prob > 0.0 and flush_rng is not None:
                for ci in range(n_sched_chunks - 1):
                    if float(flush_rng.random()) < flush_prob:
                        flush_flags[ci] = True

        word_idx = 0
        word_buffer: list[int] = []  # indices of words buffered for words_per_group grouping
        chunk_end_frame = 0
        for chunk_i, current_chunk_size in enumerate(sample_chunk_sizes):
            chunk_end_frame += current_chunk_size
            is_flush = flush_flags[chunk_i]

            # User turn: one audio tag per frame in the chunk
            messages.append({"role": "user", "content": audio_tag * current_chunk_size})

            # Collect words ready by this chunk. Normally a word is ready when
            # word_end + delay <= boundary; on a FLUSH chunk the delay is IGNORED, so
            # every word whose audio ended by the boundary is emitted now.
            while word_idx < len(alignments):
                word = alignments[word_idx]
                word_end_frame = math.ceil(word.end_time / frame_length_in_secs)
                if is_flush:
                    ready_frame = word_end_frame
                else:
                    _delay = _resolve_word_delay(
                        word.text,
                        num_delay_frames,
                        use_word_length_delay,
                        word_length_delay_stochastic,
                        delay_rng,
                        word_delay_max,
                        word_delay_midpoint,
                        word_delay_slope,
                    )
                    ready_frame = word_end_frame + _delay
                if ready_frame <= chunk_end_frame:
                    word_buffer.append(word_idx)
                    word_idx += 1
                else:
                    break

            # Emit when the group is full, on the last chunk, OR on a flush chunk
            # (force-emit; an empty flush still emits blank so the model learns the
            # empty-flush case). Flush turns are tagged so the packer inserts <flush>.
            is_last_chunk = chunk_i == n_sched_chunks - 1
            if word_buffer and (len(word_buffer) >= words_per_group or is_last_chunk or is_flush):
                if word_spans and transcript:
                    first_span = word_spans[word_buffer[0]]
                    last_span = word_spans[word_buffer[-1]]
                    if first_span is not None and last_span is not None:
                        content = transcript[first_span[0] : last_span[1]]
                    else:
                        content = " ".join(alignments[i].text for i in word_buffer)
                else:
                    content = " ".join(alignments[i].text for i in word_buffer)
                msg = {"role": "assistant", "content": content}
                if is_flush:
                    msg["flush"] = True
                messages.append(msg)
                word_buffer = []
            else:
                msg = {"role": "assistant", "content": blank_token}
                if is_flush:
                    msg["flush"] = True
                messages.append(msg)

        # Append any residual words that weren't emitted (e.g., due to delay pushing
        # them past the last chunk boundary, or alignment end_time > audio_duration).
        if word_idx < len(alignments):
            residual_indices = list(range(word_idx, len(alignments)))
            if word_spans and transcript:
                first_span = word_spans[residual_indices[0]]
                last_span = word_spans[residual_indices[-1]]
                if first_span is not None and last_span is not None:
                    content = transcript[first_span[0] : last_span[1]]
                else:
                    content = " ".join(alignments[i].text for i in residual_indices)
            else:
                content = " ".join(alignments[i].text for i in residual_indices)
            if messages[-1]["role"] == "assistant" and messages[-1]["content"] == blank_token:
                messages[-1]["content"] = content
            elif messages[-1]["role"] == "assistant":
                messages[-1]["content"] += " " + content
            else:
                messages.append({"role": "assistant", "content": content})

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
    chunk_sizes: Optional[List[int]] = None,
    use_word_length_delay: bool = False,
    word_length_delay_stochastic: bool = False,
    word_delay_max: int = 3,
    word_delay_midpoint: float = 4.5,
    word_delay_slope: float = 1.0,
    delay_rng: Optional[np.random.Generator] = None,
    enable_flush: bool = False,
    flush_prob: float = 0.0,
    flush_rng: Optional[np.random.Generator] = None,
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
                chunk_sizes=chunk_sizes,
                use_word_length_delay=use_word_length_delay,
                word_length_delay_stochastic=word_length_delay_stochastic,
                word_delay_max=word_delay_max,
                word_delay_midpoint=word_delay_midpoint,
                word_delay_slope=word_delay_slope,
                delay_rng=delay_rng,
                enable_flush=enable_flush,
                flush_prob=flush_prob,
                flush_rng=flush_rng,
            )
        )
    return batch_messages
