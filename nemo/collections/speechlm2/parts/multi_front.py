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
"""Multi-front (self-correcting) streaming decode for prompt-controlled SCRIPT.

A prompt-controlled model serves every chunk size from ONE set of weights, so two
"fronts" are the same model under two different control prompts:

    fast front (chunk 2, 0.16 s)   -- emitted immediately, PROVISIONAL
    slow front (chunk 14, 1.12 s)  -- emitted a block later, AUTHORITATIVE

Because the slow chunk is an exact multiple of the fast one (14 = 7 x 2), the
14-frame block is the natural unit: during a block the user sees the fast front's
words appear, and when the block closes the slow front replaces them.

Serialisation
-------------
One block is one ``<temp_start>provisional<temp_end>authoritative`` pair, and an
utterance is their concatenation::

    <temp_start>the quick brown fax<temp_end>the quick brown fox<temp_start>jumps ovr<temp_end>jumps over

Everything between the tags is what was shown during the block; everything after
``<temp_end>`` is what replaced it. The final transcript is the concatenation of
the authoritative spans, and a no-op correction is visible by inspection
(``<temp_start>fox<temp_end>fox``) rather than needing its own tag.

Timing is implicit: block ``i`` spans frames ``[i*slow, (i+1)*slow)``. The format
deliberately drops the per-fast-chunk boundaries inside a block, so within-block
emission times are NOT recoverable from the string -- ask the decoder for the
per-chunk trace if you need them.

Resync
------
At each block boundary the fast front's history is reset to the committed
(authoritative) text. It therefore never extrapolates more than one block from
corrected text, which both stops its errors compounding and keeps the splice
coherent. ``resync=False`` gives the independent-history baseline for comparison.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

TEMP_START = "<temp_start>"
TEMP_END = "<temp_end>"


@dataclass(frozen=True)
class Block:
    """One slow-chunk block: what was shown, and what replaced it."""

    provisional: str  # fast-front text displayed during the block
    final: str  # slow-front text that replaced it

    @property
    def corrected(self) -> bool:
        """Did the slow front actually change anything?"""
        return self.provisional.split() != self.final.split()


def serialize(blocks: Sequence[Block]) -> str:
    """Blocks -> the tagged string."""
    return "".join(f"{TEMP_START}{b.provisional}{TEMP_END}{b.final}" for b in blocks)


def parse(s: str) -> List[Block]:
    """The tagged string -> blocks. Inverse of :func:`serialize`.

    Raises on a malformed string rather than silently returning a short list: a
    truncated trace that parses to "fewer blocks" would quietly understate every
    metric computed from it.
    """
    blocks: List[Block] = []
    pos = 0
    n = len(s)
    while pos < n:
        if not s.startswith(TEMP_START, pos):
            raise ValueError(f"expected {TEMP_START!r} at offset {pos} in {s!r}")
        pos += len(TEMP_START)
        end = s.find(TEMP_END, pos)
        if end < 0:
            raise ValueError(f"unterminated {TEMP_START!r} at offset {pos} in {s!r}")
        provisional = s[pos:end]
        pos = end + len(TEMP_END)
        nxt = s.find(TEMP_START, pos)
        final = s[pos:] if nxt < 0 else s[pos:nxt]
        pos = n if nxt < 0 else nxt
        blocks.append(Block(provisional=provisional, final=final))
    return blocks


def final_text(s: str) -> str:
    """The transcript a user is left with: the authoritative spans, joined."""
    return " ".join(b.final for b in parse(s) if b.final).strip()


def provisional_text(s: str) -> str:
    """What a user would have read if nothing was ever corrected.

    This is the fast front's transcript, and scoring it against the reference
    gives the quality of what is shown FIRST -- the number that matters for how
    often a reader sees something wrong, as opposed to what they end up with.
    """
    return " ".join(b.provisional for b in parse(s) if b.provisional).strip()


@dataclass(frozen=True)
class RevisionStats:
    """How much the slow front actually changed."""

    blocks: int
    corrected_blocks: int
    provisional_words: int
    revised_words: int

    @property
    def block_correction_rate(self) -> float:
        return self.corrected_blocks / self.blocks if self.blocks else 0.0

    @property
    def word_revision_rate(self) -> float:
        return self.revised_words / self.provisional_words if self.provisional_words else 0.0


def revision_stats(s: str) -> RevisionStats:
    """Count what the second front bought.

    ``word_revision_rate`` counts provisional words NOT confirmed in place by the
    slow front -- measured as a common-prefix match, because a reader perceives a
    change at the first word that moves, and everything after it re-renders.
    """
    blocks = parse(s)
    corrected = sum(1 for b in blocks if b.corrected)
    prov_words = revised = 0
    for b in blocks:
        p, f = b.provisional.split(), b.final.split()
        prov_words += len(p)
        keep = 0
        for pw, fw in zip(p, f):
            if pw != fw:
                break
            keep += 1
        revised += len(p) - keep
    return RevisionStats(
        blocks=len(blocks),
        corrected_blocks=corrected,
        provisional_words=prov_words,
        revised_words=revised,
    )


def multi_front_decode(
    model,
    audios,
    audio_lens,
    system_prompt: str = "Transcribe the audio into text.",
    fast_chunk: int = 2,
    slow_chunk: int = 14,
    resync: bool = True,
    **generate_kwargs,
) -> List[str]:
    """Decode a batch with two fronts and return one tagged string per utterance.

    The encoder runs ONCE per front over the whole utterance (so left context is
    never truncated); only the decode window moves block by block.

    Args:
        model: a :class:`ScriptSTTModel`. Must be prompt-controlled, since the
            two fronts differ only by the chunk size stated in the prompt.
        fast_chunk / slow_chunk: the two operating points. ``slow_chunk`` must be
            an exact multiple of ``fast_chunk`` -- the block structure, and hence
            the serialisation, is only well defined when it divides evenly.
        resync: reset the fast front's history to the committed text at each
            block boundary. ``False`` lets it carry its own history throughout,
            which is the baseline the resync design is meant to beat.
    """
    if fast_chunk <= 0 or slow_chunk <= 0:
        raise ValueError(f"chunk sizes must be positive, got fast={fast_chunk} slow={slow_chunk}")
    if slow_chunk % fast_chunk != 0:
        raise ValueError(
            f"slow_chunk ({slow_chunk}) must be an exact multiple of fast_chunk ({fast_chunk}); "
            "otherwise a block does not contain a whole number of fast chunks and the "
            "<temp_start>/<temp_end> spans would not line up with either front."
        )
    if slow_chunk == fast_chunk:
        raise ValueError("fast and slow fronts must differ; identical chunk sizes give nothing to correct")

    n_frames = _frame_counts(model, audios, audio_lens, slow_chunk)
    batch = len(n_frames)
    committed: List[List[str]] = [[] for _ in range(batch)]
    fast_hist: List[List[str]] = [[] for _ in range(batch)]
    blocks: List[List[Block]] = [[] for _ in range(batch)]

    n_blocks = max((-(-f // slow_chunk) for f in n_frames), default=0)
    for i in range(n_blocks):
        lo, hi = i * slow_chunk, (i + 1) * slow_chunk
        # History each front is conditioned on for this block. With resync the
        # fast front starts from what the slow front committed; without it, from
        # its own running transcript.
        slow_hist_text = [_join(c) for c in committed]
        fast_hist_text = slow_hist_text if resync else [_join(h) for h in fast_hist]

        prov = model.generate(
            audios,
            audio_lens,
            system_prompt=system_prompt,
            chunk_size_override=fast_chunk,
            history_text=fast_hist_text,
            frame_range=(lo, hi),
            **generate_kwargs,
        )
        auth = model.generate(
            audios,
            audio_lens,
            system_prompt=system_prompt,
            chunk_size_override=slow_chunk,
            history_text=slow_hist_text,
            frame_range=(lo, hi),
            **generate_kwargs,
        )
        for b in range(batch):
            if lo >= n_frames[b]:
                continue  # this utterance already ended; emit nothing for it
            blocks[b].append(Block(provisional=prov[b].strip(), final=auth[b].strip()))
            if auth[b].strip():
                committed[b].append(auth[b].strip())
            if prov[b].strip():
                fast_hist[b].append(prov[b].strip())
    return [serialize(bl) for bl in blocks]


def _join(parts: Sequence[str]) -> str:
    return " ".join(p for p in parts if p).strip()


def _frame_counts(model, audios, audio_lens, chunk: int) -> List[int]:
    """Encoder frame count per utterance, as the decoder will see it."""
    frames = model.encode_frames(audios, audio_lens, chunk)
    return [int(f.shape[0]) for f in frames]
