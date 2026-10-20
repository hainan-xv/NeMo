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


def provisional_text(s: str, dedupe_boundary: bool = True) -> str:
    """What the reader READS first, before any correction lands.

    Naive concatenation double-counts deferred words. When the slow front moves a
    word into the next block, the fast front paints it in block i, has it erased,
    and paints it again in block i+1 -- but the reader reads it ONCE. Counting it
    twice shows up as pure insertions: on an AMI probe it doubled the insertion
    count (13 -> 31) and overstated the first-paint WER by ~1.5 points.

    ``dedupe_boundary`` drops leading words of a block that merely repeat the
    previous block's deferred tail. Pass ``False`` for the raw concatenation.
    """
    blocks = parse(s)
    out: List[str] = []
    for i, b in enumerate(blocks):
        words = b.provisional.split()
        if dedupe_boundary and i and words:
            prev = blocks[i - 1]
            # Words the previous block SHOWED but did not commit were deferred to
            # this one; re-painting them is the same word, not a new one.
            # ALIGN to find them -- a word-count difference is not enough. When a
            # block goes 'to, uh, mute' -> 'to mute,' the slow front dropped a word
            # from the MIDDLE, and counting would wrongly treat 'mute' as deferred
            # and delete a legitimate repeat in the next block.
            deferred = _deferred_tail(prev)
            k = 0
            while k < len(deferred) and k < len(words) and _same_word(deferred[k], words[k]):
                k += 1
            words = words[k:]
        if words:
            out.append(" ".join(words))
    return " ".join(out).strip()


def _deferred_tail(block: "Block") -> List[str]:
    """Words the slow front moved to the NEXT block: a trailing deletion only.

    Anything else the slow front did (a substitution, a mid-span drop) is a
    correction, not a deferral, and must not license deleting words downstream.
    """
    import difflib

    p, f = block.provisional.split(), block.final.split()
    ops = difflib.SequenceMatcher(a=p, b=f, autojunk=False).get_opcodes()
    if not ops:
        return []
    tag, i1, i2, _, _ = ops[-1]
    return p[i1:i2] if tag == "delete" else []


def _same_word(a: str, b: str) -> bool:
    """Compare ignoring case and edge punctuation -- 'fingers,' repaints as 'fingers'."""
    strip = ".,;:!?\"'"
    return a.strip(strip).lower() == b.strip(strip).lower()


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

    # Encode ONCE per front and reuse it for every block. The encoder depends
    # only on the chunk size, so re-running it per block (2 passes x N blocks)
    # was pure waste -- 62 full encodes on a 31-block utterance instead of 2.
    frames_fast = model.encode_frames(audios, audio_lens, fast_chunk)
    frames_slow = model.encode_frames(audios, audio_lens, slow_chunk)
    n_frames = [int(f.shape[0]) for f in frames_slow]
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
            precomputed_frames=frames_fast,
            **generate_kwargs,
        )
        auth = model.generate(
            audios,
            audio_lens,
            system_prompt=system_prompt,
            chunk_size_override=slow_chunk,
            history_text=slow_hist_text,
            frame_range=(lo, hi),
            precomputed_frames=frames_slow,
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


# --------------------------------------------------------------------- display


def display_frames(s: str) -> List[Tuple[str, str]]:
    """What the screen shows, step by step, as ``(committed, provisional)``.

    Two frames per block: the provisional text being painted, then the moment the
    slow front replaces it. Keeping this separate from any escape-code rendering
    is what makes the live view testable -- the renderers below are then just
    printing.
    """
    frames: List[Tuple[str, str]] = []
    committed: List[str] = []
    for b in parse(s):
        frames.append((_join(committed), b.provisional))
        if b.final:
            committed.append(b.final)
        frames.append((_join(committed), ""))
    return frames


def render_stream(s: str, out=None, delay: float = 0.0, block_seconds: float = 0.0, width: int = 100) -> None:
    """ONE-STREAM view: words appear, then the temporary tail is deleted and replaced.

    The provisional tail is erased with backspaces, exactly as described: what was
    tentatively shown is rubbed out and the authoritative text written over it. A
    terminal that has wrapped the line cannot be un-wrapped by backspacing, so
    long transcripts are re-flowed onto a fresh line instead of corrupting the
    display.

    Args:
        delay: seconds to pause between steps. ``block_seconds`` sets it from the
            real block duration so playback runs at the speed a listener would
            have experienced.
        width: column at which to break to a fresh line before writing text that
            will later need erasing.
    """
    import sys
    import time

    out = out or sys.stdout
    pause = block_seconds / 2.0 if block_seconds else delay
    printed = 0  # chars of COMMITTED text already written (never erased)
    col = 0  # current screen column
    shown = ""  # provisional tail currently on screen, erasable
    for committed, provisional in display_frames(s):
        if shown:  # rub out the tentative text before anything replaces it
            out.write("\b" * len(shown) + " " * len(shown) + "\b" * len(shown))
            col -= len(shown)
            shown = ""
        tail = committed[printed:]
        if tail:
            out.write(tail)
            printed = len(committed)
            col += len(tail)
        if provisional:
            # A wrapped line cannot be un-wrapped by backspacing, so move to a
            # fresh line BEFORE writing anything that will have to be erased.
            if col + len(provisional) + 1 > width:
                out.write("\n")
                col = 0
            piece = (" " if col and not provisional.startswith(" ") else "") + provisional
            out.write(piece)
            shown = piece
            col += len(piece)
        out.flush()
        if pause:
            time.sleep(pause)
    out.write("\n")
    out.flush()


def render_dual(s: str, width: int = 100) -> str:
    """TWO-STREAM view: the fast front on one line, the slow front beneath it.

    Blocks are column-aligned so the two fronts can be read against each other,
    and rows are wrapped to ``width``. Corrected blocks are marked, since the
    point of the view is to see where the second front earned its keep.
    """
    blocks = parse(s)
    if not blocks:
        return ""
    sep = " | "
    rows: List[List[Block]] = [[]]
    used = 0
    for b in blocks:
        w = max(len(b.provisional), len(b.final))
        if used and used + len(sep) + w > width:
            rows.append([])
            used = 0
        rows[-1].append(b)
        used += (len(sep) if used else 0) + w
    lines: List[str] = []
    for row in rows:
        widths = [max(len(b.provisional), len(b.final)) for b in row]
        fast = sep.join(b.provisional.ljust(w) for b, w in zip(row, widths))
        slow = sep.join(b.final.ljust(w) for b, w in zip(row, widths))
        lines += [f"  fast  {fast.rstrip()}", f"  slow  {slow.rstrip()}"]
        if any(b.corrected for b in row):
            # Only when something actually changed -- a row of bare separators
            # under every agreeing block is noise that hides the real marks.
            mark = sep.join(("^" * w if b.corrected else " " * w) for b, w in zip(row, widths))
            lines.append(f"        {mark.rstrip()}")
        lines.append("")
    return "\n".join(lines).rstrip()


def strike_markup(provisional: str, final: str) -> List[Tuple[str, bool]]:
    """Word-level diff as ``(text, struck)`` segments.

    A minimal diff rather than "strike the whole block and reprint": in
    ``quietude for`` -> ``disquietude for`` only ``quietude`` was wrong, and
    striking ``for`` as well would misreport what the slow front actually
    changed. Words the slow front ADDS are plain (nothing was wrong with them);
    words it drops or replaces are struck.
    """
    import difflib

    p, f = provisional.split(), final.split()
    segs: List[Tuple[str, bool]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=p, b=f, autojunk=False).get_opcodes():
        if tag == "equal":
            segs.append((" ".join(p[i1:i2]), False))
        elif tag == "delete":
            segs.append((" ".join(p[i1:i2]), True))
        elif tag == "insert":
            segs.append((" ".join(f[j1:j2]), False))
        else:  # replace: show what was wrong, then what replaced it
            segs.append((" ".join(p[i1:i2]), True))
            segs.append((" ".join(f[j1:j2]), False))
    return [(t, k) for t, k in segs if t]


STRIKE_ON = "\033[9;2m"  # strikethrough + dim
STRIKE_OFF = "\033[0m"


def render_strikeout(s: str, ansi: bool = True) -> str:
    """ONE-STREAM view that keeps its mistakes visible, crossed out.

    Nothing is erased: the reader sees the whole self-correction history inline,
    which is what you want for inspecting behaviour rather than for a live
    caption. ``ansi=False`` emits ``~~word~~`` so the output survives a pipe into
    a file or a diff.
    """
    out: List[str] = []
    for b in parse(s):
        for text, struck in strike_markup(b.provisional, b.final):
            if struck:
                out.append(f"{STRIKE_ON}{text}{STRIKE_OFF}" if ansi else f"~~{text}~~")
            else:
                out.append(text)
    return " ".join(out)
