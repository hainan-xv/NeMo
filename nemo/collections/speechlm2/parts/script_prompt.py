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
"""Prompt-controlled SCRIPT: per-example decoding settings stated in the instruction.

A prompt-controlled model is trained with the chunk size, emission delay,
capitalization and punctuation DRAWN per example and DESCRIBED in the system
prompt, so one checkpoint serves every operating point and the caller picks one
at inference by asking for it in words.

This module owns two things and nothing else:

``apply_text_style``
    Turns the reference transcript into the styled target for one example
    (lowercase it, drop sentence punctuation, or both).

``render_control_prompt``
    Appends the natural-language description of the settings to the base
    instruction. Training and inference MUST build the prompt through this one
    function -- a wording difference between the two is silent
    out-of-distribution decoding, which is exactly the failure this centralises
    away.

The rendered form is natural language with frame units, e.g.::

    <base instruction> The audio is chunked every 14 frames with an emission
    delay of 3 frames. Use capitalization. Do not use punctuation.

Frames rather than seconds because every other latency quantity in the codebase
(``chunk_size``, ``num_delay_frames``, ``att_context_size``) is in encoder
frames, so there is no unit conversion to get wrong.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

# Hyphen and apostrophes survive only BETWEEN word characters, so "well-known"
# and "it's" keep their shape while a dangling dash or stray quote is dropped.
_KEEP_BETWEEN_WORD = "-'’"

# These survive only BETWEEN digits, which keeps "3.5", "1,200" and "9:30"
# intact instead of shattering them into separate tokens. Corpora like
# spgispeech and earnings22 are dense with figures, so this is not a corner case.
_KEEP_BETWEEN_DIGIT = ".,:"

# Everything punctuation-like that is a candidate for removal. Anything not
# listed here (letters, digits, whitespace) is never touched.
_PUNCT = set("""!"#$%&*+/;<=>?@\\^_`|~()[]{}…‘“”«»–—""") | set(_KEEP_BETWEEN_WORD) | set(_KEEP_BETWEEN_DIGIT)


def _strip_punctuation(body: str) -> str:
    """Drop sentence punctuation, keeping intra-word and intra-number marks."""
    out = []
    last = len(body) - 1
    for i, ch in enumerate(body):
        if ch not in _PUNCT:
            out.append(ch)
            continue
        prev = body[i - 1] if i > 0 else ""
        nxt = body[i + 1] if i < last else ""
        if ch in _KEEP_BETWEEN_WORD and prev.isalnum() and nxt.isalnum():
            out.append(ch)
        elif ch in _KEEP_BETWEEN_DIGIT and prev.isdigit() and nxt.isdigit():
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def apply_text_style(text: str, capitalization: bool, punctuation: bool) -> str:
    """Restyle ``text`` for the requested capitalization / punctuation setting.

    Leading whitespace is preserved. That matters more than it looks: the chunk
    text is sliced out of the transcript with ``preserve_leading_whitespace=True``
    so the first token of a chunk carries a word-start marker, and dropping the
    space would silently change the tokenization.

    Args:
        text: the chunk's text, as sliced from the reference transcript.
        capitalization: keep the original casing; lowercase everything when False.
        punctuation: keep punctuation; strip sentence marks when False.

    Returns:
        The restyled text. May be empty (a chunk whose entire content was a
        period becomes blank once punctuation is stripped), which callers must
        treat as a silent chunk.
    """
    if capitalization and punctuation:
        return text

    lead = text[: len(text) - len(text.lstrip())]
    body = text[len(lead) :]

    if not punctuation:
        body = _strip_punctuation(body)
        # Stripping marks leaves gaps ("end. Next" -> "end  Next"); collapse them
        # so the target text is not distinguishable by its spacing.
        body = re.sub(r"\s+", " ", body).strip()

    if not capitalization:
        body = body.lower()

    return lead + body if body else ""


@dataclass(frozen=True)
class ScriptControls:
    """One example's decoding settings.

    Attributes:
        chunk_size: frames per chunk. Shared across a batch -- the encoder's
            right context is ``chunk_size - 1``, so it cannot vary per example.
        num_delay_frames: emission delay in encoder frames.
        capitalization: whether targets keep their original casing.
        punctuation: whether targets keep sentence punctuation.
    """

    chunk_size: int
    num_delay_frames: int
    capitalization: bool
    punctuation: bool


def _frames(n: int) -> str:
    return f"{n} frame" + ("" if n == 1 else "s")


def render_control_prompt(base_prompt: str, controls: ScriptControls) -> str:
    """Append the natural-language description of ``controls`` to ``base_prompt``.

    The single source of truth for the control wording. Training builds the
    instruction with it and so must inference; see the module docstring.
    """
    delay = (
        "no emission delay"
        if controls.num_delay_frames == 0
        else f"an emission delay of {_frames(controls.num_delay_frames)}"
    )
    parts = [
        base_prompt.rstrip(),
        f"The audio is chunked every {_frames(controls.chunk_size)} with {delay}.",
        "Use capitalization." if controls.capitalization else "Do not use capitalization.",
        "Use punctuation." if controls.punctuation else "Do not use punctuation.",
    ]
    return " ".join(p for p in parts if p)


def sample_controls(rng, chunk_size: int, delay_candidates: List[int], cap_prob: float, punct_prob: float):
    """Draw one example's controls. ``chunk_size`` is passed in, not drawn here.

    Args:
        rng: a ``numpy.random.Generator``.
        chunk_size: the batch's chunk size.
        delay_candidates: delays to draw from, uniformly.
        cap_prob: probability that capitalization is on.
        punct_prob: probability that punctuation is on.
    """
    return ScriptControls(
        chunk_size=int(chunk_size),
        num_delay_frames=int(rng.choice(delay_candidates)),
        capitalization=bool(rng.random() < cap_prob),
        punctuation=bool(rng.random() < punct_prob),
    )


def resolve_delay_candidates(value: Optional[object], fallback: int) -> List[int]:
    """Normalize a config's delay setting into a non-empty list of frame counts."""
    if value is None:
        return [int(fallback)]
    if isinstance(value, int):
        return [int(value)]
    out = [int(v) for v in value]
    if not out:
        raise ValueError("delay_candidates is empty; give at least one delay in frames")
    if any(v < 0 for v in out):
        raise ValueError(f"delay_candidates must be non-negative frame counts, got {out}")
    return out
