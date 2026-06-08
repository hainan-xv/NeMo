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

"""Spelling / capitalization factorization utilities for the 2-stream TDT model.

The 2-stream TDT model emits, at every transducer step, two parallel labels:

* a **spelling** sub-word token, taken from a *lowercased* tokenization of the
  text, and
* a **capitalization** class describing how the casing of that sub-word should
  be restored.

Capitalization is modelled with ``NUM_CAP`` discrete per-token classes:

* ``ALL_LOWER``   - the sub-word is rendered as-is (already lowercase).
* ``ALL_UPPER``   - the whole sub-word is uppercased.
* ``FIRST_UPPER`` - only the first cased character of the sub-word is uppercased.
* ``OTHER``       - any other / mixed casing. This class is **lossy**: at decode
  time we cannot perfectly reconstruct the original casing, so we fall back to
  rendering the sub-word lowercased.

The capitalization label is computed **per sub-word token** (not per word). This
keeps encode/decode purely local to a token and makes word-spanning casing fall
out naturally: e.g. ``"NASA"`` -> tokens ``["▁na", "sa"]`` both labelled
``ALL_UPPER``; ``"Hello"`` -> ``["▁hel", "lo"]`` labelled
``[FIRST_UPPER, ALL_LOWER]``.

Two integer "spaces" are used, decoupled on purpose:

* **product space** (size ``num_cap * num_spell``) - a single integer
  ``combined = cap * num_spell + spell`` that the *prediction network* embeds and
  that the dataset/tokenizer pipeline carries as an ordinary token id.
* **sum space** (size ``num_cap + num_spell + 1(blank) + n_durations``) - the
  *joint* output / loss layout, partitioned into contiguous streams via
  ``dividers`` (see :func:`multistream_tdt_dividers`).
"""

from typing import List, Sequence, Tuple

# Capitalization classes (per sub-word token).
ALL_LOWER = 0
ALL_UPPER = 1
FIRST_UPPER = 2
OTHER = 3
NUM_CAP = 4

SPIECE_UNDERLINE = "\u2581"  # SentencePiece word-boundary marker (▁)


def token_cap_label(orig: str, low_core: str) -> int:
    """Return the capitalization class for one sub-word.

    Args:
        orig: the original-cased substring corresponding to this sub-word.
        low_core: the lowercased sub-word surface (what the tokenizer produced,
            minus any leading word-boundary marker).
    """
    if not any(c.isalpha() for c in orig):
        return ALL_LOWER
    # If the alignment drifted (e.g. tokenizer normalization changed characters)
    # we cannot trust `orig`; mark it OTHER rather than guess.
    if orig.lower() != low_core:
        return OTHER
    if orig == low_core:
        return ALL_LOWER
    if orig == orig.upper():
        return ALL_UPPER
    # First-cased-character-uppercase, everything else lowercase.
    first = next(i for i, c in enumerate(orig) if c.isalpha())
    if orig[first].isupper() and orig[:first] == low_core[:first] and orig[first + 1 :] == low_core[first + 1 :]:
        return FIRST_UPPER
    return OTHER


def apply_cap(core: str, cap: int) -> str:
    """Render a lowercased sub-word surface with the given capitalization class."""
    if cap == ALL_UPPER:
        return core.upper()
    if cap == FIRST_UPPER:
        for i, c in enumerate(core):
            if c.isalpha():
                return core[:i] + core[i].upper() + core[i + 1 :]
        return core
    # ALL_LOWER and OTHER (lossy) both render the surface unchanged.
    return core


def encode_capitalization(text: str, tokenizer) -> Tuple[List[int], List[int]]:
    """Factorize cased ``text`` into (spelling_ids, cap_ids).

    ``tokenizer`` must expose ``text_to_ids`` and ``ids_to_tokens`` (the standard
    NeMo ``TokenizerSpec`` BPE interface). Tokenization is performed on the
    lowercased text; the capitalization class of each produced sub-word is then
    recovered by aligning sub-word surfaces against the original cased string.

    Note: alignment assumes lowercasing is a length-preserving, 1:1 character
    mapping (true for ASCII / typical English ASR transcripts). For inputs where
    this does not hold, affected tokens are labelled ``OTHER`` (lossy) instead of
    crashing.
    """
    lower = text.lower()
    spell_ids = list(tokenizer.text_to_ids(lower))
    pieces = tokenizer.ids_to_tokens(spell_ids)

    cap_ids: List[int] = []
    pos = 0
    for piece in pieces:
        if piece.startswith(SPIECE_UNDERLINE):
            core = piece[len(SPIECE_UNDERLINE) :]
            while pos < len(text) and text[pos].isspace():
                pos += 1
        else:
            core = piece
        n = len(core)
        orig = text[pos : pos + n]
        pos += n
        cap_ids.append(token_cap_label(orig, core))
    return spell_ids, cap_ids


def decode_capitalization(spell_ids: Sequence[int], cap_ids: Sequence[int], tokenizer) -> str:
    """Inverse of :func:`encode_capitalization`: render cased text."""
    pieces = tokenizer.ids_to_tokens(list(spell_ids))
    out: List[str] = []
    for piece, cap in zip(pieces, cap_ids):
        if piece.startswith(SPIECE_UNDERLINE):
            out.append(" ")
            core = piece[len(SPIECE_UNDERLINE) :]
        else:
            core = piece
        out.append(apply_cap(core, int(cap)))
    return "".join(out).strip()


# --------------------------------------------------------------------------- #
# Integer space conversions
# --------------------------------------------------------------------------- #
def combine_ids(spell: int, cap: int, num_spell: int) -> int:
    """(spell, cap) -> single product-space id used by the prediction network."""
    return cap * num_spell + spell


def split_id(combined: int, num_spell: int) -> Tuple[int, int]:
    """product-space id -> (spell, cap)."""
    return combined % num_spell, combined // num_spell


def multistream_tdt_dividers(num_spell: int, num_cap: int = NUM_CAP) -> Tuple[List[int], int]:
    """Return ``(dividers, blank)`` for the joint/loss *sum* space.

    Layout (label part, then durations appended by the joint):
        [ cap(0..num_cap-1) | spell(num_cap..num_cap+num_spell-1) | blank ]
    so the spelling stream owns the (single, shared) blank as its last index, as
    required by :class:`MultistreamTDTLoss`.
    """
    dividers = [0, num_cap, num_cap + num_spell + 1]
    blank = num_cap + num_spell
    return dividers, blank


def factorize_combined_to_sum(combined, num_spell: int, num_cap: int = NUM_CAP):
    """Convert product-space target ids to factorized *sum*-space stream indices.

    Args:
        combined: tensor [..., ] (e.g. [B, U]) of product-space ids
            (``cap * num_spell + spell``).
        num_spell: spelling vocabulary size (without blank).
        num_cap: number of capitalization classes.

    Returns:
        tensor [..., 2] where ``[..., 0]`` is the cap stream index (in
        ``[0, num_cap)``) and ``[..., 1]`` is the spelling stream index (in
        ``[num_cap, num_cap + num_spell)``), i.e. absolute indices into the joint
        label part expected by :class:`MultistreamTDTLoss`.
    """
    import torch

    spell = combined % num_spell
    cap = combined // num_spell
    return torch.stack([cap, spell + num_cap], dim=-1)
