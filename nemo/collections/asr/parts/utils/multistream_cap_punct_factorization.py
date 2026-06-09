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

"""Spelling / capitalization / punctuation factorization for the 3-stream TDT model.

This extends the 2-stream (spelling + capitalization) factorization with a third,
**word-ending punctuation** stream. At every transducer step the model emits three
parallel labels:

* a **spelling** sub-word token (from a *lowercased, punctuation-stripped* tokenization),
* a **capitalization** class (per sub-word, see :mod:`multistream_factorization`), and
* a **punctuation** class (per *word*; attached to the **last** sub-word of the word).

Punctuation handling
--------------------
Only a configurable set of *word-ending* punctuation marks is modelled as a stream
(default: ``, . ? ! ; :``). Everything else - including within-word punctuation such as
hyphens/apostrophes and any mark not in the set - stays in the spelling stream as ordinary
sub-word tokens.

For each whitespace-delimited word we look at its maximal trailing run of non-alphanumeric
characters and find the **first** character in that run that belongs to the configured set:

* that character becomes the word's punctuation class,
* any characters *before* it (if any) are kept as ordinary spelling tokens, and
* any characters *after* it are **discarded** (a warning with the full utterance is logged).

A "word" that is entirely punctuation (e.g. a standalone ``.``) attaches its punctuation to
the **previous** word's last sub-word; if there is no previous word (or it is already
punctuated) the mark is dropped (with a warning).

The ``OTHER`` capitalization class and any discarded punctuation are *lossy*: the reference
text reconstructed by :func:`decode_cap_punct` will differ from the original in those spots.

Integer spaces (mirroring the 2-stream scheme, with punctuation prepended):

* **product space** (size ``num_punct * num_cap * num_spell``):
  ``combined = (punct * num_cap + cap) * num_spell + spell``.
* **sum space** (joint/loss layout): ``[ punct | cap | spell | blank | durations ]``; the
  spelling stream owns the (shared) blank as its last label index.
"""

from typing import List, Sequence, Tuple

from nemo.collections.asr.parts.utils.multistream_factorization import (
    NUM_CAP,
    SPIECE_UNDERLINE,
    apply_cap,
    encode_capitalization,
)
from nemo.utils import logging

# Punctuation classes: 0 is reserved for "no punctuation"; the configured marks take 1..N.
PUNCT_NONE = 0
DEFAULT_PUNCT_MARKS: Tuple[str, ...] = (",", ".", "?", "!", ";", ":")


def num_punct_classes(punct_marks: Sequence[str] = DEFAULT_PUNCT_MARKS) -> int:
    """Number of punctuation classes including ``PUNCT_NONE``."""
    return len(punct_marks) + 1


def split_word_punct(word: str, punct_marks: Sequence[str] = DEFAULT_PUNCT_MARKS) -> Tuple[str, int, bool]:
    """Split a single word into ``(core, punct_class, discarded)``.

    ``core`` is the spelling part (original casing, with the chosen trailing punctuation removed
    but any punctuation *before* it kept). ``punct_class`` is the 1-based class of the first
    in-set trailing punctuation (``PUNCT_NONE`` if none). ``discarded`` is True if any trailing
    punctuation *after* the chosen mark was dropped.
    """
    punct_set = set(punct_marks)
    # maximal trailing run of non-alphanumeric characters
    i = len(word)
    while i > 0 and not word[i - 1].isalnum():
        i -= 1
    head, trail = word[:i], word[i:]

    punct_class = PUNCT_NONE
    keep_before, discard_after = trail, ""
    for j, ch in enumerate(trail):
        if ch in punct_set:
            punct_class = punct_marks.index(ch) + 1
            keep_before, discard_after = trail[:j], trail[j + 1 :]
            break

    core = head + keep_before
    return core, punct_class, len(discard_after) > 0


def encode_cap_punct(
    text: str, tokenizer, punct_marks: Sequence[str] = DEFAULT_PUNCT_MARKS
) -> Tuple[List[int], List[int], List[int]]:
    """Factorize cased ``text`` into ``(spell_ids, cap_ids, punct_ids)`` (all the same length).

    ``tokenizer`` is the base BPE tokenizer (standard NeMo ``TokenizerSpec``).
    """
    words = text.split()
    core_words: List[str] = []
    core_punct: List[int] = []
    lossy = False

    for word in words:
        core, punct_class, discarded = split_word_punct(word, punct_marks)
        if discarded:
            lossy = True
        if core == "":
            # standalone punctuation: attach to the previous word's last sub-word if possible
            if punct_class != PUNCT_NONE:
                if core_punct and core_punct[-1] == PUNCT_NONE:
                    core_punct[-1] = punct_class
                else:
                    lossy = True  # nowhere to attach (or already punctuated) -> dropped
            continue
        core_words.append(core)
        core_punct.append(punct_class)

    clean_text = " ".join(core_words)
    spell_ids, cap_ids = encode_capitalization(clean_text, tokenizer)
    punct_ids = [PUNCT_NONE] * len(spell_ids)

    if spell_ids:
        pieces = tokenizer.ids_to_tokens(list(spell_ids))
        word_starts = [k for k, p in enumerate(pieces) if p.startswith(SPIECE_UNDERLINE)]
        if len(word_starts) == len(core_words):
            for w in range(len(core_words)):
                if core_punct[w] == PUNCT_NONE:
                    continue
                end = word_starts[w + 1] if w + 1 < len(word_starts) else len(pieces)
                punct_ids[end - 1] = core_punct[w]
        else:
            # sub-word/word grouping mismatch: cannot reliably place punctuation for this utterance
            lossy = True

    if lossy:
        logging.warning(
            "multistream cap+punct: dropped or could not place word-ending punctuation; "
            "kept the first in-set mark per word. utterance: %r",
            text,
        )
    return spell_ids, cap_ids, punct_ids


def decode_cap_punct(
    spell_ids: Sequence[int],
    cap_ids: Sequence[int],
    punct_ids: Sequence[int],
    tokenizer,
    punct_marks: Sequence[str] = DEFAULT_PUNCT_MARKS,
) -> str:
    """Inverse of :func:`encode_cap_punct`: render cased, punctuated text."""
    pieces = tokenizer.ids_to_tokens(list(spell_ids))
    out: List[str] = []
    for piece, cap, punct in zip(pieces, cap_ids, punct_ids):
        if piece.startswith(SPIECE_UNDERLINE):
            out.append(" ")
            core = piece[len(SPIECE_UNDERLINE) :]
        else:
            core = piece
        out.append(apply_cap(core, int(cap)))
        p = int(punct)
        if p != PUNCT_NONE:
            out.append(punct_marks[p - 1])
    return "".join(out).strip()


# --------------------------------------------------------------------------- #
# Integer space conversions
# --------------------------------------------------------------------------- #
def combine_ids_cap_punct(spell: int, cap: int, punct: int, num_spell: int, num_cap: int) -> int:
    """(spell, cap, punct) -> single product-space id used by the prediction network."""
    return (punct * num_cap + cap) * num_spell + spell


def split_id_cap_punct(combined: int, num_spell: int, num_cap: int) -> Tuple[int, int, int]:
    """product-space id -> (spell, cap, punct)."""
    spell = combined % num_spell
    rest = combined // num_spell
    cap = rest % num_cap
    punct = rest // num_cap
    return spell, cap, punct


def cap_punct_dividers(num_spell: int, num_cap: int = NUM_CAP, num_punct: int = len(DEFAULT_PUNCT_MARKS) + 1):
    """Return ``(dividers, blank)`` for the joint/loss *sum* space.

    Layout (label part, then durations appended by the joint):
        [ punct(0..) | cap(..) | spell(..) | blank ]
    so the spelling stream owns the (single, shared) blank as its last index, as required by
    :class:`MultistreamTDTLoss`.
    """
    dividers = [0, num_punct, num_punct + num_cap, num_punct + num_cap + num_spell + 1]
    blank = num_punct + num_cap + num_spell
    return dividers, blank
