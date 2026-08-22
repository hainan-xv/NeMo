# Copyright 2022 The OpenAI team and The HuggingFace Team. All rights reserved.
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
"""Vendored Open ASR Leaderboard English normalizer.

Faithful port of ``EnglishTextNormalizer`` from
github.com/huggingface/open_asr_leaderboard/normalizer/normalizer.py (as of
2026-08). The heavy, well-tested number/spelling/symbol pieces are the canonical
Whisper implementations, which are reused verbatim from the installed
``whisper_normalizer`` package (the same one our training val_wer uses) instead
of being re-vendored. The __call__ pipeline order and the leaderboard's ADDITIONS
over stock Whisper are reproduced here:

  * an expanded disfluency ``ignore_patterns`` list,
  * compound-word joining applied BEFORE number normalization,
  * name-variant folding and acronym de-spacing applied AFTER spelling.
"""
import re
import unicodedata

# Reuse ONLY the heavy, canonical Whisper number normalizer from the installed
# package (its ctor is arg-free and stable across versions). The spelling class
# and symbol stripper are vendored locally below, because their signatures/behavior
# drift between whisper_normalizer releases (e.g. EnglishSpellingNormalizer taking
# no mapping arg in some versions), which would silently change scoring.
from whisper_normalizer.english import EnglishNumberNormalizer

from .english_abbreviations import (
    english_compound_normalizer,
    english_name_normalizer,
    english_spelling_normalizer,
)

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep: str = "") -> str:
    """Replace markers/symbols/punctuation with a space and drop diacritics
    (Unicode category 'Mn' plus the manual ADDITIONAL_DIACRITICS mappings)."""

    def replace_character(char):
        if char in keep:
            return char
        elif char in ADDITIONAL_DIACRITICS:
            return ADDITIONAL_DIACRITICS[char]
        elif unicodedata.category(char) == "Mn":
            return ""
        elif unicodedata.category(char)[0] in "MSP":
            return " "
        return char

    return "".join(replace_character(c) for c in unicodedata.normalize("NFKD", s))


class EnglishSpellingNormalizer:
    """Apply British->American spelling mappings (Whisper "tysto" list)."""

    def __init__(self, english_spelling_mapping):
        self.mapping = english_spelling_mapping

    def __call__(self, s: str) -> str:
        return " ".join(self.mapping.get(word, word) for word in s.split())


class EnglishAcronymNormalizer:
    """Collapse runs of single-character tokens into one word.

    Normalizes acronym spacing so spaced and joined forms match:
      "b b c" -> "bbc", "5 g" -> "5g".
    Lone single-char words between multi-char words are left untouched
    ("a big cat" stays "a big cat"). Runs containing the common words "a"/"i"
    need 3+ tokens to collapse; otherwise 2+ is enough.
    """

    def __call__(self, s: str) -> str:
        words = s.split()
        result = []
        i = 0
        while i < len(words):
            if len(words[i]) == 1 and words[i].isalnum():
                run = [words[i]]
                j = i + 1
                while j < len(words) and len(words[j]) == 1 and words[j].isalnum():
                    run.append(words[j])
                    j += 1
                has_common_word = any(c in ("a", "i") for c in run)
                min_run = 3 if has_common_word else 2
                if len(run) >= min_run:
                    result.append("".join(run))
                else:
                    result.extend(run)
                i = j
            else:
                result.append(words[i])
                i += 1
        return " ".join(result)


class EnglishNameNormalizer:
    """Fold common name spelling variants to a single canonical form."""

    def __init__(self, english_name_mapping=english_name_normalizer):
        self.mapping = english_name_mapping

    def __call__(self, s: str) -> str:
        return " ".join(self.mapping.get(word, word) for word in s.split())


class EnglishTextNormalizer:
    def __init__(self, english_spelling_mapping=english_spelling_normalizer):
        self.ignore_patterns = (
            r"\b(hmm|mm|mhm|mmm|uh|um|ah|aha|ahh|ahm|eh|ehehe|em|hm|huh|hum|mhum|uhm|umm|uhuh)\b"
        )
        self.replacers = {
            # common contractions
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            # contractions in titles/prefixes
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            # perfect tenses, ideally it should be any past participles, but it's harder..
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",  # "'s done" is ambiguous
            r"'s got\b": " has got",
            # general contractions
            r"n't\b": " not",
            r"'re\b": " are",
            r"\b(it|he|she|what|that|who|here|there|how|when|where|why|this)'s\b": r"\1 is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_spellings = EnglishSpellingNormalizer(english_spelling_mapping)
        self.standardize_names = EnglishNameNormalizer()
        self.standardize_acronyms = EnglishAcronymNormalizer()
        # Multi-word compound mappings (leaderboard-specific).
        self.compound_words = english_compound_normalizer

    def __call__(self, s: str) -> str:
        s = s.lower()

        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = re.sub(self.ignore_patterns, "", s)
        s = re.sub(r"\s+'", "'", s)  # standardize when there's a space before an apostrophe

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)  # remove periods not followed by numbers
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")  # keep some symbols for numerics

        # Normalize hardcoded compound words (e.g. "wi fi" -> "wifi" after hyphen removal)
        for pattern, replacement in self.compound_words.items():
            s = re.sub(pattern, replacement, s)

        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)
        s = self.standardize_names(s)
        s = self.standardize_acronyms(s)

        # now remove prefix/suffix symbols that are not preceded/followed by numbers
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space

        return s
