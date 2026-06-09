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

"""Tokenizer wrapper for the 3-stream (spelling + capitalization + punctuation) TDT model.

Like :class:`MultiStreamCapTokenizer`, this wraps an ordinary BPE (SentencePiece) tokenizer so
the standard ASR data pipeline carries a *single* "product-space" id per token:

    combined = (punct * num_cap + cap) * num_spell + spell

where ``spell`` is the lowercased, punctuation-stripped sub-word id, ``cap`` is the per-token
capitalization class, and ``punct`` is the per-word word-ending punctuation class (attached to
the word's last sub-word). The model splits ``combined`` back into the three streams.
"""

from typing import List, Sequence, Union

from nemo.collections.asr.parts.utils.multistream_cap_punct_factorization import (
    DEFAULT_PUNCT_MARKS,
    combine_ids_cap_punct,
    decode_cap_punct,
    encode_cap_punct,
    split_id_cap_punct,
)
from nemo.collections.asr.parts.utils.multistream_factorization import NUM_CAP, split_id
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class MultiStreamCapPunctTokenizer(TokenizerSpec):
    """Wrap a base BPE tokenizer to emit product-space (spelling x cap x punct) ids."""

    def __init__(
        self,
        base_tokenizer: TokenizerSpec,
        num_cap: int = NUM_CAP,
        punct_marks: Sequence[str] = DEFAULT_PUNCT_MARKS,
    ):
        self.base = base_tokenizer
        self.num_cap = num_cap
        self.punct_marks = list(punct_marks)
        self.num_punct = len(self.punct_marks) + 1
        self.num_spell = base_tokenizer.vocab_size

    # ----- size / proxies -----
    @property
    def vocab_size(self) -> int:
        return self.num_punct * self.num_cap * self.num_spell

    @property
    def tokenizer(self):
        # Many call-sites reach for `.tokenizer` (the underlying SP model).
        return getattr(self.base, "tokenizer", self.base)

    def __getattr__(self, item):
        # Proxy any unknown attribute (e.g. pad_id, special tokens) to the base. Guard against
        # lookups before `base` is set (pickling/copy) and raise AttributeError (not KeyError).
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        try:
            base = self.__dict__["base"]
        except KeyError:
            raise AttributeError(item)
        return getattr(base, item)

    # ----- text <-> product ids -----
    def text_to_ids(self, text: str) -> List[int]:
        spell, cap, punct = encode_cap_punct(text, self.base, self.punct_marks)
        return [
            combine_ids_cap_punct(s, c, p, self.num_spell, self.num_cap)
            for s, c, p in zip(spell, cap, punct)
        ]

    def ids_to_text(self, ids) -> str:
        spell, cap, punct = [], [], []
        for cid in ids:
            s, c, p = split_id_cap_punct(int(cid), self.num_spell, self.num_cap)
            spell.append(s)
            cap.append(c)
            punct.append(p)
        return decode_cap_punct(spell, cap, punct, self.base, self.punct_marks)

    # ----- token-level helpers (operate on the spelling stream) -----
    def text_to_tokens(self, text: str):
        return self.base.text_to_tokens(text.lower())

    def tokens_to_text(self, tokens):
        return self.base.tokens_to_text(tokens)

    def ids_to_tokens(self, ids):
        spell = [split_id(int(i), self.num_spell)[0] for i in ids]
        return self.base.ids_to_tokens(spell)

    def tokens_to_ids(self, tokens: Union[str, List[str]]):
        # Maps to the spelling space (capitalization / punctuation are undefined here).
        return self.base.tokens_to_ids(tokens)
