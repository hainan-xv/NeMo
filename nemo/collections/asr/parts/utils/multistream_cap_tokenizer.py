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

"""Tokenizer wrapper for the 2-stream (spelling + capitalization) TDT model.

This wraps an ordinary BPE (SentencePiece) tokenizer and makes the standard ASR
data pipeline carry a *single* "product-space" id per token:

    combined = cap * num_spell + spell

where ``spell`` is the id of the **lowercased** sub-word and ``cap`` is the
per-token capitalization class. No dataset changes are needed: the dataset still
calls ``text_to_ids`` / ``ids_to_text`` and gets/produces a flat id sequence.

The model is responsible for splitting ``combined`` back into:
    * ``spell = combined % num_spell``  -> fed to the prediction network and used
      as the spelling-stream target, and
    * ``cap   = combined // num_spell`` -> used as the capitalization-stream
      target only (it is *not* fed back to the prediction network).
"""

from typing import List, Union

from nemo.collections.asr.parts.utils.multistream_factorization import (
    NUM_CAP,
    combine_ids,
    decode_capitalization,
    encode_capitalization,
    split_id,
)
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class MultiStreamCapTokenizer(TokenizerSpec):
    """Wrap a base BPE tokenizer to emit product-space (spelling x capitalization) ids."""

    def __init__(self, base_tokenizer: TokenizerSpec, num_cap: int = NUM_CAP):
        self.base = base_tokenizer
        self.num_cap = num_cap
        self.num_spell = base_tokenizer.vocab_size

    # ----- size / proxies -----
    @property
    def vocab_size(self) -> int:
        return self.num_cap * self.num_spell

    @property
    def tokenizer(self):
        # Many call-sites reach for `.tokenizer` (the underlying SP model).
        return getattr(self.base, "tokenizer", self.base)

    def __getattr__(self, item):
        # Proxy any unknown attribute (e.g. pad_id, special tokens) to the base.
        # __getattr__ is only called when normal lookup fails, so self.base etc.
        # are unaffected.
        return getattr(self.__dict__["base"], item)

    # ----- text <-> product ids -----
    def text_to_ids(self, text: str) -> List[int]:
        spell, cap = encode_capitalization(text, self.base)
        return [combine_ids(s, c, self.num_spell) for s, c in zip(spell, cap)]

    def ids_to_text(self, ids) -> str:
        spell, cap = [], []
        for cid in ids:
            s, c = split_id(int(cid), self.num_spell)
            spell.append(s)
            cap.append(c)
        return decode_capitalization(spell, cap, self.base)

    # ----- token-level helpers (operate on the spelling stream) -----
    def text_to_tokens(self, text: str):
        return self.base.text_to_tokens(text.lower())

    def tokens_to_text(self, tokens):
        return self.base.tokens_to_text(tokens)

    def ids_to_tokens(self, ids):
        spell = [split_id(int(i), self.num_spell)[0] for i in ids]
        return self.base.ids_to_tokens(spell)

    def tokens_to_ids(self, tokens: Union[str, List[str]]):
        # Maps to the spelling space (capitalization is undefined here).
        return self.base.tokens_to_ids(tokens)
