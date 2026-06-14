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

"""A HuggingFace (byte-level BPE) tokenizer restricted to a subset of its vocabulary.

Motivation
----------
LLM tokenizers such as Qwen are multilingual with very large vocabularies (~151k). For an
English-only ASR system most of those entries are never used, yet they dominate the size of the
joint output projection / decoder embedding and (for CHAT) the ``[B, T, U, V]`` joint tensor.

This wrapper keeps a chosen subset of the original token ids (e.g. the tokens that actually appear
when tokenizing an English corpus) and exposes a *compact*, contiguous id space of size
``len(kept)`` to the model, while still using the original tokenizer's merge rules for encoding.
Coverage is guaranteed because the 256 base byte-level tokens are always retained: any token that
is not in the kept subset is transparently decomposed into its constituent byte tokens, so any
string remains representable (no hard failures on unseen words / scripts).

The mapping is reversible: ``compact_id <-> original (e.g. Qwen) id`` is available via
``compact_to_original`` / ``original_to_compact`` so the original LLM ids can always be recovered
(useful for downstream LLM fusion).
"""

import json
import os
from typing import Dict, List, Optional, Sequence

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.utils import logging

__all__ = ['RestrictedAutoTokenizer', 'bytes_to_unicode']


def bytes_to_unicode() -> Dict[int, str]:
    """GPT-2 / Qwen byte-level BPE byte<->unicode table (the 256 base "byte" characters)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _load_kept_ids(restrict_vocab_file: str) -> List[int]:
    """Load the kept original-token-id list from a JSON ({"kept_ids": [...]}) / JSON list / txt file."""
    with open(restrict_vocab_file, 'r', encoding='utf-8') as f:
        if restrict_vocab_file.endswith('.json'):
            data = json.load(f)
            if isinstance(data, dict):
                ids = data.get('kept_ids', data.get('kept_qwen_ids', None))
                if ids is None:
                    raise ValueError(f"{restrict_vocab_file} JSON must contain 'kept_ids' (or 'kept_qwen_ids').")
            else:
                ids = data
        else:
            ids = [int(line.strip()) for line in f if line.strip()]
    return [int(i) for i in ids]


class RestrictedAutoTokenizer(AutoTokenizer):
    """HuggingFace AutoTokenizer restricted to a compact subset of its vocabulary (byte fallback).

    Args:
        pretrained_model_name: HF hub name or local path (e.g. ``Qwen/Qwen3-1.7B``).
        restrict_vocab_file: path to the kept-id artifact produced by
            ``scripts/tokenizers/build_restricted_qwen_tokenizer.py`` (JSON with ``kept_ids``).
        kept_ids: alternatively pass the kept original ids directly (takes precedence over the file).
        Any other kwargs are forwarded to :class:`AutoTokenizer`.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        restrict_vocab_file: Optional[str] = None,
        kept_ids: Optional[Sequence[int]] = None,
        **auto_kwargs,
    ):
        super().__init__(pretrained_model_name, **auto_kwargs)

        if kept_ids is None:
            if not restrict_vocab_file:
                raise ValueError("RestrictedAutoTokenizer requires `restrict_vocab_file` or `kept_ids`.")
            kept_ids = _load_kept_ids(restrict_vocab_file)

        hf = self.tokenizer  # underlying HF tokenizer (encoding uses its merge rules)
        full_vocab = hf.get_vocab()  # token_str -> original id
        self._inv_full_vocab = {i: t for t, i in full_vocab.items()}

        # The 256 base byte tokens must always be kept so any string is representable via fallback.
        byte_chars = set(bytes_to_unicode().values())
        byte_token_ids = [full_vocab[c] for c in byte_chars if c in full_vocab]
        if len(byte_token_ids) != 256:
            logging.warning(
                f"RestrictedAutoTokenizer: found {len(byte_token_ids)}/256 base byte tokens in "
                f"{pretrained_model_name}; byte fallback may be incomplete."
            )

        special_ids = [i for i in self.tokenizer.all_special_ids if i is not None]

        kept = set(int(i) for i in kept_ids) | set(byte_token_ids) | set(special_ids)
        self.compact_to_original: List[int] = sorted(kept)
        self.original_to_compact: Dict[int, int] = {q: c for c, q in enumerate(self.compact_to_original)}
        self._byte_token_ids = set(byte_token_ids)
        self._fallback_cache: Dict[int, List[int]] = {}

        # Make the model's vocab-sizing (EncDecRNNTBPEModel reads
        # ``self.tokenizer.tokenizer.get_vocab()``) see the COMPACT vocabulary, in compact-id order,
        # while leaving the Rust-backed encode/decode of the underlying tokenizer untouched.
        compact_vocab = {self._inv_full_vocab[q]: c for c, q in enumerate(self.compact_to_original)}
        hf.get_vocab = lambda: dict(compact_vocab)  # noqa: E731 (instance shadow, intentional)

        logging.info(
            f"RestrictedAutoTokenizer('{pretrained_model_name}'): compact vocab = "
            f"{len(self.compact_to_original)} tokens (from {len(full_vocab)} original; "
            f"{len(byte_token_ids)} byte + {len(special_ids)} special always kept)."
        )

    # --- helpers -----------------------------------------------------------------------------
    def _original_to_byte_originals(self, oid: int) -> List[int]:
        """Decompose an out-of-subset original token id into its constituent byte-token ids."""
        if oid not in self._fallback_cache:
            surface = self._inv_full_vocab[oid]
            full_vocab = {t: i for i, t in self._inv_full_vocab.items()}
            self._fallback_cache[oid] = [full_vocab[ch] for ch in surface]
        return self._fallback_cache[oid]

    def _original_ids_to_compact(self, original_ids: Sequence[int]) -> List[int]:
        out: List[int] = []
        for oid in original_ids:
            c = self.original_to_compact.get(int(oid))
            if c is not None:
                out.append(c)
            else:
                for bid in self._original_to_byte_originals(int(oid)):
                    out.append(self.original_to_compact[bid])
        return out

    # --- overridden TokenizerSpec API (operates in COMPACT id space) ---------------------------
    @property
    def vocab_size(self):
        """Size of the compact (restricted) vocabulary."""
        return len(self.compact_to_original)

    def text_to_ids(self, text):
        """Tokenize ``text`` with the original merges, then map to compact ids (byte fallback)."""
        original_ids = self.tokenizer.encode(text, add_special_tokens=self.include_special_tokens)
        return self._original_ids_to_compact(original_ids)

    def ids_to_text(self, ids, remove_special_tokens=True):
        """Map compact ids back to original ids and decode to text."""
        original_ids = [self.compact_to_original[int(i)] for i in ids]
        return self.tokenizer.decode(original_ids, skip_special_tokens=remove_special_tokens)

    def tokens_to_ids(self, tokens):
        """Convert token strings to compact ids (byte fallback for out-of-subset tokens)."""
        original_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if isinstance(original_ids, int):
            original_ids = [original_ids]
        return self._original_ids_to_compact(original_ids)

    def ids_to_tokens(self, ids):
        """Convert compact ids to original token strings."""
        original_ids = [self.compact_to_original[int(i)] for i in ids]
        return self.tokenizer.convert_ids_to_tokens(original_ids)

    def token_to_id(self, token):
        """Compact id for a single token (None if it is out-of-subset and not a byte token)."""
        ids = self.tokens_to_ids([token])
        return ids[0] if ids else None

    @property
    def vocab(self):
        """List of token strings indexed by compact id."""
        return [self._inv_full_vocab[q] for q in self.compact_to_original]

    @property
    def inv_vocab(self):
        """Mapping token string -> compact id."""
        return {self._inv_full_vocab[q]: c for c, q in enumerate(self.compact_to_original)}
