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
"""Swap the SpeechLM's text vocabulary for the ASR encoder's.

WHY. The LLM carries its vocabulary in the input embedding and the output head.
For Qwen3-1.7B that is 151,936 x 2,048 = **311 M parameters** -- 13% of the 2.4 B
model -- and a softmax over 151,936 classes at *every* decode step. The ASR
encoder we already load ships a 1,024-piece SentencePiece BPE vocabulary: 148x
smaller, 2.1 M parameters, and a softmax 148x cheaper.

THE COST is sequence length. The ASR vocabulary is far more granular: measured on
AMI references it needs **1.877 tokens/word against Qwen's 1.160, i.e. 1.62x more
tokens**. In SCRIPT that lengthens every branch, so attention pays back part of
what the head saves. Which side wins is an empirical question about where decode
time actually goes -- this module exists so the experiment can be run.

WHAT IS LOST. The vocabulary is cased but only lightly punctuated (9 punctuation
pieces), so prompt-controlled capitalisation/punctuation stops being meaningful.
The leaderboard normalizer strips both before scoring, so measured WER is
unaffected; a cased/punctuated deliverable would not be. Coverage is otherwise
good: 99.6% of AMI words are representable, the rest fall back to <unk>.

WHY IT MIGHT ALSO HELP ACCURACY. The insertion study found all three LLM-based
systems fabricate ~21-23k net filler words on the leaderboard (``you``, ``know``,
``and``) against the RNN-T's ~5k -- a property of decoding with a large text
prior, invariant to the streaming layout. The RNN-T uses *this* vocabulary. A
smaller, ASR-shaped text interface is a plausible lever on that imbalance, which
the position-dependent emission penalty could not move.

INITIALISATION. Replacing the vocabulary throws away the mapping between token
ids and the LLM body's knowledge; the body still knows English but the interface
is gone. Rather than start from noise, each new piece is initialised as the MEAN
of the donor LLM's embeddings for the same string ("donor averaging"), so
``▁the`` inherits Qwen's notion of " the". Special tokens that exist verbatim in
the donor keep their original vector.

BACKWARD COMPATIBILITY. Nothing here runs unless a model explicitly asks for it.
Existing checkpoints keep Qwen's tokenizer, ids and embedding table untouched.
"""

import os
import shutil
import tarfile
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor

from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging

# SentencePiece marks a word-initial piece with this; the SCRIPT word-start
# helper already recognises it alongside GPT-2's "Ġ".
SPM_SPACE = "▁"


def extract_spm_from_nemo(nemo_path: str, out_dir: str) -> str:
    """Pull the SentencePiece model out of a ``.nemo`` archive.

    A ``.nemo`` is a tar of the config, weights and tokenizer artifacts, with the
    tokenizer named ``<hash>_tokenizer.model``. Returns the extracted path.
    """
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(nemo_path, "r:") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith("tokenizer.model")]
        if not members:
            raise FileNotFoundError(
                f"{nemo_path} contains no *tokenizer.model; is it a NeMo ASR model with a "
                f"SentencePiece tokenizer? Found: {[m.name for m in tf.getmembers()][:8]}"
            )
        if len(members) > 1:
            raise ValueError(f"{nemo_path} has {len(members)} tokenizer models: {[m.name for m in members]}")
        member = members[0]
        dest = os.path.join(out_dir, "tokenizer.model")
        with tf.extractfile(member) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return dest


class AsrVocabTokenizer(TokenizerSpec):
    """The ASR SentencePiece vocabulary behind the interface the SpeechLM expects.

    The model reaches for two different shapes: NeMo's (``text_to_ids`` /
    ``ids_to_text``) and HuggingFace's (``self.tokenizer.convert_tokens_to_ids``,
    ``.eos_token_id``, ``convert_ids_to_tokens``). This exposes both, so swapping
    the vocabulary needs no changes at the call sites -- everything there looks
    special tokens up by STRING, and the strings are preserved.

    Special tokens are appended after the SentencePiece pieces, so a piece keeps
    the id the ASR model gave it.

    Subclasses TokenizerSpec because NeMo DISPATCHES ON IT: Lhotse's dataloader
    wraps the tokenizer in TokenizerWrapper, which routes a TokenizerSpec through
    ``text_to_ids`` and anything else through ``tokenizer(text)`` -- the parser
    protocol. The tokenizer this replaces (NeMo's AutoTokenizer) is a
    TokenizerSpec, so being one keeps the dispatch identical instead of landing
    in a branch meant for character parsers.
    """

    def __init__(
        self,
        spm_path: str,
        special_tokens: Sequence[str] = (),
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
    ):
        specials = list(dict.fromkeys(t for t in special_tokens if t))  # de-duplicate, keep order
        # legacy=True is the mode that allows appending special tokens BEYOND the
        # trained SentencePiece range; without it the class refuses them outright
        # ("Provide special tokens at train time"), and we cannot retrain the ASR
        # tokenizer just to carry <|vision_start|>.
        self._spm = SentencePieceTokenizer(model_path=spm_path, special_tokens=specials, legacy=True)
        self._specials = specials
        self._eos_token = eos_token
        self._pad_token = pad_token or eos_token
        self._vocab_cache: Optional[Dict[str, int]] = None
        # `self.tokenizer` mirrors the HF wrapper attribute the call sites use.
        self.tokenizer = self

    # --- NeMo tokenizer interface -------------------------------------------------
    def text_to_ids(self, text: str) -> List[int]:
        return self._spm.text_to_ids(text)

    def ids_to_text(self, ids: Sequence[int]) -> str:
        return self._spm.ids_to_text(list(ids))

    def text_to_tokens(self, text: str) -> List[str]:
        return self._spm.text_to_tokens(text)

    @property
    def vocab_size(self) -> int:
        return len(self)

    def __len__(self) -> int:
        return self._spm.vocab_size

    # --- HuggingFace-shaped interface ---------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = False, **kwargs) -> List[int]:
        """HF-style alias for :meth:`text_to_ids`.

        ``add_special_tokens`` is accepted and ignored: this tokenizer never
        prepends bos/eos of its own, so False (what every call site passes) is
        already the behaviour.
        """
        return self.text_to_ids(text)

    def decode(self, ids, skip_special_tokens: bool = False, **kwargs) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if int(i) not in self._spm.id_to_special_token]
        return self.ids_to_text(ids)

    def get_vocab(self) -> Dict[str, int]:
        """token -> id for the whole vocabulary, built once and cached."""
        if self._vocab_cache is None:
            self._vocab_cache = {self._id_to_token(i): i for i in range(len(self))}
        return self._vocab_cache

    @property
    def vocab(self) -> Dict[str, int]:
        return self.get_vocab()

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._token_to_id(tokens)
        return [self._token_to_id(t) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._id_to_token(ids)
        return [self._id_to_token(int(i)) for i in ids]

    def _token_to_id(self, token: str) -> int:
        if token in self._spm.special_token_to_id:
            return int(self._spm.special_token_to_id[token])
        return int(self._spm.token_to_id(token))

    def _id_to_token(self, idx: int) -> str:
        if idx in self._spm.id_to_special_token:
            return self._spm.id_to_special_token[idx]
        return self._spm.ids_to_tokens([idx])[0]

    @property
    def eos_token_id(self) -> Optional[int]:
        return None if self._eos_token is None else self._token_to_id(self._eos_token)

    @property
    def eos_token(self) -> Optional[str]:
        return self._eos_token

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.pad_id

    # --- NeMo id accessors ---------------------------------------------------------
    # This SentencePiece model declares no bos/eos/pad of its own (all -1), so
    # they resolve through the carried-over donor specials instead. pad_id must
    # be a REAL id: the collators pad with it, and -1 would index out of range.
    @property
    def pad_id(self) -> int:
        if self._pad_token is None:
            raise ValueError("AsrVocabTokenizer has no pad token; pass pad_token= or eos_token=")
        return self._token_to_id(self._pad_token)

    # NeMo tokenizers expose the token STRINGS as bos/eos/pad and the ids as
    # bos_id/eos_id/pad_id; HF uses *_token / *_token_id. Both spellings appear
    # across speechlm2, so both are provided.
    @property
    def eos(self) -> Optional[str]:
        return self._eos_token

    @property
    def pad(self) -> Optional[str]:
        return self._pad_token

    @property
    def bos(self) -> None:
        return None

    @property
    def bos_token(self) -> None:
        return None

    @property
    def pad_token(self) -> Optional[str]:
        return self._pad_token

    def token_to_id(self, token: str) -> int:
        return self._token_to_id(token)

    @property
    def unk_token_id(self) -> int:
        """SentencePiece's <unk>, id 0 here.

        Exposed because the SCRIPT dataset validates its delimiters with
        ``getattr(tok, "unk_token_id", None)`` and skips the check when it is
        absent -- an out-of-vocabulary marker would then silently map to <unk>
        and train against the wrong id.
        """
        return int(self._spm.tokenizer.unk_id())

    @property
    def unk_id(self) -> int:
        return self.unk_token_id

    @property
    def eos_id(self) -> Optional[int]:
        return self.eos_token_id

    @property
    def bos_id(self) -> Optional[int]:
        # No bos in this vocabulary; the SCRIPT/streaming path never uses one.
        return None

    def tokens_to_ids(self, tokens) -> List[int]:
        return [self._token_to_id(t) for t in tokens]

    def ids_to_tokens(self, ids) -> List[str]:
        return [self._id_to_token(int(i)) for i in ids]

    def tokens_to_text(self, tokens, remove_special_tokens: bool = False) -> str:
        if remove_special_tokens:
            tokens = [t for t in tokens if t not in self._spm.special_token_to_id]
        return self._spm.tokens_to_text(list(tokens))

    def add_special_tokens(self, tokens) -> int:
        """Accepts HF's ``{"additional_special_tokens": [...]}`` or a bare list."""
        if isinstance(tokens, dict):
            tokens = tokens.get("additional_special_tokens", [])
        new = [t for t in tokens if t and t not in self._spm.special_token_to_id]
        if new:
            self._spm.add_special_tokens(new)
            self._specials.extend(new)
            self._vocab_cache = None  # ids shifted; a stale map would silently mis-resolve
        return len(new)


def _donor_ids_for(piece: str, donor_tok) -> List[int]:
    """Donor-vocabulary ids whose embeddings should seed ``piece``.

    A SentencePiece piece is either word-initial (``▁the``) or a continuation
    (``ing``). The donor marks word boundaries with a leading space, so the
    former is looked up as " the" and the latter as "ing" -- getting this
    backwards seeds every word-initial piece with a mid-word vector.
    """
    if piece.startswith(SPM_SPACE):
        text = " " + piece[len(SPM_SPACE) :]
    else:
        text = piece
    if not text:
        return []
    return donor_tok.encode(text, add_special_tokens=False)


def build_embedding_from_donor(
    new_tok: AsrVocabTokenizer,
    donor_tok,
    donor_weight: Tensor,
    verbose: bool = True,
) -> Tensor:
    """Seed a new embedding table from a donor LLM's, by string.

    For each new piece, average the donor's embeddings for the same text. A
    special token that exists verbatim in the donor keeps its own vector, so
    ``<|vision_start|>`` stays exactly what the pretrained model already learned.

    Pieces the donor cannot represent fall back to the mean of the whole donor
    table, which is a far better starting point than noise: it puts the vector in
    the right region of the space with no directional bias.
    """
    n_new = len(new_tok)
    dim = donor_weight.shape[1]
    out = torch.empty(n_new, dim, dtype=donor_weight.dtype)
    table_mean = donor_weight.mean(dim=0)

    n_exact = n_avg = n_fallback = 0
    donor_vocab = donor_tok.get_vocab()
    for i in range(n_new):
        piece = new_tok.convert_ids_to_tokens(i)
        if piece in donor_vocab:  # special tokens shared with the donor
            out[i] = donor_weight[donor_vocab[piece]]
            n_exact += 1
            continue
        ids = _donor_ids_for(piece, donor_tok)
        if ids:
            out[i] = donor_weight[torch.tensor(ids, dtype=torch.long)].mean(dim=0)
            n_avg += 1
        else:
            out[i] = table_mean
            n_fallback += 1
    if verbose:
        logging.info(
            f"ASR-vocab embedding init: {n_new} pieces "
            f"({n_exact} copied verbatim, {n_avg} donor-averaged, {n_fallback} table-mean fallback)"
        )
    return out


def summarize_swap(donor_tok, new_tok: AsrVocabTokenizer, hidden_size: int) -> Dict[str, float]:
    """Parameter accounting for the swap, so the trade is stated rather than assumed."""
    old_n, new_n = len(donor_tok), len(new_tok)
    return {
        "old_vocab": old_n,
        "new_vocab": new_n,
        "shrink_factor": old_n / max(new_n, 1),
        "old_embed_params": old_n * hidden_size,
        "new_embed_params": new_n * hidden_size,
        "saved_params": (old_n - new_n) * hidden_size,
    }
