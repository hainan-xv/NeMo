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
"""Two text vocabularies on one LLM body, selected by the batch's chunk size.

WHY. Measured on the leaderboard, a 1024-piece vocabulary costs SCRIPT almost
nothing (5.60 against 5.58 at 151,936 pieces) while costing CHAT 29%. The small
vocabulary is also a better fit for SHORT chunks: its pieces are finer, so a word
can be spread across several 0.16 s chunks instead of having to be emitted whole
from one. The large vocabulary keeps its edge where a chunk is long enough to
carry whole words. This module lets one model use each where it is better.

HOW. The swap surface is two attributes. Everything downstream reads
``self.embed_tokens.weight`` (via ``_embed_tokens`` / ``_embed_ref_tensor``) and
``self.llm.lm_head`` (via ``_lm_head_module``), so re-pointing those two -- plus
the tokenizer and the special ids resolved from it -- moves the whole model to
the other vocabulary. Both tables stay registered, so both keep optimizer state.

WHAT THIS COSTS. The spine carries transcript history as text tokens, so the two
vocabularies tokenize history differently. A decode cannot cross the boundary
mid-stream: the fast/slow two-front trick, which shares one KV prefix between
two chunk sizes, is not available across the vocabulary split.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from nemo.utils import logging


@dataclass
class VocabBundle:
    """One vocabulary: its tokenizer, its tables, and the ids resolved from it.

    The ids are resolved ONCE per vocabulary at build time rather than per batch,
    because resolving them is a string lookup and getting them wrong is silent --
    a wrong ``eot`` id ends every branch on a token the model never supervised.
    """

    name: str
    tokenizer: Any
    embed: nn.Embedding
    head: nn.Linear
    vision_start_id: int
    vision_end_id: int
    eot_id: int
    pad_id: int

    def __len__(self) -> int:
        """Rows in the embedding table -- NOT len(tokenizer).

        NeMo's AutoTokenizer wrapper has no __len__, and more importantly the
        table is the authoritative size: _register_special_tokens() may have
        grown it past the tokenizer's nominal vocabulary, and it is the table
        the LLM actually indexes.
        """
        return int(self.embed.weight.shape[0])


def resolve_special_ids(tokenizer, audio_open: str, audio_close: str) -> Dict[str, int]:
    """Resolve the ids SCRIPT's packed layout is built from, for one tokenizer.

    Raises rather than returning ``unk``: a vocabulary missing the audio
    delimiters cannot express the layout at all, and finding that out at step 0
    is much cheaper than finding it out from a loss curve.
    """
    hf = getattr(tokenizer, "tokenizer", tokenizer)
    unk = getattr(hf, "unk_token_id", None)
    out = {}
    for key, tok in (("vision_start_id", audio_open), ("vision_end_id", audio_close)):
        tid = hf.convert_tokens_to_ids(tok)
        if tid is None or (unk is not None and tid == unk):
            raise ValueError(
                f"tokenizer {getattr(tokenizer, 'name', type(tokenizer).__name__)} has no id for {tok!r} "
                f"(got {tid}); the audio span cannot be delimited in this vocabulary."
            )
        out[key] = int(tid)
    eot = getattr(hf, "eos_token_id", None)
    if eot is None:
        raise ValueError("tokenizer has no eos_token_id; it is required as the branch end-of-turn token.")
    out["eot_id"] = int(eot)
    pad = getattr(hf, "pad_token_id", None)
    out["pad_id"] = int(pad if pad is not None else eot)
    return out


def build_asr_vocab_bundle(
    pretrained_asr: str,
    donor_tokenizer,
    donor_weight: torch.Tensor,
    audio_open: str,
    audio_close: str,
) -> VocabBundle:
    """Build the small-vocabulary bundle from the ASR encoder's SentencePiece model.

    The embedding is initialised from the donor by averaging the donor's vectors
    for each new piece's subwords -- not randomly. That is what makes the small
    table usable from step 0 rather than needing to relearn English from scratch.
    """
    import tempfile

    from nemo.collections.speechlm2.parts.asr_vocab import (
        AsrVocabTokenizer,
        build_embedding_from_donor,
        extract_spm_from_nemo,
    )

    hf_donor = getattr(donor_tokenizer, "tokenizer", donor_tokenizer)
    specials = list(dict.fromkeys(list(hf_donor.get_added_vocab().keys()) + list(hf_donor.all_special_tokens)))
    spm_path = extract_spm_from_nemo(pretrained_asr, tempfile.mkdtemp(prefix="dual_vocab_"))
    tok = AsrVocabTokenizer(
        spm_path, special_tokens=specials, eos_token=hf_donor.eos_token, pad_token=hf_donor.pad_token
    )

    hidden = int(donor_weight.shape[1])
    # float32 for the averaging, then back to the model's dtype.
    new_w = build_embedding_from_donor(tok, hf_donor, donor_weight.float()).to(donor_weight.dtype)
    emb = nn.Embedding(len(tok), hidden, dtype=donor_weight.dtype, device=donor_weight.device)
    with torch.no_grad():
        emb.weight.copy_(new_w)
    head = nn.Linear(hidden, len(tok), bias=False, dtype=donor_weight.dtype, device=donor_weight.device)
    head.weight = emb.weight  # tied, as Qwen3 ties input and output

    ids = resolve_special_ids(tok, audio_open, audio_close)
    logging.info(
        "dual-vocab: built the small bundle from %s -- %d pieces, embedding %.1fM parameters",
        pretrained_asr,
        len(tok),
        emb.weight.numel() / 1e6,
    )
    return VocabBundle(name="asr", tokenizer=tok, embed=emb, head=head, **ids)


def validate_chunk_split(chunk_size: Sequence[int], small_chunks: Sequence[int]) -> List[int]:
    """``small_chunks`` must be a non-empty PROPER subset of ``chunk_size``.

    Both halves must be non-empty: if every chunk routes to one vocabulary the
    other table never receives a gradient, and DDP would be averaging a table
    that no rank ever used -- a silently dead 311M parameters, or a silently
    dead model head.
    """
    cs, sm = [int(c) for c in chunk_size], [int(c) for c in small_chunks]
    missing = sorted(set(sm) - set(cs))
    if missing:
        raise ValueError(f"small_vocab_chunks {sm} contains {missing}, which are not in chunk_size {cs}")
    if not sm:
        raise ValueError("small_vocab_chunks is empty; use text_vocab_from_asr=false for a single-vocab arm")
    large = [c for c in cs if c not in set(sm)]
    if not large:
        raise ValueError(
            f"small_vocab_chunks {sm} covers every chunk in {cs}; that is a single-vocab arm. "
            "Use text_vocab_from_asr=true instead so only one table is built."
        )
    return large


def split_dataset_cfg(base, keep_chunks: Sequence[int], all_chunks: Sequence[int]):
    """A copy of the dataset config restricted to ``keep_chunks``.

    ``band_words_by_chunk`` is PARALLEL to ``chunk_size``, so it must be sliced
    by the same POSITIONS. Truncating it instead would keep the lengths
    consistent -- so the dataset's own validation still passes -- while pairing
    the wide band with the wrong chunk. Nothing downstream could detect that.
    """
    from copy import deepcopy

    from omegaconf import DictConfig, OmegaConf, open_dict

    out = deepcopy(base)
    if not isinstance(out, DictConfig):
        out = OmegaConf.create(out)
    bwc = list(base.get("band_words_by_chunk", None) or [])
    with open_dict(out):
        out.chunk_size = list(keep_chunks)
        if bwc:
            if len(bwc) != len(all_chunks):
                raise ValueError(f"band_words_by_chunk has {len(bwc)} entries but chunk_size has {len(all_chunks)}")
            pos = {int(c): i for i, c in enumerate(all_chunks)}
            out.band_words_by_chunk = [bwc[pos[int(c)]] for c in keep_chunks]
    return out


CANONICAL_LARGE_KEY = "_vocab_tables.llm_embed.weight"
ALIAS_KEYS = ("embed_tokens.weight", "llm.lm_head.weight")


def canonicalize_saved_aliases(state_dict: Dict[str, Any], llm_weight) -> None:
    """Force the aliased keys to the LARGE table before a checkpoint is written.

    ``embed_tokens`` and ``llm.lm_head`` are re-pointed per batch, so they would
    otherwise be saved with whatever shape was active -- 1050 rows after a
    chunk-2 batch, 151936 after a chunk-14 one. A restart builds the model with
    the large vocabulary active and dies on the mismatch before step 1. The
    small table is untouched; it lives under its own key and is the only copy.
    """
    for k in ALIAS_KEYS:
        if k in state_dict:
            state_dict[k] = llm_weight


def repair_loaded_aliases(state_dict: Dict[str, Any]) -> bool:
    """Rebuild the aliases from the canonical large table when loading.

    Returns True if anything was repaired. This is what makes checkpoints
    written BEFORE the save hook existed still resumable, rather than the arm
    having to restart from step 0.
    """
    canon = state_dict.get(CANONICAL_LARGE_KEY, None)
    if canon is None:
        return False
    fixed = False
    for k in ALIAS_KEYS:
        if k in state_dict and state_dict[k].shape != canon.shape:
            state_dict[k] = canon
            fixed = True
    return fixed
