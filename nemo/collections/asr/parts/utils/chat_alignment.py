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
"""Forced-alignment paths for CHAT training.

The RNN-T loss marginalises over every alignment, which costs a [B, T, U, V]
tensor. Conditioning on ONE alignment instead reduces the scored positions from
T*U to U+T and makes a large vocabulary trainable. This module turns word
timings into that single path.

Lives in the ASR collection so ``EncDecCHATBPEModel`` needs nothing from
speechlm2; the rules are identical to the ones validated there.
"""

import math
from typing import List, Optional, Sequence, Tuple

__all__ = ["assign_words_to_chunks", "build_forced_path", "word_spans", "chunk_texts", "band_nodes"]


def word_spans(words: Sequence[str], transcript: str) -> List[Optional[Tuple[int, int]]]:
    """Character spans of each aligned word inside the ORIGINAL transcript.

    Forced aligners emit bare word forms -- ``Media``, never ``Media.`` -- so a
    target built from them has no punctuation at all, and a PnC model trained on
    it can never produce any. Locating each word in the transcript instead
    recovers the true surface form: the span is extended through any trailing
    non-alphanumeric characters, which is what picks up commas, periods and
    quotes.

    Matching is case-insensitive and advances a cursor, so a repeated word maps
    to its own occurrence rather than always to the first. A word that cannot be
    located yields ``None`` rather than a wrong span.
    """
    spans: List[Optional[Tuple[int, int]]] = []
    lower = transcript.lower()
    pos = 0
    for w in words:
        idx = lower.find(w.lower(), pos)
        if idx == -1:
            spans.append(None)
            continue
        end = idx + len(w)
        while end < len(transcript) and not transcript[end].isalnum() and not transcript[end].isspace():
            end += 1
        spans.append((idx, end))
        pos = end
    return spans


def chunk_texts(groups: Sequence[Sequence[int]], words: Sequence[str], transcript: str) -> List[str]:
    """The text each chunk is responsible for, sliced from the transcript.

    Returned WITHOUT leading or trailing whitespace. That matters: SentencePiece
    turns a leading space into a standalone ``▁`` piece, so prepending one to
    every chunk trains the model to emit a junk token at each chunk boundary --
    which decodes as a double space and wastes one emission slot per chunk. The
    word-boundary marker is already supplied by the tokenizer's own dummy
    prefix, so the space is not needed for it either.

    Falls back to joining the aligner's word forms when a span cannot be
    located, which loses that word's punctuation but never loses the word.
    """
    spans = word_spans(words, transcript) if transcript else [None] * len(words)
    out = []
    for idxs in groups:
        if not idxs:
            out.append("")
            continue
        found = [spans[i] for i in idxs if spans[i] is not None]
        if found:
            out.append(transcript[found[0][0] : found[-1][1]].strip())
        else:
            out.append(" ".join(words[i] for i in idxs).strip())
    return out


def assign_words_to_chunks(
    word_end_times: Sequence[float],
    num_chunks: int,
    chunk_size: int,
    frame_length_in_secs: float,
    num_delay_frames: int = 0,
) -> List[List[int]]:
    """Which word indices each chunk is responsible for emitting.

    A word is emitted at the chunk containing its LAST frame, plus a delay. The
    delay matters because a word's final frames are often what disambiguate it:
    emitting at the chunk where it ends gives the encoder no right context at
    all.

    Words whose end time (plus delay) falls past the final chunk -- because the
    delay pushed them out, or because the alignment runs past the clip -- are
    folded into the last chunk rather than dropped, which would otherwise show
    up as deletions at the end of every utterance.
    """
    chunks: List[List[int]] = [[] for _ in range(max(num_chunks, 0))]
    if not chunks:
        return chunks
    for i, end in enumerate(word_end_times):
        ready = math.ceil(end / frame_length_in_secs) + num_delay_frames
        idx = min(ready // chunk_size, num_chunks - 1)
        chunks[max(idx, 0)].append(i)
    return chunks


def build_forced_path(
    chunk_tokens: List[List[int]],
    blank_id: int,
    recover_words: int = 0,
    word_starts: Optional[List[List[int]]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """(t_idx, u_idx, labels) for one utterance.

    Each chunk emits its tokens then exactly one blank -- silent chunks
    included, since that is the only signal for "emit nothing here". ``u``
    counts EMITTED LABELS and so does not advance on a blank: the prediction
    network is conditioned on emitted labels alone.

    HISTORY RECOVERY (``recover_words`` > 0) additionally scores each chunk on
    the previous chunk's last k words, starting from the prefix that precedes
    them::

        chunk t-1 :  ... W1 W2 BLANK            <- unchanged, blank stays put
        chunk t   :  W1 W2 [own words] BLANK    <- begins k words earlier in u

    Nothing is removed, so no chunk is ever trained to stop before its last
    word. It teaches the model to recover when its history is short, which is
    what makes retract-style decoding legal. Never reaches further back than the
    immediately-previous chunk, so the decoder's matching rule can be satisfied
    exactly.
    """
    t_idx: List[int] = []
    u_idx: List[int] = []
    labels: List[int] = []
    u = 0
    chunk_u_start: List[int] = []

    for t, toks in enumerate(chunk_tokens):
        chunk_u_start.append(u)

        if recover_words > 0 and t > 0:
            prev = chunk_tokens[t - 1]
            starts = word_starts[t - 1] if word_starts else []
            if starts:
                take_from = starts[-recover_words] if len(starts) >= recover_words else starts[0]
                uu = chunk_u_start[t - 1] + take_from
                for tok in prev[take_from:]:
                    t_idx.append(t)
                    u_idx.append(uu)
                    labels.append(int(tok))
                    uu += 1

        for tok in toks:
            t_idx.append(t)
            u_idx.append(u)
            labels.append(int(tok))
            u += 1
        t_idx.append(t)
        u_idx.append(u)
        labels.append(int(blank_id))

    return t_idx, u_idx, labels


def band_nodes(tokens_per_chunk: Sequence[int], band: int) -> List[Tuple[int, int]]:
    """Lattice nodes ``(t, u)`` within ``band`` chunks of the forced path.

    The middle ground between the two losses. Conditioning on ONE alignment
    scores ``U + T`` positions and trusts the aligner completely; the full RNN-T
    lattice scores ``T * U`` and ignores it. This keeps the alignment as a PRIOR:
    a word may be emitted up to ``band`` chunks earlier or later than the aligner
    placed it, and the loss sums over every valid path in that band.

    ``t`` indexes CHUNKS -- CHAT's loss already uses chunks as its time axis --
    and ``u`` is the number of labels emitted, so ``u`` runs ``0..U``.

    At chunk ``t`` the forced path occupies ``u`` in ``[S(t), S(t+1)]`` where
    ``S`` is the cumulative token count. Widening by ``band`` chunks gives
    ``[S(t-band), S(t+band+1)]``. So ``band=0`` returns exactly the forced path's
    own nodes (``U + T`` of them) and the cost grows roughly linearly in
    ``2*band + 1`` -- about 3x at ``band=1``, against ``T*U/(U+T)`` (an order of
    magnitude more) for the full lattice.

    Returns nodes grouped by ``t`` and ascending in ``u``.
    """
    T = len(tokens_per_chunk)
    if T == 0:
        return []
    cum = [0]
    for n in tokens_per_chunk:
        cum.append(cum[-1] + int(n))

    nodes: List[Tuple[int, int]] = []
    for t in range(T):
        u_lo = cum[max(0, t - band)]
        u_hi = cum[min(T - 1, t + band) + 1]
        for u in range(u_lo, u_hi + 1):
            nodes.append((t, u))
    return nodes
