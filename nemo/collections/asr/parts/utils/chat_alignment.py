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
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = [
    "assert_clean_transcript",
    "clean_transcript",
    "assign_words_to_chunks",
    "build_forced_path",
    "word_spans",
    "word_core_end",
    "chunk_texts",
    "band_nodes",
]


def word_spans(
    words: Sequence[str],
    transcript: str,
    report: Optional[Callable[[str, str, str], None]] = None,
    respell: bool = True,
) -> List[Optional[Tuple[int, int]]]:
    """Character spans of each aligned word inside the ORIGINAL transcript.

    Forced aligners emit bare word forms -- ``Media``, never ``Media.`` -- so a
    target built from them has no punctuation at all, and a PnC model trained on
    it can never produce any. Locating each word in the transcript instead
    recovers the true surface form: the span is extended through any trailing
    non-alphanumeric characters, which is what picks up commas, periods and
    quotes.

    Matching is case-insensitive and advances a cursor, so a repeated word maps
    to its own occurrence rather than always to the first.

    THE RESPELLING RETRY. The aligner ran on NORMALISED text, so it strips
    punctuation that sits INSIDE a word: the transcript's ``forward-looking``
    reaches us as ``forwardlooking``, ``3-year`` as ``3year``, ``e-commerce`` as
    ``ecommerce``. Those are not a literal substring of the transcript, so the
    plain search above returns -1 and the word ends up with no span -- and a word
    with no span cannot be sliced into its chunk, so it silently vanishes from
    the training target. Measured over the aligned manifests that was 0.462% of
    all aligner words (0.528% on spgispeech), and 0% on LibriSpeech, which has no
    punctuation to strip.

    So when the literal search fails the word is retried with punctuation ignored
    on BOTH sides, by matching against an alphanumeric-only view of the
    transcript and mapping the hit back to original offsets. That recovers 93.8%
    of the failures. Anything still unfound (a word genuinely absent from the
    transcript, e.g. a verbalised number) yields ``None`` rather than a wrong
    span, exactly as before.

    ``respell=False`` restores the pre-fix behaviour exactly -- the retry is
    still RUN, so the rate is still reported, but its answer is discarded and the
    word yields ``None`` as before. That keeps an already-launched control arm on
    the objective it was launched with while still telling us what it is losing.

    ``report(kind, aligner_word, transcript_form)`` is called for each mismatch:
    ``"respelled"`` (found by the retry and used), ``"dropped"`` (found by the
    retry but discarded because ``respell=False``), or ``"missing"`` (not in the
    transcript at all). So callers can log how often this fires on real data
    instead of inferring it.
    """
    # Alphanumeric-only view plus a map back to original character offsets, and a
    # prefix count so the cursor can be carried across without rescanning.
    norm_chars: List[str] = []
    back: List[int] = []
    alnum_before: List[int] = [0] * (len(transcript) + 1)
    for i, ch in enumerate(transcript):
        alnum_before[i + 1] = alnum_before[i]
        if ch.isalnum():
            norm_chars.append(ch.lower())
            back.append(i)
            alnum_before[i + 1] += 1
    norm = "".join(norm_chars)

    spans: List[Optional[Tuple[int, int]]] = []
    lower = transcript.lower()
    pos = 0
    for w in words:
        idx = lower.find(w.lower(), pos)
        if idx != -1:
            end = idx + len(w)
        else:
            w_norm = "".join(c.lower() for c in w if c.isalnum())
            j = norm.find(w_norm, alnum_before[pos]) if w_norm else -1
            if j == -1:
                spans.append(None)
                if report is not None:
                    report("missing", w, "")
                continue
            hit_start, hit_end = back[j], back[j + len(w_norm) - 1] + 1
            if not respell:
                spans.append(None)
                if report is not None:
                    report("dropped", w, transcript[hit_start:hit_end])
                continue
            idx, end = hit_start, hit_end
            if report is not None:
                report("respelled", w, transcript[idx:end])
        while end < len(transcript) and not transcript[end].isalnum() and not transcript[end].isspace():
            end += 1
        spans.append((idx, end))
        pos = end
    return spans


def word_core_end(transcript: str, span: Tuple[int, int]) -> int:
    """End of a span's ALPHANUMERIC content, i.e. the span minus trailing punctuation.

    ``word_spans`` deliberately extends each span through trailing punctuation so
    the surface form carries it. Deciding WHEN that punctuation is emitted needs
    the two parts separated, which is what this returns: for ``'Media.'`` at
    (0, 6) it returns 5, so ``[0, 5)`` is the word and ``[5, 6)`` is the period.
    """
    start, end = span
    core = start
    for i in range(start, end):
        if transcript[i].isalnum():
            core = i + 1
    return core


def chunk_texts(
    groups: Sequence[Sequence[int]],
    words: Sequence[str],
    transcript: str,
    report: Optional[Callable[[str, str, str], None]] = None,
    respell: bool = True,
) -> List[str]:
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
    spans = word_spans(words, transcript, report=report, respell=respell) if transcript else [None] * len(words)
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


def assert_clean_transcript(transcript: str, source: str = "") -> None:
    """Raise if the transcript would tokenize to an ORPHAN word-start token.

    Qwen's byte-level BPE marks word starts with a leading space. Text that runs
    two spaces together, or ends in one, therefore produces a piece that is a
    word marker with NO WORD ATTACHED, followed by another word-marked piece::

        'a  b'  -> ['a', 'G', 'Gb']     <- orphan
        'a b '  -> ['a', 'Gb', 'G']     <- orphan
        'a b'   -> ['a', 'Gb']          <- fine

    (``G`` stands for the byte-level space marker.) Measured on Qwen3-1.7B. A
    single space, a leading space, a tab or a newline all merge into the
    following piece and are harmless -- only repeated spaces and a trailing one
    orphan a token.

    This matters because the chunk targets are a SPLIT of one tokenization of the
    whole transcript. An orphan marker consumes an emission slot in whichever
    chunk it lands in, so the model is trained to emit a token that carries no
    word, and the chunk's token count no longer matches its word count. This is
    the same family as the standalone U+2581 piece that _token_split_points
    already guards against.

    Observed in mcv11_dev_clean_pcstrip_en_2k_qwen_aligned.json, where 25% of
    utterances have a double space after a comma -- the text was rebuilt with the
    comma merged onto the previous word and an empty placeholder left behind. A
    1.32M-utterance sample of the granary training manifests found ZERO, so this
    should never fire on training data; if it does, the data changed.
    """
    if not transcript:
        return
    if "  " in transcript:
        i = transcript.index("  ")
        raise ValueError(
            f"transcript has consecutive spaces at offset {i}"
            f"{f' in {source}' if source else ''}: {transcript[max(0, i - 30):i + 30]!r}. "
            "This tokenizes to an orphan word-start piece and corrupts the chunk targets. "
            "Fix the manifest (re.sub(r'  +', ' ', text)) rather than relaxing this check."
        )
    if transcript != transcript.rstrip():
        raise ValueError(
            f"transcript has trailing whitespace"
            f"{f' in {source}' if source else ''}: {transcript[-40:]!r}. "
            "This tokenizes to an orphan word-start piece at the end of the targets."
        )


def clean_transcript(transcript: str):
    """``(cleaned, n_fixed)`` -- collapse whitespace that would orphan a token.

    REPAIRS rather than raises. The same code path builds training AND validation
    targets, and the mcv11 validation manifest has a double space in 25% of its
    utterances (the comma was merged onto the previous word, leaving an empty
    placeholder). A hard assertion there kills the job at the first validation
    epoch -- which is exactly what it did.

    Collapsing is the right repair, not a workaround: the orphan piece carries no
    word, so removing it makes the targets match the words they are supposed to
    encode. See :func:`assert_clean_transcript` for what counts as orphaning and
    why a tab or a leading space does not.

    Returns the count so the caller can report it. Silence would turn a data
    defect into an invisible one, which is how this started.
    """
    if not transcript:
        return transcript, 0
    fixed = " ".join(transcript.split())
    return fixed, int(fixed != transcript)


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


def band_nodes(tokens_per_chunk: Sequence[int], band: int, side: str = "both") -> List[Tuple[int, int]]:
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

    if side not in ("both", "later", "earlier"):
        raise ValueError(f"side must be 'both', 'later' or 'earlier', got {side!r}")

    nodes: List[Tuple[int, int]] = []
    for t in range(T):
        # u is the number of labels emitted BY chunk t, so the two directions are:
        #   lower u  -> fewer labels emitted so far -> a word DEFERRED ("later")
        #   higher u -> more labels emitted already -> a word PULLED FORWARD
        # "later" is the half aligner error can justify: a word whose audio ends
        # just after a chunk boundary cannot honestly be emitted before that audio
        # arrives, which is the same thing num_delay_frames guards. It also costs
        # about half of the two-sided band.
        u_lo = cum[max(0, t - band)] if side in ("both", "later") else cum[t]
        u_hi = cum[min(T - 1, t + band) + 1] if side in ("both", "earlier") else cum[t + 1]
        for u in range(u_lo, u_hi + 1):
            nodes.append((t, u))
    return nodes
