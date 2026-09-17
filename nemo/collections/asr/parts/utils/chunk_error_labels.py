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
"""Which chunks a frozen ASR model got WRONG, for training a corrector.

The labels a verifier/corrector trains on: per chunk, either ACCEPT (the ASR's
output for this chunk needs no change) or a correction target.

THE RULE, and why it is not per-chunk string equality. An ASR that emits the
right WORDS on a different chunk boundary is not wrong -- it is early or late.
Comparing chunk strings directly would label both the chunk that lost a word and
the one that gained it, teaching a corrector to fight emission timing instead of
fixing content. Timing is exactly what this project's banded losses exist to
leave loose, so penalising it here would be working against the model.

So:

  1. If the concatenated hypothesis equals the reference, EVERY chunk is ACCEPT.
     No alignment, no attribution -- the transcript is right, so nothing to fix
     regardless of where the boundaries fell.

  2. Otherwise align the hypothesis words to the reference words. Matched words
     are correct wherever they landed. Only words inside an edit mark their
     chunk as wrong.

Chunks are indexed on the HYPOTHESIS partition, because that is the decision the
corrector actually makes: at inference it is handed hypothesis chunk k and must
accept or rewrite THAT span. Indexing on the reference instead looks equivalent
and is not -- the ASR's emission lags the aligner's chunking by around a word, so
reference chunk k and hypothesis chunk k hold different words, and a label
computed on one applied to the other lands on the wrong chunk. Observed: the
error "real time" (for reference "real-time") marked hypothesis chunk 16
' kernel to provide' -- text that is correct -- while chunk 17, which held the
actual error, was labelled ACCEPT. Applying such a target deletes the words the
hypothesis emitted late and duplicates the ones it emitted early, so the
"correction" raises WER.

Deletions still have somewhere to go: a reference word the ASR never emitted has
no hypothesis position, so it is charged to the chunk it would have been read
into -- the one owning the next aligned hypothesis word, or the last chunk if the
deletion is trailing.

Every chunk also reports the reference words the alignment assigns to it, not
just the wrong ones. The caller needs the accepted chunks' spans too, to build
the conditioning history.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

__all__ = ["align_words", "label_chunks", "simple_normalize"]


def simple_normalize(word: str) -> str:
    """Casefold and strip surrounding punctuation, for the LABEL DECISION only.

    The ASR emits cased, punctuated text; the aligner's reference words are raw.
    Compared literally, almost every chunk looks wrong -- measured, that inflated
    the wrong-chunk rate from 7% to 75%, which is the difference between a
    usable corpus and nonsense.

    This is used ONLY to decide whether a chunk needs correcting. The correction
    TARGET stays the true reference text, because that is what the model should
    learn to emit -- casing and punctuation included.
    """
    return word.strip(".,!?;:\"'()[]").casefold()


def align_words(hyp: Sequence[str], ref: Sequence[str]) -> List[Tuple[str, int, int]]:
    """Levenshtein alignment as ``(op, hyp_i, ref_j)`` with ``-1`` for absent.

    ``op`` is one of ``equal`` / ``sub`` / ``ins`` (hypothesis-only) / ``del``
    (reference-only).
    """
    n, m = len(hyp), len(ref)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    out: List[Tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if hyp[i - 1] == ref[j - 1] else 1):
            out.append(("equal" if hyp[i - 1] == ref[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append(("ins", i - 1, -1))
            i -= 1
        else:
            out.append(("del", -1, j - 1))
            j -= 1
    out.reverse()
    return out


def label_chunks(
    hyp_chunks: Sequence[Sequence[str]],
    ref_chunks: Sequence[Sequence[str]],
    normalize=None,
):
    """``(labels, n_wrong, owned)``, all indexed by HYPOTHESIS chunk.

    Args:
        hyp_chunks: the ASR's words grouped by ITS OWN chunk. This is the index
            space of the result -- see the module docstring for why it is not the
            reference's.
        ref_chunks: reference words grouped by chunk. Used only to recover the
            reference word SEQUENCE; its chunk boundaries do not survive.

    Returns:
        labels: ``None`` for ACCEPT, else the correction target text.
        n_wrong: how many chunks are labelled wrong.
        owned: for EVERY chunk, the reference words the alignment assigns to it.
            Concatenated over all chunks this reproduces the reference exactly,
            which is what lets accept-or-correct decisions be stitched back into
            a transcript.
    """
    # Drop tokens that normalise to NOTHING, remembering which chunk each
    # surviving word came from. CHAT is trained on the original punctuated
    # transcript, so a chunk can legitimately begin with a standalone "." or ","
    # -- which split() makes its own word and simple_normalize maps to "". An
    # empty string matches nothing, so every such chunk scored as an error even
    # when the hypothesis was perfect: measured, it pushed the accept rate from
    # ~0.93 down to ~0.77 and taught the corrector to reject nearly everything.
    norm = normalize or (lambda w: w)

    hyp_words: List[str] = []
    hyp_owner: List[int] = []
    for k, c in enumerate(hyp_chunks):
        for w in c:
            x = norm(w)
            if x:
                hyp_words.append(x)
                hyp_owner.append(k)

    ref_words: List[str] = []
    ref_orig: List[str] = []
    for c in ref_chunks:
        for w in c:
            x = norm(w)
            if x:
                ref_words.append(x)
                ref_orig.append(w)

    n = len(hyp_chunks)
    if n == 0:
        return [], 0, []

    owned: List[List[int]] = [[] for _ in range(n)]

    # Rule 1: the transcript is right, so nothing is wrong anywhere -- whatever
    # the boundaries did. Each chunk still owns the words it emitted.
    if hyp_words == ref_words:
        for i in range(len(ref_words)):
            owned[hyp_owner[i]].append(i)
        return [None] * n, 0, [[ref_orig[j] for j in o] for o in owned]

    bad = set()
    pending_del: List[int] = []
    for op, hi, rj in align_words(hyp_words, ref_words):
        if op == "del":
            # No hypothesis position exists for this reference word; hold it for
            # the next hypothesis-anchored op and charge it to that chunk.
            pending_del.append(rj)
            continue
        owner = hyp_owner[hi]
        if pending_del:
            owned[owner].extend(pending_del)
            bad.add(owner)
            pending_del = []
        if op == "ins":
            # A spurious word: this chunk owns no reference word for it, but it
            # must still be rewritten -- to nothing, if it owns nothing at all.
            bad.add(owner)
            continue
        owned[owner].append(rj)
        if op == "sub":
            bad.add(owner)
    if pending_del:  # trailing deletions land on the final chunk
        owned[n - 1].extend(pending_del)
        bad.add(n - 1)

    out_words = [[ref_orig[j] for j in sorted(o)] for o in owned]
    labels = [(" ".join(out_words[k]) if k in bad else None) for k in range(n)]
    return labels, len(bad), out_words
