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

Chunks are indexed on the REFERENCE partition, not the hypothesis's. That is
what makes deletions expressible: a word the ASR never emitted has no
hypothesis position and therefore no hypothesis chunk, but it always has a
reference chunk -- and the correction target is reference text anyway.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

__all__ = ["align_words", "label_chunks"]


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


def label_chunks(hyp_words: Sequence[str], ref_chunks: Sequence[Sequence[str]]):
    """``(labels, n_wrong)`` where ``labels[t]`` is None for ACCEPT else the target.

    Args:
        hyp_words: the ASR's full hypothesis, flattened. Chunk boundaries in the
            HYPOTHESIS are deliberately not an input -- see the module docstring.
        ref_chunks: reference words grouped by chunk.
    """
    ref_words: List[str] = [w for c in ref_chunks for w in c]
    # Which reference chunk each reference word belongs to.
    owner: List[int] = [t for t, c in enumerate(ref_chunks) for _ in c]

    # Rule 1: the transcript is right, so nothing is wrong anywhere.
    if list(hyp_words) == ref_words:
        return [None] * len(ref_chunks), 0

    bad = set()
    pending_ins = 0  # insertions seen before the next reference-anchored op
    for op, _hi, rj in align_words(hyp_words, ref_words):
        if op == "ins":
            pending_ins += 1
            continue
        if rj >= 0:
            t = owner[rj]
            if op in ("sub", "del"):
                bad.add(t)
            if pending_ins:
                # Spurious words belong to the chunk that was about to be read;
                # attributing them to the PREVIOUS chunk would blame a chunk the
                # reference says was fine.
                bad.add(t)
        pending_ins = 0
    if pending_ins:  # trailing insertions land on the final chunk
        bad.add(len(ref_chunks) - 1)

    labels = [(" ".join(ref_chunks[t]) if t in bad else None) for t in range(len(ref_chunks))]
    return labels, len(bad)
