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
"""Chunk-synchronous joint decoding of a CHAT transducer and a SCRIPT SpeechLM.

CHAT drives the loop -- it is the faster model and its transducer already steps
along exactly the axes we need. At every step it is at chunk ``t`` with history
``h`` and produces ``p_chat(token | t, h)``. SCRIPT is asked for
``p_script(token | t, h)`` at the SAME ``(t, h)``, and the two are combined
log-linearly::

    score(token) = lam * log p_chat(token | t, h) + (1 - lam) * log p_script(token | t, h)

over the shared vocabulary AND over the end-of-chunk symbol, so the two models
vote on when the chunk ends just as they vote on what is in it.

WHY THIS IS LEGITIMATE, and it is the whole reason the idea works: the two
families are trained on the same aligner output, the same Qwen3 vocabulary and
the same 14-frame chunk grid. At any ``(t, h)`` they are predicting the next
piece of the same tokenization of the same words, so their log-probs are
commensurable and can simply be added. Nothing here resamples, re-tokenizes or
re-aligns.

THE ONE THING THAT IS NOT SHARED, and why it needs no machinery. The two
chunk-assignment rules disagree when ``ready`` is an exact multiple of
``chunk_size`` (~7% of word positions at chunk 14): CHAT places such a word one
chunk LATER than SCRIPT. That is a training-target convention, and at inference
neither model emits one-hot -- a boundary word still draws probability from both
models, just split slightly differently across two adjacent chunks. So it shows
up as a mild disagreement in the sum, which is exactly what an ensemble is for,
not as a structural conflict. An earlier draft of this file modelled the offset
explicitly with per-hypothesis skew state; it was solving a problem that only
exists for one-hot distributions.

The END slot is the only translation, and each scorer owns it (see
``joint_decode_adapters``): CHAT's blank already sits at index ``V``, SCRIPT's
``<|im_end|>`` is moved there. The fusion below therefore never sees a raw id.
"""

from __future__ import annotations

from typing import Any, List, Protocol, Sequence

__all__ = ["ChunkScorer", "chunk_sync_joint_decode"]


class ChunkScorer(Protocol):
    """One model's view of decoding, reduced to what fusion needs.

    Implemented by ChatChunkScorer and ScriptChunkScorer. The protocol exists so
    the fusion can be tested with stubs: the arithmetic and the END handling are
    where bugs live, and neither needs a GPU to exercise.
    """

    def init_state(self) -> Any:
        """Opaque per-utterance state (caches, token history)."""

    def logprobs(self, state: Any, chunk_idx: int) -> Sequence[float]:
        """Next-token log-probs at this chunk: length ``V + 1``, END at index ``V``."""

    def advance(self, state: Any, token: int) -> Any:
        """State after emitting ``token`` (never END) inside the current chunk."""

    def close_chunk(self, state: Any, chunk_idx: int) -> Any:
        """State after finishing ``chunk_idx``."""


def chunk_sync_joint_decode(
    chat: ChunkScorer,
    script: ChunkScorer,
    num_chunks: int,
    vocab_size: int,
    lam: float = 0.5,
    max_tokens_per_chunk: int = 32,
) -> List[int]:
    """Greedy decode of one utterance, both models scoring every step.

    Args:
        chat, script: the two scorers, each returning ``V + 1`` log-probs.
        num_chunks: chunks in this utterance; both models see the same grid.
        vocab_size: ``V``. Index ``V`` in a scorer's output is END.
        lam: weight on CHAT. ``1.0`` is CHAT alone and ``0.0`` is SCRIPT alone --
            both are pinned by tests, since a fusion that does not reduce
            correctly at the endpoints is not doing what it claims in between.
        max_tokens_per_chunk: guard against a chunk that never ends. A chunk is
            ~1.12 s of audio, so this only fires on a degenerate hypothesis.

    Returns:
        The emitted token ids, END excluded.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    chat_state = chat.init_state()
    script_state = script.init_state()
    tokens: List[int] = []

    for t in range(num_chunks):
        for _ in range(max_tokens_per_chunk):
            c_lp = chat.logprobs(chat_state, t)
            s_lp = script.logprobs(script_state, t)
            if len(c_lp) != len(s_lp):
                raise ValueError(f"scorer vocab mismatch: chat={len(c_lp)} script={len(s_lp)}")

            best_i, best_v = 0, None
            for i in range(len(c_lp)):
                v = lam * c_lp[i] + (1.0 - lam) * s_lp[i]
                if best_v is None or v > best_v:
                    best_i, best_v = i, v

            if best_i == vocab_size:  # END: this chunk is finished
                break
            tokens.append(best_i)
            chat_state = chat.advance(chat_state, best_i)
            script_state = script.advance(script_state, best_i)

        chat_state = chat.close_chunk(chat_state, t)
        script_state = script.close_chunk(script_state, t)

    return tokens
