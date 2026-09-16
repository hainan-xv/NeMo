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

The two model families are trained on the SAME aligner output, the SAME Qwen3
vocabulary and the SAME 14-frame chunk grid, which makes a token-level ensemble
possible in a way it normally is not: at every step both models are predicting
the next piece of the same tokenization of the same words, so their log-probs
can simply be added. Nothing here resamples, re-tokenizes or re-aligns.

WHAT IS ACTUALLY SHARED, verified rather than assumed:
  * vocabulary -- both load Qwen3-1.7B, 151,669 pieces, so text token INDICES
    are identical and no mapping table is needed.
  * terminator -- CHAT's transducer blank sits at ``vocab_size`` (outside the
    tokenizer), SCRIPT closes a chunk with ``<|im_end|>``. Different indices for
    the same event, so each scorer reports it in a canonical slot (``END``) and
    the fusion never sees the raw ids.

WHAT IS NOT SHARED, and why this file is more than a weighted sum:
  The two chunk-assignment rules disagree on exact chunk boundaries --

      CHAT    t = (ceil(end/frame_len) + delay) // chunk_size
      SCRIPT  t = first chunk with ready <= (t+1)*chunk_size  ==  (ready-1) // chunk_size

  which differ precisely when ``ready`` is a multiple of ``chunk_size``. That is
  1/chunk_size of positions -- ~7% at chunk 14 -- and always in the same
  direction: CHAT places a boundary word ONE CHUNK LATER than SCRIPT. The offset
  is systematic, not noise, so a naive per-chunk sum would penalise the correct
  hypothesis at every boundary rather than average two opinions of it. Neither
  model is trained to tolerate it either: SCRIPT's stochastic word-delay
  augmentation (``word_delay_prob``) defaults to 0 and is off in the v2 recipe.

  So a beam carries an explicit SKEW in {0, -1}: CHAT is scored at chunk ``t``
  while SCRIPT is scored at chunk ``t + skew``. The sign follows the direction of
  the mismatch -- CHAT places a boundary word LATER, so to agree with it SCRIPT
  must be read one chunk BEHIND. (Reading SCRIPT ahead, the intuitive-looking
  ``+1``, moves the two further apart and deadlocks chunk 0.)

  Both values are seeded from the START rather than forked at a later boundary:
  the offset is a property of the alignment convention, so it is already in
  effect at chunk 0, and a beam that cannot get past chunk 0 can never reach a
  boundary at which to fork. A skewed beam necessarily runs off one end of the
  utterance -- SCRIPT has no chunk -1, and CHAT finishes a chunk early -- and a
  model whose chunk is out of range contributes ``_neutral``, i.e. abstains and
  lets the other decide. Set ``allow_skew=False`` for the naive behaviour, which
  is the ablation the equivalence tests pin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Sequence, Tuple

__all__ = ["END", "ChunkScorer", "Beam", "chunk_sync_joint_decode"]

# Canonical slot for "this chunk is finished", in the scorers' output vectors.
# The raw ids differ per model (CHAT blank at vocab_size, SCRIPT <|im_end|>), and
# keeping the translation inside each scorer is what lets the fusion below be a
# plain elementwise add.
END = -1

NEG_INF = -1e30


class ChunkScorer(Protocol):
    """One model's view of decoding, reduced to what fusion needs.

    Implementations wrap EncDecCHATBPEModel and ScriptSTTModel respectively. The
    protocol exists so the search can be tested against stubs -- the fusion
    arithmetic and the skew bookkeeping are where the bugs live, and neither
    needs a GPU or a 2.4B-parameter model to exercise.
    """

    def init_state(self) -> Any:
        """Opaque per-hypothesis state (caches, step index, token history)."""

    def logprobs(self, state: Any, chunk_idx: int) -> Sequence[float]:
        """Next-token log-probs for this chunk.

        Length ``V + 1``; index ``V`` is the END slot. Must be normalised, since
        the fusion weights assume comparable scales between the two models.
        """

    def advance(self, state: Any, token: int) -> Any:
        """State after emitting ``token`` (never END) within the current chunk."""

    def close_chunk(self, state: Any, chunk_idx: int) -> Any:
        """State after finishing ``chunk_idx``; the next call uses ``chunk_idx+1``."""


@dataclass
class Beam:
    """One hypothesis: the tokens, both models' states, and the skew."""

    tokens: List[int] = field(default_factory=list)
    chat_state: Any = None
    script_state: Any = None
    # +1 means SCRIPT is one chunk AHEAD of CHAT, which is the direction the
    # rule mismatch predicts. 0 means the two agree.
    skew: int = 0
    score: float = 0.0
    # Per-chunk token count, to enforce max_tokens_per_chunk.
    emitted_this_chunk: int = 0

    def key(self) -> Tuple:
        """Identity for recombination: same tokens AND same skew are the same path."""
        return (tuple(self.tokens), self.skew)


def _fuse(chat_lp: Sequence[float], script_lp: Sequence[float], lam: float) -> List[float]:
    """Log-linear interpolation, the standard shallow-fusion form.

    Deliberately NOT renormalised. Beam search compares hypotheses of equal
    length at equal depth, so a shared per-step constant cancels; paying a
    logsumexp over 151k entries per step per beam would dominate the runtime for
    no change in the argmax.
    """
    n = len(chat_lp)
    if len(script_lp) != n:
        raise ValueError(f"scorer vocab mismatch: chat={n} script={len(script_lp)}")
    # A -inf vetoes the token, because a symbol one model has structurally
    # excluded must not be reachable just because the other scores it finitely.
    # But ONLY from a model that carries weight: at lam=1 the ensemble has to
    # reduce to CHAT exactly, and a veto from the zero-weight side would leave it
    # silently unable to emit CHAT's own argmax.
    chat_votes = lam > 0.0
    script_votes = lam < 1.0
    out = [0.0] * n
    for i in range(n):
        a, b = chat_lp[i], script_lp[i]
        if (chat_votes and a <= NEG_INF) or (script_votes and b <= NEG_INF):
            out[i] = NEG_INF
        else:
            out[i] = (lam * a if chat_votes else 0.0) + ((1.0 - lam) * b if script_votes else 0.0)
    return out


def _neutral(vocab_size: int) -> List[float]:
    """ "No opinion": log 1 everywhere, so the other model decides alone.

    Used when a model's chunk index falls outside the utterance, which a skewed
    beam necessarily produces at one end or the other.
    """
    return [0.0] * (vocab_size + 1)


def chunk_sync_joint_decode(
    chat: ChunkScorer,
    script: ChunkScorer,
    num_chunks: int,
    vocab_size: int,
    lam: float = 0.5,
    beam_size: int = 4,
    max_tokens_per_chunk: int = 32,
    allow_skew: bool = True,
    skew_penalty: float = 0.0,
) -> List[int]:
    """Decode one utterance, synchronising the two models at every chunk boundary.

    Args:
        chat, script: the two scorers.
        num_chunks: chunks in this utterance (both models see the same grid).
        vocab_size: ``V``; scorer vectors are ``V + 1`` long with END last.
        lam: weight on CHAT. 1.0 is CHAT alone, 0.0 is SCRIPT alone -- both are
            exercised by the tests as the degenerate cases.
        beam_size: hypotheses kept across a chunk boundary.
        max_tokens_per_chunk: guard against a model that never emits END. A chunk
            is 14 frames (~1.12 s) of audio, so a real chunk holds a handful of
            tokens; this only fires on a degenerate hypothesis.
        allow_skew: explore SCRIPT running one chunk ahead (see module docstring).
        skew_penalty: log-domain cost for adopting skew=+1. 0.0 lets the models
            decide purely on likelihood; a small positive value biases toward the
            aligned reading when the evidence is a wash.

    Returns:
        The best hypothesis' token ids, END excluded.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    if num_chunks <= 0:
        return []

    beams = [Beam(chat_state=chat.init_state(), script_state=script.init_state(), skew=0)]
    if allow_skew:
        beams.append(
            Beam(
                chat_state=chat.init_state(),
                script_state=script.init_state(),
                skew=-1,
                score=-skew_penalty,
            )
        )

    # One extra pass: a skew=-1 beam is a chunk behind on the SCRIPT side, so its
    # final chunk of audio is still unread when CHAT has finished. CHAT abstains
    # there via _neutral.
    total_passes = num_chunks + (1 if allow_skew else 0)

    for t in range(total_passes):
        for b in beams:
            b.emitted_this_chunk = 0

        # --- expand within the chunk until every beam has emitted END ---
        active, finished = beams, []
        while active:
            nxt: List[Beam] = []
            for b in active:
                if b.emitted_this_chunk >= max_tokens_per_chunk:
                    finished.append(b)
                    continue

                # Either model may be outside the utterance on a skewed beam;
                # whoever is out of range abstains rather than being clamped onto
                # a chunk it has already consumed.
                s_chunk = t + b.skew
                c_lp = chat.logprobs(b.chat_state, t) if t < num_chunks else _neutral(vocab_size)
                s_lp = script.logprobs(b.script_state, s_chunk) if 0 <= s_chunk < num_chunks else _neutral(vocab_size)
                if t >= num_chunks and not (0 <= s_chunk < num_chunks):
                    # Both abstain: nothing left to decode on this beam.
                    finished.append(b)
                    continue
                fused = _fuse(c_lp, s_lp, lam)

                # Top (beam_size + 1) candidates: END plus enough tokens that a
                # full beam can still be filled if END wins.
                order = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)
                for idx in order[: beam_size + 1]:
                    sc = fused[idx]
                    if sc <= NEG_INF:
                        continue
                    if idx == vocab_size:  # END
                        nb = Beam(
                            tokens=list(b.tokens),
                            chat_state=b.chat_state,
                            script_state=b.script_state,
                            skew=b.skew,
                            score=b.score + sc,
                            emitted_this_chunk=b.emitted_this_chunk,
                        )
                        finished.append(nb)
                    else:
                        nxt.append(
                            Beam(
                                tokens=b.tokens + [idx],
                                chat_state=chat.advance(b.chat_state, idx),
                                script_state=script.advance(b.script_state, idx),
                                skew=b.skew,
                                score=b.score + sc,
                                emitted_this_chunk=b.emitted_this_chunk + 1,
                            )
                        )
            active = sorted(nxt, key=lambda x: x.score, reverse=True)[:beam_size]

        # --- chunk boundary: close both models, then consider the skew flip ---
        closed: List[Beam] = []
        for b in finished:
            if t < num_chunks:
                b.chat_state = chat.close_chunk(b.chat_state, t)
            s_chunk = t + b.skew
            if 0 <= s_chunk < num_chunks:
                b.script_state = script.close_chunk(b.script_state, s_chunk)
            closed.append(b)

        # Recombine identical (tokens, skew) paths, keeping the best score, then
        # prune. Without this the skew fork doubles the beam every chunk.
        best: dict = {}
        for b in closed:
            k = b.key()
            if k not in best or b.score > best[k].score:
                best[k] = b
        beams = sorted(best.values(), key=lambda x: x.score, reverse=True)[:beam_size]
        if not beams:  # every path vetoed; nothing sensible to continue from
            return []

    return beams[0].tokens
