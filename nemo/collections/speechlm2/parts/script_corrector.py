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
"""SpeechLM as a VERIFIER over a frozen ASR, instead of a second opinion to blend.

Score fusion only ever sees CHAT's distribution one position at a time, so it
cannot react to a hypothesis it has not finished reading. This conditions on the
COMPLETED chunk hypothesis, which is what lets a corrector fix an error that is
only visible once the phrase is whole -- a homophone that is wrong given the
following word, a mangled proper noun, a dropped article.

Per chunk the model sees

    instruction | text history | <vision_start> audio <vision_end> | <box_start> hypothesis <box_end>

and produces either

    ACCEPT                       CHAT's hypothesis stands
    <corrected text> <eot>       replace this chunk with that

WHY A SINGLE ACCEPT TOKEN. Measured on this data, 93% of chunks need no change.
Emitting one token for those, rather than regenerating text the model is about to
agree with, is what makes a verifier cheaper than the generator it supervises.
It also gives a clean quantity to threshold at inference: P(ACCEPT).

EVERY SPECIAL ID IS AN EXISTING QWEN TOKEN, repurposed. The embedding matrix is
never resized -- the same trick SCRIPT already uses for its audio delimiters, and
the reason a corrector can warm-start from a SCRIPT checkpoint at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

__all__ = [
    "CorrectorIds",
    "CorrectorExample",
    "build_corrector_example",
    "corrector_examples_for_utterance",
]


@dataclass(frozen=True)
class CorrectorIds:
    """Repurposed Qwen3 special ids. Defaults are unused-by-Qwen3 specials."""

    accept: int = 151646  # <|object_ref_start|>
    hyp_start: int = 151648  # <|box_start|>
    hyp_end: int = 151649  # <|box_end|>
    vision_start: int = 151652  # audio window open  (shared with SCRIPT)
    vision_end: int = 151653  # audio window close (shared with SCRIPT)
    eot: int = 151645  # <|im_end|>

    def validate(self, vocab_size: int) -> None:
        ids = [self.accept, self.hyp_start, self.hyp_end, self.vision_start, self.vision_end, self.eot]
        if len(set(ids)) != len(ids):
            raise ValueError(f"corrector special ids must be distinct, got {ids}")
        for i in ids:
            if not 0 <= i < vocab_size:
                raise ValueError(f"special id {i} outside vocabulary of {vocab_size}")


@dataclass
class CorrectorExample:
    """One chunk's training example.

    ``input_ids`` is the full sequence (prompt followed by target);
    ``labels`` is the same length with the PROMPT masked to ``ignore_index``, so
    loss is taken only on what the model must produce. Masking the prompt matters
    more here than for a plain generator: the prompt contains CHAT's hypothesis,
    and training the model to predict that text would teach it to imitate the ASR
    it is supposed to be checking.
    """

    input_ids: List[int]
    labels: List[int]
    is_accept: bool
    prompt_len: int


def build_corrector_example(
    instruction_ids: Sequence[int],
    history_ids: Sequence[int],
    audio_len: int,
    hyp_ids: Sequence[int],
    target_ids: Optional[Sequence[int]],
    ids: CorrectorIds = CorrectorIds(),
    audio_placeholder: int = 0,
    ignore_index: int = -100,
) -> CorrectorExample:
    """Build one chunk example.

    Args:
        instruction_ids: tokenized system prompt.
        history_ids: the transcript SO FAR. At training time this is the
            reference history, which is what inference approximates once earlier
            chunks have been corrected.
        audio_len: number of audio frames in this chunk's window. The frames
            themselves are spliced in by the model; ``audio_placeholder`` merely
            reserves the positions.
        hyp_ids: CHAT's tokens for this chunk. MAY BE EMPTY -- a chunk where the
            ASR emitted nothing is a real case and must still be checkable,
            since that is exactly how a deletion presents.
        target_ids: the corrected tokens, or ``None`` for ACCEPT.
    """
    ids.validate(vocab_size=10**9)  # bounds checked properly by the caller's tokenizer

    prompt: List[int] = list(instruction_ids) + list(history_ids)
    prompt += [ids.vision_start] + [audio_placeholder] * max(0, audio_len) + [ids.vision_end]
    prompt += [ids.hyp_start] + list(hyp_ids) + [ids.hyp_end]

    if target_ids is None:
        target = [ids.accept]
        is_accept = True
    else:
        # The eot terminates a correction. ACCEPT needs none: it is a single
        # token and its own terminator, which is the point of using one.
        target = list(target_ids) + [ids.eot]
        is_accept = False

    input_ids = prompt + target
    labels = [ignore_index] * len(prompt) + list(target)
    return CorrectorExample(input_ids=input_ids, labels=labels, is_accept=is_accept, prompt_len=len(prompt))


def corrector_examples_for_utterance(
    instruction_ids: Sequence[int],
    ref_chunk_words: Sequence[Sequence[str]],
    ref_chunk_ids: Sequence[Sequence[int]],
    hyp_chunk_ids: Sequence[Sequence[int]],
    hyp_chunk_words: Sequence[Sequence[str]],
    audio_lens: Sequence[int],
    ids: CorrectorIds = CorrectorIds(),
    ignore_index: int = -100,
) -> List[CorrectorExample]:
    """One utterance -> one example per chunk, labelled ACCEPT or corrected.

    HISTORY IS THE REFERENCE, not the hypothesis. That is the choice that makes
    the training distribution match inference: by the time the corrector sees
    chunk k at inference, chunks 0..k-1 have already been corrected, so their
    text is (intended to be) the reference. Conditioning on CHAT's raw history
    instead would train the model on a context it never meets, and would let one
    early ASR error poison every later chunk's example.

    Labels come from :func:`label_chunks`, so the timing rule applies: a
    hypothesis with the right words on different chunk boundaries is entirely
    ACCEPT. Which means ``hyp_chunk_words`` is FLATTENED before labelling -- the
    hypothesis's own chunk boundaries are deliberately not used.
    """
    from nemo.collections.asr.parts.utils.chunk_error_labels import label_chunks

    n = len(ref_chunk_words)
    if not (len(ref_chunk_ids) == len(hyp_chunk_ids) == len(hyp_chunk_words) == len(audio_lens) == n):
        raise ValueError(
            "per-chunk inputs disagree on length: "
            f"ref_words={len(ref_chunk_words)} ref_ids={len(ref_chunk_ids)} "
            f"hyp_ids={len(hyp_chunk_ids)} hyp_words={len(hyp_chunk_words)} audio={len(audio_lens)}"
        )

    flat_hyp = [w for c in hyp_chunk_words for w in c]
    labels, _ = label_chunks(flat_hyp, ref_chunk_words)

    out: List[CorrectorExample] = []
    history: List[int] = []
    for t in range(n):
        out.append(
            build_corrector_example(
                instruction_ids,
                history,
                audio_lens[t],
                hyp_chunk_ids[t],
                None if labels[t] is None else list(ref_chunk_ids[t]),
                ids=ids,
                ignore_index=ignore_index,
            )
        )
        # Advance on the REFERENCE, matching the conditioning above.
        history = history + list(ref_chunk_ids[t])
    return out
