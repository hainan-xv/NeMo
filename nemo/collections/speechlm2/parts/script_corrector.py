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
    "CorrectorBatch",
    "collate_corrector_examples",
    "decision_stats",
    "word_errors",
    "format_sample",
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


def _span_ids(words, tokenize, after_text: bool) -> List[int]:
    """Token ids for one chunk's reference span.

    The segments are concatenated AS IDS, so a span that follows text needs a
    leading space: without it the BPE glues the previous chunk's last word to
    this chunk's first and the detokenized transcript silently loses a word
    boundary -- the same class of bug as per-chunk detokenization, which measured
    41% WER against a true ~8%.
    """
    if not words:
        return []
    return list(tokenize((" " if after_text else "") + " ".join(words)))


def corrector_examples_for_utterance(
    instruction_ids: Sequence[int],
    ref_chunk_words: Sequence[Sequence[str]],
    hyp_chunk_words: Sequence[Sequence[str]],
    hyp_chunk_ids: Sequence[Sequence[int]],
    audio_lens: Sequence[int],
    tokenize,
    ids: CorrectorIds = CorrectorIds(),
    ignore_index: int = -100,
    normalize=None,
):
    """One utterance -> ``(examples, labels, target_ids)``, one entry per HYPOTHESIS chunk.

    The index space is the hypothesis's, because that is what the corrector is
    handed at inference: chunk k's example must judge the words the ASR actually
    emitted in chunk k. Labelling on the reference partition instead puts the
    ``<incorrect>`` mark on a neighbouring chunk whenever emission lags the
    aligner -- see :mod:`nemo.collections.asr.parts.utils.chunk_error_labels`.

    HISTORY IS THE CORRECTED TEXT, not the hypothesis. By the time the corrector
    sees chunk k at inference, chunks 0..k-1 have been accepted or rewritten, so
    their text is (intended to be) the reference words those chunks own.
    Conditioning on CHAT's raw history instead would train the model on a context
    it never meets, and would let one early ASR error poison every later chunk.

    ``target_ids`` is returned for EVERY chunk, accepted ones included: the
    caller needs those spans to stitch a transcript back together and to score
    it, and recomputing them would mean running the alignment twice.
    """
    from nemo.collections.asr.parts.utils.chunk_error_labels import label_chunks

    n = len(hyp_chunk_ids)
    if not (len(hyp_chunk_words) == len(audio_lens) == n):
        raise ValueError(
            "per-chunk inputs disagree on length: "
            f"hyp_words={len(hyp_chunk_words)} hyp_ids={len(hyp_chunk_ids)} audio={len(audio_lens)}"
        )

    labels, _, owned = label_chunks(hyp_chunk_words, ref_chunk_words, normalize=normalize)

    out: List[CorrectorExample] = []
    target_ids: List[List[int]] = []
    history: List[int] = []
    for t in range(n):
        seg = _span_ids(owned[t], tokenize, bool(history))
        target_ids.append(seg)
        out.append(
            build_corrector_example(
                instruction_ids,
                history,
                audio_lens[t],
                hyp_chunk_ids[t],
                None if labels[t] is None else seg,
                ids=ids,
                ignore_index=ignore_index,
            )
        )
        history = history + seg
    return out, labels, target_ids


@dataclass
class CorrectorBatch:
    """Padded batch, plus where each example's audio frames must be spliced in.

    ``audio_slots`` is ``[(row, position, frame_index)]``: the model replaces the
    placeholder embedding at ``(row, position)`` with encoder frame
    ``frame_index`` of that row's chunk. Carrying explicit positions rather than
    recomputing them from the ids is deliberate -- a placeholder id is an
    ORDINARY token id, so a search for it would also match real text that happens
    to use that id, silently scattering audio into the transcript.
    """

    input_ids: "List[List[int]]"
    labels: "List[List[int]]"
    attention_mask: "List[List[int]]"
    audio_slots: "List[tuple]"
    is_accept: "List[bool]"


def collate_corrector_examples(
    examples: Sequence[CorrectorExample],
    pad_id: int,
    ids: CorrectorIds = CorrectorIds(),
    ignore_index: int = -100,
) -> CorrectorBatch:
    """Right-pad to the longest example.

    Padding is MASKED IN THE LABELS as well as the attention mask. Both matter
    and for different reasons: the attention mask keeps pad out of the context,
    while the label mask keeps it out of the loss. Getting only the first right
    still trains the model to emit pad, which looks like a mysteriously high
    ACCEPT rate because pad and ACCEPT are both "short output".
    """
    if not examples:
        raise ValueError("no examples to collate")
    width = max(len(e.input_ids) for e in examples)

    input_ids, labels, attn, slots = [], [], [], []
    for r, e in enumerate(examples):
        pad = width - len(e.input_ids)
        input_ids.append(list(e.input_ids) + [pad_id] * pad)
        labels.append(list(e.labels) + [ignore_index] * pad)
        attn.append([1] * len(e.input_ids) + [0] * pad)

        # Audio positions: between this example's vision_start and vision_end.
        seq = e.input_ids
        lo = seq.index(ids.vision_start)
        hi = seq.index(ids.vision_end, lo + 1)
        for k, pos in enumerate(range(lo + 1, hi)):
            slots.append((r, pos, k))

    return CorrectorBatch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attn,
        audio_slots=slots,
        is_accept=[e.is_accept for e in examples],
    )


def decision_stats(pred_accept: Sequence[bool], label_accept: Sequence[bool]) -> dict:
    """Accept/reject quality, reported from the REJECT side.

    With ~93% of chunks needing no change, a model that always accepts scores
    93% accuracy and is worthless -- it never corrects anything. So the numbers
    that matter are recall and precision on REJECT: of the chunks that really
    were wrong, how many did it catch, and of those it flagged, how many really
    were wrong. ``pred_accept_frac`` is the collapse detector: it drifting to
    1.00 while the loss still falls is exactly the failure this imbalance
    invites.
    """
    if len(pred_accept) != len(label_accept):
        raise ValueError(f"length mismatch: {len(pred_accept)} vs {len(label_accept)}")
    n = len(label_accept)
    if n == 0:
        return {}
    tp = sum(1 for p, l in zip(pred_accept, label_accept) if not p and not l)  # correctly rejected
    fp = sum(1 for p, l in zip(pred_accept, label_accept) if not p and l)  # wrongly rejected
    fn = sum(1 for p, l in zip(pred_accept, label_accept) if p and not l)  # missed a real error
    return {
        "pred_accept_frac": sum(pred_accept) / n,
        "label_accept_frac": sum(label_accept) / n,
        "reject_precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "reject_recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def word_errors(hyp_words: Sequence[str], ref_words: Sequence[str]) -> tuple:
    """``(edits, ref_len)`` so WER can be accumulated ADDITIVELY across a batch.

    Returning the pair rather than a rate matters: corpus WER is total edits over
    total reference words, and averaging per-utterance rates instead over-weights
    short utterances. That is the same additive rule the leaderboard scorer uses,
    so these numbers stay comparable to the ones in the results table.
    """
    from nemo.collections.asr.parts.utils.chunk_error_labels import align_words

    ops = align_words(list(hyp_words), list(ref_words))
    edits = sum(1 for op, _, _ in ops if op != "equal")
    return edits, len(ref_words)


def format_sample(
    ref_chunks: Sequence[str],
    hyp_chunks: Sequence[str],
    labels: Sequence,
    step=None,
    preds: Optional[Sequence[bool]] = None,
    gens: Optional[Sequence[Optional[str]]] = None,
) -> str:
    """Human-readable dump of one utterance, label AND model prediction.

    Prints the hypothesis WITH ITS CHUNK BOUNDARIES, which is the thing no metric
    shows: the labeller deliberately ignores where the ASR broke its output, so a
    boundary that looks alarming next to the reference may be entirely correct.
    Seeing both side by side is the only way to tell a real error from a timing
    difference, and mistaking one for the other is the failure this whole
    labelling rule exists to avoid.

    ``preds``/``gens`` add what the MODEL currently says, beside what it should
    say. Aggregate metrics cannot distinguish "rejects the right 6%" from
    "rejects everything" once an oracle stitches the output back together -- the
    per-chunk view can, and a "!!" marks every disagreement so a collapse is
    visible at a glance rather than inferred from a rate.
    """
    head = f"=== corrector sample{'' if step is None else f' @ step {step}'} ==="
    lines = [
        head,
        "  ref : " + " | ".join(ref_chunks),
        "  hyp : " + " | ".join(hyp_chunks),
    ]
    n_agree = n_cmp = 0
    for t, lab in enumerate(labels):
        h = hyp_chunks[t] if t < len(hyp_chunks) else ""
        lab_s = "<correct>  " if lab is None else "<incorrect>"
        row = f"    chunk {t}: label {lab_s}"
        if preds is not None and t < len(preds):
            n_cmp += 1
            agree = (preds[t] is True) == (lab is None)
            n_agree += agree
            row += f" | pred {'<correct>  ' if preds[t] else '<incorrect>'}{'' if agree else ' !!'}"
        row += f" | hyp {h!r}"
        if lab is not None:
            row += f" -> {lab!r}"
        if gens is not None and t < len(gens) and gens[t] is not None:
            row += f" | model {gens[t]!r}"
        lines.append(row)
    if n_cmp:
        lines.append(f"    agreement: {n_agree}/{n_cmp} chunks")
    return "\n".join(lines)
