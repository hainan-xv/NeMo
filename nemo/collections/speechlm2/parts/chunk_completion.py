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
"""Chunk-completion (spine + branches) layout for streaming SpeechLM.

This implements the "conditional text-completion" formulation of streaming ASR:
for each audio chunk the model is asked to *extract the words carried by this
chunk, given the transcript so far*, i.e. it models

    p(words_k | text_history_<k, audio_k).

Instead of building one interleaved ``audio-text-audio-text`` causal stream, we
build a single packed sequence per utterance made of two kinds of tokens:

* A **spine**: ``[instruction] w_1 w_2 ... w_N`` — the pure-text history. It is a
  plain causal text sequence and is computed exactly once; every chunk reuses
  the relevant prefix of it (``instruction`` + the words emitted so far). Spine
  tokens NEVER attend to any audio, so their representation is pure text and
  reusable.
* One **branch** per chunk k: ``<vision_start> [audio_k frames] <vision_end>
  w_k <eot>``. A branch attends only to (a) its own history prefix of the spine
  ``[instruction] w_1..w_{k-1}``, (b) its own audio frames, and (c) its own
  earlier branch tokens — nothing else (not other branches, not other chunks'
  audio, not spine words at/after ``w_k``). The training loss is taken on the
  branch's target words.

Because a word must be *predicted from audio* in its branch (so it attends the
audio) but must serve as *pure-text history* in the spine (so it does not), each
word appears in both roles. The growing history is still computed once (the
spine), so training is a single O(L) forward, not O(N^2).

The key property, verified by the parity test, is that a branch's logits in the
packed sequence are bit-identical to running the standalone example
``[instruction] w_1..w_{k-1} <vision_start> audio_k <vision_end> w_k`` on its
own. That is what makes ``position_ids`` overlap across branches / spine safe:
the 4D mask keeps the branches isolated, and each branch's positions are exactly
those of its standalone example.

All builders here are pure tensor/logic (no model, no tokenizer), so they can be
unit-tested in isolation.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
from torch import Tensor

from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX, IGNORE_INDEX

# Segment id 0 is reserved for the spine; branches are 1..N.
SPINE_SEG_ID = 0


def _branch_positions(
    pref: int, window_len: int, n_words: int, contiguous_text_positions: bool
) -> List[int]:
    """RoPE position ids for one branch's tokens.

    Physical branch layout is ``<vs> [window_len audio] <ve> [n_words words] <eot>``
    (``n_bt = window_len + n_words + 3`` tokens; ``n_words = len(branch_words)``).

    Two conventions:

    * **Default** (``contiguous_text_positions=False``): positions are a plain
      contiguous run ``pref, pref+1, ..., pref+n_bt-1`` — the audio consumes
      ``window_len+2`` position slots *between* the history (ending at ``pref-1``)
      and the predicted words, so the transcript has a per-chunk positional gap.

    * **Contiguous text** (``contiguous_text_positions=True``, "Option A"): the
      words + eot are placed *contiguously with the history* at ``pref, pref+1,
      ...`` (word ``j`` -> ``pref+j``, eot -> ``pref+n_words``), so ``[history |
      words]`` reads as one uninterrupted stream. The prelude (``<vs>`` + audio +
      ``<ve>``) is overlaid *just before* ``pref`` (positions
      ``pref-(window_len+2) .. pref-1``, clamped at 0), overlapping the tail of
      the history's position range. This keeps the position-based causal rule of
      :func:`build_chunk_completion_mask` intact (audio positions ``< pref`` so
      the words still attend all their audio), and a word ends up at the *same*
      position as its spine twin. When ``pref < window_len+2`` (history shorter
      than the audio window, e.g. an early chunk with a large window) the earliest
      prelude positions clamp to 0 and tie; this is benign (it only makes those
      audio frames attend each other, never leaks history/words) but breaks
      exact parity with an order-causal standalone, so parity tests use configs
      with ``pref >= window_len+2``.
    """
    n_bt = window_len + n_words + 3  # <vs> + audio + <ve> + words + <eot>
    if not contiguous_text_positions:
        return [pref + off for off in range(n_bt)]
    n_prelude = window_len + 2  # <vs> + audio + <ve>
    prelude = [max(0, pref - n_prelude + i) for i in range(n_prelude)]
    words = [pref + j for j in range(n_words + 1)]  # words + eot, contiguous with history
    return prelude + words


def _audio_window_start(cur_chunk_start: int, win_end: int, m_window_start: int, audio_window_frames: int) -> int:
    """Global start frame of a branch's audio window (window is ``[start, win_end)``).

    Two ways to size the window (see :func:`build_packed_chunk_example`):

    * ``audio_window_frames > 0`` (FRAME-based): the last ``audio_window_frames``
      encoder frames ending at the chunk boundary ``win_end`` — a fixed acoustic
      context independent of the chunk size. The start is never past the current
      chunk's own start (``cur_chunk_start``), so a chunk LARGER than the window
      still shows all of its own frames (the frame count is a floor, not a cap),
      and it is clamped at 0 for the first chunk(s). Takes precedence over
      ``audio_history_chunks``.
    * otherwise (CHUNK-based): ``m_window_start`` = the start of chunk
      ``max(0, k - audio_history_chunks)``.
    """
    if audio_window_frames > 0:
        return max(0, min(cur_chunk_start, win_end - audio_window_frames))
    return m_window_start


@dataclass
class ChunkSpec:
    """One chunk of an utterance.

    Args:
        audio_len: number of encoder audio frames in this chunk (``C_k``).
        target_ids: token ids of the words revealed by this chunk (``w_k``); may
            be empty for a silent chunk (the branch then only emits ``eot_id``).
        last_word_ids: token ids of just this chunk's LAST word (a trailing slice
            of ``target_ids``). Only needed for history-word-recovery
            regularization; empty otherwise.
    """

    audio_len: int
    target_ids: List[int] = field(default_factory=list)
    last_word_ids: List[int] = field(default_factory=list)


@dataclass
class PackedChunkExample:
    """A single utterance packed as spine + per-chunk branches.

    All tensors are 1-D of length ``T`` (the packed sequence length). Batch them
    with :func:`collate_packed_chunk_examples`.

    Attributes:
        input_ids: (T,) token ids; audio-frame positions hold ``AUDIO_TOKEN_IDX``.
        position_ids: (T,) RoPE position id per token (spine = its index; a branch
            token = ``prefix_len_k + local_offset``).
        seg_ids: (T,) 0 for spine tokens, ``k>=1`` for branch-k tokens.
        prefix_len: (T,) for a branch token, the length of its chunk's history
            prefix (``= m + sum_{j<k} |w_j|``); 0 for spine tokens.
        target_ids: (T,) next-token targets; ``IGNORE_INDEX`` everywhere except
            the branch positions that predict the chunk's words + end-of-turn.
        is_audio: (T,) True at audio-frame positions.
        audio_frame_index: (T,) global encoder-frame index each audio position
            maps to (``-1`` at non-audio positions). With ``audio_history_chunks>0``
            a branch's window spans multiple chunks and frames are reused across
            branches, so the fill is an explicit gather by this index rather than a
            positional cumsum. For ``audio_history_chunks==0`` it is exactly
            ``0,1,2,...`` (i.e. equivalent to the cumsum fill).
        spine_len: number of spine tokens ``P`` (spine occupies ``input_ids[:P]``).
    """

    input_ids: Tensor
    position_ids: Tensor
    seg_ids: Tensor
    prefix_len: Tensor
    target_ids: Tensor
    is_audio: Tensor
    audio_frame_index: Tensor
    spine_len: int


def build_packed_chunk_example(
    instruction_ids: List[int],
    chunks: List[ChunkSpec],
    vision_start_id: int,
    vision_end_id: int,
    eot_id: int,
    supervise_eot: bool = True,
    audio_history_chunks: int = 0,
    recover_prev: Optional[List[bool]] = None,
    contiguous_text_positions: bool = False,
    audio_window_frames: int = 0,
    corrupt_prev: Optional[List[Optional[List[int]]]] = None,
    delete_id: Optional[int] = None,
) -> PackedChunkExample:
    """Build the packed spine+branch layout for one utterance.

    Layout (physical order): the spine first, then the branches in chunk order::

        [instruction w_1 w_2 ... w_N]  (spine)
        [<vs> A_1.. <ve> w_1 <eot>]    (branch 1)
        [<vs> A_2.. <ve> w_2 <eot>]    (branch 2)
        ...

    Args:
        instruction_ids: token ids of the instruction/system prompt prefix.
        chunks: per-chunk specs (audio frame count + revealed word token ids).
        vision_start_id / vision_end_id: delimiter token ids wrapping the audio.
        eot_id: end-of-turn token id appended to each branch (what the model
            emits when a chunk has no more words).
        supervise_eot: if True, the branch predicts ``eot_id`` after its last
            word (so the model learns to stop); the ``eot`` position itself is
            never supervised.
        audio_history_chunks: number ``M`` of PREVIOUS chunks whose audio is also
            included in each branch's window. Branch ``k`` then wraps the frames of
            chunks ``[max(0, k-M) .. k]`` (fewer for early chunks; no padding). ``0``
            = original behavior (only the current chunk's audio).
        audio_window_frames: if ``> 0``, size each branch's audio window by a FIXED
            number of encoder frames (the last ``audio_window_frames`` frames ending
            at the chunk boundary) instead of by whole chunks — a constant acoustic
            context regardless of chunk size. Never smaller than the current chunk
            (so chunks larger than the window keep all their frames). Takes
            precedence over ``audio_history_chunks``; ``0`` = disabled. See
            :func:`_audio_window_start`.
        recover_prev: optional per-chunk bool list. When ``recover_prev[k]`` is
            True (needs a non-empty previous-chunk last word), branch ``k`` DROPS
            the previous chunk's last word from its history and prepends it to its
            target, so the model learns to recover a missing history word. With
            ``M>=1`` that word's audio is inside branch ``k``'s window; with
            ``M==0`` it is not, so recovery must come from the current chunk's
            (left-context-carrying) audio + the remaining text history. The spine
            and all OTHER branches (incl. the one that normally emits that word)
            are unchanged.
        corrupt_prev: optional per-chunk list for self-correction. When
            ``corrupt_prev[k]`` is a non-empty token list ``W'`` (and ``delete_id``
            is given and the previous chunk has a non-empty last word), branch ``k``
            is built as if the previous chunk's last word had been mis-committed as
            ``W'``: the correct previous word is excluded from the attended history,
            ``W'`` is shown as this branch's history tail, and the target becomes
            ``[delete_id] + w_prev + w_k`` (delete the wrong word, re-emit the truth,
            then this chunk's words). The spine and all other branches are unchanged.
            Mutually exclusive with ``recover_prev``; non-contiguous layout only.
        delete_id: the special "delete last word" token id used by ``corrupt_prev``.
        contiguous_text_positions: when True (Option A), each branch's predicted
            words + eot get positions ``pref, pref+1, ...`` (contiguous with the
            history), and the branch's ``<vs>``/audio/``<ve>`` prelude is overlaid
            just before ``pref`` — so the transcript reads as one uninterrupted
            stream with the audio parked on the history's tail positions. False
            (default) keeps the original contiguous-branch positions (audio
            consumes position slots between history and words). See
            :func:`_branch_positions`.

    Returns:
        PackedChunkExample.
    """
    m = len(instruction_ids)
    M = max(int(audio_history_chunks), 0)

    # --- spine: instruction followed by all chunk words, in order ---
    spine_ids: List[int] = list(instruction_ids)
    prefix_lens: List[int] = []  # history-prefix length per chunk
    frame_starts: List[int] = []  # global encoder-frame index where each chunk begins
    running = m
    running_frames = 0
    for ch in chunks:
        prefix_lens.append(running)
        frame_starts.append(running_frames)
        spine_ids.extend(ch.target_ids)
        running += len(ch.target_ids)
        running_frames += ch.audio_len
    P = len(spine_ids)

    input_ids: List[int] = list(spine_ids)
    position_ids: List[int] = list(range(P))
    seg_ids: List[int] = [SPINE_SEG_ID] * P
    prefix_len: List[int] = [0] * P
    target_ids: List[int] = [IGNORE_INDEX] * P  # spine is context only, no loss
    is_audio: List[bool] = [False] * P
    audio_frame_index: List[int] = [-1] * P

    # --- branches ---
    for kc, ch in enumerate(chunks):  # kc: 0-based chunk index
        k = kc + 1  # 1-based branch/segment id
        pref = prefix_lens[kc]

        # Audio window (global frame indices): chunk-based (audio_history_chunks) or
        # fixed frame count (audio_window_frames), ending at the chunk boundary.
        win_end = frame_starts[kc] + ch.audio_len
        win_start = _audio_window_start(
            frame_starts[kc], win_end, frame_starts[max(0, kc - M)], audio_window_frames
        )
        window_frames = list(range(win_start, win_end))
        window_len = len(window_frames)

        # Per-branch history edit (mutually exclusive, both need a previous chunk
        # with a non-empty last word, both only in the non-contiguous layout):
        #  * recovery: DROP the prev chunk's correct last word and re-emit it.
        #  * self-correction: the prev chunk's last word was mis-committed as W'
        #    (corrupt_prev[kc]); show W' as this branch's history tail and target
        #    ``<del> w_prev w_k`` -- delete the wrong word and re-emit the truth.
        do_recover = (
            recover_prev is not None
            and kc >= 1
            and recover_prev[kc]
            and len(chunks[kc - 1].last_word_ids) > 0
        )
        do_corrupt = (
            corrupt_prev is not None
            and kc >= 1
            and delete_id is not None
            and corrupt_prev[kc] is not None
            and len(corrupt_prev[kc]) > 0
            and len(chunks[kc - 1].last_word_ids) > 0
        )
        if do_recover and do_corrupt:
            raise ValueError("a chunk cannot both recover and self-correct its previous word")
        if (do_recover or do_corrupt) and contiguous_text_positions:
            raise ValueError("recovery / self-correction are not supported with contiguous_text_positions")

        context: List[int] = []  # unsupervised branch-leading tokens (the mis-committed word W')
        if do_corrupt:
            wprime = list(corrupt_prev[kc])
            wprev = list(chunks[kc - 1].last_word_ids)
            pref = pref - len(wprev)  # exclude the correct prev word from the attended history
            context = wprime  # the wrong word, shown as the (committed) history tail
            branch_words = [delete_id] + wprev + list(ch.target_ids)
        elif do_recover:
            dropped = list(chunks[kc - 1].last_word_ids)
            pref = pref - len(dropped)  # exclude the dropped word's spine tokens
            branch_words = dropped + list(ch.target_ids)
        else:
            branch_words = list(ch.target_ids)

        # branch tokens: [W' context] <vs> [window audio] <ve> [branch_words] <eot>
        branch_tokens: List[int] = (
            list(context) + [vision_start_id] + [AUDIO_TOKEN_IDX] * window_len + [vision_end_id]
        )
        branch_is_audio: List[bool] = [False] * len(context) + [False] + [True] * window_len + [False]
        branch_frame_idx: List[int] = [-1] * len(context) + [-1] + window_frames + [-1]
        branch_tokens.extend(branch_words)
        branch_is_audio.extend([False] * len(branch_words))
        branch_frame_idx.extend([-1] * len(branch_words))
        branch_tokens.append(eot_id)
        branch_is_audio.append(False)
        branch_frame_idx.append(-1)

        # Next-token targets: <ve> predicts branch_words[0]; word i predicts word
        # i+1; the last word predicts eot. context/<vs>/audio/eot are not supervised.
        n_bt = len(branch_tokens)
        next_targets: List[int] = [IGNORE_INDEX] * n_bt
        ve_idx = len(context) + 1 + window_len  # index of <ve> within the branch
        first_word_pos = ve_idx  # <ve> predicts the first (delete/recovered/normal) target
        for j, tok in enumerate(branch_words):
            next_targets[first_word_pos + j] = tok
        last_word_pos = first_word_pos + len(branch_words)  # predicts eot
        if supervise_eot:
            next_targets[last_word_pos] = eot_id

        if context:
            # W' occupies the history-tail positions pref..pref+|W'|-1; the rest follows.
            branch_positions = [pref + i for i in range(len(context))] + _branch_positions(
                pref + len(context), window_len, len(branch_words), contiguous_text_positions
            )
        else:
            branch_positions = _branch_positions(pref, window_len, len(branch_words), contiguous_text_positions)

        input_ids.extend(branch_tokens)
        position_ids.extend(branch_positions)
        seg_ids.extend([k] * n_bt)
        prefix_len.extend([pref] * n_bt)
        target_ids.extend(next_targets)
        is_audio.extend(branch_is_audio)
        audio_frame_index.extend(branch_frame_idx)

    return PackedChunkExample(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        position_ids=torch.tensor(position_ids, dtype=torch.long),
        seg_ids=torch.tensor(seg_ids, dtype=torch.long),
        prefix_len=torch.tensor(prefix_len, dtype=torch.long),
        target_ids=torch.tensor(target_ids, dtype=torch.long),
        is_audio=torch.tensor(is_audio, dtype=torch.bool),
        audio_frame_index=torch.tensor(audio_frame_index, dtype=torch.long),
        spine_len=P,
    )


def build_chunk_completion_mask(
    seg_ids: Tensor,
    position_ids: Tensor,
    prefix_len: Tensor,
    valid: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    """Build the 4D additive attention mask for a packed chunk-completion batch.

    A query at position ``q`` may attend to key ``j`` iff the key is valid and one
    of:

    * **spine query, spine key**: causal within the spine (``pos[j] <= pos[q]``).
      Spine tokens attend only other spine tokens -> pure-text history.
    * **branch query, spine key**: ``j`` is in the branch's history prefix, i.e.
      ``pos[j] < prefix_len[q]``. (Spine positions equal their index, so this is
      exactly ``instruction + w_1..w_{k-1}``.)
    * **branch query, same-branch key**: causal within the branch
      (``pos[j] <= pos[q]``) — covers the branch's own audio + earlier words.

    Everything else is blocked: a branch never sees another branch, another
    chunk's audio, or spine words at/after its own; the spine never sees any
    branch/audio.

    Args:
        seg_ids: (B, T) 0 = spine, >=1 = branch id.
        position_ids: (B, T) RoPE positions (also used for causality/prefix).
        prefix_len: (B, T) per-branch-token history-prefix length (0 for spine).
        valid: (B, T) bool, False at padding.
        dtype: floating dtype for the additive mask.

    Returns:
        (B, 1, T, T) additive mask: 0 where allowed, ``finfo(dtype).min`` where blocked.
    """
    q_seg = seg_ids[:, :, None]  # (B, T, 1)
    k_seg = seg_ids[:, None, :]  # (B, 1, T)
    q_pos = position_ids[:, :, None]
    k_pos = position_ids[:, None, :]
    q_prefix = prefix_len[:, :, None]
    k_valid = valid[:, None, :]

    q_is_spine = q_seg == SPINE_SEG_ID
    k_is_spine = k_seg == SPINE_SEG_ID
    causal = k_pos <= q_pos
    same_branch = (q_seg == k_seg) & (~q_is_spine)

    spine_to_spine = q_is_spine & k_is_spine & causal
    branch_to_prefix = (~q_is_spine) & k_is_spine & (k_pos < q_prefix)
    branch_to_own = same_branch & causal

    allowed = (spine_to_spine | branch_to_prefix | branch_to_own) & k_valid  # (B, T, T)

    additive = torch.zeros_like(allowed, dtype=dtype)
    additive = additive.masked_fill(~allowed, torch.finfo(dtype).min)
    return additive.unsqueeze(1)  # (B, 1, T, T)


@dataclass
class BatchedPackedChunk:
    """A right-padded batch of :class:`PackedChunkExample`."""

    input_ids: Tensor  # (B, T)
    position_ids: Tensor  # (B, T)
    seg_ids: Tensor  # (B, T)
    prefix_len: Tensor  # (B, T)
    target_ids: Tensor  # (B, T)
    is_audio: Tensor  # (B, T)
    audio_frame_index: Tensor  # (B, T) global encoder-frame index at audio positions, -1 elsewhere
    valid: Tensor  # (B, T) bool, False at right-padding
    spine_lens: Tensor  # (B,)


def collate_packed_chunk_examples(
    examples: List[PackedChunkExample],
    pad_id: int,
) -> BatchedPackedChunk:
    """Right-pad a list of packed examples into a batch.

    Padding tokens are text ``pad_id`` with ``seg_ids = -1`` (never matches a real
    segment), ``valid = False`` (so they are masked out as keys).
    """
    B = len(examples)
    T = max(int(ex.input_ids.numel()) for ex in examples)

    def _pad(vals: List[Tensor], pad_value, dtype) -> Tensor:
        out = torch.full((B, T), pad_value, dtype=dtype)
        for i, v in enumerate(vals):
            out[i, : v.numel()] = v.to(dtype)
        return out

    input_ids = _pad([e.input_ids for e in examples], pad_id, torch.long)
    position_ids = _pad([e.position_ids for e in examples], 0, torch.long)
    seg_ids = _pad([e.seg_ids for e in examples], -1, torch.long)  # -1 = padding segment
    prefix_len = _pad([e.prefix_len for e in examples], 0, torch.long)
    target_ids = _pad([e.target_ids for e in examples], IGNORE_INDEX, torch.long)
    is_audio = _pad([e.is_audio for e in examples], False, torch.bool)
    audio_frame_index = _pad([e.audio_frame_index for e in examples], -1, torch.long)

    valid = torch.zeros((B, T), dtype=torch.bool)
    for i, e in enumerate(examples):
        valid[i, : e.input_ids.numel()] = True

    spine_lens = torch.tensor([e.spine_len for e in examples], dtype=torch.long)

    return BatchedPackedChunk(
        input_ids=input_ids,
        position_ids=position_ids,
        seg_ids=seg_ids,
        prefix_len=prefix_len,
        target_ids=target_ids,
        is_audio=is_audio,
        audio_frame_index=audio_frame_index,
        valid=valid,
        spine_lens=spine_lens,
    )


@dataclass
class SeparateChunkExample:
    """One standalone per-chunk example (the reference formulation).

    ``[instruction w_1..w_{k-1}] <vs> audio_window <ve> w_k <eot>`` with plain
    causal attention. ``position_ids`` are ``0..len-1`` in the default convention;
    under ``contiguous_text_positions`` they match the packed branch's overlaid
    scheme (the prefix is ``0..pref-1`` and the branch follows
    :func:`_branch_positions`), so the parity test must pass them to the model.
    """

    input_ids: Tensor  # (L,)
    target_ids: Tensor  # (L,)
    is_audio: Tensor  # (L,)
    audio_frame_index: Tensor  # (L,) global encoder-frame index at audio positions, -1 elsewhere
    position_ids: Tensor  # (L,) RoPE positions (0..L-1 by default; overlaid under contiguous mode)
    # index range [branch_start, L) of this example's branch (audio + words + eot)
    branch_start: int


def build_separate_chunk_examples(
    instruction_ids: List[int],
    chunks: List[ChunkSpec],
    vision_start_id: int,
    vision_end_id: int,
    eot_id: int,
    supervise_eot: bool = True,
    audio_history_chunks: int = 0,
    recover_prev: Optional[List[bool]] = None,
    contiguous_text_positions: bool = False,
    audio_window_frames: int = 0,
    corrupt_prev: Optional[List[Optional[List[int]]]] = None,
    delete_id: Optional[int] = None,
) -> List[SeparateChunkExample]:
    """Build the naive per-chunk examples used as a correctness oracle.

    Each chunk becomes an independent standard-causal example whose history is
    the *plain text* ``instruction + w_1..w_{k-1}`` (minus the previous chunk's
    last word when recovering), with the same audio window as the packed branch.
    The packed builder must reproduce these branches' logits exactly (parity test).

    ``contiguous_text_positions`` and ``audio_window_frames`` must match the packed
    builder: the former selects the position-id convention (see
    :func:`_branch_positions`, written into each example's ``position_ids``), the
    latter the audio-window sizing (see :func:`_audio_window_start`).
    """
    M = max(int(audio_history_chunks), 0)
    frame_starts: List[int] = []
    running_frames = 0
    for ch in chunks:
        frame_starts.append(running_frames)
        running_frames += ch.audio_len

    out: List[SeparateChunkExample] = []
    history: List[int] = list(instruction_ids)
    for kc, ch in enumerate(chunks):
        win_end = frame_starts[kc] + ch.audio_len
        win_start = _audio_window_start(
            frame_starts[kc], win_end, frame_starts[max(0, kc - M)], audio_window_frames
        )
        window_frames = list(range(win_start, win_end))
        window_len = len(window_frames)

        do_recover = (
            recover_prev is not None
            and kc >= 1
            and recover_prev[kc]
            and len(chunks[kc - 1].last_word_ids) > 0
        )
        do_corrupt = (
            corrupt_prev is not None
            and kc >= 1
            and delete_id is not None
            and corrupt_prev[kc] is not None
            and len(corrupt_prev[kc]) > 0
            and len(chunks[kc - 1].last_word_ids) > 0
        )
        context: List[int] = []
        if do_corrupt:
            wprime = list(corrupt_prev[kc])
            wprev = list(chunks[kc - 1].last_word_ids)
            prefix = list(history[: len(history) - len(wprev)])  # drop the correct prev word
            context = wprime  # show the mis-committed word as the history tail
            branch_words = [delete_id] + wprev + list(ch.target_ids)
        elif do_recover:
            dropped = list(chunks[kc - 1].last_word_ids)
            prefix = list(history[: len(history) - len(dropped)])  # drop prev chunk's last word
            branch_words = dropped + list(ch.target_ids)
        else:
            prefix = list(history)
            branch_words = list(ch.target_ids)

        branch_tokens = list(context) + [vision_start_id] + [AUDIO_TOKEN_IDX] * window_len + [vision_end_id]
        branch_is_audio = [False] * len(context) + [False] + [True] * window_len + [False]
        branch_frame_idx = [-1] * len(context) + [-1] + window_frames + [-1]
        branch_tokens.extend(branch_words)
        branch_is_audio.extend([False] * len(branch_words))
        branch_frame_idx.extend([-1] * len(branch_words))
        branch_tokens.append(eot_id)
        branch_is_audio.append(False)
        branch_frame_idx.append(-1)

        input_ids = prefix + branch_tokens
        is_audio = [False] * len(prefix) + branch_is_audio
        audio_frame_index = [-1] * len(prefix) + branch_frame_idx

        target_ids = [IGNORE_INDEX] * len(input_ids)
        ve_idx = len(prefix) + len(context) + 1 + window_len  # position of <ve>
        for j, tok in enumerate(branch_words):
            target_ids[ve_idx + j] = tok
        last_word_pos = ve_idx + len(branch_words)
        if supervise_eot:
            target_ids[last_word_pos] = eot_id

        # Positions: prefix is 0..pref-1; W' context (if any) takes the history-tail
        # positions; the branch follows the same convention as the packed builder.
        pref = len(prefix)
        if context:
            position_ids = (
                list(range(pref))
                + [pref + i for i in range(len(context))]
                + _branch_positions(pref + len(context), window_len, len(branch_words), contiguous_text_positions)
            )
        else:
            position_ids = list(range(pref)) + _branch_positions(
                pref, window_len, len(branch_words), contiguous_text_positions
            )

        out.append(
            SeparateChunkExample(
                audio_frame_index=torch.tensor(audio_frame_index, dtype=torch.long),
                input_ids=torch.tensor(input_ids, dtype=torch.long),
                target_ids=torch.tensor(target_ids, dtype=torch.long),
                is_audio=torch.tensor(is_audio, dtype=torch.bool),
                position_ids=torch.tensor(position_ids, dtype=torch.long),
                branch_start=len(prefix),
            )
        )
        # append this chunk's words to the running plain-text history
        history.extend(ch.target_ids)
    return out


@torch.no_grad()
def stream_decode_chunk_completion(
    llm,
    embed_tokens: Callable[[Tensor], Tensor],
    instruction_ids: List[int],
    frames: Tensor,
    chunk_size: int,
    vision_start_id: int,
    vision_end_id: int,
    eot_id: int,
    max_new_tokens: int = 64,
    device: Optional[torch.device] = None,
    audio_history_chunks: int = 0,
    contiguous_text_positions: bool = False,
    max_history_tokens: int = 0,
    audio_window_frames: int = 0,
) -> List[List[int]]:
    """Greedy streaming decode of one utterance in the chunk-completion model.

    Mirrors the packed training layout exactly, so greedy decoding here matches a
    teacher-forced packed forward of the emitted tokens (see the parity test):

    * Keep a growing **plain-text spine KV cache** = ``instruction + emitted words``.
    * For each chunk: run ``<vs> audio <ve>`` on top of the spine KV (positions
      offset to ``spine_len``), greedily decode words until ``eot`` (or
      ``max_new_tokens``).
    * **Crop** the branch KV (audio + delimiters + generated words) back off the
      cache — audio is never persisted — then re-append the emitted words as
      **plain text** at spine positions ``spine_len..``. This makes the history
      the model conditions on at inference identical to the spine at training.

    Args:
        llm: an HF causal LM taking ``inputs_embeds``, ``position_ids``,
            ``past_key_values``, ``use_cache=True`` and returning ``.logits`` +
            ``.past_key_values``.
        embed_tokens: maps a ``(1, n)`` long tensor of ids to ``(1, n, H)`` embeddings.
        instruction_ids: instruction/system-prompt token ids (the spine root).
        frames: ``(T_enc, H)`` audio encoder frame embeddings for this utterance.
        chunk_size: number of encoder frames per chunk.
        vision_start_id / vision_end_id / eot_id: delimiter + end-of-turn ids.
        max_new_tokens: max words decoded per chunk.

    Returns:
        A list (per chunk) of emitted token-id lists (``eot`` excluded).
    """
    from transformers.cache_utils import DynamicCache

    if device is None:
        device = frames.device

    def embed_ids(ids: List[int]) -> Tensor:
        return embed_tokens(torch.tensor(ids, dtype=torch.long, device=device)[None])  # (1, n, H)

    spine = DynamicCache()
    instr_emb = embed_ids(list(instruction_ids))
    dtype = instr_emb.dtype
    frames = frames.to(device=device, dtype=dtype)

    pos = torch.arange(len(instruction_ids), device=device)[None]
    out = llm(inputs_embeds=instr_emb, position_ids=pos, past_key_values=spine, use_cache=True, return_dict=True)
    spine = out.past_key_values
    spine_len = len(instruction_ids)

    n_frames = frames.shape[0]
    num_chunks = math.ceil(n_frames / chunk_size) if n_frames > 0 else 0
    M = max(int(audio_history_chunks), 0)
    instr_len = len(instruction_ids)

    emitted_flat: List[int] = []
    emitted_per_chunk: List[List[int]] = []
    W = max(int(audio_window_frames), 0)
    for k in range(num_chunks):
        # Audio window: chunk-based (audio_history_chunks) or fixed frame count
        # (audio_window_frames), ending at this chunk's boundary.
        win_end = (k + 1) * chunk_size
        win_start = _audio_window_start(k * chunk_size, win_end, max(0, k - M) * chunk_size, W)
        cf = frames[win_start:win_end]
        c = cf.shape[0]
        if c == 0:
            break
        # branch prelude: <vs> [audio] <ve>
        prelude = torch.cat(
            [embed_ids([vision_start_id])[0], cf, embed_ids([vision_end_id])[0]], dim=0
        )[None]  # (1, c+2, H)
        # Positions: default = spine_len..spine_len+c+1 (audio consumes slots, words
        # follow after). Contiguous (Option A) = prelude overlaid just before
        # spine_len (clamped >=0) so the first word lands at spine_len, contiguous
        # with the history.
        if contiguous_text_positions:
            bpos = torch.tensor(
                [[max(0, spine_len - (c + 2) + i) for i in range(c + 2)]], device=device
            )
            cur = spine_len  # first decoded word position
        else:
            bpos = torch.arange(spine_len, spine_len + c + 2, device=device)[None]
            cur = spine_len + c + 2
        out = llm(inputs_embeds=prelude, position_ids=bpos, past_key_values=spine, use_cache=True, return_dict=True)
        cache = out.past_key_values
        logits = out.logits[:, -1]  # position of <ve>: predicts first word

        words: List[int] = []
        for _ in range(max_new_tokens):
            nxt = int(logits.argmax(dim=-1).item())
            if nxt == eot_id:
                break
            words.append(nxt)
            temb = embed_ids([nxt])
            out = llm(
                inputs_embeds=temb,
                position_ids=torch.tensor([[cur]], device=device),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out.past_key_values
            cur += 1
            logits = out.logits[:, -1]

        # Evict the branch KV (audio + delimiters + generated words).
        cache.crop(spine_len)
        spine = cache

        # Commit the emitted words to the spine as PLAIN TEXT at spine positions.
        if words:
            wemb = embed_ids(words)
            wpos = torch.arange(spine_len, spine_len + len(words), device=device)[None]
            out = llm(
                inputs_embeds=wemb, position_ids=wpos, past_key_values=spine, use_cache=True, return_dict=True
            )
            spine = out.past_key_values
            spine_len += len(words)
        emitted_flat.extend(words)

        # Optional max-history cap: keep only the most recent ``max_history_tokens``
        # emitted tokens as spine history (instruction always kept). Rebuild the
        # spine (crop to instruction, re-prefill the kept tail at contiguous
        # positions) so the retained history matches the batched decoder's capped
        # prefill. Bounds cost to linear in duration.
        if max_history_tokens and (spine_len - instr_len) > max_history_tokens:
            kept = emitted_flat[-max_history_tokens:]
            spine.crop(instr_len)
            kemb = embed_ids(kept)
            kpos = torch.arange(instr_len, instr_len + len(kept), device=device)[None]
            out = llm(inputs_embeds=kemb, position_ids=kpos, past_key_values=spine, use_cache=True, return_dict=True)
            spine = out.past_key_values
            spine_len = instr_len + len(kept)

        emitted_per_chunk.append(words)

    return emitted_per_chunk


@torch.no_grad()
def batched_stream_decode_chunk_completion(
    llm,
    embed_tokens: Callable[[Tensor], Tensor],
    instruction_ids_list: List[List[int]],
    frames_list: List[Tensor],
    chunk_size: int,
    vision_start_id: int,
    vision_end_id: int,
    eot_id: int,
    pad_id: int,
    max_new_tokens: int = 64,
    device: Optional[torch.device] = None,
    audio_history_chunks: int = 0,
    contiguous_text_positions: bool = False,
    max_history_tokens: int = 0,
    return_chunk_ids: bool = False,
    audio_window_frames: int = 0,
    delete_id: Optional[int] = None,
    is_word_start: Optional[Callable[[int], bool]] = None,
):
    """Batched greedy chunk-completion decode for B utterances at once.

    Chunk-synchronous: for chunk index ``k`` every still-active stream is decoded
    together. For each stream the model sees, per chunk, the *plain-text history*
    plus only that chunk's audio -- exactly the training conditioning
    ``p(words_k | text_history_<k, audio_k)`` -- built as::

        [instruction] [emitted words so far] <vs> [audio_k frames] <ve> -> words

    Because the history is compact text (not audio), re-prefilling it each chunk
    is cheap; this avoids the fragile variable-length KV surgery a persistent
    per-stream spine cache would need across a batch, while keeping ``position_ids``
    contiguous per stream (identical to training). Equivalent to running
    :func:`stream_decode_chunk_completion` on each utterance independently (see
    the batched-parity test).

    Args:
        llm / embed_tokens: as in :func:`stream_decode_chunk_completion`.
        instruction_ids_list: per-utterance instruction token ids.
        frames_list: per-utterance ``(T_enc_b, H)`` encoder frame embeddings.
        chunk_size: encoder frames per chunk.
        vision_start_id / vision_end_id / eot_id / pad_id: delimiter / end / pad ids.
        max_new_tokens: max words decoded per chunk per stream.

    Returns:
        ``B`` lists of emitted token ids (flattened across chunks; ``eot`` excluded).
    """
    B = len(frames_list)
    if B == 0:
        return []
    if device is None:
        device = frames_list[0].device
    H = frames_list[0].shape[-1]
    dtype = embed_tokens(torch.zeros(1, 1, dtype=torch.long, device=device)).dtype

    n_frames = [int(f.shape[0]) for f in frames_list]
    num_chunks = [math.ceil(n / chunk_size) if n > 0 else 0 for n in n_frames]
    max_chunks = max(num_chunks) if B else 0
    M = max(int(audio_history_chunks), 0)
    W = max(int(audio_window_frames), 0)

    emitted: List[List[int]] = [[] for _ in range(B)]
    # Per emitted token, the audio chunk index during which it was decoded (for
    # word-latency: a word's emission time = end of the chunk of its last subword).
    chunk_ids: List[List[int]] = [[] for _ in range(B)]
    # Self-correction: token index in emitted[b] where each committed word begins,
    # so a leading <del> can pop the last committed word. Only tracked when the
    # delete token is active.
    word_starts: List[List[int]] = [[] for _ in range(B)]

    def _append_token(b: int, tok: int, k: int) -> None:
        if not emitted[b] or (is_word_start is not None and is_word_start(tok)):
            word_starts[b].append(len(emitted[b]))
        emitted[b].append(tok)
        chunk_ids[b].append(k)

    def _pop_last_word(b: int) -> None:
        if word_starts[b]:
            start = word_starts[b].pop()
            del emitted[b][start:]
            del chunk_ids[b][start:]

    for k in range(max_chunks):
        active = [b for b in range(B) if k < num_chunks[b]]
        if not active:
            break
        na = len(active)

        # --- build per-stream prefill: instr + emitted + <vs> audio_k <ve> ---
        seqs: List[List[int]] = []
        chunk_frames: List[Tensor] = []
        for b in active:
            # Audio window: chunk-based (audio_history_chunks) or fixed frame count
            # (audio_window_frames), ending at this chunk's boundary.
            win_end = (k + 1) * chunk_size
            win_start = _audio_window_start(k * chunk_size, win_end, max(0, k - M) * chunk_size, W)
            fr = frames_list[b][win_start:win_end].to(device=device, dtype=dtype)
            c = int(fr.shape[0])
            # Optionally cap the CONDITIONING history to the most recent
            # ``max_history_tokens`` emitted tokens (instruction always kept). This
            # bounds the per-chunk prefill so cost is linear in duration instead of
            # quadratic; all emitted tokens are still recorded for the output.
            hist = emitted[b]
            if max_history_tokens and len(hist) > max_history_tokens:
                hist = hist[-max_history_tokens:]
            toks = (
                list(instruction_ids_list[b])
                + list(hist)
                + [vision_start_id]
                + [AUDIO_TOKEN_IDX] * c
                + [vision_end_id]
            )
            seqs.append(toks)
            chunk_frames.append(fr)

        L = max(len(s) for s in seqs)
        max_c = max(int(fr.shape[0]) for fr in chunk_frames)

        # Left-pad so every row's last position is the <ve> (shared query column).
        input_tokens = torch.full((na, L), pad_id, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            input_tokens[i, L - len(s) :] = torch.tensor(s, dtype=torch.long, device=device)

        audio_embs = torch.zeros(na, max_c, H, dtype=dtype, device=device)
        for i, fr in enumerate(chunk_frames):
            audio_embs[i, : fr.shape[0]] = fr

        # Interleave audio into the AUDIO_TOKEN_IDX slots (per-row cumsum gather).
        audio_mask = input_tokens == AUDIO_TOKEN_IDX
        text_tokens = input_tokens.where(~audio_mask, torch.zeros_like(input_tokens))
        text_embeds = embed_tokens(text_tokens)
        frame_idx = audio_mask.long().cumsum(dim=1) - 1
        gather_idx = frame_idx.clamp(min=0).unsqueeze(-1).expand(na, L, H)
        audio_at = torch.gather(audio_embs, dim=1, index=gather_idx)
        embeds = torch.where(audio_mask.unsqueeze(-1), audio_at, text_embeds)

        valid = input_tokens != pad_id  # (na, L) bool
        if contiguous_text_positions:
            # Per row: history (instr+emitted) at 0..pref-1, the <vs>/audio/<ve>
            # prelude overlaid just before pref (clamped >=0), and the first decoded
            # word at pref -> the transcript stays contiguous with the history.
            position_ids = torch.zeros(na, L, dtype=torch.long, device=device)
            cur_pos = torch.zeros(na, dtype=torch.long, device=device)
            for i, (s, fr) in enumerate(zip(seqs, chunk_frames)):
                seq_len = len(s)
                start = L - seq_len
                c = int(fr.shape[0])
                pref = seq_len - (c + 2)  # history length (instr + emitted)
                row_pos = list(range(pref)) + [max(0, pref - (c + 2) + j) for j in range(c + 2)]
                position_ids[i, start:] = torch.tensor(row_pos, dtype=torch.long, device=device)
                cur_pos[i] = pref
        else:
            position_ids = (valid.long().cumsum(dim=1) - 1).clamp(min=0)
            cur_pos = position_ids[:, -1] + 1  # (na,)

        out = llm(
            inputs_embeds=embeds,
            attention_mask=valid.long(),
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        logits = out.logits[:, -1]  # <ve> position -> predicts first word
        attn_running = valid.long()

        finished = [False] * na
        words: List[List[int]] = [[] for _ in range(na)]
        do_delete = [False] * na  # a leading <del> deletes the last committed word
        for _ in range(max_new_tokens):
            nxt = logits.argmax(dim=-1)  # (na,)
            for i in range(na):
                if finished[i]:
                    continue
                tid = int(nxt[i].item())
                if tid == eot_id:
                    finished[i] = True
                elif delete_id is not None and tid == delete_id and len(words[i]) == 0 and not do_delete[i]:
                    do_delete[i] = True  # leading delete: pop last word at commit, don't emit <del>
                else:
                    words[i].append(tid)
            if all(finished):
                break
            # Feed the next token (eot for finished rows; ignored downstream).
            feed = nxt.clone()
            for i in range(na):
                if finished[i]:
                    feed[i] = eot_id
            temb = embed_tokens(feed.unsqueeze(1))  # (na, 1, H)
            attn_running = torch.cat([attn_running, torch.ones(na, 1, dtype=attn_running.dtype, device=device)], dim=1)
            out = llm(
                inputs_embeds=temb,
                attention_mask=attn_running,
                position_ids=cur_pos.unsqueeze(1),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out.past_key_values
            cur_pos = cur_pos + 1
            logits = out.logits[:, -1]

        for i, b in enumerate(active):
            if delete_id is None:
                emitted[b].extend(words[i])
                chunk_ids[b].extend([k] * len(words[i]))
            else:
                if do_delete[i]:
                    _pop_last_word(b)  # remove the mis-committed previous word
                for tok in words[i]:
                    _append_token(b, tok, k)

    if return_chunk_ids:
        return emitted, chunk_ids
    return emitted
