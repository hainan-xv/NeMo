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
"""SCRIPT (spine + branches) layout for streaming SpeechLM.

SCRIPT frames streaming ASR as *conditional text completion*: for each audio
chunk the model is asked to extract the words carried by that chunk, given the
transcript so far, i.e. it models

    p(words_k | text_history_<k, audio_k).

Rather than one interleaved ``audio-text-audio-text`` causal stream, each
utterance is packed into a single sequence made of two kinds of tokens:

* A **spine**: ``[instruction] w_1 w_2 ... w_N`` — the pure-text history. It is a
  plain causal text sequence, computed exactly once; every chunk reuses the
  relevant prefix of it (``instruction`` + the words emitted so far). Spine
  tokens NEVER attend to audio, so their representation is pure text and can be
  shared by all chunks.
* One **branch** per chunk ``k``: ``<vision_start> [audio_k frames] <vision_end>
  w_k <eot>``. A branch attends only to (a) its own history prefix of the spine
  ``[instruction] w_1..w_{k-1}``, (b) its own audio frames, and (c) its own
  earlier branch tokens — nothing else. The training loss is taken on the
  branch's target words.

A word must be *predicted from audio* in its branch (so it attends audio) but
must act as *pure text history* in the spine (so it must not), which is why each
word appears in both roles. The growing history is still materialized once, so
training is a single O(L) forward rather than O(N^2).

The invariant that makes this correct — and that
``test_parity_packed_vs_separate_examples`` checks — is that a branch's logits
inside the packed sequence are numerically identical to running the standalone
example ``[instruction] w_1..w_{k-1} <vs> audio_k <ve> w_k`` on its own. That is
what makes overlapping ``position_ids`` across branches safe: the 4D mask keeps
branches isolated, and each branch's positions are exactly those of its
standalone example.

Everything here is pure tensor/list logic (no model, no tokenizer), so it can be
unit-tested in isolation.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
from torch import Tensor

from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX, IGNORE_INDEX

# Segment id 0 is reserved for the spine; branches are 1..N. Right-padding uses -1.
SPINE_SEG_ID = 0
PAD_SEG_ID = -1


def audio_window_start(
    chunk_index: int,
    frame_starts: List[int],
    audio_history_chunks: int,
    win_end: Optional[int] = None,
    audio_window_frames: int = 0,
) -> int:
    """First global encoder-frame index of branch ``chunk_index``'s audio window.

    The window always ends at the chunk's own boundary (``win_end``) and never
    looks ahead. There are two ways to size it:

    * **Chunk-based** (default) — with ``audio_history_chunks == M`` the window
      begins at the start of chunk ``max(0, k - M)``, so the branch sees the audio
      of chunks ``[k-M .. k]``. ``M == 0`` gives each branch only its own chunk.
      The window therefore SCALES with the chunk size.

    * **Frame-based** (``audio_window_frames = F > 0``) — the window is the last
      ``F`` frames ending at the chunk boundary, giving a CONSTANT acoustic
      context regardless of the emission granularity. Takes precedence over
      ``audio_history_chunks``.

    Two properties of the frame-based mode are worth stating explicitly:

    * ``F`` is a **floor, not a cap**. The window never starts after the current
      chunk's own start, so a chunk longer than ``F`` still shows all of its own
      frames. A branch must be able to see the audio of the words it is being
      asked to predict, so clipping into the current chunk would be incoherent.
    * Early chunks are **clamped at frame 0**, not left-padded, so the first few
      branches simply have shorter windows.

    Sizing matters when the emission delay is non-zero: a word held back by the
    delay is emitted from a LATER chunk, and only a window reaching back far
    enough still contains that word's acoustics.

    This helper is shared by the training packer and the inference decoder so the
    two cannot drift apart.
    """
    F = max(int(audio_window_frames), 0)
    cur_chunk_start = frame_starts[chunk_index]
    if F > 0:
        if win_end is None:
            raise ValueError("win_end is required when audio_window_frames > 0")
        return max(0, min(cur_chunk_start, int(win_end) - F))
    return frame_starts[max(0, chunk_index - max(int(audio_history_chunks), 0))]


def _branch_positions(pref: int, window_len: int, n_words: int) -> List[int]:
    """RoPE position ids for one branch.

    The branch is laid out as ``<vs> [window_len audio] <ve> [n_words words]
    <eot>`` (``window_len + n_words + 3`` tokens) and takes a contiguous run of
    positions starting right after its history prefix, ``pref, pref+1, ...``.
    The audio therefore consumes ``window_len + 2`` position slots between the
    history (which ends at ``pref - 1``) and the predicted words — exactly the
    positions the equivalent standalone example would use.
    """
    return [pref + off for off in range(window_len + n_words + 3)]


@dataclass
class ChunkSpec:
    """One chunk of an utterance.

    Args:
        audio_len: number of encoder audio frames in this chunk (``C_k``).
        target_ids: token ids of the words revealed by this chunk (``w_k``). May
            be empty for a silent chunk, in which case the branch only predicts
            ``eot_id``.
    """

    audio_len: int
    target_ids: List[int] = field(default_factory=list)


@dataclass
class PackedChunkExample:
    """A single utterance packed as spine + per-chunk branches.

    All tensors are 1-D of length ``T`` (the packed sequence length). Batch them
    with :func:`collate_packed_chunk_examples`.

    Attributes:
        input_ids: (T,) token ids; audio-frame positions hold ``AUDIO_TOKEN_IDX``.
        position_ids: (T,) RoPE position per token (spine = its index; a branch
            token = ``prefix_len_k + local_offset``).
        seg_ids: (T,) ``0`` for spine tokens, ``k >= 1`` for branch-k tokens.
        prefix_len: (T,) for a branch token, the length of its chunk's history
            prefix (``m + sum_{j<k} |w_j|``); ``0`` for spine tokens.
        target_ids: (T,) next-token targets; ``IGNORE_INDEX`` everywhere except
            the branch positions that predict the chunk's words + end-of-turn.
        is_audio: (T,) True at audio-frame positions.
        audio_frame_index: (T,) global encoder-frame index each audio position
            maps to (``-1`` elsewhere). With ``audio_history_chunks > 0`` a frame
            appears in several branches, so the model fills audio slots by an
            explicit gather on this index rather than a positional cumsum. With
            ``audio_history_chunks == 0`` it is exactly ``0, 1, 2, ...``.
        spine_len: number of spine tokens ``P`` (the spine is ``input_ids[:P]``).
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
    audio_window_frames: int = 0,
) -> PackedChunkExample:
    """Build the packed spine+branch layout for one utterance.

    Physical order is the spine first, then the branches in chunk order::

        [instruction w_1 w_2 ... w_N]   (spine)
        [<vs> A_1.. <ve> w_1 <eot>]     (branch 1)
        [<vs> A_2.. <ve> w_2 <eot>]     (branch 2)
        ...

    Args:
        instruction_ids: token ids of the instruction / system prompt prefix.
        chunks: per-chunk specs (audio frame count + revealed word token ids).
        vision_start_id / vision_end_id: delimiter token ids wrapping the audio.
        eot_id: end-of-turn token id closing each branch — what the model emits
            when a chunk has no more words.
        supervise_eot: if True the branch is trained to predict ``eot_id`` after
            its last word (so it learns to stop). The ``eot`` position itself is
            never supervised.
        audio_history_chunks: ``M``, the number of PREVIOUS chunks whose audio is
            also included in each branch's window; branch ``k`` then wraps the
            frames of chunks ``[max(0, k-M) .. k]`` (fewer for early chunks, with
            no padding). ``0`` = each branch sees only its own chunk's audio.
        audio_window_frames: if ``> 0``, size every window by this FIXED number of
            frames ending at the chunk boundary instead of by whole chunks, giving
            a constant acoustic context independent of the chunk size. Acts as a
            floor (a longer chunk keeps all its own frames) and takes precedence
            over ``audio_history_chunks``. See :func:`audio_window_start`.

    Returns:
        PackedChunkExample.
    """
    m = len(instruction_ids)

    # --- spine: instruction followed by every chunk's words, in order ---
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
    target_ids: List[int] = [IGNORE_INDEX] * P  # the spine is context only — no loss
    is_audio: List[bool] = [False] * P
    audio_frame_index: List[int] = [-1] * P

    # --- branches ---
    for kc, ch in enumerate(chunks):  # kc: 0-based chunk index
        k = kc + 1  # 1-based branch / segment id
        pref = prefix_lens[kc]

        win_end = frame_starts[kc] + ch.audio_len
        win_start = audio_window_start(kc, frame_starts, audio_history_chunks, win_end, audio_window_frames)
        window_frames = list(range(win_start, win_end))
        window_len = len(window_frames)

        branch_words = list(ch.target_ids)

        # branch tokens: <vs> [window audio] <ve> [branch_words] <eot>
        branch_tokens: List[int] = [vision_start_id] + [AUDIO_TOKEN_IDX] * window_len + [vision_end_id]
        branch_is_audio: List[bool] = [False] + [True] * window_len + [False]
        branch_frame_idx: List[int] = [-1] + window_frames + [-1]
        branch_tokens.extend(branch_words)
        branch_is_audio.extend([False] * len(branch_words))
        branch_frame_idx.extend([-1] * len(branch_words))
        branch_tokens.append(eot_id)
        branch_is_audio.append(False)
        branch_frame_idx.append(-1)

        # Next-token targets: <ve> predicts branch_words[0], word i predicts word
        # i+1, and the last word predicts eot. <vs>/audio/<ve>/eot are not
        # supervised themselves.
        n_bt = len(branch_tokens)
        next_targets: List[int] = [IGNORE_INDEX] * n_bt
        ve_idx = 1 + window_len  # index of <ve> within the branch
        for j, tok in enumerate(branch_words):
            next_targets[ve_idx + j] = tok
        if supervise_eot:
            next_targets[ve_idx + len(branch_words)] = eot_id

        input_ids.extend(branch_tokens)
        position_ids.extend(_branch_positions(pref, window_len, len(branch_words)))
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


def build_script_mask(
    seg_ids: Tensor,
    position_ids: Tensor,
    prefix_len: Tensor,
    valid: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    """Build the 4D additive attention mask for a packed SCRIPT batch.

    A query at position ``q`` may attend key ``j`` iff the key is valid and one of:

    * **spine query, spine key** — causal within the spine (``pos[j] <= pos[q]``),
      so spine tokens see only other spine tokens and stay pure text.
    * **branch query, spine key** — ``j`` lies in the branch's history prefix
      (``pos[j] < prefix_len[q]``). Spine positions equal their index, so this is
      exactly ``instruction + w_1..w_{k-1}``.
    * **branch query, same-branch key** — causal within the branch
      (``pos[j] <= pos[q]``), covering its own audio and its earlier words.

    Everything else is blocked: a branch never sees another branch, another
    chunk's audio, or spine words at or after its own; the spine never sees any
    branch or any audio.

    Args:
        seg_ids: (B, T) ``0`` = spine, ``>= 1`` = branch id, ``-1`` = padding.
        position_ids: (B, T) RoPE positions (also used for causality / prefix).
        prefix_len: (B, T) per-branch-token history-prefix length (0 for spine).
        valid: (B, T) bool, False at padding.
        dtype: floating dtype for the additive mask.

    Returns:
        (B, 1, T, T) additive mask: ``0`` where allowed, ``finfo(dtype).min``
        where blocked.
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
    audio_frame_index: Tensor  # (B, T) global encoder-frame index at audio slots, -1 elsewhere
    valid: Tensor  # (B, T) bool, False at right-padding
    spine_lens: Tensor  # (B,)


def collate_packed_chunk_examples(examples: List[PackedChunkExample], pad_id: int) -> BatchedPackedChunk:
    """Right-pad a list of packed examples into a batch.

    Padding slots hold the text ``pad_id`` with ``seg_ids = PAD_SEG_ID`` (which
    never matches a real segment) and ``valid = False``, so they are masked out
    as attention keys.
    """
    B = len(examples)
    T = max(int(ex.input_ids.numel()) for ex in examples)

    def _pad(vals: List[Tensor], pad_value, dtype) -> Tensor:
        out = torch.full((B, T), pad_value, dtype=dtype)
        for i, v in enumerate(vals):
            out[i, : v.numel()] = v.to(dtype)
        return out

    valid = torch.zeros((B, T), dtype=torch.bool)
    for i, e in enumerate(examples):
        valid[i, : e.input_ids.numel()] = True

    return BatchedPackedChunk(
        input_ids=_pad([e.input_ids for e in examples], pad_id, torch.long),
        position_ids=_pad([e.position_ids for e in examples], 0, torch.long),
        seg_ids=_pad([e.seg_ids for e in examples], PAD_SEG_ID, torch.long),
        prefix_len=_pad([e.prefix_len for e in examples], 0, torch.long),
        target_ids=_pad([e.target_ids for e in examples], IGNORE_INDEX, torch.long),
        is_audio=_pad([e.is_audio for e in examples], False, torch.bool),
        audio_frame_index=_pad([e.audio_frame_index for e in examples], -1, torch.long),
        valid=valid,
        spine_lens=torch.tensor([e.spine_len for e in examples], dtype=torch.long),
    )


@dataclass
class SeparateChunkExample:
    """One standalone per-chunk example — the reference formulation.

    ``[instruction w_1..w_{k-1}] <vs> audio_window <ve> w_k <eot>`` under plain
    causal attention, with ``position_ids = 0..L-1``. Used only by the parity
    test: the packed branch's logits must match this example's.

    Attributes:
        input_ids: (L,) token ids; audio slots hold ``AUDIO_TOKEN_IDX``.
        target_ids: (L,) next-token targets (``IGNORE_INDEX`` outside the words).
        is_audio: (L,) True at audio-frame positions.
        audio_frame_index: (L,) global encoder-frame index at audio slots, -1 elsewhere.
        branch_start: index where this example's branch begins (audio + words + eot).
    """

    input_ids: Tensor
    target_ids: Tensor
    is_audio: Tensor
    audio_frame_index: Tensor
    branch_start: int


def build_separate_chunk_examples(
    instruction_ids: List[int],
    chunks: List[ChunkSpec],
    vision_start_id: int,
    vision_end_id: int,
    eot_id: int,
    supervise_eot: bool = True,
    audio_history_chunks: int = 0,
    audio_window_frames: int = 0,
) -> List[SeparateChunkExample]:
    """Build the standalone per-chunk examples the packed layout must reproduce.

    This is the naive O(N^2) formulation — one independent sequence per chunk,
    each re-materializing the whole history. It exists as the ground truth for
    :func:`build_packed_chunk_example`; the parity test forwards both and
    asserts the branch logits agree.
    """
    m = len(instruction_ids)
    prefix_lens: List[int] = []
    frame_starts: List[int] = []
    running = m
    running_frames = 0
    for ch in chunks:
        prefix_lens.append(running)
        frame_starts.append(running_frames)
        running += len(ch.target_ids)
        running_frames += ch.audio_len

    history: List[int] = list(instruction_ids)
    out: List[SeparateChunkExample] = []
    for kc, ch in enumerate(chunks):
        win_end = frame_starts[kc] + ch.audio_len
        win_start = audio_window_start(kc, frame_starts, audio_history_chunks, win_end, audio_window_frames)
        window_frames = list(range(win_start, win_end))
        window_len = len(window_frames)

        branch_words = list(ch.target_ids)
        prefix = list(history)
        branch_start = len(prefix)

        ids = prefix + [vision_start_id] + [AUDIO_TOKEN_IDX] * window_len + [vision_end_id]
        aud = [False] * len(prefix) + [False] + [True] * window_len + [False]
        fidx = [-1] * len(prefix) + [-1] + window_frames + [-1]
        ids.extend(branch_words)
        aud.extend([False] * len(branch_words))
        fidx.extend([-1] * len(branch_words))
        ids.append(eot_id)
        aud.append(False)
        fidx.append(-1)

        tgt = [IGNORE_INDEX] * len(ids)
        ve_idx = branch_start + 1 + window_len
        for j, tok in enumerate(branch_words):
            tgt[ve_idx + j] = tok
        if supervise_eot:
            tgt[ve_idx + len(branch_words)] = eot_id

        out.append(
            SeparateChunkExample(
                input_ids=torch.tensor(ids, dtype=torch.long),
                target_ids=torch.tensor(tgt, dtype=torch.long),
                is_audio=torch.tensor(aud, dtype=torch.bool),
                audio_frame_index=torch.tensor(fidx, dtype=torch.long),
                branch_start=branch_start,
            )
        )
        history.extend(branch_words)

    return out


@torch.no_grad()
def batched_stream_decode_script(
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
    audio_window_frames: int = 0,
    max_history_tokens: int = 0,
    return_chunk_ids: bool = False,
    is_word_start: Optional[Callable[[int], bool]] = None,
    insert_word_start_id: Optional[int] = None,
):
    """Batched greedy SCRIPT decode for ``B`` utterances at once.

    Chunk-synchronous: for chunk index ``k`` every still-active stream is decoded
    together. Per chunk each stream is shown its *plain-text history* plus that
    chunk's audio window — exactly the training conditioning
    ``p(words_k | text_history_<k, audio_k)`` — built as::

        [instruction] [emitted words so far] <vs> [audio window] <ve> -> words

    Because the history is compact text rather than audio, re-prefilling it each
    chunk is cheap. That avoids the fragile variable-length KV surgery a
    persistent per-stream spine cache would need across a batch, while keeping
    ``position_ids`` contiguous per stream exactly as in training.

    SCRIPT emits whole words per chunk and the history already holds the previous
    chunk's finished words, so a chunk's FIRST decoded token should begin a new
    word; otherwise it merges onto the previous one ("border ruffian" ->
    "bordereruffian"). This is not enforced by restricting what the model may
    emit (which can starve a chunk into emitting nothing). Instead the model
    decodes freely and, when a chunk's first token is not a word start,
    ``insert_word_start_id`` (a leading-space token) is inserted in front of it.
    Both arguments are no-ops unless ``is_word_start`` is given.

    Args:
        llm: causal LM accepting ``inputs_embeds`` / ``attention_mask`` /
            ``position_ids`` and returning ``logits`` with a KV cache.
        embed_tokens: token-id -> embedding lookup.
        instruction_ids_list: per-utterance instruction token ids.
        frames_list: per-utterance ``(T_enc_b, H)`` encoder frame embeddings.
        chunk_size: encoder frames per chunk.
        vision_start_id / vision_end_id / eot_id / pad_id: delimiter / end / pad ids.
        max_new_tokens: max tokens decoded per chunk per stream.
        audio_history_chunks: ``M`` — must match the training setting.
        audio_window_frames: ``F`` — fixed-frame window; must match training.
        max_history_tokens: if ``> 0``, cap the CONDITIONING history to the most
            recent N emitted tokens (the instruction is always kept). Bounds the
            per-chunk prefill so cost is linear rather than quadratic in
            duration; all emitted tokens are still returned.
        return_chunk_ids: also return, per emitted token, the chunk index during
            which it was decoded (used for word-level emission-latency metrics).

    Returns:
        ``B`` lists of emitted token ids (flattened across chunks, ``eot``
        excluded), or ``(emitted, chunk_ids)`` when ``return_chunk_ids``.
    """
    B = len(frames_list)
    if B == 0:
        return ([], []) if return_chunk_ids else []
    if device is None:
        device = frames_list[0].device
    H = frames_list[0].shape[-1]
    dtype = embed_tokens(torch.zeros(1, 1, dtype=torch.long, device=device)).dtype

    n_frames = [int(f.shape[0]) for f in frames_list]
    num_chunks = [math.ceil(n / chunk_size) if n > 0 else 0 for n in n_frames]
    max_chunks = max(num_chunks) if B else 0
    M = max(int(audio_history_chunks), 0)
    F = max(int(audio_window_frames), 0)
    # At inference every chunk is exactly ``chunk_size`` frames (the final one is
    # zero-padded up to it below), so chunk starts are a simple arithmetic run.
    _chunk_starts = [k * chunk_size for k in range(max_chunks)]

    emitted: List[List[int]] = [[] for _ in range(B)]
    chunk_ids: List[List[int]] = [[] for _ in range(B)]

    for k in range(max_chunks):
        active = [b for b in range(B) if k < num_chunks[b]]
        if not active:
            break
        na = len(active)

        # --- per-stream prefill: instruction + history + <vs> audio_k <ve> ---
        seqs: List[List[int]] = []
        chunk_frames: List[Tensor] = []
        for b in active:
            win_end = (k + 1) * chunk_size
            # Same rule as the training packer, via the shared helper, so the
            # window cannot drift between training and inference.
            win_start = audio_window_start(k, _chunk_starts, M, win_end, F)
            fr = frames_list[b][win_start:win_end].to(device=device, dtype=dtype)
            # Match TRAINING exactly on the FINAL (partial) chunk. In training every
            # chunk's audio turn is a full ``chunk_size`` frames and the slots past the
            # real audio are ZERO-filled by the gather in the model's indexed embed
            # builder, so a branch always has a full-length window ending in trailing
            # silence. A raw slice here would truncate that tail, changing both the
            # audio-token count and the end-of-audio cue relative to training.
            want = win_end - win_start
            if fr.shape[0] < want:
                fr = torch.cat([fr, fr.new_zeros(want - fr.shape[0], fr.shape[1])], dim=0)
            c = int(fr.shape[0])

            hist = emitted[b]
            if max_history_tokens and len(hist) > max_history_tokens:
                hist = hist[-max_history_tokens:]
            seqs.append(
                list(instruction_ids_list[b])
                + list(hist)
                + [vision_start_id]
                + [AUDIO_TOKEN_IDX] * c
                + [vision_end_id]
            )
            chunk_frames.append(fr)

        L = max(len(s) for s in seqs)
        max_c = max(int(fr.shape[0]) for fr in chunk_frames)

        # Left-pad so every row's LAST position is its <ve> — the shared query
        # column whose logits predict each stream's first word.
        input_tokens = torch.full((na, L), pad_id, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            input_tokens[i, L - len(s) :] = torch.tensor(s, dtype=torch.long, device=device)

        audio_embs = torch.zeros(na, max_c, H, dtype=dtype, device=device)
        for i, fr in enumerate(chunk_frames):
            audio_embs[i, : fr.shape[0]] = fr

        # Fill the AUDIO_TOKEN_IDX slots (per-row cumsum gather — within one chunk
        # the window frames are contiguous, so a cumsum is exact).
        audio_mask = input_tokens == AUDIO_TOKEN_IDX
        text_tokens = input_tokens.where(~audio_mask, torch.zeros_like(input_tokens))
        text_embeds = embed_tokens(text_tokens)
        frame_idx = audio_mask.long().cumsum(dim=1) - 1
        gather_idx = frame_idx.clamp(min=0).unsqueeze(-1).expand(na, L, H)
        audio_at = torch.gather(audio_embs, dim=1, index=gather_idx)
        embeds = torch.where(audio_mask.unsqueeze(-1), audio_at, text_embeds)

        valid = input_tokens != pad_id  # (na, L)
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
        logits = out.logits[:, -1]  # <ve> position -> predicts the first word
        attn_running = valid.long()

        finished = [False] * na
        words: List[List[int]] = [[] for _ in range(na)]
        for _ in range(max_new_tokens):
            nxt = logits.argmax(dim=-1)  # (na,)
            for i in range(na):
                if finished[i]:
                    continue
                tid = int(nxt[i].item())
                if tid == eot_id:
                    finished[i] = True
                else:
                    words[i].append(tid)
            if all(finished):
                break
            feed = nxt.clone()
            for i in range(na):
                if finished[i]:
                    feed[i] = eot_id  # harmless filler; the row is ignored downstream
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
            toks = words[i]
            # Chunk-start fix-up: guarantee a word boundary without having
            # restricted what the model was allowed to emit.
            if (
                insert_word_start_id is not None
                and is_word_start is not None
                and toks
                and emitted[b]
                and not is_word_start(toks[0])
            ):
                toks = [insert_word_start_id] + toks
            emitted[b].extend(toks)
            chunk_ids[b].extend([k] * len(toks))

    return (emitted, chunk_ids) if return_chunk_ids else emitted
