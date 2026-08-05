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
"""Shared-audio packed layout for the SCRIPT SpeechLM.

This is a **memory-efficient** variant of
:mod:`nemo.collections.speechlm2.parts.script`. In the original packed
layout each chunk's branch *copies* its audio window as fresh ``AUDIO_TOKEN_IDX``
positions, so with a fixed frame window ``W`` and a small chunk size the packed
sequence length blows up (audio tokens = ``W * num_chunks``, frames re-duplicated
across overlapping windows).

Here the encoder frames are laid down **once** as a shared audio track, and each
branch attends its fixed frame window ``[win_start_k, win_end_k)`` of that track
through the 4D mask — no per-branch copies. So the packed audio contributes only
``F = total_frames`` positions regardless of the window size:

    [spine : instruction w_1 .. w_N]         seg = 0  (pure-text history, causal)
    [audio : f_0 f_1 .. f_{F-1}]              seg = AUDIO_SEG (self-contained, causal)
    [branch k : <ve> w_k .. <eot>]            seg = k

A branch's ``<ve>`` anchor + words attend (a) their history prefix of the spine,
(b) their audio window of the shared track, and (c) their own earlier tokens —
nothing else. The audio track attends only earlier audio (never text), so its
representation is branch-independent (safe to share) and never leaks future text.

Positions are assigned so training and streaming inference agree without knowing
the full utterance length: spine word -> its word index (``0..P-1``); audio frame
-> its frame index (``0..F-1``); a branch's anchor+words -> contiguous with the
history (``pref_k, pref_k+1, ...``). (Audio and spine position ranges overlap, but
they are different segments that never attend each other, and audio vs text embeds
differ in content, so RoPE sharing a value is harmless.)

All builders here are pure tensor/logic (no model, no tokenizer).
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from torch import Tensor

from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX, IGNORE_INDEX
from nemo.collections.speechlm2.parts.script import ChunkSpec, _audio_window_start

SPINE_SEG_ID = 0
AUDIO_SEG_ID = -2  # padding uses -1; spine 0; branches 1..N; audio track -2


@dataclass
class SharedAudioPackedExample:
    """One utterance packed as spine (text) + shared audio track + branches.

    All tensors are 1-D of length ``T``. Batch with
    :func:`collate_shared_audio_examples`.

    Attributes:
        input_ids: (T,) token ids; audio-track positions hold ``AUDIO_TOKEN_IDX``.
        position_ids: (T,) RoPE positions (spine=word index, audio=frame index,
            branch=contiguous with its history prefix).
        seg_ids: (T,) 0 spine, ``AUDIO_SEG_ID`` audio, ``k>=1`` branch-k.
        prefix_len: (T,) branch token -> its history-prefix length into the spine.
        win_start / win_end: (T,) branch token -> its audio window frame bounds
            ``[win_start, win_end)``; 0 for non-branch tokens.
        audio_frame_index: (T,) audio token -> its global encoder-frame index
            (``0..F-1``); ``-1`` elsewhere. Doubles as the embed gather index.
        target_ids: (T,) next-token targets; ``IGNORE_INDEX`` except branch words + eot.
        is_audio: (T,) True at audio-track positions.
        spine_len: number of spine tokens ``P``.
        audio_len: number of audio-track frames ``F``.
    """

    input_ids: Tensor
    position_ids: Tensor
    seg_ids: Tensor
    prefix_len: Tensor
    win_start: Tensor
    win_end: Tensor
    audio_frame_index: Tensor
    target_ids: Tensor
    is_audio: Tensor
    spine_len: int
    audio_len: int


def build_shared_audio_chunk_example(
    instruction_ids: List[int],
    chunks: List[ChunkSpec],
    vision_end_id: int,
    eot_id: int,
    supervise_eot: bool = True,
    audio_window_frames: int = 0,
    audio_history_chunks: int = 0,
) -> SharedAudioPackedExample:
    """Build the shared-audio spine + audio-track + branches layout.

    Args:
        instruction_ids: instruction/system-prompt token ids (spine root).
        chunks: per-chunk specs (audio frame count + revealed word token ids).
        vision_end_id: the per-branch anchor token (``<ve>``) that predicts the
            first word (the audio is shared, so no per-branch ``<vs>``/audio copy).
        eot_id: end-of-turn token appended to each branch.
        supervise_eot: if True the branch predicts ``eot_id`` after its last word.
        audio_window_frames: fixed frame window ``W`` (last ``W`` frames ending at
            the chunk boundary, floored to the current chunk). ``0`` pairs with
            ``audio_history_chunks``.
        audio_history_chunks: chunk-based window ``M`` (used when
            ``audio_window_frames == 0``).

    Returns:
        SharedAudioPackedExample.
    """
    m = len(instruction_ids)
    M = max(int(audio_history_chunks), 0)
    W = max(int(audio_window_frames), 0)

    # --- spine: instruction + all chunk words ---
    spine_ids: List[int] = list(instruction_ids)
    prefix_lens: List[int] = []
    frame_starts: List[int] = []
    running = m
    running_frames = 0
    for ch in chunks:
        prefix_lens.append(running)
        frame_starts.append(running_frames)
        spine_ids.extend(ch.target_ids)
        running += len(ch.target_ids)
        running_frames += ch.audio_len
    P = len(spine_ids)
    F = running_frames

    input_ids: List[int] = list(spine_ids)
    position_ids: List[int] = list(range(P))
    seg_ids: List[int] = [SPINE_SEG_ID] * P
    prefix_len: List[int] = [0] * P
    win_start: List[int] = [0] * P
    win_end: List[int] = [0] * P
    audio_frame_index: List[int] = [-1] * P
    target_ids: List[int] = [IGNORE_INDEX] * P
    is_audio: List[bool] = [False] * P

    # --- shared audio track: all frames once, positions = frame index ---
    input_ids.extend([AUDIO_TOKEN_IDX] * F)
    position_ids.extend(range(F))
    seg_ids.extend([AUDIO_SEG_ID] * F)
    prefix_len.extend([0] * F)
    win_start.extend([0] * F)
    win_end.extend([0] * F)
    audio_frame_index.extend(range(F))
    target_ids.extend([IGNORE_INDEX] * F)
    is_audio.extend([True] * F)

    # --- branches: anchor <ve> + words + eot; window enforced via the mask ---
    for kc, ch in enumerate(chunks):
        k = kc + 1
        pref = prefix_lens[kc]
        we = frame_starts[kc] + ch.audio_len
        ws = _audio_window_start(frame_starts[kc], we, frame_starts[max(0, kc - M)], W)

        words = list(ch.target_ids)
        branch_tokens = [vision_end_id] + words + [eot_id]
        nbt = len(branch_tokens)

        next_targets = [IGNORE_INDEX] * nbt
        for j, tok in enumerate(words):  # anchor predicts w0, w_i predicts w_{i+1}
            next_targets[j] = tok
        if supervise_eot:
            next_targets[len(words)] = eot_id  # anchor (empty chunk) / last word predicts eot

        branch_positions = [pref + i for i in range(nbt)]

        input_ids.extend(branch_tokens)
        position_ids.extend(branch_positions)
        seg_ids.extend([k] * nbt)
        prefix_len.extend([pref] * nbt)
        win_start.extend([ws] * nbt)
        win_end.extend([we] * nbt)
        audio_frame_index.extend([-1] * nbt)
        target_ids.extend(next_targets)
        is_audio.extend([False] * nbt)

    return SharedAudioPackedExample(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        position_ids=torch.tensor(position_ids, dtype=torch.long),
        seg_ids=torch.tensor(seg_ids, dtype=torch.long),
        prefix_len=torch.tensor(prefix_len, dtype=torch.long),
        win_start=torch.tensor(win_start, dtype=torch.long),
        win_end=torch.tensor(win_end, dtype=torch.long),
        audio_frame_index=torch.tensor(audio_frame_index, dtype=torch.long),
        target_ids=torch.tensor(target_ids, dtype=torch.long),
        is_audio=torch.tensor(is_audio, dtype=torch.bool),
        spine_len=P,
        audio_len=F,
    )


def build_shared_audio_chunk_mask(
    seg_ids: Tensor,
    position_ids: Tensor,
    prefix_len: Tensor,
    win_start: Tensor,
    win_end: Tensor,
    audio_frame_index: Tensor,
    valid: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    """Build the (B, 1, T, T) additive attention mask for the shared-audio layout.

    A query may attend a valid key iff one of:

    * **spine -> spine**: causal by position (pure-text history).
    * **audio -> audio**: causal by frame index (self-contained audio track; audio
      never attends text, so it never leaks future words and is branch-independent).
    * **branch -> spine**: key is in the branch's history prefix (``k_pos < prefix``).
    * **branch -> audio**: key frame is in the branch's window
      (``win_start <= frame < win_end``).
    * **branch -> same-branch**: causal by position (its anchor + earlier words).

    Everything else is blocked (spine never sees audio/branches; audio never sees
    text; a branch never sees another branch or audio outside its window).
    """
    q_seg = seg_ids[:, :, None]
    k_seg = seg_ids[:, None, :]
    q_pos = position_ids[:, :, None]
    k_pos = position_ids[:, None, :]
    q_prefix = prefix_len[:, :, None]
    q_ws = win_start[:, :, None]
    q_we = win_end[:, :, None]
    q_frame = audio_frame_index[:, :, None]
    k_frame = audio_frame_index[:, None, :]
    k_valid = valid[:, None, :]

    q_is_spine = q_seg == SPINE_SEG_ID
    k_is_spine = k_seg == SPINE_SEG_ID
    q_is_audio = q_seg == AUDIO_SEG_ID
    k_is_audio = k_seg == AUDIO_SEG_ID
    q_is_branch = q_seg >= 1
    same_branch = (q_seg == k_seg) & q_is_branch

    spine_to_spine = q_is_spine & k_is_spine & (k_pos <= q_pos)
    audio_to_audio = q_is_audio & k_is_audio & (k_frame <= q_frame)
    branch_to_prefix = q_is_branch & k_is_spine & (k_pos < q_prefix)
    branch_to_audio = q_is_branch & k_is_audio & (q_ws <= k_frame) & (k_frame < q_we)
    branch_to_own = same_branch & (k_pos <= q_pos)

    allowed = (
        spine_to_spine | audio_to_audio | branch_to_prefix | branch_to_audio | branch_to_own
    ) & k_valid

    additive = torch.zeros_like(allowed, dtype=dtype)
    additive = additive.masked_fill(~allowed, torch.finfo(dtype).min)
    return additive.unsqueeze(1)


@dataclass
class BatchedSharedAudioChunk:
    """A right-padded batch of :class:`SharedAudioPackedExample`."""

    input_ids: Tensor
    position_ids: Tensor
    seg_ids: Tensor
    prefix_len: Tensor
    win_start: Tensor
    win_end: Tensor
    audio_frame_index: Tensor
    target_ids: Tensor
    is_audio: Tensor
    valid: Tensor


def collate_shared_audio_examples(
    examples: List[SharedAudioPackedExample],
    pad_id: int,
) -> BatchedSharedAudioChunk:
    """Right-pad a list of shared-audio examples into a batch."""
    B = len(examples)
    T = max(int(e.input_ids.numel()) for e in examples)

    def _pad(vals, pad_value, dtype):
        out = torch.full((B, T), pad_value, dtype=dtype)
        for i, v in enumerate(vals):
            out[i, : v.numel()] = v.to(dtype)
        return out

    valid = torch.zeros((B, T), dtype=torch.bool)
    for i, e in enumerate(examples):
        valid[i, : e.input_ids.numel()] = True

    return BatchedSharedAudioChunk(
        input_ids=_pad([e.input_ids for e in examples], pad_id, torch.long),
        position_ids=_pad([e.position_ids for e in examples], 0, torch.long),
        seg_ids=_pad([e.seg_ids for e in examples], -1, torch.long),  # -1 = padding segment
        prefix_len=_pad([e.prefix_len for e in examples], 0, torch.long),
        win_start=_pad([e.win_start for e in examples], 0, torch.long),
        win_end=_pad([e.win_end for e in examples], 0, torch.long),
        audio_frame_index=_pad([e.audio_frame_index for e in examples], -1, torch.long),
        target_ids=_pad([e.target_ids for e in examples], IGNORE_INDEX, torch.long),
        is_audio=_pad([e.is_audio for e in examples], False, torch.bool),
        valid=valid,
    )


@dataclass
class SeparateSharedAudioExample:
    """Standalone per-chunk oracle for the shared-audio layout (parity test).

    ``[instruction w_<k] [audio f_0..f_{we-1}] [<ve> w_k <eot>]`` run with
    :func:`build_shared_audio_chunk_mask`; only frames ``< win_end`` are needed
    (audio is causal, so window-frame reps match the full packed example).
    """

    input_ids: Tensor
    position_ids: Tensor
    seg_ids: Tensor
    prefix_len: Tensor
    win_start: Tensor
    win_end: Tensor
    audio_frame_index: Tensor
    target_ids: Tensor
    is_audio: Tensor
    branch_start: int


def build_separate_shared_audio_examples(
    instruction_ids: List[int],
    chunks: List[ChunkSpec],
    vision_end_id: int,
    eot_id: int,
    supervise_eot: bool = True,
    audio_window_frames: int = 0,
    audio_history_chunks: int = 0,
) -> List[SeparateSharedAudioExample]:
    """Build the standalone per-chunk oracles the packed builder must reproduce."""
    m = len(instruction_ids)
    M = max(int(audio_history_chunks), 0)
    W = max(int(audio_window_frames), 0)

    frame_starts: List[int] = []
    prefix_lens: List[int] = []
    running = m
    running_frames = 0
    for ch in chunks:
        prefix_lens.append(running)
        frame_starts.append(running_frames)
        running += len(ch.target_ids)
        running_frames += ch.audio_len

    spine_all = list(instruction_ids)
    for ch in chunks:
        spine_all.extend(ch.target_ids)

    out: List[SeparateSharedAudioExample] = []
    for kc, ch in enumerate(chunks):
        pref = prefix_lens[kc]
        we = frame_starts[kc] + ch.audio_len
        ws = _audio_window_start(frame_starts[kc], we, frame_starts[max(0, kc - M)], W)

        # spine prefix = instruction + w_<k (history the branch may attend).
        spine = spine_all[:pref]
        # audio track = frames 0..we-1 (enough for causal window-frame reps).
        words = list(ch.target_ids)
        branch = [vision_end_id] + words + [eot_id]

        input_ids = spine + [AUDIO_TOKEN_IDX] * we + branch
        seg_ids = [SPINE_SEG_ID] * pref + [AUDIO_SEG_ID] * we + [kc + 1] * len(branch)
        position_ids = list(range(pref)) + list(range(we)) + [pref + i for i in range(len(branch))]
        prefix_len = [0] * (pref + we) + [pref] * len(branch)
        win_start = [0] * (pref + we) + [ws] * len(branch)
        win_end = [0] * (pref + we) + [we] * len(branch)
        audio_frame_index = [-1] * pref + list(range(we)) + [-1] * len(branch)
        is_audio = [False] * pref + [True] * we + [False] * len(branch)

        target_ids = [IGNORE_INDEX] * len(input_ids)
        branch_start = pref + we
        for j, tok in enumerate(words):
            target_ids[branch_start + j] = tok
        if supervise_eot:
            target_ids[branch_start + len(words)] = eot_id

        out.append(
            SeparateSharedAudioExample(
                input_ids=torch.tensor(input_ids, dtype=torch.long),
                position_ids=torch.tensor(position_ids, dtype=torch.long),
                seg_ids=torch.tensor(seg_ids, dtype=torch.long),
                prefix_len=torch.tensor(prefix_len, dtype=torch.long),
                win_start=torch.tensor(win_start, dtype=torch.long),
                win_end=torch.tensor(win_end, dtype=torch.long),
                audio_frame_index=torch.tensor(audio_frame_index, dtype=torch.long),
                target_ids=torch.tensor(target_ids, dtype=torch.long),
                is_audio=torch.tensor(is_audio, dtype=torch.bool),
                branch_start=branch_start,
            )
        )
    return out


@torch.no_grad()
def batched_shared_audio_decode(
    llm,
    embed_tokens: Callable[[Tensor], Tensor],
    instruction_ids_list: List[List[int]],
    frames_list: List[Tensor],
    chunk_size: int,
    vision_end_id: int,
    eot_id: int,
    pad_id: int,
    max_new_tokens: int = 64,
    device: Optional[torch.device] = None,
    audio_window_frames: int = 0,
    audio_history_chunks: int = 0,
    max_history_tokens: int = 0,
) -> List[List[int]]:
    """Batched greedy streaming decode for the shared-audio model.

    Chunk-synchronous. For each chunk every active stream is re-prefilled as a
    small shared-audio sequence ``[history text] [audio window frames] [<ve>]`` with
    the shared-audio 4D mask (audio attends only audio; history only history; the
    anchor attends history + the window), then words are decoded greedily until
    ``eot``. Per-step memory is bounded by one chunk's window (not all chunks
    packed), so there is no training-time blow-up. Equivalent to the packed
    training conditioning ``p(words_k | history_<k, audio_window_k)``.
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

    for k in range(max_chunks):
        active = [b for b in range(B) if k < num_chunks[b]]
        if not active:
            break
        na = len(active)

        # Per-stream pieces. The audio track is the FULL causal prefix [0, we) so
        # each frame's rep attends all earlier frames (matches training); the anchor
        # then attends only the window [ws, we) via the mask.
        hist_list: List[List[int]] = []
        we_list: List[int] = []
        ws_list: List[int] = []
        for b in active:
            we = min((k + 1) * chunk_size, n_frames[b])
            ws = _audio_window_start(k * chunk_size, we, max(0, k - M) * chunk_size, W)
            we_list.append(we)
            ws_list.append(ws)
            hist = list(instruction_ids_list[b])
            tail = list(emitted[b])
            if max_history_tokens and len(tail) > max_history_tokens:
                tail = tail[-max_history_tokens:]
            hist_list.append(hist + tail)

        Hh = max(len(h) for h in hist_list)
        Cc = max(we_list) if we_list else 0
        # Row layout (left-padded): [hist][audio 0..we][<ve>][decoded words...].
        base_L = Hh + Cc + 1

        def _empty(fill, T, dt):
            return torch.full((na, T), fill, dtype=dt, device=device)

        input_ids = _empty(pad_id, base_L, torch.long)
        seg = _empty(-1, base_L, torch.long)
        pos = torch.zeros((na, base_L), dtype=torch.long, device=device)
        prefix_len = torch.zeros((na, base_L), dtype=torch.long, device=device)
        win_s = torch.zeros((na, base_L), dtype=torch.long, device=device)
        win_e = torch.zeros((na, base_L), dtype=torch.long, device=device)
        afi = _empty(-1, base_L, torch.long)
        valid = torch.zeros((na, base_L), dtype=torch.bool, device=device)
        audio_embs = torch.zeros((na, Cc, H), dtype=dtype, device=device)

        for i, b in enumerate(active):
            hist = hist_list[i]
            hlen = len(hist)
            we = we_list[i]
            ws = ws_list[i]
            h0 = base_L - (hlen + we + 1)
            a0 = h0 + hlen
            v0 = a0 + we
            input_ids[i, h0:a0] = torch.tensor(hist, dtype=torch.long, device=device)
            input_ids[i, a0:v0] = AUDIO_TOKEN_IDX
            input_ids[i, v0] = vision_end_id
            seg[i, h0:a0] = SPINE_SEG_ID
            seg[i, a0:v0] = AUDIO_SEG_ID
            seg[i, v0] = 1
            pos[i, h0:a0] = torch.arange(hlen, device=device)
            pos[i, a0:v0] = torch.arange(we, device=device)  # frame index positions
            pos[i, v0] = hlen  # anchor contiguous with history
            prefix_len[i, v0] = hlen
            win_s[i, v0] = ws
            win_e[i, v0] = we
            afi[i, a0:v0] = torch.arange(we, device=device)
            valid[i, h0:] = True
            audio_embs[i, :we] = frames_list[b][:we].to(device=device, dtype=dtype)

        def _forward(ids, sg, ps, pf, wsx, wex, af, vd):
            amask = ids == AUDIO_TOKEN_IDX
            tids = ids.where(~amask, torch.zeros_like(ids))
            temb = embed_tokens(tids)
            Lc = ids.shape[1]
            ford = amask.long().cumsum(dim=1) - 1
            gidx = ford.clamp(min=0).unsqueeze(-1).expand(na, Lc, H)
            aat = torch.gather(audio_embs, dim=1, index=gidx)
            emb = torch.where(amask.unsqueeze(-1), aat, temb)
            msk = build_shared_audio_chunk_mask(sg, ps, pf, wsx, wex, af, vd, emb.dtype)
            o = llm(inputs_embeds=emb, attention_mask=msk, position_ids=ps, use_cache=False, return_dict=True)
            return o.logits[:, -1]

        logits = _forward(input_ids, seg, pos, prefix_len, win_s, win_e, afi, valid)

        finished = [False] * na
        for _step in range(max_new_tokens):
            nxt = logits.argmax(dim=-1)
            for i in range(na):
                if finished[i]:
                    continue
                tid = int(nxt[i].item())
                if tid == eot_id:
                    finished[i] = True
                else:
                    emitted[active[i]].append(tid)
            if all(finished):
                break
            feed = nxt.clone()
            for i in range(na):
                if finished[i]:
                    feed[i] = eot_id
            input_ids = torch.cat([input_ids, feed.unsqueeze(1)], dim=1)
            seg = torch.cat([seg, _empty(1, 1, torch.long)], dim=1)
            pos = torch.cat([pos, (pos[:, -1] + 1).unsqueeze(1)], dim=1)
            prefix_len = torch.cat([prefix_len, prefix_len[:, -1:].clone()], dim=1)
            win_s = torch.cat([win_s, win_s[:, -1:].clone()], dim=1)
            win_e = torch.cat([win_e, win_e[:, -1:].clone()], dim=1)
            afi = torch.cat([afi, _empty(-1, 1, torch.long)], dim=1)
            valid = torch.cat([valid, torch.ones((na, 1), dtype=torch.bool, device=device)], dim=1)
            logits = _forward(input_ids, seg, pos, prefix_len, win_s, win_e, afi, valid)

    return emitted
