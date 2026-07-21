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


@dataclass
class ChunkSpec:
    """One chunk of an utterance.

    Args:
        audio_len: number of encoder audio frames in this chunk (``C_k``).
        target_ids: token ids of the words revealed by this chunk (``w_k``); may
            be empty for a silent chunk (the branch then only emits ``eot_id``).
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
        position_ids: (T,) RoPE position id per token (spine = its index; a branch
            token = ``prefix_len_k + local_offset``).
        seg_ids: (T,) 0 for spine tokens, ``k>=1`` for branch-k tokens.
        prefix_len: (T,) for a branch token, the length of its chunk's history
            prefix (``= m + sum_{j<k} |w_j|``); 0 for spine tokens.
        target_ids: (T,) next-token targets; ``IGNORE_INDEX`` everywhere except
            the branch positions that predict the chunk's words + end-of-turn.
        is_audio: (T,) True at audio-frame positions.
        spine_len: number of spine tokens ``P`` (spine occupies ``input_ids[:P]``).
    """

    input_ids: Tensor
    position_ids: Tensor
    seg_ids: Tensor
    prefix_len: Tensor
    target_ids: Tensor
    is_audio: Tensor
    spine_len: int


def build_packed_chunk_example(
    instruction_ids: List[int],
    chunks: List[ChunkSpec],
    vision_start_id: int,
    vision_end_id: int,
    eot_id: int,
    supervise_eot: bool = True,
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

    Returns:
        PackedChunkExample.
    """
    m = len(instruction_ids)

    # --- spine: instruction followed by all chunk words, in order ---
    spine_ids: List[int] = list(instruction_ids)
    prefix_lens: List[int] = []  # history-prefix length per chunk
    running = m
    for ch in chunks:
        prefix_lens.append(running)
        spine_ids.extend(ch.target_ids)
        running += len(ch.target_ids)
    P = len(spine_ids)

    input_ids: List[int] = list(spine_ids)
    position_ids: List[int] = list(range(P))
    seg_ids: List[int] = [SPINE_SEG_ID] * P
    prefix_len: List[int] = [0] * P
    target_ids: List[int] = [IGNORE_INDEX] * P  # spine is context only, no loss
    is_audio: List[bool] = [False] * P

    # --- branches ---
    for k, ch in enumerate(chunks, start=1):
        pref = prefix_lens[k - 1]
        # branch tokens: <vs> [audio*C] <ve> w_k... <eot>
        branch_tokens: List[int] = [vision_start_id] + [AUDIO_TOKEN_IDX] * ch.audio_len + [vision_end_id]
        branch_is_audio: List[bool] = [False] + [True] * ch.audio_len + [False]
        # target words + end-of-turn
        branch_tokens.extend(ch.target_ids)
        branch_is_audio.extend([False] * len(ch.target_ids))
        branch_tokens.append(eot_id)
        branch_is_audio.append(False)

        # next-token targets within the branch: the token AFTER <ve> is the first
        # word (or eot if empty); each word predicts the next word; the last word
        # predicts eot. <vs>/audio/eot positions are not supervised.
        n_bt = len(branch_tokens)
        branch_targets: List[int] = [IGNORE_INDEX] * n_bt
        # index of <ve> within the branch = 1 (vs) + audio_len
        ve_idx = 1 + ch.audio_len
        # positions ve_idx .. ve_idx + len(w_k) predict w_k[0..], then eot
        for j, tok in enumerate(ch.target_ids):
            branch_targets[ve_idx + j] = tok  # predicted by the previous position
        # shift: target at position i is input at i+1 -> rebuild as next-token
        # We assembled branch_targets as "token at this position"; convert to
        # next-token form (target[i] = token that should follow input[i]).
        next_targets: List[int] = [IGNORE_INDEX] * n_bt
        # words: predicted starting right after <ve>
        first_word_pos = ve_idx  # <ve> position predicts first word
        for j, tok in enumerate(ch.target_ids):
            next_targets[first_word_pos + j] = tok
        last_word_pos = first_word_pos + len(ch.target_ids)  # position that predicts eot
        if supervise_eot:
            next_targets[last_word_pos] = eot_id

        branch_positions = [pref + off for off in range(n_bt)]

        input_ids.extend(branch_tokens)
        position_ids.extend(branch_positions)
        seg_ids.extend([k] * n_bt)
        prefix_len.extend([pref] * n_bt)
        target_ids.extend(next_targets)
        is_audio.extend(branch_is_audio)

    return PackedChunkExample(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        position_ids=torch.tensor(position_ids, dtype=torch.long),
        seg_ids=torch.tensor(seg_ids, dtype=torch.long),
        prefix_len=torch.tensor(prefix_len, dtype=torch.long),
        target_ids=torch.tensor(target_ids, dtype=torch.long),
        is_audio=torch.tensor(is_audio, dtype=torch.bool),
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
        valid=valid,
        spine_lens=spine_lens,
    )


@dataclass
class SeparateChunkExample:
    """One standalone per-chunk example (the reference formulation).

    ``[instruction w_1..w_{k-1}] <vs> audio_k <ve> w_k <eot>`` with plain causal
    attention and contiguous positions ``0..len-1``.
    """

    input_ids: Tensor  # (L,)
    target_ids: Tensor  # (L,)
    is_audio: Tensor  # (L,)
    # index range [branch_start, L) of this example's branch (audio + words + eot)
    branch_start: int


def build_separate_chunk_examples(
    instruction_ids: List[int],
    chunks: List[ChunkSpec],
    vision_start_id: int,
    vision_end_id: int,
    eot_id: int,
    supervise_eot: bool = True,
) -> List[SeparateChunkExample]:
    """Build the naive per-chunk examples used as a correctness oracle.

    Each chunk becomes an independent standard-causal example whose history is
    the *plain text* ``instruction + w_1..w_{k-1}``. The packed builder must
    reproduce these branches' logits exactly (see the parity test).
    """
    out: List[SeparateChunkExample] = []
    history: List[int] = list(instruction_ids)
    for ch in chunks:
        branch_tokens = [vision_start_id] + [AUDIO_TOKEN_IDX] * ch.audio_len + [vision_end_id]
        branch_is_audio = [False] + [True] * ch.audio_len + [False]
        branch_tokens.extend(ch.target_ids)
        branch_is_audio.extend([False] * len(ch.target_ids))
        branch_tokens.append(eot_id)
        branch_is_audio.append(False)

        prefix = list(history)
        input_ids = prefix + branch_tokens
        is_audio = [False] * len(prefix) + branch_is_audio

        target_ids = [IGNORE_INDEX] * len(input_ids)
        ve_idx = len(prefix) + 1 + ch.audio_len  # position of <ve>
        for j, tok in enumerate(ch.target_ids):
            target_ids[ve_idx + j] = tok
        last_word_pos = ve_idx + len(ch.target_ids)
        if supervise_eot:
            target_ids[last_word_pos] = eot_id

        out.append(
            SeparateChunkExample(
                input_ids=torch.tensor(input_ids, dtype=torch.long),
                target_ids=torch.tensor(target_ids, dtype=torch.long),
                is_audio=torch.tensor(is_audio, dtype=torch.bool),
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

    emitted_per_chunk: List[List[int]] = []
    for k in range(num_chunks):
        cf = frames[k * chunk_size : (k + 1) * chunk_size]
        c = cf.shape[0]
        if c == 0:
            break
        # branch prelude: <vs> [audio] <ve>
        prelude = torch.cat(
            [embed_ids([vision_start_id])[0], cf, embed_ids([vision_end_id])[0]], dim=0
        )[None]  # (1, c+2, H)
        bpos = torch.arange(spine_len, spine_len + c + 2, device=device)[None]
        out = llm(inputs_embeds=prelude, position_ids=bpos, past_key_values=spine, use_cache=True, return_dict=True)
        cache = out.past_key_values
        cur = spine_len + c + 2
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
) -> List[List[int]]:
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

    emitted: List[List[int]] = [[] for _ in range(B)]

    for k in range(max_chunks):
        active = [b for b in range(B) if k < num_chunks[b]]
        if not active:
            break
        na = len(active)

        # --- build per-stream prefill: instr + emitted + <vs> audio_k <ve> ---
        seqs: List[List[int]] = []
        chunk_frames: List[Tensor] = []
        for b in active:
            fr = frames_list[b][k * chunk_size : (k + 1) * chunk_size].to(device=device, dtype=dtype)
            c = int(fr.shape[0])
            toks = (
                list(instruction_ids_list[b])
                + list(emitted[b])
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
        position_ids = (valid.long().cumsum(dim=1) - 1).clamp(min=0)

        out = llm(
            inputs_embeds=embeds,
            attention_mask=valid.long(),
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        logits = out.logits[:, -1]  # <ve> position -> predicts first word
        cur_pos = position_ids[:, -1] + 1  # (na,)
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
            emitted[b].extend(words[i])

    return emitted
