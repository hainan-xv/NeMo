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
"""State-machine ("FSM") decoding for SCRIPT.

The counterpart of ``StreamingSTTModel._generate_dynamic_streaming`` for the
SCRIPT layout, built to mirror its structure so the two systems can be compared
under the same decode discipline. On the interleaved model the FSM path scores
materially better than the bulk-prefill path (6.03 vs 6.42 macro at chunk 14, and
the gap explodes at chunk 2), so the obvious question is whether SCRIPT is
leaving the same thing on the table. This module exists to answer that with
numbers rather than argument.

TWO THINGS DIFFER from :func:`~...parts.script.batched_stream_decode_script`, and
they are separable on purpose -- ``streaming_encode`` and ``fsm_decode`` can be
enabled independently, so a difference can be attributed:

1. **Perception.** The default SCRIPT eval encodes the WHOLE utterance in one
   offline pass with ``att_context_size = [left, chunk-1]``. That is
   dependency-equivalent to streaming (a frame never sees past its own chunk
   boundary -- see ``test_offline_encode_dependency_is_chunk_limited``) but it is
   NOT the same computation as running the cache-aware encoder chunk by chunk.
   On the interleaved model that exact distinction was worth 17.79 vs 5.51 macro,
   so it is the prime suspect here too. ``streaming_encode_frames`` runs the
   cache-aware path instead, mirroring the FSM's incremental perception.

2. **Decode stepping.** The default path bulk-prefills
   ``[instruction, history, <vs>, audio, <ve>]`` and then decodes. The FSM walks
   an explicit per-stream state machine one token at a time
   (LISTENING -> VE -> GENERATING -> EOT), mirroring the interleaved FSM's
   HEADER/LISTENING/FOOTER/GENERATING/ASST_FOOTER states.

   NOTE: for SCRIPT this second change is expected to be a no-op up to
   floating-point associativity -- the conditioning, the positions and the
   termination rule are identical, and unlike the interleaved chunked path SCRIPT
   never force-feeds a footer after an unterminated decode (the failure mode that
   makes the FSM win there). It is implemented anyway so that the comparison is
   structural rather than a claim.
"""

import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from nemo.collections.speechlm2.parts.script import AUDIO_TOKEN_IDX, BRANCH_SCHEME, audio_window_start


@torch.no_grad()
def streaming_encode_frames(model, audios: Tensor, audio_lens: Tensor, chunk_size: int) -> List[Tensor]:
    """Encode with the CACHE-AWARE streaming encoder, chunk by chunk.

    Mirrors the perception half of ``_generate_dynamic_streaming``: a mel feature
    buffer is fed ``chunk_size`` frames' worth of samples at a time, and the
    conformer runs with ``streaming=True`` carrying its channel/time caches
    across chunks.

    Returns the same structure as :meth:`ScriptSTTModel.encode_frames` -- ``B``
    tensors of ``(n_frames_b, H)`` -- so it is a drop-in replacement.
    """
    B = int(audios.shape[0])
    device = audios.device
    ref = model._embed_ref_tensor if hasattr(model, "_embed_ref_tensor") else next(model.perception.parameters())

    cache_lc, cache_lt, cache_lcl = model.perception.get_initial_cache_state(
        batch_size=B, dtype=ref.dtype, device=device
    )
    buf = model.get_audio_feature_buffer(batch_size=B, chunk_size_override=chunk_size)

    spf = model._samples_per_encoder_frame()
    chunk_samples = chunk_size * spf
    n_samples = [int(x) for x in audio_lens.tolist()]
    # Frames the offline encoder would produce, so the two paths return the same
    # lengths and the caller cannot tell them apart structurally.
    n_frames = [max(0, int(math.ceil(ns / spf))) for ns in n_samples]
    n_chunks = [int(math.ceil(nf / chunk_size)) if nf > 0 else 0 for nf in n_frames]
    max_chunks = max(n_chunks) if B else 0

    out: List[List[Tensor]] = [[] for _ in range(B)]
    for k in range(max_chunks):
        wavs = []
        for b in range(B):
            s = k * chunk_samples
            e = min(s + chunk_samples, n_samples[b])
            w = audios[b, s:e] if e > s else audios.new_zeros(0)
            if int(w.shape[0]) < chunk_samples:
                w = F.pad(w, (0, chunk_samples - int(w.shape[0])))
            wavs.append(w)

        features, right_paddings = buf.update(wavs)
        processed_signal = torch.stack(features).type_as(ref)
        processed_signal_length = torch.tensor(
            [processed_signal.shape[-1] - int(rp) for rp in right_paddings], device=device
        ).long()

        embs, _, new_cache = model.perception(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_lc,
            cache_last_time=cache_lt,
            cache_last_channel_len=cache_lcl,
            streaming=True,
        )
        if new_cache is not None:
            cache_lc = new_cache["cache_last_channel"]
            cache_lt = new_cache["cache_last_time"]
            cache_lcl = new_cache["cache_last_channel_len"]

        got = int(embs.shape[1])
        if got < chunk_size:  # encoder returned short; zero-pad to the chunk grid
            embs = F.pad(embs, (0, 0, 0, chunk_size - got))
        for b in range(B):
            if k < n_chunks[b]:
                out[b].append(embs[b, :chunk_size])

    frames: List[Tensor] = []
    for b in range(B):
        if not out[b]:
            frames.append(audios.new_zeros(0, int(model.perception.encoder.d_model)))
        else:
            frames.append(torch.cat(out[b], dim=0)[: n_frames[b]].contiguous())
    return frames


# ---------------------------------------------------------------------------
# FSM decode
# ---------------------------------------------------------------------------

# Mirrors the interleaved model's HEADER/LISTENING/FOOTER/GENERATING/ASST_FOOTER/DONE.
LISTENING, VE, GENERATING, EOT, DONE = range(5)


@torch.no_grad()
def fsm_stream_decode_script(
    llm,
    embed_tokens,
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
    is_word_start=None,
    insert_word_start_id: Optional[int] = None,
    read_id: Optional[int] = None,
    write_id: Optional[int] = None,
    gate_in_history: bool = False,
    position_scheme: str = BRANCH_SCHEME,
):
    """Per-stream state machine over SCRIPT's chunk structure.

    Same conditioning as :func:`batched_stream_decode_script` --
    ``p(words_k | text_history_<k, audio_k)`` -- but advanced through explicit
    states, one token per step, rather than prefilled in bulk.
    """
    if position_scheme != BRANCH_SCHEME:
        # This path lays out positions branch-style (the branch starts right
        # after the history). A continuous-scheme checkpoint expects its branch
        # shifted left by window+2; decoding it here would silently apply the
        # wrong RoPE geometry and look like a quality regression rather than a
        # configuration error.
        raise NotImplementedError(
            f"fsm_stream_decode_script only implements position_scheme='{BRANCH_SCHEME}', "
            f"got '{position_scheme}'. Use the default decode path for that checkpoint."
        )

    B = len(frames_list)
    if B == 0:
        return []
    if device is None:
        device = frames_list[0].device
    H = int(frames_list[0].shape[-1])
    dtype = embed_tokens(torch.zeros(1, 1, dtype=torch.long, device=device)).dtype

    n_frames = [int(f.shape[0]) for f in frames_list]
    n_chunks = [math.ceil(n / chunk_size) if n > 0 else 0 for n in n_frames]
    max_chunks = max(n_chunks) if B else 0
    M, Fw = max(int(audio_history_chunks), 0), max(int(audio_window_frames), 0)
    chunk_starts = [k * chunk_size for k in range(max_chunks)]

    emitted: List[List[int]] = [[] for _ in range(B)]

    for k in range(max_chunks):
        active = [b for b in range(B) if k < n_chunks[b]]
        if not active:
            break
        na = len(active)

        # ---- state per active stream ----
        state = [LISTENING] * na
        win, wlen = [], []
        for b in active:
            win_end = (k + 1) * chunk_size
            win_start = audio_window_start(k, chunk_starts, M, win_end, Fw)
            fr = frames_list[b][win_start:win_end].to(device=device, dtype=dtype)
            want = win_end - win_start
            if int(fr.shape[0]) < want:  # final partial chunk: zero-pad, as in training
                fr = torch.cat([fr, fr.new_zeros(want - int(fr.shape[0]), fr.shape[1])], dim=0)
            win.append(fr)
            wlen.append(int(fr.shape[0]))

        # ---- prefix: instruction + history + <vs>, prefilled once per chunk ----
        # The history is plain text and grows, so it is re-materialised per chunk
        # exactly as in the default path; the FSM governs what happens AFTER it.
        seqs = []
        for i, b in enumerate(active):
            hist = emitted[b]
            if max_history_tokens and len(hist) > max_history_tokens:
                hist = hist[-max_history_tokens:]
            seqs.append(list(instruction_ids_list[b]) + list(hist) + [vision_start_id])
        L = max(len(s) for s in seqs)
        toks = torch.full((na, L), pad_id, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            toks[i, L - len(s) :] = torch.tensor(s, dtype=torch.long, device=device)
        valid = toks != pad_id
        pos = (valid.long().cumsum(dim=1) - 1).clamp(min=0)
        cur_pos = pos[:, -1] + 1
        out = llm(
            inputs_embeds=embed_tokens(toks.where(valid, torch.zeros_like(toks))),
            attention_mask=valid.long(),
            position_ids=pos,
            use_cache=True,
            return_dict=True,
        )
        cache, attn = out.past_key_values, valid.long()

        # ---- LISTENING: feed the window one frame per step ----
        max_w = max(wlen)
        for t in range(max_w):
            step = torch.zeros(na, 1, H, dtype=dtype, device=device)
            for i in range(na):
                if t < wlen[i]:
                    step[i, 0] = win[i][t]
            attn = torch.cat([attn, torch.ones(na, 1, dtype=attn.dtype, device=device)], dim=1)
            out = llm(
                inputs_embeds=step,
                attention_mask=attn,
                position_ids=cur_pos.unsqueeze(1),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache, cur_pos = out.past_key_values, cur_pos + 1
        for i in range(na):
            state[i] = VE

        # ---- VE: the <ve> token, whose logits predict the first word ----
        ve = torch.full((na, 1), vision_end_id, dtype=torch.long, device=device)
        attn = torch.cat([attn, torch.ones(na, 1, dtype=attn.dtype, device=device)], dim=1)
        out = llm(
            inputs_embeds=embed_tokens(ve),
            attention_mask=attn,
            position_ids=cur_pos.unsqueeze(1),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache, cur_pos, logits = out.past_key_values, cur_pos + 1, out.logits[:, -1]
        for i in range(na):
            state[i] = GENERATING

        # ---- GENERATING: one token per step until <eot> ----
        words: List[List[int]] = [[] for _ in range(na)]
        for _ in range(max_new_tokens):
            nxt = logits.argmax(dim=-1)
            for i in range(na):
                if state[i] != GENERATING:
                    continue
                tid = int(nxt[i].item())
                if tid == eot_id:
                    state[i] = EOT
                else:
                    words[i].append(tid)
            if all(s != GENERATING for s in state):
                break
            feed = nxt.clone()
            for i in range(na):
                if state[i] != GENERATING:
                    feed[i] = eot_id  # inert filler; the row is ignored downstream
            attn = torch.cat([attn, torch.ones(na, 1, dtype=attn.dtype, device=device)], dim=1)
            out = llm(
                inputs_embeds=embed_tokens(feed.unsqueeze(1)),
                attention_mask=attn,
                position_ids=cur_pos.unsqueeze(1),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache, cur_pos, logits = out.past_key_values, cur_pos + 1, out.logits[:, -1]

        # ---- commit: identical bookkeeping to the default path ----
        for i, b in enumerate(active):
            tk = words[i]
            gate = None
            if read_id is not None and tk and tk[0] == read_id:
                gate, tk = read_id, []
            elif write_id is not None and tk and tk[0] == write_id:
                gate, tk = write_id, tk[1:]
            if (
                insert_word_start_id is not None
                and is_word_start is not None
                and tk
                and emitted[b]
                and not is_word_start(tk[0])
            ):
                tk = [insert_word_start_id] + tk
            if gate_in_history and gate is not None:
                tk = [gate] + tk
            emitted[b].extend(tk)

    return emitted
