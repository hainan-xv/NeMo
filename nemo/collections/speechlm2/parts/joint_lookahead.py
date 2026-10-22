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
"""Joint decoding of one span under two different amounts of right context.

Motivation, measured. A frame at offset ``i`` of a chunk of ``C`` frames sees
``C-1-i`` frames of right context, so the chunk-final frames see none -- and
inverting the chunk-size sweep shows per-frame error falling 8.06 -> 5.78 ->
5.25 -> 4.90 as right context grows 0-1 -> 2-6 -> 7-13 -> 14-27. The curve is
steeply convex at the bottom, so the chunk-final frames are where the errors
concentrate. Controlling for word identity by shifting the grid, the same word
is wrong 3.33% of the time when chunk-final against 2.87% when mid-chunk.

What this does. The commit unit is ``U`` frames, but the span is decoded TWICE
from the same history and the same start:

    S  window [t, t+U)   chunk U declared   <- this IS the commit unit
    L  window [t, t+C)   chunk C declared   <- same words, more right context

and the two log-prob distributions are averaged at every token. Because both
stand on the same history and the same tokenization of the same words, their
log-probs are commensurable -- the same argument ``chat_fusion`` makes for
fusing an external chat model.

WHY NOT TWO SHIFTED GRIDS. Token-level averaging needs both members predicting
the same next token. Two independently-phased decoders hold different histories
and emit at interleaved times, so there is no common token to average; they can
only be combined post hoc, and a position-based post-hoc selector was measured
to recover just 0.09 WER of a 0.72 oracle. Sharing the history and varying only
the lookahead is what makes the ensemble well defined.

THE EOT ASYMMETRY. L's window runs ``C-U`` frames past the commit boundary, so
L always wants to keep emitting into the next unit. Averaging its stop decision
in would systematically over-emit. S owns the boundary and therefore owns the
stop: L's ``<eot>`` is masked out of the average entirely.

ENCODER PHASE. Right context comes from the ENCODER's chunk grid, which
``encode_frames`` anchors at frame 0. For L's window ``[t, t+C)`` to actually
carry lookahead to ``t+C`` there must be a grid boundary at ``t``, which holds
only for every other commit unit when ``U = C/2``. So L is encoded twice, on
grids anchored at 0 and at U, and the unit's parity picks the pass. The second
pass is built by starting the encode later in the WAVEFORM, never by prepending
silence: padding with 0.56 s of zeros was measured to cost 0.32 WER on its own,
which would swamp anything this buys.
"""

import math
from typing import List, Optional

import torch
from torch import Tensor

from nemo.collections.speechlm2.parts.script import AUDIO_TOKEN_IDX, BRANCH_SCHEME
from nemo.collections.speechlm2.parts.script_prompt import ScriptControls, render_control_prompt

SAMPLES_PER_FRAME = 1280


def _prefill_rows(model, seqs, chunk_frames, device, dtype):
    """Left-padded prefill whose last column is every row's <ve>."""
    pad_id = model.text_pad_id
    n = len(seqs)
    L = max(len(s) for s in seqs)
    H = chunk_frames[0].shape[-1]
    max_c = max(int(f.shape[0]) for f in chunk_frames)

    input_tokens = torch.full((n, L), pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        input_tokens[i, L - len(s) :] = torch.tensor(s, dtype=torch.long, device=device)

    audio_embs = torch.zeros(n, max_c, H, dtype=dtype, device=device)
    for i, fr in enumerate(chunk_frames):
        audio_embs[i, : fr.shape[0]] = fr

    audio_mask = input_tokens == AUDIO_TOKEN_IDX
    text_tokens = input_tokens.where(~audio_mask, torch.zeros_like(input_tokens))
    text_embeds = model._embed_tokens(text_tokens)
    frame_idx = audio_mask.long().cumsum(dim=1) - 1
    gather_idx = frame_idx.clamp(min=0).unsqueeze(-1).expand(n, L, H)
    embeds = torch.where(audio_mask.unsqueeze(-1), torch.gather(audio_embs, 1, gather_idx), text_embeds)

    valid = input_tokens != pad_id
    position_ids = (valid.long().cumsum(dim=1) - 1).clamp(min=0)
    out = model.llm(
        inputs_embeds=embeds,
        attention_mask=valid.long(),
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )
    return out, valid.long(), position_ids[:, -1] + 1


@torch.no_grad()
def joint_lookahead_decode(
    model,
    audios: Tensor,
    audio_lens: Tensor,
    system_prompt: str,
    commit_chunk: int,
    look_chunk: int,
    weight: float = 0.5,
    max_new_tokens: int = 64,
    num_delay_frames: Optional[int] = None,
    s_first_tokens: int = 0,
    history_text: Optional[List[str]] = None,
    _trace: Optional[list] = None,
) -> List[str]:
    """Decode ``audios`` committing ``commit_chunk`` frames at a time.

    Args:
        commit_chunk: U, the commit granularity, and S's declared chunk size.
        look_chunk: C, L's declared chunk size. Must be exactly ``2*U``; unit
            ``k`` takes its lookahead from encoder phase ``k % 2``.
        s_first_tokens: how many tokens at the start of each unit come from S
            alone. Protects words S deferred from the previous unit, which L
            knows nothing about. 0 disables.
        weight: mixing weight on L's log-probs. 0 reproduces S alone (a plain
            chunk-U decode), 1 is L alone with S supplying only the stop
            decision, 0.5 is the even ensemble. Sweeping it is the cheapest way
            to tell whether the ensemble or merely the longer lookahead is doing
            the work.
    """
    U, C = int(commit_chunk), int(look_chunk)
    # TWO PHASES ONLY: the original grid and the one cut in the middle. A
    # general C = n*U would need n encoder passes and n-way parity, which is
    # more machinery than the idea needs while it is still being tested.
    if C != 2 * U:
        raise ValueError(f"look_chunk must be exactly 2*commit_chunk; got U={U}, C={C}")
    for name, cs in (("commit_chunk", U), ("look_chunk", C)):
        trained = list(getattr(model.core_cfg, "chunk_size", []) or [])
        if trained and cs not in trained:
            # A size the model never saw still decodes to fluent text, so this
            # would look like it worked while measuring nothing.
            raise ValueError(f"{name}={cs} is not a trained chunk size {trained}")
    if int(getattr(model.core_cfg, "audio_history_chunks", 0) or 0) or int(
        getattr(model.core_cfg, "audio_window_frames", 0) or 0
    ):
        raise NotImplementedError("joint_lookahead_decode assumes a per-chunk audio window")
    if getattr(model, "_decode_position_scheme", BRANCH_SCHEME) != BRANCH_SCHEME:
        raise NotImplementedError("joint_lookahead_decode implements the branch position scheme only")

    if getattr(model, "_read_id", None) is not None or getattr(model, "_write_id", None) is not None:
        raise NotImplementedError("joint_lookahead_decode does not implement the read/write gate")

    device = model.device
    B = audios.shape[0]
    dtype = model._embed_tokens(torch.zeros(1, 1, dtype=torch.long, device=device)).dtype
    vs, ve, eot = model._vision_start_id, model._vision_end_id, model._eot_id
    # Chunk-start fix-up, exactly as batched_stream_decode_script does it: when a
    # unit's first token is not a word start, a space token is prepended so the
    # history matches the training spine. Skipping it makes the conditioning
    # drift token-by-token from the reference decode, which is what made
    # weight=0 disagree with a plain chunk-U decode instead of reproducing it.
    insert_ws = model._get_word_start_insert_id() if model.core_cfg.force_word_start else None
    is_ws = model._is_word_start if insert_ws is not None else None

    # S: one grid of chunk U anchored at 0 serves every commit unit.
    frames_S = model.encode_frames(audios, audio_lens, U)
    # L on BOTH phases: phase 0's grid is anchored at frame 0, phase 1's at U.
    # Phase 1 is built by starting the encode U frames later in the WAVEFORM,
    # never by prepending silence -- 0.56 s of leading zeros costs 0.32 WER.
    off = U * SAMPLES_PER_FRAME
    frames_L = [
        model.encode_frames(audios, audio_lens, C),
        model.encode_frames(audios[:, off:], (audio_lens - off).clamp(min=0), C),
    ]

    # The instruction is NOT the bare prompt: with prompt_control the model was
    # trained on an instruction carrying its chunk size, emission delay,
    # capitalisation and punctuation. Passing the bare prompt puts it out of
    # distribution -- it emitted on the FIRST chunk where the reference stayed
    # silent, and fragmented the utterance across 47 chunks instead of 40.
    # Each member must also declare its OWN chunk size, since that is exactly
    # what differs between them.
    prompts = {}
    for tag, cs in (("S", U), ("L", C)):
        if model.core_cfg.prompt_control:
            controls = ScriptControls(
                chunk_size=cs,
                num_delay_frames=(
                    int(model.core_cfg.val_num_delay_frames) if num_delay_frames is None else int(num_delay_frames)
                ),
                capitalization=bool(model.core_cfg.val_capitalization),
                punctuation=bool(model.core_cfg.val_punctuation),
            )
            prompts[tag] = render_control_prompt(
                system_prompt,
                controls,
                include_chunk_size=bool(getattr(model.core_cfg, "prompt_include_chunk_size", True)),
            )
        else:
            prompts[tag] = system_prompt

    n_units = [max(0, math.ceil(int(f.shape[0]) / U)) for f in frames_S]
    hist: List[List[int]] = [
        list(model.tokenizer.text_to_ids(h)) if history_text else [] for h in (history_text or [""] * B)
    ]
    instr = {t: model.tokenizer.text_to_ids(prompts[t] + "\n") for t in ("S", "L")}

    for k in range(max(n_units) if B else 0):
        active = [b for b in range(B) if k < n_units[b]]
        if not active:
            break
        # Even units start a phase-0 chunk, odd units a phase-1 chunk, so each
        # gets lookahead running C frames past its own start.
        p = k % 2
        seqs, frs = [], []
        for b in active:
            for which in ("S", "L"):
                if which == "S":
                    fr = frames_S[b][k * U : (k + 1) * U]
                    want = U
                else:
                    # Phase p's frame j is real frame j + p*U.
                    base = k * U - p * U
                    fr = frames_L[p][b][base : base + C]
                    want = C
                fr = fr.to(device=device, dtype=dtype)
                if fr.shape[0] < want:  # final partial unit: zero-pad as training did
                    fr = torch.cat([fr, fr.new_zeros(want - fr.shape[0], fr.shape[1])], 0)
                frs.append(fr)
                seqs.append(list(instr[which]) + list(hist[b]) + [vs] + [AUDIO_TOKEN_IDX] * int(fr.shape[0]) + [ve])

        out, attn, cur_pos = _prefill_rows(model, seqs, frs, device, dtype)
        cache, logits = out.past_key_values, out.logits[:, -1]
        na = len(active)
        words: List[List[int]] = [[] for _ in range(na)]
        done = [False] * na
        for step_idx in range(max_new_tokens):
            lp = torch.log_softmax(logits.float(), dim=-1)  # (2*na, V)
            lpS, lpL = lp[0::2], lp[1::2]
            # DEFERRED WORDS BELONG TO S ALONE. S declares chunk U with an
            # emission delay, so it holds back a word ending near t+U and emits
            # it at the START of the next unit -- from text context, since its
            # window no longer contains that word's audio. L, whose window is
            # [t+U, t+U+C), has no notion of that word and wants the first word
            # of its own window instead. Averaging the two suppresses the
            # deferred word, and nothing downstream can recover it: no later
            # window contains its audio either. Measured, that showed up as
            # deletions 3.07 vs chunk-14's 2.71 while substitutions were nearly
            # matched (2.86 vs 2.78) -- the lookahead was working and the
            # deferral was leaking.
            #
            # Costs almost nothing to protect: a unit's first word sits at the
            # START of S's window, where S already has its maximum right
            # context. The words that need L are the later ones, near S's blind
            # boundary.
            # L's window runs past the commit boundary, so L always wants to keep
            # going. S owns the boundary, so S owns the stop: mask L's <eot> out
            # of the mixture instead of letting it vote the span longer.
            #
            # STOP IS DECIDED BY S ALONE, OUTSIDE THE MIXTURE. Leaving <eot> at
            # S's raw log-prob while averaging every word column biases the
            # decode toward stopping: when S and L disagree about the next word,
            # the average sits below either member's mode, but the untouched
            # <eot> column does not move -- so stop wins comparisons it should
            # lose. Measured, that was deletions 3.07 (highest of any arm),
            # insertions 1.46 (lowest) and 0.984 words emitted per reference
            # word (fewest): a stop bias, not a deferral problem.
            #
            # So S decides stop exactly as it would decoding alone, and L only
            # influences WHICH word is emitted once S has decided to emit one.
            # This also keeps the weight=0 identity exact.
            if step_idx < s_first_tokens:
                mixed = lpS.clone()
            else:
                mixed = (1.0 - weight) * lpS + weight * lpL
            stop_now = lpS.argmax(dim=-1) == eot
            mixed = mixed.clone()
            mixed[:, eot] = torch.finfo(mixed.dtype).min
            nxt = torch.where(stop_now, torch.full_like(stop_now, eot, dtype=torch.long), mixed.argmax(dim=-1))
            step = []
            for i in range(na):
                t = int(nxt[i])
                if done[i] or t == eot:
                    done[i] = True
                    step.append(eot)  # filler for a finished row; it is ignored
                else:
                    words[i].append(t)
                    step.append(t)
            if all(done):
                break
            # Feed the SAME chosen token to both members so their caches stay on
            # one shared sequence -- that is what keeps the next step's two
            # distributions commensurable.
            tok = torch.tensor(step, dtype=torch.long, device=device).repeat_interleave(2).unsqueeze(1)
            attn = torch.cat([attn, torch.ones(attn.shape[0], 1, dtype=attn.dtype, device=device)], dim=1)
            res = model.llm(
                inputs_embeds=model._embed_tokens(tok),
                attention_mask=attn,
                position_ids=cur_pos.unsqueeze(1),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache, logits = res.past_key_values, res.logits[:, -1]
            cur_pos = cur_pos + 1
        for i, b in enumerate(active):
            toks = words[i]
            if _trace is not None and b == 0:
                _trace.append((k, list(toks)))
            if insert_ws is not None and toks and hist[b] and not is_ws(toks[0]):
                toks = [insert_ws] + toks
            hist[b].extend(toks)

    return [model.tokenizer.ids_to_text(h) if h else "" for h in hist]
