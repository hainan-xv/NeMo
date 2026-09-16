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
"""ChunkScorer adapters for EncDecCHATBPEModel and ScriptSTTModel.

The search in :mod:`joint_decode` is deliberately model-agnostic; these are the
two implementations that make it real. Both must emit a vector of exactly
``V + 1`` log-probs per step with the SAME meaning in every slot, which is the
only thing that makes an elementwise fusion legitimate:

    slot 0 .. V-1   Qwen3 text tokens, identical indices in both models
    slot V          END -- "this chunk is finished"

The two models express END differently and neither uses slot ``V`` natively:

    CHAT    the transducer blank, already at index ``V`` (``num_classes_with_blank
            - 1``) because blank sits just past the tokenizer. Nothing to remap.
    SCRIPT  ``<|im_end|>`` at 151645, which is INSIDE the text range. Left alone
            it would be scored twice -- once as its own text token and once as
            END -- so it is MOVED: copied to slot V and vetoed at 151645.

SCRIPT also has two in-vocab audio delimiters (``<|vision_start|>`` 151652 /
``<|vision_end|>`` 151653). They are structural markers, never transcript, so
they are vetoed too. Missing that would let the fusion emit a delimiter as text.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch

__all__ = ["NEG_INF", "ChatChunkScorer", "ScriptChunkScorer", "remap_script_logprobs"]

# Veto value for slots that must never be emitted. Finite rather than -inf so a
# weighted sum can never produce nan from 0 * -inf.
NEG_INF = -1e30


def remap_script_logprobs(
    lp: torch.Tensor,
    vocab_size: int,
    eot_id: int,
    veto_ids: Sequence[int] = (),
) -> torch.Tensor:
    """SCRIPT's LM distribution -> the canonical ``[V + 1]`` layout.

    Args:
        lp: ``[vocab_llm]`` log-probs straight from the LM head.
        vocab_size: ``V``, the CHAT/tokenizer text vocabulary size.
        eot_id: SCRIPT's end-of-chunk token, moved into slot ``V``.
        veto_ids: structural ids that must never surface as text.

    The LLM's embedding matrix is usually WIDER than the tokenizer (Qwen3 pads
    the rows for alignment), so the tail past ``V`` is dropped rather than
    assumed absent -- those rows are untrained and would otherwise compete.
    """
    if lp.dim() != 1:
        raise ValueError(f"expected a 1-D distribution, got shape {tuple(lp.shape)}")
    if lp.numel() < vocab_size:
        raise ValueError(f"LM vocab {lp.numel()} is smaller than text vocab {vocab_size}")

    out = torch.full((vocab_size + 1,), NEG_INF, dtype=torch.float32, device=lp.device)
    out[:vocab_size] = lp[:vocab_size].float()
    # END first, THEN the veto -- eot_id < vocab_size, so vetoing it before the
    # copy would blank out the very value being moved.
    out[vocab_size] = lp[eot_id].float()
    out[eot_id] = NEG_INF
    for vid in veto_ids:
        if 0 <= vid < vocab_size:
            out[vid] = NEG_INF
    return out


class ChatChunkScorer:
    """EncDecCHATBPEModel as a ChunkScorer.

    ``joint_on_path`` already returns ``[N, V + 1]`` log-probs at chosen
    ``(b, t, u)`` triples with blank last, so the layout needs no translation --
    ``t`` is the chunk index and ``u`` the number of labels emitted, exactly the
    axes the search steps along.
    """

    def __init__(self, model, encoded: torch.Tensor, encoded_len: torch.Tensor):
        """
        Args:
            model: a constructed EncDecCHATBPEModel.
            encoded: RAW encoder output ``[1, T_frames, D]``. joint_on_path does
                its own chunking, so passing pre-chunked features would chunk
                twice and silently shift every boundary.
            encoded_len: ``[1]`` valid frame count.
        """
        self.m = model
        self.encoded = encoded
        self.encoded_len = encoded_len
        self.blank = model.joint.num_classes_with_blank - 1
        self.vocab_size = self.blank  # blank sits just past the text vocabulary

    def init_state(self) -> Tuple[int, ...]:
        return ()  # the emitted token prefix IS the state

    @torch.no_grad()
    def logprobs(self, state: Tuple[int, ...], chunk_idx: int) -> torch.Tensor:
        toks = torch.tensor([list(state) or [0]], dtype=torch.long, device=self.encoded.device)
        lens = torch.tensor([len(state)], dtype=torch.long, device=self.encoded.device)
        g, _, _ = self.m.decoder(targets=toks, target_length=lens)
        g = g.transpose(1, 2)
        idx = lambda v: torch.tensor([v], dtype=torch.long, device=self.encoded.device)  # noqa: E731
        out = self.m.joint.joint_on_path(self.encoded, g, idx(0), idx(chunk_idx), idx(len(state)), self.encoded_len)
        return out[0].float()

    def advance(self, state: Tuple[int, ...], token: int) -> Tuple[int, ...]:
        return state + (token,)

    def close_chunk(self, state: Tuple[int, ...], chunk_idx: int) -> Tuple[int, ...]:
        # u keeps counting across chunks in the transducer lattice, so the token
        # prefix carries over untouched; only t advances, and the search owns t.
        return state


class ScriptChunkScorer:
    """ScriptSTTModel as a ChunkScorer.

    Rebuilds, per chunk, the same conditioning ``batched_stream_decode_script``
    uses -- ``instruction + text history + <vision_start> audio_chunk
    <vision_end>`` -- and then steps one token at a time inside the chunk.

    History is PLAIN TEXT, re-embedded each chunk, which is what the model was
    trained on: the spine carries no audio and the branch sees one chunk's
    window. Carrying a KV cache across chunk boundaries would condition on
    something training never showed it.
    """

    def __init__(
        self,
        model,
        frames: torch.Tensor,
        instruction_ids: Sequence[int],
        chunk_size: int,
        vocab_size: int,
    ):
        """
        Args:
            model: a constructed ScriptSTTModel.
            frames: encoded audio ``[T_frames, D]`` from ``encode_frames``.
            instruction_ids: tokenized system prompt; MUST be byte-identical to
                the training instruction or every chunk is out of distribution.
            chunk_size: frames per chunk, matching CHAT's grid.
            vocab_size: ``V``, taken from CHAT so both scorers agree.
        """
        self.m = model
        self.frames = frames
        self.instruction_ids = list(instruction_ids)
        self.chunk_size = chunk_size
        self.vocab_size = vocab_size
        self.eot_id = model._eot_id
        self.veto = (model._vision_start_id, model._vision_end_id)

    def init_state(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        # (committed history across chunks, tokens emitted in the current chunk)
        return ((), ())

    def _chunk_frames(self, chunk_idx: int) -> torch.Tensor:
        lo = chunk_idx * self.chunk_size
        hi = min(lo + self.chunk_size, self.frames.shape[0])
        return self.frames[lo:hi]

    @torch.no_grad()
    def logprobs(self, state, chunk_idx: int) -> torch.Tensor:
        history, cur = state
        dev = self.frames.device
        emb = self.m._embed_tokens

        text_ids = list(self.instruction_ids) + list(history)
        prefix = emb(torch.tensor([text_ids], dtype=torch.long, device=dev))
        vs = emb(torch.tensor([[self.m._vision_start_id]], dtype=torch.long, device=dev))
        ve = emb(torch.tensor([[self.m._vision_end_id]], dtype=torch.long, device=dev))
        audio = self._chunk_frames(chunk_idx).unsqueeze(0)

        parts = [prefix, vs, audio, ve]
        if cur:
            parts.append(emb(torch.tensor([list(cur)], dtype=torch.long, device=dev)))
        embeds = torch.cat(parts, dim=1)

        logits = self.m.llm(inputs_embeds=embeds).logits[0, -1]
        return remap_script_logprobs(logits.float().log_softmax(-1), self.vocab_size, self.eot_id, self.veto)

    def advance(self, state, token: int):
        history, cur = state
        return (history, cur + (token,))

    def close_chunk(self, state, chunk_idx: int):
        history, cur = state
        return (history + cur, ())
