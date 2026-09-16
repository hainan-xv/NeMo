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
"""CHAT-side scoring for chunk-synchronous joint decoding, fused into SCRIPT's
existing decode loop.

WHY IT LIVES HERE RATHER THAN IN A DECODER OF ITS OWN. The first version of this
work reimplemented the whole chunk-synchronous loop from scratch so the two
models could be combined symmetrically. It was correct -- it produced sensible
transcripts at every mixing weight -- and unusably slow: 390 s for a 5-second
clip, because it had none of the batching or KV caching that
``batched_stream_decode_script`` already has.

``batched_stream_decode_script`` is ALREADY chunk-synchronous: it walks chunk k,
decodes every active stream together, and keeps a KV cache across the tokens of a
chunk. So the fusion is one insert immediately before its ``argmax``, and the
result is mathematically identical -- the same greedy choice over

    lam * log p_chat(token | t, h) + (1 - lam) * log p_script(token | t, h)

-- while inheriting every optimisation the production decode already has.

INDEX SPACE. The fusion happens in SCRIPT's vocabulary indexing, not a neutral
one, precisely so nothing downstream changes: the loop's ``eot_id`` test, its
embedding lookup and its cache all keep working untouched. CHAT's transducer
blank, which lives at ``V`` just past the tokenizer, is therefore folded onto
SCRIPT's ``eot_id`` -- the two are the same event ("this chunk is over") wearing
different indices.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

__all__ = ["ChatFusionScorer", "fuse_into_script_logits"]

NEG_INF = -1e30


class ChatFusionScorer:
    """Per-step CHAT log-probs at the ``(b, t, u)`` triples SCRIPT is standing on.

    ``t`` is the chunk index and ``u`` the number of labels emitted so far, which
    are exactly the axes ``batched_stream_decode_script`` already tracks -- ``k``
    for the chunk and ``len(emitted[b]) + len(words[i])`` for the labels.
    """

    def __init__(self, model, encoded: torch.Tensor, encoded_len: torch.Tensor):
        """
        Args:
            model: a constructed EncDecCHATBPEModel.
            encoded: RAW encoder output ``[B, T_frames, D]``. ``joint_on_path``
                chunks internally, so handing it pre-chunked features would chunk
                twice and silently shift every boundary.
            encoded_len: ``[B]`` valid frame counts.
        """
        self.m = model
        self.encoded = encoded
        self.encoded_len = encoded_len
        self.blank = model.joint.num_classes_with_blank - 1
        self.vocab_size = self.blank

    @torch.no_grad()
    def logprobs(self, b_idx: Sequence[int], chunk_idx: int, u_list: Sequence[int]) -> torch.Tensor:
        """``[len(b_idx), V + 1]`` log-probs, blank last.

        The prediction network is recomputed from each stream's full prefix. That
        is O(U) per step rather than incremental, but it is a small recurrent net
        run ONCE for the whole batch, against a 1.7B-parameter LLM forward on the
        other side -- it is not where the time goes, and keeping it stateless
        avoids a second copy of the transducer's state bookkeeping.
        """
        dev = self.encoded.device
        prefixes = [list(p) for p in u_list] if u_list and isinstance(u_list[0], (list, tuple)) else None
        if prefixes is None:
            raise ValueError("u_list must carry each stream's token PREFIX, not just its length")

        n = len(b_idx)
        lens = torch.tensor([len(p) for p in prefixes], dtype=torch.long, device=dev)
        width = max(1, int(lens.max().item()))
        targets = torch.zeros((n, width), dtype=torch.long, device=dev)
        for i, p in enumerate(prefixes):
            if p:
                targets[i, : len(p)] = torch.tensor(p, dtype=torch.long, device=dev)

        g, _, _ = self.m.decoder(targets=targets, target_length=lens)
        g = g.transpose(1, 2)

        # joint_on_path indexes g by u, so g must be gathered per stream at that
        # stream's own u -- the triples are (row i, chunk k, u_i), not a grid.
        bt = torch.tensor(list(b_idx), dtype=torch.long, device=dev)
        tt = torch.full((n,), int(chunk_idx), dtype=torch.long, device=dev)
        ut = lens.clone()

        enc = self.encoded[bt]
        enc_len = self.encoded_len[bt]
        rows = torch.arange(n, dtype=torch.long, device=dev)
        out = self.m.joint.joint_on_path(enc, g, rows, tt, ut, enc_len)
        return out.float()


def fuse_into_script_logits(
    script_logits: torch.Tensor,
    chat_lp: torch.Tensor,
    lam: float,
    eot_id: int,
    veto_ids: Sequence[int] = (),
) -> torch.Tensor:
    """Combine in SCRIPT's index space, returning a drop-in replacement for ``logits``.

    Args:
        script_logits: ``[n, vocab_llm]`` RAW logits from the LLM head.
        chat_lp: ``[n, V + 1]`` CHAT log-probs, blank last.
        lam: weight on CHAT. ``0.0`` reduces to SCRIPT exactly.
        eot_id: SCRIPT's end-of-chunk id; CHAT's blank is folded onto it.
        veto_ids: ids that must never be emitted as text (audio delimiters).

    Both sides are converted to log-probs first. Mixing a raw logit with a
    log-prob would make ``lam`` meaningless -- the two live on different scales,
    and the effective weight would drift with the logits' magnitude.
    """
    n, vocab_llm = script_logits.shape
    V = chat_lp.shape[1] - 1
    if V > vocab_llm:
        raise ValueError(f"CHAT vocab {V} exceeds SCRIPT LM width {vocab_llm}")

    s_lp = script_logits.float().log_softmax(-1)
    out = torch.full_like(s_lp, NEG_INF)

    # Text range first...
    out[:, :V] = lam * chat_lp[:, :V] + (1.0 - lam) * s_lp[:, :V]
    # ...then END, which overwrites eot_id's text-range value on purpose: CHAT's
    # blank and SCRIPT's <|im_end|> are the same event, and eot_id < V, so the
    # line above has just written the WRONG thing into that column.
    out[:, eot_id] = lam * chat_lp[:, V] + (1.0 - lam) * s_lp[:, eot_id]
    for vid in veto_ids:
        if 0 <= vid < vocab_llm:
            out[:, vid] = NEG_INF
    return out
