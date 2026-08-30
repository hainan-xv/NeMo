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
"""Structure-aware attention for the SCRIPT packed layout.

The packed sequence keeps every branch in ONE sequence and uses a 4D mask to
forbid the cross-branch pairs (:func:`~...parts.script.build_script_mask`). That
is correct but wasteful in the extreme: the mask is only ~2% dense at the shapes
that matter, so nearly all of the ``T x T`` scores are computed and discarded,
and the ``(B, heads, T, T)`` score tensor is the largest allocation in the step.

This module computes only the blocks the mask permits, while leaving the
sequence flat -- so the QKV/MLP projections stay one big GEMM with no padding
waste, one kernel launch, and no per-utterance Python loop::

    QKV projection            one GEMM over the flat sequence   (unchanged)
                                        |
              +-------------------------+-------------------------+
        spine -> spine             branch -> spine           branch -> own
        causal (P, P)              (N, b) queries x P        block-diagonal
                                        +---- one joint softmax ----+
                                        |
    scatter back to flat order  ->  MLP                            (unchanged)

The two branch blocks share a SINGLE softmax over the concatenated key axis, so
the result is exactly the dense-mask result -- this is a decomposition, not an
approximation. ``test_structured_attention_matches_dense`` and its gradient
counterpart are the gate on that.

Branches are ragged (early chunks have shorter audio windows; chunks reveal
different numbers of words), so they are gathered onto a padded ``(N, b_max)``
grid inside the attention op and scattered back afterwards. That padding never
reaches the linear layers.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor

from nemo.collections.speechlm2.parts.script import PackedChunkExample


@dataclass
class ScriptAttentionPlan:
    """Where the spine and each branch live inside the flat packed sequence.

    Built once per batch in the dataloader (it is pure indexing), then consumed
    by every layer of the forward.

    Attributes:
        spine_pos: (B, P) flat index of each spine token. Spine positions equal
            their index, so grid column ``j`` is position ``j``.
        spine_valid: (B, P) False where an utterance's spine is shorter than the
            batch maximum.
        branch_pos: (B, N, b) flat index of each branch token.
        branch_valid: (B, N, b) False at ragged/padded branch slots.
        branch_prefix: (B, N) how many spine tokens each branch may attend.
        branch_is_audio: (B, N, b) True at audio slots. Always populated; it is
            only CONSULTED when ``bidirectional_audio`` is set, so one plan can
            serve either rule.
        bidirectional_audio: let each branch's audio block attend itself both
            ways, matching :func:`~...parts.script.build_script_mask` when that
            function is given ``is_audio``.
    """

    spine_pos: Tensor
    spine_valid: Tensor
    branch_pos: Tensor
    branch_valid: Tensor
    branch_prefix: Tensor
    branch_is_audio: Optional[Tensor] = None
    bidirectional_audio: bool = False

    def to(self, device) -> "ScriptAttentionPlan":
        return ScriptAttentionPlan(
            spine_pos=self.spine_pos.to(device),
            spine_valid=self.spine_valid.to(device),
            branch_pos=self.branch_pos.to(device),
            branch_valid=self.branch_valid.to(device),
            branch_prefix=self.branch_prefix.to(device),
            branch_is_audio=None if self.branch_is_audio is None else self.branch_is_audio.to(device),
            bidirectional_audio=self.bidirectional_audio,
        )


def build_attention_plan(examples: List[PackedChunkExample], bidirectional_audio: bool = False) -> ScriptAttentionPlan:
    """Derive the plan from the same examples the flat collate consumes."""
    B = len(examples)
    spine_lens = [int(e.spine_len) for e in examples]
    P = max(spine_lens)

    rows_per_example = []
    for e in examples:
        n = int(e.seg_ids.max())
        rows = [(e.seg_ids == k + 1).nonzero(as_tuple=True)[0] for k in range(n)]
        rows_per_example.append(rows)
    N = max((len(r) for r in rows_per_example), default=0)
    b = max((int(x.numel()) for rows in rows_per_example for x in rows), default=0)

    spine_pos = torch.zeros(B, P, dtype=torch.long)
    spine_valid = torch.zeros(B, P, dtype=torch.bool)
    branch_pos = torch.zeros(B, N, b, dtype=torch.long)
    branch_valid = torch.zeros(B, N, b, dtype=torch.bool)
    branch_prefix = torch.zeros(B, N, dtype=torch.long)
    branch_is_audio = torch.zeros(B, N, b, dtype=torch.bool)

    for i, (e, rows) in enumerate(zip(examples, rows_per_example)):
        p = spine_lens[i]
        spine_pos[i, :p] = torch.arange(p)
        spine_valid[i, :p] = True
        for k, r in enumerate(rows):
            m = int(r.numel())
            branch_pos[i, k, :m] = r
            branch_valid[i, k, :m] = True
            branch_prefix[i, k] = int(e.prefix_len[r[0]])
            branch_is_audio[i, k, :m] = e.is_audio[r]

    return ScriptAttentionPlan(
        spine_pos,
        spine_valid,
        branch_pos,
        branch_valid,
        branch_prefix,
        branch_is_audio=branch_is_audio,
        bidirectional_audio=bidirectional_audio,
    )


# ---------------------------------------------------------------------------
# Active plan
#
# HF's attention interface has no channel for custom per-batch metadata, so the
# plan is published for the duration of one forward. Every layer sees the same
# structure, and with no plan set the attention falls back to SDPA -- which is
# what keeps validation and generate() working unchanged.
# ---------------------------------------------------------------------------

_ACTIVE_PLAN: Optional[ScriptAttentionPlan] = None


@contextmanager
def script_attention_plan(plan: Optional[ScriptAttentionPlan]):
    global _ACTIVE_PLAN
    prev = _ACTIVE_PLAN
    _ACTIVE_PLAN = plan
    try:
        yield
    finally:
        _ACTIVE_PLAN = prev


def active_plan() -> Optional[ScriptAttentionPlan]:
    return _ACTIVE_PLAN


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand grouped-query KV heads to match the query head count."""
    if n_rep == 1:
        return x
    B, h_kv, T, d = x.shape
    return x[:, :, None].expand(B, h_kv, n_rep, T, d).reshape(B, h_kv * n_rep, T, d)


def script_structured_attention(
    module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor] = None,
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs,
):
    """HF attention backend computing only the SCRIPT-permitted blocks.

    Falls back to SDPA whenever no plan is active, so the same model can run the
    structured path for training and the ordinary path for decoding.
    """
    plan = active_plan()
    if plan is None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        return sdpa_attention_forward(
            module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs
        )

    B, h, T, d = query.shape
    key = _repeat_kv(key, h // key.shape[1])
    value = _repeat_kv(value, h // value.shape[1])
    scale = scaling if scaling is not None else d**-0.5
    neg = torch.finfo(query.dtype).min

    sp_pos, sp_valid = plan.spine_pos, plan.spine_valid
    br_pos, br_valid, br_pref = plan.branch_pos, plan.branch_valid, plan.branch_prefix
    P, N, b = sp_pos.shape[1], br_pos.shape[1], br_pos.shape[2]

    def gather(x, idx):  # (B,h,T,d) -> (B,h,*idx.shape,d)
        flat = idx.reshape(B, -1)
        out = torch.gather(x, 2, flat[:, None, :, None].expand(B, h, flat.shape[1], d))
        return out.reshape(B, h, *idx.shape[1:], d)

    q_sp, k_sp, v_sp = gather(query, sp_pos), gather(key, sp_pos), gather(value, sp_pos)

    # --- spine -> spine: ordinary causal over the spine only ---
    tri = torch.ones(P, P, dtype=torch.bool, device=query.device).tril()
    m_sp = (tri[None, None] & sp_valid[:, None, None, :]) & sp_valid[:, None, :, None]
    s = torch.einsum("bhid,bhjd->bhij", q_sp, k_sp) * scale
    o_sp = torch.softmax(s.masked_fill(~m_sp, neg), dim=-1) @ v_sp

    # --- branches: gathered onto the regular (N, b) grid ---
    q_br, k_br, v_br = gather(query, br_pos), gather(key, br_pos), gather(value, br_pos)

    # branch -> its own slice of the spine
    s_sp = torch.einsum("bhnid,bhpd->bhnip", q_br, k_sp) * scale
    allow = (torch.arange(P, device=query.device)[None, None, :] < br_pref[:, :, None]) & sp_valid[:, None, :]
    s_sp = s_sp.masked_fill(~allow[:, None, :, None, :], neg)

    # branch -> itself, block-diagonal: the (N*b)^2 matrix is never formed
    s_own = torch.einsum("bhnid,bhnjd->bhnij", q_br, k_br) * scale
    j = torch.arange(b, device=query.device)
    causal = (j[None, :] <= j[:, None])[None, None, None]
    if plan.bidirectional_audio and plan.branch_is_audio is not None:
        # (B, N, b, b) rather than the shared (b, b): the audio run's length
        # varies per branch whenever audio_window_frames is 0, so this cannot be
        # a single triangular matrix. Still h-times smaller than s_own.
        aud = plan.branch_is_audio
        causal = causal | (aud[:, :, :, None] & aud[:, :, None, :])[:, None]
    s_own = s_own.masked_fill(~(causal & br_valid[:, None, :, None, :]), neg)

    # ONE softmax over [spine keys | own keys] -- this is what makes it exact
    p = torch.softmax(torch.cat([s_sp, s_own], dim=-1), dim=-1)
    if dropout > 0.0:
        p = torch.nn.functional.dropout(p, p=dropout, training=module.training)
    o_br = torch.einsum("bhnip,bhpd->bhnid", p[..., :P], v_sp)
    o_br = o_br + torch.einsum("bhnij,bhnjd->bhnid", p[..., P:], v_br)

    # --- scatter back into flat order ---
    out = torch.zeros_like(query)
    out = out.scatter(
        2, sp_pos[:, None, :, None].expand(B, h, P, d), o_sp.masked_fill(~sp_valid[:, None, :, None], 0.0)
    )
    flat_pos = br_pos.reshape(B, N * b)
    o_br = (o_br * br_valid[:, None, :, :, None]).reshape(B, h, N * b, d)
    out = out.scatter_add(2, flat_pos[:, None, :, None].expand(B, h, N * b, d), o_br)

    return out.transpose(1, 2).contiguous(), None


def register_script_attention() -> None:
    """Make the backend selectable as ``attn_implementation="script"``."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["script"] = script_structured_attention
