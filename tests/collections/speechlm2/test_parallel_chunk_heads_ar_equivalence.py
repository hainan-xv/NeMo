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
"""K=1 parallel-chunk-head ⇔ autoregressive (AR) equivalence.

When the parallel block size is K=1, the multi-token-per-chunk machinery
degenerates to plain autoregression: the heads emit one token per LLM forward
and re-anchor on that token's hidden state — exactly the AR decode loop.

For the *logits* to be bit-identical to AR, the parallel head must be a
pass-through over the anchor hidden state. That holds iff:
  * num_slots == 1,
  * depth_layers == 0 (no depth transformer), and
  * slot_embeds == 0 (no additive slot bias),
in which case ``heads(h, lm_head) == lm_head(h)``.

These tests assert that equivalence at three levels:
  1. Head forward:    slot-0 logits == lm_head(anchor)   (bit-exact).
  2. Training target: K=1 per-block target == AR next-token target.
  3. Decode loop:     K=1 ``_parallel_chunk_step_decode`` == greedy AR loop.
"""

from types import SimpleNamespace

import pytest
import torch

from nemo.collections.speechlm2.modules.parallel_chunk_heads import ParallelChunkHeads

# Mirrors nemo.collections.speechlm2.data.streaming_stt_dataset.IGNORE_INDEX.
IGNORE_INDEX = -100


def _passthrough_heads(hidden_size: int) -> ParallelChunkHeads:
    """A K=1 head with no depth transformer and zeroed slot embeddings, so that
    ``heads(h, lm_head)`` reduces exactly to ``lm_head(h)``."""
    heads = ParallelChunkHeads(hidden_size=hidden_size, num_slots=1, depth_layers=0)
    with torch.no_grad():
        heads.slot_embeds.zero_()
    heads.eval()
    return heads


# ===========================================================================
# 1. Head forward equivalence: heads(anchor) == lm_head(anchor)
# ===========================================================================
class TestHeadForwardEquivalence:

    def test_k1_passthrough_head_equals_lm_head(self):
        torch.manual_seed(0)
        H, V, M = 16, 32, 5
        lm_head = torch.nn.Linear(H, V, bias=False)
        heads = _passthrough_heads(H)

        anchor = torch.randn(M, H)
        par_logits = heads(anchor, lm_head)  # (M, 1, V)
        ar_logits = lm_head(anchor)  # (M, V)

        assert par_logits.shape == (M, 1, V)
        # Same float ops in the same order → bit-exact.
        torch.testing.assert_close(par_logits[:, 0, :], ar_logits, rtol=0.0, atol=0.0)

    def test_depth_transformer_breaks_exact_equivalence(self):
        """Sanity check on the *condition*: with a real depth transformer (>=1
        layer), K=1 is NOT bit-identical to lm_head — equivalence requires the
        pass-through configuration above."""
        torch.manual_seed(0)
        H, V, M = 16, 32, 5
        lm_head = torch.nn.Linear(H, V, bias=False)
        heads = ParallelChunkHeads(hidden_size=H, num_slots=1, depth_layers=1)
        heads.eval()

        anchor = torch.randn(M, H)
        par_logits = heads(anchor, lm_head)[:, 0, :]
        ar_logits = lm_head(anchor)
        assert not torch.allclose(par_logits, ar_logits, atol=1e-4)


# ===========================================================================
# 2. Training-target equivalence: K=1 per-block target == AR next-token
# ===========================================================================
class TestTrainingTargetEquivalence:

    @staticmethod
    def _build(all_input_ids, K):
        from nemo.collections.speechlm2.data.streaming_stt_dataset import (
            IGNORE_INDEX as DS_IGNORE,
            StreamingSTTDataset,
        )

        assert DS_IGNORE == IGNORE_INDEX
        stub = SimpleNamespace(
            cfg=SimpleNamespace(parallel_chunk_slots=K, compact_template=True),
            _write_id=100,
            _compact_eos_id=101,
        )
        return StreamingSTTDataset._build_parallel_chunk_targets(stub, all_input_ids)

    def test_k1_anchor_targets_match_ar_next_token(self):
        # [audio, audio, <write>, c0, c1, c2, <eos>]
        ids = [200, 200, 100, 1, 2, 3, 101]
        anchors, targets = self._build([torch.tensor(ids)], K=1)

        # One chunk: write_id at index 2, eos at index 6. Anchors should be the
        # positions whose next token is supervised: write_id .. eos-1 = 2,3,4,5.
        valid = anchors[0] >= 0
        anchor_list = anchors[0][valid].tolist()
        assert anchor_list == [2, 3, 4, 5]

        # K=1 → exactly one target slot per block, equal to the AR next token.
        for blk, a in enumerate(anchor_list):
            slot = targets[0, blk]
            assert slot.numel() == 1
            assert int(slot[0]) == ids[a + 1]  # AR: input[a] predicts input[a+1]

    def test_general_invariant_target_equals_next_k_tokens(self):
        """For ANY K, a supervised (block g, slot s) target equals the token s+1
        positions after the block's anchor (i.e. stream[anchor + s + 1]). For
        s=0 that's precisely the AR next token."""
        ids = [200, 100, 1, 2, 3, 4, 5, 6, 101]  # write@1, content 1..6, eos@8
        for K in (1, 2, 4):
            anchors, targets = self._build([torch.tensor(ids)], K=K)
            B, C, Kdim = targets.shape
            assert Kdim == K
            for c in range(C):
                a = int(anchors[0, c])
                if a < 0:
                    continue
                for s in range(K):
                    tgt = int(targets[0, c, s])
                    if tgt == IGNORE_INDEX:
                        continue
                    pos = a + s + 1
                    assert pos < len(ids)
                    assert tgt == ids[pos], f"K={K} block c={c} slot s={s}"

    def test_multi_chunk_each_chunk_independent(self):
        # Two chunks back to back.
        ids = [100, 1, 2, 101, 100, 3, 101]  # chunk1: c0=1,c1=2 ; chunk2: c0=3
        anchors, targets = self._build([torch.tensor(ids)], K=1)
        valid = anchors[0] >= 0
        anchor_list = anchors[0][valid].tolist()
        # chunk1 anchors: 0,1,2 (write,1,2 → predict 1,2,eos); chunk2: 4,5.
        assert anchor_list == [0, 1, 2, 4, 5]
        for blk, a in enumerate(anchor_list):
            assert int(targets[0, blk, 0]) == ids[a + 1]


# ===========================================================================
# 3. Decode-loop equivalence: K=1 parallel decode == greedy AR loop
# ===========================================================================
class _FakeLLM(torch.nn.Module):
    """Deterministic position-wise stand-in for the LLM.

    hidden = trunk(inputs_embeds); logits = lm_head(hidden). No attention is
    needed: both the parallel decode and the reference AR loop call this same
    module, so they must produce identical token streams iff the parallel
    plumbing (feed + re-anchor + stop) matches AR.
    """

    def __init__(self, trunk: torch.nn.Module, lm_head: torch.nn.Module):
        super().__init__()
        self.trunk = trunk
        self.lm_head = lm_head

    def forward(
        self,
        inputs_embeds=None,
        past_key_values=None,
        attention_mask=None,
        use_cache=True,
        return_dict=True,
        output_hidden_states=False,
        **kwargs,
    ):
        hidden = self.trunk(inputs_embeds)
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits, hidden_states=(hidden,), past_key_values=past_key_values)


class TestDecodeLoopEquivalence:

    def _make_mock(self, H, V, eos_id):
        embed = torch.nn.Embedding(V, H)
        trunk = torch.nn.Linear(H, H)
        lm_head = torch.nn.Linear(H, V, bias=False)
        for m in (embed, trunk, lm_head):
            m.eval()
        fake_llm = _FakeLLM(trunk, lm_head)
        heads = _passthrough_heads(H)
        mock = SimpleNamespace(
            parallel_chunk_heads=heads,
            core_cfg=SimpleNamespace(use_chunk_local_audio_attn=False, use_modality_position_ids=False),
            text_pad_id=0,
            _eos_id=eos_id,
            has_blank=False,
            blank_token_id=-1,
            llm=fake_llm,
            embed_tokens=embed,
        )
        return mock, embed, trunk, lm_head

    @staticmethod
    def _ref_ar_greedy(anchor, embed, trunk, lm_head, eos_id, max_new_tokens):
        """Plain greedy AR over the same fake LLM (batch size 1)."""
        cur = anchor  # (1, H)
        toks = []
        for _ in range(max_new_tokens):
            logits = lm_head(cur)  # (1, V)
            tid = int(logits.argmax(dim=-1)[0])
            if tid == eos_id:
                break
            toks.append(tid)
            emb = embed(torch.tensor([[tid]]))  # (1, 1, H)
            hidden = trunk(emb)  # (1, 1, H)
            cur = hidden[:, -1, :]  # (1, H)
        return toks

    def test_k1_decode_matches_greedy_ar(self):
        from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel

        torch.manual_seed(1234)
        H, V, EOS = 24, 40, 7
        mock, embed, trunk, lm_head = self._make_mock(H, V, EOS)

        anchor_hidden = torch.randn(1, H)
        L0 = 3
        state = SimpleNamespace(
            attention_mask=torch.ones(1, L0, dtype=torch.long),
            cache=None,
            seq_lens=[L0],
        )

        max_new_tokens = 8
        gen, footer = StreamingSTTModel._parallel_chunk_step_decode(
            mock, anchor_hidden, state, max_new_tokens=max_new_tokens
        )
        ref = self._ref_ar_greedy(anchor_hidden, embed, trunk, lm_head, EOS, max_new_tokens)

        assert gen[0] == ref

    def test_k1_decode_stops_on_eos(self):
        """Rig lm_head so the second token is EOS; parallel decode must emit
        exactly one token and report footer_consumed (closer in cache)."""
        from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel

        torch.manual_seed(0)
        H, V, EOS = 8, 5, 4
        embed = torch.nn.Embedding(V, H)
        trunk = torch.nn.Identity()
        lm_head = torch.nn.Linear(H, V, bias=True)

        # Build a deterministic two-step program:
        #   from anchor  -> argmax = token 2
        #   from emb(2)   -> argmax = EOS (=4)
        with torch.no_grad():
            anchor = torch.zeros(1, H)
            anchor[0, 0] = 1.0
            lm_head.weight.zero_()
            lm_head.bias.zero_()
            # anchor (e0=1) → logit for token 2 highest.
            lm_head.weight[2, 0] = 10.0
            # emb(token 2) → make EOS highest. Use embedding row 2's features.
            e2 = embed(torch.tensor(2))  # (H,)
            lm_head.weight[EOS] = e2 * 10.0  # dot(e2, e2)*10 dominates
        for m in (embed, lm_head):
            m.eval()

        fake_llm = _FakeLLM(trunk, lm_head)
        heads = _passthrough_heads(H)
        mock = SimpleNamespace(
            parallel_chunk_heads=heads,
            core_cfg=SimpleNamespace(use_chunk_local_audio_attn=False, use_modality_position_ids=False),
            text_pad_id=0,
            _eos_id=EOS,
            has_blank=False,
            blank_token_id=-1,
            llm=fake_llm,
            embed_tokens=embed,
        )
        state = SimpleNamespace(
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            cache=None,
            seq_lens=[2],
        )

        gen, footer = StreamingSTTModel._parallel_chunk_step_decode(
            mock, anchor, state, max_new_tokens=8
        )
        assert gen[0] == [2]
        assert footer[0] is True
