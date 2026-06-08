# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from nemo.collections.asr.parts.utils.multistream_factorization import (
    ALL_LOWER,
    ALL_UPPER,
    FIRST_UPPER,
    NUM_CAP,
    OTHER,
    combine_ids,
    decode_capitalization,
    encode_capitalization,
    factorize_combined_to_sum,
    multistream_tdt_dividers,
    split_id,
)


class _MockSP:
    """Deterministic SentencePiece-like tokenizer (3-char pieces, ▁ word marker)."""

    def __init__(self):
        self.piece2id = {}
        self.id2piece = []

    def _piece(self, p):
        if p not in self.piece2id:
            self.piece2id[p] = len(self.id2piece)
            self.id2piece.append(p)
        return self.piece2id[p]

    def text_to_ids(self, text):
        ids = []
        for word in text.split(" "):
            if word == "":
                continue
            chunks = [word[i : i + 3] for i in range(0, len(word), 3)] or [""]
            for j, ch in enumerate(chunks):
                ids.append(self._piece(("\u2581" + ch) if j == 0 else ch))
        return ids

    def ids_to_tokens(self, ids):
        return [self.id2piece[i] for i in ids]


class TestCapitalizationCodec:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("hello world", "hello world"),
            ("Hello world", "Hello world"),
            ("NASA launched today", "NASA launched today"),
            ("I am here", "I am here"),
        ],
    )
    def test_roundtrip_exact(self, text, expected):
        tok = _MockSP()
        spell, cap = encode_capitalization(text, tok)
        assert decode_capitalization(spell, cap, tok) == expected

    @pytest.mark.unit
    def test_cap_labels(self):
        tok = _MockSP()
        _, cap = encode_capitalization("Hello", tok)
        assert cap[0] == FIRST_UPPER and all(c == ALL_LOWER for c in cap[1:])

        _, cap = encode_capitalization("NASA", tok)
        assert all(c == ALL_UPPER for c in cap)

        _, cap = encode_capitalization("hello", tok)
        assert all(c == ALL_LOWER for c in cap)

    @pytest.mark.unit
    def test_other_is_lossy_lowercase(self):
        tok = _MockSP()
        spell, cap = encode_capitalization("the McDonald store", tok)
        assert OTHER in cap
        # OTHER falls back to lowercase rendering.
        assert decode_capitalization(spell, cap, tok) == "the mcdonald store"


class TestIntegerFactorization:
    @pytest.mark.unit
    def test_combine_split_roundtrip(self):
        num_spell = 1024
        for spell, cap in [(0, 0), (5, 1), (1023, 3), (700, 2)]:
            combined = combine_ids(spell, cap, num_spell)
            assert split_id(combined, num_spell) == (spell, cap)

    @pytest.mark.unit
    def test_dividers(self):
        num_spell = 1024
        dividers, blank = multistream_tdt_dividers(num_spell, NUM_CAP)
        assert dividers == [0, NUM_CAP, NUM_CAP + num_spell + 1]
        assert blank == NUM_CAP + num_spell

    @pytest.mark.unit
    def test_factorize_to_sum_space(self):
        num_spell = 1024
        combined = torch.tensor([[combine_ids(7, 2, num_spell), combine_ids(3, 0, num_spell)]])
        fac = factorize_combined_to_sum(combined, num_spell, NUM_CAP)
        # [cap_index, spell_index_offset_by_num_cap]
        assert fac.tolist() == [[[2, 7 + NUM_CAP], [0, 3 + NUM_CAP]]]


class TestJointLayoutWiring:
    """The joint output layout [cap | spell | blank | durations] must line up with
    the dividers/blank that MultistreamTDTLoss expects."""

    @pytest.mark.unit
    def test_layout_matches_loss(self):
        from nemo.collections.asr.losses.rnnt_pytorch import MultistreamTDTLoss

        num_spell, num_cap = 10, NUM_CAP
        durations = [0, 1, 2]
        n_dur = len(durations)
        dividers, blank = multistream_tdt_dividers(num_spell, num_cap)

        # joint output dim as the standard NeMo joint would produce it:
        # num_classes (= num_cap + num_spell) + 1 (blank) + num_extra (= n_dur)
        D = (num_cap + num_spell) + 1 + n_dur
        assert dividers[-1] == num_cap + num_spell + 1  # label part width (incl. blank)
        assert blank == num_cap + num_spell == D - n_dur - 1

        B, T, U = 2, 5, 3
        acts = torch.randn(B, T, T + 1, D, dtype=torch.float64, requires_grad=True)
        act_lens = torch.tensor([T, T - 1])
        label_lens = torch.tensor([U, U - 1])

        # product-space targets -> split -> factorized [B, U, 2]
        combined = torch.zeros(B, U, dtype=torch.long)
        for b in range(B):
            for u in range(int(label_lens[b])):
                spell = torch.randint(0, num_spell, (1,)).item()
                cap = torch.randint(0, num_cap, (1,)).item()
                combined[b, u] = combine_ids(spell, cap, num_spell)
        fac = factorize_combined_to_sum(combined, num_spell, num_cap)

        loss = MultistreamTDTLoss(blank=blank, durations=durations, dividers=dividers, reduction='sum')
        out = loss(acts=acts, labels=fac, act_lens=act_lens, label_lens=label_lens)
        out.backward()
        assert torch.isfinite(out)
        assert acts.grad is not None and torch.isfinite(acts.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
