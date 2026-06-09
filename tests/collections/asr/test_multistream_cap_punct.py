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

from nemo.collections.asr.parts.utils.multistream_cap_punct_factorization import (
    DEFAULT_PUNCT_MARKS,
    PUNCT_NONE,
    cap_punct_dividers,
    combine_ids_cap_punct,
    decode_cap_punct,
    encode_cap_punct,
    split_id_cap_punct,
    split_word_punct,
)
from nemo.collections.asr.parts.utils.multistream_factorization import NUM_CAP


class _MockSP:
    """Deterministic SentencePiece-like tokenizer (3-char pieces, U+2581 word marker)."""

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


class TestSplitWordPunct:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "word,core,punct_class,discarded",
        [
            ("Hello,", "Hello", 1, False),  # "," -> class 1
            ("world.", "world", 2, False),  # "." -> class 2
            ("wait...", "wait", 2, True),  # keep first ".", discard ".."
            ("really?!", "really", 3, True),  # keep "?", discard "!"
            ("end).", "end)", 2, False),  # ")" kept (before "."), "." is the punct
            ("end.)", "end", 2, True),  # "." is the punct, ")" after it -> discarded
            ("100%", "100%", PUNCT_NONE, False),  # "%" not in set -> normal token
            ("well-known", "well-known", PUNCT_NONE, False),  # within-word hyphen -> normal
            (".", "", 2, False),  # standalone period
            ("?!", "", 3, True),  # standalone, keep "?", discard "!"
            (")", ")", PUNCT_NONE, False),  # non-set punct -> normal token (not standalone)
        ],
    )
    def test_split(self, word, core, punct_class, discarded):
        c, p, d = split_word_punct(word, DEFAULT_PUNCT_MARKS)
        assert (c, p, d) == (core, punct_class, discarded)


class TestCapPunctCodec:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Hello, world.", "Hello, world."),
            ("hello world", "hello world"),
            ("Is this real?", "Is this real?"),
            ("Wait; stop: now!", "Wait; stop: now!"),
            ("100% done", "100% done"),  # non-set punct stays as a normal token
            ("well-known fact", "well-known fact"),  # within-word hyphen stays normal
            ("end). go", "end). go"),  # ")" normal, "." punct
            ("Hello .", "Hello."),  # standalone punct attaches to previous word
        ],
    )
    def test_roundtrip_exact(self, text, expected):
        tok = _MockSP()
        spell, cap, punct = encode_cap_punct(text, tok, DEFAULT_PUNCT_MARKS)
        assert decode_cap_punct(spell, cap, punct, tok, DEFAULT_PUNCT_MARKS) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("wait...", "wait."),  # extra trailing punct discarded (lossy)
            ("really?!", "really?"),
            ("end.) now", "end. now"),  # ")" after the "." is discarded
        ],
    )
    def test_roundtrip_lossy_keeps_first(self, text, expected):
        tok = _MockSP()
        spell, cap, punct = encode_cap_punct(text, tok, DEFAULT_PUNCT_MARKS)
        assert decode_cap_punct(spell, cap, punct, tok, DEFAULT_PUNCT_MARKS) == expected

    @pytest.mark.unit
    def test_punct_attached_to_last_subword(self):
        tok = _MockSP()
        # "Hello," -> pieces [U+2581hel, lo]; punct must land on the LAST piece ("lo").
        spell, cap, punct = encode_cap_punct("Hello, there", tok, DEFAULT_PUNCT_MARKS)
        assert punct[0] == PUNCT_NONE  # "hel"
        assert punct[1] == 1  # "lo" carries the comma
        assert all(p == PUNCT_NONE for p in punct[2:])


class TestIntegerFactorization:
    @pytest.mark.unit
    def test_combine_split_roundtrip(self):
        num_spell, num_cap = 1024, NUM_CAP
        for spell, cap, punct in [(0, 0, 0), (5, 1, 2), (1023, 3, 6), (700, 2, 1)]:
            combined = combine_ids_cap_punct(spell, cap, punct, num_spell, num_cap)
            assert split_id_cap_punct(combined, num_spell, num_cap) == (spell, cap, punct)

    @pytest.mark.unit
    def test_dividers(self):
        num_spell, num_cap = 1024, NUM_CAP
        num_punct = len(DEFAULT_PUNCT_MARKS) + 1
        dividers, blank = cap_punct_dividers(num_spell, num_cap, num_punct)
        assert dividers == [0, num_punct, num_punct + num_cap, num_punct + num_cap + num_spell + 1]
        assert blank == num_punct + num_cap + num_spell


class TestJointLayoutWiring:
    """[punct | cap | spell | blank | durations] must line up with the loss dividers/blank."""

    @pytest.mark.unit
    def test_layout_matches_loss(self):
        from nemo.collections.asr.losses.rnnt_pytorch import MultistreamTDTLoss

        num_spell, num_cap, num_punct = 10, NUM_CAP, len(DEFAULT_PUNCT_MARKS) + 1
        durations = [0, 1, 2]
        n_dur = len(durations)
        dividers, blank = cap_punct_dividers(num_spell, num_cap, num_punct)

        D = (num_punct + num_cap + num_spell) + 1 + n_dur
        assert dividers[-1] == num_punct + num_cap + num_spell + 1
        assert blank == num_punct + num_cap + num_spell == D - n_dur - 1

        B, T, U = 2, 5, 3
        acts = torch.randn(B, T, T + 1, D, dtype=torch.float64, requires_grad=True)
        act_lens = torch.tensor([T, T - 1])
        label_lens = torch.tensor([U, U - 1])

        combined = torch.zeros(B, U, dtype=torch.long)
        for b in range(B):
            for u in range(int(label_lens[b])):
                spell = torch.randint(0, num_spell, (1,)).item()
                cap = torch.randint(0, num_cap, (1,)).item()
                punct = torch.randint(0, num_punct, (1,)).item()
                combined[b, u] = combine_ids_cap_punct(spell, cap, punct, num_spell, num_cap)
        spell = combined % num_spell
        rest = combined // num_spell
        cap = rest % num_cap
        punct = rest // num_cap
        fac = torch.stack([punct, cap + num_punct, spell + num_punct + num_cap], dim=-1)

        loss = MultistreamTDTLoss(blank=blank, durations=durations, dividers=dividers, reduction='sum')
        out = loss(acts=acts, labels=fac, act_lens=act_lens, label_lens=label_lens)
        out.backward()
        assert torch.isfinite(out)
        assert acts.grad is not None and torch.isfinite(acts.grad).all()


# --------------------------------------------------------------------------- #
# Batched decoder equivalence: with num_punct == 1 (only PUNCT_NONE), the 3-stream
# label-looping decoder must reproduce the (already-validated) 2-stream decoder for
# identical joint outputs, since combined = (0*num_cap + cap)*num_spell + spell.
# --------------------------------------------------------------------------- #
class _FakeDecoder:
    """Minimal prediction-net stub. State is the [B] tensor of last spelling labels."""

    def initialize_state(self, x):
        return torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)

    def predict(self, labels, state, add_sos, batch_size):
        lab = labels.squeeze(1).long()
        dec_out = lab.to(torch.float32).view(batch_size, 1, 1)  # carries the label for the fake joint
        return dec_out, lab.clone()

    def batch_replace_states_mask(self, src_states, dst_states, mask):
        dst_states[mask] = src_states[mask]


class _FakeJoint:
    """Fake joint reading (time, last-spell-label) and emitting logits from a fixed table.

    ``extra_punct`` prepends a single (constant) punctuation column so the 3-stream layout
    ``[punct(1) | cap | spell | blank | dur]`` shares cap/spell/blank/dur with the 2-stream table.
    """

    def __init__(self, table, num_durations, extra_punct):
        self.table = table  # [T, num_spell+1, W2] where W2 = num_cap+num_spell+1+n_dur
        self.num_durations = num_durations
        self.extra_punct = extra_punct

    def project_encoder(self, x):
        return x

    def project_prednet(self, x):
        return x

    def joint_after_projection(self, enc_step, dec_out):
        t = enc_step[:, 0, 0].long()
        lab = dec_out[:, 0, 0].long().clamp_min(0)
        logits = self.table[t, lab]  # [B, W2]
        if self.extra_punct:
            pad = torch.zeros(logits.shape[0], 1, dtype=logits.dtype, device=logits.device)
            logits = torch.cat([pad, logits], dim=-1)
        return logits.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, W]


class TestBatchedDecoderEquivalence:
    @pytest.mark.unit
    def test_matches_two_stream_when_num_punct_is_one(self):
        from nemo.collections.asr.parts.submodules.transducer_decoding.multistream_cap_punct_tdt_label_looping import (  # noqa: E501
            GreedyBatchedMultiStreamCapPunctTDTLabelLoopingComputer,
        )
        from nemo.collections.asr.parts.submodules.transducer_decoding.multistream_tdt_label_looping import (
            GreedyBatchedMultiStreamTDTLabelLoopingComputer,
        )

        torch.manual_seed(0)
        num_cap, num_spell = 3, 5
        durations = [0, 1, 2, 3]
        n_dur = len(durations)
        W2 = num_cap + num_spell + 1 + n_dur
        T = 12
        table = torch.randn(T, num_spell + 1, W2)

        B = 4
        D = 1
        encoder_output = torch.arange(T, dtype=torch.float32).view(1, T, D).repeat(B, 1, 1)
        out_len = torch.tensor([T, T - 1, T - 5, 1], dtype=torch.long)

        two = GreedyBatchedMultiStreamTDTLabelLoopingComputer(
            decoder=_FakeDecoder(),
            joint=_FakeJoint(table, n_dur, extra_punct=False),
            blank_index=num_cap + num_spell,
            durations=durations,
            num_cap=num_cap,
            num_spell=num_spell,
            max_symbols_per_step=10,
        )
        three = GreedyBatchedMultiStreamCapPunctTDTLabelLoopingComputer(
            decoder=_FakeDecoder(),
            joint=_FakeJoint(table, n_dur, extra_punct=True),
            blank_index=1 + num_cap + num_spell,
            durations=durations,
            num_punct=1,
            num_cap=num_cap,
            num_spell=num_spell,
            max_symbols_per_step=10,
        )

        hyps2, _, _ = two(x=encoder_output.clone(), out_len=out_len.clone())
        hyps3, _, _ = three(x=encoder_output.clone(), out_len=out_len.clone())

        assert torch.equal(hyps2.current_lengths, hyps3.current_lengths)
        for b in range(B):
            n = int(hyps2.current_lengths[b])
            assert torch.equal(hyps2.transcript[b, :n], hyps3.transcript[b, :n])
            assert torch.equal(hyps2.timestamps[b, :n], hyps3.timestamps[b, :n])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
