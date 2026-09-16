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
"""Fusing CHAT into SCRIPT's decode loop, in SCRIPT's index space.

The dangerous property here is that a mistake is SILENT: the loop still runs,
still emits text, and only the WER moves. So these pin the two claims the method
rests on -- that lam=0 leaves the production decode untouched, and that CHAT's
blank lands on SCRIPT's <|im_end|> rather than on some text token.
"""

import pytest
import torch

from nemo.collections.speechlm2.parts.chat_fusion import NEG_INF, fuse_into_script_logits

V = 8  # CHAT text vocab; CHAT vectors are V+1 with blank last
LLM = 12  # SCRIPT's LM head is wider (Qwen pads its embedding rows)
EOT = 5  # inside the text range, like the real 151645 < 151669
VS, VE = 6, 7  # audio delimiters, also in-vocab


def _chat(vals):
    return torch.tensor([vals], dtype=torch.float32).log_softmax(-1)


def test_lam_zero_is_script_alone_up_to_a_constant():
    """The production path must be recoverable exactly.

    argmax is invariant to a shared additive constant, so the test compares the
    ARGMAX and the differences, not raw values.
    """
    torch.manual_seed(0)
    logits = torch.randn(1, LLM)
    chat_lp = _chat([0.0] * (V + 1))
    out = fuse_into_script_logits(logits, chat_lp, 0.0, EOT, ())
    ref = logits.float().log_softmax(-1)
    # only the columns fusion defines are comparable (0..V-1 plus eot)
    cols = [i for i in range(V)]
    assert out[0, cols].argmax().item() == ref[0, cols].argmax().item()
    assert torch.allclose(out[0, :V], ref[0, :V], atol=1e-5)


def test_chat_blank_lands_on_eot_not_on_a_text_token():
    """CHAT's blank is at index V; SCRIPT's end-of-chunk is eot_id < V.

    If the blank were left at V the loop would never see it (V is past nothing
    useful in SCRIPT's space) and chunks would only end when SCRIPT alone said
    so -- silently discarding half the ensemble's opinion about boundaries.
    """
    logits = torch.zeros(1, LLM)
    # CHAT is certain the chunk is over.
    chat_lp = torch.full((1, V + 1), -20.0)
    chat_lp[0, V] = 0.0
    out = fuse_into_script_logits(logits, chat_lp, 1.0, EOT, ())
    assert out[0].argmax().item() == EOT, "blank must win at eot_id when CHAT is certain"


def test_eot_column_is_not_left_as_chat_text_score():
    """eot_id < V, so the text-range write touches that column first and must be
    overwritten by the END combination. Getting the order wrong is invisible."""
    logits = torch.zeros(1, LLM)
    chat_lp = torch.full((1, V + 1), -20.0)
    chat_lp[0, EOT] = -1.0  # CHAT's score for eot AS A TEXT TOKEN
    chat_lp[0, V] = 0.0  # CHAT's score for BLANK
    out = fuse_into_script_logits(logits, chat_lp, 1.0, EOT, ())
    s_lp = logits.float().log_softmax(-1)
    expected = 1.0 * chat_lp[0, V] + 0.0 * s_lp[0, EOT]
    assert out[0, EOT].item() == pytest.approx(expected.item(), abs=1e-5)


def test_audio_delimiters_are_vetoed():
    logits = torch.zeros(1, LLM)
    logits[0, VS] = 50.0  # try hard to make a delimiter win
    chat_lp = _chat([0.0] * (V + 1))
    out = fuse_into_script_logits(logits, chat_lp, 0.5, EOT, (VS, VE))
    assert out[0, VS].item() <= NEG_INF and out[0, VE].item() <= NEG_INF
    assert out[0].argmax().item() not in (VS, VE)


def test_untrained_lm_tail_past_the_text_vocab_is_unreachable():
    """Qwen3's embedding is wider than the tokenizer; those rows never trained."""
    logits = torch.zeros(1, LLM)
    logits[0, LLM - 1] = 99.0
    out = fuse_into_script_logits(logits, _chat([0.0] * (V + 1)), 0.5, EOT, ())
    assert out[0, LLM - 1].item() <= NEG_INF
    assert out[0].argmax().item() < V


def test_both_sides_are_log_probs_so_lam_means_something():
    """Mixing a raw logit with a log-prob would make lam scale-dependent.

    Adding a constant to the SCRIPT logits must not change the fused result,
    which is only true if they are normalised first.
    """
    torch.manual_seed(1)
    logits = torch.randn(1, LLM)
    chat_lp = torch.randn(1, V + 1).log_softmax(-1)
    a = fuse_into_script_logits(logits, chat_lp, 0.5, EOT, ())
    b = fuse_into_script_logits(logits + 7.5, chat_lp, 0.5, EOT, ())
    assert torch.allclose(a[0, :V], b[0, :V], atol=1e-5)


def test_rejects_a_chat_vocab_wider_than_the_lm():
    with pytest.raises(ValueError, match="exceeds SCRIPT LM width"):
        fuse_into_script_logits(torch.zeros(1, 4), torch.zeros(1, 10), 0.5, 1, ())


def test_batched_streams_are_fused_independently():
    """The loop fuses all active streams at once; rows must not bleed."""
    logits = torch.zeros(2, LLM)
    chat_lp = torch.full((2, V + 1), -20.0)
    chat_lp[0, 1] = 0.0  # stream 0 -> token 1
    chat_lp[1, 3] = 0.0  # stream 1 -> token 3
    out = fuse_into_script_logits(logits, chat_lp, 1.0, EOT, ())
    assert out[0].argmax().item() == 1
    assert out[1].argmax().item() == 3


# --------------------------------------------------------------------------
# Confidence gating: hand a step to CHAT outright when it is sure.
# --------------------------------------------------------------------------


SCRIPT_FAVOURITE = 3  # the token SCRIPT is made to want in these tests


def _chat_dist(top_tok, margin, floor=-6.0, runner_up=1):
    """CHAT log-probs: ``top_tok`` on top, beating the runner-up by ``margin``.

    ``floor`` is deliberately MILD (-6, not -30). A very negative floor makes
    CHAT veto every token it did not pick, so SCRIPT can never pull a step at
    lam=0.5 no matter how unsure CHAT is -- which made an earlier version of
    these tests assert an outcome the arithmetic could not produce.
    """
    assert runner_up not in (top_tok, SCRIPT_FAVOURITE)
    lp = torch.full((1, V + 1), floor)
    lp[0, top_tok] = 0.0
    lp[0, runner_up] = -margin
    return lp


def _script_prefers(tok, strength=20.0, n=1):
    lg = torch.zeros(n, LLM)
    lg[:, tok] = strength
    return lg


def test_threshold_zero_reproduces_chat_alone():
    """tau=0 gates EVERY step, so it must equal lam=1 decoding exactly.

    This is one end of the sweep; without it a threshold curve cannot be
    anchored.
    """
    logits = _script_prefers(SCRIPT_FAVOURITE)
    chat_lp = _chat_dist(2, 3.0)
    gated = fuse_into_script_logits(logits, chat_lp, 0.5, EOT, (), margin_threshold=0.0)
    chat_only = fuse_into_script_logits(logits, chat_lp, 1.0, EOT, ())
    assert gated[0].argmax().item() == chat_only[0].argmax().item() == 2


def test_threshold_infinity_is_plain_fusion():
    """The other end: no step gated, identical to ungated fusion."""
    torch.manual_seed(3)
    logits = torch.randn(1, LLM)
    chat_lp = torch.randn(1, V + 1).log_softmax(-1)
    a = fuse_into_script_logits(logits, chat_lp, 0.5, EOT, ())
    b = fuse_into_script_logits(logits, chat_lp, 0.5, EOT, (), margin_threshold=float("inf"))
    assert torch.allclose(a, b, atol=1e-6)


def test_gate_hands_confident_steps_to_chat():
    """With the gate on, a confident CHAT keeps the step it would otherwise lose.

    Ungated the step goes to SCRIPT: fused[tok3] = .5*(-6) + .5*(0) = -3 beats
    fused[tok2] = .5*(0) + .5*(-20) = -10. Gating on a margin above tau replaces
    the row with CHAT's own, so token 2 wins instead.
    """
    logits = _script_prefers(SCRIPT_FAVOURITE)
    sure = _chat_dist(2, 5.0)
    assert fuse_into_script_logits(logits, sure, 0.5, EOT, ())[0].argmax().item() == SCRIPT_FAVOURITE
    gated = fuse_into_script_logits(logits, sure, 0.5, EOT, (), margin_threshold=1.0)
    assert gated[0].argmax().item() == 2, "confident CHAT must keep the step"


def test_gate_leaves_unsure_steps_to_the_ensemble():
    logits = _script_prefers(SCRIPT_FAVOURITE)
    unsure = _chat_dist(2, 0.1)
    out = fuse_into_script_logits(logits, unsure, 0.5, EOT, (), margin_threshold=1.0)
    assert out[0].argmax().item() == SCRIPT_FAVOURITE, "unsure CHAT must let SCRIPT pull"


def test_gate_is_per_row_not_per_batch():
    """Streams decode together; one confident row must not gate another."""
    logits = _script_prefers(SCRIPT_FAVOURITE, n=2)
    chat_lp = torch.cat([_chat_dist(2, 5.0), _chat_dist(2, 0.1)], dim=0)
    out = fuse_into_script_logits(logits, chat_lp, 0.5, EOT, (), margin_threshold=1.0)
    assert out[0].argmax().item() == 2, "row 0 confident -> CHAT"
    assert out[1].argmax().item() == SCRIPT_FAVOURITE, "row 1 unsure -> fused"


def test_stats_count_overrides_by_confidence_bucket():
    """Row 0 disagrees with SCRIPT (override); row 1 AGREES with it (no override).

    Agreement is used for the no-override row rather than a large margin,
    because margin alone does not prevent an override -- a very confident SCRIPT
    can outvote a confident CHAT. That is a real property of the method, not
    something the test should paper over.
    """
    from nemo.collections.speechlm2.parts.chat_fusion import FusionStats

    st = FusionStats()
    logits = _script_prefers(SCRIPT_FAVOURITE, n=2)
    chat_lp = torch.cat(
        [
            _chat_dist(2, 0.1),                  # unsure, disagrees -> overridden
            _chat_dist(SCRIPT_FAVOURITE, 9.0),   # certain, AGREES   -> not overridden
        ],
        dim=0,
    )
    fuse_into_script_logits(logits, chat_lp, 0.5, EOT, (), stats=st)

    assert st.steps == 2
    assert st.overrides == 1, "only the disagreeing row counts as an override"
    assert st.bucket_overrides[st._bucket(0.1)] == 1
    assert st.bucket_overrides[st._bucket(9.0)] == 0
    assert "CHAT margin" in st.report()


def test_stats_do_not_change_the_decision():
    """Instrumentation must be observation only."""
    from nemo.collections.speechlm2.parts.chat_fusion import FusionStats

    torch.manual_seed(2)
    logits = torch.randn(3, LLM)
    chat_lp = torch.randn(3, V + 1).log_softmax(-1)
    a = fuse_into_script_logits(logits, chat_lp, 0.5, EOT, ())
    b = fuse_into_script_logits(logits, chat_lp, 0.5, EOT, (), stats=FusionStats())
    assert torch.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------------------
# On-demand fusion: decode a chunk with CHAT alone, escalate only if unsure.
# --------------------------------------------------------------------------


class _StubScorer:
    """Scripted CHAT: table[chunk][position] -> (top_token, margin)."""

    def __init__(self, table, vocab_size=V):
        self.table = table
        self.vocab_size = vocab_size
        self.calls = 0

    def logprobs(self, b_idx, chunk_idx, prefixes):
        self.calls += 1
        rows = []
        for p in prefixes:
            tok, margin = self.table[chunk_idx][len(p)]
            lp = torch.full((self.vocab_size + 1,), -20.0)
            lp[tok] = 0.0
            # runner-up sits `margin` below the top
            lp[(tok + 1) % self.vocab_size if tok != (tok + 1) % self.vocab_size else 0] = -margin
            rows.append(lp)
        return torch.stack(rows)


def test_chat_only_chunk_returns_tokens_and_the_weakest_margin():
    from nemo.collections.speechlm2.parts.chat_fusion import chat_only_chunk

    END = V
    # two tokens then END; margins 5.0 then 0.3 -> weakest is 0.3
    tbl = {0: {0: (2, 5.0), 1: (3, 0.3), 2: (END, 9.0)}}
    sc = _StubScorer(tbl)
    toks, worst = chat_only_chunk(sc, [0], 0, [[]], margin_threshold=2.0)
    assert toks == [[2, 3]]
    assert worst[0] == pytest.approx(0.3, abs=1e-6), "must report the MINIMUM margin, not the last or mean"


def test_one_unsure_step_taints_the_whole_chunk():
    """An early uncertain token changes every token after it, so the chunk is
    escalated as a unit rather than per position."""
    from nemo.collections.speechlm2.parts.chat_fusion import chat_only_chunk

    END = V
    tbl = {0: {0: (2, 0.1), 1: (3, 9.0), 2: (END, 9.0)}}  # only the FIRST step is unsure
    toks, worst = chat_only_chunk(_StubScorer(tbl), [0], 0, [[]], margin_threshold=2.0)
    assert worst[0] < 2.0, "the chunk must be flagged even though later steps were certain"


def test_chunk_ends_at_the_end_slot_without_emitting_it():
    from nemo.collections.speechlm2.parts.chat_fusion import chat_only_chunk

    END = V
    tbl = {0: {0: (END, 9.0)}}
    toks, worst = chat_only_chunk(_StubScorer(tbl), [0], 0, [[]], margin_threshold=2.0)
    assert toks == [[]] and V not in toks[0]


def test_streams_are_tracked_independently():
    """Batched decoding: one stream ending must not truncate another."""
    from nemo.collections.speechlm2.parts.chat_fusion import chat_only_chunk

    END = V

    class _TwoStream(_StubScorer):
        def logprobs(self, b_idx, chunk_idx, prefixes):
            rows = []
            for bi, p in zip(b_idx, prefixes):
                # stream 0 stops immediately; stream 1 emits two tokens
                tok, margin = (END, 9.0) if bi == 0 else ([4, 5, END][len(p)], 9.0)
                lp = torch.full((V + 1,), -20.0)
                lp[tok] = 0.0
                lp[0 if tok != 0 else 1] = -margin
                rows.append(lp)
            return torch.stack(rows)

    toks, worst = chat_only_chunk(_TwoStream({}), [0, 1], 0, [[], []], margin_threshold=2.0)
    assert toks[0] == [] and toks[1] == [4, 5]


def test_max_new_tokens_bounds_a_chunk_that_never_ends():
    from nemo.collections.speechlm2.parts.chat_fusion import chat_only_chunk

    tbl = {0: {i: (2, 9.0) for i in range(50)}}
    toks, _ = chat_only_chunk(_StubScorer(tbl), [0], 0, [[]], max_new_tokens=4, margin_threshold=2.0)
    assert toks == [[2, 2, 2, 2]]
