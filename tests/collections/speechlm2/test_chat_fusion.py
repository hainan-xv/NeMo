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
