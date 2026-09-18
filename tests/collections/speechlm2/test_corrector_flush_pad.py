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
"""Training hypotheses need trailing audio, or CHAT never flushes its tail.

CHAT's emission lags its audio, so an unpadded utterance loses its final words.
Measured at eval: librispeech test-other scored 5.57 at pad 0.0 and 4.00 at pad
0.5 for the SAME model. Training on unpadded hypotheses teaches the corrector to
append missing trailing words -- a task that vanishes once the pad is there, and
which made the corrector look like it was improving when it was learning to
repair a harness artefact.

The extra chunk the pad introduces is safe specifically because labels live in
HYPOTHESIS-chunk space: label_chunks assigns reference words by alignment, so a
word flushed late still owns the reference word it matches.
"""

import inspect

import pytest

from nemo.collections.asr.parts.utils.chunk_error_labels import label_chunks, simple_normalize
from nemo.collections.speechlm2.models.script_corrector_model import ScriptCorrectorModel


@pytest.mark.unit
def test_prepare_pads_the_audio_before_encoding():
    src = inspect.getsource(ScriptCorrectorModel._prepare)
    assert "chat_flush_seconds" in src
    i_pad, i_enc = src.index("n_pad"), src.index("self.chat.encoder(")
    assert i_pad < i_enc, "the pad must be applied BEFORE encoding, or it changes nothing"


@pytest.mark.unit
def test_pad_is_applied_to_both_signal_and_length():
    """Padding the tensor without extending `length` leaves the extra frames
    masked out, which silently does nothing."""
    src = inspect.getsource(ScriptCorrectorModel._prepare)
    assert "sig = torch.nn.functional.pad" in src
    assert "sig_len = sig_len + n_pad" in src


@pytest.mark.unit
def test_a_late_flushed_word_still_owns_its_reference_word():
    """THE property that makes the extra trailing chunk harmless. The hypothesis
    emits 'dog' one chunk late, into a chunk the reference partition leaves
    empty; it must still be ACCEPTED, not marked a spurious insertion."""
    ref = [["the", "quick"], ["brown", "dog"], []]
    hyp = [["the", "quick"], ["brown"], ["dog"]]
    labels, n_wrong, owned = label_chunks(hyp, ref, normalize=simple_normalize)
    assert n_wrong == 0, f"a late flush must not be an error: {labels}"
    assert owned[2] == ["dog"], "the flushed word owns the reference word it matches"
    assert [w for c in owned for w in c] == ["the", "quick", "brown", "dog"]


@pytest.mark.unit
def test_a_truncated_tail_IS_still_an_error():
    """The complement: if the tail is genuinely missing (what unpadded training
    produced), the labeller must still flag it -- otherwise the pad would be
    hiding real errors rather than preventing fake ones."""
    ref = [["the", "quick"], ["brown", "dog"]]
    hyp = [["the", "quick"], ["brown"]]
    labels, n_wrong, owned = label_chunks(hyp, ref, normalize=simple_normalize)
    assert n_wrong == 1
    assert "dog" in labels[1]
