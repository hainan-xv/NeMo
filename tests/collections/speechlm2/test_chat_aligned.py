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
"""The forced-alignment path CHAT is trained on."""

import pytest
import torch

from nemo.collections.speechlm2.data.chat_dataset import build_path


@pytest.mark.unit
def test_path_layout_is_tokens_then_a_blank_per_chunk():
    BLANK = 99
    t, u, lab = build_path([[5, 6], [], [7]], BLANK)
    assert t == [0, 0, 0, 1, 2, 2]
    assert u == [0, 1, 2, 2, 2, 3]
    assert lab == [5, 6, BLANK, BLANK, 7, BLANK]


@pytest.mark.unit
def test_u_advances_only_on_real_tokens():
    """u indexes the prediction network, which never consumes a blank."""
    BLANK = 0
    chunks = [[1, 2], [], [3], [], [], [4, 5, 6]]
    t, u, lab = build_path(chunks, BLANK)
    emitted = 0
    for ti, ui, li in zip(t, u, lab):
        assert ui == emitted, "u must equal the number of labels emitted so far"
        if li != BLANK:
            emitted += 1
    assert emitted == 6


@pytest.mark.unit
def test_every_chunk_contributes_exactly_one_blank():
    """Silent chunks included -- that is how the model learns to emit nothing."""
    BLANK = 7
    for chunks in ([[1], [2]], [[], [], []], [[1, 2, 3]], []):
        t, u, lab = build_path(chunks, BLANK)
        assert sum(1 for x in lab if x == BLANK) == len(chunks)
        # ...and it is always the chunk's last step.
        for c in range(len(chunks)):
            steps = [i for i, x in enumerate(t) if x == c]
            assert lab[steps[-1]] == BLANK
            assert all(lab[i] != BLANK for i in steps[:-1])


@pytest.mark.unit
def test_path_length_is_tokens_plus_chunks_not_the_lattice():
    """The whole point: cost is O(U + T), not O(T * U)."""
    chunks = [[1, 2], [], [3], [4, 5]]
    U = sum(len(c) for c in chunks)
    T = len(chunks)
    t, u, lab = build_path(chunks, 0)
    assert len(t) == len(u) == len(lab) == U + T
    assert U + T < T * U or T * U == 0


@pytest.mark.unit
def test_chunk_index_is_non_decreasing_and_covers_every_chunk():
    """t must be monotone -- a transducer cannot go back to an earlier chunk."""
    chunks = [[1], [], [2, 3], [], [4]]
    t, _, _ = build_path(chunks, 0)
    assert t == sorted(t)
    assert set(t) == set(range(len(chunks)))


@pytest.mark.unit
def test_prediction_input_row_u_is_the_state_after_u_labels():
    """pred_input[u] must be the token whose output state u_idx=u refers to."""
    chunks = [[11, 12], [], [13]]
    SOS = 0
    pred = [SOS] + [tok for c in chunks for tok in c]
    t, u, lab = build_path(chunks, 99)
    # For every step, u indexes a valid prediction-network position.
    assert max(u) < len(pred), "u must never exceed the prediction network's length"
    # The token emitted at step i is produced FROM state u[i]; after emitting it,
    # the next state is u[i]+1, whose input token is exactly that label.
    for i, (ui, li) in enumerate(zip(u, lab)):
        if li != 99:
            assert pred[ui + 1] == li
