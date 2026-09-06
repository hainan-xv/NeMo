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
"""Forced-alignment path construction for the CHAT transducer."""

import pytest

from nemo.collections.asr.parts.utils.chat_alignment import assign_words_to_chunks, build_forced_path


class TestAssignWordsToChunks:
    @pytest.mark.unit
    def test_word_lands_in_the_chunk_holding_its_last_frame(self):
        # frame = 0.08 s, chunk = 14 frames = 1.12 s.
        # 0.5 s -> frame 7 -> chunk 0;  1.5 s -> frame 19 -> chunk 1.
        assert assign_words_to_chunks([0.5, 1.5], 3, 14, 0.08) == [[0], [1], []]

    @pytest.mark.unit
    def test_delay_pushes_a_word_to_a_later_chunk(self):
        # frame 7 + 8 = 15 -> chunk 1 rather than chunk 0.
        assert assign_words_to_chunks([0.5], 3, 14, 0.08, num_delay_frames=8) == [[], [0], []]

    @pytest.mark.unit
    def test_words_past_the_end_fold_into_the_last_chunk(self):
        """Never dropped: a dropped word is a deletion the model is trained on."""
        assert assign_words_to_chunks([0.5, 99.0], 2, 14, 0.08) == [[0], [1]]

    @pytest.mark.unit
    def test_no_chunks(self):
        assert assign_words_to_chunks([0.5], 0, 14, 0.08) == []


class TestBuildForcedPath:
    BLANK = 99

    @pytest.mark.unit
    def test_every_chunk_ends_with_exactly_one_blank(self):
        t, u, lab = build_forced_path([[1, 2], [], [3]], self.BLANK)
        assert lab == [1, 2, self.BLANK, self.BLANK, 3, self.BLANK]
        assert t == [0, 0, 0, 1, 2, 2]
        # u counts EMITTED LABELS only, so it does not advance on a blank.
        assert u == [0, 1, 2, 2, 2, 3]

    @pytest.mark.unit
    def test_path_length_is_u_plus_t(self):
        chunks = [[1, 2], [], [3, 4, 5], [6]]
        t, _, lab = build_forced_path(chunks, self.BLANK)
        assert len(lab) == sum(len(c) for c in chunks) + len(chunks)
        assert len(t) == len(lab)

    @pytest.mark.unit
    def test_empty_utterance_still_emits_a_blank_per_chunk(self):
        _, _, lab = build_forced_path([[], []], self.BLANK)
        assert lab == [self.BLANK, self.BLANK]

    @pytest.mark.unit
    def test_recovery_is_purely_additive(self):
        """History recovery must never REMOVE a scored step.

        Deleting from the output would train the model to stop early; the whole
        point is that it only ever changes what history a chunk is given.
        """
        chunks = [[1, 2, 3], [4, 5], [6]]
        starts = [[0, 2], [0], [0]]  # word boundaries inside each chunk
        base = build_forced_path(chunks, self.BLANK)[2]
        rec = build_forced_path(chunks, self.BLANK, recover_words=1, word_starts=starts)[2]
        assert len(rec) > len(base)
        # Every original step survives, in order.
        it = iter(rec)
        assert all(any(x == b for x in it) for b in base)

    @pytest.mark.unit
    def test_recovery_rescores_the_previous_chunks_last_word(self):
        # chunk 0 = [1, 2, 3] with words starting at 0 and 2, so its last word is
        # [3]. Chunk 1 should be scored on [3] before its own tokens.
        chunks = [[1, 2, 3], [4]]
        t, u, lab = build_forced_path(chunks, self.BLANK, recover_words=1, word_starts=[[0, 2], [0]])
        assert lab == [1, 2, 3, self.BLANK, 3, 4, self.BLANK]
        assert t == [0, 0, 0, 0, 1, 1, 1]
        # The recovered "3" is scored at u=2 -- the state BEFORE it was emitted --
        # which is exactly the state a decoder reaches after retracting one word.
        assert u == [0, 1, 2, 3, 2, 3, 4]

    @pytest.mark.unit
    def test_recovery_never_reaches_past_the_previous_chunk(self):
        """ "If less than two, we don't go back" -- clamp to that chunk's start."""
        chunks = [[1], [2]]  # chunk 0 holds ONE word, but we ask for two
        _, u, lab = build_forced_path(chunks, self.BLANK, recover_words=2, word_starts=[[0], [0]])
        assert lab == [1, self.BLANK, 1, 2, self.BLANK]
        assert min(u) == 0  # never negative, never into chunk -1

    @pytest.mark.unit
    def test_recovery_skips_a_silent_previous_chunk(self):
        chunks = [[1], [], [2]]
        t, u, lab = build_forced_path(chunks, self.BLANK, recover_words=1, word_starts=[[0], [], [0]])
        # Chunk 1 is silent but its PREDECESSOR is not, so it still recovers "1"
        # and then emits its blank. Chunk 2's predecessor IS empty, so it has
        # nothing to recover and starts straight at its own token.
        assert lab == [1, self.BLANK, 1, self.BLANK, 2, self.BLANK]
        assert t == [0, 0, 1, 1, 2, 2]
        assert u == [0, 1, 0, 1, 1, 2]

    @pytest.mark.unit
    def test_first_chunk_never_recovers(self):
        _, _, lab = build_forced_path([[1]], self.BLANK, recover_words=1, word_starts=[[0]])
        assert lab == [1, self.BLANK]
