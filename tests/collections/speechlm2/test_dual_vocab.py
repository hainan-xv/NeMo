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
"""Dual-vocabulary SCRIPT: the parts that fail SILENTLY if they are wrong."""

import pytest
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# The chunk split
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_chunk_split_rejects_a_chunk_that_is_not_trained():
    """A typo'd chunk must fail at startup, not route to a vocabulary silently."""
    from nemo.collections.speechlm2.parts.dual_vocab import validate_chunk_split

    with pytest.raises(ValueError, match="not in chunk_size"):
        validate_chunk_split([2, 7, 14], [2, 8])


@pytest.mark.unit
def test_chunk_split_rejects_covering_every_chunk():
    """If the small vocabulary owns every chunk, the LARGE table never trains.

    DDP would then average a 311M-parameter table that no rank ever used, and
    the arm would look like a single-vocab run with 311M dead parameters.
    """
    from nemo.collections.speechlm2.parts.dual_vocab import validate_chunk_split

    with pytest.raises(ValueError, match="single-vocab"):
        validate_chunk_split([2, 7], [2, 7])


@pytest.mark.unit
def test_chunk_split_returns_the_large_half():
    from nemo.collections.speechlm2.parts.dual_vocab import validate_chunk_split

    assert validate_chunk_split([2, 7, 14, 28, 42], [2, 7]) == [14, 28, 42]


# ---------------------------------------------------------------------------
# Special ids
# ---------------------------------------------------------------------------


class _Tok:
    """Minimal stand-in exposing what resolve_special_ids reads."""

    def __init__(self, table, eos=9, unk=0, pad=None):
        self._t, self.eos_token_id, self.unk_token_id, self.pad_token_id = table, eos, unk, pad

    def convert_tokens_to_ids(self, tok):
        return self._t.get(tok)


@pytest.mark.unit
def test_special_ids_reject_a_vocabulary_missing_the_audio_delimiters():
    """Returning `unk` here would delimit every audio span with the SAME id as
    any other unknown token, so the layout would be unparseable -- and nothing
    downstream checks it."""
    from nemo.collections.speechlm2.parts.dual_vocab import resolve_special_ids

    with pytest.raises(ValueError, match="no id for"):
        resolve_special_ids(_Tok({"<vs>": 0, "<ve>": 5}), "<vs>", "<ve>")  # 0 == unk
    with pytest.raises(ValueError, match="no id for"):
        resolve_special_ids(_Tok({"<ve>": 5}), "<vs>", "<ve>")  # missing entirely


@pytest.mark.unit
def test_special_ids_fall_back_to_eos_for_pad():
    from nemo.collections.speechlm2.parts.dual_vocab import resolve_special_ids

    ids = resolve_special_ids(_Tok({"<vs>": 3, "<ve>": 4}, eos=9, pad=None), "<vs>", "<ve>")
    assert ids == {"vision_start_id": 3, "vision_end_id": 4, "eot_id": 9, "pad_id": 9}


# ---------------------------------------------------------------------------
# Splitting the dataset config -- the band list must follow the chunk list
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_band_words_by_chunk_is_sliced_by_POSITION_not_truncated():
    """band_words_by_chunk is PARALLEL to chunk_size.

    Splitting [2,7,14,28,42] with bands [3,1,1,1,1] must give the small half
    [3,1] and the large half [1,1,1]. Truncating instead of indexing would pair
    the wide band with chunk 14 and nothing would notice: the lengths still
    match, so the dataset's own validation passes and the wrong chunk simply
    trains with the wrong band.
    """
    from nemo.collections.speechlm2.parts.dual_vocab import split_dataset_cfg as _split_dataset_cfg

    base = OmegaConf.create({"chunk_size": [2, 7, 14, 28, 42], "band_words_by_chunk": [3, 1, 1, 1, 1]})
    small = _split_dataset_cfg(base, [2, 7], [2, 7, 14, 28, 42])
    large = _split_dataset_cfg(base, [14, 28, 42], [2, 7, 14, 28, 42])
    assert list(small.chunk_size) == [2, 7] and list(small.band_words_by_chunk) == [3, 1]
    assert list(large.chunk_size) == [14, 28, 42] and list(large.band_words_by_chunk) == [1, 1, 1]


@pytest.mark.unit
def test_split_preserves_a_non_parallel_band_absence():
    from nemo.collections.speechlm2.parts.dual_vocab import split_dataset_cfg as _split_dataset_cfg

    base = OmegaConf.create({"chunk_size": [2, 14], "band_words": 1})
    out = _split_dataset_cfg(base, [14], [2, 14])
    assert list(out.chunk_size) == [14]
    assert out.get("band_words_by_chunk", None) in (None, [])


@pytest.mark.unit
def test_split_rejects_a_band_list_of_the_wrong_length():
    from nemo.collections.speechlm2.parts.dual_vocab import split_dataset_cfg as _split_dataset_cfg

    base = OmegaConf.create({"chunk_size": [2, 7, 14], "band_words_by_chunk": [3, 1]})
    with pytest.raises(ValueError, match="entries but chunk_size"):
        _split_dataset_cfg(base, [2], [2, 7, 14])
