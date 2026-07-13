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

import math

import torch
from lhotse.dataset.collation import collate_vectors

from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class LhotseSpeechToTextChatCEDataset(LhotseSpeechToTextBpeDataset):
    """
    Lhotse dataset for the alignment-guided cross-entropy (CE) CHAT model.

    In addition to the standard ASR batch, it returns a per-token chunk-index tensor
    ``token_chunk_idx`` [B, U] that assigns every target token to the encoder chunk
    containing its word's ENDING timestamp (+ optional ``num_delay_frames``). All
    subword tokens of a word share the word's ending chunk.

    Word-level alignments are read from ``cut.custom["alignments"]`` (Granary v2
    pre-aligned format: a list of ``{text, start_time, end_time}``).

    NOTE: tokens are produced by tokenizing each aligned word individually and
    concatenating them, so ``tokens`` and ``token_chunk_idx`` are aligned by
    construction. This can differ slightly from whole-utterance tokenization.

    Args:
        tokenizer: the ASR sub-word tokenizer.
        chunk_size: CHAT chunk size in encoder frames (must match model.joint.chunk_size).
        frame_length_in_secs: seconds per encoder frame = window_stride * subsampling_factor
            (e.g. 0.01 * 8 = 0.08).
        num_delay_frames: number of encoder frames to delay word emission after its end.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        chunk_size: int,
        frame_length_in_secs: float = 0.08,
        num_delay_frames: int = 0,
    ):
        super().__init__(tokenizer=tokenizer, return_cuts=False)
        if chunk_size is None or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive int, got {chunk_size}")
        self.chunk_size = int(chunk_size)
        self.frame_length_in_secs = float(frame_length_in_secs)
        self.num_delay_frames = int(num_delay_frames)

    def __getitem__(self, cuts):
        audio, audio_lens, cuts = self.load_audio(cuts)

        tokens_list, chunk_idx_list = [], []
        for c in cuts:
            lang = c.supervisions[0].language if c.supervisions else None
            custom = c.custom or {}
            alignments = custom.get("alignments", []) or []

            toks, cidx = [], []
            for word in alignments:
                text = word["text"]
                end_time = word["end_time"]
                word_tokens = self.tokenizer(text or "", lang)
                if word_tokens is None or len(word_tokens) == 0:
                    continue
                # word-ending timestamp -> encoder frame -> chunk index
                end_frame = math.ceil(end_time / self.frame_length_in_secs) + self.num_delay_frames
                chunk = end_frame // self.chunk_size
                toks.extend(int(t) for t in word_tokens)
                cidx.extend([chunk] * len(word_tokens))

            tokens_list.append(torch.tensor(toks, dtype=torch.long))
            chunk_idx_list.append(torch.tensor(cidx, dtype=torch.long))

        token_lens = torch.tensor([t.size(0) for t in tokens_list], dtype=torch.long)
        tokens = collate_vectors(tokens_list, padding_value=0)
        token_chunk_idx = collate_vectors(chunk_idx_list, padding_value=0)
        return audio, audio_lens, tokens, token_lens, token_chunk_idx
