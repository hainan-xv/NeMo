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
"""Dataset producing SCRIPT's packed spine+branch batches."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from nemo.collections.speechlm2.data.streaming_stt_dataset import (
    StreamingSTTDataConfig,
    StreamingSTTDataset,
)
from nemo.collections.speechlm2.parts.alignments import WordAlignment
from nemo.collections.speechlm2.parts.script import (
    ChunkSpec,
    build_packed_chunk_example,
    collate_packed_chunk_examples,
)
from nemo.collections.speechlm2.parts.script_messages import get_llm_messages_for_batch
from nemo.collections.speechlm2.parts.utils import to_dataclass
from nemo.utils import logging


@dataclass
class ScriptSTTDataConfig(StreamingSTTDataConfig):
    """:class:`StreamingSTTDataConfig` plus SCRIPT's own knobs.

    Attributes:
        audio_history_chunks: ``M`` — how many PREVIOUS chunks' audio each branch
            also sees. Must match ``model.audio_history_chunks`` so that training
            and inference build the same window.
        audio_window_frames: ``F`` — if ``> 0``, give every branch a FIXED window
            of ``F`` frames ending at its chunk boundary instead of a whole number
            of chunks, so the acoustic context is constant across chunk sizes.
            Takes precedence over ``audio_history_chunks``. Must match
            ``model.audio_window_frames``.
        chunk_size_seed: base seed for the per-batch chunk-size draw. Offset per
            dataloader worker so workers do not draw identical sequences.
    """

    audio_history_chunks: int = 0
    audio_window_frames: int = 0
    chunk_size_seed: int = 1234


@dataclass
class ScriptBatch:
    """A packed spine+branch batch for the SCRIPT SpeechLM.

    Attributes:
        audios / audio_lens: raw waveforms ``(B, T_samples)`` and sample counts ``(B,)``.
        input_tokens: (B, T) token ids; audio-frame slots hold ``AUDIO_TOKEN_IDX``.
        position_ids: (B, T) RoPE positions (spine index, or branch prefix+offset).
        seg_ids: (B, T) ``0`` spine, ``>= 1`` branch id, ``-1`` padding.
        prefix_len: (B, T) per-branch-token history-prefix length.
        target_tokens: (B, T) next-token targets; ``IGNORE_INDEX`` except branch words.
        is_audio: (B, T) True at audio-frame slots.
        audio_frame_index: (B, T) global encoder-frame index each audio slot maps
            to (``-1`` elsewhere). Set only when ``audio_history_chunks > 0``,
            where a frame is reused across branches and the model must gather by
            explicit index rather than by positional cumsum.
        valid: (B, T) False at right-padding.
        text / cuts: passthrough for metrics and per-cut prompts.
        chunk_size: the fixed chunk size drawn for this batch.
    """

    audios: Optional[torch.Tensor] = None
    audio_lens: Optional[torch.Tensor] = None
    input_tokens: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    seg_ids: Optional[torch.Tensor] = None
    prefix_len: Optional[torch.Tensor] = None
    target_tokens: Optional[torch.Tensor] = None
    is_audio: Optional[torch.Tensor] = None
    audio_frame_index: Optional[torch.Tensor] = None
    valid: Optional[torch.Tensor] = None
    text: Optional[List[str]] = None
    cuts: Optional[object] = None
    chunk_size: Optional[int] = None


class ScriptSTTDataset(StreamingSTTDataset):
    """:class:`StreamingSTTDataset` variant emitting the packed spine+branch layout.

    Only fixed chunking is supported (``chunk_size > 0``, or a list of positive
    sizes for multi chunk-size training). The audio span delimiters default to
    Qwen's in-vocab ``<|vision_start|>`` / ``<|vision_end|>``, so no embedding
    resize is needed; the branch end-of-turn token is the tokenizer's EOS
    (``<|im_end|>`` for Qwen).
    """

    audio_open_token: str = "<|vision_start|>"
    audio_close_token: str = "<|vision_end|>"

    def __init__(self, cfg, tokenizer, defer_get_batch: bool = False):
        super().__init__(cfg, tokenizer, defer_get_batch=defer_get_batch)

        # The base __init__ coerces cfg through StreamingSTTDataConfig, which
        # silently drops SCRIPT's extra keys. Re-coerce through the extended
        # dataclass, then re-apply the one in-place normalization the base does
        # (Hydra loads "\\n" literally, so escapes must be interpreted).
        self.cfg: ScriptSTTDataConfig = to_dataclass(ScriptSTTDataConfig, cfg)
        self.cfg.blank_token = self.cfg.blank_token.encode().decode('unicode_escape')

        if isinstance(self.cfg.chunk_size, int) and self.cfg.chunk_size <= 0:
            raise ValueError(
                f"ScriptSTTDataset supports fixed chunking only; got chunk_size={self.cfg.chunk_size}. "
                "Use a positive int, or a list of positive ints for multi chunk-size training."
            )

        self._audio_history_chunks = max(int(self.cfg.audio_history_chunks), 0)
        self._audio_window_frames = max(int(self.cfg.audio_window_frames), 0)
        if self._audio_window_frames > 0 and self._audio_history_chunks > 0:
            logging.warning(
                "Both audio_window_frames=%d and audio_history_chunks=%d are set; "
                "the fixed-frame window takes precedence and audio_history_chunks is ignored.",
                self._audio_window_frames,
                self._audio_history_chunks,
            )

        hf_tok = self.tokenizer.tokenizer
        self.vision_start_id = hf_tok.convert_tokens_to_ids(self.audio_open_token)
        self.vision_end_id = hf_tok.convert_tokens_to_ids(self.audio_close_token)
        unk = getattr(hf_tok, "unk_token_id", None)
        for name, tid, tok in (
            ("audio_open_token", self.vision_start_id, self.audio_open_token),
            ("audio_close_token", self.vision_end_id, self.audio_close_token),
        ):
            if tid is None or (unk is not None and tid == unk):
                raise ValueError(
                    f"{name}={tok!r} is not a single in-vocabulary token for this tokenizer (got id={tid}). "
                    "Choose a delimiter that already exists in the vocab."
                )
        self.eot_id = hf_tok.eos_token_id
        if self.eot_id is None:
            raise ValueError("Tokenizer has no eos_token_id; it is required as the branch end-of-turn token.")

        # Per-worker RNG for the per-batch chunk-size draw.
        self._chunk_rngs: dict = {}

        logging.info(
            "ScriptSTTDataset: audio delimiters %r=%d / %r=%d, eot_id=%d, "
            "audio_history_chunks=%d, audio_window_frames=%d",
            self.audio_open_token,
            self.vision_start_id,
            self.audio_close_token,
            self.vision_end_id,
            self.eot_id,
            self._audio_history_chunks,
            self._audio_window_frames,
        )

    def _get_chunk_rng(self) -> np.random.Generator:
        """RNG for the per-batch chunk-size draw, seeded per dataloader worker.

        Workers must not draw identical chunk-size sequences, so the base seed is
        offset by the worker id.
        """
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        if wid not in self._chunk_rngs:
            self._chunk_rngs[wid] = np.random.default_rng(int(self.cfg.chunk_size_seed) + wid)
        return self._chunk_rngs[wid]

    def _messages_to_chunks(self, messages: List[dict]) -> List[ChunkSpec]:
        """Parse alternating user(audio)/assistant(words) turns into ChunkSpecs.

        ``messages[0]`` is the system prompt (used separately as the
        instruction). Each user turn's content is ``audio_tag`` repeated once per
        frame; the assistant turn that follows holds the words that chunk
        reveals, or the blank sentinel for a silent chunk.
        """
        chunks: List[ChunkSpec] = []
        audio_tag = self.cfg.audio_tag
        i, n = 1, len(messages)  # skip the system turn
        while i < n:
            m = messages[i]
            if m["role"] != "user":
                i += 1
                continue
            audio_len = m["content"].count(audio_tag)
            words = ""
            if i + 1 < n and messages[i + 1]["role"] == "assistant":
                words = messages[i + 1]["content"]
                i += 2
            else:
                i += 1
            # The blank sentinel (including "" in no-blank mode) means a silent chunk.
            if words == self.cfg.blank_token:
                words = ""
            target_ids = self.tokenizer.text_to_ids(words) if words.strip() else []
            chunks.append(ChunkSpec(audio_len=audio_len, target_ids=target_ids))
        return chunks

    def get_batch_data(
        self,
        cuts,
        audios: torch.Tensor,
        audio_lens: torch.Tensor,
        alignments: List[List[WordAlignment]],
        text: List[str],
    ) -> ScriptBatch:
        audio_durations_secs = (audio_lens.float() / self.cfg.sample_rate).tolist()

        # One fixed chunk size per batch (multi chunk-size training), or the scalar.
        if self._chunk_size_candidates is not None:
            chunk_size = int(self._get_chunk_rng().choice(self._chunk_size_candidates))
        else:
            chunk_size = int(self.cfg.chunk_size)

        system_prompts = [cut.custom.get(self.cfg.prompt_field, self.cfg.system_prompt) for cut in cuts]

        batch_messages = get_llm_messages_for_batch(
            system_role=self.cfg.system_role,
            system_prompt=system_prompts,
            audio_tag=self.cfg.audio_tag,
            blank_token=self.cfg.blank_token,
            chunk_size=chunk_size,
            num_delay_frames=self.cfg.num_delay_frames,
            audio_durations_secs=audio_durations_secs,
            frame_length_in_secs=self.cfg.frame_length_in_secs,
            alignments=alignments,
            transcripts=text,
        )

        examples = []
        for messages, sysp in zip(batch_messages, system_prompts):
            # Instruction/history separator: the trailing newline keeps the first
            # history word from BPE-merging into the instruction text.
            instruction_ids = self.tokenizer.text_to_ids(sysp + "\n")
            examples.append(
                build_packed_chunk_example(
                    instruction_ids=instruction_ids,
                    chunks=self._messages_to_chunks(messages),
                    vision_start_id=self.vision_start_id,
                    vision_end_id=self.vision_end_id,
                    eot_id=self.eot_id,
                    audio_history_chunks=self._audio_history_chunks,
                    audio_window_frames=self._audio_window_frames,
                )
            )

        packed = collate_packed_chunk_examples(examples, pad_id=self.tokenizer.pad_id)

        return ScriptBatch(
            audios=audios,
            audio_lens=audio_lens,
            input_tokens=packed.input_ids,
            position_ids=packed.position_ids,
            seg_ids=packed.seg_ids,
            prefix_len=packed.prefix_len,
            target_tokens=packed.target_ids,
            is_audio=packed.is_audio,
            # Only needed when a window reuses frames across branches. With M == 0
            # the audio slots are a plain 0,1,2,... run, so the model can take the
            # cheaper (and numerically identical) cumsum interleave path.
            audio_frame_index=(
                packed.audio_frame_index if (self._audio_history_chunks > 0 or self._audio_window_frames > 0) else None
            ),
            valid=packed.valid,
            text=text,
            cuts=cuts,
            chunk_size=chunk_size,
        )
