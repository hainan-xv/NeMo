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
"""Dataset for the chunk-completion (spine + branch) streaming SpeechLM.

Reuses the exact chunk/word-readiness assignment of :class:`StreamingSTTDataset`
(via :func:`get_llm_messages_for_batch`), then re-lays the utterance out as a
packed spine + per-chunk branches (see
:mod:`nemo.collections.speechlm2.parts.chunk_completion`). The model then does a
single O(L) forward with a custom 4D mask + ``position_ids`` and takes the loss
on the branch target words.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch

from nemo.collections.speechlm2.data.streaming_stt_dataset import (
    StreamingSTTDataset,
    get_llm_messages_for_batch,
)
from nemo.collections.speechlm2.parts.alignments import WordAlignment
from nemo.collections.speechlm2.parts.chunk_completion import (
    ChunkSpec,
    build_packed_chunk_example,
    collate_packed_chunk_examples,
)
from nemo.utils import logging


@dataclass
class ChunkCompletionBatch:
    """A packed spine+branch batch for the chunk-completion SpeechLM.

    Attributes:
        audios / audio_lens: raw waveforms and lengths (encoder input).
        input_tokens: (B, T) token ids; audio-frame positions hold AUDIO_TOKEN_IDX.
        position_ids: (B, T) RoPE positions (spine index / branch prefix+offset).
        seg_ids: (B, T) 0 spine, >=1 branch id, -1 padding.
        prefix_len: (B, T) per-branch-token history-prefix length.
        target_tokens: (B, T) next-token targets; IGNORE_INDEX except branch words.
        is_audio: (B, T) True at audio-frame positions.
        valid: (B, T) False at right-padding.
        text / cuts: passthrough.
        chunk_size: fixed-chunk size (frames) drawn for this batch.
    """

    audios: Optional[torch.Tensor] = None
    audio_lens: Optional[torch.Tensor] = None
    input_tokens: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    seg_ids: Optional[torch.Tensor] = None
    prefix_len: Optional[torch.Tensor] = None
    target_tokens: Optional[torch.Tensor] = None
    is_audio: Optional[torch.Tensor] = None
    valid: Optional[torch.Tensor] = None
    text: Optional[List[str]] = None
    cuts: Optional[object] = None
    chunk_size: Optional[int] = None


class ChunkCompletionSTTDataset(StreamingSTTDataset):
    """StreamingSTTDataset variant that emits the packed spine+branch layout.

    Only fixed chunking (``chunk_size > 0`` or a list of positive sizes) is
    supported. The audio-span delimiters default to Qwen's in-vocab
    ``<|vision_start|>`` / ``<|vision_end|>`` (no embedding resize); the
    end-of-turn token is the tokenizer EOS (``<|im_end|>`` for Qwen).
    """

    audio_open_token: str = "<|vision_start|>"
    audio_close_token: str = "<|vision_end|>"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.cfg.chunk_size == 0:
            raise ValueError("ChunkCompletionSTTDataset does not support dynamic chunking (chunk_size=0).")
        if isinstance(self.cfg.chunk_size, int) and self.cfg.chunk_size < 0:
            raise ValueError("ChunkCompletionSTTDataset does not support offline mode (chunk_size<0).")

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
                    f"{name}={tok!r} is not a single in-vocabulary token for this tokenizer "
                    f"(got id={tid}). Choose a delimiter that exists in the vocab."
                )
        self.eot_id = hf_tok.eos_token_id
        if self.eot_id is None:
            raise ValueError("Tokenizer has no eos_token_id; required as the branch end-of-turn token.")
        logging.info(
            "ChunkCompletionSTTDataset: audio span delimiters "
            f"{self.audio_open_token!r}={self.vision_start_id} / {self.audio_close_token!r}={self.vision_end_id}, "
            f"eot_id={self.eot_id}"
        )

        # --- Multi-delay-prompt training ---
        # ACTIVATION: set num_delay_frames = -1. Then each batch samples one
        # (delay, prompt) entry uniformly from delay_prompts and trains the whole
        # batch with that entry's delay + natural-language instruction. Any
        # num_delay_frames >= 0 keeps the classic fixed-delay behavior
        # (backward compatible); delay_prompts is then ignored.
        self._delay_prompts = None
        _dp = getattr(self.cfg, "delay_prompts", None)
        if _dp:
            parsed = []
            for e in _dp:
                delay = int(e["delay"])
                prompt = str(e["prompt"])
                if delay < 0:
                    raise ValueError(f"delay_prompts entry has negative delay: {delay}")
                parsed.append({"delay": delay, "prompt": prompt})
            self._delay_prompts = parsed or None

        self._multi_delay = int(self.cfg.num_delay_frames) == -1
        if self._multi_delay:
            if not self._delay_prompts:
                raise ValueError(
                    "num_delay_frames=-1 activates multi-delay-prompt training, but "
                    "data.dataset.delay_prompts is empty/unset. Provide a list of "
                    "{delay, prompt} entries (e.g. delays 0/2/4)."
                )
            logging.info(
                "ChunkCompletionSTTDataset: multi-delay-prompt training (num_delay_frames=-1) over "
                + ", ".join(f"delay={p['delay']}" for p in self._delay_prompts)
                + " (one sampled per batch)."
            )
        elif self._delay_prompts:
            logging.warning(
                "ChunkCompletionSTTDataset: delay_prompts are set but num_delay_frames=%d (not -1); "
                "multi-delay-prompt training is DISABLED. Set num_delay_frames=-1 to enable it.",
                int(self.cfg.num_delay_frames),
            )

    def _sample_delay_prompt(self, rng):
        """Sample one (num_delay_frames, prompt) uniformly, or None if disabled."""
        if not self._delay_prompts:
            return None
        entry = self._delay_prompts[int(rng.integers(len(self._delay_prompts)))]
        return entry["delay"], entry["prompt"]

    def _messages_to_chunks(self, messages: List[dict]) -> List[ChunkSpec]:
        """Parse alternating user(audio)/assistant(words) turns into ChunkSpecs.

        ``messages[0]`` is the system prompt (used as the instruction elsewhere).
        Each user turn's content is ``audio_tag`` repeated once per frame; the
        following assistant turn holds the words revealed by that chunk (or the
        blank sentinel / empty string for a silent chunk).
        """
        chunks: List[ChunkSpec] = []
        audio_tag = self.cfg.audio_tag
        i = 1  # skip system
        n = len(messages)
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
            # Blank sentinel (incl. "" in no-blank mode) -> silent chunk (no words).
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
    ) -> ChunkCompletionBatch:
        audio_durations_secs = (audio_lens.float() / self.cfg.sample_rate).tolist()

        # Draw one fixed chunk size per batch (multi chunk-size) or use the scalar.
        if self._chunk_size_candidates is not None:
            chunk_size = int(self._get_chunk_rng().choice(self._chunk_size_candidates))
        else:
            chunk_size = int(self.cfg.chunk_size)

        # Multi-delay-prompt (num_delay_frames=-1): sample ONE (delay, prompt) for
        # the whole batch. Otherwise use the fixed scalar delay + per-cut prompt.
        if self._multi_delay:
            num_delay_frames, forced_prompt = self._sample_delay_prompt(self._get_chunk_rng())
            system_prompts = [forced_prompt] * len(cuts)
        else:
            num_delay_frames = self.cfg.num_delay_frames
            system_prompts = [cut.custom.get(self.cfg.prompt_field, self.cfg.system_prompt) for cut in cuts]

        batch_messages = get_llm_messages_for_batch(
            system_role=self.cfg.system_role,
            system_prompt=system_prompts,
            audio_tag=self.cfg.audio_tag,
            blank_token=self.cfg.blank_token,
            chunk_size=chunk_size,
            num_delay_frames=num_delay_frames,
            audio_durations_secs=audio_durations_secs,
            frame_length_in_secs=self.cfg.frame_length_in_secs,
            alignments=alignments,
            transcripts=text,
            words_per_group=self.cfg.words_per_group,
            chunk_step=max(int(getattr(self.cfg, "chunk_step", 1)), 1),
            chunk_sizes=None,
            use_word_length_delay=self.cfg.use_word_length_delay,
            word_length_delay_stochastic=self.cfg.word_length_delay_stochastic,
            word_delay_max=self.cfg.word_delay_max,
            word_delay_midpoint=self.cfg.word_delay_midpoint,
            word_delay_slope=self.cfg.word_delay_slope,
            delay_rng=(
                self._get_delay_rng()
                if (self.cfg.use_word_length_delay and self.cfg.word_length_delay_stochastic)
                else None
            ),
        )

        examples = []
        for messages, sysp in zip(batch_messages, system_prompts):
            chunks = self._messages_to_chunks(messages)
            # Instruction/history separator: a newline keeps the first history word
            # from BPE-merging into the instruction text.
            instruction_ids = self.tokenizer.text_to_ids(sysp + "\n")
            examples.append(
                build_packed_chunk_example(
                    instruction_ids=instruction_ids,
                    chunks=chunks,
                    vision_start_id=self.vision_start_id,
                    vision_end_id=self.vision_end_id,
                    eot_id=self.eot_id,
                )
            )

        packed = collate_packed_chunk_examples(examples, pad_id=self.tokenizer.pad_id)

        return ChunkCompletionBatch(
            audios=audios,
            audio_lens=audio_lens,
            input_tokens=packed.input_ids,
            position_ids=packed.position_ids,
            seg_ids=packed.seg_ids,
            prefix_len=packed.prefix_len,
            target_tokens=packed.target_ids,
            is_audio=packed.is_audio,
            valid=packed.valid,
            text=text,
            cuts=cuts,
            chunk_size=chunk_size,
        )
