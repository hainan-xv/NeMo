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
"""Dataset for the SCRIPT (spine + branch) streaming SpeechLM.

Reuses the exact chunk/word-readiness assignment of :class:`StreamingSTTDataset`
(via :func:`get_llm_messages_for_batch`), then re-lays the utterance out as a
packed spine + per-chunk branches (see
:mod:`nemo.collections.speechlm2.parts.script`). The model then does a
single O(L) forward with a custom 4D mask + ``position_ids`` and takes the loss
on the branch target words.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.speechlm2.data.streaming_stt_dataset import (
    StreamingSTTDataset,
    get_llm_messages_for_batch,
)
from nemo.collections.speechlm2.parts.alignments import WordAlignment
from nemo.collections.speechlm2.parts.script import (
    ChunkSpec,
    build_packed_chunk_example,
    build_packed_redecode_example,
    collate_packed_chunk_examples,
)
from nemo.collections.speechlm2.parts.shared_audio_chunk import (
    build_shared_audio_chunk_example,
    collate_shared_audio_examples,
)
from nemo.utils import logging


@dataclass
class ScriptBatch:
    """A packed spine+branch batch for the SCRIPT SpeechLM.

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
    # (B, T) global encoder-frame index each audio slot maps to (-1 elsewhere).
    # Set only when audio_history_chunks > 0 (windowed audio reuses frames across
    # branches, so the model gathers by this index instead of the cumsum fill).
    audio_frame_index: Optional[torch.Tensor] = None
    # Shared-audio layout only (audio laid once, windowed via the mask): per-branch
    # audio-window frame bounds [win_start, win_end). None in the per-branch-audio layout.
    win_start: Optional[torch.Tensor] = None
    win_end: Optional[torch.Tensor] = None
    valid: Optional[torch.Tensor] = None
    text: Optional[List[str]] = None
    cuts: Optional[object] = None
    chunk_size: Optional[int] = None
    # Self-correction only: per-example (instruction_ids, chunks) so the model can
    # rebuild corrupted branches from its own forced-decoding errors. None otherwise.
    chunk_meta: Optional[List] = None


class ScriptSTTDataset(StreamingSTTDataset):
    """StreamingSTTDataset variant that emits the packed spine+branch layout.

    Only fixed chunking (``chunk_size > 0`` or a list of positive sizes) is
    supported. The audio-span delimiters default to Qwen's in-vocab
    ``<|vision_start|>`` / ``<|vision_end|>`` (no embedding resize); the
    end-of-turn token is the tokenizer EOS (``<|im_end|>`` for Qwen).
    """

    audio_open_token: str = "<|vision_start|>"
    audio_close_token: str = "<|vision_end|>"

    # Default prompt template for exact-delay / text-representation modes.
    # {delay} is filled with the exact integer delay (frames); {format_clause}
    # with the capitalization/punctuation instruction for the batch.
    _DEFAULT_PROMPT_TEMPLATE: str = (
        "You are doing streaming speech recognition. Given the transcript so far and "
        "the next audio chunk, output the words spoken in that chunk. Emit each chunk's "
        "words with a fixed delay of {delay} frames. {format_clause}"
    )
    # The 4 capitalization x punctuation format clauses (keys: cap/nocap _ punct/nopunct).
    _DEFAULT_FORMAT_CLAUSES = {
        "cap_punct": "Write the text with normal capitalization and punctuation.",
        "cap_nopunct": "Write the text with normal capitalization but no punctuation.",
        "nocap_punct": "Write the text in all lowercase, keeping punctuation.",
        "nocap_nopunct": "Write the text in all lowercase with no punctuation.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The base StreamingSTTDataset coerces cfg to the StreamingSTTDataConfig
        # dataclass (via to_dataclass), which drops SCRIPT-specific keys (delay_prompts,
        # self_correction, audio_history_chunks, chunk_size_seed, word_delay_seed, ...).
        # Re-attach them from the raw cfg so the getattr(self.cfg, ...) reads below behave
        # as they do on the branch SCRIPT was developed against.
        _raw_cfg = kwargs.get("cfg", args[0] if args else None)
        if _raw_cfg is not None:
            _raw = OmegaConf.to_container(_raw_cfg, resolve=True) if isinstance(_raw_cfg, DictConfig) else dict(_raw_cfg)
            for _k, _v in _raw.items():
                if not hasattr(self.cfg, _k):
                    setattr(self.cfg, _k, _v)

        # Per-rank/worker RNG streams. The base class on this branch does not create
        # these, so SCRIPT owns them here (used by _get_chunk_rng / _get_delay_rng).
        if not hasattr(self, "_chunk_rngs"):
            self._chunk_rngs: dict[int, np.random.Generator] = {}
        if not hasattr(self, "_delay_rngs"):
            self._delay_rngs: dict[int, np.random.Generator] = {}

        if self.cfg.chunk_size == 0:
            raise ValueError("ScriptSTTDataset does not support dynamic chunking (chunk_size=0).")
        if isinstance(self.cfg.chunk_size, int) and self.cfg.chunk_size < 0:
            raise ValueError("ScriptSTTDataset does not support offline mode (chunk_size<0).")

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
            "ScriptSTTDataset: audio span delimiters "
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

        # --- Exact-delay + text-representation prompting ---
        # Parsed BEFORE the multi-delay validation so the exact-delay mode can
        # activate prompt-controlled delay WITHOUT a delay_prompts list.
        self._exact_delay = bool(getattr(self.cfg, "exact_delay", False))
        self._exact_max_delay = int(getattr(self.cfg, "exact_max_delay", 0) or 0)
        self._vary_text_repr = bool(getattr(self.cfg, "vary_text_repr", False))
        self._text_repr_keep = set(getattr(self.cfg, "text_repr_keep_chars", "'") or "")
        self._prompt_template = str(getattr(self.cfg, "prompt_template", None) or self._DEFAULT_PROMPT_TEMPLATE)
        # --- Separate TRAINING base prompt ---
        # When set, TRAINING builds its per-batch prompt from this base (then appends
        # the sampled format clause and the rendered chunk-size clause), while the
        # fully-rendered ``system_prompt`` below is reserved for VALIDATION (read
        # verbatim by model.val_system_prompt). This mirrors the exact-delay path
        # (where system_prompt is val-only and training uses a template): it lets a
        # FIXED-delay recipe use chunk-size / text-representation prompting whose
        # train and val renders are byte-identical, WITHOUT double-appending the
        # chunk/format clauses onto a system_prompt that already contains them.
        # Unset (default) => training uses system_prompt as the base (unchanged).
        self._train_system_prompt = getattr(self.cfg, "train_system_prompt", None)
        if self._train_system_prompt is not None:
            self._train_system_prompt = str(self._train_system_prompt)
            logging.info(
                "ScriptSTTDataset: train_system_prompt set; TRAINING uses it as the "
                "prompt base (format/chunk clauses appended), VALIDATION reads "
                "data.dataset.system_prompt verbatim."
            )
        # --- Chunk-size prompting ---
        # When set, the per-batch chunk size (frames) is rendered into the prompt via
        # this template's {chunk_size} placeholder and appended to the system prompt,
        # so the model is TOLD its chunk size (and can be asked for a specific one at
        # inference). Off by default (backward compatible: chunk size stays implicit).
        self._chunk_size_prompt_template = getattr(self.cfg, "chunk_size_prompt_template", None)
        if self._chunk_size_prompt_template is not None:
            self._chunk_size_prompt_template = str(self._chunk_size_prompt_template)
            if "{chunk_size}" not in self._chunk_size_prompt_template:
                raise ValueError("chunk_size_prompt_template must contain '{chunk_size}'.")
            logging.info(
                "ScriptSTTDataset: chunk-size prompting ON (template=%r; the per-batch "
                "chunk size in frames is rendered into the prompt).",
                self._chunk_size_prompt_template,
            )
        self._format_clauses = dict(self._DEFAULT_FORMAT_CLAUSES)
        _fc = getattr(self.cfg, "format_clauses", None)
        if _fc:
            self._format_clauses.update({str(k): str(v) for k, v in dict(_fc).items()})
        if self._vary_text_repr and "{format_clause}" not in self._prompt_template:
            # Non-exact modes append the clause instead; only the templated
            # (exact-delay) path needs the placeholder. Warn if exact + missing.
            if self._exact_delay:
                raise ValueError("prompt_template must contain '{format_clause}' when vary_text_repr=True.")

        self._multi_delay = int(self.cfg.num_delay_frames) == -1
        if self._multi_delay:
            if self._exact_delay:
                if self._exact_max_delay < 1:
                    raise ValueError("exact_delay=True requires exact_max_delay >= 1.")
                if "{delay}" not in self._prompt_template:
                    raise ValueError("prompt_template must contain '{delay}' when exact_delay=True.")
                logging.info(
                    "ScriptSTTDataset: EXACT-delay prompting (num_delay_frames=-1); "
                    "delay ~ Uniform[0, %d] per batch, rendered into the prompt.",
                    self._exact_max_delay,
                )
            elif not self._delay_prompts:
                raise ValueError(
                    "num_delay_frames=-1 activates multi-delay-prompt training, but "
                    "data.dataset.delay_prompts is empty/unset. Provide a list of "
                    "{delay, prompt} entries (e.g. delays 0/2/4), or set exact_delay=true "
                    "with exact_max_delay."
                )
            else:
                logging.info(
                    "ScriptSTTDataset: multi-delay-prompt training (num_delay_frames=-1) over "
                    + ", ".join(f"delay={p['delay']}" for p in self._delay_prompts)
                    + " (one sampled per batch)."
                )
        else:
            if self._exact_delay:
                logging.warning(
                    "ScriptSTTDataset: exact_delay is set but num_delay_frames=%d (not -1); "
                    "exact-delay prompting is DISABLED. Set num_delay_frames=-1 to enable it.",
                    int(self.cfg.num_delay_frames),
                )
            elif self._delay_prompts:
                logging.warning(
                    "ScriptSTTDataset: delay_prompts are set but num_delay_frames=%d (not -1); "
                    "multi-delay-prompt training is DISABLED. Set num_delay_frames=-1 to enable it.",
                    int(self.cfg.num_delay_frames),
                )
        if self._vary_text_repr:
            logging.info(
                "ScriptSTTDataset: text-representation variation ON (4 capitalization x "
                "punctuation combos, one sampled per batch; keep-chars=%r).",
                "".join(sorted(self._text_repr_keep)),
            )

        # --- Audio history window + history-word recovery ---
        self._audio_history_chunks = max(int(getattr(self.cfg, "audio_history_chunks", 0) or 0), 0)
        self._audio_window_frames = max(int(getattr(self.cfg, "audio_window_frames", 0) or 0), 0)
        if self._audio_window_frames > 0 and self._audio_history_chunks > 0:
            raise ValueError(
                "audio_window_frames and audio_history_chunks are mutually exclusive "
                "(pick a frame-based OR a chunk-based audio window)."
            )
        if self._audio_window_frames > 0:
            logging.info(
                "ScriptSTTDataset: audio_window_frames=%d (fixed acoustic context; "
                "each branch sees the last %d encoder frames ending at the chunk boundary, "
                "regardless of chunk size).",
                self._audio_window_frames,
                self._audio_window_frames,
            )
        self._history_word_recovery_prob = float(getattr(self.cfg, "history_word_recovery_prob", 0.0) or 0.0)
        if not (0.0 <= self._history_word_recovery_prob <= 1.0):
            raise ValueError(
                f"history_word_recovery_prob must be in [0, 1], got {self._history_word_recovery_prob}"
            )
        if self._audio_history_chunks > 0:
            logging.info(
                "ScriptSTTDataset: audio_history_chunks=%d%s",
                self._audio_history_chunks,
                (
                    f" | history_word_recovery_prob={self._history_word_recovery_prob}"
                    if self._history_word_recovery_prob > 0.0
                    else ""
                ),
            )
        elif self._history_word_recovery_prob > 0.0:
            # Recovery with NO audio window: the recovering branch drops the previous
            # chunk's last word from history but does NOT see that chunk's audio, so
            # the model must recover it from the current chunk's (left-context-
            # carrying) encoder frames + the remaining text history.
            logging.info(
                "ScriptSTTDataset: history_word_recovery_prob=%.3f with "
                "audio_history_chunks=0 (no prior-chunk audio window; recover the "
                "dropped word from the current chunk's audio left-context + text history).",
                self._history_word_recovery_prob,
            )

        # --- Contiguous-text positions ("Option A") ---
        self._contiguous_text_positions = bool(getattr(self.cfg, "contiguous_text_positions", False))
        if self._contiguous_text_positions:
            logging.info(
                "ScriptSTTDataset: contiguous_text_positions=True (words placed "
                "contiguous with history; audio prelude overlaid on history tail positions)."
            )

        # --- Self-correction (delete-last-word), forced/DAgger variant ---
        self._self_correction = bool(getattr(self.cfg, "self_correction", False))
        self._self_correction_prompt_suffix = getattr(self.cfg, "self_correction_prompt_suffix", None)
        # Per-batch prompt-controlled toggle: with prob p a batch trains WITH the
        # correction objective (emit chunk_meta so the model runs its forced-decode
        # DAgger step + the ON clause is stated in the prompt); with prob 1-p it
        # trains as a plain SCRIPT batch (chunk_meta=None -> normal step +
        # the OFF clause is stated). p=1.0 (default) reproduces the always-on model.
        self._self_correction_batch_prob = float(getattr(self.cfg, "self_correction_batch_prob", 1.0) or 0.0)
        self._no_self_correction_prompt_suffix = getattr(self.cfg, "no_self_correction_prompt_suffix", None)
        # Correction scope: "word" (delete only the previous chunk's last word) or
        # "chunk" (delete the entire previous chunk). Forced/DAgger variant only;
        # the model reads the same knob from its own config for the rebuild + decode.
        self._self_correction_scope = str(getattr(self.cfg, "self_correction_scope", "word") or "word").lower()
        if self._self_correction_scope not in ("word", "chunk"):
            raise ValueError(
                f"self_correction_scope must be 'word' or 'chunk', got {self._self_correction_scope!r}"
            )
        if self._self_correction:
            if self._contiguous_text_positions:
                raise ValueError("self_correction is not compatible with contiguous_text_positions.")
            if bool(getattr(self.cfg, "shared_audio_track", False)):
                raise ValueError("self_correction is not compatible with shared_audio_track (yet).")
            if self._history_word_recovery_prob > 0.0:
                # The DAgger step (ScriptSTTModel._self_correction_training_step) rebuilds
                # each branch from chunk_meta with corrupt_prev ONLY, so any recover_prev
                # the dataset baked into the forced-history batch would be silently dropped.
                # build_packed_chunk_example documents corrupt_prev/recover_prev as mutually
                # exclusive; enforce it here rather than train an inconsistent objective.
                raise ValueError(
                    "self_correction (forced/DAgger) and history_word_recovery_prob are mutually "
                    "exclusive: the DAgger rebuild applies corrupt_prev only and would drop the "
                    "recovery objective. Enable at most one."
                )
            if not (0.0 <= self._self_correction_batch_prob <= 1.0):
                raise ValueError(
                    f"self_correction_batch_prob must be in [0, 1], got {self._self_correction_batch_prob}"
                )
            logging.info(
                "ScriptSTTDataset: self_correction=True (forced/DAgger) | scope=%s | batch_prob=%.2f "
                "(fraction of batches trained WITH the correction objective; the rest are plain "
                "SCRIPT batches with the 'do not correct' clause).",
                self._self_correction_scope,
                self._self_correction_batch_prob,
            )

        # --- Self-correction via prefix heuristic (data-side) ---
        self._self_correction_prefix = bool(getattr(self.cfg, "self_correction_prefix", False))
        self._sc_prefix_prob = float(getattr(self.cfg, "self_correction_prefix_prob", 0.2) or 0.0)
        self._sc_delete_id = None
        if self._self_correction_prefix:
            if self._self_correction:
                raise ValueError("self_correction (forced) and self_correction_prefix are mutually exclusive.")
            if self._self_correction_scope != "word":
                # The prefix heuristic bakes a random char-prefix of the previous
                # chunk's last WORD; whole-chunk scope is only wired for the forced
                # (DAgger) variant. Guard rather than silently ignore the scope.
                raise NotImplementedError(
                    "self_correction_scope='chunk' is only supported with the forced self_correction "
                    "variant, not self_correction_prefix."
                )
            if self._contiguous_text_positions:
                raise ValueError("self_correction_prefix is not compatible with contiguous_text_positions.")
            if bool(getattr(self.cfg, "shared_audio_track", False)):
                raise ValueError("self_correction_prefix is not compatible with shared_audio_track (yet).")
            if not (0.0 <= self._sc_prefix_prob <= 1.0):
                raise ValueError(f"self_correction_prefix_prob must be in [0, 1], got {self._sc_prefix_prob}")
            delete_token = str(getattr(self.cfg, "delete_token", "<|object_ref_start|>"))
            hf_tok = self.tokenizer.tokenizer
            self._sc_delete_id = hf_tok.convert_tokens_to_ids(delete_token)
            unk = getattr(hf_tok, "unk_token_id", None)
            if self._sc_delete_id is None or (unk is not None and self._sc_delete_id == unk):
                raise ValueError(
                    f"delete_token {delete_token!r} is not a valid in-vocab token (id={self._sc_delete_id})."
                )
            logging.info(
                "ScriptSTTDataset: self_correction_prefix ON (prob=%.2f, delete_token=%r id=%s) -- "
                "the previous chunk's last word is truncated to a random char prefix; target = <del> w_prev w_k.",
                self._sc_prefix_prob, delete_token, self._sc_delete_id,
            )

        # --- Windowed re-decoding self-correction ---
        # Each chunk c gets one branch per lookahead level j (0..R): predict chunk c
        # from clean history y_0..y_{c-1} and the (M+1)-chunk audio window ending at
        # the arrival chunk c+j (j chunks of lookahead). j=0 is the base branch.
        self._redecode = bool(getattr(self.cfg, "redecode", False))
        self._redecode_depth = int(getattr(self.cfg, "redecode_depth", 0) or 0)
        self._redecode_train_prob = float(getattr(self.cfg, "redecode_train_prob", 1.0) or 0.0)
        if self._redecode:
            if self._audio_history_chunks < 1:
                raise ValueError("redecode requires audio_history_chunks >= 1 (the window is M+1 chunks).")
            if self._redecode_depth <= 0:
                self._redecode_depth = self._audio_history_chunks  # default R = M
            if not (1 <= self._redecode_depth <= self._audio_history_chunks):
                raise ValueError(
                    f"redecode_depth must be in [1, audio_history_chunks={self._audio_history_chunks}], "
                    f"got {self._redecode_depth}"
                )
            if not (0.0 <= self._redecode_train_prob <= 1.0):
                raise ValueError(f"redecode_train_prob must be in [0, 1], got {self._redecode_train_prob}")
            for name, on in (
                ("audio_window_frames", self._audio_window_frames > 0),
                ("shared_audio_track", bool(getattr(self.cfg, "shared_audio_track", False))),
                ("contiguous_text_positions", self._contiguous_text_positions),
                ("self_correction", self._self_correction),
                ("self_correction_prefix", self._self_correction_prefix),
                ("history_word_recovery_prob>0", self._history_word_recovery_prob > 0.0),
            ):
                if on:
                    raise ValueError(f"redecode is not compatible with {name}.")
            logging.info(
                "ScriptSTTDataset: redecode ON (windowed re-decoding) | window=%d chunks (M=%d) | "
                "depth R=%d | train_prob=%.2f -- each chunk is trained at lookahead levels 0..R "
                "(j=0 is the base branch; higher j adds one more chunk of lookahead).",
                self._audio_history_chunks + 1, self._audio_history_chunks,
                self._redecode_depth, self._redecode_train_prob,
            )

        # --- Shared-audio packed layout ---
        self._shared_audio_track = bool(getattr(self.cfg, "shared_audio_track", False))
        if self._shared_audio_track:
            if self._contiguous_text_positions:
                raise ValueError("shared_audio_track is not compatible with contiguous_text_positions.")
            if self._history_word_recovery_prob > 0.0:
                raise ValueError("shared_audio_track does not support history_word_recovery_prob (yet).")
            logging.info(
                "ScriptSTTDataset: shared_audio_track=True (encoder frames laid ONCE; "
                "each branch windowed via the mask -> packed length independent of the audio window)."
            )

    def _get_chunk_rng(self) -> np.random.Generator:
        """Deterministic RNG for chunk-size selection, unique per rank/worker.

        Provided by SCRIPT because the base StreamingSTTDataset on this branch selects
        chunk sizes with the global ``random`` module instead of a per-rank stream.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            key = worker_info.id
            seed = worker_info.seed
        else:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            key = -(rank + 1)
            seed = int(getattr(self.cfg, "chunk_size_seed", 42)) + rank
        if key not in self._chunk_rngs:
            self._chunk_rngs[key] = np.random.default_rng(seed % (2**32))
        return self._chunk_rngs[key]

    def _get_delay_rng(self) -> np.random.Generator:
        """RNG for stochastic word-length delays, rank/worker-local and seeded
        independently from the chunk-size RNG."""
        word_delay_seed = int(getattr(self.cfg, "word_delay_seed", 1234))
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            key = worker_info.id
            seed = worker_info.seed + word_delay_seed
        else:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            key = -(rank + 1)
            seed = word_delay_seed + rank
        if key not in self._delay_rngs:
            self._delay_rngs[key] = np.random.default_rng(seed % (2**32))
        return self._delay_rngs[key]

    def _sample_delay_prompt(self, rng):
        """Sample one (num_delay_frames, prompt) uniformly, or None if disabled."""
        if not self._delay_prompts:
            return None
        entry = self._delay_prompts[int(rng.integers(len(self._delay_prompts)))]
        return entry["delay"], entry["prompt"]

    # --- Exact-delay + text-representation prompting helpers ---

    @staticmethod
    def _repr_key(cap: bool, punct: bool) -> str:
        return f"{'cap' if cap else 'nocap'}_{'punct' if punct else 'nopunct'}"

    def _format_clause(self, cap: bool, punct: bool) -> str:
        return self._format_clauses[self._repr_key(cap, punct)]

    def _build_exact_prompt(self, delay: int, cap: bool, punct: bool) -> str:
        """Render the prompt template with the exact delay and (optional) format clause."""
        clause = self._format_clause(cap, punct) if self._vary_text_repr else ""
        return self._prompt_template.format(delay=int(delay), format_clause=clause).strip()

    def _append_format_clause(self, prompt: str, cap: bool, punct: bool) -> str:
        """Append the (cap, punct) format clause to an existing prompt string."""
        clause = self._format_clause(cap, punct)
        return (prompt.rstrip() + " " + clause).strip() if clause else prompt

    def _append_chunk_clause(self, prompt: str, chunk_size: int) -> str:
        """Append the chunk-size clause (rendered with the batch chunk size in frames)."""
        if not self._chunk_size_prompt_template:
            return prompt
        clause = self._chunk_size_prompt_template.format(chunk_size=int(chunk_size)).strip()
        return (prompt.rstrip() + " " + clause).strip() if clause else prompt

    def _append_correction_clause(self, prompt: str, sc_on: bool) -> str:
        """Append the self-correction ON/OFF clause so the prompt states whether the
        model is allowed to delete + rewrite a previously committed word."""
        suffix = self._self_correction_prompt_suffix if sc_on else self._no_self_correction_prompt_suffix
        if not suffix:
            return prompt
        return (prompt.rstrip() + " " + str(suffix)).strip()

    def _strip_punct(self, s: str) -> str:
        """Remove punctuation (keep alphanumerics, whitespace, and text_repr_keep_chars),
        collapsing whitespace runs introduced by the removal."""
        kept = "".join(ch for ch in s if ch.isalnum() or ch.isspace() or ch in self._text_repr_keep)
        return " ".join(kept.split())

    def _apply_text_repr(self, content: str, cap: bool, punct: bool) -> str:
        """Transform one chunk's target text for the (cap, punct) setting.

        Preserves a single leading space (the byte-level-BPE word-boundary marker)
        and leaves the blank/silent sentinel untouched. A chunk that becomes empty
        after punctuation removal (e.g. punctuation-only content) collapses to ''
        and is treated as a silent chunk downstream.
        """
        if not content or content == self.cfg.blank_token:
            return content
        lead = " " if content[:1] == " " else ""
        body = content[1:] if lead else content
        if not punct:
            body = self._strip_punct(body)
        if not cap:
            body = body.lower()
        return (lead + body) if body else ""

    def _apply_text_repr_to_messages(self, messages: List[dict], cap: bool, punct: bool) -> List[dict]:
        """Return a copy of ``messages`` with each assistant turn's content
        transformed for the batch's (cap, punct) setting (system/user untouched)."""
        out = []
        for m in messages:
            if m["role"] == "assistant":
                out.append({"role": "assistant", "content": self._apply_text_repr(m["content"], cap, punct)})
            else:
                out.append(m)
        return out

    def _sample_text_repr(self, rng) -> tuple:
        """Sample a batch-shared (cap, punct) setting; (True, True) if disabled."""
        if not self._vary_text_repr:
            return True, True
        return bool(rng.integers(0, 2)), bool(rng.integers(0, 2))

    def _last_word_ids(self, content: str, target_ids: List[int]) -> List[int]:
        """Token ids of just the LAST word of ``content`` (a trailing slice of
        ``target_ids``), for history-word recovery.

        Splits off the final whitespace-delimited word and locates its tokens by
        the length of the tokenized head (byte-level BPE keeps word boundaries at
        spaces, so ``len(tok(head))`` tokens form the prefix). Falls back to the
        whole thing on any boundary mismatch (conservative — recovers more).
        """
        c = content.rstrip()
        if not c.strip():
            return []
        head, sep, _tail = c.rpartition(" ")
        if sep == "" or head.strip() == "":
            return list(target_ids)  # single word -> whole chunk text is one word
        prefix_ids = self.tokenizer.text_to_ids(head)
        n_prefix = len(prefix_ids)
        if 0 < n_prefix < len(target_ids):
            return list(target_ids[n_prefix:])
        return list(target_ids)

    def _prefix_word_ids(self, last_word_ids: List[int], rng) -> Optional[List[int]]:
        """A random CHARACTER prefix of a word (given as token ids), returned as token
        ids, for the prefix self-correction heuristic. Keeps a uniformly random number
        of the word's characters in ``[1, len-1]`` (a strict truncation), preserving
        the word's leading space so it still tokenizes as a word. Returns None when the
        word is too short or the prefix retokenizes back to the same word.
        """
        if not last_word_ids:
            return None
        w_full = self.tokenizer.ids_to_text(list(last_word_ids))
        lead = " " if w_full[:1] == " " else ""
        core = w_full[len(lead):]
        if len(core) < 2:
            return None
        k = int(rng.integers(1, len(core)))  # keep 1..len(core)-1 chars
        prefix = self.tokenizer.text_to_ids(lead + core[:k])
        if not prefix or list(prefix) == list(last_word_ids):
            return None
        return list(prefix)

    def _sample_prefix_corrupt(self, chunks: List[ChunkSpec], rng) -> Optional[List[Optional[List[int]]]]:
        """Per-chunk corrupt_prev for the prefix heuristic: with prob prefix_prob, set
        chunk k's entry to a random char prefix of chunk k-1's last word (skip when the
        previous chunk produced no words). None entries = no corruption."""
        corrupt = [None] * len(chunks)
        for kc in range(1, len(chunks)):
            prev_lw = chunks[kc - 1].last_word_ids
            if not prev_lw:  # skip if the previous chunk had no output
                continue
            if rng.random() < self._sc_prefix_prob:
                wprime = self._prefix_word_ids(prev_lw, rng)
                if wprime:
                    corrupt[kc] = wprime
        return corrupt

    def _messages_to_chunks(self, messages: List[dict], compute_last_word: bool = False) -> List[ChunkSpec]:
        """Parse alternating user(audio)/assistant(words) turns into ChunkSpecs.

        ``messages[0]`` is the system prompt (used as the instruction elsewhere).
        Each user turn's content is ``audio_tag`` repeated once per frame; the
        following assistant turn holds the words revealed by that chunk (or the
        blank sentinel / empty string for a silent chunk). When ``compute_last_word``
        is True, each ChunkSpec also gets its last word's tokens (for recovery).
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
            last_word_ids = (
                self._last_word_ids(words, target_ids) if (compute_last_word and target_ids) else []
            )
            chunks.append(ChunkSpec(audio_len=audio_len, target_ids=target_ids, last_word_ids=last_word_ids))
        return chunks

    def _sample_recover_flags(self, chunks: List[ChunkSpec], rng) -> Optional[List[bool]]:
        """Per-chunk recovery flags: True where we drop the previous chunk's last
        word and recover it. None when recovery is disabled."""
        if self._history_word_recovery_prob <= 0.0:
            return None
        flags: List[bool] = []
        for kc, _ch in enumerate(chunks):
            eligible = kc >= 1 and len(chunks[kc - 1].last_word_ids) > 0
            flags.append(bool(eligible and rng.random() < self._history_word_recovery_prob))
        return flags

    def get_batch_data(
        self,
        cuts,
        audios: torch.Tensor,
        audio_lens: torch.Tensor,
        alignments: List[List[WordAlignment]],
        text: List[str],
    ) -> ScriptBatch:
        audio_durations_secs = (audio_lens.float() / self.cfg.sample_rate).tolist()

        # Draw one fixed chunk size per batch (multi chunk-size) or use the scalar.
        if self._chunk_size_candidates is not None:
            chunk_size = int(self._get_chunk_rng().choice(self._chunk_size_candidates))
        else:
            chunk_size = int(self.cfg.chunk_size)

        # Batch-shared delay + text-representation (whole batch shares one setting
        # so a single prompt is reused). Modes:
        #   * exact_delay: delay ~ Uniform[0, max]; the number is rendered into the
        #     prompt (with the optional cap/punct clause).
        #   * delay_prompts: sample one (delay, natural-language prompt) entry.
        #   * fixed: scalar num_delay_frames + per-cut prompt.
        # vary_text_repr adds a batch-shared (cap, punct) draw whose clause is put
        # in / appended to the prompt and whose transform is applied to the targets.
        rng = self._get_chunk_rng()
        cap, punct = self._sample_text_repr(rng)
        # Per-batch self-correction decision (forced variant only). When on, this
        # batch trains WITH the delete objective and states so in the prompt; when
        # off it is a plain SCRIPT batch with the "do not correct" clause.
        sc_on = bool(self._self_correction and rng.random() < self._self_correction_batch_prob)
        if self._multi_delay:
            if self._exact_delay:
                num_delay_frames = int(rng.integers(0, self._exact_max_delay + 1))
                system_prompts = [self._build_exact_prompt(num_delay_frames, cap, punct)] * len(cuts)
            else:
                num_delay_frames, forced_prompt = self._sample_delay_prompt(rng)
                if self._vary_text_repr:
                    forced_prompt = self._append_format_clause(forced_prompt, cap, punct)
                system_prompts = [forced_prompt] * len(cuts)
        else:
            num_delay_frames = self.cfg.num_delay_frames
            # Training base = train_system_prompt when set (system_prompt is then the
            # val-only fully-rendered prompt); else system_prompt. Per-cut overrides
            # via prompt_field still win.
            base_default = (
                self._train_system_prompt if self._train_system_prompt is not None else self.cfg.system_prompt
            )
            system_prompts = [cut.custom.get(self.cfg.prompt_field, base_default) for cut in cuts]
            if self._vary_text_repr:
                system_prompts = [self._append_format_clause(p, cap, punct) for p in system_prompts]

        # State the per-batch chunk size and the self-correction ON/OFF choice in the
        # prompt (no-ops when their templates/suffixes are unset -> backward compatible).
        if self._chunk_size_prompt_template:
            system_prompts = [self._append_chunk_clause(p, chunk_size) for p in system_prompts]
        if self._self_correction:
            system_prompts = [self._append_correction_clause(p, sc_on) for p in system_prompts]

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

        recover_on = self._history_word_recovery_prob > 0.0
        # Self-correction needs each chunk's last-word tokens (to know the correct word
        # to re-emit). The forced version also passes chunk metadata for the model;
        # the prefix version bakes the corruption in here (no chunk_meta). For the
        # forced variant this is per-batch: only the correction-ON batches carry
        # chunk_meta (and thus route to the model's forced-decode step).
        compute_last_word = recover_on or sc_on or self._self_correction_prefix
        transform_text = self._vary_text_repr and (not cap or not punct)
        examples = []
        chunk_meta = [] if sc_on else None
        for messages, sysp in zip(batch_messages, system_prompts):
            if transform_text:
                messages = self._apply_text_repr_to_messages(messages, cap, punct)
            chunks = self._messages_to_chunks(messages, compute_last_word=compute_last_word)
            recover_prev = self._sample_recover_flags(chunks, self._get_chunk_rng()) if recover_on else None
            # Prefix self-correction: truncate the previous chunk's last word in-place
            # (baked correction target), sampled per chunk.
            corrupt_prev = self._sample_prefix_corrupt(chunks, rng) if self._self_correction_prefix else None
            # Instruction/history separator: a newline keeps the first history word
            # from BPE-merging into the instruction text. The self-correction ON/OFF
            # clause is already baked into ``sysp`` (system_prompts) above.
            instruction_ids = self.tokenizer.text_to_ids(sysp + "\n")
            if chunk_meta is not None:
                chunk_meta.append((instruction_ids, chunks))
            if self._redecode:
                if self._redecode_train_prob >= 1.0:
                    include_mode = None
                else:
                    _r = self._get_chunk_rng()
                    _p = self._redecode_train_prob
                    include_mode = lambda c, j, _r=_r, _p=_p: _r.random() < _p
                examples.append(
                    build_packed_redecode_example(
                        instruction_ids=instruction_ids,
                        chunks=chunks,
                        vision_start_id=self.vision_start_id,
                        vision_end_id=self.vision_end_id,
                        eot_id=self.eot_id,
                        audio_history_chunks=self._audio_history_chunks,
                        redecode_depth=self._redecode_depth,
                        include_mode=include_mode,
                    )
                )
            elif self._shared_audio_track:
                examples.append(
                    build_shared_audio_chunk_example(
                        instruction_ids=instruction_ids,
                        chunks=chunks,
                        vision_end_id=self.vision_end_id,
                        eot_id=self.eot_id,
                        audio_window_frames=self._audio_window_frames,
                        audio_history_chunks=self._audio_history_chunks,
                    )
                )
            else:
                examples.append(
                    build_packed_chunk_example(
                        instruction_ids=instruction_ids,
                        chunks=chunks,
                        vision_start_id=self.vision_start_id,
                        vision_end_id=self.vision_end_id,
                        eot_id=self.eot_id,
                        audio_history_chunks=self._audio_history_chunks,
                        recover_prev=recover_prev,
                        contiguous_text_positions=self._contiguous_text_positions,
                        audio_window_frames=self._audio_window_frames,
                        corrupt_prev=corrupt_prev,
                        delete_id=self._sc_delete_id if self._self_correction_prefix else None,
                    )
                )

        if self._shared_audio_track:
            packed = collate_shared_audio_examples(examples, pad_id=self.tokenizer.pad_id)
            return ScriptBatch(
                audios=audios,
                audio_lens=audio_lens,
                input_tokens=packed.input_ids,
                position_ids=packed.position_ids,
                seg_ids=packed.seg_ids,
                prefix_len=packed.prefix_len,
                target_tokens=packed.target_ids,
                is_audio=packed.is_audio,
                audio_frame_index=packed.audio_frame_index,  # always set (audio laid once)
                win_start=packed.win_start,
                win_end=packed.win_end,
                valid=packed.valid,
                text=text,
                cuts=cuts,
                chunk_size=chunk_size,
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
            # Carry the explicit frame index whenever a window (chunk- or frame-based)
            # reuses frames across branches; otherwise the model keeps the
            # (byte-identical) cumsum interleave path.
            audio_frame_index=(
                packed.audio_frame_index
                if (self._audio_history_chunks > 0 or self._audio_window_frames > 0)
                else None
            ),
            valid=packed.valid,
            text=text,
            cuts=cuts,
            chunk_size=chunk_size,
            chunk_meta=chunk_meta,
        )
