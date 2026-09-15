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

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from nemo.collections.speechlm2.data.streaming_stt_dataset import (
    StreamingSTTDataConfig,
    StreamingSTTDataset,
)
from nemo.collections.speechlm2.parts.alignments import WordAlignment
from nemo.collections.speechlm2.parts.script import (
    ChunkSpec,
    build_packed_banded_example,
    build_packed_chunk_example,
    build_twod_chunk_example,
    collate_packed_banded_examples,
    collate_packed_chunk_examples,
    collate_twod_chunk_examples,
)
from nemo.collections.speechlm2.parts.script_messages import get_llm_messages_for_batch
from nemo.collections.speechlm2.parts.script_prompt import (
    render_control_prompt,
    resolve_delay_candidates,
    sample_controls,
)
from nemo.collections.speechlm2.parts.utils import to_dataclass
from nemo.utils import logging

# How often (in utterances, per dataloader worker) to report the rate at which
# target_construction='partition' fell back to per-chunk tokenization. A rate
# that climbs means the run is quietly training the legacy objective, which no
# other signal in the log would show.
_PARTITION_LOG_EVERY = 2000


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
        twod_layout: emit the 2-D layout (spine + branches on a batch axis)
            instead of one flat packed sequence. Mathematically identical -- see
            ``test_parity_twod_vs_flat`` -- but it never materialises the
            cross-branch attention pairs. Must match ``model.twod_layout``.
        chunk_size_seed: base seed for the per-batch chunk-size draw. Offset per
            dataloader worker so workers do not draw identical sequences.
        read_write: give every branch an EXPLICIT emit/no-emit gate. A silent
            chunk's branch predicts ``<read> <eot>``; a chunk that reveals words
            predicts ``<write> w_k <eot>``. Without it (the default) the decision
            is implicit -- a silent branch simply predicts ``<eot>`` first.
            The gate is BRANCH-ONLY: the spine still holds words alone, so the
            model does not condition on its own past gate decisions the way an
            interleaved SpeechLM does. Must match ``model.read_write``.
        gate_in_history: also put the gate token into the HISTORY, so the spine
            becomes the concatenation of what each branch emitted rather than
            words alone. This is what gives the model elapsed-time information:
            without it the history grows only with WORDS, so a branch cannot tell
            whether one chunk or fifty of silence preceded it. Costs one spine
            token per chunk, which bites hardest at small chunk sizes (a 30s clip
            at chunk_size=2 has ~188 chunks against ~110 word tokens). Requires
            ``read_write``. Must match ``model.gate_in_history``.
        full_context: OFFLINE upper bound. The utterance becomes ONE chunk: the
            LLM sees every encoder frame at once and predicts the whole
            transcript, instead of alternating chunk-by-chunk. The ENCODER is
            unchanged -- ``att_context_size`` still follows the sampled chunk
            size, so frames remain chunk-limited and streaming-equivalent. That
            makes this an ablation of the chunked TEXT structure alone: any gap
            to SCRIPT is the cost of emitting incrementally, not of restricted
            acoustic context.
        position_scheme: ``branch`` | ``continuous`` | ``sampled``. ``sampled``
            draws one of the two schemes PER BATCH (like chunk_size), with
            probability ``continuous_prob`` of ``continuous``. Note the two are
            not two views of one fact: under ``branch`` the offset between
            consecutive words reveals whether a chunk boundary fell between them,
            and ``continuous`` erases exactly that. Sampling therefore trades a
            usable cue for robustness to both layouts -- worth measuring, not
            obviously a win.
        continuous_prob: P(continuous) when ``position_scheme='sampled'``.
        position_seed: base seed for the per-batch scheme draw, offset per
            dataloader worker. Separate from the chunk and control seeds so
            enabling sampling does not perturb either of those streams.
        read_token / write_token: the two gate tokens. Defaults are unused
            in-vocab Qwen specials, so no embedding resize is needed and a
            read/write run can still warm-start from a plain SCRIPT checkpoint.
        prompt_control: train a PROMPT-CONTROLLED model. Capitalization,
            punctuation and the emission delay are drawn per example, the targets
            are restyled to match, and all four settings (including the batch's
            chunk size) are stated in that example's instruction. One checkpoint
            then serves every operating point. Off by default, so existing
            recipes and checkpoints are unaffected.
        delay_candidates: delays in frames to draw from when ``prompt_control``
            is on. ``None`` means always use ``num_delay_frames``.
        capitalization_prob: probability an example keeps its casing.
        punctuation_prob: probability an example keeps its punctuation.
        control_seed: base seed for the per-example control draw. Kept separate
            from ``chunk_size_seed`` so changing one does not reshuffle the other.
        respell_targets: locate an aligner word that is not a literal substring of
            the transcript by retrying with punctuation ignored, and anchor both
            searches to whole-word boundaries. The aligner ran on normalised text,
            so ``forty-eight`` arrives as ``fortyeight`` and ``U.S.`` as ``US``;
            without this 0.37% of words are either supervised several chunks late
            or mislocated inside a later word. Default False so an existing run
            that requeues keeps the objective it was launched with.
        target_construction: how each chunk's target ids are produced.
            ``legacy`` tokenizes every chunk's text on its own. The chunk TEXTS
            already tile the transcript, but their separate tokenizations need
            not concatenate to the tokenization of the whole: a cut inside a
            hyphen or apostrophe word makes ``forty-`` + ``eight`` differ from
            ``forty-eight``, and SentencePiece additionally emits a bare U+2581
            for the leading space (which ``_tokenize_target`` then has to strip).
            ``partition`` tokenizes the joined text ONCE and splits the id
            sequence at each chunk's character boundary, so the concatenated
            per-chunk ids equal the whole-text ids BY CONSTRUCTION. Verified per
            utterance, with a per-chunk fallback when a boundary does not split
            cleanly.
            This is also the precondition for a banded loss: a band moves the cut
            between adjacent chunks, and only under ``partition`` is the spine id
            sequence independent of where the cut falls -- under ``legacy``
            moving a word re-tokenizes both neighbours, so two paths reaching the
            same (chunk, cut) state do NOT share a spine prefix and the dynamic
            program is invalid.
            Default legacy so an existing run that requeues keeps the objective
            it was launched with.
    """

    audio_history_chunks: int = 0
    audio_window_frames: int = 0
    twod_layout: bool = False
    chunk_size_seed: int = 1234
    read_write: bool = False
    read_token: str = "<|box_start|>"
    write_token: str = "<|box_end|>"
    gate_in_history: bool = False
    position_scheme: str = "branch"
    full_context: bool = False
    continuous_prob: float = 0.5
    position_seed: int = 91011
    prompt_control: bool = False
    delay_candidates: Optional[List[int]] = None
    capitalization_prob: float = 0.5
    punctuation_prob: float = 0.5
    control_seed: int = 5678
    respell_targets: bool = False
    target_construction: str = "legacy"
    loss_type: str = "forced"
    band_words: int = 1
    band_side: str = "later"


@dataclass
class ScriptBatch:
    """A packed spine+branch batch for the SCRIPT SpeechLM.

    Attributes:
        audios / audio_lens: raw waveforms ``(B, T_samples)`` and sample counts ``(B,)``.
        input_tokens: (B, T) token ids; audio-frame slots hold ``AUDIO_TOKEN_IDX``.
        position_ids: (B, T) RoPE positions (spine index, or branch prefix+offset).
        order_ids: (B, T) structural indices used for masking only -- deliberately
            independent of position_ids, so a position scheme cannot change who
            attends to whom.
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
    order_ids: Optional[torch.Tensor] = None
    seg_ids: Optional[torch.Tensor] = None
    prefix_len: Optional[torch.Tensor] = None
    target_tokens: Optional[torch.Tensor] = None
    is_audio: Optional[torch.Tensor] = None
    audio_frame_index: Optional[torch.Tensor] = None
    valid: Optional[torch.Tensor] = None
    text: Optional[List[str]] = None
    cuts: Optional[object] = None
    chunk_size: Optional[int] = None
    # 2-D layout only (twod_layout=True); the flat fields above are then unset.
    twod: Optional[object] = None
    # Banded loss only (loss_type='banded'); `twod` is then unset.
    banded: Optional[object] = None


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
        self._twod_layout = bool(self.cfg.twod_layout)
        if self._audio_window_frames > 0 and self._audio_history_chunks > 0:
            logging.warning(
                "Both audio_window_frames=%d and audio_history_chunks=%d are set; "
                "the fixed-frame window takes precedence and audio_history_chunks is ignored.",
                self._audio_window_frames,
                self._audio_history_chunks,
            )

        hf_tok = self.tokenizer.tokenizer

        # Does a leading space become its own token? SentencePiece: yes (the
        # boundary is already in the piece). Byte-level BPE: no (folded in).
        self._leading_space_is_a_token = len(self.tokenizer.text_to_ids(" word")) > len(
            self.tokenizer.text_to_ids("word")
        )
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
        # Read/write gate ids, validated exactly like the audio delimiters: they
        # must already be single tokens in the vocabulary, so enabling the gate
        # never resizes the embedding table (which would break warm-starting
        # from a plain SCRIPT checkpoint).
        self._read_write = bool(self.cfg.read_write)
        self._gate_in_history = bool(self.cfg.gate_in_history)
        target_construction = str(self.cfg.target_construction or "legacy").lower()
        if target_construction not in ("legacy", "partition"):
            raise ValueError(
                f"target_construction must be 'legacy' or 'partition', got {self.cfg.target_construction!r}"
            )
        self._target_partition = target_construction == "partition"
        self._loss_type = str(self.cfg.loss_type or "forced").lower()
        if self._loss_type not in ("forced", "banded"):
            raise ValueError(f"loss_type must be 'forced' or 'banded', got {self.cfg.loss_type!r}")
        self._band_words = max(int(self.cfg.band_words), 0)
        self._band_side = str(self.cfg.band_side or "later").lower()
        if self._band_side not in ("both", "later", "earlier"):
            raise ValueError(f"band_side must be 'both', 'later' or 'earlier', got {self.cfg.band_side!r}")
        self._banded = self._loss_type == "banded"
        if self._banded and (self._twod_layout or not self._target_partition):
            raise ValueError(
                "loss_type='banded' requires twod_layout=false and target_construction='partition'; "
                f"got twod_layout={self._twod_layout}, target_construction={target_construction!r}"
            )
        self._word_start_ids = None
        self._partition_utts = 0
        self._partition_fallbacks = 0
        if self._gate_in_history and not self._read_write:
            raise ValueError(
                "gate_in_history=True requires read_write=True: without the gate there is no "
                "token to put in the history."
            )
        self.read_id = self.write_id = None
        if self._read_write:
            self.read_id = hf_tok.convert_tokens_to_ids(self.cfg.read_token)
            self.write_id = hf_tok.convert_tokens_to_ids(self.cfg.write_token)
            for name, tid, tok in (
                ("read_token", self.read_id, self.cfg.read_token),
                ("write_token", self.write_id, self.cfg.write_token),
            ):
                if tid is None or (unk is not None and tid == unk):
                    raise ValueError(
                        f"{name}={tok!r} is not a single in-vocabulary token for this tokenizer (got id={tid}). "
                        "Choose one that already exists in the vocab, or the embedding table would need resizing."
                    )
            if self.read_id == self.write_id:
                raise ValueError(f"read_token and write_token must differ; both are {self.cfg.read_token!r}")

        self.eot_id = hf_tok.eos_token_id
        if self.eot_id is None:
            raise ValueError("Tokenizer has no eos_token_id; it is required as the branch end-of-turn token.")

        # Per-worker RNGs: one for the per-batch chunk-size draw, one for the
        # per-example control draw. Separate streams so that turning prompt
        # control on does not perturb the chunk-size sequence.
        self._chunk_rngs: dict = {}
        self._control_rngs: dict = {}

        if self.cfg.position_scheme not in ("branch", "continuous", "sampled"):
            raise ValueError(
                f"position_scheme must be 'branch', 'continuous' or 'sampled', " f"got {self.cfg.position_scheme!r}"
            )
        if not 0.0 <= float(self.cfg.continuous_prob) <= 1.0:
            raise ValueError(f"continuous_prob must be in [0, 1], got {self.cfg.continuous_prob}")
        self._position_rngs: dict = {}
        if self.cfg.position_scheme == "sampled":
            logging.info(
                "ScriptSTTDataset: position scheme SAMPLED per batch — P(continuous)=%.2f",
                self.cfg.continuous_prob,
            )

        self._prompt_control = bool(self.cfg.prompt_control)
        self._delay_candidates = resolve_delay_candidates(self.cfg.delay_candidates, self.cfg.num_delay_frames)
        for name, p in (
            ("capitalization_prob", self.cfg.capitalization_prob),
            ("punctuation_prob", self.cfg.punctuation_prob),
        ):
            if not 0.0 <= float(p) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {p}")
        if not self._prompt_control and self.cfg.delay_candidates:
            logging.warning(
                "delay_candidates=%s is set but prompt_control is off; the delay stays fixed at "
                "num_delay_frames=%d. Set data.dataset.prompt_control=true to sample it.",
                list(self.cfg.delay_candidates),
                self.cfg.num_delay_frames,
            )
        if self._prompt_control:
            logging.info(
                "ScriptSTTDataset: prompt control ON — delays=%s, P(cap)=%.2f, P(punct)=%.2f",
                self._delay_candidates,
                self.cfg.capitalization_prob,
                self.cfg.punctuation_prob,
            )

        logging.info(
            "ScriptSTTDataset: audio delimiters %r=%d / %r=%d, eot_id=%d, "
            "audio_history_chunks=%d, audio_window_frames=%d, twod_layout=%s",
            self.audio_open_token,
            self.vision_start_id,
            self.audio_close_token,
            self.vision_end_id,
            self.eot_id,
            self._audio_history_chunks,
            self._audio_window_frames,
            self._twod_layout,
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

    def _get_position_rng(self) -> np.random.Generator:
        """RNG for the per-batch position-scheme draw, seeded per worker."""
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        if wid not in self._position_rngs:
            self._position_rngs[wid] = np.random.default_rng(int(self.cfg.position_seed) + wid)
        return self._position_rngs[wid]

    def _get_control_rng(self) -> np.random.Generator:
        """RNG for the per-example control draw, seeded per dataloader worker."""
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        if wid not in self._control_rngs:
            self._control_rngs[wid] = np.random.default_rng(int(self.cfg.control_seed) + wid)
        return self._control_rngs[wid]

    def _tokenize_target(self, text: str) -> List[int]:
        """Tokenize a chunk's words, without a redundant word-start token.

        SCRIPT marks a word start with a LEADING SPACE in the chunk text. Byte-
        level BPE folds that into the token (" station" -> one token, Gstation),
        but SentencePiece already encodes the boundary inside the piece
        (U+2581 st), so the leading space becomes a separate, contentless token:

            ' station' -> ['_', '_st', 'at', 'ion']     <- the bare '_' is spurious
            'station'  -> ['_st', 'at', 'ion']

        Trained on those targets the model learns to emit that bare separator, so
        decoding yields "The  station" with a doubled space. Beyond the cosmetics
        it wastes one token per chunk on a vocabulary already paying 1.62x in
        length, shifts every within-chunk word position by one, and permanently
        satisfies the force_word_start guard -- which then never fires when a
        chunk really would merge onto the previous word.

        So drop ONE leading space when, and only when, this tokenizer would turn
        it into its own token. Probed behaviourally rather than by tokenizer
        class, so it stays correct across families.
        """
        if self._leading_space_is_a_token and text.startswith(" "):
            text = text[1:]
        return self.tokenizer.text_to_ids(text)

    def _word_start_id_set(self) -> frozenset:
        """Token ids whose surface form begins a word, scanned once from the vocab.

        Both the byte-level BPE marker (``\u0120``) and the SentencePiece one
        (``\u2581``) are recognised, so this works across tokenizer families. A
        one-time vocabulary scan makes the per-token test O(1); asking the
        tokenizer to convert ids for every token of every utterance instead would
        dominate the collate.
        """
        if self._word_start_ids is None:
            hf_tok = self.tokenizer.tokenizer
            vocab = hf_tok.get_vocab()
            self._word_start_ids = frozenset(
                tid for piece, tid in vocab.items() if isinstance(piece, str) and piece[:1] in ("\u0120", "\u2581")
            )
        return self._word_start_ids

    def _word_start_positions(self, ids: List[int]) -> List[int]:
        """Positions in ``ids`` that begin a word. Position 0 always does."""
        ws = self._word_start_id_set()
        out = [0] if ids else []
        out.extend(i for i, t in enumerate(ids) if i > 0 and int(t) in ws)
        return out

    def _partition_target_ids(self, texts: List[str]) -> Optional[List[List[int]]]:
        """Split ONE tokenization of the joined chunk texts at the chunk bounds.

        Returns one id list per chunk, or ``None`` when the split is not exact --
        in which case the caller falls back to per-chunk tokenization for this
        utterance only.

        Two conditions are checked, and BOTH have bitten this project before:

        1. No boundary may fall mid-word. The chunk texts tile the transcript, so
           a boundary normally lands on the space that starts the next chunk. It
           does not when a whitespace-only chunk was blanked out upstream -- its
           characters leave the joined text, and the two neighbours fuse into
           ``wordAwordB``, which tokenizes as one word and trains the model to
           run them together. This is the byte-BPE form of the bug that produced
           'the bestselling singleby a Germanartist' on the CHAT side.
        2. Each prefix's ids must be a prefix of the whole's ids. This is
           VERIFIED rather than trusted: an earlier CHAT implementation took the
           split from SentencePiece's per-token character offsets, which worked
           locally and returned nothing in the training container, so every
           utterance silently took the fallback and the arm quietly trained the
           control objective. Only ``text_to_ids`` is used here, which every NeMo
           tokenizer has.

        Condition 1 additionally gives the banded loss a property CHAT's band
        lacks: CHAT's ``band_nodes`` sees only token counts, so its band permits
        cuts inside a word. Here a cut is always at a word boundary.
        """
        full = "".join(texts)
        if not full:
            return [[] for _ in texts]

        bounds: List[int] = []
        run = 0
        for t in texts:
            run += len(t)
            bounds.append(run)

        for b in bounds[:-1]:
            if b <= 0 or b >= len(full):
                continue
            if not (full[b - 1].isspace() or full[b].isspace()):
                return None  # a cut inside a word

        full_ids = self.tokenizer.text_to_ids(full)
        counts: List[int] = []
        for b in bounds:
            if b <= 0:
                counts.append(0)
                continue
            if b >= len(full):
                counts.append(len(full_ids))
                continue
            prefix = self.tokenizer.text_to_ids(full[:b])
            n = len(prefix)
            if n > len(full_ids) or list(full_ids[:n]) != list(prefix):
                return None
            counts.append(n)
        if counts != sorted(counts):
            return None

        out: List[List[int]] = []
        prev = 0
        for n in counts:
            out.append([int(i) for i in full_ids[prev:n]])
            prev = n
        return out

    def _messages_to_chunks(self, messages: List[dict]) -> List[ChunkSpec]:
        """Parse alternating user(audio)/assistant(words) turns into ChunkSpecs.

        ``messages[0]`` is the system prompt (used separately as the
        instruction). Each user turn's content is ``audio_tag`` repeated once per
        frame; the assistant turn that follows holds the words that chunk
        reveals, or the blank sentinel for a silent chunk.
        """
        audio_tag = self.cfg.audio_tag
        parsed: List[Tuple[int, str]] = []
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
            parsed.append((audio_len, words))

        texts = [w for _, w in parsed]
        per_chunk_ids: Optional[List[List[int]]] = None
        if self._target_partition:
            per_chunk_ids = self._partition_target_ids(texts)
            self._partition_utts += 1
            if per_chunk_ids is None:
                self._partition_fallbacks += 1
            if self._partition_utts % _PARTITION_LOG_EVERY == 0:
                logging.info(
                    "SCRIPT target partition: %d/%d utterances fell back to per-chunk " "tokenization (%.2f%%)",
                    self._partition_fallbacks,
                    self._partition_utts,
                    100.0 * self._partition_fallbacks / max(self._partition_utts, 1),
                )
        if per_chunk_ids is None:
            per_chunk_ids = [self._tokenize_target(w) if w.strip() else [] for w in texts]

        chunks: List[ChunkSpec] = []
        for (audio_len, _words), target_ids in zip(parsed, per_chunk_ids):
            # The gate goes on the BRANCH only; target_ids (which also feeds the
            # spine) stays the plain word sequence.
            gate = None
            if self._read_write:
                gate = self.write_id if target_ids else self.read_id
            chunks.append(ChunkSpec(audio_len=audio_len, target_ids=target_ids, gate_id=gate))
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

        # One position scheme per batch, drawn like the chunk size.
        position_scheme = self.cfg.position_scheme
        if position_scheme == "sampled":
            position_scheme = (
                "continuous" if self._get_position_rng().random() < float(self.cfg.continuous_prob) else "branch"
            )

        system_prompts = [cut.custom.get(self.cfg.prompt_field, self.cfg.system_prompt) for cut in cuts]

        # Prompt control: draw each example's settings, restyle its targets to
        # match, and state all four in its own instruction. Off -> the batch
        # shares the configured delay and the transcript's own style, and the
        # prompt is left exactly as before.
        if self._prompt_control:
            rng = self._get_control_rng()
            controls = [
                sample_controls(
                    rng,
                    chunk_size=chunk_size,
                    delay_candidates=self._delay_candidates,
                    cap_prob=float(self.cfg.capitalization_prob),
                    punct_prob=float(self.cfg.punctuation_prob),
                )
                for _ in cuts
            ]
            system_prompts = [render_control_prompt(p, c) for p, c in zip(system_prompts, controls)]
            delays = [c.num_delay_frames for c in controls]
            caps = [c.capitalization for c in controls]
            puncts = [c.punctuation for c in controls]
        else:
            delays, caps, puncts = self.cfg.num_delay_frames, True, True

        batch_messages = get_llm_messages_for_batch(
            respell=self.cfg.respell_targets,
            system_role=self.cfg.system_role,
            system_prompt=system_prompts,
            audio_tag=self.cfg.audio_tag,
            blank_token=self.cfg.blank_token,
            chunk_size=chunk_size,
            num_delay_frames=delays,
            audio_durations_secs=audio_durations_secs,
            frame_length_in_secs=self.cfg.frame_length_in_secs,
            alignments=alignments,
            transcripts=text,
            capitalization=caps,
            punctuation=puncts,
        )

        builder = build_twod_chunk_example if self._twod_layout else build_packed_chunk_example
        examples = []
        for bi, (messages, sysp) in enumerate(zip(batch_messages, system_prompts)):
            # Instruction/history separator: the trailing newline keeps the first
            # history word from BPE-merging into the instruction text.
            instruction_ids = self.tokenizer.text_to_ids(sysp + "\n")
            if self.cfg.full_context:
                # ONE chunk: every frame, the whole transcript. The batch's
                # chunk_size is left untouched so the ENCODER keeps its
                # chunk-limited look-ahead -- only the LLM-side layout changes.
                n_frames = math.ceil(audio_durations_secs[bi] / self.cfg.frame_length_in_secs)
                full_text = text[bi] if text is not None else ""
                chunks = [
                    ChunkSpec(
                        audio_len=max(1, int(n_frames)),
                        target_ids=self._tokenize_target(full_text) if full_text.strip() else [],
                    )
                ]
            else:
                chunks = self._messages_to_chunks(messages)
            if self._banded:
                transcript_ids = [t for ch in chunks for t in ch.target_ids]
                examples.append(
                    build_packed_banded_example(
                        instruction_ids=instruction_ids,
                        chunks=chunks,
                        word_starts=self._word_start_positions(transcript_ids),
                        band_words=self._band_words,
                        band_side=self._band_side,
                        vision_start_id=self.vision_start_id,
                        vision_end_id=self.vision_end_id,
                        eot_id=self.eot_id,
                        audio_history_chunks=self._audio_history_chunks,
                        audio_window_frames=self._audio_window_frames,
                        position_scheme=position_scheme,
                    )
                )
                continue
            examples.append(
                builder(
                    instruction_ids=instruction_ids,
                    chunks=chunks,
                    vision_start_id=self.vision_start_id,
                    vision_end_id=self.vision_end_id,
                    eot_id=self.eot_id,
                    audio_history_chunks=self._audio_history_chunks,
                    audio_window_frames=self._audio_window_frames,
                    gate_in_history=self._gate_in_history,
                    position_scheme=position_scheme,
                )
            )

        if self._banded:
            # The banded batch fills the ORDINARY flat fields, so the model reuses
            # _script_input_embeds and _training_attention untouched; `banded`
            # carries only the lattice the loss needs on top.
            bnd = collate_packed_banded_examples(examples, pad_id=self.tokenizer.pad_id)
            return ScriptBatch(
                audios=audios,
                audio_lens=audio_lens,
                input_tokens=bnd.input_ids,
                position_ids=bnd.position_ids,
                order_ids=bnd.order_ids,
                seg_ids=bnd.seg_ids,
                prefix_len=bnd.prefix_len,
                target_tokens=bnd.target_ids,
                is_audio=bnd.is_audio,
                audio_frame_index=bnd.audio_frame_index,
                valid=bnd.valid,
                banded=bnd,
                text=text,
                cuts=cuts,
                chunk_size=chunk_size,
            )

        if self._twod_layout:
            return ScriptBatch(
                audios=audios,
                audio_lens=audio_lens,
                twod=collate_twod_chunk_examples(examples, pad_id=self.tokenizer.pad_id),
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
            order_ids=packed.order_ids,
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
