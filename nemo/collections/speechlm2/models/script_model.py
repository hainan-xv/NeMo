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
"""SCRIPT streaming SpeechLM — packed spine + per-chunk branches."""

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.tensor.parallel import loss_parallel
from transformers import GenerationConfig

from nemo.collections.speechlm2.data.script_dataset import ScriptBatch, ScriptSTTDataset
from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX, IGNORE_INDEX
from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel, StreamingSTTModelConfig
from nemo.collections.speechlm2.parts.metrics.wer import WER
from nemo.collections.speechlm2.parts.optim_setup import is_frozen
from nemo.collections.speechlm2.parts.script import (
    batched_stream_decode_script,
    broadcast_spine_cache,
    build_script_mask,
    build_spine_mask,
    build_twod_branch_mask,
)
from nemo.collections.speechlm2.parts.script_attention import script_attention_plan
from nemo.collections.speechlm2.parts.script_banded import NEG_INF, banded_forward, span_scores
from nemo.collections.speechlm2.parts.script_fsm import fsm_stream_decode_script, streaming_encode_frames
from nemo.collections.speechlm2.parts.script_prompt import (
    ScriptControls,
    apply_text_style,
    render_control_prompt,
)
from nemo.collections.speechlm2.parts.utils import to_dataclass
from nemo.utils import logging


def _maybe_plan(plan):
    """No-op context when the structured backend is not in use."""
    return script_attention_plan(plan)


@dataclass
class ScriptSTTModelConfig(StreamingSTTModelConfig):
    """:class:`StreamingSTTModelConfig` plus SCRIPT's own knobs.

    Attributes:
        audio_history_chunks: ``M`` — how many PREVIOUS chunks' audio each branch
            also sees. MUST equal ``data.dataset.audio_history_chunks``, since
            training and inference build the window from this same number.
        audio_window_frames: ``F`` — if ``> 0``, every branch gets a FIXED window
            of ``F`` frames ending at its chunk boundary, so the acoustic context
            is constant regardless of chunk size. Takes precedence over
            ``audio_history_chunks``. MUST equal ``data.dataset.audio_window_frames``.
        twod_layout: train with the 2-D layout (spine forwarded once, branches on
            a batch axis) instead of one flat packed sequence. Provably identical
            -- see ``test_parity_twod_vs_flat`` and its gradient counterpart --
            but it never creates the cross-branch attention pairs the flat mask
            only exists to forbid. Must match ``data.dataset.twod_layout``.
        twod_branch_micro_batch: with ``twod_layout``, process at most this many
            branches at a time, recomputing each group's activations in backward
            (``torch.utils.checkpoint``). Because branches are a BATCH axis rather
            than one long sequence, this makes activation memory a function of the
            micro-batch instead of the utterance length -- the flat layout cannot
            do this, since its branches are a single inseparable sequence. ``0``
            (default) processes every branch at once. The loss is unchanged:
            each group contributes a SUM and the batch-wide target count is the
            single denominator.
        bidirectional_audio: let each branch's audio block attend ITSELF both ways,
            so an early frame in a chunk sees a later one. Streaming-legal: the
            whole block has already arrived when the branch starts, and the model
            waits for the chunk boundary either way. It costs no extra FLOPs (the
            block is computed and then masked regardless) and makes the LLM-side
            rule agree with the encoder, which is already bidirectional within a
            chunk under `chunked_limited`. Text after the audio stays causal.
            NOTE: the LLM was pretrained strictly causally and has never seen a
            negative relative RoPE offset, so warm-starting into this rule is a
            genuine distribution shift the fine-tune has to absorb.
        attn_backend: how the SCRIPT mask is applied during TRAINING.
            ``"dense"`` builds the full ``(B, 1, T, T)`` additive mask -- correct
            but it materialises a score tensor that is ~98% masked out, and at
            long T the run is bandwidth-bound on it.
            ``"flex"`` expresses the same rule as a FlexAttention predicate, so
            fully-masked 128x128 blocks are skipped and no mask is materialised.
            Measured at T=11001: 7.78s -> 1.12s per step and 37.3 -> 26.4 GiB.
            ``"script"`` uses the structured decomposition
            (:mod:`...parts.script_attention`) -- also exact, slightly slower than
            flex, but needs no ``torch.compile``.
            All three compute the SAME function; decoding always uses SDPA.
        activation_checkpointing: recompute LLM layer activations in backward.
            Roughly halves activation memory for ~30% more compute, and is what
            makes the long-sequence configurations fit at all.
        val_chunk_size: chunk size used for the decode-only validation pass when
            training with multiple chunk sizes. Defaults to 14 when available,
            else the largest candidate.
        val_max_new_tokens_per_chunk: cap on tokens decoded per chunk during
            validation. Defaults to the validation chunk size.
        val_system_prompt: instruction used at validation. Defaults to the
            dataset's ``system_prompt``; set explicitly to pin a fixed operating
            point that matches training.
        val_prompt_field: per-cut field that may override ``val_system_prompt``.
        max_history_tokens: if ``> 0``, cap the conditioning history at inference
            to the most recent N emitted tokens (the instruction is always kept),
            making per-chunk cost linear rather than quadratic in duration.
        encode_batch_size: sub-batch size for the offline encode. A single
            full-batch encode of long, length-sorted clips can overflow 32-bit
            CUDA indexing in the subsampling convolution.
        force_word_start: insert a leading-space token when a chunk's first
            decoded token is not a word start, so the chunk cannot merge onto the
            previous chunk's last word. Overridable per ``generate`` call.
        log_detailed_train_metrics: also log sequence length / target counts.
        read_write: the branches carry an EXPLICIT emit/no-emit gate -- a silent
            chunk predicts ``<read> <eot>``, an emitting chunk ``<write> w_k <eot>``.
            The gate is stripped at decode so it never enters the history, which
            stays the running transcript. MUST match ``data.dataset.read_write``.
        gate_in_history: keep the gate in the conditioning history, so the spine
            is the concatenation of what each branch emitted. This is what gives
            the model elapsed-time information -- otherwise the history grows only
            with words and a branch cannot tell how long a silence lasted. The
            gate is still stripped from the returned TEXT. Requires ``read_write``
            and MUST match ``data.dataset.gate_in_history``.
        full_context: OFFLINE upper bound -- the LLM sees every encoder frame at
            once and emits the whole transcript in one turn. The ENCODER is
            unchanged (``att_context_size`` still follows the chunk size), so this
            ablates the chunked TEXT structure alone. MUST match
            ``data.dataset.full_context``. Note a full transcript needs a much
            larger ``max_new_tokens`` than a chunk does.
        position_scheme: ``branch`` | ``continuous`` | ``sampled``. MUST match
            ``data.dataset.position_scheme``.
        val_position_scheme: which concrete layout to DECODE with when the model
            was trained with ``sampled``. Decoding has no notion of sampling.
        read_token / write_token: the gate tokens. Defaults are unused in-vocab
            Qwen specials, so no embedding resize is needed. MUST match the
            dataset's.
        prompt_control: the model was trained PROMPT-CONTROLLED — its instruction
            states the chunk size, emission delay, capitalization and punctuation.
            When on, :meth:`generate` renders those settings into the prompt
            through the same function the dataset uses, so decoding cannot drift
            out of distribution by wording the instruction differently. MUST match
            ``data.dataset.prompt_control``.
        val_num_delay_frames / val_capitalization / val_punctuation: the operating
            point validation decodes at, used only when ``prompt_control`` is on.
            References are restyled to match, or the WER would penalise the model
            for honouring the style it was asked for.
    """

    audio_history_chunks: int = 0
    audio_window_frames: int = 0
    twod_layout: bool = False
    twod_branch_micro_batch: int = 0
    attn_backend: str = "dense"
    activation_checkpointing: bool = False
    bidirectional_audio: bool = False
    prompt_control: bool = False
    read_write: bool = False
    read_token: str = "<|box_start|>"
    write_token: str = "<|box_end|>"
    gate_in_history: bool = False
    position_scheme: str = "branch"
    full_context: bool = False
    # Provenance only -- the target construction is a pure DATASET concern, but
    # save_hyperparameters() stamps cfg.model into every checkpoint and nothing
    # in cfg.data.dataset is recorded there. Without this, a .nemo cannot be told
    # apart from one trained on the old targets. script_train.py's paired assert
    # keeps the two sides from drifting.
    respell_targets: bool = False
    target_construction: str = "legacy"
    # BANDED LOSS. "forced" (default) trains cross-entropy over the single
    # word-to-chunk assignment the aligner chose. "banded" marginalises over every
    # assignment whose chunk boundaries sit within band_words words of it, the way
    # CHAT's banded RNN-T does -- so a word the aligner placed a chunk early is no
    # longer scored as an error the model must reproduce.
    loss_type: str = "forced"
    band_words: int = 1
    # ONE-SIDED by default. "later" lets a word the aligner placed in chunk t be
    # emitted in t+1 instead, never earlier -- which is the direction aligner error
    # actually needs, since a word whose audio finishes just after a boundary
    # cannot legitimately be emitted before that audio arrives. It also costs a
    # THIRD less than a two-sided band: candidates per chunk drop 3 -> 2, and the
    # packed sequence scales with that count.
    band_side: str = "later"
    # Consecutive OOM batches tolerated before training_step re-raises. A few
    # skips are a rare bad draw (chunk_size is sampled per batch while
    # bucket_batch_size is keyed on duration only); a streak means the batch
    # simply does not fit, and limping on would train on a quietly easier
    # distribution -- the batches that OOM are the long-audio, small-chunk ones.
    oom_skip_limit: int = 25
    val_position_scheme: str = "continuous"
    val_chunk_size: Optional[int] = None
    val_max_new_tokens_per_chunk: Optional[int] = None
    val_system_prompt: Optional[str] = None
    val_prompt_field: str = "system_prompt"
    val_num_delay_frames: int = 3
    val_capitalization: bool = True
    val_punctuation: bool = True
    max_history_tokens: int = 0
    encode_batch_size: int = 8
    force_word_start: bool = True
    log_detailed_train_metrics: bool = False
    # TRAINING WER, under the ASR collection's name so one wandb panel carries both
    # families. It is a REAL autoregressive decode, matching what CHAT's
    # training_batch_wer measures -- a teacher-forced argmax would be cheaper but
    # would not be the same statistic, and logging it under the same name is the
    # kind of false comparability this exists to remove.
    #
    # Decoding is expensive for an LLM, so it is bounded on both axes: every
    # train_wer_every_n_steps steps, over at most train_wer_max_utts utterances.
    # At the defaults that is ~4 short decodes per 500 steps, well under 1% of
    # step time. 0 disables.
    train_wer_every_n_steps: int = 500
    train_wer_max_utts: int = 4


class ScriptSTTModel(StreamingSTTModel):
    """Streaming ASR as conditional text completion.

    Each utterance is packed as a pure-text **spine** (the instruction plus every
    word, in order) followed by one **branch** per audio chunk
    (``<vs> audio_k <ve> w_k <eot>``). A 4D mask
    (:func:`~nemo.collections.speechlm2.parts.script.build_script_mask`) keeps
    each branch attending only its own history prefix of the spine, its own
    audio, and its own earlier tokens — so the whole utterance trains in a single
    O(L) forward while every chunk still sees exactly the conditioning it would
    see standing alone: ``p(words_k | text_history_<k, audio_k)``.

    Inference mirrors that conditioning chunk by chunk, re-prefilling the compact
    text history and attaching only the current chunk's audio window.

    **On "offline" encoding.** Validation and inference encode the whole
    utterance in one ``perception`` call and then slice frames per chunk. That is
    a choice about *how* the representation is computed, not a relaxation of the
    streaming constraint: :meth:`encode_frames` pins the encoder's right context
    to ``chunk_size - 1`` for chunk-limited encoders, so a frame's receptive
    field never crosses its own chunk boundary. The dependency structure is
    identical to true frame-by-frame streaming; only the batching differs.
    ``test_offline_encode_dependency_is_chunk_limited`` pins this down by
    perturbing future audio and asserting earlier frames do not move.
    """

    def __init__(
        self,
        cfg: dict,
        forced_aligner=None,
        data_cfg=None,
        val_data_cfg=None,
        dataset_cls=ScriptSTTDataset,
    ) -> None:
        super().__init__(
            cfg,
            forced_aligner=forced_aligner,
            data_cfg=data_cfg,
            val_data_cfg=val_data_cfg,
            dataset_cls=dataset_cls,
        )
        # The base __init__ coerces cfg through StreamingSTTModelConfig, which
        # silently drops SCRIPT's extra keys. Re-coerce through the extended
        # dataclass so they survive (and stay typed).
        self.core_cfg: ScriptSTTModelConfig = to_dataclass(ScriptSTTModelConfig, cfg)

        self._audio_history_chunks = max(int(self.core_cfg.audio_history_chunks), 0)
        self._audio_window_frames = max(int(self.core_cfg.audio_window_frames), 0)
        self._twod_layout = bool(self.core_cfg.twod_layout)
        self._loss_type = str(self.core_cfg.loss_type or "forced").lower()
        if self._loss_type not in ("forced", "banded"):
            raise ValueError(f"loss_type must be 'forced' or 'banded', got {self.core_cfg.loss_type!r}")
        self._band_words = max(int(self.core_cfg.band_words), 0)
        if self._loss_type == "banded":
            # Each of these makes the (chunk, cut) dynamic program invalid rather
            # than merely different, so they are refused instead of worked around.
            if self._twod_layout:
                raise ValueError(
                    "loss_type='banded' requires twod_layout=false. The band emits one branch "
                    "SEGMENT per candidate cut in the flat packed layout, which needs no mask "
                    "change because build_script_mask already gives each segment its own history "
                    "via prefix_len. Measured on one node, the 2-D path costs 5.11 s/step against "
                    "flat's 0.55 s/step for the same objective, because transformers' "
                    "DynamicLayer.update materialises the broadcast spine cache per branch."
                )
            if str(self.core_cfg.target_construction or "legacy").lower() != "partition":
                raise ValueError(
                    "loss_type='banded' requires target_construction='partition'. Under the legacy "
                    "per-chunk tokenization the spine ids change with where the cut falls, so two "
                    "paths reaching the same (chunk, cut) do not share a history prefix and the "
                    "dynamic program is not valid."
                )
            if bool(self.core_cfg.gate_in_history):
                raise ValueError(
                    "loss_type='banded' is incompatible with gate_in_history=true: the gate written "
                    "into the spine is 'write if the chunk emitted anything else read', so it flips "
                    "exactly when the band empties a chunk -- which would make it part of the DP state."
                )
            if bool(self.core_cfg.read_write):
                raise ValueError(
                    "loss_type='banded' is incompatible with read_write=true: the branch gate's "
                    "identity depends on whether the chunk is empty, which varies across the span "
                    "lengths a single branch scores."
                )

        self._bidirectional_audio = bool(self.core_cfg.bidirectional_audio)
        self._attn_backend = str(self.core_cfg.attn_backend or "dense").lower()
        if self._attn_backend not in ("dense", "flex", "script"):
            raise ValueError(f"attn_backend must be dense|flex|script, got {self._attn_backend!r}")
        if self._attn_backend == "script":
            from nemo.collections.speechlm2.parts.script_attention import register_script_attention

            register_script_attention()
        if self.core_cfg.activation_checkpointing:
            base = self.llm.get_base_model() if hasattr(self.llm, "get_base_model") else self.llm
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            logging.info("ScriptSTTModel: activation checkpointing enabled on the LLM")

        # Audio-span delimiters and end-of-turn token, resolved once. These must
        # match ScriptSTTDataset, which builds the training layout with them.
        hf_tok = self.tokenizer.tokenizer
        self._vision_start_id = hf_tok.convert_tokens_to_ids(ScriptSTTDataset.audio_open_token)
        self._vision_end_id = hf_tok.convert_tokens_to_ids(ScriptSTTDataset.audio_close_token)
        self._eot_id = hf_tok.eos_token_id
        if self._eot_id is None:
            raise ValueError("Tokenizer has no eos_token_id; it is required as the branch end-of-turn token.")

        # Read/write gate. Resolved here so decode strips exactly the ids the
        # dataset supervised; a mismatch would leave gate tokens in the output.
        # Decoding needs a CONCRETE layout: "sampled" is a training-time
        # augmentation and has no meaning for a single forward pass. Resolve it
        # here so the choice is explicit and logged, rather than defaulting
        # silently to whichever branch of an if-statement runs first.
        if self.core_cfg.position_scheme not in ("branch", "continuous", "sampled"):
            raise ValueError(f"model.position_scheme={self.core_cfg.position_scheme!r} is not valid")
        self._decode_position_scheme = self.core_cfg.position_scheme
        if self._decode_position_scheme == "sampled":
            self._decode_position_scheme = self.core_cfg.val_position_scheme
            if self._decode_position_scheme not in ("branch", "continuous"):
                raise ValueError(
                    f"val_position_scheme must be 'branch' or 'continuous' when position_scheme="
                    f"'sampled', got {self.core_cfg.val_position_scheme!r}"
                )
            logging.info(
                "ScriptSTTModel: trained with SAMPLED positions; decoding with %r",
                self._decode_position_scheme,
            )

        self._read_write = bool(self.core_cfg.read_write)
        self._gate_in_history = bool(self.core_cfg.gate_in_history)
        if self._gate_in_history and not self._read_write:
            raise ValueError("model.gate_in_history=True requires model.read_write=True")
        self._read_id = self._write_id = None
        if self._read_write:
            self._read_id = hf_tok.convert_tokens_to_ids(self.core_cfg.read_token)
            self._write_id = hf_tok.convert_tokens_to_ids(self.core_cfg.write_token)
            unk = getattr(hf_tok, "unk_token_id", None)
            for name, tid, tok in (
                ("read_token", self._read_id, self.core_cfg.read_token),
                ("write_token", self._write_id, self.core_cfg.write_token),
            ):
                if tid is None or (unk is not None and tid == unk):
                    raise ValueError(f"model.{name}={tok!r} is not a single in-vocabulary token (got id={tid}).")
            if self._read_id == self._write_id:
                raise ValueError("model.read_token and model.write_token must differ")
            logging.info(
                "ScriptSTTModel: read/write gate ON — read=%r(%d) write=%r(%d)",
                self.core_cfg.read_token,
                self._read_id,
                self.core_cfg.write_token,
                self._write_id,
            )

        # Lazily resolved leading-space token used to guarantee a word boundary at
        # the start of a chunk; see _get_word_start_insert_id.
        self._word_start_insert_id: Optional[int] = None

        self._val_system_prompt = self.core_cfg.val_system_prompt
        if self._val_system_prompt is None and data_cfg is not None:
            self._val_system_prompt = data_cfg.get("system_prompt", None)
        if self._val_system_prompt is None:
            self._val_system_prompt = "Transcribe the audio into text."

        logging.info(
            "ScriptSTTModel: audio delimiters %d / %d, eot_id=%d, " "audio_history_chunks=%d, audio_window_frames=%d",
            self._vision_start_id,
            self._vision_end_id,
            self._eot_id,
            self._audio_history_chunks,
            self._audio_window_frames,
        )

    # ------------------------------------------------------------------
    # Input construction
    # ------------------------------------------------------------------

    def _build_input_embeds_indexed(
        self, input_tokens: Tensor, audios: Tensor, audio_lens: Tensor, audio_frame_index: Tensor
    ) -> Tensor:
        """Fill audio slots by EXPLICIT global frame index rather than by cumsum.

        Needed when ``audio_history_chunks > 0``: a branch's window spans several
        chunks and the same encoder frame appears in more than one branch, so the
        1:1 positional mapping the cumsum fill assumes no longer holds.
        Out-of-range indices (the final chunk's ceiling past the real audio)
        gather a zero-padded frame, which is exactly what the decoder pads to.
        """
        audio_mask = input_tokens == AUDIO_TOKEN_IDX  # (B, L)
        text_tokens = input_tokens.where(~audio_mask, torch.zeros_like(input_tokens))
        text_embeds = self._embed_tokens(text_tokens)  # (B, L, H)
        audio_embs, _ = self.perception(input_signal=audios, input_signal_length=audio_lens)  # (B, T_enc, H)

        B, L = input_tokens.shape
        H = audio_embs.shape[2]
        T_enc = audio_embs.shape[1]
        max_idx = int(audio_frame_index.max().item()) if audio_frame_index.numel() else -1
        if max_idx >= T_enc:
            audio_embs = F.pad(audio_embs, (0, 0, 0, max_idx - T_enc + 1))
        gather_idx = audio_frame_index.clamp(min=0).unsqueeze(-1).expand(B, L, H)
        audio_at = torch.gather(audio_embs, dim=1, index=gather_idx)  # (B, L, H)
        return torch.where(audio_mask.unsqueeze(-1), audio_at, text_embeds)

    def _script_input_embeds(self, batch: ScriptBatch) -> Tensor:
        """Interleave encoder frames into the packed ``AUDIO_TOKEN_IDX`` slots."""
        if batch.audio_frame_index is not None:
            return self._build_input_embeds_indexed(
                batch.input_tokens, batch.audios, batch.audio_lens, batch.audio_frame_index
            )
        return self._build_input_embeds(batch.input_tokens, batch.audios, batch.audio_lens)["input_embeds"]

    # ------------------------------------------------------------------
    # Attention backend
    # ------------------------------------------------------------------

    @contextmanager
    def _attn_implementation(self, name: str):
        """Temporarily switch the LLM's attention backend.

        Training may use flex/structured attention, but DECODING must not: the
        decode loop passes an ordinary 2-D padding mask, which those backends do
        not interpret. Scoping the switch to the training forward keeps
        ``generate()`` on plain SDPA without a second model.
        """
        llm = self.llm.get_base_model() if hasattr(self.llm, "get_base_model") else self.llm
        prev = getattr(llm.config, "_attn_implementation", None)
        if prev == name:
            yield
            return
        llm.set_attn_implementation(name)
        try:
            yield
        finally:
            if prev is not None:
                llm.set_attn_implementation(prev)

    @staticmethod
    def _script_mask_mod(batch: ScriptBatch, bidirectional_audio: bool = False):
        """The SCRIPT rule as a FlexAttention predicate.

        A direct transcription of :func:`build_script_mask`; the equality of the
        two is asserted in the tests.
        """
        # order_ids, NOT position_ids: masking is structural.
        if batch.order_ids is None:
            raise ValueError(
                "ScriptBatch.order_ids is None -- the dataset did not populate the structural "
                "indices the mask is built from. Without them the mask would silently fall back "
                "to RoPE geometry, which is exactly what position schemes are allowed to change."
            )
        seg, pos, pref, val = batch.seg_ids, batch.order_ids, batch.prefix_len, batch.valid
        aud = batch.is_audio if bidirectional_audio else None
        if bidirectional_audio and aud is None:
            raise ValueError(
                "bidirectional_audio is set but ScriptBatch.is_audio is None -- without it the "
                "flex predicate would silently fall back to the causal rule while the dense path "
                "used the bidirectional one, and the two backends would train different models."
            )

        def mask_mod(b, h, q, kv):
            qs, ks = seg[b, q], seg[b, kv]
            qp, kp = pos[b, q], pos[b, kv]
            q_spine, k_spine = qs == 0, ks == 0
            causal = kp <= qp
            same_branch = (qs == ks) & (~q_spine)
            own = same_branch & causal
            if aud is not None:
                own = own | (same_branch & aud[b, q] & aud[b, kv])
            return ((q_spine & k_spine & causal) | ((~q_spine) & k_spine & (kp < pref[b, q])) | own) & val[b, kv]

        return mask_mod

    def _reject_fsm_with_bidirectional_audio(self, state_machine: bool) -> None:
        """The FSM decode path cannot implement bidirectional audio.

        It ingests the window ONE FRAME AT A TIME behind a KV cache, so an early
        frame can never see a later one whatever mask it is handed -- that is the
        point of the state machine. Silently decoding a bidirectionally-trained
        model that way is a train/decode mismatch that would surface only as an
        unexplained WER gap, so refuse instead.
        """
        if state_machine and self._bidirectional_audio:
            raise ValueError(
                "use_state_machine_inference is incompatible with bidirectional_audio: the FSM "
                "ingests the window one frame at a time behind a KV cache, so an early frame can "
                "never see a later one no matter what mask it is given. Use the default decode "
                "path, or train with bidirectional_audio=false."
            )

    def _training_attention(self, batch: ScriptBatch, dtype):
        """(attn_implementation, attention_mask) for this batch's backend."""
        if self._bidirectional_audio and batch.is_audio is None:
            raise ValueError(
                "bidirectional_audio is set but ScriptBatch.is_audio is None -- the mask would "
                "silently fall back to the causal rule, so the model would train under a rule it "
                "was not configured for and nothing would say so."
            )
        if self._attn_backend == "flex":
            from torch.nn.attention.flex_attention import create_block_mask

            B, T = batch.seg_ids.shape
            block_mask = create_block_mask(
                self._script_mask_mod(batch, self._bidirectional_audio),
                B=B,
                H=None,
                Q_LEN=T,
                KV_LEN=T,
                device=batch.seg_ids.device,
            )
            return "flex_attention", block_mask
        if self._attn_backend == "script":
            return "script", None
        return "eager", build_script_mask(
            batch.seg_ids,
            batch.order_ids,
            batch.prefix_len,
            batch.valid,
            dtype,
            is_audio=batch.is_audio if self._bidirectional_audio else None,
        )

    # ------------------------------------------------------------------
    # 2-D layout: spine forwarded once, branches on a batch axis
    # ------------------------------------------------------------------

    def _twod_branch_embeds(self, branch_ids: Tensor, branch_frame_index: Tensor, utt_frames: Tensor) -> Tensor:
        """Embed one utterance's branches, splicing in its encoder frames.

        Args:
            branch_ids: (N, b) tokens; audio slots hold ``AUDIO_TOKEN_IDX``.
            branch_frame_index: (N, b) global frame index at audio slots, -1 elsewhere.
            utt_frames: (T_enc, H) this utterance's encoder output.
        """
        audio_mask = branch_ids == AUDIO_TOKEN_IDX
        text_ids = branch_ids.where(~audio_mask, torch.zeros_like(branch_ids))
        embeds = self._embed_tokens(text_ids)  # (N, b, H)
        # A branch window may run past the real audio on the final chunk; those
        # slots gather a zero frame, matching the flat path's padded gather.
        need = int(branch_frame_index.max().item()) + 1
        if need > utt_frames.shape[0]:
            utt_frames = F.pad(utt_frames, (0, 0, 0, need - utt_frames.shape[0]))
        gathered = utt_frames[branch_frame_index.clamp(min=0)]  # (N, b, H)
        return torch.where(audio_mask.unsqueeze(-1), gathered, embeds)

    def _twod_spine_cache(self, two, dtype):
        """Forward every utterance's spine once and return the shared K/V cache.

        The spine is plain causal text -- exactly its role in the flat layout,
        where spine tokens never attend audio.
        """
        spine_embeds = self._embed_tokens(two.spine_ids)
        out = self._llm_forward(
            inputs_embeds=spine_embeds,
            attention_mask=build_spine_mask(two.spine_positions, two.spine_valid, dtype),
            position_ids=two.spine_positions,
            use_cache=True,
            return_dict=True,
        )
        return out.past_key_values

    def _branch_loss_sum(self, two, audio_embs, cache, dtype, i: int, lo: int, hi: int) -> Tensor:
        """Summed cross-entropy for branches ``[lo, hi)`` of utterance ``i``.

        Returning the SUM (not the mean) lets the caller divide once by the
        batch-wide target count, so any micro-batch split gives the same loss.

        Kept as one function so it can be wrapped in ``torch.utils.checkpoint``:
        the logits are ``(hi-lo, b, vocab)`` and the vocabulary is ~152k, so
        materialising every branch's logits at once is the single largest term in
        the step. Under checkpointing only one micro-batch's worth exists at a
        time, recomputed during backward.
        """
        n = hi - lo
        # This utterance's REAL spine length: the batched spine forward pads to the
        # batch maximum, and the branches must see only the real columns.
        spine_len = int(two.spine_lens[i])
        valid = two.branch_valid[i, lo:hi]
        embeds = self._twod_branch_embeds(two.branch_ids[i, lo:hi], two.branch_frame_index[i, lo:hi], audio_embs[i])
        out = self._llm_forward(
            inputs_embeds=embeds,
            attention_mask=build_twod_branch_mask(
                two.branch_prefix[i, lo:hi],
                valid,
                int(two.spine_lens[i]),
                dtype,
                # Audio slots still hold the placeholder id at this point, so the
                # branch grid identifies them without a separate field.
                branch_is_audio=((two.branch_ids[i, lo:hi] == AUDIO_TOKEN_IDX) if self._bidirectional_audio else None),
            ),
            position_ids=two.branch_positions[i, lo:hi],
            # Fresh shallow copy per micro-batch, so nothing downstream can mutate
            # the shared spine cache that later micro-batches still need.
            past_key_values=broadcast_spine_cache(cache, i, n, spine_len),
            use_cache=False,
            return_dict=True,
        )
        targets = two.branch_targets[i, lo:hi]
        # Padding slots already carry IGNORE_INDEX, so they contribute nothing.
        with loss_parallel():
            return F.cross_entropy(
                out["logits"].flatten(0, 1),
                targets.flatten(0, 1),
                reduction="sum",
                ignore_index=IGNORE_INDEX,
            )

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Drive the training-WER decode AFTER the step is completely finished.

        It must not live inside training_step. Doing so ran extra forward passes
        and toggled module train/eval state between the forward and its
        recomputation, which corrupts activation checkpointing:

            torch.utils.checkpoint: Recomputed values for the following tensors
            have different metadata than during the forward pass.

        Both DFW SCRIPT arms died that way at exactly step 500 -- the first step
        where train_wer_every_n_steps fired. By on_train_batch_end the backward
        and optimizer step are done, so nothing this does can perturb them.
        """
        super().on_train_batch_end(outputs, batch, batch_idx)
        self._maybe_log_training_wer(batch)

    def _maybe_log_training_wer(self, batch: ScriptBatch) -> None:
        """Log ``training_batch_wer`` on a bounded sample of the training batch.

        Named to match the ASR collection so CHAT and SCRIPT land on one wandb
        panel, and like CHAT's it is a real decode rather than a teacher-forced
        score, so the two measure the same thing.

        Deliberately cheap and deliberately noisy: a handful of utterances every
        few hundred steps is enough to see a training curve diverge from
        validation, and not enough to slow the run. It is NOT a substitute for
        val_wer -- it is measured on whatever chunk size this batch drew, which for
        a multi chunk-size arm varies step to step.

        Module train/eval flags are saved and restored EXACTLY, per module. A bare
        self.train() afterwards would be wrong: _training_step_inner deliberately
        puts frozen submodules back into eval, and clobbering that would silently
        re-enable dropout in a frozen encoder for every later step.

        Any failure is swallowed: a metric must never be able to kill a run.
        """
        every = int(getattr(self.core_cfg, "train_wer_every_n_steps", 0) or 0)
        if every <= 0 or self._trainer is None:
            return
        step = int(self.trainer.global_step)
        if step == 0 or step % every != 0:
            return

        n = max(int(getattr(self.core_cfg, "train_wer_max_utts", 4) or 4), 1)
        refs = [t for t in (batch.text or [])][:n]
        if not refs or batch.audios is None:
            return

        was_training = {name: m.training for name, m in self.named_modules()}
        try:
            self.eval()
            with torch.no_grad():
                hyps = self.generate(
                    audios=batch.audios[: len(refs)],
                    audio_lens=batch.audio_lens[: len(refs)],
                    system_prompt=self._val_system_prompt,
                    max_new_tokens=self.val_max_new_tokens_per_chunk,
                    chunk_size_override=batch.chunk_size,
                )
            metric = WER(normalize=False, verbose=False)
            metric.update("train", refs=refs, hyps=[str(h) for h in hyps])
            for name, value in metric.compute().items():
                if name == "wer":
                    self.log("training_batch_wer", value.to(self.device), on_step=True)
        except Exception as e:  # pragma: no cover - never let a metric kill training
            logging.warning("training_batch_wer skipped: %r", e)
        finally:
            for name, m in self.named_modules():
                if name in was_training:
                    m.train(was_training[name])
            # generate() may have retuned the encoder's look-ahead; put it back.
            if getattr(self, "_last_chunk_size", None) is not None:
                self._set_encoder_att_context(self._last_chunk_size)

    def _banded_training_step(self, batch: ScriptBatch, batch_idx: int):
        """Marginalise the loss over every in-band word-to-chunk assignment.

        Runs the ORDINARY flat packed forward -- same embeds, same SCRIPT mask,
        one fused call -- and differs only in the reduction. The batch carries
        ``C`` branch segments per chunk instead of one; the mask needs no change,
        because it already gives each segment its own history through
        ``kp < prefix_len[q]`` and the candidates differ only in ``prefix_len``.

        The flat layout rather than the 2-D one is a measured choice. At one node
        the 2-D path cost 5.11 s/step against flat's 0.55 s/step for the SAME
        objective (``band_words=0`` is exactly the forced loss), because the 2-D
        premise that the spine K/V is broadcast and never copied is defeated by
        transformers' ``DynamicLayer.update``, which concatenates the stride-0
        expansion once per branch. Flat has no spine cache at all.
        """
        lat = batch.banded

        input_embeds = self._script_input_embeds(batch)
        impl, mask = self._training_attention(batch, input_embeds.dtype)
        plan = getattr(batch, "attn_plan", None)
        if self._attn_backend == "script" and plan is None:
            raise ValueError(
                "attn_backend='script' but the batch carries no attn_plan, so the structured "
                "kernel would fall back to unmasked SDPA -- branches would attend each other."
            )
        if plan is not None and self._bidirectional_audio:
            plan = replace(plan, bidirectional_audio=True)
        with self._attn_implementation(impl), _maybe_plan(plan if self._attn_backend == "script" else None):
            out = self._llm_forward(
                inputs_embeds=input_embeds,
                attention_mask=mask,
                position_ids=batch.position_ids,
                use_cache=False,
                return_dict=True,
            )

        logits = out["logits"]  # (B, L, V) -- left in bf16 on purpose
        b_size, seq_len, vocab = logits.shape
        k1 = int(lat.span_len) + 1

        # Absolute <ve> index per segment; segments differ in width because the
        # audio window narrows for early chunks, so this cannot be derived.
        # Segments are NOT uniform width any more: each is sized to its own chunk's
        # span length, which is what stopped one dense chunk from inflating the
        # whole utterance. So clamp each segment's reads to its OWN last position
        # rather than to the sequence end -- otherwise a short segment would read
        # forward into the next one and silently score the wrong tokens.
        #
        # The over-read positions are harmless: span_valid marks every k beyond the
        # segment's real length invalid, and span_scores' cumsum is prefix-only, so
        # a valid k never depends on a garbage k' > k.
        last = (lat.branch_ve_abs + lat.branch_span_len).unsqueeze(-1)  # (B, N, 1)
        idx = lat.branch_ve_abs.unsqueeze(-1) + torch.arange(k1, device=logits.device)
        idx = torch.minimum(idx, last).clamp(max=seq_len - 1).reshape(b_size, -1)  # (B, N * k1)

        # Only the (B, L) logsumexp is upcast; .float() on (B, L, 151936) would
        # allocate gigabytes and be recomputed under activation checkpointing.
        lse = torch.logsumexp(logits, dim=-1).float()
        lse_sel = lse.gather(1, idx)
        stop_lp = logits[..., self._eot_id].float().gather(1, idx) - lse_sel

        # Row and column together, so no (B, N * k1, vocab) intermediate exists.
        tgt = batch.target_tokens.gather(1, idx)
        rows = (torch.arange(b_size, device=logits.device).unsqueeze(1) * seq_len + idx).reshape(-1)
        tok_lp = logits.reshape(-1, vocab)[rows, tgt.clamp(min=0).reshape(-1)].view(b_size, -1).float() - lse_sel
        tok_lp = torch.where(tgt == IGNORE_INDEX, torch.zeros_like(tok_lp), tok_lp)

        t_max, n_cand = int(lat.cut.shape[1]), int(lat.n_cand)
        tok_lp = tok_lp.view(b_size, t_max, n_cand, k1)
        stop_lp = stop_lp.view(b_size, t_max, n_cand, k1)

        span_logprob = span_scores(tok_lp[..., :-1], stop_lp)
        span_logprob = torch.where(lat.span_valid, span_logprob, span_logprob.new_full((), NEG_INF))

        nll = banded_forward(span_logprob, lat.cut, lat.cut_valid, lat.n_chunks, lat.n_tokens)

        # An utterance no in-band path can complete would otherwise contribute
        # -NEG_INF and swamp the batch. Drop it, loudly.
        finite = nll < (-NEG_INF / 2)
        n_targets = int(lat.n_tokens[finite].sum()) if bool(finite.any()) else 0
        if n_targets == 0:
            logging.warning("Batch %d has no reachable in-band path — skipping (zero loss).", batch_idx)
            zero = sum(p.sum() for p in self.parameters() if p.requires_grad)
            return {"loss": zero * 0.0}
        if not bool(finite.all()):
            logging.warning(
                "Batch %d: %d/%d utterances had no reachable in-band path and were dropped.",
                batch_idx,
                int((~finite).sum()),
                b_size,
            )

        loss = nll[finite].sum() / n_targets
        self.log_dict(
            {
                "train_loss": loss,
                "learning_rate": torch.as_tensor(
                    self.trainer.optimizers[0].param_groups[0]["lr"] if self._trainer is not None else 0.0
                ),
            },
            on_step=True,
        )
        return {"loss": loss}

    def _twod_training_step(self, batch: ScriptBatch, batch_idx: int):
        two = batch.twod
        audio_embs, _ = self.perception(input_signal=batch.audios, input_signal_length=batch.audio_lens)
        dtype = audio_embs.dtype
        cache = self._twod_spine_cache(two, dtype)

        # Counted up front so every micro-batch is scaled by the SAME denominator;
        # the result is then a single mean over all supervised positions in the
        # batch, identical to the flat path regardless of how branches are split.
        n_targets = int((two.branch_targets != IGNORE_INDEX).sum())

        mb = max(int(self.core_cfg.twod_branch_micro_batch), 0)
        use_ckpt = mb > 0 and torch.is_grad_enabled()

        total = None
        for i in range(two.spine_ids.shape[0]):
            n = int(two.branch_counts[i])
            if n == 0:
                continue
            step = mb if mb > 0 else n
            for lo in range(0, n, step):
                hi = min(lo + step, n)
                if use_ckpt:
                    s = torch.utils.checkpoint.checkpoint(
                        self._branch_loss_sum, two, audio_embs, cache, dtype, i, lo, hi, use_reentrant=False
                    )
                else:
                    s = self._branch_loss_sum(two, audio_embs, cache, dtype, i, lo, hi)
                total = s if total is None else total + s

        if n_targets == 0 or total is None:
            logging.warning("Batch %d has no supervised targets — skipping (zero loss).", batch_idx)
            return {"loss": torch.zeros((), device=batch.audios.device, requires_grad=True)}

        loss = total / n_targets
        self.log_dict(
            {
                "train_loss": loss,
                "learning_rate": torch.as_tensor(
                    self.trainer.optimizers[0].param_groups[0]["lr"] if self._trainer is not None else 0.0
                ),
            },
            on_step=True,
        )
        return {"loss": loss}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: ScriptBatch, batch_idx: int):
        """OOM-tolerant wrapper around :meth:`_training_step_inner`.

        WHY THIS EXISTS. SCRIPT draws ``chunk_size`` per batch from
        ``[2, 4, 7, 10, 14, 28]`` while ``bucket_batch_size`` is keyed on DURATION
        only, so a batch sized to fit at chunk 14 can be ~7x more branches at
        chunk 2. Job 13380534 died in Qwen3's ``lm_head`` after 3h50m and 32k
        steps when that draw finally coincided with a long-audio bucket -- a tail
        event that costs a whole 8-node block each time it lands.

        WHY A PLAIN try/except WOULD HANG. Under DDP every rank must reach the
        same collectives in the same order. One rank returning early while the
        other 63 run backward leaves the gradient all-reduce waiting forever --
        no traceback, no checkpoint, the entire block burned. Strictly worse than
        crashing. Lightning also rejects the obvious shortcut: returning ``None``
        from ``training_step`` raises "Skipping the training_step by returning
        None in distributed training is not supported".

        So the verdict is ALL-REDUCED and every rank then takes the same branch,
        returning a graph-connected zero built from the PARAMETERS -- which keeps
        DDP seeing a gradient for each one instead of aborting on unused
        parameters. Same shape as the CHAT non-finite guard in
        ``nemo/collections/asr/models/chat_bpe_models.py``.

        FORWARD ONLY. An OOM during backward is not recovered: DDP overlaps
        all-reduce with gradient computation, so NCCL collectives are already in
        flight and the process group may be inconsistent. Those still crash, by
        design. The observed failure is in the forward pass.

        Skips are COUNTED and logged, because a zero loss is indistinguishable
        from a model that has learned the task perfectly -- and because the
        batches that OOM are systematically the long-audio, small-chunk ones, so
        a rising count means training on a quietly easier distribution. After
        ``oom_skip_limit`` consecutive skips it re-raises rather than limping on.
        """
        oom = False
        result = None
        try:
            result = self._training_step_inner(batch, batch_idx)
        except torch.cuda.OutOfMemoryError as e:
            oom = True
            self._last_oom = repr(e)[:200]

        # UNCONDITIONAL, on every rank and every step: a rank that succeeded has
        # no other way to learn that a peer did not, and if it proceeds to
        # backward alone the collective waits forever. The cost is one 4-byte
        # all-reduce per step, which is a sync point DDP would reach at backward
        # anyway.
        if self._any_rank_oom(oom):
            # EVERY rank drops the batch, including ranks whose forward
            # succeeded -- their loss is discarded so the whole batch is dropped
            # coherently rather than leaving one rank contributing zeros into an
            # otherwise real gradient average.
            del result
            return self._skip_oom_batch(batch_idx)
        return result

    def _any_rank_oom(self, local: bool) -> bool:
        """All-reduce the OOM verdict so every rank takes the same branch."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return local
        flag = torch.tensor([1 if local else 0], device=self.device, dtype=torch.int32)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        return bool(flag.item())

    # REINSTATED after the revert. The revert was right on the evidence then --
    # zero catches in two 4-hour runs, one NCCL deadlock caused. The evidence
    # changed within two runs: a FATAL backward OOM at chunk_size=2 and another
    # at chunk_size=7, the latter missing by 80 MB on a 79 GiB card, once in
    # ~8,280 steps. They are rare, marginal, and unrecoverable without this.
    #
    # The deadlock that justified the revert was a bug in _allreduce_grads, not
    # in the idea: the flat buffer's size depended on which parameters had
    # populated grads, so ranks reduced mismatched shapes. That is fixed and
    # regression-tested (test_allreduce_buffer_shape_is_rank_invariant).
    # ------------------------------------------------------------------
    # Backward-pass OOM guard
    # ------------------------------------------------------------------
    #
    # training_step's guard covers the FORWARD only: Lightning runs backward
    # after training_step returns, inside the optimizer closure, so an OOM there
    # escapes entirely. Measured on the multi-lookahead arms, that is the actual
    # killer -- an uncaught torch.OutOfMemoryError in _engine_run_backward takes
    # down one rank (e.g. rank 19), the other 63 tasks block in the gradient
    # collective, and Slurm SIGKILLs the step. It reads as a silent hang because
    # the traceback is on a rank nobody thinks to read. 100% of these OOMs occur
    # at chunk_size=2, the smallest-chunk/longest-audio corner.
    #
    # WHY no_sync IS REQUIRED AND NOT AN OPTIMISATION. DDP normally all-reduces
    # gradient buckets *during* backward. If a rank aborts partway, its peers are
    # already blocked waiting on buckets it will never contribute -- so catching
    # the exception locally is not enough; they deadlock before any vote can be
    # taken. Running backward under no_sync() removes every collective from
    # backward, which makes the failure purely local and therefore recoverable.
    # Gradients are then reduced explicitly below, once all ranks agree the step
    # is sound.
    #
    # THE COST IS REAL: one flat all-reduce per step instead of bucketed
    # all-reduces overlapped with backward compute. That trades some throughput
    # for not losing the entire job, which on an arm that was failing 12 times a
    # day is a large net win -- but it is a genuine slowdown on every step, not
    # only the rare OOM ones.
    def _ddp_module(self):
        """The DistributedDataParallel wrapper, if DDP is the active strategy."""
        for obj in (getattr(getattr(self, "trainer", None), "strategy", None), getattr(self, "trainer", None)):
            m = getattr(obj, "model", None)
            if m is not None and hasattr(m, "no_sync"):
                return m
        return None

    def backward(self, *args, **kwargs):
        ddp = self._ddp_module()
        oom = False
        try:
            if ddp is not None:
                with ddp.no_sync():
                    super().backward(*args, **kwargs)
            else:
                super().backward(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            oom = True
            self._last_oom = repr(e)[:200]

        if self._any_rank_oom(oom):
            # Drop the step on EVERY rank. set_to_none=False so the buffers stay
            # allocated and the optimizer sees a well-formed (zero) gradient
            # rather than a missing one.
            self.zero_grad(set_to_none=False)
            torch.cuda.empty_cache()
            self._skip_optimizer_step = True
            self._record_oom_skip(batch_idx=-1, phase="backward")
            return

        self._skip_optimizer_step = False
        if ddp is not None:
            self._allreduce_grads()

    def _allreduce_grads(self):
        """Average gradients across ranks -- the sync no_sync() suppressed.

        Flattened into one buffer per dtype: 948M trainable parameters issued as
        individual all-reduces would cost far more in launch latency than the
        bucketing this replaces.
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        world = torch.distributed.get_world_size()
        if world == 1:
            return
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        # EVERY TRAINABLE PARAMETER, on every rank, whether or not backward
        # populated its gradient.
        #
        # Filtering on `prm.grad is not None` made the flat buffer's SIZE depend
        # on which parameters a rank's batch happened to exercise -- and on
        # whether the previous step was skipped, which sets grads to None. Ranks
        # then all-reduced tensors of different shapes and NCCL blocked forever:
        # "Watchdog caught collective operation timeout: WorkNCCL(OpType=ALLREDUCE)"
        # at step 510, with no Python error because nothing raised. Real DDP
        # tolerates unused parameters; this hand-rolled replacement must do so
        # explicitly.
        #
        # Materialising a zero grad is the correct value as well as the safe one:
        # a parameter that received no gradient contributes zero to the average.
        by_dtype = defaultdict(list)
        for prm in self.parameters():
            if not prm.requires_grad:
                continue
            if prm.grad is None:
                prm.grad = torch.zeros_like(prm)
            by_dtype[prm.grad.dtype].append(prm)
        for params in by_dtype.values():
            grads = [prm.grad.data for prm in params]
            flat = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(flat)
            flat.div_(world)
            for g, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
                g.copy_(synced)

    def optimizer_step(self, *args, **kwargs):
        """Skip the update when the step was dropped for OOM.

        Zeroed gradients are NOT equivalent to skipping: Adam still advances its
        moment estimates and the LR schedule on a zero grad, so a dropped batch
        would quietly decay the model rather than be a no-op.
        """
        if getattr(self, "_skip_optimizer_step", False):
            self._skip_optimizer_step = False
            self.zero_grad(set_to_none=True)
            return
        out = super().optimizer_step(*args, **kwargs)
        # An update actually landed, so the streak is broken. This is the only
        # place that can truthfully say so: it is reached only when forward AND
        # backward both completed on every rank.
        self._oom_streak = 0
        return out

    def _record_oom_skip(self, batch_idx: int, phase: str) -> None:
        """Shared accounting for forward- and backward-pass OOM skips."""
        self._oom_skips = getattr(self, "_oom_skips", 0) + 1
        self._oom_streak = getattr(self, "_oom_streak", 0) + 1
        limit = int(getattr(self.core_cfg, "oom_skip_limit", 25) or 25)
        if self._oom_streak >= limit:
            raise RuntimeError(
                f"{self._oom_streak} consecutive OOM batches (limit {limit}) at the {phase} pass. This is no "
                f"longer a rare draw -- the batch size does not fit this configuration. Scale "
                f"bucket_batch_size for this recipe, or drop the smallest chunk size. "
                f"Last error: {getattr(self, '_last_oom', '')}"
            )
        logging.error(
            "CUDA OOM in %s pass (batch %s, chunk_size=%s); dropping the step on all ranks. "
            "total skipped=%d, consecutive=%d. Last: %s",
            phase,
            batch_idx,
            getattr(self, "_last_chunk_size", "?"),
            self._oom_skips,
            self._oom_streak,
            getattr(self, "_last_oom", ""),
        )
        self.log("train_batches_skipped_oom", float(self._oom_skips), prog_bar=True, on_step=True)

    def _skip_oom_batch(self, batch_idx: int):
        """Drop the batch on EVERY rank, keeping DDP in lockstep."""
        # ONE memory summary, on the FIRST OOM only, before empty_cache() erases
        # the evidence.
        #
        # The OOM message says how much was in use but never WHAT was holding it,
        # which left the banded arm's repeated failures to be answered by guessing
        # at bucket_batch_size -- four cuts, four more failures. This prints the
        # allocator's own breakdown (params vs activations vs fragmentation) so the
        # next decision is made on evidence. Rank 0 only and once per process, so
        # it cannot flood the log.
        if not getattr(self, "_oom_summary_logged", False):
            self._oom_summary_logged = True
            try:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    logging.error(
                        "FIRST CUDA OOM -- allocator state before empty_cache():\n%s",
                        torch.cuda.memory_summary(abbreviated=True),
                    )
            except Exception:  # pragma: no cover - diagnostics must never add a failure
                pass

        # Release whatever the failed forward left behind before the next batch.
        torch.cuda.empty_cache()

        self._record_oom_skip(batch_idx, phase="forward")

        # Graph-connected zero that TOUCHES every trainable parameter, so DDP
        # still receives a gradient for each and does not abort on unused ones.
        zero = sum(p.sum() for p in self.parameters() if p.requires_grad)
        return {"loss": zero * 0.0}

    def _training_step_inner(self, batch: ScriptBatch, batch_idx: int):
        # Keep frozen modules in eval mode (disables dropout / BN updates).
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        # Pin the encoder look-ahead to this batch's chunk size so a frame's
        # receptive field never crosses its chunk boundary.
        self._set_encoder_att_context(batch.chunk_size)
        self._last_chunk_size = batch.chunk_size

        if self._loss_type == "banded":
            return self._banded_training_step(batch, batch_idx)
        if self._twod_layout:
            return self._twod_training_step(batch, batch_idx)

        input_embeds = self._script_input_embeds(batch)
        impl, mask = self._training_attention(batch, input_embeds.dtype)
        plan = getattr(batch, "attn_plan", None)
        if self._attn_backend == "script" and plan is None:
            # The structured backend needs the plan to know the block layout; with
            # none active it falls through to plain SDPA, which is BOTH causal and
            # blind to the cross-branch rule the whole design rests on. Refuse
            # rather than train a silently different model.
            raise ValueError(
                "attn_backend='script' but the batch carries no attn_plan, so the structured "
                "kernel would fall back to unmasked SDPA -- branches would attend each other. "
                "Use attn_backend='dense' or 'flex', or populate batch.attn_plan via "
                "script_attention.build_attention_plan()."
            )
        if plan is not None and self._bidirectional_audio:
            plan = replace(plan, bidirectional_audio=True)
        with self._attn_implementation(impl), _maybe_plan(plan if self._attn_backend == "script" else None):
            out = self._llm_forward(
                inputs_embeds=input_embeds,
                attention_mask=mask,
                position_ids=batch.position_ids,
                use_cache=False,
                return_dict=True,
            )
        logits = out["logits"]  # (B, T, V)

        target_ids = batch.target_tokens
        num_targets = (target_ids != IGNORE_INDEX).long().sum()
        if num_targets == 0:
            logging.warning("Batch %d has no supervised targets — skipping (zero loss).", batch_idx)
            return {"loss": torch.zeros((), device=logits.device, requires_grad=True)}

        with loss_parallel():
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                target_ids.flatten(0, 1),
                reduction="mean",
                ignore_index=IGNORE_INDEX,
            )

        metrics = {
            "train_loss": loss,
            "learning_rate": torch.as_tensor(
                self.trainer.optimizers[0].param_groups[0]["lr"] if self._trainer is not None else 0.0
            ),
        }
        if self.core_cfg.log_detailed_train_metrics:
            B, T = batch.input_tokens.shape
            metrics.update(
                {
                    "num_targets": num_targets.float(),
                    "sequence_length": float(T),
                    "target_to_input_ratio": num_targets / (B * T),
                }
            )
        # NOT reset here: a forward that succeeds can still OOM in backward, so
        # resetting on forward success would make oom_skip_limit unreachable for
        # backward OOMs -- the job would skip every step forever while logging a
        # zero loss. The reset lives in optimizer_step, where an update landed.
        self.log_dict(metrics, on_step=True)
        return {"loss": loss}

    # ------------------------------------------------------------------
    # Validation — decode-only WER
    # ------------------------------------------------------------------

    @property
    def val_chunk_size(self) -> Optional[int]:
        """Chunk size for the validation decode.

        With multi chunk-size training the largest candidate may imply a
        look-ahead configuration that is slow or unsupported for streaming, so
        validation pins one well-supported size instead.
        """
        if not getattr(self, "_chunk_size_candidates", None):
            return None
        configured = self.core_cfg.val_chunk_size
        if configured is not None:
            return int(configured)
        return 14 if 14 in self._chunk_size_candidates else max(self._chunk_size_candidates)

    @property
    def val_max_new_tokens_per_chunk(self) -> int:
        """Per-chunk autoregressive cap for validation."""
        configured = self.core_cfg.val_max_new_tokens_per_chunk
        if configured is not None:
            if configured <= 0:
                raise ValueError("val_max_new_tokens_per_chunk must be positive")
            return int(configured)
        return self.val_chunk_size or 64

    def _validation_system_prompts(self, batch) -> Union[str, List[str]]:
        if getattr(batch, "cuts", None) is None:
            return self._val_system_prompt
        return [(cut.custom or {}).get(self.core_cfg.val_prompt_field, self._val_system_prompt) for cut in batch.cuts]

    def on_validation_epoch_start(self) -> None:
        self._partial_wer_refs: dict = defaultdict(list)
        self._partial_wer_hyps: dict = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        # Gather the decoded strings and compute a true corpus WER. Averaging
        # rank-local WERs would be wrong when ranks see different word counts.
        local = {
            name: {"refs": self._partial_wer_refs[name], "hyps": self._partial_wer_hyps[name]}
            for name in self._partial_wer_refs
        }
        if torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        else:
            gathered = [local]

        # TWO normalizations over the SAME decoded strings, because one number
        # cannot serve both purposes.
        #
        #   val_wer          Whisper-normalised. speechlm2's historical default and
        #                    this model's checkpoint monitor, so its meaning must
        #                    not change -- every existing run's save_top_k was
        #                    selected against it.
        #   val_wer_verbatim raw text, no normalisation. This is what the ASR
        #                    collection (and therefore the CHAT models) reports as
        #                    its val_wer.
        #
        # Logging both makes the families comparable on a like-for-like axis. They
        # were NOT before: CHAT reading ~0.15 and SCRIPT ~0.087 on the same
        # manifest was mostly the normaliser, not the model, and that gap was read
        # as a real quality difference more than once.
        #
        # The second metric is nearly free -- it re-scores strings that are already
        # decoded and gathered, with no extra forward pass.
        wer = WER(normalize=True, verbose=False)
        wer_verbatim = WER(normalize=False, verbose=False)
        has_data = False
        for rank_data in gathered:
            for name, values in rank_data.items():
                has_data = has_data or bool(values["refs"])
                wer.update(name, refs=values["refs"], hyps=values["hyps"])
                wer_verbatim.update(name, refs=values["refs"], hyps=values["hyps"])
        if has_data:
            for metric_name, metric_value in wer.compute().items():
                log_name = "val_wer" if metric_name == "wer" else f"val_{metric_name}"
                self.log(log_name, metric_value.to(self.device), on_epoch=True, sync_dist=False)
            for metric_name, metric_value in wer_verbatim.compute().items():
                log_name = "val_wer_verbatim" if metric_name == "wer" else f"val_{metric_name}_verbatim"
                self.log(log_name, metric_value.to(self.device), on_epoch=True, sync_dist=False)

        self._partial_wer_refs.clear()
        self._partial_wer_hyps.clear()

    def validation_step(self, batch, batch_idx: int):
        if isinstance(batch, dict):  # multiple validation dataloaders
            for name, dataset_batch in batch.items():
                if dataset_batch is not None:
                    self._eval_step(dataset_batch, name, batch_idx)
        else:
            self._eval_step(batch, "val", batch_idx)

    def _eval_step(self, batch, name: str, batch_idx: int = 0) -> None:
        # Validation is decode-only: autoregressive WER needs neither word
        # alignments nor constructed target turns, just audio and reference text.
        refs = list(batch.text)
        # A prompt-controlled model decodes at a fixed validation operating point.
        # Restyle the references to that same style, or WER would count the model
        # honouring the requested style as an error on every word.
        if self.core_cfg.prompt_control:
            refs = [apply_text_style(r, self.core_cfg.val_capitalization, self.core_cfg.val_punctuation) for r in refs]
        hyps = self.generate(
            audios=batch.audios,
            audio_lens=batch.audio_lens,
            system_prompt=self._validation_system_prompts(batch),
            max_new_tokens=self.val_max_new_tokens_per_chunk,
            generation_config=GenerationConfig(do_sample=False),
            chunk_size_override=self.val_chunk_size,
        )
        self._partial_wer_refs[name].extend(refs)
        self._partial_wer_hyps[name].extend(hyps)

        if batch_idx % self.core_cfg.log_every_n_steps == 0 and refs and hyps:
            logging.info(
                "[%s] decode batch %d (max %d tokens/chunk)\n  ref: `%s`\n  hyp: `%s`",
                name,
                batch_idx,
                self.val_max_new_tokens_per_chunk,
                refs[0],
                hyps[0],
            )

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _is_word_start(self, token_id: int) -> bool:
        """Whether ``token_id``'s surface form begins a new word.

        Both the GPT-2/Qwen byte-level marker (``Ġ``) and the SentencePiece
        marker (``▁``) are recognised, so this works across tokenizer families.
        """
        tok = self.tokenizer.tokenizer.convert_ids_to_tokens(int(token_id))
        return isinstance(tok, str) and (tok.startswith("Ġ") or tok.startswith("▁"))

    def _get_word_start_insert_id(self) -> Optional[int]:
        """Token id of a lone leading-space subword, cached after first lookup.

        Inserted in front of a chunk's first token when that token is not itself a
        word start, so the chunk cannot merge onto the previous chunk's last word
        ("border ruffian" -> "bordereruffian"). Returns ``None`` — disabling the
        guard — if the tokenizer has no such standalone token.
        """
        if self._word_start_insert_id is None:
            hf_tok = self.tokenizer.tokenizer
            unk = getattr(hf_tok, "unk_token_id", None)
            self._word_start_insert_id = -1  # sentinel: resolved but absent
            for marker in ("Ġ", "▁"):
                tid = hf_tok.convert_tokens_to_ids(marker)
                if tid is not None and tid >= 0 and (unk is None or tid != unk) and self._is_word_start(tid):
                    self._word_start_insert_id = int(tid)
                    break
            if self._word_start_insert_id == -1:
                logging.warning(
                    "ScriptSTTModel: tokenizer has no standalone word-start token; "
                    "chunk-start word-boundary insertion is disabled."
                )
        return self._word_start_insert_id if self._word_start_insert_id != -1 else None

    def encode_frames(self, audios: Tensor, audio_lens: Tensor, chunk_size: int) -> List[Tensor]:
        """Encode a batch of waveforms into per-utterance encoder-frame sequences.

        The encoder's right context is pinned to ``chunk_size - 1`` first, so
        each output frame depends only on audio up to its own chunk's boundary —
        the same dependency structure as frame-by-frame streaming. Computing all
        frames in one pass is purely a batching optimization.

        Sub-batched because a single full-batch encode of long, length-sorted
        clips can overflow 32-bit CUDA indexing in the subsampling convolution.
        """
        self._set_encoder_att_context(chunk_size)
        B = audios.shape[0]
        frames: List[Optional[Tensor]] = [None] * B
        step = max(1, int(self.core_cfg.encode_batch_size))
        for i in range(0, B, step):
            hi = min(i + step, B)
            idx = torch.arange(i, hi, device=audio_lens.device)
            lens = audio_lens[idx]
            sig = audios[idx, : int(lens.max().item())]
            emb, emb_len = self.perception(input_signal=sig, input_signal_length=lens)  # (b, T_enc, H)
            for j, b in enumerate(range(i, hi)):
                frames[b] = emb[j, : int(emb_len[j].item())].clone()
        return frames

    @torch.no_grad()
    def generate(
        self,
        audios: Tensor,
        audio_lens: Tensor,
        system_prompt: Union[str, List[str]] = "Transcribe the audio into text.",
        max_new_tokens: int = 64,
        generation_config: Optional[GenerationConfig] = None,
        chunk_size_override: Optional[int] = None,
        **generation_kwargs,
    ) -> List[str]:
        """Chunk-by-chunk streaming transcription.

        Encodes the audio once (see :meth:`encode_frames` on why that does not
        weaken the streaming constraint), then runs the batched greedy
        spine+branch decode: for each chunk every active stream is shown its
        compact text history plus that chunk's audio window, exactly as in
        training.

        Args:
            audios / audio_lens: waveforms ``(B, T)`` and sample counts ``(B,)``.
            system_prompt: one instruction, or one per utterance. For a
                prompt-controlled model this is the BASE instruction; the control
                sentence is appended here, so callers pass the same base string
                the recipe trained with.
            max_new_tokens: cap on tokens decoded per chunk.
            chunk_size_override: decode at this chunk size instead of the
                configured / representative one.
            num_delay_frames / capitalization / punctuation: prompt-controlled
                models only — the operating point to request. Default to the
                ``val_*`` config values. Ignored when ``prompt_control`` is off.

        Returns:
            ``B`` transcripts.
        """
        cs = self._resolve_inference_chunk_size(chunk_size_override)
        if cs <= 0:
            raise ValueError(f"SCRIPT generate requires a positive chunk size, got {cs}")

        B = audios.shape[0]
        if isinstance(system_prompt, str):
            system_prompt = [system_prompt] * B

        # Prompt control: state the operating point in the instruction, using the
        # SAME renderer the dataset used during training.
        asked = {
            k: generation_kwargs.pop(k)
            for k in ("num_delay_frames", "capitalization", "punctuation")
            if k in generation_kwargs
        }
        if self.core_cfg.prompt_control:
            controls = ScriptControls(
                chunk_size=cs,
                num_delay_frames=int(asked.get("num_delay_frames", self.core_cfg.val_num_delay_frames)),
                capitalization=bool(asked.get("capitalization", self.core_cfg.val_capitalization)),
                punctuation=bool(asked.get("punctuation", self.core_cfg.val_punctuation)),
            )
            system_prompt = [render_control_prompt(p, controls) for p in system_prompt]
        elif asked:
            # Silently ignoring these would look like the knobs work when the model
            # never learned them, which is the expensive kind of mistake.
            raise ValueError(
                f"generate() got {sorted(asked)} but this model has prompt_control=False, so it was never "
                "trained to honour them. Set model.prompt_control=true (and train that way) or drop these arguments."
            )

        # FSM decode knobs. Separable on purpose: streaming_encode swaps the
        # OFFLINE encode for the cache-aware one (the change that was worth 12
        # WER points on the interleaved model), state_machine swaps the bulk
        # prefill for explicit per-stream stepping. Either can be used alone.
        streaming_encode = bool(generation_kwargs.pop("streaming_encode", False))
        state_machine = bool(generation_kwargs.pop("use_state_machine_inference", False))
        max_history_tokens = int(generation_kwargs.pop("max_history_tokens", self.core_cfg.max_history_tokens))
        # Chunk-synchronous joint decoding with a CHAT transducer. Off unless a
        # scorer is supplied, so the production decode path is unchanged.
        chat_fusion = generation_kwargs.pop("chat_fusion", None)
        fusion_lam = float(generation_kwargs.pop("fusion_lam", 0.5))
        fusion_margin_threshold = float(generation_kwargs.pop("fusion_margin_threshold", float("inf")))
        fusion_stats = generation_kwargs.pop("fusion_stats", None)
        fusion_skip_threshold = float(generation_kwargs.pop("fusion_skip_threshold", float("inf")))
        fusion_skipped = generation_kwargs.pop("fusion_skipped", None)
        # Guarantee that each chunk's first emitted token starts a new word. On by
        # default: without it a chunk whose first token is a continuation merges
        # onto the previous chunk's last word.
        force_word_start = bool(generation_kwargs.pop("force_word_start", self.core_cfg.force_word_start))
        insert_word_start_id = self._get_word_start_insert_id() if force_word_start else None

        if streaming_encode:
            frames_list = streaming_encode_frames(self, audios, audio_lens, cs)
        else:
            frames_list = self.encode_frames(audios, audio_lens, cs)
        # Same instruction/history separator the dataset uses when building the spine.
        instruction_ids_list = [self.tokenizer.text_to_ids(system_prompt[b] + "\n") for b in range(B)]

        # Inference-only: a per-emission-index word-insertion penalty (bonus on
        # <eot>). Popped here so a model that ignores it never sees the kwarg.
        emission_penalty_lambda = float(generation_kwargs.pop("emission_penalty_lambda", 0.0) or 0.0)
        emission_penalty = generation_kwargs.pop("emission_penalty", None)
        if emission_penalty is not None:
            emission_penalty = [float(x) for x in emission_penalty]
        return_chunk_ids = bool(generation_kwargs.pop("return_chunk_ids", False))
        if return_chunk_ids and state_machine:
            raise ValueError(
                "return_chunk_ids is not supported by the state-machine decode path; it emits "
                "per-stream rather than per-chunk, so a chunk index would be meaningless."
            )
        self._reject_fsm_with_bidirectional_audio(state_machine)
        decode_fn = fsm_stream_decode_script if state_machine else batched_stream_decode_script
        if chat_fusion is not None and state_machine:
            # The FSM decoder has its own emission logic and no fusion hook, so it
            # would silently IGNORE chat_fusion and report a SCRIPT-only number as
            # if it were a joint one.
            raise ValueError(
                "chat_fusion is not supported with use_state_machine_inference=True; "
                "the FSM decoder has no fusion hook and would silently ignore it."
            )
        # full_context: keep the ENCODER at chunk size `cs` (already applied by
        # encode_frames) but hand the decoder a chunk large enough that every
        # utterance is a single branch -- the LLM then sees all the audio at once.
        decode_chunk = cs
        if self.core_cfg.full_context:
            decode_chunk = max(1, max((int(f.shape[0]) for f in frames_list), default=1))

        emitted = decode_fn(
            llm=self.llm,
            embed_tokens=self._embed_tokens,
            instruction_ids_list=instruction_ids_list,
            frames_list=frames_list,
            chunk_size=decode_chunk,
            vision_start_id=self._vision_start_id,
            vision_end_id=self._vision_end_id,
            eot_id=self._eot_id,
            read_id=self._read_id,
            write_id=self._write_id,
            gate_in_history=self._gate_in_history,
            position_scheme=self._decode_position_scheme,
            pad_id=self.text_pad_id,
            max_new_tokens=max_new_tokens,
            device=self.device,
            audio_history_chunks=self._audio_history_chunks,
            audio_window_frames=self._audio_window_frames,
            max_history_tokens=max_history_tokens,
            chat_fusion=chat_fusion,
            fusion_lam=fusion_lam,
            fusion_margin_threshold=fusion_margin_threshold,
            fusion_stats=fusion_stats,
            fusion_skip_threshold=fusion_skip_threshold,
            fusion_skipped=fusion_skipped,
            is_word_start=self._is_word_start if insert_word_start_id is not None else None,
            insert_word_start_id=insert_word_start_id,
            **({"bidirectional_audio": True} if self._bidirectional_audio else {}),
            **({"return_chunk_ids": True} if return_chunk_ids else {}),
            **({"emission_penalty": emission_penalty} if emission_penalty else {}),
            **({"emission_penalty_lambda": emission_penalty_lambda} if emission_penalty_lambda else {}),
        )
        chunk_ids = None
        if return_chunk_ids:
            emitted, chunk_ids = emitted
        # The history may legitimately contain gate tokens (gate_in_history);
        # they are conditioning, not transcript, so never let them reach the text.
        drop = {t for t in (self._read_id, self._write_id) if t is not None}
        if drop:
            if chunk_ids is None:
                emitted = [[t for t in ids if t not in drop] for ids in emitted]
            else:
                # Filter tokens and their chunk labels TOGETHER, or the two lists
                # silently desync and every word after a gate is attributed to
                # the wrong chunk.
                kept = [[(t, k) for t, k in zip(ids, ks) if t not in drop] for ids, ks in zip(emitted, chunk_ids)]
                emitted = [[t for t, _ in pair] for pair in kept]
                chunk_ids = [[k for _, k in pair] for pair in kept]
        texts = [self.tokenizer.ids_to_text(ids) if ids else "" for ids in emitted]
        if not return_chunk_ids:
            return texts
        # Per chunk, the text it emitted -- so a word can be located inside its
        # chunk. Detokenised per group rather than sliced out of `texts`, because
        # the tokenizer's word-start convention makes offsets unreliable.
        per_chunk = []
        for ids, ks in zip(emitted, chunk_ids):
            groups: Dict[int, List[int]] = {}
            for t, k in zip(ids, ks):
                groups.setdefault(int(k), []).append(t)
            per_chunk.append([[k, self.tokenizer.ids_to_text(v) if v else ""] for k, v in sorted(groups.items())])
        return texts, per_chunk
