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
from dataclasses import dataclass
from typing import List, Optional, Union

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
    prompt_control: bool = False
    read_write: bool = False
    read_token: str = "<|box_start|>"
    write_token: str = "<|box_end|>"
    gate_in_history: bool = False
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
    def _script_mask_mod(batch: ScriptBatch):
        """The SCRIPT rule as a FlexAttention predicate.

        A direct transcription of :func:`build_script_mask`; the equality of the
        two is asserted in the tests.
        """
        seg, pos, pref, val = batch.seg_ids, batch.position_ids, batch.prefix_len, batch.valid

        def mask_mod(b, h, q, kv):
            qs, ks = seg[b, q], seg[b, kv]
            qp, kp = pos[b, q], pos[b, kv]
            q_spine, k_spine = qs == 0, ks == 0
            causal = kp <= qp
            return (
                (q_spine & k_spine & causal)
                | ((~q_spine) & k_spine & (kp < pref[b, q]))
                | ((qs == ks) & (~q_spine) & causal)
            ) & val[b, kv]

        return mask_mod

    def _training_attention(self, batch: ScriptBatch, dtype):
        """(attn_implementation, attention_mask) for this batch's backend."""
        if self._attn_backend == "flex":
            from torch.nn.attention.flex_attention import create_block_mask

            B, T = batch.seg_ids.shape
            block_mask = create_block_mask(
                self._script_mask_mod(batch), B=B, H=None, Q_LEN=T, KV_LEN=T, device=batch.seg_ids.device
            )
            return "flex_attention", block_mask
        if self._attn_backend == "script":
            return "script", None
        return "eager", build_script_mask(batch.seg_ids, batch.position_ids, batch.prefix_len, batch.valid, dtype)

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
            attention_mask=build_twod_branch_mask(two.branch_prefix[i, lo:hi], valid, int(two.spine_lens[i]), dtype),
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
                "loss": loss,
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
        # Keep frozen modules in eval mode (disables dropout / BN updates).
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        # Pin the encoder look-ahead to this batch's chunk size so a frame's
        # receptive field never crosses its chunk boundary.
        self._set_encoder_att_context(batch.chunk_size)

        if self._twod_layout:
            return self._twod_training_step(batch, batch_idx)

        input_embeds = self._script_input_embeds(batch)
        impl, mask = self._training_attention(batch, input_embeds.dtype)
        plan = getattr(batch, "attn_plan", None)
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
            "loss": loss,
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

        wer = WER(normalize=True, verbose=False)
        has_data = False
        for rank_data in gathered:
            for name, values in rank_data.items():
                has_data = has_data or bool(values["refs"])
                wer.update(name, refs=values["refs"], hyps=values["hyps"])
        if has_data:
            for metric_name, metric_value in wer.compute().items():
                log_name = "val_wer" if metric_name == "wer" else f"val_{metric_name}"
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

        max_history_tokens = int(generation_kwargs.pop("max_history_tokens", self.core_cfg.max_history_tokens))
        # Guarantee that each chunk's first emitted token starts a new word. On by
        # default: without it a chunk whose first token is a continuation merges
        # onto the previous chunk's last word.
        force_word_start = bool(generation_kwargs.pop("force_word_start", self.core_cfg.force_word_start))
        insert_word_start_id = self._get_word_start_insert_id() if force_word_start else None

        frames_list = self.encode_frames(audios, audio_lens, cs)
        # Same instruction/history separator the dataset uses when building the spine.
        instruction_ids_list = [self.tokenizer.text_to_ids(system_prompt[b] + "\n") for b in range(B)]

        emitted = batched_stream_decode_script(
            llm=self.llm,
            embed_tokens=self._embed_tokens,
            instruction_ids_list=instruction_ids_list,
            frames_list=frames_list,
            chunk_size=cs,
            vision_start_id=self._vision_start_id,
            vision_end_id=self._vision_end_id,
            eot_id=self._eot_id,
            read_id=self._read_id,
            write_id=self._write_id,
            gate_in_history=self._gate_in_history,
            pad_id=self.text_pad_id,
            max_new_tokens=max_new_tokens,
            device=self.device,
            audio_history_chunks=self._audio_history_chunks,
            audio_window_frames=self._audio_window_frames,
            max_history_tokens=max_history_tokens,
            is_word_start=self._is_word_start if insert_word_start_id is not None else None,
            insert_word_start_id=insert_word_start_id,
        )
        # The history may legitimately contain gate tokens (gate_in_history);
        # they are conditioning, not transcript, so never let them reach the text.
        drop = {t for t in (self._read_id, self._write_id) if t is not None}
        if drop:
            emitted = [[t for t in ids if t not in drop] for ids in emitted]
        return [self.tokenizer.ids_to_text(ids) if ids else "" for ids in emitted]
