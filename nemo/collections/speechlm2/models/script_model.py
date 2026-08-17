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
"""SCRIPT streaming SpeechLM.

A variant of :class:`StreamingSTTModel` that frames streaming ASR as a
conditional *text-completion* task: for each audio chunk, extract the words it
carries given the transcript so far, ``p(words_k | text_history_<k, audio_k)``.

Rather than one interleaved ``audio-text-audio-text`` causal stream, each
utterance is packed as a pure-text **spine** (the reusable history) plus one
**branch** per chunk (its audio + target words); see
:mod:`nemo.collections.speechlm2.parts.script`. Training is a single
O(L) forward with a custom 4D mask + ``position_ids``; the CE loss is taken on
the branch target words. Inference is a streaming loop that keeps a plain-text
spine KV cache and evicts each chunk's audio after decoding it.

Only the training and generation paths differ from the parent — audio encoding
(``perception``), embedding interleave (``_build_input_embeds``), and the
validation/WER accumulation are reused as-is.
"""

import math
from collections import defaultdict
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor
from torch.distributed.tensor.parallel import loss_parallel
from transformers import GenerationConfig

from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.speechlm2.data.script_dataset import ScriptBatch, ScriptSTTDataset
from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX, IGNORE_INDEX
from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel
from nemo.collections.speechlm2.parts.metrics.wer import WER
from nemo.collections.speechlm2.parts.script import (
    batched_stream_decode_redecode,
    batched_stream_decode_script,
    batched_stream_decode_script_last_layer,
    build_script_mask,
    build_packed_chunk_example,
    collate_packed_chunk_examples,
    run_script_layers_split,
)
from nemo.collections.speechlm2.parts.shared_audio_chunk import (
    batched_shared_audio_decode,
    build_shared_audio_chunk_mask,
)
from nemo.collections.speechlm2.parts.optim_setup import is_frozen
from nemo.utils import logging


class ScriptSTTModel(StreamingSTTModel):
    """StreamingSTTModel with the packed spine+branch SCRIPT objective."""

    audio_open_token: str = "<|vision_start|>"
    audio_close_token: str = "<|vision_end|>"
    # Sub-batch size for the encoder pass during generate(). Encoding the whole
    # (possibly large) eval batch of long clips at once overflows 32-bit CUDA
    # indexing in the subsampling conv; encoding in sub-batches keeps each conv
    # tensor bounded. The LLM decode is still fully batched across the whole batch.
    encode_batch_size: int = 8

    def __init__(self, cfg: dict, forced_aligner=None, data_cfg=None, dataset_cls=ScriptSTTDataset) -> None:
        # NOTE: keep this signature identical to StreamingSTTModel.__init__ (no
        # *args/**kwargs). huggingface_hub's PyTorchModelHubMixin.from_pretrained
        # inspects the __init__ signature; a **kwargs here makes it expand the
        # whole saved config (att_context_size, chunk_size, ...) into keyword
        # args, which then break super().__init__.
        super().__init__(cfg, forced_aligner=forced_aligner, data_cfg=data_cfg, dataset_cls=dataset_cls)

        # The base StreamingSTTModelConfig dataclass keeps only its own fields when
        # coerced via to_dataclass(); SCRIPT-specific keys (audio_history_chunks,
        # self_correction, script_last_layer_*, redecode, ...) are dropped. Re-attach
        # them from the raw model cfg (self.cfg, a DictConfig) so the getattr(self.core_cfg, ...)
        # reads below see the values a user set in their SCRIPT yaml.
        _raw_model_cfg = OmegaConf.to_container(self.cfg, resolve=True)
        for _k, _v in _raw_model_cfg.items():
            if not hasattr(self.core_cfg, _k):
                setattr(self.core_cfg, _k, _v)

        # Base-config fields that this class reads directly as self.core_cfg.<name>
        # (not via getattr) but that the clean automodel StreamingSTTModelConfig does
        # not declare (they existed on the base SCRIPT was developed against). Seed
        # the original defaults when absent from both the recipe and the base so the
        # direct reads below don't AttributeError.
        _script_core_cfg_defaults = {
            "log_detailed_train_metrics": False,
            # Decode-only val_wer knobs (present on the base SCRIPT was developed
            # against; absent from the clean automodel base). None => derive the cap
            # / decode chunk size from the (repr) chunk size at validation time.
            # See the "Validation" section below.
            "val_max_new_tokens_per_chunk": None,
            "val_chunk_size": None,
        }
        for _k, _v in _script_core_cfg_defaults.items():
            if not hasattr(self.core_cfg, _k):
                setattr(self.core_cfg, _k, _v)

        # Decode-only validation prompts. The clean automodel base reports only a
        # teacher-forced val_acc and never sets these; SCRIPT restores true
        # autoregressive val_wer (see the "Validation" section) which needs the
        # per-sample system prompt. Mirror the base SCRIPT behaviour: default prompt
        # from the data config, per-utterance override read from
        # cut.custom[<prompt_field>]. data_cfg is None under from_pretrained.
        self.val_system_prompt = (
            data_cfg.get("system_prompt", "Transcribe the audio into text.")
            if data_cfg is not None
            else "Transcribe the audio into text."
        )
        self.val_prompt_field = (
            data_cfg.get("prompt_field", "system_prompt") if data_cfg is not None else "system_prompt"
        )

        # Audio history window (M previous chunks in each branch). Read from the
        # MODEL config so it survives from_pretrained (data_cfg is None then).
        self._audio_history_chunks = max(int(getattr(self.core_cfg, "audio_history_chunks", 0) or 0), 0)
        # Fixed-frame audio window (takes precedence over audio_history_chunks).
        self._audio_window_frames = max(int(getattr(self.core_cfg, "audio_window_frames", 0) or 0), 0)
        # Shared-audio packed layout (encoder frames laid once, windowed via the mask).
        self._shared_audio_track = bool(getattr(self.core_cfg, "shared_audio_track", False))

        # Byte-level BPE word-start test (leading-space marker), always available.
        # SCRIPT emits WHOLE words per chunk, so a chunk's first decoded token must be
        # word-initial; used for decode-time diagnostics / optional enforcement.
        _hf_tok = self.tokenizer.tokenizer

        def _is_word_start(tid: int) -> bool:
            tok = _hf_tok.convert_ids_to_tokens(int(tid))
            return isinstance(tok, str) and (tok.startswith("\u0120") or tok.startswith("\u2581"))

        self._is_word_start = _is_word_start
        self._word_start_insert_id = None  # lazily resolved leading-space token id

        # --- Self-correction (delete-last-word) ---
        self._self_correction = bool(getattr(self.core_cfg, "self_correction", False))
        self._self_correction_prob = float(getattr(self.core_cfg, "self_correction_prob", 1.0) or 0.0)
        self._self_correction_log_every = int(getattr(self.core_cfg, "self_correction_log_every", 200) or 0)
        # Correction scope: "word" deletes only the previous chunk's last word;
        # "chunk" deletes the entire previous chunk (one <del> -> whole last chunk).
        # Read from the MODEL config so from_pretrained rebuilds it for decode.
        self._self_correction_scope = str(getattr(self.core_cfg, "self_correction_scope", "word") or "word").lower()
        if self._self_correction_scope not in ("word", "chunk"):
            raise ValueError(
                f"self_correction_scope must be 'word' or 'chunk', got {self._self_correction_scope!r}"
            )
        self._delete_id = None
        if self._self_correction:
            hf_tok = self.tokenizer.tokenizer
            delete_token = str(getattr(self.core_cfg, "delete_token", "<|object_ref_start|>"))
            self._delete_id = hf_tok.convert_tokens_to_ids(delete_token)
            unk2 = getattr(hf_tok, "unk_token_id", None)
            if self._delete_id is None or (unk2 is not None and self._delete_id == unk2):
                raise ValueError(
                    f"self_correction delete_token {delete_token!r} is not a valid in-vocab token "
                    f"(id={self._delete_id}). Pick an unused tokenizer special token."
                )
            self._sc_rng = torch.Generator(device="cpu")
            self._sc_rng.manual_seed(1234)
            logging.info(
                "[SCRIPT] self-correction ON | delete_token=%r (id=%s) | inject_prob=%.2f | scope=%s "
                "-- learns to emit <del> to fix a mis-committed previous %s.",
                delete_token, self._delete_id, self._self_correction_prob, self._self_correction_scope,
                "chunk" if self._self_correction_scope == "chunk" else "word",
            )
        # Contiguous-text positions ("Option A"): must match how the data was
        # packed at train time; read from the MODEL config for from_pretrained.
        self._contiguous_text_positions = bool(getattr(self.core_cfg, "contiguous_text_positions", False))

        # --- Last-layer restricted history ---
        # When > 0, the TOP ``script_last_layer_restrict_num_layers`` LLM layer(s)
        # restrict a chunk (branch) query to only the most recent
        # ``script_last_layer_history_tokens`` history tokens (its audio + own
        # already-emitted tokens stay fully attended); lower layers are unchanged.
        # Both training and decode drive the layer stack manually to apply a
        # per-layer mask. Read from the MODEL config so from_pretrained rebuilds it.
        self._last_layer_history_tokens = max(int(getattr(self.core_cfg, "script_last_layer_history_tokens", 0) or 0), 0)
        self._last_layer_restrict_num_layers = max(
            int(getattr(self.core_cfg, "script_last_layer_restrict_num_layers", 1) or 1), 1
        )
        self._last_layer_restrict = self._last_layer_history_tokens > 0
        if self._last_layer_restrict:
            n_llm_layers = int(self.llm.config.num_hidden_layers)
            if not (1 <= self._last_layer_restrict_num_layers < n_llm_layers):
                raise ValueError(
                    f"script_last_layer_restrict_num_layers must be in [1, {n_llm_layers - 1}] "
                    f"(need >= 1 unrestricted lower layer), got {self._last_layer_restrict_num_layers}"
                )
            # These features change the packed layout / decode path in ways not yet
            # composed with the manual per-layer mask driver; guard rather than
            # silently produce a train/inference mismatch.
            for name, on in (
                ("shared_audio_track", self._shared_audio_track),
                ("self_correction", self._self_correction),
                ("contiguous_text_positions", self._contiguous_text_positions),
            ):
                if on:
                    raise ValueError(
                        f"script_last_layer_history_tokens is not supported together with {name} yet."
                    )
            logging.info("=" * 72)
            logging.info(
                "[SCRIPT] last-layer restricted history ON | history_tokens=%d | top_layers=%d/%d "
                "-- the top layer(s) attend only the last %d history tokens + this chunk's audio "
                "(lower layers unchanged).",
                self._last_layer_history_tokens,
                self._last_layer_restrict_num_layers,
                int(self.llm.config.num_hidden_layers),
                self._last_layer_history_tokens,
            )
            logging.info("=" * 72)

        # --- Windowed re-decoding self-correction (redecode) ---
        # Each chunk is re-decoded at lookahead levels 0..R over an (M+1)-chunk audio
        # window, conditioning on clean history; inference emits low-latency previews
        # and locks each chunk R chunks later. Read from the MODEL config so
        # from_pretrained rebuilds the decode path.
        self._redecode = bool(getattr(self.core_cfg, "redecode", False))
        self._redecode_depth = int(getattr(self.core_cfg, "redecode_depth", 0) or 0)
        if self._redecode:
            if self._audio_history_chunks < 1:
                raise ValueError("redecode requires audio_history_chunks >= 1 (window is M+1 chunks).")
            if self._redecode_depth <= 0:
                self._redecode_depth = self._audio_history_chunks  # default R = M
            if not (1 <= self._redecode_depth <= self._audio_history_chunks):
                raise ValueError(
                    f"redecode_depth must be in [1, audio_history_chunks={self._audio_history_chunks}], "
                    f"got {self._redecode_depth}"
                )
            for name, on in (
                ("audio_window_frames", self._audio_window_frames > 0),
                ("shared_audio_track", self._shared_audio_track),
                ("contiguous_text_positions", self._contiguous_text_positions),
                ("self_correction", self._self_correction),
                ("script_last_layer_history_tokens", self._last_layer_restrict),
            ):
                if on:
                    raise ValueError(f"redecode is not compatible with {name}.")
            logging.info("=" * 72)
            logging.info(
                "[SCRIPT] windowed re-decoding ON | window=%d chunks (M=%d) | depth R=%d "
                "-- each chunk is re-decoded with growing lookahead on clean history; "
                "inference emits previews and locks each chunk %d chunk(s) later.",
                self._audio_history_chunks + 1, self._audio_history_chunks, self._redecode_depth,
                self._redecode_depth,
            )
            logging.info("=" * 72)

        # Long-form encode: clips longer than this (seconds) are encoded with the
        # cache-aware STREAMING encoder in generate() (bounded memory) instead of a
        # single offline perception() forward (chunked_limited attention is still
        # O(T^2) -> OOM on multi-minute audio). Shorter clips keep the exact offline
        # path so leaderboard-length WER/throughput is unchanged. 0 disables (always
        # offline); a large value effectively disables streaming.
        self._stream_encode_min_sec = float(getattr(self.core_cfg, "stream_encode_min_sec", 40.0) or 0.0)

        # Inference max-history cap: retain at most this many of the most recently
        # emitted transcript tokens as the conditioning history (instruction always
        # kept); 0 = unlimited. Bounds the per-chunk prefill so long-form decode
        # cost stays linear (the text spine otherwise grows with the word count).
        self._max_history_tokens = int(getattr(self.core_cfg, "max_history_tokens", 0) or 0)

        hf_tok = self.tokenizer.tokenizer
        self._cc_vision_start_id = hf_tok.convert_tokens_to_ids(self.audio_open_token)
        self._cc_vision_end_id = hf_tok.convert_tokens_to_ids(self.audio_close_token)
        self._cc_eot_id = hf_tok.eos_token_id
        unk = getattr(hf_tok, "unk_token_id", None)
        for name, tid in (("audio_open", self._cc_vision_start_id), ("audio_close", self._cc_vision_end_id)):
            if tid is None or (unk is not None and tid == unk):
                raise ValueError(f"SCRIPT delimiter {name} is not a valid in-vocab token (id={tid}).")

        # The 4D packed mask requires a mask-capable attention backend.
        attn_impl = getattr(self.llm.config, "_attn_implementation", None)
        if attn_impl not in ("eager", "sdpa"):
            logging.warning(
                "SCRIPT needs a 4D-mask-capable attention backend; found %r. Forcing 'sdpa'.",
                attn_impl,
            )
            self.llm.config._attn_implementation = "sdpa"

        logging.info("=" * 72)
        logging.info(
            "[SCRIPT] packed spine+branch objective active | audio span %r/%r | eot=%s | "
            "audio_history_chunks=%d | audio_window_frames=%d | shared_audio_track=%s | "
            "contiguous_text_positions=%s | attn_impl=%s -- models p(words_k | text_history_<k, audio_{k-M..k}).",
            self.audio_open_token,
            self.audio_close_token,
            self._cc_eot_id,
            self._audio_history_chunks,
            self._audio_window_frames,
            self._shared_audio_track,
            self._contiguous_text_positions,
            self.llm.config._attn_implementation,
        )
        logging.info("=" * 72)

    # ------------------------------------------------------------------
    # Validation / test: true autoregressive WER (decode-only)
    #
    # The clean automodel StreamingSTTModel reports a teacher-forced val_acc and
    # never logs val_wer, so a ``ModelCheckpoint(monitor='val_wer')`` recipe
    # crashes at the first validation epoch ("could not find the monitored key").
    # The base SCRIPT was developed against ran a real chunk-synchronous greedy
    # decode and logged corpus val_wer; that path is restored here (ported from
    # that base) so checkpoint selection tracks actual streaming transcription
    # quality instead of next-token teacher-forced accuracy.
    # ------------------------------------------------------------------

    @property
    def val_max_new_tokens_per_chunk(self) -> int:
        """Resolve the autoregressive per-chunk generation cap for validation."""
        configured = getattr(self.core_cfg, "val_max_new_tokens_per_chunk", None)
        if configured is not None:
            if configured <= 0:
                raise ValueError("val_max_new_tokens_per_chunk must be positive")
            return int(configured)
        if self._chunk_size_repr > 0:
            return self._chunk_size_repr
        return 64

    def _validation_system_prompts(self, batch) -> Union[str, List[str]]:
        if getattr(batch, "cuts", None) is None:
            return self.val_system_prompt
        return [
            (cut.custom or {}).get(self.val_prompt_field, self.val_system_prompt)
            for cut in batch.cuts
        ]

    def on_validation_epoch_start(self) -> None:
        self._partial_wer_refs: dict = defaultdict(list)
        self._partial_wer_hyps: dict = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        # Gather decoded strings, then compute true corpus WER on every rank.
        # Averaging rank-local WERs would be incorrect when ranks see different
        # numbers of reference words.
        local_wer_data = {
            name: {"refs": self._partial_wer_refs[name], "hyps": self._partial_wer_hyps[name]}
            for name in self._partial_wer_refs
        }
        if torch.distributed.is_initialized():
            gathered_wer_data = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered_wer_data, local_wer_data)
        else:
            gathered_wer_data = [local_wer_data]

        wer = WER(normalize=True, verbose=False)
        has_wer_data = False
        for rank_data in gathered_wer_data:
            for name, values in rank_data.items():
                has_wer_data = has_wer_data or bool(values["refs"])
                wer.update(name, refs=values["refs"], hyps=values["hyps"])
        if has_wer_data:
            for metric_name, metric_value in wer.compute().items():
                log_name = "val_wer" if metric_name == "wer" else f"val_{metric_name}"
                self.log(log_name, metric_value.to(self.device), on_epoch=True, sync_dist=False)

        self._partial_wer_refs.clear()
        self._partial_wer_hyps.clear()

    def validation_step(self, batch, batch_idx: int):
        # Support multiple validation dataloaders ({name: batch} dict).
        if isinstance(batch, dict):
            for name, dataset_batch in batch.items():
                if dataset_batch is not None:
                    self._eval_step(dataset_batch, name, batch_idx)
        else:
            self._eval_step(batch, "val", batch_idx)

    def _eval_step(self, batch, name: str, batch_idx: int = 0) -> None:
        # Validation is decode-only: unlike the teacher-forced loss, autoregressive
        # WER needs neither word alignments nor constructed target turns -- just the
        # raw audio and the reference text.
        refs = list(batch.text)
        # Multi chunk-size: decode validation at a fixed, well-supported chunk
        # (default 14) instead of the longest candidate, whose look-ahead may be an
        # unsupported/slow streaming config that stalls the decode.
        # generate(chunk_size_override=...) pins the size for this decode.
        val_chunk = None
        if getattr(self, "_chunk_size_candidates", None):
            val_chunk = getattr(self.core_cfg, "val_chunk_size", None)
            if val_chunk is None:
                val_chunk = 14 if 14 in self._chunk_size_candidates else max(self._chunk_size_candidates)
            val_chunk = int(val_chunk)
        hyps = self.generate(
            audios=batch.audios,
            audio_lens=batch.audio_lens,
            system_prompt=self._validation_system_prompts(batch),
            max_new_tokens=self.val_max_new_tokens_per_chunk,
            generation_config=GenerationConfig(do_sample=False),
            chunk_size_override=val_chunk,
        )
        self._partial_wer_refs[name].extend(refs)
        self._partial_wer_hyps[name].extend(hyps)

        if batch_idx % self.core_cfg.log_every_n_steps == 0 and refs and hyps:
            logging.info(
                "[%s] autoregressive batch %d (max %d tokens/chunk)\n  ref: `%s`\n  hyp: `%s`",
                name,
                batch_idx,
                self.val_max_new_tokens_per_chunk,
                refs[0],
                hyps[0],
            )

    # Test delegates to the validation (decode-only WER) logic.
    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def _resolve_llm_core(self):
        """Return ``(layers, norm, rotary_emb, lm_head)`` for the LLM.

        SCRIPT drives the decoder layer stack manually (custom per-layer masks).
        The base StreamingSTTModel on this branch does not expose this helper, so it
        is provided here. Unwraps the PEFT/LoRA wrapper (``get_base_model()`` returns
        the underlying ``*ForCausalLM`` with adapters still injected) and walks down
        to the decoder body that owns ``layers`` / ``norm`` / ``rotary_emb``.
        """
        llm = self.llm
        base = llm.get_base_model() if hasattr(llm, "get_base_model") else llm
        lm_head = getattr(base, "lm_head", None)
        if lm_head is None:
            lm_head = self.llm.lm_head
        core = base
        while not hasattr(core, "layers") and hasattr(core, "model"):
            core = core.model
        if not hasattr(core, "layers"):
            raise AttributeError(f"Could not locate decoder layers on LLM of type {type(base).__name__}")
        return core.layers, core.norm, core.rotary_emb, lm_head

    def _build_input_embeds_indexed(
        self, input_tokens: Tensor, audios: Tensor, audio_lens: Tensor, audio_frame_index: Tensor
    ) -> dict:
        """Like ``_build_input_embeds`` but fills each audio slot by an EXPLICIT
        global encoder-frame index (gather), not a positional cumsum.

        Needed when ``audio_history_chunks > 0``: a branch's window spans multiple
        chunks and the same frame appears in multiple branches, so the 1:1 cumsum
        mapping no longer holds. Out-of-range indices (last-chunk ceiling) gather a
        zero-padded frame, matching ``interleave_embeddings``.
        """
        audio_mask = input_tokens == AUDIO_TOKEN_IDX  # (B, L)
        text_tokens = input_tokens.where(~audio_mask, torch.zeros_like(input_tokens))
        text_embeds = self.embed_tokens(text_tokens)  # (B, L, H)
        audio_embs, _ = self.perception(input_signal=audios, input_signal_length=audio_lens)  # (B, T_enc, H)
        B, L = input_tokens.shape
        H = audio_embs.shape[2]
        T_enc = audio_embs.shape[1]
        max_idx = int(audio_frame_index.max().item()) if audio_frame_index.numel() else -1
        if max_idx >= T_enc:
            audio_embs = F.pad(audio_embs, (0, 0, 0, max_idx - T_enc + 1))
        gather_idx = audio_frame_index.clamp(min=0).unsqueeze(-1).expand(B, L, H)
        audio_at = torch.gather(audio_embs, dim=1, index=gather_idx)  # (B, L, H)
        embeds = torch.where(audio_mask.unsqueeze(-1), audio_at, text_embeds)
        attention_mask = input_tokens != self.text_pad_id
        return {"input_embeds": embeds, "attention_mask": attention_mask}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _chunk_logits(self, input_tokens, position_ids, seg_ids, prefix_len, valid, audios, audio_lens, audio_frame_index):
        """Forward the (regular per-branch-audio) packed batch -> (B, T, V) logits."""
        if audio_frame_index is not None:
            inputs = self._build_input_embeds_indexed(input_tokens, audios, audio_lens, audio_frame_index)
        else:
            inputs = self._build_input_embeds(input_tokens, audios, audio_lens)
        input_embeds = inputs["input_embeds"]
        mask = build_script_mask(seg_ids, position_ids, prefix_len, valid, input_embeds.dtype)
        out = self.llm(
            inputs_embeds=input_embeds, attention_mask=mask, position_ids=position_ids,
            use_cache=False, return_dict=True,
        )
        return out["logits"]

    def training_step(self, batch: ScriptBatch, batch_idx: int):
        if self._self_correction and getattr(batch, "chunk_meta", None) is not None:
            return self._self_correction_training_step(batch, batch_idx)

        # Keep frozen modules in eval mode (disables dropout / BN updates).
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        # Match the encoder look-ahead to the per-batch chunk size (no-op unless
        # multi chunk-size + att_context_size set).
        self._set_encoder_att_context(getattr(batch, "chunk_size", None))

        # Interleave encoder frames into the packed AUDIO_TOKEN_IDX positions.
        # With an audio history window (audio_frame_index set) frames are reused
        # across branches -> gather by explicit index; else the cumsum fill.
        if getattr(batch, "audio_frame_index", None) is not None:
            inputs = self._build_input_embeds_indexed(
                batch.input_tokens, batch.audios, batch.audio_lens, batch.audio_frame_index
            )
        else:
            inputs = self._build_input_embeds(batch.input_tokens, batch.audios, batch.audio_lens)
        input_embeds = inputs["input_embeds"]

        if self._shared_audio_track:
            mask = build_shared_audio_chunk_mask(
                batch.seg_ids, batch.position_ids, batch.prefix_len,
                batch.win_start, batch.win_end, batch.audio_frame_index, batch.valid, input_embeds.dtype,
            )
        else:
            mask = build_script_mask(
                batch.seg_ids, batch.position_ids, batch.prefix_len, batch.valid, input_embeds.dtype
            )

        if self._last_layer_restrict:
            # Top layer(s) use a restricted (last-N-history) mask; lower layers use
            # the normal mask. Drive the layer stack manually (one mask per layer).
            mask_restricted = build_script_mask(
                batch.seg_ids, batch.position_ids, batch.prefix_len, batch.valid, input_embeds.dtype,
                history_window=self._last_layer_history_tokens,
            )
            layers, norm, rotary_emb, lm_head = self._resolve_llm_core()
            logits = run_script_layers_split(
                layers=layers, norm=norm, rotary_emb=rotary_emb, lm_head=lm_head,
                inputs_embeds=input_embeds, position_ids=batch.position_ids,
                mask_lower=mask, mask_top=mask_restricted,
                num_top_layers=self._last_layer_restrict_num_layers,
            )  # (B, T, V)
        else:
            out = self.llm(
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
            logging.warning("Batch %d: num_targets is 0 — skipping (zero loss).", batch_idx)
            return {"loss": torch.tensor(0.0, device=logits.device, requires_grad=True)}

        with loss_parallel():
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                target_ids.flatten(0, 1),
                reduction="mean",
                ignore_index=IGNORE_INDEX,
            )

        train_metrics = {
            "loss": loss,
            "learning_rate": torch.as_tensor(
                self.trainer.optimizers[0].param_groups[0]["lr"] if self._trainer is not None else 0
            ),
        }
        if self.core_cfg.log_detailed_train_metrics:
            B, T = batch.input_tokens.shape
            train_metrics.update(
                {
                    "num_targets": num_targets.float(),
                    "sequence_length": float(T),
                    "target_to_input_ratio": num_targets / (B * T),
                }
            )
        self.log_dict(train_metrics, on_step=True)
        return {"loss": loss}

    def _self_correction_training_step(self, batch: ScriptBatch, batch_idx: int):
        """DAgger-style self-correction step.

        (1) A no-grad teacher-forced pass on the ground-truth-history batch gives the
            model's argmax per branch. (2) Where a chunk's LAST word is mispredicted,
            the NEXT branch is rebuilt with that wrong word ``W'`` as its committed
            history tail and target ``<del> w_prev w_k``. (3) A grad forward on the
            rebuilt batch takes the CE loss. Error stats are logged and a sample
            correction is printed every ``self_correction_log_every`` steps.
        """
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()
        self._set_encoder_att_context(getattr(batch, "chunk_size", None))

        # 1) Forced (teacher-forced) argmax on the correct-history batch.
        with torch.no_grad():
            logits = self._chunk_logits(
                batch.input_tokens, batch.position_ids, batch.seg_ids, batch.prefix_len,
                batch.valid, batch.audios, batch.audio_lens, batch.audio_frame_index,
            )
            argmax = logits.argmax(dim=-1)  # (B, T)

        # 2) Detect last-word errors; build per-example corrupt_prev; rebuild.
        B = batch.input_tokens.shape[0]
        n_eligible = n_error = n_corrupt = 0
        sample = None
        corrupt_examples = []
        for b in range(B):
            instruction_ids, chunks = batch.chunk_meta[b]
            corrupt_prev = [None] * len(chunks)
            seg_b, tgt_b, am_b = batch.seg_ids[b], batch.target_tokens[b], argmax[b]
            chunk_scope = self._self_correction_scope == "chunk"
            for kc in range(len(chunks) - 1):  # an error in chunk kc corrupts chunk kc+1
                # The corrected unit is chunk kc's last word (word scope) or all of
                # its words (chunk scope). Both need chunk kc to have emitted words.
                truth = list(chunks[kc].target_ids) if chunk_scope else list(chunks[kc].last_word_ids)
                nword = len(chunks[kc].target_ids)
                if len(truth) == 0 or nword == 0:
                    continue
                n_eligible += 1
                sup = ((seg_b == kc + 1) & (tgt_b != IGNORE_INDEX)).nonzero(as_tuple=True)[0]
                if sup.numel() < nword:
                    continue
                # Predicted unit: the whole chunk (chunk scope) or just its last word.
                pred_words = am_b[sup[:nword]]
                pred_unit = pred_words.tolist() if chunk_scope else pred_words[nword - len(truth):].tolist()
                if pred_unit != truth:
                    n_error += 1
                    if float(torch.rand((), generator=self._sc_rng).item()) < self._self_correction_prob:
                        corrupt_prev[kc + 1] = pred_unit
                        n_corrupt += 1
                        if sample is None:
                            sample = (truth, pred_unit, list(chunks[kc + 1].target_ids))
            corrupt_examples.append(
                build_packed_chunk_example(
                    instruction_ids=instruction_ids,
                    chunks=chunks,
                    vision_start_id=self._cc_vision_start_id,
                    vision_end_id=self._cc_vision_end_id,
                    eot_id=self._cc_eot_id,
                    corrupt_prev=corrupt_prev,
                    delete_id=self._delete_id,
                    correction_scope=self._self_correction_scope,
                    audio_history_chunks=self._audio_history_chunks,
                    audio_window_frames=self._audio_window_frames,
                )
            )

        packed = collate_packed_chunk_examples(corrupt_examples, pad_id=self.tokenizer.pad_id)
        dev = self.device
        windowed = self._audio_history_chunks > 0 or self._audio_window_frames > 0
        target_tokens = packed.target_ids.to(dev)

        # 3) Grad forward on the corrupted batch.
        logits = self._chunk_logits(
            packed.input_ids.to(dev), packed.position_ids.to(dev), packed.seg_ids.to(dev),
            packed.prefix_len.to(dev), packed.valid.to(dev), batch.audios, batch.audio_lens,
            packed.audio_frame_index.to(dev) if windowed else None,
        )
        num_targets = (target_tokens != IGNORE_INDEX).long().sum()
        if num_targets == 0:
            return {"loss": torch.tensor(0.0, device=logits.device, requires_grad=True)}
        with loss_parallel():
            loss = F.cross_entropy(
                logits.flatten(0, 1), target_tokens.flatten(0, 1),
                reduction="mean", ignore_index=IGNORE_INDEX,
            )

        # 4) Stats + periodic sample printout.
        err_rate = (n_error / n_eligible) if n_eligible else 0.0
        self.log_dict(
            {
                "loss": loss,
                "learning_rate": torch.as_tensor(
                    self.trainer.optimizers[0].param_groups[0]["lr"] if self._trainer is not None else 0
                ),
                "sc/eligible_chunks": float(n_eligible),
                "sc/errors": float(n_error),
                "sc/error_rate": float(err_rate),
                "sc/injected": float(n_corrupt),
            },
            on_step=True,
        )
        if self._self_correction_log_every and (batch_idx % self._self_correction_log_every == 0) and sample is not None:
            w_prev, wprime, w_cur = sample
            logging.info(
                "[self-correction] step %d | eligible=%d errors=%d (%.1f%%) injected=%d | "
                "example: model committed %r for true %r -> target: <del> %r then chunk %r",
                batch_idx, n_eligible, n_error, 100.0 * err_rate, n_corrupt,
                self.tokenizer.ids_to_text(wprime), self.tokenizer.ids_to_text(w_prev),
                self.tokenizer.ids_to_text(w_prev), self.tokenizer.ids_to_text(w_cur) if w_cur else "",
            )
        return {"loss": loss}

    # ``backward`` (loss_parallel wrapper) is inherited from StreamingSTTModel.

    # ------------------------------------------------------------------
    # Inference (streaming spine-KV decode; audio evicted per chunk)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_frames_streaming(self, wav: Tensor, n_samples: int, N: int, chunk_samples: int) -> Tensor:
        """Encode ONE clip's waveform into ``(T_enc, H)`` LLM-space frames via the
        cache-aware STREAMING encoder path, keeping memory bounded to one window.

        This mirrors the interleaving model's ``_generate_dynamic_streaming`` refill
        loop (feature buffer -> ``perception(..., streaming=True)`` -> carried
        encoder cache), but simply concatenates every step's frames instead of
        feeding them to the LLM one at a time. Because the encoder is
        ``chunked_limited`` with a fixed per-layer cache, this reproduces the same
        limited-context frames the model was trained on — but without the offline
        forward's O(T^2) attention that OOMs on multi-minute clips.

        Requires ``setup_streaming_params()`` to have run (the caller does so via
        ``_set_encoder_att_context(cs, recompute_streaming=True)``).
        """
        device = wav.device
        enc_param = next(self.perception.encoder.parameters())
        cache_lc, cache_lt, cache_lcl = self.perception.get_initial_cache_state(
            batch_size=1, dtype=enc_param.dtype, device=device
        )
        buf = self.get_audio_feature_buffer(batch_size=1, chunk_size_override=N)
        out: List[Tensor] = []
        pos = 0
        while pos < n_samples:
            end = min(pos + chunk_samples, n_samples)
            seg = wav[pos:end]
            length = end - pos
            if seg.shape[0] < chunk_samples:
                seg = F.pad(seg, (0, chunk_samples - seg.shape[0]))
            features, right_paddings = buf.update([Frame(samples=seg, stream_id=0, length=length)])
            processed_signal = torch.stack(features).type_as(self.embed_tokens.weight)  # (1, D, T_mel)
            processed_signal_length = torch.tensor(
                [processed_signal.shape[-1] - int(right_paddings[0])], device=device
            ).long()
            emb, emb_len, new_cache = self.perception(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=cache_lc,
                cache_last_time=cache_lt,
                cache_last_channel_len=cache_lcl,
                streaming=True,
            )
            cache_lc = new_cache["cache_last_channel"]
            cache_lt = new_cache["cache_last_time"]
            cache_lcl = new_cache["cache_last_channel_len"]
            n_enc = int(emb_len[0].item()) if emb_len is not None else emb.shape[1]
            if n_enc > 0:
                out.append(emb[0, :n_enc])
            pos = end
        if not out:
            return torch.zeros(0, self.llm.config.hidden_size, device=device, dtype=self.embed_tokens.weight.dtype)
        return torch.cat(out, dim=0)  # (T_enc, H)

    def _get_word_start_insert_id(self) -> Optional[int]:
        """Token id of a lone leading-space (word-start) subword, INSERTED in front
        of a chunk's first token when that token is not itself a word-start, so the
        chunk cannot merge onto the previous word. Resolved once and cached; returns
        ``None`` (insertion disabled) if the tokenizer has no such token."""
        if self._word_start_insert_id is None:
            hf_tok = self.tokenizer.tokenizer
            cand = None
            for marker in ("\u0120", "\u2581"):  # GPT-2/Qwen "Ġ", SentencePiece "▁"
                tid = hf_tok.convert_tokens_to_ids(marker)
                unk = getattr(hf_tok, "unk_token_id", None)
                if tid is not None and tid >= 0 and (unk is None or tid != unk) and self._is_word_start(tid):
                    cand = int(tid)
                    break
            if cand is None:
                logging.warning("[SCRIPT] no lone word-start (space) token found; chunk word-start insertion disabled.")
                self._word_start_insert_id = -1  # sentinel: resolved-but-absent
            else:
                self._word_start_insert_id = cand
        return self._word_start_insert_id if self._word_start_insert_id != -1 else None

    def _state_machine_encode_batched(self, audios: Tensor, audio_lens: Tensor, cs: int) -> List[Tensor]:
        """Batched cache-aware STREAMING encode: advance every still-active stream
        one ``chunk_samples`` step per iteration through a single shared feature
        buffer + per-stream encoder cache, in ONE ``perception(streaming=True)``
        call per step (mirrors ``StreamingSTTModel._generate_dynamic_streaming``'s
        refill loop). Returns per-utterance ``(T_enc_b, H)`` frame tensors.

        Because the encoder is ``chunked_limited`` with per-stream caches, batching
        is numerically identical to encoding each utterance alone
        (:meth:`_encode_frames_streaming`) -- it only parallelizes the work.
        """
        device = self.device
        sr = self.core_cfg.sample_rate
        N = max(int(cs), 1)
        chunk_samples = math.ceil(N * self.core_cfg.frame_length_in_secs * sr)
        B = audios.shape[0]
        n_samples = [int(audio_lens[b].item()) for b in range(B)]

        # Streaming inference needs the cache-aware config recomputed for cs.
        self._set_encoder_att_context(cs, recompute_streaming=True)
        enc = self.perception.encoder
        if getattr(enc, "streaming_cfg", None) is None:
            enc.setup_streaming_params()

        enc_param = next(self.perception.encoder.parameters())
        cache_lc, cache_lt, cache_lcl = self.perception.get_initial_cache_state(
            batch_size=B, dtype=enc_param.dtype, device=device
        )
        buf = self.get_audio_feature_buffer(batch_size=B, chunk_size_override=N)

        frames_by_stream: List[List[Tensor]] = [[] for _ in range(B)]  # per-stream list of (n_enc, H) chunks
        audio_pos = [0] * B

        while True:
            active = [b for b in range(B) if audio_pos[b] < n_samples[b]]
            if not active:
                break
            idx_t = torch.tensor(active, device=device)

            frames = []
            for b in active:
                start = audio_pos[b]
                end = min(start + chunk_samples, n_samples[b])
                seg = audios[b, start:end]
                length = end - start
                if seg.shape[0] < chunk_samples:
                    seg = F.pad(seg, (0, chunk_samples - seg.shape[0]))
                frames.append(Frame(samples=seg, stream_id=b, length=length))
                audio_pos[b] = end

            features, right_paddings = buf.update(frames)
            processed_signal = torch.stack(features).type_as(self.embed_tokens.weight)  # (S, D, T_mel)
            processed_signal_length = torch.tensor(
                [processed_signal.shape[-1] - int(rp) for rp in right_paddings], device=device
            ).long()

            # Slice the per-stream encoder cache to the active subset, then scatter back.
            if cache_lc is not None:
                sub_lc = cache_lc.index_select(1, idx_t)
                sub_lt = cache_lt.index_select(1, idx_t)
                sub_lcl = cache_lcl[idx_t]
            else:
                sub_lc = sub_lt = sub_lcl = None

            emb, emb_len, new_cache = self.perception(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=sub_lc,
                cache_last_time=sub_lt,
                cache_last_channel_len=sub_lcl,
                streaming=True,
            )
            if new_cache is not None:
                for i, b in enumerate(active):
                    cache_lc[:, b] = new_cache["cache_last_channel"][:, i]
                    cache_lt[:, b] = new_cache["cache_last_time"][:, i]
                    cache_lcl[b] = new_cache["cache_last_channel_len"][i]

            for i, b in enumerate(active):
                n_enc = int(emb_len[i].item()) if emb_len is not None else emb.shape[1]
                if n_enc > 0:
                    frames_by_stream[b].append(emb[i, :n_enc])

        H = self.llm.config.hidden_size
        dtype = self.embed_tokens.weight.dtype
        out: List[Tensor] = []
        for b in range(B):
            if frames_by_stream[b]:
                out.append(torch.cat(frames_by_stream[b], dim=0))  # (T_enc_b, H)
            else:
                out.append(torch.zeros(0, H, device=device, dtype=dtype))
        return out

    def _state_machine_decode(
        self,
        audios: Tensor,
        audio_lens: Tensor,
        instruction_ids_list: List[List[int]],
        cs: int,
        max_new_tokens: int,
        max_history_tokens: int,
        return_chunk_ids: bool = False,
        warn_chunk_word_start: bool = False,
        insert_word_start_id: Optional[int] = None,
    ):
        """Batched SCRIPT state-machine decode: batched cache-aware streaming encode
        (:meth:`_state_machine_encode_batched`) followed by the batched spine/branch
        decoder (:func:`batched_stream_decode_script`). Returns that decoder's
        contract: ``emitted`` (per-utterance flat token lists) and, when requested,
        per-token ``chunk_ids``."""
        frames_list = self._state_machine_encode_batched(audios, audio_lens, cs)
        return batched_stream_decode_script(
            llm=self.llm,
            embed_tokens=self.embed_tokens,
            instruction_ids_list=instruction_ids_list,
            frames_list=frames_list,
            chunk_size=cs,
            vision_start_id=self._cc_vision_start_id,
            vision_end_id=self._cc_vision_end_id,
            eot_id=self._cc_eot_id,
            pad_id=self.text_pad_id,
            max_new_tokens=max_new_tokens,
            device=self.device,
            audio_history_chunks=self._audio_history_chunks,
            contiguous_text_positions=self._contiguous_text_positions,
            max_history_tokens=max_history_tokens,
            audio_window_frames=self._audio_window_frames,
            return_chunk_ids=return_chunk_ids,
            is_word_start=self._is_word_start if (warn_chunk_word_start or insert_word_start_id is not None) else None,
            warn_chunk_word_start=warn_chunk_word_start,
            insert_word_start_id=insert_word_start_id,
        )

    def generate(
        self,
        audios: Tensor,
        audio_lens: Tensor,
        system_prompt: Union[str, List[str]] = "Transcribe the audio into text.",
        max_new_tokens: int = 64,
        generation_config: Optional[GenerationConfig] = None,
        chunk_size_override: Optional[int] = None,
        use_state_machine_inference: bool = False,
        **generation_kwargs,
    ) -> List[str]:
        """Chunk-by-chunk streaming transcription.

        Encodes the audio once (same ``perception`` call as training), then for
        each utterance runs the greedy spine+branch decode
        (:func:`stream_decode_script`): a growing plain-text spine KV
        cache of ``instruction + emitted words`` plus a transient per-chunk audio
        branch that is evicted after its words are decoded.

        With ``use_state_machine_inference=True`` the audio is instead ingested
        incrementally through the cache-aware streaming encoder (shared feature
        buffer + per-stream carried cache, one ``perception`` call per step across
        the whole batch) and decoded with the batched spine/branch decoder
        (:meth:`_state_machine_decode`), mirroring ``StreamingSTTModel``. Supported
        for plain SCRIPT only (not redecode / last_layer / shared_audio).
        """
        cs = self._resolve_inference_chunk_size(chunk_size_override)
        if cs <= 0:
            raise ValueError(f"SCRIPT generate requires a positive chunk size, got {cs}")

        B = audios.shape[0]
        if isinstance(system_prompt, str):
            system_prompt = [system_prompt] * B

        # Chunk-start invariant controls (each SCRIPT chunk should begin a new word).
        # ``debug_chunk_word_start`` logs when a chunk's first token is not a word-start.
        # ``force_chunk_word_start`` fixes it by INSERTING a leading-space token in front
        # (never restricts what the model may emit, so it cannot starve a chunk empty).
        warn_chunk_word_start = bool(generation_kwargs.pop("debug_chunk_word_start", False))
        force_chunk_word_start = bool(generation_kwargs.pop("force_chunk_word_start", False))
        insert_word_start_id = self._get_word_start_insert_id() if force_chunk_word_start else None

        if use_state_machine_inference:
            if self._redecode or self._last_layer_restrict or self._shared_audio_track:
                raise NotImplementedError(
                    "use_state_machine_inference is only supported for plain SCRIPT decode "
                    "(not redecode / last_layer_restrict / shared_audio_track)."
                )
            if bool(generation_kwargs.pop("self_correct", False)):
                raise NotImplementedError("self_correct is not supported with use_state_machine_inference.")
            if bool(generation_kwargs.pop("return_raw", False)):
                raise NotImplementedError("return_raw is not supported with use_state_machine_inference.")
            mh = generation_kwargs.pop("max_history_tokens", None)
            max_history_tokens = int(mh) if mh is not None else self._max_history_tokens
            return_word_latency = bool(generation_kwargs.pop("return_word_latency", False))
            instruction_ids_list = [self.tokenizer.text_to_ids(system_prompt[b] + "\n") for b in range(B)]
            out = self._state_machine_decode(
                audios,
                audio_lens,
                instruction_ids_list,
                cs,
                max_new_tokens,
                max_history_tokens,
                return_chunk_ids=return_word_latency,
                warn_chunk_word_start=warn_chunk_word_start,
                insert_word_start_id=insert_word_start_id,
            )
            emitted = out[0] if isinstance(out, tuple) else out
            hyps = [self.tokenizer.ids_to_text(ids) if ids else "" for ids in emitted]
            if return_word_latency:
                chunk_ids = out[1]
                frame_len = float(self.core_cfg.frame_length_in_secs)
                lat = [self._word_emission_times(emitted[b], chunk_ids[b], cs, frame_len) for b in range(B)]
                return hyps, lat
            return hyps

        # --- 1) Encode audio -> per-utterance frame tensors ---
        # Short clips: exact batched OFFLINE encode (unchanged; keeps leaderboard
        # WER/throughput). Long clips: cache-aware STREAMING encode so encoder
        # memory stays bounded (offline chunked_limited attention is O(T^2) and
        # OOMs on multi-minute audio). ``stream_encode`` (True/False) forces either
        # path; otherwise clips longer than ``_stream_encode_min_sec`` stream.
        sr = self.core_cfg.sample_rate
        force = generation_kwargs.pop("stream_encode", None)
        thr = float(getattr(self, "_stream_encode_min_sec", 0.0) or 0.0)
        if isinstance(force, bool):
            stream_flags = [force] * B
        else:
            stream_flags = [thr > 0.0 and (int(audio_lens[b].item()) / sr) > thr for b in range(B)]

        frames_list: List[Optional[Tensor]] = [None] * B

        # Long clips: cache-aware streaming encode (bounded memory), per utterance.
        stream_idx = [b for b in range(B) if stream_flags[b]]
        if stream_idx:
            self._set_encoder_att_context(cs, recompute_streaming=True)
            enc = self.perception.encoder
            if getattr(enc, "streaming_cfg", None) is None:
                enc.setup_streaming_params()
            N = max(int(cs), 1)
            chunk_samples = math.ceil(N * self.core_cfg.frame_length_in_secs * sr)
            for b in stream_idx:
                n = int(audio_lens[b].item())
                frames_list[b] = self._encode_frames_streaming(audios[b, :n], n, N, chunk_samples)

        # Short clips: batched OFFLINE encode. Sub-batched because a single
        # full-batch encode of long, length-sorted clips overflows 32-bit CUDA
        # indexing in the subsampling conv; sub-batching keeps each conv tensor bounded.
        offline_idx = [b for b in range(B) if not stream_flags[b]]
        if offline_idx:
            self._set_encoder_att_context(cs)
            enc_bs = max(1, int(self.encode_batch_size))
            for i in range(0, len(offline_idx), enc_bs):
                grp = offline_idx[i : i + enc_bs]
                grp_t = torch.tensor(grp, device=audio_lens.device)
                sl = audio_lens[grp_t]
                max_len = int(sl.max().item())
                sig = audios[grp_t, :max_len]
                emb, emb_len = self.perception(input_signal=sig, input_signal_length=sl)  # (b, T_enc, H)
                for j, b in enumerate(grp):
                    frames_list[b] = emb[j, : int(emb_len[j].item())].clone()

        # --- 2) Batched chunk-synchronous streaming decode ---
        # Per-call override of the max-history cap, else the model-config default.
        mh = generation_kwargs.pop("max_history_tokens", None)
        max_history_tokens = int(mh) if mh is not None else self._max_history_tokens
        # Optionally return per-word emission latency (proxy): the time from audio
        # start to when each word was emitted, i.e. the end of the chunk of the
        # word's LAST subword. Off by a constant from true latency (the mean word
        # onset time), which is fine for comparing models/latency settings.
        return_word_latency = bool(generation_kwargs.pop("return_word_latency", False))
        # Return the RAW emission stream with <del> markers kept in place (so callers
        # can dump "A B <del> C" and see where self-correction fired). Only meaningful
        # with self-correction; harmless (== hyp) otherwise.
        return_raw = bool(generation_kwargs.pop("return_raw", False))
        # Popped here (before any decode-path branch) so it never leaks into the
        # non-redecode decoders as an unexpected kwarg. Only used by the redecode
        # path below; harmlessly ignored otherwise.
        self_correct = bool(generation_kwargs.pop("self_correct", False))
        instruction_ids_list = [self.tokenizer.text_to_ids(system_prompt[b] + "\n") for b in range(B)]

        if self._redecode:
            if return_raw:
                raise NotImplementedError("return_raw is not supported with redecode (no <del> token).")
            # Optionally also return the j=0 (zero-lookahead) preview transcript, i.e.
            # the low-latency streaming operating point alongside the locked output.
            return_provisional = bool(generation_kwargs.pop("return_provisional", False))
            # Self-correction switch. DEFAULT OFF: run a plain non-corrective stream
            # (effective redecode_depth=0) -- each new chunk is decoded once and
            # appended, past chunks are never revisited. self_correct=True re-decodes
            # each chunk with growing lookahead (up to the trained depth R) and emits
            # the LOCKED/corrected stream (which lags R chunks). The model is trained
            # either way; this only chooses which stream generate() emits.
            eff_depth = self._redecode_depth if self_correct else 0
            out = batched_stream_decode_redecode(
                llm=self.llm,
                embed_tokens=self.embed_tokens,
                instruction_ids_list=instruction_ids_list,
                frames_list=frames_list,
                chunk_size=cs,
                vision_start_id=self._cc_vision_start_id,
                vision_end_id=self._cc_vision_end_id,
                eot_id=self._cc_eot_id,
                pad_id=self.text_pad_id,
                audio_history_chunks=self._audio_history_chunks,
                redecode_depth=eff_depth,
                max_new_tokens=max_new_tokens,
                device=self.device,
                return_chunk_ids=return_word_latency,
                return_provisional=return_provisional,
            )
            extras = list(out[1:]) if isinstance(out, tuple) else []
            emitted = out[0] if isinstance(out, tuple) else out
            hyps = [self.tokenizer.ids_to_text(ids) if ids else "" for ids in emitted]
            result = [hyps]
            ei = 0
            if return_word_latency:
                lock_steps = extras[ei]
                ei += 1
                frame_len = float(self.core_cfg.frame_length_in_secs)
                result.append(
                    [self._word_emission_times(emitted[b], lock_steps[b], cs, frame_len) for b in range(B)]
                )
            if return_provisional:
                prov = extras[ei]
                ei += 1
                result.append([self.tokenizer.ids_to_text(ids) if ids else "" for ids in prov])
            return tuple(result) if len(result) > 1 else hyps

        if self._last_layer_restrict:
            if return_raw:
                raise NotImplementedError("return_raw is not supported with the last-layer restricted decode.")
            layers, norm, rotary_emb, lm_head = self._resolve_llm_core()
            out = batched_stream_decode_script_last_layer(
                layers=layers, norm=norm, rotary_emb=rotary_emb, lm_head=lm_head,
                embed_tokens=self.embed_tokens,
                instruction_ids_list=instruction_ids_list,
                frames_list=frames_list,
                chunk_size=cs,
                vision_start_id=self._cc_vision_start_id,
                vision_end_id=self._cc_vision_end_id,
                eot_id=self._cc_eot_id,
                pad_id=self.text_pad_id,
                num_top_layers=self._last_layer_restrict_num_layers,
                history_tokens=self._last_layer_history_tokens,
                max_new_tokens=max_new_tokens,
                device=self.device,
                audio_history_chunks=self._audio_history_chunks,
                audio_window_frames=self._audio_window_frames,
                max_history_tokens=max_history_tokens,
                return_chunk_ids=return_word_latency,
            )
            emitted = out[0] if isinstance(out, tuple) else out
            hyps = [self.tokenizer.ids_to_text(ids) if ids else "" for ids in emitted]
            if return_word_latency:
                chunk_ids = out[1]
                frame_len = float(self.core_cfg.frame_length_in_secs)
                lat = [self._word_emission_times(emitted[b], chunk_ids[b], cs, frame_len) for b in range(B)]
                return hyps, lat
            return hyps

        if self._shared_audio_track:
            if return_word_latency or return_raw:
                raise NotImplementedError("return_word_latency / return_raw not supported with shared_audio_track yet.")
            emitted = batched_shared_audio_decode(
                llm=self.llm,
                embed_tokens=self.embed_tokens,
                instruction_ids_list=instruction_ids_list,
                frames_list=frames_list,
                chunk_size=cs,
                vision_end_id=self._cc_vision_end_id,
                eot_id=self._cc_eot_id,
                pad_id=self.text_pad_id,
                max_new_tokens=max_new_tokens,
                device=self.device,
                audio_window_frames=self._audio_window_frames,
                audio_history_chunks=self._audio_history_chunks,
                max_history_tokens=max_history_tokens,
            )
            return [self.tokenizer.ids_to_text(ids) if ids else "" for ids in emitted]

        out = batched_stream_decode_script(
            llm=self.llm,
            embed_tokens=self.embed_tokens,
            instruction_ids_list=instruction_ids_list,
            frames_list=frames_list,
            chunk_size=cs,
            vision_start_id=self._cc_vision_start_id,
            vision_end_id=self._cc_vision_end_id,
            eot_id=self._cc_eot_id,
            pad_id=self.text_pad_id,
            max_new_tokens=max_new_tokens,
            device=self.device,
            audio_history_chunks=self._audio_history_chunks,
            contiguous_text_positions=self._contiguous_text_positions,
            max_history_tokens=max_history_tokens,
            return_chunk_ids=return_word_latency,
            audio_window_frames=self._audio_window_frames,
            delete_id=self._delete_id if self._self_correction else None,
            is_word_start=(
                self._is_word_start
                if (self._self_correction or warn_chunk_word_start or insert_word_start_id is not None)
                else None
            ),
            correction_scope=self._self_correction_scope,
            return_raw=return_raw,
            warn_chunk_word_start=warn_chunk_word_start,
            insert_word_start_id=insert_word_start_id,
        )
        # The decoder returns emitted, then (chunk_ids if requested), then (raw if
        # requested), in that fixed order. Unpack accordingly.
        extras = list(out[1:]) if isinstance(out, tuple) else []
        emitted = out[0] if isinstance(out, tuple) else out
        hyps = [self.tokenizer.ids_to_text(ids) if ids else "" for ids in emitted]

        result = [hyps]
        ei = 0
        if return_word_latency:
            chunk_ids = extras[ei]
            ei += 1
            frame_len = float(self.core_cfg.frame_length_in_secs)
            result.append([self._word_emission_times(emitted[b], chunk_ids[b], cs, frame_len) for b in range(B)])
        if return_raw:
            raw = extras[ei]
            ei += 1
            result.append([self._raw_to_text(r) for r in raw])
        return tuple(result) if len(result) > 1 else hyps

    def _raw_to_text(self, tokens: List[int]) -> str:
        """Render the raw emission stream with the delete token shown as ``<del>``,
        so a correction reads e.g. ``the cat <del> cat sat``."""
        if not tokens:
            return ""
        parts: List[str] = []
        seg: List[int] = []
        for t in tokens:
            if self._delete_id is not None and t == self._delete_id:
                parts.append(self.tokenizer.ids_to_text(seg) if seg else "")
                parts.append("<del>")
                seg = []
            else:
                seg.append(t)
        parts.append(self.tokenizer.ids_to_text(seg) if seg else "")
        return " ".join(p for p in parts if p != "").strip()

    def _word_emission_times(self, tokens: List[int], chunk_ids: List[int], chunk_size: int, frame_len: float):
        """Per-word emission time (s from audio start): end of the chunk of the
        word's LAST subword. Words are whitespace-delimited (matching WER). Token
        ``i`` completes a word when the next token starts a new word (its
        detokenized prefix gains a word); the final token completes the last word.
        """
        if not tokens:
            return []
        end_t = lambda k: (k + 1) * chunk_size * frame_len  # end time of chunk k
        times: List[float] = []
        wc_prev = 0
        for i in range(len(tokens)):
            wc_i = len(self.tokenizer.ids_to_text(tokens[: i + 1]).split())
            if i > 0 and wc_i > wc_prev:  # token i started a new word -> token i-1 finished one
                times.append(end_t(chunk_ids[i - 1]))
            wc_prev = wc_i
        times.append(end_t(chunk_ids[-1]))  # final word
        return times
