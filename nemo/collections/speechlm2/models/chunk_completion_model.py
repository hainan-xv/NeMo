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
"""Chunk-completion streaming SpeechLM.

A variant of :class:`StreamingSTTModel` that frames streaming ASR as a
conditional *text-completion* task: for each audio chunk, extract the words it
carries given the transcript so far, ``p(words_k | text_history_<k, audio_k)``.

Rather than one interleaved ``audio-text-audio-text`` causal stream, each
utterance is packed as a pure-text **spine** (the reusable history) plus one
**branch** per chunk (its audio + target words); see
:mod:`nemo.collections.speechlm2.parts.chunk_completion`. Training is a single
O(L) forward with a custom 4D mask + ``position_ids``; the CE loss is taken on
the branch target words. Inference is a streaming loop that keeps a plain-text
spine KV cache and evicts each chunk's audio after decoding it.

Only the training and generation paths differ from the parent — audio encoding
(``perception``), embedding interleave (``_build_input_embeds``), and the
validation/WER accumulation are reused as-is.
"""

import math
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.tensor.parallel import loss_parallel
from transformers import GenerationConfig

from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.speechlm2.data.chunk_completion_dataset import ChunkCompletionBatch, ChunkCompletionSTTDataset
from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX, IGNORE_INDEX
from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel
from nemo.collections.speechlm2.parts.chunk_completion import (
    batched_stream_decode_chunk_completion,
    build_chunk_completion_mask,
)
from nemo.collections.speechlm2.parts.optim_setup import is_frozen
from nemo.utils import logging


class ChunkCompletionSTTModel(StreamingSTTModel):
    """StreamingSTTModel with the packed spine+branch chunk-completion objective."""

    audio_open_token: str = "<|vision_start|>"
    audio_close_token: str = "<|vision_end|>"
    # Sub-batch size for the encoder pass during generate(). Encoding the whole
    # (possibly large) eval batch of long clips at once overflows 32-bit CUDA
    # indexing in the subsampling conv; encoding in sub-batches keeps each conv
    # tensor bounded. The LLM decode is still fully batched across the whole batch.
    encode_batch_size: int = 8

    def __init__(self, cfg: dict, forced_aligner=None, data_cfg=None, dataset_cls=ChunkCompletionSTTDataset) -> None:
        # NOTE: keep this signature identical to StreamingSTTModel.__init__ (no
        # *args/**kwargs). huggingface_hub's PyTorchModelHubMixin.from_pretrained
        # inspects the __init__ signature; a **kwargs here makes it expand the
        # whole saved config (att_context_size, chunk_size, ...) into keyword
        # args, which then break super().__init__.
        super().__init__(cfg, forced_aligner=forced_aligner, data_cfg=data_cfg, dataset_cls=dataset_cls)

        # Audio history window (M previous chunks in each branch). Read from the
        # MODEL config so it survives from_pretrained (data_cfg is None then).
        self._audio_history_chunks = max(int(getattr(self.core_cfg, "audio_history_chunks", 0) or 0), 0)
        # Fixed-frame audio window (takes precedence over audio_history_chunks).
        self._audio_window_frames = max(int(getattr(self.core_cfg, "audio_window_frames", 0) or 0), 0)
        # Contiguous-text positions ("Option A"): must match how the data was
        # packed at train time; read from the MODEL config for from_pretrained.
        self._contiguous_text_positions = bool(getattr(self.core_cfg, "contiguous_text_positions", False))

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
                raise ValueError(f"chunk-completion delimiter {name} is not a valid in-vocab token (id={tid}).")

        # The 4D packed mask requires a mask-capable attention backend.
        attn_impl = getattr(self.llm.config, "_attn_implementation", None)
        if attn_impl not in ("eager", "sdpa"):
            logging.warning(
                "chunk-completion needs a 4D-mask-capable attention backend; found %r. Forcing 'sdpa'.",
                attn_impl,
            )
            self.llm.config._attn_implementation = "sdpa"

        logging.info("=" * 72)
        logging.info(
            "[chunk-completion] packed spine+branch objective active | audio span %r/%r | eot=%s | "
            "audio_history_chunks=%d | audio_window_frames=%d | contiguous_text_positions=%s | "
            "attn_impl=%s -- models p(words_k | text_history_<k, audio_{k-M..k}).",
            self.audio_open_token,
            self.audio_close_token,
            self._cc_eot_id,
            self._audio_history_chunks,
            self._audio_window_frames,
            self._contiguous_text_positions,
            self.llm.config._attn_implementation,
        )
        logging.info("=" * 72)

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

    def training_step(self, batch: ChunkCompletionBatch, batch_idx: int):
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

        mask = build_chunk_completion_mask(
            batch.seg_ids, batch.position_ids, batch.prefix_len, batch.valid, input_embeds.dtype
        )

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

        Encodes the audio once (same ``perception`` call as training), then for
        each utterance runs the greedy spine+branch decode
        (:func:`stream_decode_chunk_completion`): a growing plain-text spine KV
        cache of ``instruction + emitted words`` plus a transient per-chunk audio
        branch that is evicted after its words are decoded.
        """
        cs = self._resolve_inference_chunk_size(chunk_size_override)
        if cs <= 0:
            raise ValueError(f"chunk-completion generate requires a positive chunk size, got {cs}")

        B = audios.shape[0]
        if isinstance(system_prompt, str):
            system_prompt = [system_prompt] * B

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
        instruction_ids_list = [self.tokenizer.text_to_ids(system_prompt[b] + "\n") for b in range(B)]
        out = batched_stream_decode_chunk_completion(
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
        )
        if return_word_latency:
            emitted, chunk_ids = out
            hyps = [self.tokenizer.ids_to_text(ids) if ids else "" for ids in emitted]
            frame_len = float(self.core_cfg.frame_length_in_secs)
            latencies = [self._word_emission_times(emitted[b], chunk_ids[b], cs, frame_len) for b in range(B)]
            return hyps, latencies
        return [self.tokenizer.ids_to_text(ids) if ids else "" for ids in out]

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
