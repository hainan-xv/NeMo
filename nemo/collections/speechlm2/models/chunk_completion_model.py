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

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.tensor.parallel import loss_parallel
from transformers import GenerationConfig

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
        # Contiguous-text positions ("Option A"): must match how the data was
        # packed at train time; read from the MODEL config for from_pretrained.
        self._contiguous_text_positions = bool(getattr(self.core_cfg, "contiguous_text_positions", False))

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
            "audio_history_chunks=%d | contiguous_text_positions=%s | attn_impl=%s -- "
            "models p(words_k | text_history_<k, audio_{k-M..k}).",
            self.audio_open_token,
            self.audio_close_token,
            self._cc_eot_id,
            self._audio_history_chunks,
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
        self._set_encoder_att_context(cs)

        B = audios.shape[0]
        if isinstance(system_prompt, str):
            system_prompt = [system_prompt] * B

        # --- 1) Encode audio in sub-batches -> per-utterance frame tensors ---
        # A single full-batch encode of long, length-sorted clips overflows 32-bit
        # CUDA indexing in the subsampling conv; sub-batching keeps it bounded.
        enc_bs = max(1, int(self.encode_batch_size))
        frames_list: List[Tensor] = []
        for i in range(0, B, enc_bs):
            sl = audio_lens[i : i + enc_bs]
            max_len = int(sl.max().item())
            sig = audios[i : i + enc_bs, :max_len]
            emb, emb_len = self.perception(input_signal=sig, input_signal_length=sl)  # (b, T_enc, H)
            for j in range(emb.shape[0]):
                frames_list.append(emb[j, : int(emb_len[j].item())].clone())

        # --- 2) Batched chunk-synchronous streaming decode ---
        instruction_ids_list = [self.tokenizer.text_to_ids(system_prompt[b] + "\n") for b in range(B)]
        emitted = batched_stream_decode_chunk_completion(
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
        )
        return [self.tokenizer.ids_to_text(ids) if ids else "" for ids in emitted]
