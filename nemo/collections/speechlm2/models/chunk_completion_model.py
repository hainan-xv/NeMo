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
from nemo.collections.speechlm2.data.streaming_stt_dataset import IGNORE_INDEX
from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel
from nemo.collections.speechlm2.parts.chunk_completion import (
    build_chunk_completion_mask,
    stream_decode_chunk_completion,
)
from nemo.collections.speechlm2.parts.optim_setup import is_frozen
from nemo.utils import logging


class ChunkCompletionSTTModel(StreamingSTTModel):
    """StreamingSTTModel with the packed spine+branch chunk-completion objective."""

    audio_open_token: str = "<|vision_start|>"
    audio_close_token: str = "<|vision_end|>"

    def __init__(self, cfg: dict, *args, dataset_cls=ChunkCompletionSTTDataset, **kwargs) -> None:
        super().__init__(cfg, *args, dataset_cls=dataset_cls, **kwargs)

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
            "[chunk-completion] packed spine+branch objective active | audio span %r/%r | eot=%s | attn_impl=%s "
            "-- models p(words_k | text_history_<k, audio_k).",
            self.audio_open_token,
            self.audio_close_token,
            self._cc_eot_id,
            self.llm.config._attn_implementation,
        )
        logging.info("=" * 72)

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

        audio_embs, audio_emb_lens = self.perception(
            input_signal=audios, input_signal_length=audio_lens
        )  # (B, T_enc, H), (B,)
        B = audio_embs.shape[0]

        if isinstance(system_prompt, str):
            system_prompt = [system_prompt] * B

        hyps: List[str] = []
        for b in range(B):
            n_frames = int(audio_emb_lens[b].item())
            frames = audio_embs[b, :n_frames]  # (n_frames, H)
            instruction_ids = self.tokenizer.text_to_ids(system_prompt[b] + "\n")
            emitted_per_chunk = stream_decode_chunk_completion(
                llm=self.llm,
                embed_tokens=self.embed_tokens,
                instruction_ids=instruction_ids,
                frames=frames,
                chunk_size=cs,
                vision_start_id=self._cc_vision_start_id,
                vision_end_id=self._cc_vision_end_id,
                eot_id=self._cc_eot_id,
                max_new_tokens=max_new_tokens,
                device=self.device,
            )
            flat = [tok for chunk in emitted_per_chunk for tok in chunk]
            hyps.append(self.tokenizer.ids_to_text(flat) if flat else "")
        return hyps
