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
import math
import os
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.distributed.tensor.parallel import loss_parallel
from transformers import AutoModel, GenerationConfig

from nemo.collections.asr.inference.streaming.buffering.cache_feature_bufferer import BatchedCacheFeatureBufferer
from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.utils.context_manager import CacheAwareContext
from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.streaming_stt_dataset import (
    AUDIO_TOKEN_IDX,
    IGNORE_INDEX,
    StreamingSTTBatch,
    StreamingSTTDataset,
    build_compact_turn_markers,
    decode_with_blank,
    parse_chat_template_ids,
)
from nemo.collections.speechlm2.parts.alignments import ForcedAligner, get_word_alignments_for_batch
from nemo.collections.speechlm2.parts.chunk_local_attn import (
    build_chunk_ids,
    build_chunk_local_attention_bias,
    build_chunk_local_inference_bias,
    build_chunk_local_position_ids,
    build_two_stream_attention_bias,
    build_two_stream_inference_bias,
)
from nemo.collections.speechlm2.modules.parallel_chunk_heads import ParallelChunkHeads
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf, move_embedding, setup_perception
from nemo.collections.speechlm2.parts.utils import freeze_module, to_dataclass, unfreeze_module
from nemo.utils import logging


DEFAULT_MAX_NEW_TOKENS_PER_CHUNK = 10


def token_in_vocab(token: str, tokenizer: AutoTokenizer) -> bool:
    token_pieces = tokenizer.text_to_tokens(token)
    if len(token_pieces) == 1:
        return True
    else:
        return False


def interleave_embeddings(
    input_tokens: Tensor,
    audio_mask: Tensor,
    text_embeds: Tensor,
    audio_embs: Tensor,
    pad_id: int,
) -> dict[str, Tensor]:
    """
    Merge pre-computed text and audio embeddings into a single sequence,
    guided by ``audio_mask``.

    All operations are fully batched (no Python loops over batch items):

    1. ``cumsum`` on the audio mask gives a 0-based frame index per audio position.
    2. ``torch.gather`` selects the correct audio frame for each position.
    3. ``torch.where`` picks audio or text embeddings per position.

    Args:
        input_tokens: (B, L) token IDs — only used to derive the attention mask
            (non-``pad_id`` positions).
        audio_mask: (B, L) bool — True at ``AUDIO_TOKEN_IDX`` positions.
        text_embeds: (B, L, H) embeddings produced by the text embedding layer.
            Values at audio positions are unused and may be arbitrary.
        audio_embs: (B, T_enc, H) frame-level embeddings from the audio encoder.
            If there are more audio tokens than encoder frames (last-chunk
            ceiling), the tensor is zero-padded automatically.
        pad_id: text token ID used for padding — these positions get
            ``attention_mask = False``.

    Returns:
        dict with:
            ``input_embeds`` — (B, L, H) interleaved embeddings.
            ``attention_mask`` — (B, L) bool, False only at padding positions.
    """
    B, L = input_tokens.shape

    if not audio_mask.any():
        # Pure text — nothing to interleave.
        attention_mask = input_tokens != pad_id
        return {"input_embeds": text_embeds, "attention_mask": attention_mask}

    # Sequential 0-based frame index for each audio-token position.
    frame_indices = audio_mask.long().cumsum(dim=1) - 1  # (B, L)

    # Pad encoder output if the dataset produced more audio tokens than
    # the encoder returned (last chunk ceiling).
    max_frame_idx = frame_indices.max().item()
    T_enc = audio_embs.shape[1]
    if max_frame_idx >= T_enc:
        audio_embs = F.pad(audio_embs, (0, 0, 0, max_frame_idx - T_enc + 1))

    # Gather the correct audio frame for every position in L.
    H = audio_embs.shape[2]
    gather_idx = frame_indices.clamp(min=0).unsqueeze(-1).expand(B, L, H)
    audio_at_all_pos = torch.gather(audio_embs, dim=1, index=gather_idx)  # (B, L, H)

    # Merge: audio embeddings at audio positions, text embeddings elsewhere.
    embeds = torch.where(audio_mask.unsqueeze(-1), audio_at_all_pos, text_embeds)

    # Attend to every non-padding position.
    # pad_id is ≥ 0 and AUDIO_TOKEN_IDX is −100, so this is safe.
    attention_mask = input_tokens != pad_id  # (B, L)

    return {"input_embeds": embeds, "attention_mask": attention_mask}


@dataclass
class StreamingSTTModelConfig:
    pretrained_llm: str
    pretrained_asr: str
    load_llm_weights: bool
    blank_token: str
    load_asr_weights: bool
    freeze_speech_encoder: bool
    freeze_modality_adapter: bool
    freeze_modality_proj: bool
    freeze_llm_model: bool
    freeze_llm_head: bool
    freeze_embed_tokens: bool
    chunk_size: int
    audio_tag: str = "<audio>"
    att_context_size: Optional[List[int]] = None
    audio_pad_to: Optional[int] = None
    sample_rate: int = 16000
    frame_length_in_secs: float = 0.08
    # --- Optional trainable subsampling of the encoder output ---
    # When > 1, a trainable Conv1d (kernel_size = stride = factor) is applied to
    # the perception/encoder output, shrinking the time axis by this factor
    # BEFORE the audio embeddings enter the LLM. With factor=1 (default) the
    # layer is not created at all and the model is byte-for-byte unchanged.
    #
    # Semantics ("frames fixed"): ``chunk_size`` keeps meaning *encoder frames*
    # per chunk (audio duration per chunk is unchanged, alignment/timeline math
    # stays in 80 ms frames). The LLM then sees ``ceil(chunk_size / factor)``
    # audio tokens per chunk. ``chunk_size`` need NOT be divisible by the factor:
    # when it isn't, each chunk's tail is padded up to a multiple of the factor by
    # repeating its last frame before subsampling (see _apply_audio_subsampling).
    # Fixed chunking (chunk_size > 0) is required. This is propagated to the
    # dataset so it emits the matching number of ``<audio>`` placeholders.
    encoder_subsampling_factor: int = 1
    # --- Regularization on the ENCODER OUTPUT (audio embeddings) before the LLM ---
    # Applied to the (projected, post-subsampling) audio embeddings inside
    # `_build_input_embeds`, TRAINING ONLY (skipped in eval/inference via the
    # `self.training` gate). Motivation: adjacent encoder frames carry redundant
    # information, so lightly corrupting the audio-embedding stream regularizes a
    # finite-data, multi-epoch finetune with a trainable audio encoder (text LLMs
    # skip this because pretraining is ~single-epoch -- a different regime).
    #
    #  * audio_embed_dropout > 0: standard (inverted) dropout on the audio
    #    embeddings -- each element zeroed w.p. p, survivors scaled by 1/(1-p) so the
    #    expected magnitude is preserved.
    #  * SpecAugment-style masking: zero whole spans of the audio-embedding stream
    #    (NOT rescaled), resampled fresh per forward.
    #      - audio_embed_specaug_time_masks spans of up to
    #        audio_embed_specaug_time_width contiguous FRAMES (targets redundant
    #        adjacent frames);
    #      - audio_embed_specaug_feature_masks spans of up to
    #        audio_embed_specaug_feature_width hidden DIMS.
    audio_embed_dropout: float = 0.0
    audio_embed_specaug_time_masks: int = 0
    audio_embed_specaug_time_width: int = 0
    audio_embed_specaug_feature_masks: int = 0
    audio_embed_specaug_feature_width: int = 0
    # --- Selective LoRA on the SPEECH ENCODER (Conformer) ---
    # Independent of the LLM LoRA (``cfg.lora``). When lora_encoder=True, LoRA
    # adapters are injected into the encoder attention projections
    # (``lora_encoder_target_modules``, default linear_q/linear_v) on the LAST
    # ``lora_encoder_num_last_layers`` Conformer layers (0 = all layers). Set
    # ``freeze_speech_encoder=true`` alongside this so the encoder BASE stays
    # frozen and only the injected ``lora_`` params train. Warm-starting a
    # fine-tuned encoder still works: warm_start_from_ckpt remaps the pre-adapter
    # checkpoint keys into PEFT's ``base_layer`` slots. With lora_encoder=False
    # (default) nothing changes.
    lora_encoder: bool = False
    lora_encoder_num_last_layers: int = 0
    lora_encoder_r: int = 32
    lora_encoder_alpha: int = 64
    lora_encoder_dropout: float = 0.0
    lora_encoder_target_modules: Optional[List[str]] = None
    blank_loss_weight: float = 1.0
    supervise_im_end_in_loss: bool = False
    project_unaligned_text_to_chunks: bool = False
    max_audio_chunks_per_turn: int = 1
    # Optional discrete set of chunks-per-turn group sizes for fixed-chunk
    # multi-chunk training. Comma-separated string (e.g. "1,2,7"). When set, each
    # turn's group size is sampled uniformly from EXACTLY these values instead of
    # uniformly over [1, max_audio_chunks_per_turn]. None (default) preserves the
    # legacy uniform-range behavior. Propagated to the dataset config.
    audio_chunks_per_turn_choices: Optional[str] = None
    # Max new text tokens the AR decode may emit per chunk before being force-
    # stopped. This is a decode-time cap only (training is teacher-forced and
    # uncapped). It drives the in-training WER preview and is persisted in the
    # model config so downstream eval can read the intended value. The default
    # matches the historical hardcoded constant, so existing models are
    # unchanged.
    max_new_tokens_per_chunk: int = DEFAULT_MAX_NEW_TOKENS_PER_CHUNK
    use_modality_position_ids: bool = False
    # Additive constant applied to every text/template RoPE position so that the
    # text stream occupies ``[offset, offset + num_text)`` while audio occupies
    # ``[0, num_audio)``. With ``offset > 0`` the two modalities live in disjoint
    # RoPE ranges. With ``offset == 0`` audio and text both count from 0 and share
    # the SAME range (the "both from 0" design); modality is then distinguished by
    # the embeddings rather than by position. The offset is a pure additive
    # constant applied identically at train and inference, so it never changes
    # relative (RoPE) geometry within the text stream.
    modality_position_offset: int = 32768
    # --- Chunk-local audio attention ---
    # When True, applies a chunk-local attention mask: a query token may attend
    # to any causally-prior non-pad key EXCEPT audio keys belonging to a chunk
    # more than ``num_visible_audio_chunks - 1`` chunks older than the query's
    # own chunk. Position IDs are built with two independent contiguous
    # counters (audio counter and text counter), both starting at 0; pads get
    # position 0. Implemented entirely in
    # ``nemo.collections.speechlm2.parts.chunk_local_attn`` (helpers) and below
    # in this file (state + plumbing). Mutually exclusive with
    # ``use_modality_position_ids``. Requires ``supervise_im_end_in_loss=True``
    # (``<|im_end|>`` is the only stop signal in this scheme) and fixed
    # chunking (``chunk_size > 0``).
    use_chunk_local_audio_attn: bool = False
    num_visible_audio_chunks: int = 1
    # --- Two-stream ("block audio->text") attention ---
    # When True, the LLM self-attention is masked into two causal streams:
    #   * audio tokens attend to (causally-prior) AUDIO tokens only — a pure
    #     causal speech encoder living inside the shared LLM (audio never peeks
    #     at text / template / system-prompt tokens);
    #   * text tokens attend to every causally-prior valid token of EITHER
    #     modality (all past text AND all past audio) — unchanged vs baseline.
    # This is the mildest modality separation: it only removes the audio->text
    # attention edges. The single-sequence interleaved layout and the training
    # data are UNCHANGED; only the attention bias differs. Implemented on top
    # of the chunk-local streaming machinery, so it inherits the same
    # per-modality contiguous RoPE counters (audio and text each start at 0)
    # and the ``<|im_end|>`` end-of-chunk stop signal. Mutually exclusive with
    # ``use_chunk_local_audio_attn`` and ``use_modality_position_ids``.
    # Requires ``supervise_im_end_in_loss=True`` and fixed chunking
    # (``chunk_size > 0``).
    use_two_stream_attn: bool = False
    # Auxiliary raw-transcript LM objective. When > 0, training adds a second
    # causal-LM loss over ``batch.text`` only: no audio placeholders, no chat
    # template, no blank token, and no assistant ``<|im_end|>`` control token.
    # Inference is unchanged.
    text_only_lm_loss_weight: float = 0.0
    use_text_only_lm_score_at_inference: bool = True
    # --- Encoder reuse (Tier-1 amortization) ---
    # When > 1, each training step encodes the audio with the perception
    # encoder ONCE and then reuses that encoder output across ``encoder_reuse_k``
    # independently-resampled chunk/turn partitions of the SAME utterances,
    # running ``encoder_reuse_k`` LLM forwards and averaging their losses. This
    # amortizes the (expensive) speech encoder over K diverse "completion"
    # views per utterance. ``encoder_reuse_k=1`` (default) is the legacy path
    # and is byte-for-byte unchanged. Only active when an online forced aligner
    # is configured (the model needs ``dataset.get_batch_data`` + alignments to
    # resample partitions on the fly); otherwise it transparently falls back to
    # K=1 with a one-time warning.
    encoder_reuse_k: int = 1
    log_every_n_steps: int = 10
    dtype: str = "bfloat16"
    # --- Compact template ---
    # Compact template: use a write token to trigger text generation, and the EOS token
    # is automatically generated by the tokenizer.
    compact_template: bool = False
    write_token: str = "<|im_start|>"
    # --- Flush token (explicit end-of-audio "dump the tail" control signal) ---
    # When True, a special ``flush_token`` is registered in the vocabulary and,
    # in training, ALWAYS appended as a final user turn after the last audio
    # chunk. Words whose (delayed) emission was deferred past the last real chunk
    # are emitted by the assistant AFTER the flush token; otherwise the post-flush
    # assistant turn is empty and emits only the end-of-utterance ``<|im_end|>``.
    # This gives the model an explicit, in-distribution signal to flush the tail,
    # replacing the inference-time trailing-silence hack. At inference the flush
    # turn is fed once after the audio chunks and the residual is decoded until
    # ``<|im_end|>``. The new embedding is initialized from ``<|im_end|>``.
    use_flush: bool = False
    flush_token: str = "<flush>"
    # Training-only augmentation: with this probability a cut is replaced by a
    # truncated copy (cut at a chunk boundary) so the model also learns to flush
    # MID-utterance. The cutoff (in chunks) is sampled once per batch from
    # ``[1, n_chunks(shortest cut in batch)]`` and applied to the selected cuts.
    # 0.0 (default) disables truncation (only the always-on end flush remains).
    flush_truncate_prob: float = 0.0
    # --- Aux chunk-boundary classifier head ---
    # Master switch. Only valid in dynamic-chunking mode (chunk_size == 0).
    # When True, a small K-layer transformer head is built on top of the LLM's
    # last hidden state and trained with BCE at audio frame positions; the LM
    # head is no longer supervised at audio positions. When False (default),
    # the boundary decision falls back to the LM head signaling via the blank
    # / user_footer_first token (legacy behavior). The module is NOT built
    # unless this flag is True.
    use_chunk_classifier: bool = False
    chunk_classifier_loss_weight: float = 0.5
    chunk_classifier_num_layers: int = 2
    chunk_classifier_init_from_llm: bool = True
    chunk_classifier_threshold: float = 0.5
    chunk_classifier_use_at_inference: bool = False
    freeze_chunk_classifier: bool = False
    # Auto-balance the BCE: pos_weight = num_neg/num_pos per batch. Most audio
    # frames are "keep listening" (label=0); the few "emit" frames (label=1) get
    # drowned out without rebalancing.
    chunk_classifier_auto_balance: bool = True
    # Detach the LLM's last hidden state before feeding it to the aux backbone.
    # When True, the aux BCE loss does NOT propagate back into the LLM (body,
    # lm_head, or embed_tokens) — the LLM is supervised purely by the masked
    # LM CE on text positions, and the aux head learns on a fixed feature.
    # Recommended default for clean A/B against pure text-LLM training.
    chunk_classifier_stop_grad_to_llm: bool = True
    # When True, keep the LM-CE supervision at audio positions (blank /
    # user_footer_first targets) alongside the aux head's BCE. The LM head
    # learns the boundary signal too — useful when LoRA / lm_head needs more
    # gradient density per batch (default text-only positions can be sparse,
    # ~20% of input length, slowing convergence). When False (default), audio
    # positions are masked to IGNORE_INDEX in the LM CE — the aux head owns
    # the boundary decision exclusively.
    chunk_classifier_keep_lm_supervision_at_audio: bool = False
    # --- Parallel chunk heads (multi-token-per-chunk prediction) ---
    # When parallel_loss_weight > 0 OR parallel_chunk_decode is True, build a
    # ParallelChunkHeads module: K learned slot embeddings + a small depth
    # transformer (causal over slots) + tied lm_head. Training supervises the
    # heads at per-block anchors; a chunk's emit-stream (content + <|im_end|>)
    # is split into K-token blocks, so K is the per-forward parallelism factor,
    # NOT a hard cap on chunk length. Inference emits one K-block per forward
    # and iterates blocks within a chunk until <|im_end|> appears (decoupling K
    # from the max tokens a chunk can emit). Requires compact_template=True
    # (anchor = write_id).
    parallel_chunk_slots: int = 4
    parallel_loss_weight: float = 0.0
    parallel_chunk_decode: bool = False
    parallel_depth_layers: int = 1
    parallel_depth_num_heads: int = 8
    # When True, train on the parallel-chunk loss ONLY: the AR per-token CE (and
    # the text-only-LM / chunk-classifier aux terms) are kept in the autograd
    # graph multiplied by 0.0 so their parameters still receive a (zero)
    # gradient — this keeps DDP find_unused_parameters=false from tripping —
    # but they contribute nothing to the optimized objective. Use with
    # parallel_loss_weight=1.0 for a pure multi-token-head training run.
    parallel_only_loss: bool = False
    # When True, an audio chunk that emits no text is supervised/decoded as a
    # bare <|im_end|> (chunk-end) instead of the explicit <blank> token. This
    # drops one token per empty chunk: the emit-stream becomes ``write_id ->
    # <|im_end|>`` rather than ``write_id -> <blank> -> <|im_end|>``. It is both
    # more compact and mathematically cleaner (every chunk's emission ends with
    # <|im_end|>; "empty" simply means <|im_end|> fires immediately). Only
    # affects the compact template (parallel scheme); requires
    # compact_template=True. Default False preserves the legacy <blank> scheme.
    empty_chunk_eos_only: bool = False
    # --- Blank-continuation parallel scheme (strict <|im_end|> placement) ---
    # When True, the parallel-chunk targets use the "blank-continuation" scheme:
    #   * <|im_end|> may appear ONLY in slot 0 of a block — a chunk always closes
    #     with a terminator block ``[<|im_end|>, IGNORE...]``;
    #   * a partial / cut block ends with a single <blank> marker meaning "more
    #     tokens follow in the next block" (continue), e.g. ``[t1, t2, <blank>, IGN]``;
    #   * a full K-token block carries no marker (fullness itself means continue).
    # Decoding mirrors this: <blank> => cut & continue (NOT a chunk stop); only a
    # slot-0 <|im_end|> ends the chunk; the trunk is fed real tokens only (blanks
    # are synthetic head targets, never entered into the KV cache). Requires
    # compact_template=True and a valid <blank> token. Default False keeps the
    # legacy fixed-K scheme (block g = stream[g*K:(g+1)*K], blank/im_end inline).
    parallel_blank_continuation: bool = False
    # Training-only augmentation for the blank-continuation scheme. With this
    # per-block probability a block randomly "cuts": a cut point is chosen
    # uniformly among the block's non-first real positions, a <blank> is placed
    # there, and the remaining tokens are pushed to the next block. Applied
    # independently to each block (so a chunk may be cut multiple times),
    # teaching the model flexible block boundaries. 0.0 disables augmentation
    # (blocks are greedily packed to K). Only effective when
    # parallel_blank_continuation=True.
    parallel_cut_prob: float = 0.0


@dataclass
class StreamingState:
    """Holds the KV cache and other state for B streaming audio sessions.

    All tensors have batch dimension B (B=1 for single-stream inference).
    The LLM cache ``past_key_values`` has shape ``(layers, (B, heads, seq, dim))``
    for K and V.  The perception cache has batch dim on axis 1.
    """

    cache: tuple | None = None  # HF past_key_values with batch dim B
    generated_tokens: list[list[int]] = field(default_factory=list)  # B lists of per-chunk token IDs
    seq_lens: list[int] = field(default_factory=list)  # per-stream sequence lengths
    audio_cache: CacheAwareContext | None = None  # perception cache with batch dim B
    audio_feature_buffer: BatchedCacheFeatureBufferer | None = None
    attention_mask: Optional[Tensor] = None  # (B, seq_len) mask for left-padded prefill
    # (B, seq_len, H) running buffer of LLM last hidden states; used by the aux
    # chunk-boundary classifier when chunk_classifier_use_at_inference is True.
    # None when disabled — keeps the field cheap to always carry on the state.
    aux_hidden_buffer: Optional[Tensor] = None
    audio_position_counter: Optional[Tensor] = None
    text_position_counter: Optional[Tensor] = None
    # --- Chunk-local audio attention bookkeeping ---
    # Per-stream "current chunk id" counter. Starts at -1 (no audio chunks
    # processed yet); advances each time we feed a new audio chunk.
    chunk_id_counter: Optional[Tensor] = None  # (B,) long
    # Per-token chunk id and is-audio flags for every cached token (matches
    # ``attention_mask`` length). Required to build the chunk-local 4-D
    # additive bias at every streaming forward call.
    chunk_id_history: Optional[Tensor] = None  # (B, L_cached) long
    is_audio_history: Optional[Tensor] = None  # (B, L_cached) bool
    # Cached raw-transcript LM state for optional inference-time score fusion.
    text_lm_cache: tuple | None = None
    text_lm_attention_mask: Optional[Tensor] = None
    text_lm_position_counter: Optional[Tensor] = None
    text_lm_next_logits: Optional[Tensor] = None
    text_lm_next_logits_valid: Optional[Tensor] = None
    batch_size: int = 1

    @property
    def seq_len(self) -> int:
        """Max seq_len across streams (= KV cache dimension)."""
        return max(self.seq_lens) if self.seq_lens else 0


class StreamingSTTModel(LightningModule, HFHubMixin):

    def __init__(
        self,
        cfg: dict,
        forced_aligner: Optional[ForcedAligner] = None,
        data_cfg: Optional[DictConfig] = None,
        dataset_cls=StreamingSTTDataset,
    ) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to StreamingSTTModel as a Python dict to support hyperparameter "
            f"serialization in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        print(
            "\n"
            "============================================================\n"
            ">>> HAINAN /code CHECK: StreamingSTTModel.__init__ running\n"
            f">>> module file = {__file__}\n"
            "============================================================\n",
            flush=True,
        )
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)
        self.core_cfg: StreamingSTTModelConfig = to_dataclass(StreamingSTTModelConfig, cfg)
        # Decide whether to build / use the parallel chunk heads at all.
        # Conservative: only enable when training requests it (weight > 0) or
        # inference explicitly opts in. Otherwise the heads are not built and
        # the dataset skips emitting the new tensors.
        self._parallel_heads_enabled = bool(
            self.core_cfg.parallel_loss_weight > 0 or self.core_cfg.parallel_chunk_decode
        )
        if self._parallel_heads_enabled and not self.core_cfg.compact_template:
            logging.warning(
                "Parallel chunk heads are only supported with compact_template=True; "
                "disabling parallel heads (parallel_loss_weight=%s, parallel_chunk_decode=%s).",
                self.core_cfg.parallel_loss_weight,
                self.core_cfg.parallel_chunk_decode,
            )
            self._parallel_heads_enabled = False
        if data_cfg is not None:
            from omegaconf import open_dict

            with open_dict(data_cfg):
                data_cfg.supervise_im_end_in_loss = self.core_cfg.supervise_im_end_in_loss
                data_cfg.project_unaligned_text_to_chunks = self.core_cfg.project_unaligned_text_to_chunks
                data_cfg.max_audio_chunks_per_turn = self.core_cfg.max_audio_chunks_per_turn
                # Only set the discrete-choices key when the model actually
                # requests it, so existing runs keep byte-identical data_cfg
                # snapshots (no spurious key in saved configs / hparam logs).
                if getattr(self.core_cfg, "audio_chunks_per_turn_choices", None) is not None:
                    data_cfg.audio_chunks_per_turn_choices = self.core_cfg.audio_chunks_per_turn_choices
                # compact_template / write_token: the dataset has its OWN copies
                # of these keys (StreamingSTTDataConfig) and they must agree with
                # the model's choice -- otherwise the dataset emits regular chat
                # template tokens while the model trains / runs inference under
                # compact-template assumptions, silently breaking both training
                # (no <write> anchors -> parallel heads never fire, severe
                # train/inference token mismatch) and decoding. Gate on
                # compact_template=True to keep existing non-compact runs
                # byte-identical (no new keys appear in their saved configs).
                if bool(self.core_cfg.compact_template):
                    data_cfg.compact_template = True
                    data_cfg.write_token = str(self.core_cfg.write_token)
                # Empty-chunk-as-<eos> is supported for BOTH the compact template
                # (handled in the compact tokenizer) and the standard non-compact
                # chat template (handled by post-hoc <blank>-stripping). Only set
                # the key when actually enabled so existing runs that never touch
                # this flag keep byte-identical saved configs.
                if bool(self.core_cfg.empty_chunk_eos_only):
                    data_cfg.empty_chunk_eos_only = True
                # Flush token + its training-time truncation augmentation. Only
                # set when the feature is requested so existing runs keep
                # byte-identical data_cfg snapshots.
                if bool(self.core_cfg.use_flush):
                    data_cfg.use_flush = True
                    data_cfg.flush_token = str(self.core_cfg.flush_token)
                    data_cfg.flush_truncate_prob = float(self.core_cfg.flush_truncate_prob)
                # Encoder subsampling: the dataset must emit chunk_size//factor
                # audio placeholders per chunk to match the subsampled encoder
                # output. Only set the key when the feature is requested so
                # existing runs keep byte-identical data_cfg snapshots.
                if int(getattr(self.core_cfg, "encoder_subsampling_factor", 1) or 1) > 1:
                    data_cfg.encoder_subsampling_factor = int(self.core_cfg.encoder_subsampling_factor)
                # Only touch parallel_chunk_slots in data_cfg when the feature
                # is actively requested — otherwise leave the dict identical to
                # what existing runs produce (back-compat: no spurious key, no
                # diff in config snapshots / hyperparameter logging).
                if self._parallel_heads_enabled:
                    data_cfg.parallel_chunk_slots = int(self.core_cfg.parallel_chunk_slots)
                    # Blank-continuation scheme + its training-only cut
                    # augmentation. Only set when the new scheme is actually
                    # requested so legacy parallel runs keep byte-identical
                    # data_cfg snapshots.
                    if bool(self.core_cfg.parallel_blank_continuation):
                        data_cfg.parallel_blank_continuation = True
                        data_cfg.parallel_cut_prob = float(self.core_cfg.parallel_cut_prob)
            log_msg = (
                "StreamingSTT data_cfg flags propagated from model: "
                "supervise_im_end_in_loss=%s, project_unaligned_text_to_chunks=%s, "
                "max_audio_chunks_per_turn=%s"
            )
            log_args = [
                data_cfg.supervise_im_end_in_loss,
                data_cfg.project_unaligned_text_to_chunks,
                data_cfg.max_audio_chunks_per_turn,
            ]
            if bool(self.core_cfg.compact_template):
                log_msg += ", compact_template=%s, write_token=%r"
                log_args.extend([data_cfg.compact_template, data_cfg.write_token])
            if bool(self.core_cfg.empty_chunk_eos_only):
                log_msg += ", empty_chunk_eos_only=%s"
                log_args.append(True)
            if getattr(self.core_cfg, "audio_chunks_per_turn_choices", None) is not None:
                log_msg += ", audio_chunks_per_turn_choices=%r"
                log_args.append(self.core_cfg.audio_chunks_per_turn_choices)
            if self._parallel_heads_enabled:
                log_msg += ", parallel_chunk_slots=%s"
                log_args.append(data_cfg.parallel_chunk_slots)
                if bool(self.core_cfg.parallel_blank_continuation):
                    log_msg += ", parallel_blank_continuation=%s, parallel_cut_prob=%s"
                    log_args.extend([True, data_cfg.parallel_cut_prob])
            logging.info(log_msg, *log_args)
        if self.core_cfg.use_modality_position_ids and not self.core_cfg.supervise_im_end_in_loss:
            raise ValueError("use_modality_position_ids=True requires supervise_im_end_in_loss=True")
        if self.core_cfg.use_modality_position_ids and int(self.core_cfg.modality_position_offset) < 0:
            raise ValueError("use_modality_position_ids=True requires modality_position_offset >= 0")
        if self.core_cfg.use_modality_position_ids:
            _off = int(self.core_cfg.modality_position_offset)
            logging.info(
                "Separated audio/text RoPE position IDs ENABLED: audio positions start at 0, "
                "text/template positions start at %d%s. Assistant <|im_end|> supervision is "
                "required and enabled.",
                _off,
                " (offset=0: audio and text share the same RoPE range, both starting from 0 — "
                "no disjoint-range offset; modality is distinguished by the embeddings)"
                if _off == 0
                else "",
            )
        if self.core_cfg.use_chunk_local_audio_attn:
            if self.core_cfg.use_modality_position_ids:
                raise ValueError(
                    "use_chunk_local_audio_attn=True is mutually exclusive with "
                    "use_modality_position_ids=True"
                )
            if not self.core_cfg.supervise_im_end_in_loss:
                raise ValueError(
                    "use_chunk_local_audio_attn=True requires supervise_im_end_in_loss=True "
                    "(<|im_end|> is the only end-of-chunk stop signal in this scheme)"
                )
            if int(self.core_cfg.num_visible_audio_chunks) < 1:
                raise ValueError(
                    "use_chunk_local_audio_attn=True requires num_visible_audio_chunks >= 1, got "
                    f"{self.core_cfg.num_visible_audio_chunks}"
                )
            if int(self.core_cfg.chunk_size) <= 0:
                raise ValueError(
                    "use_chunk_local_audio_attn=True currently supports fixed chunking only "
                    f"(chunk_size > 0), got chunk_size={self.core_cfg.chunk_size}"
                )
            logging.info(
                "Chunk-local audio attention ENABLED: a query may attend to audio keys only "
                "from its own chunk and the previous %d chunk(s). Audio and text use independent "
                "contiguous RoPE counters (both start at 0). Assistant <|im_end|> supervision is "
                "required and enabled.",
                int(self.core_cfg.num_visible_audio_chunks) - 1,
            )
        if self.core_cfg.use_two_stream_attn:
            if self.core_cfg.use_chunk_local_audio_attn:
                raise ValueError(
                    "use_two_stream_attn=True is mutually exclusive with "
                    "use_chunk_local_audio_attn=True"
                )
            if self.core_cfg.use_modality_position_ids:
                raise ValueError(
                    "use_two_stream_attn=True is mutually exclusive with "
                    "use_modality_position_ids=True"
                )
            if not self.core_cfg.supervise_im_end_in_loss:
                raise ValueError(
                    "use_two_stream_attn=True requires supervise_im_end_in_loss=True "
                    "(<|im_end|> is the end-of-chunk stop signal in the streaming decode "
                    "path it shares with chunk-local attention)"
                )
            if int(self.core_cfg.chunk_size) <= 0:
                raise ValueError(
                    "use_two_stream_attn=True currently supports fixed chunking only "
                    f"(chunk_size > 0), got chunk_size={self.core_cfg.chunk_size}"
                )
            logging.info(
                "Two-stream attention ENABLED: audio tokens form a pure causal audio encoder "
                "(attend to prior AUDIO only); text tokens attend to all prior text AND audio "
                "(causal). Audio and text use independent contiguous RoPE counters (both start "
                "at 0). Assistant <|im_end|> supervision is required and enabled."
            )

        # --- LLM ---
        self.tokenizer = AutoTokenizer(self.core_cfg.pretrained_llm, use_fast=True)
        self.llm = load_pretrained_hf(
            self.core_cfg.pretrained_llm,
            pretrained_weights=self.core_cfg.load_llm_weights,
        )

        # Ensure <blank> token is in the vocabulary.
        # Unescape Python escape sequences (e.g. "\\n" → "\n") because Hydra/OmegaConf
        # loads YAML strings literally without interpreting backslash escapes.
        # An empty blank_token ("") disables the blank mechanism entirely
        # (fixed chunking only — see StreamingSTTDataset for the guard).
        self.blank_token = self.core_cfg.blank_token.encode().decode('unicode_escape')

        if self.blank_token == "":
            logging.info("blank_token is empty: blank mechanism disabled")
        elif not token_in_vocab(self.blank_token, self.tokenizer):
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.blank_token]})
            self.llm.resize_token_embeddings(len(self.tokenizer.tokenizer))
            logging.info(f"Added blank token `{self.blank_token}` to tokenizer: {self.blank_token_id}")
        else:
            logging.info(f"Blank token `{str(self.blank_token)}` already in tokenizer: {self.blank_token_id}")

        # Compact-template write_token: only register if missing (default <|im_start|>
        # is already in Qwen3's vocab → uses pretrained embedding).
        if self.core_cfg.compact_template:
            wt = self.core_cfg.write_token
            if not token_in_vocab(wt, self.tokenizer):
                self.tokenizer.add_special_tokens({"additional_special_tokens": [wt]})
                self.llm.resize_token_embeddings(len(self.tokenizer.tokenizer))
                logging.info(f"compact_template: added write_token `{wt}` to tokenizer (random init)")
            else:
                logging.info(f"compact_template: using existing vocab token `{wt}` as write_token")

        # Flush token: explicit end-of-audio control signal (see config). It is a
        # fed-in (user-side) token only — never a prediction target — so we only
        # need a sensible INPUT embedding for it. Initialize from <|im_end|> (the
        # emitted end-of-utterance partner) for stable starts.
        self.flush_token = self.core_cfg.flush_token if self.core_cfg.use_flush else ""
        if self.core_cfg.use_flush:
            ft = self.core_cfg.flush_token
            if not token_in_vocab(ft, self.tokenizer):
                self.tokenizer.add_special_tokens({"additional_special_tokens": [ft]})
                self.llm.resize_token_embeddings(len(self.tokenizer.tokenizer))
                flush_id = self.tokenizer.tokenizer.convert_tokens_to_ids(ft)
                eos_id = self.tokenizer.tokenizer.eos_token_id
                if eos_id is not None:
                    with torch.no_grad():
                        in_emb = self.llm.get_input_embeddings().weight
                        in_emb[flush_id] = in_emb[eos_id].clone()
                logging.info(f"use_flush: added flush_token `{ft}` to tokenizer (init from <|im_end|>)")
            else:
                logging.info(f"use_flush: using existing vocab token `{ft}` as flush_token")

        # Separate embedding layer to avoid FSDP/TP conflicts (same pattern as SALM)
        self.embed_tokens = self.llm.model.embed_tokens
        del self.llm.model.embed_tokens

        # --- Speech encoder (perception module) ---
        self.perception = setup_perception(
            cfg=self.cfg,
            output_dim=self.llm.config.hidden_size,
            pretrained_asr=self.core_cfg.pretrained_asr,
            pretrained_weights=self.core_cfg.load_asr_weights,
            audio_pad_to=self.core_cfg.audio_pad_to,
            att_context_size=self.core_cfg.att_context_size,
        )

        # --- Optional trainable encoder-output subsampler ---
        # Conv1d with kernel_size = stride = factor performs a clean,
        # non-overlapping factor-S reduction of the encoder time axis. Created
        # only when factor > 1 so factor==1 runs are byte-for-byte unchanged.
        self.encoder_subsampling_factor = int(getattr(self.core_cfg, "encoder_subsampling_factor", 1) or 1)
        if self.encoder_subsampling_factor > 1:
            S = self.encoder_subsampling_factor
            assert self.core_cfg.chunk_size > 0, (
                f"encoder_subsampling_factor={S} requires fixed chunking (chunk_size>0), "
                f"got chunk_size={self.core_cfg.chunk_size}"
            )
            # chunk_size need NOT be divisible by S: when it isn't, each chunk's
            # tail is padded by repeating its last frame up to a multiple of S
            # (see _apply_audio_subsampling), yielding ceil(chunk_size/S) tokens.
            H = self.llm.config.hidden_size
            self.audio_subsampler = nn.Conv1d(H, H, kernel_size=S, stride=S)
            _slots = (self.core_cfg.chunk_size + S - 1) // S
            _note = "" if self.core_cfg.chunk_size % S == 0 else " (tail padded by repeating last frame)"
            logging.info(
                f"Encoder subsampling enabled: factor={S}, Conv1d(H={H}, kernel=stride={S}); "
                f"LLM sees {_slots} audio tokens per {self.core_cfg.chunk_size}-frame chunk{_note}."
            )
        else:
            self.audio_subsampler = None

        # --- Aux chunk-boundary classifier (only built when enabled) ---
        # Only valid in dynamic-chunking mode (chunk_size == 0). When disabled,
        # no module is built and existing runs / checkpoints see no behavior
        # change. The boundary decision falls back to the LM head's blank /
        # user_footer_first signal.
        if self.core_cfg.use_chunk_classifier:
            assert self.core_cfg.chunk_size == 0, (
                "use_chunk_classifier=True requires dynamic chunking "
                f"(chunk_size=0), got chunk_size={self.core_cfg.chunk_size}"
            )
            self._build_chunk_classifier()
            # Aux training/eval reads self._user_footer_first_id (the BCE positive
            # label). It's normally set lazily by _ensure_inference_cache, but
            # training runs before any inference call — so prime the cache now.
            self._ensure_inference_cache()
        elif self.core_cfg.chunk_classifier_use_at_inference:
            raise ValueError("chunk_classifier_use_at_inference=True requires use_chunk_classifier=True")

        # --- Parallel chunk heads (multi-token-per-chunk prediction) ---
        if self._parallel_heads_enabled:
            self.parallel_chunk_heads = ParallelChunkHeads(
                hidden_size=self.llm.config.hidden_size,
                num_slots=int(self.core_cfg.parallel_chunk_slots),
                depth_layers=int(self.core_cfg.parallel_depth_layers),
                num_heads=int(self.core_cfg.parallel_depth_num_heads),
            )
            logging.info(
                "ParallelChunkHeads enabled: K=%d slots, %d depth layers, %d attention heads "
                "(parallel_loss_weight=%.3f, parallel_chunk_decode=%s).",
                int(self.core_cfg.parallel_chunk_slots),
                int(self.core_cfg.parallel_depth_layers),
                int(self.core_cfg.parallel_depth_num_heads),
                float(self.core_cfg.parallel_loss_weight),
                self.core_cfg.parallel_chunk_decode,
            )
            if bool(self.core_cfg.parallel_blank_continuation):
                if not self.core_cfg.compact_template:
                    raise ValueError(
                        "parallel_blank_continuation=True requires compact_template=True "
                        "(anchors are write_id-relative)."
                    )
                if not self.has_blank:
                    raise ValueError(
                        "parallel_blank_continuation=True requires a non-empty blank_token "
                        "(the <blank> marker signals block continuation)."
                    )
                logging.info(
                    "Parallel blank-continuation scheme ENABLED: <|im_end|> only in slot 0; "
                    "partial/cut blocks end with <blank> (id=%d); decode feeds real tokens only. "
                    "Training cut augmentation prob=%.3f.",
                    int(self.blank_token_id),
                    float(self.core_cfg.parallel_cut_prob),
                )
        else:
            self.parallel_chunk_heads = None

        self._apply_freeze_config()

        # --- LoRA ---
        if "lora" in self.cfg:
            # Install LoRA after freezing the LLM body to avoid freezing the LoRA weights
            maybe_install_lora(self)
            # huggingface PEFT library freezes the whole LLM, so we need to unfreeze the lm_head if needed
            if self.core_cfg.freeze_llm_head:
                freeze_module(self.llm.lm_head)
            else:
                unfreeze_module(self.llm.lm_head)

        # --- Selective LoRA on the speech encoder (Conformer) ---
        # Injected AFTER _apply_freeze_config so the frozen encoder base is kept
        # frozen and only the freshly-added adapters train.
        if self.core_cfg.lora_encoder:
            self._install_encoder_lora()

        # The helper dataset is what rebuilds (resampled) batches at train time.
        # Two consumers need it: (1) the online forced aligner, which realigns
        # every step; and (2) offline ``encoder_reuse_k>1``, which resamples K
        # independently delay-randomized partitions per step from the cuts'
        # precomputed word alignments (no online aligner). So build it whenever a
        # data config is available, not only when an online aligner is present.
        self.forced_aligner = forced_aligner
        if data_cfg is not None and dataset_cls is not None:
            self.dataset = dataset_cls(cfg=data_cfg, tokenizer=self.tokenizer)
        else:
            self.dataset = None
            if forced_aligner is not None:
                raise ValueError(
                    "Online forced alignment requires both data_cfg and dataset_cls."
                )

        logging.info("\n" + str(ModelSummary(self, max_depth=2)))

    def _build_chunk_classifier(self) -> None:
        """Construct the aux backbone + linear head.

        The backbone reuses the LLM's architecture via ``AutoModel.from_config``
        (works for any modern HF decoder-only LLM — Llama/Qwen/Mistral/Phi/Gemma).
        ``embed_tokens`` is dropped since we always feed ``inputs_embeds``;
        same pattern as the main LLM at __init__.
        """
        K = max(int(self.core_cfg.chunk_classifier_num_layers), 1)
        aux_cfg = deepcopy(self.llm.config)
        aux_cfg.num_hidden_layers = K
        # Aux backbone is run as a full-sequence forward at both train and
        # inference time — no KV cache needed.
        aux_cfg.use_cache = False

        self.chunk_classifier_backbone = AutoModel.from_config(aux_cfg)
        # Drop V×H embedding table: we always feed inputs_embeds.
        # (Same trick as line 228-229 for the main LLM.)
        if hasattr(self.chunk_classifier_backbone, "embed_tokens"):
            del self.chunk_classifier_backbone.embed_tokens

        self.chunk_classifier_head = nn.Linear(aux_cfg.hidden_size, 1)
        nn.init.zeros_(self.chunk_classifier_head.bias)

        # Optional warm-start: copy the last K layers + final norm from the
        # main LLM. The aux backbone consumes the LLM's last hidden state, so
        # these layers operate on the right input distribution and converge
        # much faster than random init.
        if self.core_cfg.chunk_classifier_init_from_llm:
            try:
                src_layers = self.llm.model.layers[-K:]
                for i in range(K):
                    self.chunk_classifier_backbone.layers[i].load_state_dict(src_layers[i].state_dict())
                self.chunk_classifier_backbone.norm.load_state_dict(self.llm.model.norm.state_dict())
                logging.info(f"chunk_classifier: warm-started from last {K} LLM layers")
            except (AttributeError, KeyError) as e:
                logging.warning(
                    f"chunk_classifier_init_from_llm: warm-start failed ({e}); " "falling back to random init"
                )

    def _install_encoder_lora(self) -> None:
        """Inject LoRA adapters into the last-N Conformer encoder layers.

        The encoder is a NeMo ``ConformerEncoder`` (not an HF ``PreTrainedModel``),
        so we use PEFT's low-level ``inject_adapter_in_model`` rather than
        ``get_peft_model`` (which assumes an HF model). The encoder base is frozen
        and only the injected ``lora_`` params are left trainable, regardless of
        the installed PEFT version's default requires_grad behavior.
        """
        from peft import LoraConfig, inject_adapter_in_model

        encoder = self.perception.encoder
        layers = getattr(encoder, "layers", None)
        if layers is None:
            raise AttributeError(
                "lora_encoder=True but self.perception.encoder has no `.layers` "
                "ModuleList; cannot target Conformer layers."
            )
        n_layers = len(layers)
        n_last = int(self.core_cfg.lora_encoder_num_last_layers) or n_layers
        n_last = max(1, min(n_last, n_layers))
        layer_ids = set(range(n_layers - n_last, n_layers))
        wanted = set(self.core_cfg.lora_encoder_target_modules or ["linear_q", "linear_v"])

        # Build EXPLICIT fully-qualified target module names for the selected
        # layers. We do NOT use PEFT's layers_to_transform/layers_pattern: the
        # encoder is the adapter root, so its submodules are named
        # "layers.<i>.self_attn.linear_q" (no parent prefix), and PEFT's
        # layers_pattern regex expects a leading ".layers.<i>." and never matches.
        # Exact names sidestep that and are robust to the attention class name.
        full_targets = []
        for name, mod in encoder.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            parts = name.split(".")
            if len(parts) >= 3 and parts[0] == "layers" and parts[1].isdigit():
                if int(parts[1]) in layer_ids and parts[-1] in wanted:
                    full_targets.append(name)
        if not full_targets:
            attn_linears = sorted({
                n for n, m in encoder.named_modules()
                if isinstance(m, nn.Linear) and n.startswith("layers.")
            })
            raise ValueError(
                f"lora_encoder: no Linear modules matching {sorted(wanted)} found in the "
                f"last {n_last} of {n_layers} encoder layers. Present layer Linears (sample): "
                f"{attn_linears[:12]}"
            )

        enc_lora_cfg = LoraConfig(
            r=int(self.core_cfg.lora_encoder_r),
            lora_alpha=int(self.core_cfg.lora_encoder_alpha),
            lora_dropout=float(self.core_cfg.lora_encoder_dropout),
            target_modules=full_targets,
            bias="none",
        )
        inject_adapter_in_model(enc_lora_cfg, encoder)

        # Freeze the entire encoder base; train ONLY the injected adapters.
        for name, p in encoder.named_parameters():
            p.requires_grad = "lora_" in name
        n_train = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        logging.info(
            "Encoder LoRA installed: r=%d alpha=%d dropout=%.3g targets=%s on layers %s "
            "(last %d of %d Conformer layers, %d target Linears); trainable encoder params=%d",
            enc_lora_cfg.r, enc_lora_cfg.lora_alpha, enc_lora_cfg.lora_dropout,
            sorted(wanted), sorted(layer_ids), n_last, n_layers, len(full_targets), n_train,
        )

    def _apply_freeze_config(self) -> None:
        if self.core_cfg.freeze_speech_encoder:
            freeze_module(self.perception.encoder)
        else:
            unfreeze_module(self.perception.encoder)

        if self.core_cfg.freeze_modality_adapter:
            freeze_module(self.perception.modality_adapter)
        else:
            unfreeze_module(self.perception.modality_adapter)

        if self.core_cfg.freeze_modality_proj:
            freeze_module(self.perception.proj)
        else:
            unfreeze_module(self.perception.proj)

        # Freeze the LLM body (lm_head and embed_tokens are handled separately)
        if self.core_cfg.freeze_llm_model:
            freeze_module(self.llm.model)
        else:
            unfreeze_module(self.llm.model)

        # lm_head is inside self.llm, so re-apply after the LLM-wide freeze
        if self.core_cfg.freeze_llm_head:
            freeze_module(self.llm.lm_head)
        else:
            unfreeze_module(self.llm.lm_head)

        # embed_tokens is a separate top-level module (moved out of llm)
        if self.core_cfg.freeze_embed_tokens:
            freeze_module(self.embed_tokens)
        else:
            unfreeze_module(self.embed_tokens)

        # Aux chunk-boundary classifier (backbone + linear head). Only present
        # when use_chunk_classifier is True.
        if self.core_cfg.use_chunk_classifier:
            if self.core_cfg.freeze_chunk_classifier:
                freeze_module(self.chunk_classifier_backbone)
                freeze_module(self.chunk_classifier_head)
            else:
                unfreeze_module(self.chunk_classifier_backbone)
                unfreeze_module(self.chunk_classifier_head)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return int(self.embed_tokens.num_embeddings)

    @property
    def text_pad_id(self) -> int:
        pad_id = self.tokenizer.pad_id
        if pad_id is None:
            pad_id = self.tokenizer.unk_id
        if pad_id is None:
            warnings.warn(
                "The text tokenizer has no <pad> or <unk> token; using id 0 for "
                "padding (this may lead to silent bugs)."
            )
            pad_id = 0
        return pad_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def sampling_rate(self) -> int:
        return self.perception.preprocessor.featurizer.sample_rate

    @property
    def sample_rate(self) -> int:
        return self.perception.preprocessor.featurizer.sample_rate

    @property
    def frame_duration(self) -> float:
        """Duration (in seconds) of one audio frame at the perception output."""
        return self.perception.token_equivalent_duration

    @property
    def blank_token_id(self) -> int:
        # Sentinel -1 when blank is disabled — guarantees `token == blank_token_id`
        # never matches a real vocab id, so stop-on-blank checks naturally no-op.
        if self.blank_token == "":
            return -1
        return self.tokenizer.text_to_ids(self.blank_token)[0]

    @property
    def has_blank(self) -> bool:
        return self.blank_token != ""

    # ------------------------------------------------------------------
    # Core: efficient audio-text embedding interleaving
    # ------------------------------------------------------------------

    def _audio_slots_per_chunk(self) -> int:
        """Number of LLM audio tokens (post-subsampling) per ``chunk_size`` chunk.

        ``ceil(chunk_size / S)`` — when ``chunk_size`` is not divisible by the
        factor the final sub-frame is formed from a chunk whose tail was padded
        (see :meth:`_apply_audio_subsampling`). Reduces to ``chunk_size // S`` in
        the divisible case, and to ``chunk_size`` when subsampling is disabled.
        """
        S = self.encoder_subsampling_factor
        C = int(self.core_cfg.chunk_size)
        if S <= 1:
            return C
        return (C + S - 1) // S

    def _n_audio_slots(self, n_encoder_frames: int) -> int:
        """Audio-token count for an audio span of ``n_encoder_frames`` frames.

        The span is treated as ``ceil(n_encoder_frames / chunk_size)`` whole
        chunks, each contributing :meth:`_audio_slots_per_chunk` tokens. This is
        the single source of truth shared by the turn template, streaming
        inference and the offline splitter so they all agree with the dataset's
        per-chunk placeholder count even when ``chunk_size % S != 0``.
        """
        n = int(n_encoder_frames)
        S = self.encoder_subsampling_factor
        if S <= 1:
            return n
        C = int(self.core_cfg.chunk_size)
        n_chunks = (n + C - 1) // C
        return n_chunks * self._audio_slots_per_chunk()

    def _apply_audio_subsampling(
        self, audio_embs: Tensor, emb_lens: Optional[Tensor] = None
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Optionally shrink the encoder time axis with the trainable Conv1d.

        No-op (identity) when ``encoder_subsampling_factor == 1``.

        When ``chunk_size`` is DIVISIBLE by the factor ``S`` the whole ``(B, T, H)``
        input is padded up to a multiple of ``S`` and run through a single
        ``Conv1d(kernel=stride=S)`` (global, unchanged legacy behavior),
        returning ``(B, ceil(T/S), H)``.

        When ``chunk_size`` is NOT divisible by ``S`` a global stride-``S`` conv
        would let sub-frames straddle chunk boundaries and drift out of sync with
        the dataset's per-chunk placeholders. Instead we reshape into
        ``chunk_size``-frame chunks and pad EACH chunk's tail to
        ``ceil(chunk_size/S)*S`` frames by REPEATING its last frame, so every
        chunk cleanly yields ``ceil(chunk_size/S)`` sub-frames. Output is
        ``(B, ceil(T/chunk_size) * ceil(chunk_size/S), H)`` — exactly
        :meth:`_n_audio_slots` ``(T)``.
        """
        if self.audio_subsampler is None:
            return audio_embs, emb_lens
        S = self.encoder_subsampling_factor
        C = int(self.core_cfg.chunk_size)
        B, T, H = audio_embs.shape
        w = self.audio_subsampler.weight

        if C % S == 0:
            # Divisible: legacy global subsampling (numerically unchanged).
            pad = (S - (T % S)) % S
            if pad:
                audio_embs = F.pad(audio_embs, (0, 0, 0, pad))
            x = audio_embs.transpose(1, 2)  # (B, H, T)
            x = self.audio_subsampler(x.to(w.dtype))
            x = x.transpose(1, 2).to(audio_embs.dtype)  # (B, ceil(T/S), H)
            if emb_lens is not None:
                emb_lens = torch.div(emb_lens + (S - 1), S, rounding_mode="floor")
            return x, emb_lens

        # Non-divisible: per-chunk subsampling with last-frame tail padding.
        slots_per_chunk = (C + S - 1) // S
        chunk_pad = slots_per_chunk * S  # C rounded up to a multiple of S
        n_chunks = (T + C - 1) // C
        pad_T = n_chunks * C - T
        if pad_T:
            audio_embs = F.pad(audio_embs, (0, 0, 0, pad_T))  # zero tail beyond audio
        x = audio_embs.reshape(B, n_chunks, C, H)
        rep = chunk_pad - C
        if rep:
            last = x[:, :, -1:, :].expand(B, n_chunks, rep, H)  # repeat each chunk's last frame
            x = torch.cat([x, last], dim=2)  # (B, n_chunks, chunk_pad, H)
        x = x.reshape(B * n_chunks, chunk_pad, H).transpose(1, 2)  # (B*nc, H, chunk_pad)
        x = self.audio_subsampler(x.to(w.dtype))  # (B*nc, H, slots_per_chunk)
        x = x.transpose(1, 2).reshape(B, n_chunks * slots_per_chunk, H).to(audio_embs.dtype)
        if emb_lens is not None:
            valid_chunks = torch.div(emb_lens + (C - 1), C, rounding_mode="floor")
            emb_lens = valid_chunks * slots_per_chunk
        return x, emb_lens

    def _regularize_audio_embs(self, audio_embs: Tensor) -> Tensor:
        """Optionally corrupt the encoder-output embeddings before the LLM.

        Training-only regularization on the (projected, post-subsampling) audio
        embeddings ``(B, T, H)``: standard dropout and/or SpecAugment-style span
        zeroing over frames (time) and hidden dims (feature). All no-ops unless the
        corresponding config knobs are set, so existing models are unchanged. See
        the ``audio_embed_*`` fields on ``StreamingSTTModelConfig``.
        """
        if not self.training or audio_embs is None:
            return audio_embs

        p = float(getattr(self.core_cfg, "audio_embed_dropout", 0.0) or 0.0)
        if p > 0.0:
            audio_embs = F.dropout(audio_embs, p=p, training=True)

        n_time = int(getattr(self.core_cfg, "audio_embed_specaug_time_masks", 0) or 0)
        n_feat = int(getattr(self.core_cfg, "audio_embed_specaug_feature_masks", 0) or 0)
        if n_time > 0 or n_feat > 0:
            audio_embs = self._specaug_audio_embs(audio_embs, n_time, n_feat)
        return audio_embs

    def _specaug_audio_embs(self, audio_embs: Tensor, n_time: int, n_feat: int) -> Tensor:
        """SpecAugment-style span masking on ``(B, T, H)`` audio embeddings.

        Zeros ``n_time`` random contiguous frame spans (each up to
        ``audio_embed_specaug_time_width`` frames) and ``n_feat`` random hidden-dim
        spans (each up to ``audio_embed_specaug_feature_width`` dims), independently
        per sample. Masked values are set to 0 (not rescaled), mirroring classic
        SpecAugment. Sampled fresh on every call.
        """
        B, T, H = audio_embs.shape
        tw = int(getattr(self.core_cfg, "audio_embed_specaug_time_width", 0) or 0)
        fw = int(getattr(self.core_cfg, "audio_embed_specaug_feature_width", 0) or 0)
        tw = min(tw, T)
        fw = min(fw, H)
        if (n_time <= 0 or tw <= 0) and (n_feat <= 0 or fw <= 0):
            return audio_embs

        # Build a multiplicative keep-mask so the op is a single out-of-place
        # multiply (autograd-friendly, no in-place writes on a view).
        keep = audio_embs.new_ones((B, T, H))
        for b in range(B):
            if tw > 0:
                for _ in range(n_time):
                    w = int(torch.randint(1, tw + 1, (1,)).item())
                    t0 = int(torch.randint(0, T - w + 1, (1,)).item())
                    keep[b, t0 : t0 + w, :] = 0.0
            if fw > 0:
                for _ in range(n_feat):
                    w = int(torch.randint(1, fw + 1, (1,)).item())
                    f0 = int(torch.randint(0, H - w + 1, (1,)).item())
                    keep[b, :, f0 : f0 + w] = 0.0
        return audio_embs * keep

    def _build_input_embeds(
        self,
        input_tokens: Tensor,
        audios: Tensor,
        audio_lens: Tensor,
        audio_embs: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """
        Encode audio, embed text tokens, then interleave them.

        This is the high-level entry point used by ``training_step`` and
        ``_eval_step``.  The pure-tensor interleaving logic lives in
        :func:`interleave_embeddings` so it can be tested without a model.

        Args:
            input_tokens: (B, L) token IDs with ``AUDIO_TOKEN_IDX`` at audio
                positions and ``text_pad_id`` at left-padding positions.
            audios: (B, T_samples) raw waveforms.
            audio_lens: (B,) waveform lengths in samples.
            audio_embs: optional precomputed perception encoder output
                (B, T_enc, H). When provided, the (expensive) encoder forward is
                skipped and these embeddings are reused as-is. This is what
                enables ``encoder_reuse_k`` to amortize one encoder pass across
                several resampled partitions of the same audio. The caller is
                responsible for ensuring ``audio_embs`` was produced from the
                exact same ``audios`` (same content + padding) so the audio-slot
                count in ``input_tokens`` matches ``T_enc``.
        Returns:
            dict with keys ``input_embeds`` (B, L, H), ``attention_mask`` (B, L).
        """
        audio_mask = input_tokens == AUDIO_TOKEN_IDX  # (B, L)

        # --- text embeddings ---
        # Zero-out audio positions so embed_tokens gets valid indices.
        text_tokens = input_tokens.where(~audio_mask, torch.zeros_like(input_tokens))
        text_embeds = self.embed_tokens(text_tokens)  # (B, L, H)

        # --- audio embeddings ---
        if audio_embs is None:
            audio_embs, _audio_emb_lens = self.perception(
                input_signal=audios,
                input_signal_length=audio_lens,
            )  # audio_embs: (B, T_enc, H)
            # Optional trainable subsampling: (B, T_enc, H) -> (B, T_enc//S, H).
            # In the encoder-reuse path (audio_embs is not None) the caller has
            # already applied this, so we only subsample on a fresh encode.
            audio_embs, _audio_emb_lens = self._apply_audio_subsampling(audio_embs, _audio_emb_lens)

        # Regularize the encoder output before it enters the LLM (training only).
        audio_embs = self._regularize_audio_embs(audio_embs)

        # --- interleave & build attention mask ---
        inputs = interleave_embeddings(
            input_tokens=input_tokens,
            audio_mask=audio_mask,
            text_embeds=text_embeds,
            audio_embs=audio_embs,
            pad_id=self.text_pad_id,
        )
        if self.core_cfg.use_modality_position_ids:
            inputs["position_ids"] = self._build_modality_position_ids(input_tokens, inputs["attention_mask"])
        if self._uses_interleaved_stream_state:
            is_audio = audio_mask
            if self.core_cfg.use_two_stream_attn:
                attn_bias_4d = build_two_stream_attention_bias(
                    is_audio,
                    inputs["attention_mask"],
                    dtype=inputs["input_embeds"].dtype,
                )
            else:
                chunk_id = build_chunk_ids(is_audio, inputs["attention_mask"])
                attn_bias_4d = build_chunk_local_attention_bias(
                    chunk_id,
                    is_audio,
                    inputs["attention_mask"],
                    dtype=inputs["input_embeds"].dtype,
                    num_visible_audio_chunks=int(self.core_cfg.num_visible_audio_chunks),
                )
            inputs["position_ids"] = build_chunk_local_position_ids(is_audio, inputs["attention_mask"])
            # Keep the 2-D ``attention_mask`` untouched (the aux chunk
            # classifier still consumes it as a (B, L) mask); the LLM forward
            # callers will pick this 4-D additive bias instead.
            inputs["attn_bias_4d"] = attn_bias_4d
        return inputs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _build_modality_position_ids(self, input_tokens: Tensor, attention_mask: Tensor) -> Tensor:
        """Build separated RoPE positions for audio and text/template tokens.

        Audio tokens are numbered ``0, 1, 2, ...`` (contiguous over all audio
        frames in the sequence); text/template tokens are numbered
        ``offset, offset+1, ...`` (contiguous over all non-audio tokens). Each
        stream skips the other modality. ``offset == 0`` makes the two streams
        share the same numeric range, both starting from 0. This builder is the
        single source of truth for training; every inference path advances the
        same two running counters in the same order.
        """
        audio_mask = (input_tokens == AUDIO_TOKEN_IDX) & attention_mask.to(torch.bool)
        text_mask = (input_tokens != AUDIO_TOKEN_IDX) & attention_mask.to(torch.bool)
        offset = int(self.core_cfg.modality_position_offset)
        # Fast shape-only guard: audio positions cannot exceed total sequence length.
        # Avoid per-step tensor.item() here because it synchronizes GPU->CPU and
        # slows the training loop.  Only do the exact check for pathological
        # sequences that are long enough to possibly overlap.
        #
        # offset == 0 is the "shared range" design: audio and text both count
        # from 0 and intentionally share the same RoPE positions, so there is no
        # overlap to guard against.
        if offset > 0 and input_tokens.shape[1] >= offset:
            max_num_audio = int(audio_mask.long().sum(dim=1).max().item()) if audio_mask.numel() > 0 else 0
            if max_num_audio >= offset:
                raise ValueError(
                    "Separated audio/text position ids would overlap: "
                    f"max audio positions needed={max_num_audio}, modality_position_offset={offset}"
                )

        audio_positions = audio_mask.long().cumsum(dim=1) - 1
        text_positions = text_mask.long().cumsum(dim=1) - 1 + offset

        position_ids = torch.zeros_like(input_tokens)
        position_ids = torch.where(audio_mask, audio_positions, position_ids)
        position_ids = torch.where(text_mask, text_positions, position_ids)
        return position_ids

    def _build_streaming_position_ids(
        self,
        token_ids: Tensor,
        audio_mask: Tensor,
        state: StreamingState,
    ) -> Tensor:
        """Build and advance separated positions for cached streaming inputs."""
        if state.audio_position_counter is None or state.text_position_counter is None:
            raise ValueError("StreamingState is missing modality position counters")
        audio_mask = audio_mask.to(torch.bool)
        text_mask = ~audio_mask
        audio_counts = audio_mask.long().sum(dim=1)
        text_counts = text_mask.long().sum(dim=1)
        offset = int(self.core_cfg.modality_position_offset)
        # offset == 0 is the intentional "shared range" design (audio and text both
        # count from 0), so the non-overlap check does not apply there.
        if offset > 0 and os.environ.get("STREAMING_STT_CHECK_POSITION_OVERLAP") == "1":
            if bool((state.audio_position_counter + audio_counts > offset).any().item()):
                raise ValueError("Separated audio/text position ids would overlap in streaming inference")

        audio_positions = state.audio_position_counter.unsqueeze(1) + audio_mask.long().cumsum(dim=1) - 1
        text_positions = state.text_position_counter.unsqueeze(1) + text_mask.long().cumsum(dim=1) - 1
        position_ids = torch.where(audio_mask, audio_positions, text_positions)
        state.audio_position_counter = state.audio_position_counter + audio_counts
        state.text_position_counter = state.text_position_counter + text_counts
        return position_ids

    # ------------------------------------------------------------------
    # Chunk-local audio attention: per-step inference helpers
    # ------------------------------------------------------------------

    @property
    def _uses_interleaved_stream_state(self) -> bool:
        """True when per-token audio/text bookkeeping must be maintained during
        streaming (position counters, is-audio history, per-step 4-D attention
        bias). Shared by chunk-local audio attention and the two-stream scheme;
        the two differ only in the attention-bias rule."""
        return bool(
            self.core_cfg.use_chunk_local_audio_attn or self.core_cfg.use_two_stream_attn
        )

    def _build_chunk_local_step_inputs(
        self,
        state: StreamingState,
        new_is_audio: Tensor,
        active_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build (position_ids, attn_bias_4d, new_chunk_ids, new_is_audio) for one LLM step.

        Args:
            state: Mutable streaming state. ``state.chunk_id_counter``,
                ``state.audio_position_counter``, ``state.text_position_counter``,
                ``state.chunk_id_history``, ``state.is_audio_history`` and
                ``state.attention_mask`` are all read and advanced in place.
            new_is_audio: ``(B, L_q)`` bool, audio flag for the new query
                tokens. The chunk counter is advanced by the number of audio
                spans (runs of True) in each row.
            active_mask: ``(B,)`` long ``{0, 1}``, only active streams advance
                their text/audio counters for this step. When ``None``, all
                streams are active.

        Returns:
            * ``position_ids`` ``(B, L_q)`` long.
            * ``attn_bias_4d`` ``(B, 1, L_q, L_k)`` additive bias.
            * ``new_chunk_ids`` ``(B, L_q)`` long (returned for callers that
              need to append to history themselves after the LLM call).
            * ``new_is_audio`` ``(B, L_q)`` bool (same shape, returned for the
              same reason).
        """
        device = new_is_audio.device
        B, L_q = new_is_audio.shape
        if active_mask is None:
            active = torch.ones(B, dtype=torch.long, device=device)
        else:
            active = active_mask.to(device=device, dtype=torch.long)

        is_text = ~new_is_audio

        prev_audio = F.pad(new_is_audio[:, :-1], (1, 0), value=False)
        audio_span_starts_in_row = new_is_audio & ~prev_audio
        # The first audio frame fed in this call is the start of a NEW chunk
        # only if the row had no audio at the very end of its history. The
        # previous chunk's audio frames in this same call (those that have an
        # audio neighbour at idx-1) just continue the current chunk.
        if state.is_audio_history is not None and state.is_audio_history.shape[1] > 0:
            prev_tail_audio = state.is_audio_history[:, -1]
        else:
            prev_tail_audio = torch.zeros(B, dtype=torch.bool, device=device)
        # If history ended on audio AND this call begins with audio, the
        # first run of audio is a continuation, not a new span.
        starts_with_audio = new_is_audio[:, 0] if L_q > 0 else torch.zeros(B, dtype=torch.bool, device=device)
        suppress_first_start = prev_tail_audio & starts_with_audio
        if L_q > 0:
            audio_span_starts_in_row[:, 0] = audio_span_starts_in_row[:, 0] & ~suppress_first_start

        # Per-row chunk-id offset = current counter + cumsum(audio span starts) within this call.
        # The counter is -1 if no chunk has started yet.
        cumul_new_chunks = audio_span_starts_in_row.long().cumsum(dim=1)  # (B, L_q)
        new_chunk_ids = state.chunk_id_counter.unsqueeze(1) + cumul_new_chunks  # (B, L_q)
        # Inactive streams should not contribute to the counter advance for
        # their own bookkeeping (they only emit filler tokens), but the chunk
        # ids returned still describe what the model sees for THIS forward.
        # Active mask only gates counter advance below.
        total_audio_starts = audio_span_starts_in_row.long().sum(dim=1)  # (B,)

        # Position IDs: audio counter for audio tokens, text counter for text.
        audio_cumsum = new_is_audio.long().cumsum(dim=1) - 1  # (B, L_q), -1 at non-audio
        text_cumsum = is_text.long().cumsum(dim=1) - 1  # (B, L_q), -1 at audio
        audio_positions = state.audio_position_counter.unsqueeze(1) + audio_cumsum
        text_positions = state.text_position_counter.unsqueeze(1) + text_cumsum
        position_ids = torch.where(new_is_audio, audio_positions, text_positions).clamp(min=0)

        # 4-D additive bias across (history, new) for the chunk-local rule.
        if state.attention_mask is None:
            attention_mask_history = torch.empty((B, 0), dtype=torch.long, device=device)
            chunk_id_history = torch.empty((B, 0), dtype=torch.long, device=device)
            is_audio_history = torch.empty((B, 0), dtype=torch.bool, device=device)
        else:
            attention_mask_history = state.attention_mask
            chunk_id_history = state.chunk_id_history
            is_audio_history = state.is_audio_history

        if self.core_cfg.use_two_stream_attn:
            attn_bias_4d = build_two_stream_inference_bias(
                is_audio_history=is_audio_history,
                attention_mask_history=attention_mask_history,
                is_audio_new=new_is_audio,
                dtype=self.embed_tokens.weight.dtype,
            )
        else:
            attn_bias_4d = build_chunk_local_inference_bias(
                chunk_id_history=chunk_id_history,
                is_audio_history=is_audio_history,
                attention_mask_history=attention_mask_history,
                chunk_id_new=new_chunk_ids,
                is_audio_new=new_is_audio,
                dtype=self.embed_tokens.weight.dtype,
                num_visible_audio_chunks=int(self.core_cfg.num_visible_audio_chunks),
            )

        # Advance per-stream counters; inactive streams freeze in place.
        n_audio_per_row = new_is_audio.long().sum(dim=1)  # (B,)
        n_text_per_row = is_text.long().sum(dim=1)  # (B,)
        state.audio_position_counter = state.audio_position_counter + active * n_audio_per_row
        state.text_position_counter = state.text_position_counter + active * n_text_per_row
        state.chunk_id_counter = state.chunk_id_counter + active * total_audio_starts

        return position_ids, attn_bias_4d, new_chunk_ids, new_is_audio

    def _append_chunk_local_history(
        self,
        state: StreamingState,
        new_chunk_ids: Tensor,
        new_is_audio: Tensor,
    ) -> None:
        """Append per-token chunk-local bookkeeping to ``state``."""
        if state.chunk_id_history is None or state.is_audio_history is None:
            state.chunk_id_history = new_chunk_ids
            state.is_audio_history = new_is_audio
        else:
            state.chunk_id_history = torch.cat([state.chunk_id_history, new_chunk_ids], dim=1)
            state.is_audio_history = torch.cat([state.is_audio_history, new_is_audio], dim=1)

    def _build_next_text_position_ids(
        self,
        batch_size: int,
        state: StreamingState,
        device: torch.device,
        active_mask: Optional[list[bool]] = None,
    ) -> Tensor:
        """Build and advance one text position per stream for cached decoding.

        Only streams marked active in ``active_mask`` advance their counter. For
        inactive streams (finished decoders that receive a filler ``<blank>``),
        the previous counter value is reused so the per-stream text position
        does not drift past what training ever observed. Position drift in
        batched generation otherwise inflates each stream's text counter by
        ``max(steps across batch)`` per chunk instead of by that stream's own
        actual text emission.
        """
        if state.text_position_counter is None:
            raise ValueError("StreamingState is missing text position counters")
        position_ids = state.text_position_counter.view(batch_size, 1).to(device=device)
        if active_mask is None:
            active = torch.ones(batch_size, dtype=torch.long, device=device)
        else:
            active = torch.as_tensor(active_mask, dtype=torch.long, device=device)
        state.text_position_counter = state.text_position_counter + active
        return position_ids

    def _text_only_lm_score_active(self) -> bool:
        return (
            self.core_cfg.use_text_only_lm_score_at_inference
            and float(self.core_cfg.text_only_lm_loss_weight) > 0.0
        )

    def _control_token_ids_for_text_lm_score(self) -> set[int]:
        ids = set(getattr(self.tokenizer.tokenizer, "all_special_ids", []) or [])
        for tid in (
            self.blank_token_id,
            self.text_pad_id,
            getattr(self, "_eos_id", None),
            *(getattr(self, "_asst_footer_ids", []) or []),
        ):
            if tid is not None and tid >= 0:
                ids.add(int(tid))
        return ids

    def _apply_text_only_lm_score(self, logits: Tensor, state: Optional[StreamingState]) -> Tensor:
        """Fuse raw-transcript LM logits into streaming logits during decoding."""
        if (
            not self._text_only_lm_score_active()
            or state is None
            or state.text_lm_next_logits is None
            or state.text_lm_next_logits_valid is None
        ):
            return logits

        lm_logits = state.text_lm_next_logits.to(device=logits.device)
        lm_log_probs = F.log_softmax(lm_logits.float(), dim=-1).to(dtype=logits.dtype)
        control_ids = self._control_token_ids_for_text_lm_score()
        if control_ids:
            # The text-only LM is trained on raw transcript tokens only, so do
            # not score streaming control/special tokens such as <blank> and
            # <|im_end|> with this auxiliary LM.
            lm_log_probs[:, list(control_ids)] = 0
        valid = state.text_lm_next_logits_valid.to(device=logits.device, dtype=torch.bool).unsqueeze(1)
        return logits + valid.to(logits.dtype) * float(self.core_cfg.text_only_lm_loss_weight) * lm_log_probs

    def _update_text_only_lm_state(
        self,
        token_ids: Tensor,
        active_mask: list[bool],
        state: StreamingState,
    ) -> None:
        """Feed newly emitted transcript tokens into the cached raw-text LM."""
        if not self._text_only_lm_score_active():
            return
        B = token_ids.shape[0]
        device = token_ids.device
        if not any(active_mask):
            return
        active = torch.as_tensor(active_mask, dtype=torch.long, device=device)

        if state.text_lm_attention_mask is None:
            state.text_lm_attention_mask = torch.zeros(B, 0, dtype=torch.long, device=device)
        if state.text_lm_position_counter is None:
            state.text_lm_position_counter = torch.zeros(B, dtype=torch.long, device=device)
        if state.text_lm_next_logits_valid is None:
            state.text_lm_next_logits_valid = torch.zeros(B, dtype=torch.bool, device=device)

        filler_id = self.text_pad_id
        tokens_to_feed = token_ids.where(active.to(torch.bool), torch.full_like(token_ids, filler_id))
        token_emb = self.embed_tokens(tokens_to_feed.unsqueeze(1))
        position_ids = state.text_lm_position_counter.view(B, 1)
        state.text_lm_position_counter = state.text_lm_position_counter + active
        state.text_lm_attention_mask = torch.cat(
            [state.text_lm_attention_mask, active.view(B, 1).to(dtype=state.text_lm_attention_mask.dtype)],
            dim=1,
        )

        out = self.llm(
            inputs_embeds=token_emb,
            past_key_values=state.text_lm_cache,
            attention_mask=state.text_lm_attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        state.text_lm_cache = out.past_key_values
        next_logits = out.logits[:, -1, :].detach()
        if state.text_lm_next_logits is None:
            state.text_lm_next_logits = next_logits
        else:
            active_bool = active.to(torch.bool).view(B, 1)
            state.text_lm_next_logits = torch.where(active_bool, next_logits, state.text_lm_next_logits)
        state.text_lm_next_logits_valid = state.text_lm_next_logits_valid | active.to(torch.bool)

    def forward(
        self,
        input_embeds: Tensor,
        attention_mask: Tensor | None = None,
        cache=None,
        output_hidden_states: bool = False,
        position_ids: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """
        Forward pass:  embeddings → LLM → logits.

        When ``output_hidden_states=True`` the dict also contains
        ``hidden_states`` (B, L, H) — the LLM's last-layer hidden state, used
        as input to the aux chunk-boundary classifier.
        """
        out = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=cache is not None,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        ans = {"logits": out["logits"]}  # (B, L, V)
        if output_hidden_states:
            ans["hidden_states"] = out["hidden_states"][-1]  # (B, L, H)
        if cache is not None:
            ans["cache"] = out["past_key_values"]
        return ans

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_for_wer(text: str) -> str:
        """Light text normalization for WER."""
        if text is None:
            return ""
        return " ".join(text.strip().lower().split())

    @staticmethod
    def _word_edit_distance(ref_words: List[str], hyp_words: List[str]) -> int:
        """Levenshtein distance on word sequences."""
        n = len(ref_words)
        m = len(hyp_words)
        if n == 0:
            return m
        if m == 0:
            return n

        prev = list(range(m + 1))
        cur = [0] * (m + 1)
        for i in range(1, n + 1):
            cur[0] = i
            for j in range(1, m + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    cur[j] = prev[j - 1]
                else:
                    cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + 1)
            prev, cur = cur, prev
        return prev[m]

    def _compute_wer_stats(self, refs: List[str], hyps: List[str]) -> tuple[float, int, int]:
        """Return (wer, num_errors, num_ref_words)."""
        total_err = 0
        total_words = 0
        for ref, hyp in zip(refs, hyps):
            ref_words = self._normalize_for_wer(ref).split()
            hyp_words = self._normalize_for_wer(hyp).split()
            if len(ref_words) == 0:
                continue
            total_err += self._word_edit_distance(ref_words, hyp_words)
            total_words += len(ref_words)
        wer = float(total_err) / float(total_words) if total_words > 0 else 0.0
        return wer, total_err, total_words

    def _generate_hypotheses_for_wer(
        self, batch: StreamingSTTBatch, parallel: bool = False
    ) -> List[str]:
        """Run inference-time decoding (no teacher forcing) for WER.

        Args:
            batch: Streaming batch to decode.
            parallel: When True, use the parallel chunk heads to emit up to K
                tokens per chunk in 1 prefill + 1 batched feed (vs the AR loop's
                1 prefill + up to N sequential feeds). No-op when the parallel
                heads aren't initialized; falls back to AR.
        """
        max_new_tokens = int(
            getattr(self.core_cfg, "max_new_tokens_per_chunk", DEFAULT_MAX_NEW_TOKENS_PER_CHUNK)
        )
        default_prompt = "Transcribe the audio into text."
        prompt_field = "system_prompt"
        if self.dataset is not None and hasattr(self.dataset, "cfg"):
            default_prompt = getattr(self.dataset.cfg, "system_prompt", default_prompt)
            prompt_field = getattr(self.dataset.cfg, "prompt_field", prompt_field)
        system_prompt: Union[str, List[str]] = default_prompt
        if getattr(batch, "cuts", None) is not None:
            system_prompts = [cut.custom.get(prompt_field, default_prompt) for cut in batch.cuts]
            if len(system_prompts) > 0:
                system_prompt = system_prompts

        was_training = self.training
        if was_training:
            self.eval()
        try:
            with torch.no_grad():
                hyps = self.generate(
                    audios=batch.audios,
                    audio_lens=batch.audio_lens,
                    system_prompt=system_prompt,
                    max_new_tokens=max_new_tokens,
                    parallel_chunk_decode=parallel,
                )
        finally:
            if was_training:
                self.train()
                # Keep frozen submodules in eval mode (same policy as training_step).
                for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
                    if is_frozen(m):
                        m.eval()
        return hyps

    def _log_chunk_alignment_preview(self, batch: StreamingSTTBatch, batch_idx: int, stage: str = "train") -> None:
        """Print one sample's chunk→word assignment lines."""
        if not getattr(batch, "chunk_word_alignment", None):
            return
        if len(batch.chunk_word_alignment) == 0:
            return
        if len(batch.chunk_word_alignment[0]) == 0:
            return

        lines = batch.chunk_word_alignment[0]
        max_lines = 20
        body = "\n".join(f"    {line}" for line in lines[:max_lines])
        if len(lines) > max_lines:
            body += f"\n    ... ({len(lines) - max_lines} more chunks)"
        transcript = batch.text[0] if batch.text else ""
        logging.info(
            "[%s] batch %d chunk-word alignment (sample 0)\n  transcript: `%s`\n%s",
            stage,
            batch_idx,
            transcript,
            body,
        )

    def _text_only_lm_loss(self, texts: Optional[List[str]], device: torch.device) -> tuple[Tensor, Tensor]:
        """Compute auxiliary LM loss on raw transcript text only.

        This branch intentionally does not use the chat template and does not add
        streaming control tokens such as ``<blank>`` or ``<|im_end|>``.  It is a
        plain causal-LM objective over transcript token sequences.
        """
        zero = torch.zeros((), device=device)
        if not texts:
            return zero, torch.zeros((), dtype=torch.long, device=device)

        hf_tok = self.tokenizer.tokenizer
        encoded = [hf_tok.encode(t or "", add_special_tokens=False) for t in texts]
        encoded = [ids for ids in encoded if len(ids) >= 2]
        if not encoded:
            return zero, torch.zeros((), dtype=torch.long, device=device)

        pad_id = self.text_pad_id
        max_len = max(len(ids) for ids in encoded)
        input_tokens = torch.full((len(encoded), max_len), pad_id, dtype=torch.long, device=device)
        target_tokens = torch.full((len(encoded), max_len), IGNORE_INDEX, dtype=torch.long, device=device)

        for row, ids in enumerate(encoded):
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            n = ids_t.numel()
            input_tokens[row, :n] = ids_t
            target_tokens[row, : n - 1] = ids_t[1:]

        attention_mask = input_tokens != pad_id
        input_embeds = self.embed_tokens(input_tokens)
        outputs = self.forward(input_embeds, attention_mask=attention_mask)

        flat_logits = outputs["logits"].flatten(0, 1)
        flat_targets = target_tokens.flatten(0, 1)
        flat_targets = self._sanitize_ce_targets(flat_targets, where="text_only_lm")
        num_targets = (flat_targets != IGNORE_INDEX).long().sum()
        if num_targets == 0:
            return zero, num_targets

        with loss_parallel():
            per_token_loss = F.cross_entropy(
                flat_logits,
                flat_targets,
                reduction="none",
                ignore_index=IGNORE_INDEX,
            )
        return per_token_loss.sum() / num_targets, num_targets

    def _sanitize_ce_targets(self, targets: Tensor, batch_idx: int = -1, where: str = "") -> Tensor:
        """Drop any out-of-range CE target id (map it to ``IGNORE_INDEX``).

        A cross-entropy / ``nll_loss`` target must be either ``IGNORE_INDEX`` or lie
        in ``[0, vocab)``. A single id outside that range triggers an *asynchronous*
        CUDA device-side assert (``nll_loss_forward: cur_target >= 0 && cur_target <
        n_classes``) that kills the whole multi-GPU job with a traceback pointing at
        some unrelated later op. Such ids come from upstream data/alignment glitches
        (observed on granary_v2, where a rare sample carries a target token id >=
        the Qwen vocab). Rather than crash a 16-GPU run on one poisoned token, we set
        just those positions to ``IGNORE_INDEX`` so they are excluded from the loss,
        and log a warning naming the offending id(s). The common case (all ids in
        range) costs one fused comparison + ``.any()`` and returns the tensor as-is.
        """
        vocab = int(self.embed_tokens.num_embeddings)
        bad = (targets != IGNORE_INDEX) & ((targets < 0) | (targets >= vocab))
        if bool(bad.any()):
            n = int(bad.sum().item())
            offending = torch.unique(targets[bad]).tolist()
            logging.warning(
                "[ce-sanitize] batch %s %s: %d target id(s) out of range for vocab=%d "
                "(offending=%s) -> set to IGNORE_INDEX (dropped from loss).",
                batch_idx, where, n, vocab, offending[:16],
            )
            targets = targets.masked_fill(bad, IGNORE_INDEX)
        return targets

    # ------------------------------------------------------------------
    # Parallel chunk heads helpers
    # ------------------------------------------------------------------

    def _compute_parallel_chunk_loss(
        self,
        hidden_states: Tensor,
        anchor_positions: Tensor,
        chunk_targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute the K-slot parallel-chunk-head cross-entropy.

        Args:
            hidden_states: (B, L, H) LLM last hidden states.
            anchor_positions: (B, C) anchor token indices into the L axis;
                values < 0 mark padded chunks (skipped).
            chunk_targets: (B, C, K) per-slot targets with IGNORE_INDEX in
                non-supervised slots.

        Returns:
            (loss, num_supervised_slots): scalar loss averaged over supervised
            slots, and the int count of supervised slots in this batch.
            Returns (zero, zero) if no supervised slot is present.
        """
        device = hidden_states.device
        zero = torch.zeros((), device=device)
        zero_long = torch.zeros((), dtype=torch.long, device=device)
        if self.parallel_chunk_heads is None or anchor_positions is None or anchor_positions.numel() == 0:
            return zero, zero_long

        B, C = anchor_positions.shape
        valid = anchor_positions >= 0  # (B, C)
        if not valid.any():
            return zero, zero_long

        # Per-chunk supervised slot count.
        slot_valid = chunk_targets != IGNORE_INDEX  # (B, C, K)
        num_slots = int(slot_valid.sum().item())
        if num_slots == 0:
            return zero, zero_long

        # Gather hidden states at anchors. Use gather along the L axis.
        # Replace -1 indices with 0 for safety (we'll mask them out next).
        safe_anchor = anchor_positions.clamp(min=0)  # (B, C)
        # hidden_states: (B, L, H) → (B, C, H)
        H = hidden_states.size(-1)
        gather_idx = safe_anchor.unsqueeze(-1).expand(B, C, H)  # (B, C, H)
        gathered = hidden_states.gather(1, gather_idx)  # (B, C, H)

        # Flatten to (M, H) using only valid anchors, and (M, K) targets.
        flat_hidden = gathered[valid]  # (M, H)
        flat_targets = chunk_targets[valid]  # (M, K)
        flat_targets = self._sanitize_ce_targets(flat_targets, where="parallel")

        # Run parallel heads with the tied lm_head.
        par_logits = self.parallel_chunk_heads(flat_hidden, self.llm.lm_head)  # (M, K, V)

        with loss_parallel():
            par_loss = F.cross_entropy(
                par_logits.reshape(-1, par_logits.size(-1)),
                flat_targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="mean",
            )

        return par_loss, torch.as_tensor(num_slots, dtype=torch.long, device=device)

    def training_step(self, batch: StreamingSTTBatch, batch_idx: int):
        # Keep frozen modules in eval mode (disables dropout / batch-norm updates).
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        # ----- Encoder-reuse (Tier-1) orchestration --------------------------
        # K=1 (default) is the legacy single-partition path and is byte-for-byte
        # unchanged. K>1 encodes the audio with the perception encoder ONCE and
        # reuses that encoder output across K independently-resampled chunk/turn
        # partitions of the SAME utterances, averaging their LM losses. This
        # amortizes the expensive speech encoder over K diverse "completion"
        # views per utterance. To resample partitions on the fly it calls
        # ``dataset.get_batch_data`` with fresh word alignments per view; those
        # alignments come from the ONLINE forced aligner when configured, or from
        # the cuts' PRECOMPUTED (offline / pre-aligned) word alignments otherwise.
        # The offline route needs deferred (raw cuts/audio) batches, which the
        # train recipe enables when ``encoder_reuse_k>1``. If neither route is
        # available it transparently falls back to K=1.
        K = max(int(getattr(self.core_cfg, "encoder_reuse_k", 1) or 1), 1)
        # Offline reuse resamples from ``batch.cuts``; require it to be present.
        has_offline_cuts = self.forced_aligner is None and getattr(batch, "cuts", None) is not None
        can_reuse = K > 1 and self.dataset is not None and (self.forced_aligner is not None or has_offline_cuts)
        if K > 1 and not can_reuse and not getattr(self, "_encoder_reuse_disabled_warned", False):
            logging.warning(
                "encoder_reuse_k=%d requested but neither an online forced aligner nor "
                "offline cuts are available to resample partitions (dataset=%s, "
                "forced_aligner=%s, batch.cuts=%s); falling back to encoder_reuse_k=1 "
                "(no reuse). For offline/pre-aligned data make sure defer_get_batch is "
                "on (the train recipe enables it automatically when encoder_reuse_k>1).",
                K,
                self.dataset is not None,
                self.forced_aligner is not None,
                getattr(batch, "cuts", None) is not None,
            )
            self._encoder_reuse_disabled_warned = True

        if not can_reuse:
            # --- Single-partition path ---------------------------------------
            # When the batch arrived DEFERRED (raw cuts/audio, no input_tokens)
            # build it here: online aligner → realign live; otherwise → use the
            # cuts' precomputed offline alignments. A pre-built batch (the
            # default offline K=1 path) is used as-is and stays byte-for-byte
            # unchanged.
            if batch.input_tokens is None and self.dataset is not None:
                if self.forced_aligner is not None:
                    alignments = self.forced_aligner.align(batch.audios, batch.audio_lens, batch.text)
                else:
                    alignments = get_word_alignments_for_batch(batch.cuts)
                batch = self.dataset.get_batch_data(
                    cuts=batch.cuts,
                    audios=batch.audios,
                    audio_lens=batch.audio_lens,
                    alignments=alignments,
                    text=batch.text,
                )
                batch = move_data_to_device(batch, self.device)
            metrics = self._partition_forward_loss(batch, batch_idx, audio_embs=None)
            if metrics.get("skip"):
                return {"loss": metrics["loss"]}
            self._log_train_step(metrics, batch, batch_idx)
            return {"loss": metrics["loss"]}

        # --- K>1: encode once, reuse across K resampled partitions -----------
        orig = batch
        # One alignment per utterance, shared across the K views; only the random
        # delay (resampled inside each get_batch_data call) differs between views.
        if self.forced_aligner is not None:
            alignments = self.forced_aligner.align(orig.audios, orig.audio_lens, orig.text)
        else:
            alignments = get_word_alignments_for_batch(orig.cuts)
        audio_embs = None
        part_metrics = []
        last_batch = None
        # Flush-truncation plan is sampled ONCE and shared across the K views: the
        # encoder output is computed once and reused, so every view must truncate
        # the audio identically (they may differ only in delay randomization).
        # Returns None when the augmentation is inactive → no truncation.
        trunc_plan = self.dataset.sample_flush_truncation_plan(orig.audio_lens)
        for _k in range(K):
            bk = self.dataset.get_batch_data(
                cuts=orig.cuts,
                audios=orig.audios,
                audio_lens=orig.audio_lens,
                alignments=alignments,
                text=orig.text,
                truncation_plan=trunc_plan,
            )
            bk = move_data_to_device(bk, self.device)
            if audio_embs is None:
                # Encode ONCE and share. ``get_batch_data`` pads audio
                # deterministically from (audio_lens, chunk_step), so every
                # resampled partition has identical ``bk.audios`` and therefore
                # the exact same encoder output and audio-slot count.
                audio_embs, _ = self.perception(
                    input_signal=bk.audios,
                    input_signal_length=bk.audio_lens,
                )
                # Subsample ONCE so all K reused partitions share the same
                # (subsampled) encoder output and audio-slot count.
                audio_embs, _ = self._apply_audio_subsampling(audio_embs)
            mk = self._partition_forward_loss(bk, batch_idx, audio_embs=audio_embs)
            if mk.get("skip"):
                continue
            part_metrics.append(mk)
            last_batch = bk

        if not part_metrics:
            return {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)}

        # Optimized objective = mean over the K partition losses. The shared
        # ``audio_embs`` tensor lives in all K graphs, so a single backward
        # accumulates the encoder gradient from every partition.
        loss = torch.stack([m["loss"] for m in part_metrics]).mean()
        agg = {"loss": loss, "inputs": part_metrics[-1]["inputs"]}
        for key in (
            "loss_blank",
            "loss_nonblank",
            "cls_loss_log",
            "text_only_lm_loss_log",
            "text_only_lm_num_targets_log",
            "num_blank",
            "num_targets",
            "par_loss_log",
            "par_num_slots_log",
        ):
            agg[key] = torch.stack([m[key].float() for m in part_metrics]).mean()
        self._log_train_step(agg, last_batch, batch_idx)
        return {"loss": loss}

    def _debug_check_token_ranges(self, batch: StreamingSTTBatch, batch_idx: int) -> None:
        """Opt-in diagnostic: fail LOUDLY (CPU-side) on any out-of-range token/label id.

        A token id fed to the embedding table, or a CE label id, that lies outside
        ``[0, vocab)`` triggers an *asynchronous* CUDA device-side assert whose
        reported traceback points at some unrelated later op (e.g. ``move_data_to_device``).
        Setting ``NEMO_DEBUG_TOKEN_RANGE=1`` runs this cheap range check on the batch
        BEFORE the GPU kernels, so we can report the EXACT offending id and the sample
        it came from -- which is what distinguishes a data problem (an id produced by
        the manifest/aligner that exceeds our tokenizer vocab) from a code bug. This is
        a no-op unless the env var is set, so normal runs pay nothing.
        """
        vocab = int(self.embed_tokens.num_embeddings)
        texts = getattr(batch, "text", None)

        def _report(kind: str, ids, valid_mask) -> None:
            if ids is None:
                return
            checked = ids[valid_mask]
            if checked.numel() == 0:
                return
            mn = int(checked.min().item())
            mx = int(checked.max().item())
            if mn >= 0 and mx < vocab:
                return  # all in range
            bad = valid_mask & ((ids < 0) | (ids >= vocab))
            rows = torch.nonzero(bad.any(dim=1), as_tuple=False).flatten().tolist()
            offending = torch.unique(ids[bad]).tolist()
            samples = []
            for r in rows[:8]:
                t = texts[r] if (texts is not None and r < len(texts)) else "<no text>"
                samples.append(f"    row {r}: {t!r}")
            msg = (
                f"[token-range] batch {batch_idx}: {kind} id(s) OUT OF RANGE for embedding "
                f"vocab={vocab} (observed min={mn}, max={mx}). Offending id(s)="
                f"{offending[:32]} across {len(rows)} sample(s):\n" + "\n".join(samples)
            )
            logging.error(msg)
            raise RuntimeError(msg)

        _report("input_tokens", batch.input_tokens,
                None if batch.input_tokens is None else (batch.input_tokens != AUDIO_TOKEN_IDX))
        _report("target_tokens", batch.target_tokens,
                None if batch.target_tokens is None else (batch.target_tokens != IGNORE_INDEX))

    def _partition_forward_loss(
        self,
        batch: StreamingSTTBatch,
        batch_idx: int,
        audio_embs: Optional[Tensor] = None,
    ) -> dict:
        """Forward + full training loss for ONE chunk/turn partition.

        Pure compute, no logging side effects: returns the optimized ``loss``
        plus the scalar diagnostics + ``inputs`` needed by
        :meth:`_log_train_step`. When ``audio_embs`` is provided the perception
        encoder forward is skipped and those embeddings are reused (this is what
        makes ``encoder_reuse_k`` amortize one encoder pass over K partitions).
        Returns ``{"skip": True, ...}`` for an empty-target batch.
        """
        if os.environ.get("NEMO_DEBUG_TOKEN_RANGE"):
            self._debug_check_token_ranges(batch, batch_idx)
        inputs = self._build_input_embeds(
            batch.input_tokens, batch.audios, batch.audio_lens, audio_embs=audio_embs
        )
        use_aux = self.core_cfg.use_chunk_classifier
        use_parallel = (
            self._parallel_heads_enabled
            and self.core_cfg.parallel_loss_weight > 0
            and batch.chunk_anchor_positions is not None
            and batch.chunk_anchor_positions.numel() > 0
        )
        # Loud one-shot diagnostic: catch the common misconfig where the
        # parallel-heads module is built on the model side (so the metric
        # appears in wandb) but the dataset never emits chunk_anchor_positions
        # (e.g. because compact_template was not propagated to data_cfg). Left
        # silent, this presents as ``loss_parallel_chunk=0`` for an entire run.
        if (
            self._parallel_heads_enabled
            and self.core_cfg.parallel_loss_weight > 0
            and not use_parallel
            and not getattr(self, "_parallel_heads_disabled_warned", False)
        ):
            anchors = batch.chunk_anchor_positions
            logging.warning(
                "ParallelChunkHeads is ENABLED on the model (parallel_loss_weight=%.3f) "
                "but batch %d has no chunk anchors (batch.chunk_anchor_positions=%s). "
                "Most likely the dataset config does not have compact_template=True or "
                "parallel_chunk_slots>0. Verify that the dataset logged "
                "'compact_template enabled' and 'parallel_chunk_slots' at startup. "
                "Until this is fixed loss_parallel_chunk will stay at 0.",
                float(self.core_cfg.parallel_loss_weight),
                batch_idx,
                "None" if anchors is None else f"shape {tuple(anchors.shape)}",
            )
            self._parallel_heads_disabled_warned = True
        # Chunk-local audio attention swaps the 2-D mask for a 4-D additive
        # bias on the LLM forward only; the aux head keeps consuming the
        # original 2-D ``inputs["attention_mask"]``.
        llm_attn_mask = inputs.get("attn_bias_4d", inputs["attention_mask"])
        outputs = self.forward(
            inputs["input_embeds"],
            attention_mask=llm_attn_mask,
            output_hidden_states=use_aux or use_parallel,
            position_ids=inputs.get("position_ids"),
        )

        target_ids = batch.target_tokens

        # When the aux chunk classifier is active, strip audio-frame positions
        # from the LM CE so the LM head is only supervised on text. The aux
        # head (below) handles the boundary decision via BCE. Use the input-axis
        # audio mask — NOT a target-value mask — so end-of-chunk blanks at
        # text positions (line 1696/1701 in inference) remain supervised.
        if use_aux:
            audio_mask = batch.input_tokens == AUDIO_TOKEN_IDX  # (B, L)
            if not self.core_cfg.chunk_classifier_keep_lm_supervision_at_audio:
                target_ids = torch.where(audio_mask, torch.full_like(target_ids, IGNORE_INDEX), target_ids)
            # else: keep blank / user_footer_first targets at audio positions —
            # LM head is co-supervised alongside the aux BCE.
        else:
            audio_mask = None

        # Guard against rare out-of-vocab target ids (data glitch) that would
        # otherwise trip an async CUDA device-side assert in the CE below and kill
        # the whole job. Do this BEFORE counting targets so the normalizer matches.
        target_ids = self._sanitize_ce_targets(target_ids, batch_idx, where="ar")

        num_targets = (target_ids != IGNORE_INDEX).long().sum()

        if num_targets == 0:
            # DDP-safe no-op step. A fresh ``torch.tensor(0.0, requires_grad=True)``
            # here is DISCONNECTED from the model graph, so this rank's backward
            # fires no gradient-reduction hooks while other ranks (whose batches
            # DO have targets) reduce real gradients. Under multi-node DDP that
            # desyncs the NCCL collectives and hangs the job (observed as an
            # ALLREDUCE watchdog timeout). Instead return a zero-valued loss that
            # is still connected to every trainable parameter — ``logits`` depends
            # on the LLM, ``embed_tokens`` and ``perception`` — so the reducer sees
            # the same parameters become ready on every rank. The ``* 0.0`` makes
            # it a true mathematical no-op (zero gradient contribution).
            logging.warning(
                "Batch %d: num_targets is 0 — using a graph-connected zero-loss "
                "keep-alive (DDP-safe) instead of skipping.",
                batch_idx,
            )
            keepalive = outputs["logits"].sum() * 0.0
            if self._parallel_heads_enabled and self.parallel_chunk_heads is not None:
                keepalive = keepalive + sum(p.sum() for p in self.parallel_chunk_heads.parameters()) * 0.0
            return {"skip": True, "loss": keepalive}

        logits = outputs["logits"]

        # # Diagnose NaN sources (remove once stable).
        # if torch.isnan(inputs["input_embeds"]).any():
        #     logging.warning("Batch %d: NaN in input_embeds", batch_idx)
        # if torch.isnan(logits).any():
        #     logging.warning("Batch %d: NaN in logits", batch_idx)

        flat_logits = logits.flatten(0, 1)
        flat_targets = target_ids.flatten(0, 1)

        with loss_parallel():
            per_token_loss = F.cross_entropy(
                flat_logits,
                flat_targets,
                reduction="none",
                ignore_index=IGNORE_INDEX,
            )

        # --- Position weights (delay-randomized / position-weighted objective) ---
        # ``batch.target_weights`` (when present) holds a per-target-token CE weight
        # ``gamma ** p`` aligned with ``target_tokens``. ``None`` => uniform (legacy).
        # Each supervised token contributes ``w * CE`` and the normalizer is the
        # SUM of weights (so gamma=1/all-ones reduces EXACTLY to the mean CE).
        if getattr(batch, "target_weights", None) is not None:
            flat_weights = batch.target_weights.flatten(0, 1).to(per_token_loss.dtype)
        else:
            flat_weights = None

        # --- Blank vs non-blank loss breakdown ---
        blank_id = self.blank_token_id
        valid_mask = flat_targets != IGNORE_INDEX
        # When blank is disabled (blank_id=-1), is_blank is always False →
        # everything counts as non-blank, and blank_weight has no effect.
        is_blank = valid_mask & (flat_targets == blank_id)
        is_nonblank = valid_mask & (flat_targets != blank_id)
        num_blank = is_blank.sum()
        num_nonblank = is_nonblank.sum()

        # Apply blank loss weight (< 1.0 to down-weight easy blank predictions)
        blank_weight = self.core_cfg.blank_loss_weight
        if num_blank > 0 and blank_weight != 1.0 and self.has_blank:
            if flat_weights is not None:
                nb_w = flat_weights[is_nonblank]
                bl_w = flat_weights[is_blank]
                effective_num_targets = (nb_w.sum() + bl_w.sum() * blank_weight).clamp(min=1e-8)
                loss = (
                    (per_token_loss[is_nonblank] * nb_w).sum()
                    + (per_token_loss[is_blank] * bl_w).sum() * blank_weight
                ) / effective_num_targets
            else:
                effective_num_targets = num_blank * blank_weight + num_nonblank
                loss = (
                    per_token_loss[is_nonblank].sum() + per_token_loss[is_blank].sum() * blank_weight
                ) / effective_num_targets
        elif flat_weights is not None:
            w_valid = flat_weights[valid_mask]
            loss = (per_token_loss[valid_mask] * w_valid).sum() / w_valid.sum().clamp(min=1e-8)
        else:
            loss = per_token_loss.sum() / num_targets

        with torch.no_grad():
            loss_blank = per_token_loss[is_blank].sum() / num_blank.clamp(min=1)
            loss_nonblank = per_token_loss[is_nonblank].sum() / num_nonblank.clamp(min=1)

        text_only_lm_loss_log = torch.zeros((), device=loss.device)
        text_only_lm_num_targets_log = torch.zeros((), dtype=torch.long, device=loss.device)
        if self.core_cfg.text_only_lm_loss_weight > 0:
            text_only_lm_loss, text_only_lm_num_targets = self._text_only_lm_loss(batch.text, loss.device)
            loss = loss + float(self.core_cfg.text_only_lm_loss_weight) * text_only_lm_loss
            text_only_lm_loss_log = text_only_lm_loss.detach()
            text_only_lm_num_targets_log = text_only_lm_num_targets.detach()

        # --- Parallel chunk heads loss ---
        # Gather hidden states at each chunk's anchor (write_id position) and
        # supervise K parallel heads against the per-chunk target slate.
        # Auxiliary loss; added on top of the standard AR per-token CE.
        # Compute the parallel-chunk term *separately* from the AR/aux ``loss``
        # accumulator so that, in parallel-only mode, we can drop the
        # non-parallel terms while still keeping their params in the graph.
        par_term = torch.zeros((), device=loss.device)
        par_loss_log = torch.zeros((), device=loss.device)
        par_num_slots_log = torch.zeros((), dtype=torch.long, device=loss.device)
        if use_parallel:
            par_loss, par_num_slots = self._compute_parallel_chunk_loss(
                hidden_states=outputs["hidden_states"],
                anchor_positions=batch.chunk_anchor_positions,
                chunk_targets=batch.chunk_target_tokens,
            )
            if par_num_slots > 0:
                par_term = float(self.core_cfg.parallel_loss_weight) * par_loss
                par_loss_log = par_loss.detach()
                par_num_slots_log = par_num_slots.detach()

        # --- Aux chunk-boundary classifier loss ---
        # BCE on the aux head's binary "ready to emit" prediction at audio frames.
        # Supervised positions: input is an audio frame AND original target was a
        # decision token (blank=keep listening, user_footer_first=emit). Using the
        # input-axis audio mask here exactly mirrors the supervision the LM head
        # used to provide at audio positions before §4(a) masked them out.
        cls_loss_log = torch.zeros((), device=loss.device)
        if use_aux and self.has_blank and audio_mask is not None and self._user_footer_first_id is not None:
            audio_mask_flat = audio_mask.flatten(0, 1)  # (B*L,)
            orig_targets_flat = batch.target_tokens.flatten(0, 1)  # pre-LM-CE-masking
            decision_mask = audio_mask_flat & (orig_targets_flat != IGNORE_INDEX)
            num_decisions = decision_mask.sum()
            if num_decisions > 0:
                aux_input = outputs["hidden_states"]  # (B, L, H)
                if self.core_cfg.chunk_classifier_stop_grad_to_llm:
                    aux_input = aux_input.detach()
                aux_out = self.chunk_classifier_backbone(
                    inputs_embeds=aux_input,
                    attention_mask=inputs["attention_mask"],
                    return_dict=True,
                )
                flat_aux = aux_out.last_hidden_state.flatten(0, 1)  # (B*L, H)
                cls_logits = self.chunk_classifier_head(flat_aux[decision_mask]).squeeze(-1)
                cls_targets = (orig_targets_flat[decision_mask] == self._user_footer_first_id).to(cls_logits.dtype)
                # Auto-balance: pos_weight = N_neg/N_pos. Skip when either class
                # is empty in this batch (pos_weight would zero out one side).
                num_pos = cls_targets.sum()
                num_neg = cls_targets.numel() - num_pos
                if self.core_cfg.chunk_classifier_auto_balance and num_pos > 0 and num_neg > 0:
                    pos_weight = (num_neg.float() / num_pos.float()).detach()
                else:
                    pos_weight = None
                cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_targets, pos_weight=pos_weight)
                cls_w = self.core_cfg.chunk_classifier_loss_weight
                loss = loss + cls_w * cls_loss
                cls_loss_log = cls_loss.detach()
                cls_pos_ratio = num_pos.float() / num_decisions.clamp(min=1).float()
                self.log_dict(
                    {"loss_chunk_cls_pos_ratio": cls_pos_ratio},
                    on_step=True,
                )

        # --- Combine the optimized objective ---
        # At this point ``loss`` holds the non-parallel terms (AR per-token CE +
        # optional text-only-LM + optional chunk-classifier). ``par_term`` holds
        # the (weighted) parallel-chunk loss.
        if self.core_cfg.parallel_only_loss:
            # Train on the parallel loss ONLY. Keep the non-parallel terms in
            # the graph at weight 0 so every AR/aux parameter still receives a
            # (zero) gradient and DDP find_unused_parameters=false does not trip
            # when AR supervision is dropped (e.g. parallel_loss_weight=1.0).
            loss = par_term + loss * 0.0
        else:
            loss = loss + par_term
        # DDP keep-alive: ensure every ParallelChunkHeads parameter appears in
        # the backward graph on every step, even when this rank's batch has no
        # valid anchors. Without this, ``find_unused_parameters=false`` trips
        # when one rank has anchors and another does not. The 0.0 multiplier
        # makes this a true mathematical no-op (zero gradient contribution).
        if self._parallel_heads_enabled and self.parallel_chunk_heads is not None:
            params_touch = sum(p.sum() for p in self.parallel_chunk_heads.parameters())
            loss = loss + params_touch * 0.0

        return {
            "loss": loss,
            "loss_blank": loss_blank,
            "loss_nonblank": loss_nonblank,
            "cls_loss_log": cls_loss_log,
            "text_only_lm_loss_log": text_only_lm_loss_log,
            "text_only_lm_num_targets_log": text_only_lm_num_targets_log,
            "num_blank": num_blank,
            "num_targets": num_targets,
            "par_loss_log": par_loss_log,
            "par_num_slots_log": par_num_slots_log,
            "inputs": inputs,
        }

    def _log_train_step(self, metrics: dict, batch: StreamingSTTBatch, batch_idx: int) -> None:
        """Log per-step training metrics and periodic WER/decoding previews.

        Split out of ``training_step`` so the encoder-reuse (K>1) path can log a
        single aggregated payload. For K=1 the inputs are exactly the values the
        inline code used, so logging is unchanged.
        """
        loss = metrics["loss"]
        par_loss_log = metrics["par_loss_log"]
        par_num_slots_log = metrics["par_num_slots_log"]
        inputs = metrics["inputs"]

        B = inputs["input_embeds"].shape[0]
        log_payload = {
            "loss": loss,
            "learning_rate": torch.as_tensor(
                self.trainer.optimizers[0].param_groups[0]["lr"] if self._trainer is not None else 0
            ),
            "batch_size": float(B),
        }
        # Only surface the parallel-head metrics when the feature is enabled,
        # so existing runs emit the exact same metric keys as before.
        if self._parallel_heads_enabled:
            log_payload["loss_parallel_chunk"] = par_loss_log
            log_payload["parallel_chunk_num_slots"] = par_num_slots_log.float()
        self.log_dict(log_payload, on_step=True)

        should_log_debug = (
            self.core_cfg.log_every_n_steps > 0
            and self.global_step > 0
            and self.global_step % self.core_cfg.log_every_n_steps == 0
        )
        if should_log_debug:
            self._log_chunk_alignment_preview(batch, batch_idx=batch_idx, stage="train")

            # WER from actual inference-time decoding (not teacher forcing).
            # Normally ``train_wer`` is the AR-decode WER (back-compat). In
            # parallel-only training the AR path is not optimized, so we SKIP
            # the AR decode entirely (it's wasteful + its WER is meaningless)
            # and let the parallel decode own ``train_wer`` — mirroring how
            # ``val_wer`` is handled. When both paths are trained we log
            # ``train_wer`` (=AR), ``train_wer_ar`` (alias) and
            # ``train_wer_parallel`` so their divergence is visible.
            parallel_only = bool(self.core_cfg.parallel_only_loss) and self._parallel_heads_enabled
            refs = batch.text if batch.text is not None else None

            hyps_ar = None
            train_wer_ar = None
            if not parallel_only:
                hyps_ar = self._generate_hypotheses_for_wer(batch, parallel=False)
                ar_refs = refs if refs is not None else [""] * len(hyps_ar)
                train_wer_ar, _, _ = self._compute_wer_stats(ar_refs, hyps_ar)
                # NOTE: sync_dist=False on purpose. These periodic train-WER logs
                # only fire on ``should_log_debug`` steps, but a rank whose
                # microbatch has num_targets==0 takes the keepalive path and
                # early-returns before ``_log_train_step`` — so it never joins a
                # synced collective here. sync_dist=True would then desync the
                # NCCL allreduce across ranks and hang the job (watchdog timeout
                # on a NumelIn=1 ALLREDUCE). Keep these as rank-local diagnostics.
                self.log(
                    "train_wer",
                    torch.tensor(train_wer_ar, device=loss.device),
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=False,
                )
                if self._parallel_heads_enabled:
                    self.log(
                        "train_wer_ar",
                        torch.tensor(train_wer_ar, device=loss.device),
                        on_step=True,
                        on_epoch=False,
                        logger=True,
                        sync_dist=False,
                    )

            hyps_par = None
            train_wer_par = None
            if self._parallel_heads_enabled:
                hyps_par = self._generate_hypotheses_for_wer(batch, parallel=True)
                par_refs = refs if refs is not None else [""] * len(hyps_par)
                train_wer_par, _, _ = self._compute_wer_stats(par_refs, hyps_par)
                # sync_dist=False: see the note on train_wer above — the
                # keepalive (num_targets==0) path can skip this collective on
                # some ranks and hang DDP.
                self.log(
                    "train_wer_parallel",
                    torch.tensor(train_wer_par, device=loss.device),
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=False,
                )
                if parallel_only:
                    # Parallel decode IS the primary metric in parallel-only mode.
                    self.log(
                        "train_wer",
                        torch.tensor(train_wer_par, device=loss.device),
                        on_step=True,
                        on_epoch=False,
                        logger=True,
                        sync_dist=False,
                    )

            # Periodic scalar stats next to the alignment/infer preview: current
            # loss, train WER (of whichever decode path is primary) and the
            # last/best validation WER seen so far.
            train_wer_primary = train_wer_par if parallel_only else train_wer_ar
            logging.info(
                "[train] step %d stats: loss=%.4f  train_wer=%s  best_val_wer=%s  last_val_wer=%s",
                self.global_step,
                float(loss),
                ("%.4f" % train_wer_primary) if train_wer_primary is not None else "n/a",
                ("%.4f" % self._best_val_wer) if getattr(self, "_best_val_wer", None) is not None else "n/a",
                ("%.4f" % self._last_val_wer) if getattr(self, "_last_val_wer", None) is not None else "n/a",
            )

            if refs:
                if parallel_only and hyps_par is not None:
                    # Per-utterance breakdown so a single runaway (over-emission)
                    # decode that dominates this small batch's micro-averaged WER
                    # is visible. ``val_wer`` dilutes such utterances over the
                    # whole val set; a single training batch cannot — which is the
                    # usual reason ``train_wer`` >> ``val_wer`` here. If you see
                    # one utterance with hyp_w >> ref_w (wer >> 1), that's a
                    # degenerate parallel decode, NOT a metric bug.
                    per_utt = []
                    for r, h in zip(par_refs, hyps_par):
                        rw = self._normalize_for_wer(r).split()
                        hw = self._normalize_for_wer(h).split()
                        w = (self._word_edit_distance(rw, hw) / len(rw)) if len(rw) > 0 else float("nan")
                        per_utt.append((len(rw), len(hw), w))
                    worst_i = 0
                    worst_w = -1.0
                    for i, (_, _, w) in enumerate(per_utt):
                        if w == w and w > worst_w:  # skip NaN (empty ref)
                            worst_w, worst_i = w, i
                    logging.info(
                        "[train] batch %d parallel WER breakdown: batch_wer=%.4f n_utt=%d\n"
                        "  per_utt(ref_w,hyp_w,wer)=%s\n"
                        "  worst utt #%d (wer=%.3f)\n    ref: `%s`\n    hyp: `%s`",
                        batch_idx,
                        train_wer_par if train_wer_par is not None else float("nan"),
                        len(per_utt),
                        [(rw, hw, round(w, 3) if w == w else None) for rw, hw, w in per_utt],
                        worst_i,
                        worst_w if worst_w >= 0 else float("nan"),
                        par_refs[worst_i],
                        hyps_par[worst_i],
                    )
                elif hyps_ar and hyps_par is not None:
                    logging.info(
                        "[train] batch %d infer sample\n  ref: `%s`\n  hyp_ar: `%s`\n  hyp_par: `%s`\n  "
                        "wer_ar(batch)=%.4f  wer_par(batch)=%.4f",
                        batch_idx,
                        refs[0],
                        hyps_ar[0],
                        hyps_par[0],
                        train_wer_ar,
                        train_wer_par if train_wer_par is not None else float("nan"),
                    )
                elif hyps_ar:
                    logging.info(
                        "[train] batch %d infer sample\n  ref: `%s`\n  hyp: `%s`\n  wer(batch)=%.4f",
                        batch_idx,
                        refs[0],
                        hyps_ar[0],
                        train_wer_ar,
                    )

    def configure_optimizers(self):
        return configure_optimizers(self)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._partial_wer_errors: dict[str, list] = defaultdict(list)
        self._partial_wer_words: dict[str, list] = defaultdict(list)
        # Parallel-decode WER accumulators (only populated when parallel heads
        # are enabled). Same shape semantics as the AR counterparts.
        self._partial_wer_errors_parallel: dict[str, list] = defaultdict(list)
        self._partial_wer_words_parallel: dict[str, list] = defaultdict(list)
        # Per-class TP/total counts for the aux chunk classifier. Aggregated
        # across the epoch so macro acc isn't biased by per-batch composition.
        self._partial_aux_pos_correct: dict[str, list] = defaultdict(list)
        self._partial_aux_pos_total: dict[str, list] = defaultdict(list)
        self._partial_aux_neg_correct: dict[str, list] = defaultdict(list)
        self._partial_aux_neg_total: dict[str, list] = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        # In parallel-only training the AR decode path is not optimized, so the
        # AR WER is meaningless as the primary monitor. Demote it to
        # ``val_wer_ar`` and let the parallel WER own ``val_wer`` (below) so the
        # checkpoint monitor (val_wer) tracks the actually-trained objective.
        parallel_only = bool(self.core_cfg.parallel_only_loss) and self._parallel_heads_enabled

        # Monitored ``val_wer`` (whichever decode path owns it) — captured so the
        # periodic training-log stats can report the last/best validation WER.
        monitored_val_wer = None

        wers = []
        for name, errs in self._partial_wer_errors.items():
            words = self._partial_wer_words[name]
            total_err = torch.stack(errs).sum()
            total_words = torch.stack(words).sum().clamp(min=1)
            val_wer = total_err.float() / total_words.float()
            if not parallel_only:
                self.log(f"val_wer_{name}", val_wer, on_epoch=True, sync_dist=True)
            if self._parallel_heads_enabled:
                self.log(f"val_wer_ar_{name}", val_wer, on_epoch=True, sync_dist=True)
            wers.append(val_wer)
        if wers:
            mean_wer = torch.stack(wers).mean()
            if not parallel_only:
                self.log("val_wer", mean_wer, on_epoch=True, sync_dist=True)
                monitored_val_wer = mean_wer
            if self._parallel_heads_enabled:
                self.log("val_wer_ar", mean_wer, on_epoch=True, sync_dist=True)

        # Parallel-decode WER (only when parallel heads are active — for runs
        # without parallel heads, ``_partial_wer_errors_parallel`` is empty and
        # this block is a no-op, preserving the original metric set exactly).
        # When parallel_only_loss is set, this WER also *is* ``val_wer``.
        if self._parallel_heads_enabled:
            wers_par = []
            for name, errs in self._partial_wer_errors_parallel.items():
                words = self._partial_wer_words_parallel[name]
                if not errs:
                    continue
                total_err = torch.stack(errs).sum()
                total_words = torch.stack(words).sum().clamp(min=1)
                val_wer_par = total_err.float() / total_words.float()
                self.log(f"val_wer_parallel_{name}", val_wer_par, on_epoch=True, sync_dist=True)
                if parallel_only:
                    self.log(f"val_wer_{name}", val_wer_par, on_epoch=True, sync_dist=True)
                wers_par.append(val_wer_par)
            if wers_par:
                mean_par = torch.stack(wers_par).mean()
                self.log("val_wer_parallel", mean_par, on_epoch=True, sync_dist=True)
                if parallel_only:
                    self.log("val_wer", mean_par, on_epoch=True, sync_dist=True)
                    monitored_val_wer = mean_par

        # Remember last/best monitored val_wer so the periodic training log can
        # surface them next to train_wer/loss. Values are rank-local (fine for a
        # diagnostic log line; the checkpoint monitor uses the synced value).
        if monitored_val_wer is not None:
            self._last_val_wer = float(monitored_val_wer)
            prev_best = getattr(self, "_best_val_wer", None)
            if prev_best is None or self._last_val_wer < prev_best:
                self._best_val_wer = self._last_val_wer

        # --- Aux chunk classifier: macro accuracy ---
        # Sum per-class counts across the epoch and compute pos/neg accuracy
        # once at the end. Macro acc = (pos_acc + neg_acc) / 2 — class-balanced.
        macro_accs = []
        for name in self._partial_aux_pos_total.keys():
            pos_correct = torch.stack(self._partial_aux_pos_correct[name]).sum()
            pos_total = torch.stack(self._partial_aux_pos_total[name]).sum()
            neg_correct = torch.stack(self._partial_aux_neg_correct[name]).sum()
            neg_total = torch.stack(self._partial_aux_neg_total[name]).sum()
            pos_acc = pos_correct.float() / pos_total.clamp(min=1).float()
            neg_acc = neg_correct.float() / neg_total.clamp(min=1).float()
            macro = (pos_acc + neg_acc) / 2
            self.log(f"val_aux_pos_acc_{name}", pos_acc, on_epoch=True, sync_dist=True)
            self.log(f"val_aux_neg_acc_{name}", neg_acc, on_epoch=True, sync_dist=True)
            self.log(f"val_aux_macro_acc_{name}", macro, on_epoch=True, sync_dist=True)
            macro_accs.append(macro)
        if macro_accs:
            self.log("val_aux_macro_acc", torch.stack(macro_accs).mean(), on_epoch=True, sync_dist=True)

        self._partial_wer_errors.clear()
        self._partial_wer_words.clear()
        self._partial_wer_errors_parallel.clear()
        self._partial_wer_words_parallel.clear()
        self._partial_aux_pos_correct.clear()
        self._partial_aux_pos_total.clear()
        self._partial_aux_neg_correct.clear()
        self._partial_aux_neg_total.clear()

    def validation_step(self, batch, batch_idx: int):
        # Support multiple validation dataloaders ({name: batch} dict).
        if isinstance(batch, dict):
            for name, dataset_batch in batch.items():
                if dataset_batch is not None:
                    self._eval_step(dataset_batch, name, batch_idx)
        else:
            self._eval_step(batch, "val", batch_idx)

    def _eval_step(self, batch: StreamingSTTBatch, name: str, batch_idx: int = 0) -> None:
        # When the batch arrived DEFERRED (raw cuts/audio — online forced aligner
        # OR offline encoder_reuse_k>1 both defer), build it here. Online aligner
        # → realign live; otherwise → use the cuts' precomputed offline
        # alignments. Either way validation is deterministic: no random alignment
        # delay (the position weighting itself is deterministic and stays active).
        if batch.input_tokens is None and self.dataset is not None:
            if self.forced_aligner is not None:
                alignments = self.forced_aligner.align(batch.audios, batch.audio_lens, batch.text)
            else:
                alignments = get_word_alignments_for_batch(batch.cuts)
            batch = self.dataset.get_batch_data(
                cuts=batch.cuts,
                audios=batch.audios,
                audio_lens=batch.audio_lens,
                alignments=alignments,
                text=batch.text,
                randomize_fixed_chunk_groups=False,
                apply_random_delay=False,
            )
            batch = move_data_to_device(batch, self.device)

        inputs = self._build_input_embeds(batch.input_tokens, batch.audios, batch.audio_lens)
        aux_active = self.core_cfg.use_chunk_classifier and self.has_blank and self._user_footer_first_id is not None
        # See ``training_step``: under chunk-local audio attention we feed the
        # LLM the 4-D additive bias and keep the 2-D mask for the aux head.
        llm_attn_mask = inputs.get("attn_bias_4d", inputs["attention_mask"])
        outputs = self.forward(
            inputs["input_embeds"],
            attention_mask=llm_attn_mask,
            output_hidden_states=aux_active,
            position_ids=inputs.get("position_ids"),
        )

        # WER from inference-time decoding (non-teacher-forced). In
        # parallel-only training the AR path is not optimized, so skip the AR
        # decode entirely (wasteful + its WER is meaningless); the parallel
        # decode below then owns ``val_wer``.
        parallel_only = bool(self.core_cfg.parallel_only_loss) and self._parallel_heads_enabled
        refs = batch.text if batch.text is not None else None

        if not parallel_only:
            hyps = self._generate_hypotheses_for_wer(batch, parallel=False)
            ar_refs = refs if refs is not None else [""] * len(hyps)
            batch_wer, num_err, num_words = self._compute_wer_stats(ar_refs, hyps)
            self._partial_wer_errors[name].append(torch.tensor(float(num_err), device=self.device))
            self._partial_wer_words[name].append(torch.tensor(float(num_words), device=self.device))

        # Parallel-decode WER from the same batch (only when heads are enabled).
        if self._parallel_heads_enabled:
            hyps_par = self._generate_hypotheses_for_wer(batch, parallel=True)
            par_refs = refs if refs is not None else [""] * len(hyps_par)
            batch_wer_par, num_err_par, num_words_par = self._compute_wer_stats(par_refs, hyps_par)
            self._partial_wer_errors_parallel[name].append(
                torch.tensor(float(num_err_par), device=self.device)
            )
            self._partial_wer_words_parallel[name].append(
                torch.tensor(float(num_words_par), device=self.device)
            )

        # --- Aux chunk classifier: per-class correct/total counts ---
        if aux_active:
            audio_mask = batch.input_tokens == AUDIO_TOKEN_IDX  # (B, L)
            audio_mask_flat = audio_mask.flatten(0, 1)
            orig_targets_flat = batch.target_tokens.flatten(0, 1)
            decision_mask = audio_mask_flat & (orig_targets_flat != IGNORE_INDEX)
            if decision_mask.any():
                aux_out = self.chunk_classifier_backbone(
                    inputs_embeds=outputs["hidden_states"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True,
                )
                flat_aux = aux_out.last_hidden_state.flatten(0, 1)
                cls_logits = self.chunk_classifier_head(flat_aux[decision_mask]).squeeze(-1)
                cls_targets = orig_targets_flat[decision_mask] == self._user_footer_first_id
                # Use the configured threshold so val acc reflects what
                # _generate_dynamic_streaming will actually do at inference.
                thr = self.core_cfg.chunk_classifier_threshold
                cls_preds = torch.sigmoid(cls_logits) >= thr
                correct = cls_preds == cls_targets
                pos_mask = cls_targets
                neg_mask = ~cls_targets
                self._partial_aux_pos_correct[name].append((correct & pos_mask).sum().detach())
                self._partial_aux_pos_total[name].append(pos_mask.sum().detach())
                self._partial_aux_neg_correct[name].append((correct & neg_mask).sum().detach())
                self._partial_aux_neg_total[name].append(neg_mask.sum().detach())

        # Log inference-time decoded predictions vs references periodically.
        # In parallel-only mode the AR branch above is skipped, so ``hyps`` /
        # ``batch_wer`` don't exist — fall back to the parallel-decode results
        # (which own ``val_wer`` in that mode).
        if batch_idx % self.core_cfg.log_every_n_steps == 0:
            ref_text = refs[0] if refs else ""
            if parallel_only and self._parallel_heads_enabled:
                log_hyps = hyps_par
                log_wer = batch_wer_par
            else:
                log_hyps = hyps
                log_wer = batch_wer
            hyp_text = log_hyps[0] if log_hyps else ""
            logging.info(
                "[%s] batch %d infer sample\n  ref: `%s`\n  hyp: `%s`\n  wer(batch)=%.4f",
                name,
                batch_idx,
                ref_text,
                hyp_text,
                log_wer,
            )

    # ------------------------------------------------------------------
    # Test (delegates to validation logic)
    # ------------------------------------------------------------------

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    # ------------------------------------------------------------------
    # Backward + OOMptimizer
    # ------------------------------------------------------------------

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    @property
    def oomptimizer_schema(self) -> dict:
        from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType

        return {
            "cls": StreamingSTTBatch,
            "inputs": [
                {
                    "name": "input_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": int(self.text_vocab_size),
                },
                {"name": "input_token_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "output"},
                {
                    "name": "target_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": int(self.text_vocab_size),
                },
                {"name": "target_token_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "output"},
                {"name": "audios", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
            ],
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _ensure_inference_cache(self) -> None:
        """Lazily cache token templates and IDs needed for inference.

        Uses ``apply_chat_template(tokenize=False)`` on a 4-message dummy
        conversation and splits the text around a sentinel to isolate
        user-header, user-footer + assistant-header, and assistant-footer tokens.

        The 4-message pattern (two user+assistant pairs) ensures the *first*
        assistant turn is not the last — this prevents Qwen3-style chat
        templates from injecting ``<think>``/``</think>`` tags, which only
        appear on the final assistant turn.
        """
        if hasattr(self, '_inference_cache_ready'):
            return

        hf_tok = self.tokenizer.tokenizer
        chunk_size = self.core_cfg.chunk_size

        # --- Build turn template ---
        if self.core_cfg.compact_template:
            user_header_ids, user_footer_and_asst_header_ids, asst_footer_ids = build_compact_turn_markers(
                hf_tok, self.core_cfg.write_token
            )
            logging.info(
                f"compact_template: user_header={user_header_ids}, "
                f"write+mid={user_footer_and_asst_header_ids}, footer={asst_footer_ids}"
            )
        else:
            user_header_ids, user_footer_and_asst_header_ids, asst_footer_ids = parse_chat_template_ids(
                hf_tok, last_turn=(chunk_size < 0)
            )
        self._user_header_ids = user_header_ids
        self._user_footer_and_asst_header_ids = user_footer_and_asst_header_ids
        self._asst_footer_ids = asst_footer_ids

        # Always cache user_footer_first_id — needed by state machine inference
        # for both dynamic (chunk_size=0) and fixed chunking (use_state_machine_inference).
        self._user_footer_first_id = user_footer_and_asst_header_ids[0] if user_footer_and_asst_header_ids else None

        if chunk_size > 0:
            # ``chunk_size`` is in encoder frames; the LLM consumes
            # ``ceil(chunk_size / factor)`` audio tokens per chunk when the
            # trainable subsampler is enabled (factor==1 -> unchanged).
            n_audio_slots = self._n_audio_slots(chunk_size)
            turn_ids = user_header_ids + [AUDIO_TOKEN_IDX] * n_audio_slots + user_footer_and_asst_header_ids
            self._turn_template_ids = turn_ids
            n_audio = turn_ids.count(AUDIO_TOKEN_IDX)
            logging.info(
                f"Streaming turn template ({len(turn_ids)} tokens, "
                f"{n_audio} audio slots, chunk_size={chunk_size}): {turn_ids}"
            )
        elif chunk_size == 0:
            # Dynamic chunking: no fixed turn template. Audio frames are fed
            # incrementally; the user header/footer are appended on demand.
            self._turn_template_ids = None
            logging.info(
                f"Dynamic chunking mode: user_footer_first_id={self._user_footer_first_id}, "
                f"user_header_ids={user_header_ids}"
            )
        else:
            self._turn_template_ids = None
            logging.info(f"Offline mode (chunk_size={chunk_size}): no fixed turn template")

        self._eos_id = getattr(hf_tok, 'eos_token_id', None)

        # Flush inference: feed a <flush> control turn after the audio chunks and
        # decode the residual tail until <|im_end|>. Baseline fixed-chunk scheme
        # only — exotic position schemes / parallel heads fall back to no-flush
        # (the trailing-silence pad in run_eval still helps those).
        self._inference_flush_id = None
        self._inference_use_flush = False
        if bool(getattr(self.core_cfg, "use_flush", False)) and chunk_size > 0:
            fid = hf_tok.convert_tokens_to_ids(self.core_cfg.flush_token)
            unsupported = (
                self.core_cfg.use_modality_position_ids
                or self.core_cfg.use_chunk_local_audio_attn
                or self.core_cfg.use_two_stream_attn
                or (self.parallel_chunk_heads is not None)
            )
            if fid is None or int(fid) < 0:
                logging.warning("use_flush set but flush_token not in vocab; flush disabled at inference")
            elif unsupported:
                logging.warning(
                    "use_flush inference is only supported for the baseline scheme "
                    "(no modality-position-ids / chunk-local-attn / parallel heads); flush disabled"
                )
            else:
                self._inference_flush_id = int(fid)
                self._inference_use_flush = True
                logging.info(f"Inference flush enabled: flush_id={self._inference_flush_id}")

        # When eos_token_id coincides with a token in the footer (e.g. Qwen3
        # where eos = <|im_end|> = footer[0]), detecting EOS acts as an
        # early-stop shortcut that avoids generating the remaining footer
        # tokens.  When eos_token_id is NOT in the footer it serves as a
        # safety-net stop only.
        self._eos_in_footer = self._eos_id is not None and self._eos_id in self._asst_footer_ids
        logging.info(
            f"Assistant footer IDs: {self._asst_footer_ids}, "
            f"blank ID: {self.blank_token_id}, EOS ID: {self._eos_id}, "
            f"EOS in footer: {self._eos_in_footer}"
        )
        self._inference_cache_ready = True

    def _sample_token(
        self,
        logits: Tensor,
        generated_ids: list[list[int]] | list[int] | None = None,
        generation_config: Optional[GenerationConfig] = None,
        **generation_kwargs,
    ) -> Tensor:
        """Select the next token from logits.

        Applies the following transforms in order (each is skipped when the
        corresponding parameter is at its default/off value):

        1. **Suppress tokens** — force listed token IDs to ``-inf``.
        2. **No-repeat-ngram** — block n-grams that already appear in
           *generated_ids*.
        3. **Repetition penalty** — scale logits for tokens that already appear
           in *generated_ids*.
        4. **Temperature** — divide logits by temperature.
        5. **Top-k** — keep only the *k* highest-scoring tokens.
        6. **Top-p (nucleus)** — keep the smallest set of tokens whose
           cumulative probability is ≥ *top_p*.
        7. If ``do_sample`` is ``True``, sample from the filtered distribution;
           otherwise return the argmax.

        Parameters are read from *generation_kwargs* first, falling back to
        *generation_config*, then to HuggingFace defaults.

        Args:
            logits: ``(B, vocab_size)`` logits for the last position.
            generated_ids: Token IDs generated so far.  For B=1, a flat list.
                For B>1, a list of B lists (one per stream).  Used for
                repetition-aware transforms.  May be ``None`` or empty.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            generation_kwargs: Per-call overrides.

        Returns:
            ``(B,)`` tensor with the selected token IDs.
        """
        # Fast path: no config → greedy
        if generation_config is None and not generation_kwargs:
            return logits.argmax(dim=-1)

        cfg = generation_config or GenerationConfig()
        do_sample = generation_kwargs.get('do_sample', cfg.do_sample)
        temperature = generation_kwargs.get('temperature', cfg.temperature)
        top_k = generation_kwargs.get('top_k', cfg.top_k)
        top_p = generation_kwargs.get('top_p', cfg.top_p)
        repetition_penalty = generation_kwargs.get('repetition_penalty', cfg.repetition_penalty)
        no_repeat_ngram_size = generation_kwargs.get('no_repeat_ngram_size', cfg.no_repeat_ngram_size)
        suppress_tokens = generation_kwargs.get('suppress_tokens', cfg.suppress_tokens)

        # --- logit manipulation (order matters) ---

        # 1. Suppress tokens
        if suppress_tokens:
            logits[..., suppress_tokens] = float('-inf')

        histories = None
        if generated_ids:
            if isinstance(generated_ids[0], list):
                histories = generated_ids
            else:
                histories = [generated_ids for _ in range(logits.shape[0])]

        # 2. No-repeat-ngram blocking
        if no_repeat_ngram_size > 0 and histories:
            for b, history in enumerate(histories[: logits.shape[0]]):
                if not history:
                    continue
                if no_repeat_ngram_size == 1:
                    blocked = set(history)
                elif len(history) >= no_repeat_ngram_size - 1:
                    ngram_prefix = history[-(no_repeat_ngram_size - 1) :]
                    blocked = {
                        history[i + no_repeat_ngram_size - 1]
                        for i in range(len(history) - no_repeat_ngram_size + 1)
                        if history[i : i + no_repeat_ngram_size - 1] == ngram_prefix
                    }
                else:
                    blocked = set()
                if blocked:
                    logits[b, list(blocked)] = float('-inf')

        # 3. Repetition penalty
        if repetition_penalty != 1.0 and histories:
            for b, history in enumerate(histories[: logits.shape[0]]):
                if not history:
                    continue
                prev_token_ids = torch.tensor(list(set(history)), device=logits.device)
                scores = logits[b, prev_token_ids]
                # Penalize: divide positive scores, multiply negative scores
                logits[b, prev_token_ids] = torch.where(
                    scores > 0, scores / repetition_penalty, scores * repetition_penalty
                )

        # Greedy fast path (no sampling-related transforms needed)
        if not do_sample:
            return logits.argmax(dim=-1)

        # 4. Temperature scaling
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        # 5. Top-k filtering
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            kth_val = torch.topk(logits, k, dim=-1)[0][..., -1:]
            logits = logits.masked_fill(logits < kth_val, float('-inf'))

        # 6. Top-p (nucleus) filtering
        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Mark tokens whose cumulative probability (excluding themselves) >= top_p
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            indices_to_remove = sorted_mask.scatter(dim=-1, index=sorted_indices, src=sorted_mask)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # 7. Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def _parallel_chunk_step_decode(
        self,
        anchor_hidden: Tensor,
        state: 'StreamingState',
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_PER_CHUNK,
    ) -> tuple[list[list[int]], list[bool]]:
        """Parallel-head per-chunk decoding with iterative K-token blocks.

        ``K = parallel_chunk_heads.num_slots`` is the number of tokens emitted
        per LLM forward (one *block*), decoupled from how many tokens a chunk
        may contain. Each block:

          1. runs the parallel heads on the current anchor hidden → K
             next-token predictions,
          2. truncates each stream at its stop signal (see schemes below),
          3. feeds the emitted real tokens into the KV cache, and
          4. re-anchors each still-open stream on the hidden state of its last
             fed token (which, under next-token semantics, predicts the first
             token of that stream's next block).

        Two stop schemes are supported:

          * **Legacy fixed-K** (``parallel_blank_continuation=False``): both
            <|im_end|> and <blank> are chunk stops; the first one seen in a
            block ends the chunk. Budget = ``ceil(max_new_tokens / K)`` blocks.
          * **Blank-continuation** (``parallel_blank_continuation=True``): only
            <|im_end|> ends the chunk (slot 0 in the clean case). <blank> is a
            "cut & continue" marker — emit the real tokens before it, do NOT
            feed the blank, re-anchor, and continue. A full K-token block also
            continues. Budget = ``max_new_tokens + 1`` blocks (a block may emit
            as few as one real token).

        A chunk iterates blocks until a stream hits <|im_end|> (or its
        per-chunk real-token budget / block cap is exhausted, at which point a
        closing <|im_end|> is synthesized so the cache ends cleanly).

        Args:
            anchor_hidden: (B, H) hidden state at each stream's first-block
                anchor (the write_id / asst-header token of the chunk prefill).
            state: Mutable StreamingState (KV cache, attention_mask, seq_lens).
            max_new_tokens: Per-chunk cap on emitted real tokens; sets the max
                number of K-blocks to ``ceil(max_new_tokens / K)``.

        Returns:
            (generated_per_stream, footer_consumed_per_stream). Same contract as
            the single-block version: ``generated`` never includes the closer;
            ``footer_consumed[b]`` is True when this routine already put the
            chunk's closer into the cache (so the caller's footer step is a
            no-op for that stream).
        """
        assert self.parallel_chunk_heads is not None, (
            "_parallel_chunk_step_decode called but parallel_chunk_heads is not initialized"
        )
        # Chunk-local audio attn / two-stream attn / sep-pos variants are not yet
        # wired through the parallel decode path; error early instead of silently
        # mis-decoding.
        if (
            self.core_cfg.use_chunk_local_audio_attn
            or self.core_cfg.use_two_stream_attn
            or self.core_cfg.use_modality_position_ids
        ):
            raise NotImplementedError(
                "parallel_chunk_decode is not yet wired through chunk-local / two-stream audio "
                "attention or separated modality position IDs. Set parallel_chunk_decode=False "
                "for those modes."
            )
        device = anchor_hidden.device
        B = anchor_hidden.shape[0]
        K = self.parallel_chunk_heads.num_slots
        pad_id = self.text_pad_id

        # Decode scheme. In the legacy fixed-K scheme both <|im_end|> and <blank>
        # are chunk stops. In the blank-continuation scheme only <|im_end|> ends
        # the chunk; <blank> is a "cut & continue" marker (emit the real tokens
        # before it, do NOT feed the blank into the cache, re-anchor, continue).
        blank_continuation = bool(self.core_cfg.parallel_blank_continuation)
        blank_id = int(self.blank_token_id) if self.has_blank else None

        # Legacy stop tokens: <|im_end|> always; blank too if configured.
        stop_ids = set()
        if self._eos_id is not None:
            stop_ids.add(int(self._eos_id))
        if self.has_blank:
            stop_ids.add(int(self.blank_token_id))

        if blank_continuation:
            # A block may emit as few as 1 real token (a cut at slot 1), so the
            # loop is budgeted by real-token count (+1 for the terminator).
            max_blocks = max(1, int(max_new_tokens) + 1)
        else:
            max_blocks = max(1, (int(max_new_tokens) + K - 1) // K)  # ceil(max_new_tokens / K)

        generated_per_stream: list[list[int]] = [[] for _ in range(B)]
        footer_consumed: list[bool] = [False] * B
        finished: list[bool] = [False] * B
        cur_anchor = anchor_hidden  # (B, H)

        for block_idx in range(max_blocks):
            is_last_block = block_idx == max_blocks - 1

            # (1) K next-token predictions per stream from the parallel heads.
            par_logits = self.parallel_chunk_heads(cur_anchor, self.llm.lm_head)  # (B, K, V)
            sampled = par_logits.argmax(dim=-1)  # (B, K)

            # (2) Per-stream decode of this block.
            per_stream_feed_ids: list[list[int]] = [[] for _ in range(B)]
            for b in range(B):
                if finished[b]:
                    continue
                if blank_continuation:
                    # <|im_end|> ends the chunk (slot 0 in the clean case;
                    # treated as a stop at any slot defensively). <blank> cuts:
                    # stop scanning, keep the chunk open, and re-anchor on the
                    # last real token. A full K-token block (no blank/eos)
                    # likewise continues.
                    stopped = False
                    for k in range(K):
                        tid = int(sampled[b, k].item())
                        if self._eos_id is not None and tid == int(self._eos_id):
                            per_stream_feed_ids[b].append(tid)
                            footer_consumed[b] = True
                            finished[b] = True
                            stopped = True
                            break
                        if blank_id is not None and tid == blank_id:
                            break  # cut: continue to next block (blank not fed)
                        generated_per_stream[b].append(tid)
                        per_stream_feed_ids[b].append(tid)
                        if len(generated_per_stream[b]) >= int(max_new_tokens):
                            break  # budget reached mid-block
                    if not stopped:
                        budget_done = len(generated_per_stream[b]) >= int(max_new_tokens)
                        no_progress = len(per_stream_feed_ids[b]) == 0  # degenerate blank@slot0
                        if budget_done or is_last_block or no_progress:
                            # Close cleanly: synthesize the <|im_end|> so the KV
                            # cache ends correctly and the caller's footer step
                            # is a no-op for this stream.
                            finished[b] = True
                            if self._eos_id is not None:
                                per_stream_feed_ids[b].append(int(self._eos_id))
                                footer_consumed[b] = True
                    continue

                # ---- legacy fixed-K scheme: stop on <|im_end|> OR <blank> ----
                hit_stop = False
                for k in range(K):
                    tid = int(sampled[b, k].item())
                    if tid in stop_ids:
                        footer_consumed[b] = True
                        finished[b] = True
                        per_stream_feed_ids[b].append(tid)
                        hit_stop = True
                        break
                    generated_per_stream[b].append(tid)
                    per_stream_feed_ids[b].append(tid)
                if not hit_stop and is_last_block:
                    # Out of block budget without a natural stop — synthesize
                    # the closer so the KV cache ends cleanly and the caller's
                    # footer step is a no-op for this stream.
                    finished[b] = True
                    if self._eos_id is not None:
                        per_stream_feed_ids[b].append(int(self._eos_id))
                        footer_consumed[b] = True

            # (3) Batched feed of this block's emitted tokens into the KV cache.
            # Pad to the max per-stream length; padded positions are masked out.
            max_len = max((len(seq) for seq in per_stream_feed_ids), default=0)
            if max_len == 0:
                break  # all streams already finished — nothing left to feed

            feed_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
            feed_attn_add = torch.zeros((B, max_len), dtype=state.attention_mask.dtype, device=device)
            last_real_idx = [-1] * B
            for b in range(B):
                n_b = len(per_stream_feed_ids[b])
                if n_b == 0:
                    continue
                feed_ids[b, :n_b] = torch.tensor(per_stream_feed_ids[b], dtype=torch.long, device=device)
                feed_attn_add[b, :n_b] = 1
                last_real_idx[b] = n_b - 1
            feed_embeds = self.embed_tokens(feed_ids)  # (B, max_len, H)

            # Only need last-layer hidden states if a stream will continue to
            # another block (for re-anchoring).
            need_hidden = not all(finished)
            state.attention_mask = torch.cat([state.attention_mask, feed_attn_add], dim=1)
            out = self.llm(
                inputs_embeds=feed_embeds,
                past_key_values=state.cache,
                attention_mask=state.attention_mask,
                use_cache=True,
                return_dict=True,
                output_hidden_states=need_hidden,
            )
            state.cache = out.past_key_values
            for b in range(B):
                # Only real-token positions grow the logical sequence; padded
                # positions are masked out and shouldn't count as "seen text".
                state.seq_lens[b] += len(per_stream_feed_ids[b])

            if all(finished):
                break

            # (4) Re-anchor each still-open stream on the hidden state of its
            # last fed token (predicts the first token of its next block).
            last_hidden = out.hidden_states[-1]  # (B, max_len, H)
            next_anchor = cur_anchor.clone()
            for b in range(B):
                if not finished[b] and last_real_idx[b] >= 0:
                    next_anchor[b] = last_hidden[b, last_real_idx[b], :]
            cur_anchor = next_anchor

        return generated_per_stream, footer_consumed

    def _autoregressive_decode(
        self,
        logits: Tensor,
        cache: tuple,
        state: Optional['StreamingState'],
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        stop_on_blank: Union[bool, str] = True,
        **generation_kwargs,
    ) -> tuple[list[list[int]], tuple, list[bool], int, list[str], list[Optional[int]]]:
        """Autoregressive decoding (supports B streams).

        Token selection is delegated to :meth:`_sample_token`, which supports
        greedy (default), sampling (temperature / top-k / top-p), repetition
        penalty, no-repeat-ngram blocking, and token suppression.

        Generation stops per stream when any of these conditions is met:

        1. **EOS** — the tokenizer's ``eos_token_id`` is predicted.
        2. **Blank** — the ``<blank>`` token is predicted (controlled by
           ``stop_on_blank``).
        3. **Footer sequence** — the last *N* tokens match ``self._asst_footer_ids``.
        4. **Max tokens** — ``max_new_tokens`` is reached.

        Args:
            logits: ``(B, L, V)`` logits from the LLM forward pass.
            cache: HF ``past_key_values`` with batch dim B.
            max_new_tokens: Maximum tokens to generate per stream.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            stop_on_blank: Controls blank-token stopping behavior:
                - ``True`` (default): stop whenever blank is predicted.
                  Use with dedicated ``<blank>`` special tokens.
                - ``"first"``: stop only if blank is the **first** token
                  generated (= "no speech this chunk").  Use when the blank
                  token is a natural text token (e.g. ``" "``).
                - ``False``: never stop on blank.
            generation_kwargs: Per-call overrides.

        Returns:
            ``(generated_per_stream, updated_cache, footer_consumed_per_stream, num_feed_steps, stop_reasons, stop_ids)``
            where ``generated_per_stream`` is a list of B token-ID lists and
            ``num_feed_steps`` is how many tokens were fed to the LLM cache.
        """
        B = logits.shape[0]
        footer = self._asst_footer_ids
        flen = len(footer)
        generated: list[list[int]] = [[] for _ in range(B)]
        footer_consumed = [False] * B
        finished = [False] * B
        stop_reasons = ["max_tokens"] * B
        stop_ids: list[Optional[int]] = [None] * B
        num_feed_steps = 0
        # Track which streams need their token fed to the LLM this step.
        # EOS tokens must NOT be fed; blank/footer/normal tokens must be fed.
        feed_mask = [False] * B
        text_lm_feed_mask = [False] * B
        text_lm_control_ids = self._control_token_ids_for_text_lm_score() if self._text_only_lm_score_active() else set()
        use_generation_history = bool(generation_kwargs.pop("use_generation_history", False))

        next_logits = self._apply_text_only_lm_score(logits[:, -1, :], state)
        next_tokens = self._sample_token(next_logits, None, generation_config, **generation_kwargs)  # (B,)

        for _ in range(max_new_tokens):
            for b in range(B):
                feed_mask[b] = False
                text_lm_feed_mask[b] = False
                if finished[b]:
                    continue
                tid = next_tokens[b].item()

                # EOS: stop WITHOUT feeding to LLM. Append a chunk separator so
                # decode_with_blank can join per-chunk outputs correctly.
                if self._eos_id is not None and tid == self._eos_id:
                    finished[b] = True
                    stop_reasons[b] = "eos"
                    stop_ids[b] = tid
                    # Blank token when enabled, else EOS id itself (matches decode_with_blank).
                    generated[b].append(self.blank_token_id if self.has_blank else self._eos_id)
                    continue

                # All other tokens get appended and fed to LLM
                generated[b].append(tid)
                feed_mask[b] = True
                text_lm_feed_mask[b] = tid not in text_lm_control_ids

                # Blank: stop (token IS fed to LLM, IS in generated).
                # When stop_on_blank == "first", only stop if blank is the
                # first generated token (= "no speech this chunk").  This
                # avoids false stops when the blank token collides with a
                # natural text token (e.g. " ") that appears mid-sentence.
                if tid == self.blank_token_id:
                    text_lm_feed_mask[b] = False
                    if stop_on_blank is True or (stop_on_blank == "first" and len(generated[b]) == 1):
                        finished[b] = True
                        stop_reasons[b] = "blank"
                        stop_ids[b] = tid

                # Footer sequence match
                elif flen > 0 and len(generated[b]) >= flen and generated[b][-flen:] == footer:
                    generated[b] = generated[b][:-flen]
                    text_lm_feed_mask[b] = False
                    footer_consumed[b] = True
                    finished[b] = True
                    stop_reasons[b] = "footer"
                    stop_ids[b] = footer[0] if footer else tid

            # If no stream needs feeding, we're done
            if not any(feed_mask):
                break

            self._update_text_only_lm_state(next_tokens, text_lm_feed_mask, state)

            # Feed tokens to LLM. For finished streams, feed the blank token
            # (which the model was trained on) instead of a pad token, so the
            # KV cache stays clean — no foreign tokens that corrupt attention.
            # When blank is disabled, feed text_pad_id as a fallback.
            filler_id = self.blank_token_id if self.has_blank else self.text_pad_id
            tokens_to_feed = next_tokens.clone()
            for b in range(B):
                if not feed_mask[b]:
                    tokens_to_feed[b] = filler_id

            # All tokens are "real" (blank is a valid token), so all seq_lens grow
            if state is not None:
                for b in range(B):
                    state.seq_lens[b] += 1

            token_emb = self.embed_tokens(tokens_to_feed.unsqueeze(1))  # (B, 1, H)
            position_ids = None
            attn_bias_4d = None
            new_chunk_ids_for_history: Optional[Tensor] = None
            new_is_audio_for_history: Optional[Tensor] = None
            if self.core_cfg.use_modality_position_ids:
                position_ids = self._build_next_text_position_ids(
                    B, state, token_emb.device, active_mask=feed_mask
                )
            elif self._uses_interleaved_stream_state:
                new_is_audio = torch.zeros(
                    (B, 1), dtype=torch.bool, device=token_emb.device
                )
                active_t = torch.as_tensor(
                    feed_mask, dtype=torch.long, device=token_emb.device
                )
                (
                    position_ids,
                    attn_bias_4d,
                    new_chunk_ids_for_history,
                    new_is_audio_for_history,
                ) = self._build_chunk_local_step_inputs(state, new_is_audio, active_mask=active_t)

            # Only streams that are actually fed a real token this step (active
            # streams, ``feed_mask=True``) get an attendable KV slot. Finished
            # streams are fed a filler ``<blank>`` purely to keep the batched
            # cache rectangular; marking those slots attendable (``ones``) lets
            # them accumulate extra blank tokens at frozen/duplicated positions,
            # which corrupts that stream's context in *later* turns and silently
            # truncates long utterances in batched decode (the single-stream
            # path never sees them because the loop breaks on finish). Mask them
            # out so batched decode matches batch=1 exactly.
            new_attn_col = torch.tensor(
                feed_mask, dtype=state.attention_mask.dtype, device=state.attention_mask.device
            ).unsqueeze(1)  # (B, 1): 1 for active/fed streams, 0 for finished streams' filler
            state.attention_mask = torch.cat(
                [state.attention_mask, new_attn_col],
                dim=1,
            )
            llm_attn = attn_bias_4d if attn_bias_4d is not None else state.attention_mask
            out = self.llm(
                inputs_embeds=token_emb,
                past_key_values=cache,
                attention_mask=llm_attn,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
            if new_chunk_ids_for_history is not None:
                self._append_chunk_local_history(
                    state, new_chunk_ids_for_history, new_is_audio_for_history
                )
            cache = out.past_key_values
            num_feed_steps += 1

            if all(finished):
                break

            next_logits = self._apply_text_only_lm_score(out.logits[:, -1, :], state)
            next_tokens = self._sample_token(
                next_logits,
                generated if use_generation_history else None,
                generation_config,
                **generation_kwargs,
            )

        for b in range(B):
            if stop_ids[b] is None and generated[b]:
                stop_ids[b] = generated[b][-1]
        return generated, cache, footer_consumed, num_feed_steps, stop_reasons, stop_ids

    def get_audio_feature_buffer(
        self,
        batch_size: int,
        chunk_size_override: Optional[int] = None,
    ) -> BatchedCacheFeatureBufferer:
        """Get the audio feature buffer for the streaming state.

        Args:
            batch_size: Number of parallel streams.
            chunk_size_override: If provided, use this chunk size (in frames)
                instead of ``self.core_cfg.chunk_size``.  Used by dynamic
                chunking inference where the inference step size differs
                from the config chunk_size.
        """
        preprocessor_cfg: DictConfig = self.perception.cfg.preprocessor
        window_stride_in_secs = preprocessor_cfg.window_stride
        pre_encode_cache_size = self.perception.encoder.streaming_cfg.pre_encode_cache_size
        if isinstance(pre_encode_cache_size, list):
            pre_encode_cache_size = pre_encode_cache_size[1]
        pre_encode_cache_size_in_secs = pre_encode_cache_size * window_stride_in_secs
        cs = chunk_size_override if chunk_size_override is not None else max(self.core_cfg.chunk_size, 1)
        chunk_size_in_secs = cs * self.core_cfg.frame_length_in_secs
        buffer_size_in_secs = pre_encode_cache_size_in_secs + chunk_size_in_secs

        audio_feature_buffer = BatchedCacheFeatureBufferer(
            num_slots=batch_size,
            sample_rate=self.core_cfg.sample_rate,
            buffer_size_in_secs=buffer_size_in_secs,
            chunk_size_in_secs=buffer_size_in_secs,  # recalculate mel-spec for the whole buffer
            preprocessor_cfg=preprocessor_cfg,
            device=self.device,
        )
        return audio_feature_buffer

    def get_init_streaming_state(
        self,
        system_prompt: Union[str, List[str]],
        device: torch.device,
        batch_size: int = 1,
    ) -> StreamingState:
        """Forward the system prompt through the LLM and return a fresh :class:`StreamingState`.

        Args:
            system_prompt: System prompt string (shared) or list of B per-sample prompts.
            device: Target device.
            batch_size: Number of parallel streams (B).
        """
        hf_tok = self.tokenizer.tokenizer
        dtype = self.embed_tokens.weight.dtype

        if isinstance(system_prompt, str):
            prompts = [system_prompt] * batch_size
        else:
            prompts = system_prompt

        # Tokenize each prompt
        all_sys_ids = []
        for prompt in prompts:
            ids = hf_tok.apply_chat_template(
                [{"role": "system", "content": prompt}],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            all_sys_ids.append(ids)

        # Check if all prompts are the same length (common case: same prompt)
        sys_lens = [len(ids) for ids in all_sys_ids]
        needs_padding = len(set(sys_lens)) > 1

        # Capture hidden states from the prefill if the aux head will be used at
        # inference, so the aux backbone sees the same full-sequence context as
        # at training time.
        capture_hidden = self.core_cfg.use_chunk_classifier and self.core_cfg.chunk_classifier_use_at_inference

        if not needs_padding:
            # Fast path: all same length, no padding needed
            sys_embs = self.embed_tokens(
                torch.tensor(all_sys_ids[0], device=device, dtype=torch.long).unsqueeze(0)
            ).expand(batch_size, -1, -1)
            attention_mask = torch.ones(batch_size, sys_lens[0], dtype=torch.long, device=device)
            position_ids = None
            if self.core_cfg.use_modality_position_ids:
                position_ids = (
                    torch.arange(
                        int(self.core_cfg.modality_position_offset),
                        int(self.core_cfg.modality_position_offset) + sys_lens[0],
                        dtype=torch.long,
                        device=device,
                    )
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
            elif self._uses_interleaved_stream_state:
                # System prompt occupies text positions ``[0, sys_len-1]``; audio
                # counter is still 0 (no audio yet).
                position_ids = (
                    torch.arange(0, sys_lens[0], dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
            out = self.llm(
                inputs_embeds=sys_embs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                output_hidden_states=capture_hidden,
                return_dict=True,
            )
            max_sys_len = sys_lens[0]
        else:
            # Per-sample prompts with different lengths: left-pad and use attention mask
            max_sys_len = max(sys_lens)
            H = self.embed_tokens.weight.shape[-1]
            sys_embs = torch.zeros(batch_size, max_sys_len, H, device=device, dtype=dtype)
            attention_mask = torch.zeros(batch_size, max_sys_len, dtype=torch.long, device=device)
            position_ids = None
            if self.core_cfg.use_modality_position_ids:
                position_ids = torch.zeros(batch_size, max_sys_len, dtype=torch.long, device=device)
            elif self._uses_interleaved_stream_state:
                position_ids = torch.zeros(batch_size, max_sys_len, dtype=torch.long, device=device)
            for b in range(batch_size):
                embs = self.embed_tokens(
                    torch.tensor(all_sys_ids[b], device=device, dtype=torch.long).unsqueeze(0)
                ).squeeze(
                    0
                )  # (L_b, H)
                offset = max_sys_len - sys_lens[b]
                sys_embs[b, offset:] = embs
                attention_mask[b, offset:] = 1
                if position_ids is not None:
                    if self.core_cfg.use_modality_position_ids:
                        position_ids[b, offset:] = torch.arange(
                            int(self.core_cfg.modality_position_offset),
                            int(self.core_cfg.modality_position_offset) + sys_lens[b],
                            dtype=torch.long,
                            device=device,
                        )
                    else:  # chunk-local / two-stream: text counter from 0
                        position_ids[b, offset:] = torch.arange(
                            0, sys_lens[b], dtype=torch.long, device=device
                        )
            out = self.llm(
                inputs_embeds=sys_embs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                output_hidden_states=capture_hidden,
                return_dict=True,
            )

        aux_hidden_buffer = out.hidden_states[-1] if capture_hidden else None

        cache_last_channel, cache_last_time, cache_last_channel_len = self.perception.get_initial_cache_state(
            batch_size=batch_size, dtype=dtype, device=device
        )
        audio_feature_buffer = self.get_audio_feature_buffer(batch_size=batch_size)
        audio_cache = CacheAwareContext(
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
        if self.core_cfg.use_modality_position_ids:
            audio_pos_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            text_pos_counter = torch.tensor(sys_lens, dtype=torch.long, device=device) + int(
                self.core_cfg.modality_position_offset
            )
            chunk_id_counter_init = None
            chunk_id_history_init = None
            is_audio_history_init = None
        elif self._uses_interleaved_stream_state:
            audio_pos_counter = torch.zeros(batch_size, dtype=torch.long, device=device)
            text_pos_counter = torch.tensor(sys_lens, dtype=torch.long, device=device)
            chunk_id_counter_init = torch.full(
                (batch_size,), -1, dtype=torch.long, device=device
            )
            # ``attention_mask`` is (B, max_sys_len) with left-padding when prompts
            # have unequal lengths; assign the SAME shape for chunk_id_history /
            # is_audio_history so they stay aligned token-by-token with the KV cache.
            chunk_id_history_init = torch.full(
                (batch_size, max_sys_len), -1, dtype=torch.long, device=device
            )
            is_audio_history_init = torch.zeros(
                batch_size, max_sys_len, dtype=torch.bool, device=device
            )
        else:
            audio_pos_counter = None
            text_pos_counter = None
            chunk_id_counter_init = None
            chunk_id_history_init = None
            is_audio_history_init = None

        return StreamingState(
            cache=out.past_key_values,
            generated_tokens=[[] for _ in range(batch_size)],
            seq_lens=[max_sys_len] * batch_size,
            audio_cache=audio_cache,
            audio_feature_buffer=audio_feature_buffer,
            attention_mask=attention_mask,
            aux_hidden_buffer=aux_hidden_buffer,
            audio_position_counter=audio_pos_counter,
            text_position_counter=text_pos_counter,
            chunk_id_counter=chunk_id_counter_init,
            chunk_id_history=chunk_id_history_init,
            is_audio_history=is_audio_history_init,
            batch_size=batch_size,
        )

    def _encode_streaming_audio_chunk(
        self,
        audio_chunks: Tensor,
        audio_chunk_lens: Optional[Tensor],
        state: StreamingState,
        chunk_size_frames: int,
    ) -> Tensor:
        """Run one streaming-encoder update and return the chunk's audio embeddings.

        Returns ``chunk_size_frames`` embeddings, or ``chunk_size_frames //
        encoder_subsampling_factor`` when the trainable subsampler is enabled.
        """
        device = audio_chunks.device
        B = state.batch_size
        if audio_chunk_lens is None:
            audio_chunk_lens = torch.tensor([audio_chunks.shape[-1]] * B, device=device)
        frames = [
            Frame(
                samples=audio_chunks[b] if audio_chunks.dim() == 2 else audio_chunks,
                length=int(audio_chunk_lens[b].item()),
                stream_id=b,
            )
            for b in range(B)
        ]
        features, right_paddings = state.audio_feature_buffer.update(frames)
        processed_signal = torch.stack(features).type_as(self.embed_tokens.weight)
        processed_signal_length = torch.tensor(
            [processed_signal.shape[-1] - int(rp) for rp in right_paddings],
            device=device,
        ).long()

        outputs = self.perception(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=state.audio_cache.cache_last_channel,
            cache_last_time=state.audio_cache.cache_last_time,
            cache_last_channel_len=state.audio_cache.cache_last_channel_len,
            streaming=True,
        )
        audio_chunk_embs, _, new_perception_cache = outputs

        if new_perception_cache is not None:
            state.audio_cache.cache_last_channel = new_perception_cache['cache_last_channel']
            state.audio_cache.cache_last_time = new_perception_cache['cache_last_time']
            state.audio_cache.cache_last_channel_len = new_perception_cache['cache_last_channel_len']

        n_frames = audio_chunk_embs.shape[1]
        if n_frames < chunk_size_frames:
            audio_chunk_embs = F.pad(audio_chunk_embs, (0, 0, 0, chunk_size_frames - n_frames))
        elif n_frames > chunk_size_frames:
            audio_chunk_embs = audio_chunk_embs[:, :chunk_size_frames, :]
        # Trainable subsampling: (B, chunk_size_frames, H) -> (B, chunk_size_frames//S, H).
        audio_chunk_embs, _ = self._apply_audio_subsampling(audio_chunk_embs)
        return audio_chunk_embs

    @torch.no_grad()
    def _chunked_streaming_step(
        self,
        audio_chunks: Tensor,
        audio_chunk_lens: Optional[Tensor] = None,
        state: Optional[StreamingState] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_PER_CHUNK,
        generation_config: Optional[GenerationConfig] = None,
        _audio_embs: Optional[Tensor] = None,
        inference_chunk_size_frames: Optional[int] = None,
        parallel_chunk_decode: Optional[bool] = None,
        **generation_kwargs,
    ) -> list[list[int]]:
        """
        Process B raw audio chunks and generate the assistant responses.

        Args:
            audio_chunks: ``(B, T_samples)`` raw waveforms for one chunk per stream.
            audio_chunk_lens: ``(B,)`` number of valid samples per stream.
            state: Mutable :class:`StreamingState` with ``batch_size=B`` (updated in place).
            max_new_tokens: Maximum tokens to generate per chunk per stream.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            _audio_embs: Optional pre-computed audio embeddings ``(B, chunk_size, H)``.
                Diagnostic use only.
            inference_chunk_size_frames: Number of encoder frames represented
                by this inference turn. Defaults to ``self.core_cfg.chunk_size``.
            generation_kwargs: Per-call overrides for generation parameters.
        Returns:
            List of B token-ID lists (one per stream).
        """

        self._ensure_inference_cache()
        device = audio_chunks.device
        B = state.batch_size

        if _audio_embs is not None:
            audio_chunk_embs = _audio_embs.type_as(self.embed_tokens.weight)
        else:
            audio_chunk_embs = self._encode_streaming_audio_chunk(
                audio_chunks,
                audio_chunk_lens,
                state,
                chunk_size_frames=int(inference_chunk_size_frames or self.core_cfg.chunk_size),
            )

        # 3. Pad/trim to the number of audio slots in this inference turn.
        # ``chunk_size`` counts ENCODER frames; the LLM sees ``n_audio_slots =
        # chunk_size // factor`` audio tokens (the subsampler already shrank the
        # embeddings above).
        chunk_size = int(inference_chunk_size_frames or self.core_cfg.chunk_size)
        n_audio_slots = self._n_audio_slots(chunk_size)
        n_frames = audio_chunk_embs.shape[1]
        if n_frames < n_audio_slots:
            audio_chunk_embs = F.pad(audio_chunk_embs, (0, 0, 0, n_audio_slots - n_frames))
        elif n_frames > n_audio_slots:
            audio_chunk_embs = audio_chunk_embs[:, :n_audio_slots, :]

        # 4. Build input embeddings from cached turn template — (B, L, H)
        if chunk_size == self.core_cfg.chunk_size:
            turn_template_ids = self._turn_template_ids
        else:
            turn_template_ids = (
                self._user_header_ids
                + [AUDIO_TOKEN_IDX] * n_audio_slots
                + self._user_footer_and_asst_header_ids
            )
        turn_ids_t = torch.tensor(turn_template_ids, device=device).unsqueeze(0).expand(B, -1)  # (B, L)
        audio_mask = turn_ids_t == AUDIO_TOKEN_IDX  # (B, L)

        text_tokens = turn_ids_t.where(~audio_mask, torch.zeros_like(turn_ids_t))
        input_embeds = self.embed_tokens(text_tokens)  # (B, L, H)

        # Replace audio placeholder positions with actual audio embeddings
        input_embeds[audio_mask] = audio_chunk_embs.reshape(-1, audio_chunk_embs.shape[-1])
        position_ids = None
        attn_bias_4d = None
        new_chunk_ids_for_history: Optional[Tensor] = None
        new_is_audio_for_history: Optional[Tensor] = None
        if self.core_cfg.use_modality_position_ids:
            position_ids = self._build_streaming_position_ids(turn_ids_t, audio_mask, state)
        elif self._uses_interleaved_stream_state:
            (
                position_ids,
                attn_bias_4d,
                new_chunk_ids_for_history,
                new_is_audio_for_history,
            ) = self._build_chunk_local_step_inputs(state, audio_mask)

        # 5. Forward through LLM with cache
        # Resolve parallel-decode mode: explicit kwarg wins, else cfg default.
        use_parallel_decode = bool(
            self.parallel_chunk_heads is not None
            and (parallel_chunk_decode if parallel_chunk_decode is not None else self.core_cfg.parallel_chunk_decode)
        )
        input_len = input_embeds.shape[1]
        state.attention_mask = torch.cat(
            [state.attention_mask, torch.ones(B, input_len, dtype=state.attention_mask.dtype, device=device)],
            dim=1,
        )
        # Under chunk-local audio attention we hand the LLM a 4-D additive bias
        # that already encodes the (causal + chunk-local + pad-aware) mask
        # spanning ``(history + new)``; the running 2-D ``state.attention_mask``
        # is still maintained because every other code path (sep-pos, baseline)
        # consumes it.
        llm_attn = attn_bias_4d if attn_bias_4d is not None else state.attention_mask
        out = self.llm(
            inputs_embeds=input_embeds,
            past_key_values=state.cache,
            attention_mask=llm_attn,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
            output_hidden_states=use_parallel_decode,
        )
        if new_chunk_ids_for_history is not None:
            self._append_chunk_local_history(
                state, new_chunk_ids_for_history, new_is_audio_for_history
            )
        state.cache = out.past_key_values
        for b in range(B):
            state.seq_lens[b] += input_len

        if os.environ.get("STREAMING_STT_DEBUG_FIRST_LOGITS") == "1":
            with torch.no_grad():
                last_logits = out.logits[:, -1, :]
                top_v, top_i = last_logits.topk(5, dim=-1)
                for b in range(min(B, 2)):
                    pieces = self.tokenizer.ids_to_tokens(top_i[b].tolist())
                    pos_info = ""
                    if position_ids is not None:
                        pos_info = (
                            f" template_pos_first={int(position_ids[b, 0].item())}"
                            f" template_pos_last={int(position_ids[b, -1].item())}"
                        )
                    audio_emb_stats = (
                        f"audio_emb_abs_mean={audio_chunk_embs[b].abs().mean().item():.4f}"
                        f" audio_emb_norm={audio_chunk_embs[b].float().norm().item():.4f}"
                    )
                    logging.info(
                        "streaming_first_logits_debug stream=%d top5_ids=%s top5_pieces=%s top5_logits=%s "
                        "blank_logit=%.4f%s %s",
                        b,
                        top_i[b].tolist(),
                        pieces,
                        [round(v, 4) for v in top_v[b].tolist()],
                        float(last_logits[b, self.blank_token_id].item()),
                        pos_info,
                        audio_emb_stats,
                    )

        # 6. Generation: parallel-head emission (1 prefill + 1 batched feed)
        #    OR autoregressive decode (1 prefill + N sequential feeds).
        if use_parallel_decode:
            # Read hidden at the anchor (last position of the prefill, which is
            # the write_id / asst-header in the turn template). For both compact
            # and non-compact templates the last template token immediately
            # precedes the assistant content positions whose tokens we predict.
            hidden_anchor = out.hidden_states[-1][:, -1, :]  # (B, H)
            generated_per_stream, footer_consumed = self._parallel_chunk_step_decode(
                anchor_hidden=hidden_anchor,
                state=state,
                max_new_tokens=max_new_tokens,
            )
        else:
            generated_per_stream, state.cache, footer_consumed, _, _, _ = self._autoregressive_decode(
                out.logits,
                state.cache,
                state,
                max_new_tokens,
                generation_config,
                **generation_kwargs,
            )

        # 7. Finalize turn — ensure end-of-turn tokens are in the cache.
        needs_footer = [not fc for fc in footer_consumed]
        any_needs_footer = any(needs_footer)
        if any_needs_footer and self._asst_footer_ids:
            flen = len(self._asst_footer_ids)
            footer_ids = torch.tensor(self._asst_footer_ids, device=device).unsqueeze(0).expand(B, -1)
            filler_id = self.blank_token_id if self.has_blank else self.text_pad_id
            filler_ids = torch.full_like(footer_ids, filler_id)
            needs_footer_t = torch.tensor(needs_footer, dtype=torch.bool, device=device).unsqueeze(1)
            footer_input_ids = torch.where(needs_footer_t, footer_ids, filler_ids)
            asst_footer_embs = self.embed_tokens(footer_input_ids)
            footer_position_ids = None
            footer_attn_bias_4d = None
            footer_new_chunk_ids: Optional[Tensor] = None
            footer_new_is_audio: Optional[Tensor] = None
            footer_attention_mask = needs_footer_t.expand(B, flen).to(dtype=state.attention_mask.dtype)
            if self.core_cfg.use_modality_position_ids:
                footer_position_ids = torch.cat(
                    [
                        self._build_next_text_position_ids(
                            B,
                            state,
                            device,
                            active_mask=needs_footer,
                        )
                        for _ in range(flen)
                    ],
                    dim=1,
                )
                state.attention_mask = torch.cat(
                    [state.attention_mask, footer_attention_mask],
                    dim=1,
                )
            elif self._uses_interleaved_stream_state:
                new_is_audio = torch.zeros((B, flen), dtype=torch.bool, device=device)
                active_t = torch.as_tensor(needs_footer, dtype=torch.long, device=device)
                (
                    footer_position_ids,
                    footer_attn_bias_4d,
                    footer_new_chunk_ids,
                    footer_new_is_audio,
                ) = self._build_chunk_local_step_inputs(state, new_is_audio, active_mask=active_t)
                state.attention_mask = torch.cat(
                    [state.attention_mask, footer_attention_mask],
                    dim=1,
                )
            else:
                state.attention_mask = torch.cat(
                    [state.attention_mask, footer_attention_mask],
                    dim=1,
                )
            llm_attn = footer_attn_bias_4d if footer_attn_bias_4d is not None else state.attention_mask
            out = self.llm(
                inputs_embeds=asst_footer_embs,
                past_key_values=state.cache,
                attention_mask=llm_attn,
                position_ids=footer_position_ids,
                use_cache=True,
                return_dict=True,
            )
            if footer_new_chunk_ids is not None:
                self._append_chunk_local_history(state, footer_new_chunk_ids, footer_new_is_audio)
            state.cache = out.past_key_values
            for b in range(B):
                state.seq_lens[b] += flen
        elif all(footer_consumed):
            for b in range(B):
                state.seq_lens[b] += len(self._asst_footer_ids)

        # 8. Store and return
        for b in range(B):
            state.generated_tokens[b].extend(generated_per_stream[b])
        return generated_per_stream

    def _build_offline_emb_chunks(
        self,
        audio_wav: Tensor,
        n_samples: int,
        device: torch.device,
    ) -> list[Tensor]:
        """Pre-compute offline perception embeddings and slice into chunk_size groups.

        Runs the full perception module on the complete audio (the same path
        used during training), then splits the resulting embeddings into
        ``chunk_size``-frame groups that can be fed directly to the LLM turn
        template.  This bypasses both the feature buffer and the streaming
        encoder, isolating the LLM / generation logic from perception.

        Returns a list of ``(1, chunk_size, H)`` tensors, one per chunk.
        """
        chunk_size = self.core_cfg.chunk_size
        with torch.no_grad():
            offline_embs, _ = self.perception(
                input_signal=audio_wav.unsqueeze(0),
                input_signal_length=torch.tensor([n_samples], device=device),
            )
            offline_embs, _ = self._apply_audio_subsampling(offline_embs)
        # After subsampling each chunk holds ceil(chunk_size / factor) audio tokens.
        slot_size = self._audio_slots_per_chunk()
        total_frames = offline_embs.shape[1]
        chunks: list[Tensor] = []
        for start in range(0, total_frames, slot_size):
            end = min(start + slot_size, total_frames)
            chunk = offline_embs[:, start:end, :]
            if chunk.shape[1] < slot_size:
                chunk = F.pad(chunk, (0, 0, 0, slot_size - chunk.shape[1]))
            chunks.append(chunk)
        return chunks

    def _generate_offline(
        self,
        audios: Tensor,
        n_samples_list: list[int],
        system_prompt: Union[str, List[str]],
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        **generation_kwargs,
    ) -> list[str]:
        """Offline generation: process entire audio in a single LLM forward pass.

        Unlike the streaming path, this method runs offline perception on the
        full audio (no chunking, no streaming cache), builds one input sequence
        per sample (system prompt + user turn with all audio frames + assistant
        header), and prefills the LLM in a single forward pass before decoding.

        Args:
            audios: ``(B, T_samples)`` raw waveforms (zero-padded to max length).
            n_samples_list: List of B valid sample counts.
            system_prompt: System prompt string (shared) or list of B per-sample prompts.
            max_new_tokens: Maximum tokens to generate per sample.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            generation_kwargs: Per-call overrides for generation parameters.

        Returns:
            List of B transcription strings.
        """
        B = len(n_samples_list)
        if B == 0 or max(n_samples_list) == 0:
            return [""] * B
        device = audios.device
        dtype = self.embed_tokens.weight.dtype

        # 1. Encode system prompt(s)
        hf_tok = self.tokenizer.tokenizer
        prompts = [system_prompt] * B if isinstance(system_prompt, str) else system_prompt
        all_sys_embs = []
        for prompt in prompts:
            sys_ids = hf_tok.apply_chat_template(
                [{"role": "system", "content": prompt}],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            embs = self.embed_tokens(torch.tensor(sys_ids, device=device, dtype=torch.long).unsqueeze(0)).squeeze(
                0
            )  # (L_sys_b, H)
            all_sys_embs.append(embs)

        # 2. Embed turn template components (shared across batch)
        user_header_embs = self.embed_tokens(
            torch.tensor(self._user_header_ids, device=device, dtype=torch.long).unsqueeze(0)
        )  # (1, L_uh, H)
        uf_ah_embs = self.embed_tokens(
            torch.tensor(self._user_footer_and_asst_header_ids, device=device, dtype=torch.long).unsqueeze(0)
        )  # (1, L_uf, H)

        # 3. Run offline perception on the full batch
        audio_lens_t = torch.tensor(n_samples_list, device=device)
        batch_audio_embs, batch_emb_lens = self.perception(
            input_signal=audios,
            input_signal_length=audio_lens_t,
        )  # (B, T_enc_max, H), (B,)
        batch_audio_embs, batch_emb_lens = self._apply_audio_subsampling(batch_audio_embs, batch_emb_lens)
        batch_audio_embs = batch_audio_embs.type_as(self.embed_tokens.weight)
        all_audio_embs = [batch_audio_embs[b, : int(batch_emb_lens[b].item())] for b in range(B)]

        # 4. Build per-sample input sequences:
        #    sys_embs[b] + user_header_embs + audio_embs[b] + user_footer_asst_header_embs
        sample_embs_list = []
        sample_lens = []
        for b in range(B):
            seq = torch.cat(
                [all_sys_embs[b], user_header_embs.squeeze(0), all_audio_embs[b], uf_ah_embs.squeeze(0)],
                dim=0,
            )  # (L_b, H)
            sample_embs_list.append(seq)
            sample_lens.append(seq.shape[0])

        # 5. Left-pad to max length and build attention mask
        max_len = max(sample_lens)
        H = sample_embs_list[0].shape[-1]
        input_embeds = torch.zeros(B, max_len, H, device=device, dtype=dtype)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for b in range(B):
            offset = max_len - sample_lens[b]
            input_embeds[b, offset:] = sample_embs_list[b]
            attention_mask[b, offset:] = 1

        # 6. LLM prefill (single forward pass)
        out = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        # 7. Autoregressive decode
        state = StreamingState(
            cache=out.past_key_values,
            generated_tokens=[[] for _ in range(B)],
            seq_lens=[max_len] * B,
            attention_mask=attention_mask,
            batch_size=B,
        )
        generated_per_stream, _, _, _, _, _ = self._autoregressive_decode(
            out.logits,
            out.past_key_values,
            state,
            max_new_tokens,
            generation_config,
            **generation_kwargs,
        )

        # 8. Decode tokens to text
        return [decode_with_blank(toks, self.blank_token, self.tokenizer) for toks in generated_per_stream]

    def _generate_dynamic_streaming(
        self,
        audios: Tensor,
        n_samples_list: list[int],
        system_prompt: Union[str, List[str]],
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        inference_chunk_size: Optional[int] = None,
        dynamic_min_chunk_size: int = 0,
        dynamic_max_chunk_size: Optional[int] = None,
        lm_head_emit_threshold: Optional[float] = None,
        debug_logs: Optional[list] = None,
        **generation_kwargs,
    ) -> list[str]:
        """Batched dynamic-chunking generation.

        All B streams are processed in lockstep: each step feeds exactly 1
        embedding per stream to the LLM (audio frame, template token, or
        generated text token).  Perception runs with ``inference_chunk_size``
        frames and the resulting embeddings are buffered per-stream so the
        LLM still consumes them one at a time.

        Args:
            audios: ``(B, T_samples)`` raw waveforms.
            n_samples_list: List of B valid sample counts.
            system_prompt: System prompt string or list.
            max_new_tokens: Max tokens per text generation segment.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            inference_chunk_size: Number of encoder frames per perception call
                (default 1).  Embeddings are buffered and fed to the LLM one
                at a time.
            dynamic_min_chunk_size: Minimum frames before the model is allowed
                to trigger generation (default 0, no minimum).
            dynamic_max_chunk_size: Maximum frames before forcing generation.
                ``None`` means no upper bound (default).
            generation_kwargs: Per-call overrides.
        """
        B = len(n_samples_list)
        if B == 0:
            return []
        device = audios.device
        # Default inference_chunk_size: match the encoder granularity the
        # model was trained at — chunk_size for fixed chunking, 1 for dynamic
        # chunking. Even though the streaming encoder is conceptually causal,
        # cache-aware Conformer's internal chunking + lookahead window is NOT
        # identical at different batch sizes (train/inference mismatch causes
        # slight embedding drift → measurable WER regression).
        # Override via inference_chunk_size only when you've trained the
        # encoder for that granularity (e.g. chunk_step > 1 in dataset).
        if inference_chunk_size is None:
            N = max(self.core_cfg.chunk_size, 1)
        else:
            N = inference_chunk_size
        chunk_samples = math.ceil(N * self.core_cfg.frame_length_in_secs * self.core_cfg.sample_rate)

        # --- Init state ---
        state = self.get_init_streaming_state(system_prompt, device=device, batch_size=B)
        state.audio_feature_buffer = self.get_audio_feature_buffer(
            batch_size=B,
            chunk_size_override=N,
        )

        # --- Per-stream state machine ---
        HEADER, LISTENING, FOOTER, GENERATING, BLANK_FEED, ASST_FOOTER, DONE = range(7)
        # When user_header_ids is empty (compact template), skip HEADER entirely.
        _initial_state = LISTENING if not self._user_header_ids else HEADER
        stream_state = [_initial_state] * B
        template_pos = [0] * B  # position within current template seq
        audio_sample_idx = [0] * B  # next audio sample offset for perception
        gen_token_count = [0] * B  # tokens generated in current GENERATING phase
        last_gen_token = [self.text_pad_id] * B  # last generated token per stream
        all_tokens: list[list[int]] = [[] for _ in range(B)]
        use_generation_history = bool(generation_kwargs.pop("use_generation_history", False))

        # Per-stream audio embedding buffer (filled by perception, consumed 1 at a time)
        audio_emb_buf: list[list[Tensor]] = [[] for _ in range(B)]

        # Fixed-chunk mode: count frames consumed per segment to transition
        # after exactly chunk_size frames (ignoring model predictions).
        fixed_chunk_mode = self.core_cfg.chunk_size > 0
        fixed_chunk_size = self.core_cfg.chunk_size if fixed_chunk_mode else 0
        frames_in_segment = [0] * B  # frames consumed in current LISTENING segment

        # --- Audio-frame debug logging ---
        # When debug_logs is provided (a list passed in by the caller), we
        # populate it with per-LISTENING-frame diagnostic records per stream
        # — used to investigate whether the model is overfitting to predict
        # blank, whether the aux head is well-calibrated, etc.
        log_frames = debug_logs is not None
        per_stream_frame_logs: list[list[dict]] = [[] for _ in range(B)] if log_frames else []
        total_frame_idx = [0] * B  # cumulative LISTENING frames per stream

        uf_ah_ids = self._user_footer_and_asst_header_ids
        uh_ids = self._user_header_ids
        af_ids = self._asst_footer_ids
        user_footer_first_id = self._user_footer_first_id

        # Max steps: LLM's max context length minus the system prompt already in KV cache.
        max_model_len = getattr(self.llm.config, 'max_position_embeddings', 40960)
        max_steps = max_model_len - max(state.seq_lens)

        # Padding embedding for DONE streams and empty-buffer LISTENING streams.
        # Use the blank token embedding when blank is enabled (a real token the
        # model knows). Otherwise fall back to the text pad id.
        pad_token_id = self.blank_token_id if self.has_blank else self.text_pad_id
        pad_emb = self.embed_tokens(torch.tensor([pad_token_id], device=device)).squeeze(0)  # (H,)

        for _step in range(max_steps):
            # --- Refill audio embedding buffers for LISTENING streams ---
            needs_refill = [
                b
                for b in range(B)
                if stream_state[b] == LISTENING
                and len(audio_emb_buf[b]) == 0
                and audio_sample_idx[b] < n_samples_list[b]
            ]
            if needs_refill:
                # Run perception only for streams that need refill.
                # The feature buffer selectively updates via stream_id.
                # The encoder cache is sliced to the subset, then scattered back.
                idx_t = torch.tensor(needs_refill, device=device)

                # Build frames (only for refill streams)
                frames = []
                for b in needs_refill:
                    start = audio_sample_idx[b]
                    end = min(start + chunk_samples, n_samples_list[b])
                    wav = audios[b, start:end]
                    if wav.shape[0] < chunk_samples:
                        wav = F.pad(wav, (0, chunk_samples - wav.shape[0]))
                    frames.append(Frame(samples=wav, stream_id=b, length=end - start))
                    audio_sample_idx[b] = end

                # Feature buffer selectively updates only the submitted stream_ids
                features, right_paddings = state.audio_feature_buffer.update(frames)
                processed_signal = torch.stack(features).type_as(self.embed_tokens.weight)  # (S, D, T)
                processed_signal_length = torch.tensor(
                    [processed_signal.shape[-1] - int(rp) for rp in right_paddings],
                    device=device,
                ).long()

                # Slice encoder cache to the subset
                sub_cache_lc = state.audio_cache.cache_last_channel.index_select(1, idx_t)
                sub_cache_lt = state.audio_cache.cache_last_time.index_select(1, idx_t)
                sub_cache_lcl = state.audio_cache.cache_last_channel_len[idx_t]

                outputs = self.perception(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=sub_cache_lc,
                    cache_last_time=sub_cache_lt,
                    cache_last_channel_len=sub_cache_lcl,
                    streaming=True,
                )
                batch_embs, _, new_cache = outputs

                # Scatter updated cache back into the full B-sized cache
                if new_cache is not None:
                    for i, b in enumerate(needs_refill):
                        state.audio_cache.cache_last_channel[:, b] = new_cache['cache_last_channel'][:, i]
                        state.audio_cache.cache_last_time[:, b] = new_cache['cache_last_time'][:, i]
                        state.audio_cache.cache_last_channel_len[b] = new_cache['cache_last_channel_len'][i]

                # Distribute embeddings into per-stream buffers.
                # Pad to exactly N frames (matching fast path's pad/trim behavior).
                H_enc = batch_embs.shape[-1]
                for i, b in enumerate(needs_refill):
                    n_enc = batch_embs[i].shape[0]
                    for f in range(n_enc):
                        audio_emb_buf[b].append(batch_embs[i, f])
                    # Pad with zeros if encoder returned fewer than N frames
                    for _ in range(N - n_enc):
                        audio_emb_buf[b].append(torch.zeros(H_enc, device=device, dtype=batch_embs.dtype))

            # --- Build (B, 1, H) input embeddings based on per-stream state ---
            # Each entry is (H,); we stack → (B, H) then unsqueeze → (B, 1, H).
            embs_list = []
            for b in range(B):
                if stream_state[b] == LISTENING:
                    if audio_emb_buf[b]:
                        embs_list.append(audio_emb_buf[b].pop(0))  # (H,)
                    else:
                        embs_list.append(pad_emb)
                elif stream_state[b] == FOOTER:
                    tid = uf_ah_ids[template_pos[b]]
                    embs_list.append(self.embed_tokens(torch.tensor([tid], device=device)).squeeze(0))  # (H,)
                elif stream_state[b] == GENERATING:
                    embs_list.append(
                        self.embed_tokens(torch.tensor([last_gen_token[b]], device=device)).squeeze(0)  # (H,)
                    )
                elif stream_state[b] == BLANK_FEED:
                    # Only reached when has_blank is True (guarded at transition sites).
                    embs_list.append(
                        self.embed_tokens(torch.tensor([self.blank_token_id], device=device)).squeeze(0)  # (H,)
                    )
                elif stream_state[b] == ASST_FOOTER:
                    tid = af_ids[template_pos[b]]
                    embs_list.append(self.embed_tokens(torch.tensor([tid], device=device)).squeeze(0))  # (H,)
                elif stream_state[b] == HEADER:
                    tid = uh_ids[template_pos[b]]
                    embs_list.append(self.embed_tokens(torch.tensor([tid], device=device)).squeeze(0))  # (H,)
                else:  # DONE
                    embs_list.append(pad_emb)

            input_embs = torch.stack(embs_list).unsqueeze(1)  # (B, H) → (B, 1, H)

            # --- Single LLM forward ---
            use_aux = self.core_cfg.use_chunk_classifier and self.core_cfg.chunk_classifier_use_at_inference
            llm_kwargs = dict(
                inputs_embeds=input_embs,
                past_key_values=state.cache,
                use_cache=True,
                output_hidden_states=use_aux,
                return_dict=True,
            )
            if state.attention_mask is not None:
                state.attention_mask = torch.cat(
                    [state.attention_mask, torch.ones(B, 1, dtype=state.attention_mask.dtype, device=device)],
                    dim=1,
                )
                llm_kwargs["attention_mask"] = state.attention_mask
            out = self.llm(**llm_kwargs)
            state.cache = out.past_key_values
            for b in range(B):
                state.seq_lens[b] += 1

            # --- Aux chunk-boundary classifier: full-sequence forward ---
            # Append the new LLM hidden state to the running buffer, then run
            # the K-layer aux backbone over the entire accumulated buffer
            # (no aux KV cache — matches training, where the aux backbone sees
            # the full sequence in one pass). Cost is K layers × current length
            # per step; cheap at K≈2.
            aux_last_hidden = None
            if use_aux:
                new_h = out.hidden_states[-1]  # (B, 1, H)
                if state.aux_hidden_buffer is None:
                    state.aux_hidden_buffer = new_h
                else:
                    state.aux_hidden_buffer = torch.cat([state.aux_hidden_buffer, new_h], dim=1)
                aux_out = self.chunk_classifier_backbone(
                    inputs_embeds=state.aux_hidden_buffer,
                    attention_mask=state.attention_mask,
                    return_dict=True,
                )
                aux_last_hidden = aux_out.last_hidden_state  # (B, L_so_far, H)

            # --- Per-stream state transitions ---
            for b in range(B):
                if stream_state[b] == DONE:
                    continue

                if stream_state[b] == LISTENING:
                    frames_in_segment[b] += 1
                    total_frame_idx[b] += 1
                    seg_idx_now = frames_in_segment[b]  # captured before any reset
                    decision_str = "keep_listening"
                    aux_p_log: Optional[float] = None  # only set when aux head consulted
                    if fixed_chunk_mode:
                        # Fixed chunking: transition after exactly chunk_size frames
                        if frames_in_segment[b] >= fixed_chunk_size:
                            stream_state[b] = FOOTER
                            template_pos[b] = 0
                            frames_in_segment[b] = 0
                            decision_str = "emit_forced_chunk_size"
                        elif not audio_emb_buf[b] and audio_sample_idx[b] >= n_samples_list[b]:
                            stream_state[b] = DONE
                            decision_str = "done_audio_end"
                    else:
                        # Dynamic chunking: transition when model predicts <user_footer>,
                        # subject to [min_chunk_size, max_chunk_size] bounds.
                        if dynamic_max_chunk_size is not None and frames_in_segment[b] >= dynamic_max_chunk_size:
                            # Forced transition — hit upper bound
                            stream_state[b] = FOOTER
                            template_pos[b] = 0
                            frames_in_segment[b] = 0
                            decision_str = "emit_forced_max"
                        elif frames_in_segment[b] < dynamic_min_chunk_size:
                            # Below minimum — ignore model prediction, keep listening
                            if not audio_emb_buf[b] and audio_sample_idx[b] >= n_samples_list[b]:
                                # Audio exhausted before reaching min — still emit
                                stream_state[b] = FOOTER
                                template_pos[b] = 0
                                frames_in_segment[b] = 0
                                decision_str = "emit_forced_audio_end_below_min"
                            else:
                                decision_str = "below_min_keep"
                        else:
                            # In [min, max] window — use model prediction.
                            # Either the aux classifier head (when enabled) or
                            # the LM head's vocab sample (legacy path).
                            if use_aux and aux_last_hidden is not None:
                                h_last = aux_last_hidden[b, -1, :]  # (H,)
                                aux_logit_b = self.chunk_classifier_head(h_last)
                                aux_p_log = float(torch.sigmoid(aux_logit_b).item())
                                emit = aux_p_log >= self.core_cfg.chunk_classifier_threshold
                            elif lm_head_emit_threshold is not None:
                                # Threshold-based LM-head decision: fire when
                                # p(user_footer_first_id) ≥ threshold. Lower
                                # values catch boundaries where the LM is
                                # moderately confident but loses argmax to blank.
                                lm_probs_emit = torch.softmax(out.logits[b, -1, :].float(), dim=-1)
                                p_ufid_emit = (
                                    float(lm_probs_emit[user_footer_first_id].item())
                                    if user_footer_first_id is not None
                                    else 0.0
                                )
                                emit = p_ufid_emit >= lm_head_emit_threshold
                            else:
                                token = self._sample_token(
                                    out.logits[b : b + 1, -1, :],
                                    None,
                                    generation_config,
                                    **generation_kwargs,
                                ).item()
                                emit = token == user_footer_first_id
                            if emit:
                                stream_state[b] = FOOTER
                                template_pos[b] = 0
                                frames_in_segment[b] = 0
                                decision_str = "emit_model"
                            elif (
                                not audio_emb_buf[b] and audio_sample_idx[b] >= n_samples_list[b] and not all_tokens[b]
                            ):
                                # Audio exhausted in [min, max] window AND no
                                # text emitted yet for this stream — force a
                                # final FOOTER → GENERATING sweep so we don't
                                # produce an empty prediction. Once any chunk
                                # has been emitted, the model's "keep listening
                                # at end of audio" signal is trustworthy ("I'm
                                # done"), so we go to DONE without forcing —
                                # avoiding the trailing-hallucination failure
                                # mode where forced-emit invents extra text.
                                stream_state[b] = FOOTER
                                template_pos[b] = 0
                                frames_in_segment[b] = 0
                                decision_str = "emit_forced_audio_end"
                            elif not audio_emb_buf[b] and audio_sample_idx[b] >= n_samples_list[b]:
                                # Already emitted at least once and model says
                                # blank at end-of-audio: trust it and stop.
                                stream_state[b] = DONE
                                decision_str = "done_audio_end"

                    # Per-frame debug log (LM head + aux head diagnostics).
                    if log_frames:
                        lm_logits_b = out.logits[b, -1, :]
                        lm_probs_b = torch.softmax(lm_logits_b.float(), dim=-1)
                        topk_p, topk_id = torch.topk(lm_probs_b, 5)
                        lm_top5 = [
                            {"id": int(topk_id[k].item()), "prob": float(topk_p[k].item())}
                            for k in range(topk_p.numel())
                        ]
                        p_ufid = (
                            float(lm_probs_b[user_footer_first_id].item())
                            if user_footer_first_id is not None
                            else None
                        )
                        p_blank = float(lm_probs_b[self.blank_token_id].item()) if self.has_blank else None
                        # If aux is on but we didn't consult it (below min / above
                        # max / fixed-chunk path), still compute it for visibility.
                        if aux_p_log is None and use_aux and aux_last_hidden is not None:
                            aux_logit_b = self.chunk_classifier_head(aux_last_hidden[b, -1, :])
                            aux_p_log = float(torch.sigmoid(aux_logit_b).item())
                        per_stream_frame_logs[b].append(
                            {
                                "step": _step,
                                "total_frame_idx": total_frame_idx[b],
                                "frame_idx_in_segment": seg_idx_now,
                                "lm_top5": lm_top5,
                                "lm_p_user_footer_first": p_ufid,
                                "lm_p_blank": p_blank,
                                "aux_p_emit": aux_p_log,
                                "decision": decision_str,
                            }
                        )

                elif stream_state[b] == FOOTER:
                    template_pos[b] += 1
                    if template_pos[b] >= len(uf_ah_ids):
                        stream_state[b] = GENERATING
                        gen_token_count[b] = 0
                        # Use the logit from this step as the first generation logit
                        first_token = self._sample_token(
                            out.logits[b : b + 1, -1, :],
                            all_tokens[b] if use_generation_history else None,
                            generation_config,
                            **generation_kwargs,
                        ).item()
                        first_is_stop = (
                            self._eos_id is not None and first_token == self._eos_id
                        ) or first_token == self.blank_token_id
                        if first_is_stop:
                            # Immediately done generating — append chunk separator
                            # (blank when enabled, else EOS so decode_with_blank splits chunks)
                            all_tokens[b].append(self.blank_token_id if self.has_blank else self._eos_id)
                            if fixed_chunk_mode and self.has_blank:
                                # Feed blank to LLM first (matches training sequence)
                                stream_state[b] = BLANK_FEED
                            elif af_ids:
                                stream_state[b] = ASST_FOOTER
                                template_pos[b] = 0
                            else:
                                self._dynamic_finish_generating(
                                    b,
                                    stream_state,
                                    template_pos,
                                    audio_emb_buf,
                                    audio_sample_idx,
                                    n_samples_list,
                                    _initial_state,
                                    DONE,
                                )
                        else:
                            all_tokens[b].append(first_token)
                            last_gen_token[b] = first_token
                            gen_token_count[b] = 1

                elif stream_state[b] == GENERATING:
                    token = self._sample_token(
                        out.logits[b : b + 1, -1, :],
                        all_tokens[b] if use_generation_history else None,
                        generation_config,
                        **generation_kwargs,
                    ).item()
                    # Stop on EOS, blank, or max tokens (matching _autoregressive_decode)
                    is_eos = self._eos_id is not None and token == self._eos_id
                    is_blank = token == self.blank_token_id
                    is_max = gen_token_count[b] >= max_new_tokens
                    if is_eos or is_blank or is_max:
                        # Append chunk separator (blank when enabled, else EOS).
                        # decode_with_blank splits per-chunk outputs on this.
                        all_tokens[b].append(self.blank_token_id if self.has_blank else self._eos_id)
                        # Do NOT feed <blank> here. Training for non-empty
                        # chunks ends as `text <asst_footer>` (no blank between
                        # text and asst_footer — blank only appears as the
                        # *content* of empty chunks, which exits via FOOTER's
                        # first_is_stop path above). Feeding <blank> at this
                        # position pollutes the KV cache with an OOD token,
                        # manifesting as premature EOS in subsequent chunks
                        # (heavy deletion errors, especially in compact mode
                        # where the single asst_footer token can't recover the
                        # context).
                        if af_ids:
                            stream_state[b] = ASST_FOOTER
                            template_pos[b] = 0
                        else:
                            self._dynamic_finish_generating(
                                b,
                                stream_state,
                                template_pos,
                                audio_emb_buf,
                                audio_sample_idx,
                                n_samples_list,
                                _initial_state,
                                DONE,
                            )
                    else:
                        all_tokens[b].append(token)
                        last_gen_token[b] = token
                        gen_token_count[b] += 1

                elif stream_state[b] == BLANK_FEED:
                    # Blank was fed to LLM this step. Transition to ASST_FOOTER.
                    if af_ids:
                        stream_state[b] = ASST_FOOTER
                        template_pos[b] = 0
                    else:
                        self._dynamic_finish_generating(
                            b,
                            stream_state,
                            template_pos,
                            audio_emb_buf,
                            audio_sample_idx,
                            n_samples_list,
                            _initial_state,
                            DONE,
                        )

                elif stream_state[b] == ASST_FOOTER:
                    template_pos[b] += 1
                    if template_pos[b] >= len(af_ids):
                        self._dynamic_finish_generating(
                            b,
                            stream_state,
                            template_pos,
                            audio_emb_buf,
                            audio_sample_idx,
                            n_samples_list,
                            _initial_state,
                            DONE,
                        )

                elif stream_state[b] == HEADER:
                    template_pos[b] += 1
                    if template_pos[b] >= len(uh_ids):
                        stream_state[b] = LISTENING

            if all(s == DONE for s in stream_state):
                break

        if log_frames:
            debug_logs.extend(per_stream_frame_logs)
        return [decode_with_blank(toks, self.blank_token, self.tokenizer) for toks in all_tokens]

    @staticmethod
    def _dynamic_finish_generating(
        b,
        stream_state,
        template_pos,
        audio_emb_buf,
        audio_sample_idx,
        n_samples_list,
        next_listen_state,
        DONE,
    ):
        """Transition stream b from GENERATING to next_listen_state (HEADER or LISTENING) or DONE.

        ``next_listen_state`` is HEADER when user_header_ids is non-empty, or LISTENING
        when user_header_ids is empty (compact template) — the HEADER state is skipped
        since there are no header tokens to feed.
        """
        has_more_audio = bool(audio_emb_buf[b]) or audio_sample_idx[b] < n_samples_list[b]
        if has_more_audio:
            stream_state[b] = next_listen_state
            template_pos[b] = 0
        else:
            stream_state[b] = DONE

    def _generate_chunked_streaming(
        self,
        audios: Tensor,
        n_samples_list: list[int],
        system_prompt: Union[str, List[str]],
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        use_offline_embs: bool = False,
        inference_audio_chunks_per_turn: int = 1,
        parallel_chunk_decode: Optional[bool] = None,
        **generation_kwargs,
    ) -> list[str]:
        """Chunk-by-chunk streaming generation for B samples in lockstep.

        Args:
            audios: ``(B, T_samples)`` raw waveforms (zero-padded to max length).
            n_samples_list: List of B valid sample counts.
            system_prompt: System prompt string (shared) or list of B per-sample prompts.
            max_new_tokens: Maximum tokens to generate per chunk per stream.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            use_offline_embs: When True, bypass streaming perception with offline embeddings.
            inference_audio_chunks_per_turn: Number of fixed training chunks to
                group into one user turn at inference. Default 1 preserves
                legacy fixed-chunk decoding.
            generation_kwargs: Per-call overrides for generation parameters.

        Returns:
            List of B transcription strings.
        """
        assert self.core_cfg.chunk_size > 0, (
            f"chunk_size must be positive for streaming mode, got {self.core_cfg.chunk_size}. "
            f"Use generate() which dispatches to _generate_offline() for chunk_size < 0."
        )
        B = len(n_samples_list)
        if B == 0 or max(n_samples_list) == 0:
            return [""] * B
        device = audios.device

        chunk_size = self.core_cfg.chunk_size
        chunks_per_turn = max(int(inference_audio_chunks_per_turn), 1)
        turn_chunk_size = chunk_size * chunks_per_turn
        chunk_samples = math.ceil(chunk_size * self.core_cfg.frame_length_in_secs * self.core_cfg.sample_rate)
        turn_chunk_samples = chunk_samples * chunks_per_turn

        num_chunks_per_stream = [math.ceil(ns / chunk_samples) if ns > 0 else 0 for ns in n_samples_list]
        num_turns_per_stream = [math.ceil(nc / chunks_per_turn) if nc > 0 else 0 for nc in num_chunks_per_stream]

        if B > 1 and getattr(self, "_inference_use_flush", False) and len(set(num_turns_per_stream)) > 1:
            # The final <flush> turn depends on the exact KV state immediately
            # after an utterance's last real audio chunk. A mixed-length batch
            # would advance shorter streams with zero-audio filler turns while
            # longer streams continue, then flush all streams from the longest
            # stream's time. Bucket by equal turn count so each bucket can still
            # decode batched, but no stream receives post-utterance filler before
            # its flush.
            prompts = [system_prompt] * B if isinstance(system_prompt, str) else list(system_prompt)
            grouped: dict[int, list[int]] = {}
            for b, n_turns in enumerate(num_turns_per_stream):
                grouped.setdefault(n_turns, []).append(b)

            results: list[Optional[str]] = [None] * B
            for indices in grouped.values():
                sub_audios = audios[indices]
                sub_lens = [n_samples_list[i] for i in indices]
                sub_prompts = [prompts[i] for i in indices]
                sub_results = self._generate_chunked_streaming(
                    sub_audios,
                    sub_lens,
                    sub_prompts,
                    max_new_tokens,
                    generation_config,
                    use_offline_embs=use_offline_embs,
                    inference_audio_chunks_per_turn=inference_audio_chunks_per_turn,
                    parallel_chunk_decode=parallel_chunk_decode,
                    **generation_kwargs,
                )
                for i, hyp in zip(indices, sub_results):
                    results[i] = hyp
            return [r if r is not None else "" for r in results]

        state = self.get_init_streaming_state(system_prompt, device=device, batch_size=B)

        offline_emb_chunks_list = None
        if use_offline_embs:
            offline_emb_chunks_list = [
                self._build_offline_emb_chunks(audios[b, : n_samples_list[b]], n_samples_list[b], device)
                for b in range(B)
            ]

        max_turns = max(num_turns_per_stream)
        all_token_ids: list[list[int]] = [[] for _ in range(B)]

        for turn_i in range(max_turns):
            # Build B audio chunks (zero-pad finished streams)
            chunks = []
            chunk_lens = []
            for b in range(B):
                start = turn_i * turn_chunk_samples
                end = min(start + turn_chunk_samples, n_samples_list[b])
                if start >= n_samples_list[b]:
                    # Stream b has finished — send zeros with zero valid length
                    chunks.append(torch.zeros(turn_chunk_samples, device=device, dtype=audios.dtype))
                    chunk_lens.append(0)
                else:
                    wav = audios[b, start:end]
                    if wav.shape[0] < turn_chunk_samples:
                        wav = F.pad(wav, (0, turn_chunk_samples - wav.shape[0]))
                    chunks.append(wav)
                    chunk_lens.append(end - start)

            audio_batch = torch.stack(chunks)  # (B, turn_chunk_samples)
            lens_batch = torch.tensor(chunk_lens, device=device)

            extra_kwargs = {}
            if offline_emb_chunks_list is not None:
                emb_chunks = []
                for b in range(B):
                    start_chunk = turn_i * chunks_per_turn
                    end_chunk = min(start_chunk + chunks_per_turn, len(offline_emb_chunks_list[b]))
                    if start_chunk < len(offline_emb_chunks_list[b]):
                        emb = torch.cat(offline_emb_chunks_list[b][start_chunk:end_chunk], dim=1)
                        if emb.shape[1] < turn_chunk_size:
                            emb = F.pad(emb, (0, 0, 0, turn_chunk_size - emb.shape[1]))
                        emb_chunks.append(emb)
                    else:
                        H = offline_emb_chunks_list[0][0].shape[-1]
                        emb_chunks.append(torch.zeros(1, turn_chunk_size, H, device=device, dtype=audios.dtype))
                extra_kwargs["_audio_embs"] = torch.cat(emb_chunks, dim=0)
            elif chunks_per_turn > 1:
                # Keep encoder streaming granularity identical to legacy
                # fixed-chunk inference.  The multi-chunk option only groups
                # the resulting audio embeddings into a larger LLM user turn.
                turn_emb_chunks = []
                for sub_i in range(chunks_per_turn):
                    sub_chunks = []
                    sub_lens = []
                    for b in range(B):
                        sub_start = (turn_i * chunks_per_turn + sub_i) * chunk_samples
                        sub_end = min(sub_start + chunk_samples, n_samples_list[b])
                        if sub_start >= n_samples_list[b]:
                            sub_chunks.append(torch.zeros(chunk_samples, device=device, dtype=audios.dtype))
                            sub_lens.append(0)
                        else:
                            wav = audios[b, sub_start:sub_end]
                            if wav.shape[0] < chunk_samples:
                                wav = F.pad(wav, (0, chunk_samples - wav.shape[0]))
                            sub_chunks.append(wav)
                            sub_lens.append(sub_end - sub_start)
                    sub_audio_batch = torch.stack(sub_chunks)
                    sub_lens_batch = torch.tensor(sub_lens, device=device)
                    turn_emb_chunks.append(
                        self._encode_streaming_audio_chunk(
                            sub_audio_batch,
                            sub_lens_batch,
                            state,
                            chunk_size_frames=chunk_size,
                        )
                    )
                extra_kwargs["_audio_embs"] = torch.cat(turn_emb_chunks, dim=1)

            chunk_tokens = self._chunked_streaming_step(
                audio_batch,
                lens_batch,
                state,
                max_new_tokens,
                generation_config,
                **extra_kwargs,
                inference_chunk_size_frames=turn_chunk_size,
                parallel_chunk_decode=parallel_chunk_decode,
                **generation_kwargs,
            )
            for b in range(B):
                # Only collect tokens for streams that are still active
                if turn_i < num_turns_per_stream[b]:
                    all_token_ids[b].extend(chunk_tokens[b])

        # Explicit end-of-audio flush: feed a single <flush> control turn and
        # decode the residual tail (words the model was trained to defer past the
        # last chunk) until <|im_end|>. Enabled only when the model was trained
        # with use_flush (see _ensure_inference_cache).
        if getattr(self, "_inference_use_flush", False):
            flush_tokens = self._flush_step(state, max_new_tokens, generation_config, **generation_kwargs)
            for b in range(B):
                all_token_ids[b].extend(flush_tokens[b])

        return [decode_with_blank(toks, self.blank_token, self.tokenizer) for toks in all_token_ids]

    def _flush_step(
        self,
        state: StreamingState,
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        **generation_kwargs,
    ) -> list[list[int]]:
        """Feed one ``<flush>`` control turn (no audio) into the running state and
        autoregressively decode the residual tail until ``<|im_end|>``.

        Mirrors the text-turn portion of :meth:`_chunked_streaming_step` for the
        baseline scheme (no modality-position-ids / chunk-local-attn / parallel
        heads — those are gated off in ``_ensure_inference_cache``). Returns a
        list of B token-ID lists (one per stream).
        """
        device = self.embed_tokens.weight.device
        B = state.batch_size
        # Turn layout: [user_header] <flush> [user_footer + asst_header]. In the
        # compact template user_header is empty and the footer is the write token.
        turn_ids = (
            list(self._user_header_ids)
            + [int(self._inference_flush_id)]
            + list(self._user_footer_and_asst_header_ids)
        )
        turn_ids_t = torch.tensor(turn_ids, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        input_embeds = self.embed_tokens(turn_ids_t)
        input_len = input_embeds.shape[1]
        state.attention_mask = torch.cat(
            [state.attention_mask, torch.ones(B, input_len, dtype=state.attention_mask.dtype, device=device)],
            dim=1,
        )
        out = self.llm(
            inputs_embeds=input_embeds,
            past_key_values=state.cache,
            attention_mask=state.attention_mask,
            use_cache=True,
            return_dict=True,
        )
        state.cache = out.past_key_values
        for b in range(B):
            state.seq_lens[b] += input_len
        generated_per_stream, state.cache, _footer_consumed, _, _, _ = self._autoregressive_decode(
            out.logits,
            state.cache,
            state,
            max_new_tokens,
            generation_config,
            **generation_kwargs,
        )
        return generated_per_stream

    @torch.no_grad()
    def generate(
        self,
        audios: Tensor,
        audio_lens: Tensor,
        system_prompt: Union[str, List[str]] = "Transcribe the audio into text.",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_PER_CHUNK,
        generation_config: Optional[GenerationConfig] = None,
        use_offline_embs: bool = False,
        inference_audio_chunks_per_turn: int = 1,
        use_state_machine_inference: bool = False,
        dynamic_min_chunk_size: int = 0,
        dynamic_max_chunk_size: Optional[int] = None,
        lm_head_emit_threshold: Optional[float] = None,
        debug_logs: Optional[list] = None,
        parallel_chunk_decode: Optional[bool] = None,
        **generation_kwargs,
    ) -> list[str]:
        """
        Transcribe full audio(s).

        Args:
            audios: (B, T_samples) raw waveforms.
            audio_lens: (B,) waveform lengths in samples.
            system_prompt: System prompt string (shared) or list of B per-sample prompts.
            max_new_tokens: Maximum tokens to generate per chunk per stream.
            generation_config: Optional HuggingFace GenerationConfig object.
            use_offline_embs: When True, bypass streaming perception with
                offline embeddings. Diagnostic use only.
            inference_audio_chunks_per_turn: For fixed chunking, group this
                many configured audio chunks into one user turn at inference.
                Default 1 preserves legacy inference.
            dynamic_min_chunk_size: For dynamic chunking — minimum frames before
                the model is allowed to trigger generation (default 0).
            dynamic_max_chunk_size: For dynamic chunking — maximum frames before
                forcing generation. ``None`` means no upper bound (default).
            generation_kwargs: Per-call overrides for generation parameters.

        Returns:
            List of transcription strings, one per sample.
        """
        self._ensure_inference_cache()

        with move_embedding(self):
            B = audios.shape[0]
            n_samples_list = [int(audio_lens[b].item()) for b in range(B)]
            if self.core_cfg.use_modality_position_ids and (
                self.core_cfg.chunk_size <= 0 or use_state_machine_inference
            ):
                raise ValueError(
                    "use_modality_position_ids=True currently supports fixed-chunk streaming inference only"
                )
            if self.core_cfg.use_chunk_local_audio_attn and (
                self.core_cfg.chunk_size <= 0 or use_state_machine_inference
            ):
                raise ValueError(
                    "use_chunk_local_audio_attn=True currently supports fixed-chunk streaming inference only"
                )
            if self.core_cfg.use_two_stream_attn and (
                self.core_cfg.chunk_size <= 0 or use_state_machine_inference
            ):
                raise ValueError(
                    "use_two_stream_attn=True currently supports fixed-chunk streaming inference only"
                )

            if self.core_cfg.chunk_size < 0:
                results = self._generate_offline(
                    audios,
                    n_samples_list,
                    system_prompt,
                    max_new_tokens,
                    generation_config,
                    **generation_kwargs,
                )
            elif self.core_cfg.chunk_size == 0 or use_state_machine_inference:
                # Dynamic chunking (chunk_size=0) or state machine inference opted in for chunk_size > 0.
                # Note that for chunk_size > 0, use_state_machine_inference is not recommended.
                results = self._generate_dynamic_streaming(
                    audios,
                    n_samples_list,
                    system_prompt,
                    max_new_tokens,
                    generation_config,
                    dynamic_min_chunk_size=dynamic_min_chunk_size,
                    dynamic_max_chunk_size=dynamic_max_chunk_size,
                    lm_head_emit_threshold=lm_head_emit_threshold,
                    debug_logs=debug_logs,
                    **generation_kwargs,
                )
            else:
                # Static chunking (chunk_size > 0): bulk prefill + auto-regressive decode.
                results = self._generate_chunked_streaming(
                    audios,
                    n_samples_list,
                    system_prompt,
                    max_new_tokens,
                    generation_config,
                    use_offline_embs=use_offline_embs,
                    inference_audio_chunks_per_turn=inference_audio_chunks_per_turn,
                    parallel_chunk_decode=parallel_chunk_decode,
                    **generation_kwargs,
                )

        return results
