# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""GPT-style causal language model over the ASR token vocabulary.

This module borrows the attention (Q/K/V/O) + feed-forward (and positional / final-norm)
weights from a pretrained HuggingFace GPT-2 backbone, but uses a FRESH token embedding and a
FRESH output head sized to the ASR vocabulary. The pretrained model's own tokenizer / vocabulary
(``wte`` token embedding and ``lm_head``) are intentionally discarded -- only the generic
sequence-modeling priors in the transformer blocks transfer.

Index ``vocab_size`` is reserved as the LM begin-of-sequence (BOS) token in the embedding, so the
embedding has ``vocab_size + 1`` rows while the head predicts only the ``vocab_size`` real tokens.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.utils import logging


class GPTLanguageModel(nn.Module):
    """Causal GPT-2-style LM with a fresh embedding/head sized to ``vocab_size``.

    Args:
        vocab_size: number of real (non-blank) ASR tokens. The embedding gets one extra row
            (index ``vocab_size``) used as the LM BOS token; the head predicts ``vocab_size`` logits.
        hf_model: HuggingFace GPT-2 model name (e.g. ``gpt2``) whose transformer blocks (and learned
            positional embeddings + final layer norm) are reused. May be ``None`` only when
            ``init_from_pretrained=False`` and ``n_layer/n_embd/n_head`` are provided.
        init_from_pretrained: if True, load pretrained GPT-2 weights for the backbone; otherwise the
            backbone is randomly initialized (useful for quick local sanity checks without download).
        max_ctx: maximum number of history tokens kept during decoding (must be <= GPT-2 n_positions).
        n_layer, n_embd, n_head: optional architecture overrides used when not loading pretrained
            weights (or to tweak a from-pretrained config).
    """

    def __init__(
        self,
        vocab_size: int,
        hf_model: Optional[str] = "gpt2",
        init_from_pretrained: bool = True,
        max_ctx: int = 1024,
        n_layer: Optional[int] = None,
        n_embd: Optional[int] = None,
        n_head: Optional[int] = None,
    ):
        super().__init__()
        from transformers import GPT2Config, GPT2Model

        if init_from_pretrained:
            if not hf_model:
                raise ValueError("`init_from_pretrained=True` requires a `hf_model` name (e.g. 'gpt2').")
            backbone = GPT2Model.from_pretrained(hf_model)
        else:
            if hf_model:
                config = GPT2Config.from_pretrained(hf_model)
            else:
                config = GPT2Config()
            if n_layer is not None:
                config.n_layer = n_layer
            if n_embd is not None:
                config.n_embd = n_embd
            if n_head is not None:
                config.n_head = n_head
            backbone = GPT2Model(config)

        # We always feed `inputs_embeds`, so the backbone's own token embedding (`wte`) is never used.
        # Replace it with an Identity so its (large) parameters are not registered -- otherwise they
        # would receive no gradient and trigger DDP unused-parameter errors / waste checkpoint space.
        backbone.wte = nn.Identity()

        self.backbone = backbone
        hidden = self.backbone.config.n_embd
        self.hidden_size = hidden
        self.vocab_size = vocab_size
        self.bos_id = vocab_size
        self.max_ctx = min(max_ctx, self.backbone.config.n_positions)

        self.embed = nn.Embedding(vocab_size + 1, hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)

        logging.info(
            f"Initialized GPTLanguageModel (hidden={hidden}, layers={self.backbone.config.n_layer}, "
            f"vocab={vocab_size}, bos_id={self.bos_id}, max_ctx={self.max_ctx}, "
            f"pretrained={init_from_pretrained}, source={hf_model})."
        )

    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the LM over a (right-padded) token sequence.

        Args:
            token_ids: long tensor ``[B, L]`` with ids in ``[0, vocab_size]`` (``vocab_size`` = BOS).
            attention_mask: optional ``[B, L]`` mask (1 = attend, 0 = pad).

        Returns:
            logits over the real vocabulary, shape ``[B, L, vocab_size]``.
        """
        inputs_embeds = self.embed(token_ids)
        out = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = self.head(out.last_hidden_state)
        return logits
