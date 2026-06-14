# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Baseline RNN-T / TDT BPE model that uses a HuggingFace (LLM) tokenizer as its vocabulary.

This is the *plain* transducer baseline: text is treated as a flat sequence of sub-word tokens
coming from a HuggingFace tokenizer (e.g. the Qwen LLM tokenizer), with **no** spelling /
capitalization / punctuation factorization. The only thing that differs from a standard
:class:`EncDecRNNTBPEModel` is the tokenizer: instead of a SentencePiece (``bpe``) or WordPiece
(``wpe``) tokenizer, we plug in :class:`nemo.collections.common.tokenizers.huggingface.AutoTokenizer`.

Everything downstream (prediction network, joint, TDT loss, RNN-T / TDT decoding, WER) is the
standard machinery and is automatically sized to the (large) HuggingFace vocabulary, because
:meth:`EncDecRNNTBPEModel.__init__` derives ``decoder.vocab_size`` / ``joint.num_classes`` from
``self.tokenizer`` right after ``_setup_tokenizer``.

Enable it via the model config::

    tokenizer:
      type: huggingface            # (or 'hf')
      hf_model: Qwen/Qwen3-1.7B     # HF hub name or a local path
      hf_kwargs:                    # optional, forwarded to the HF AutoTokenizer wrapper
        use_fast: true
"""

from typing import Optional

from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.common.tokenizers.huggingface.restricted_auto_tokenizer import RestrictedAutoTokenizer
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging


class EncDecRNNTBPEHFTokenizerModel(EncDecRNNTBPEModel):
    """RNN-T / TDT BPE model whose sub-word vocabulary is a HuggingFace (LLM) tokenizer.

    Behaves exactly like :class:`EncDecRNNTBPEModel` (no factorization), only the tokenizer setup is
    overridden so ``tokenizer.type: huggingface`` loads an HF ``AutoTokenizer`` (e.g. Qwen).
    """

    def _setup_tokenizer(self, tokenizer_cfg: DictConfig):
        tok_type = str(tokenizer_cfg.get('type', '') or '').lower()
        if tok_type not in ('hf', 'huggingface', 'hf_restricted', 'huggingface_restricted'):
            # Anything else (bpe / wpe / agg) -> the standard ASR tokenizer setup.
            super()._setup_tokenizer(tokenizer_cfg)
            return

        restricted = tok_type in ('hf_restricted', 'huggingface_restricted')

        hf_model = tokenizer_cfg.get('hf_model', None) or tokenizer_cfg.get('dir', None)
        if not hf_model:
            raise ValueError(
                "`tokenizer.type=huggingface` requires `tokenizer.hf_model` "
                "(a HuggingFace hub name like 'Qwen/Qwen3-1.7B' or a local path)."
            )

        hf_kwargs = tokenizer_cfg.get('hf_kwargs', {}) or {}
        if not isinstance(hf_kwargs, dict):
            hf_kwargs = OmegaConf.to_container(hf_kwargs, resolve=True)

        # The HF (Restricted)AutoTokenizer wrapper is a TokenizerSpec and exposes the inner HF
        # tokenizer at `.tokenizer`, whose `get_vocab()` is what EncDecRNNTBPEModel.__init__ uses to
        # size the prediction net + joint. RestrictedAutoTokenizer makes that report the COMPACT
        # (English-pruned) vocabulary so the model is sized to the smaller id space.
        if restricted:
            restrict_vocab_file = tokenizer_cfg.get('restrict_vocab_file', None) or tokenizer_cfg.get('dir', None)
            if not restrict_vocab_file:
                raise ValueError(
                    "`tokenizer.type=huggingface_restricted` requires `tokenizer.restrict_vocab_file` "
                    "(the kept-id JSON from scripts/tokenizers/build_restricted_qwen_tokenizer.py)."
                )
            # Save the kept-id artifact into the .nemo so resume / restore re-creates the same vocab.
            restrict_vocab_file = self.register_artifact('tokenizer.restrict_vocab_file', str(restrict_vocab_file))
            self.tokenizer = RestrictedAutoTokenizer(
                pretrained_model_name=str(hf_model), restrict_vocab_file=str(restrict_vocab_file), **hf_kwargs
            )
        else:
            self.tokenizer = AutoTokenizer(pretrained_model_name=str(hf_model), **hf_kwargs)

        # Downstream RNN-T BPE code only branches on `bpe` (sub-word) vs `wpe`; an HF byte-level
        # BPE is sub-word, so advertise it as `bpe`.
        self.tokenizer_type = 'bpe'

        # Keep cfg in sync so a saved .nemo / resume re-instantiates the same HF tokenizer.
        # (`self.cfg` does not exist yet during the first __init__ pass; the guard handles that.)
        if getattr(self, 'cfg', None) is not None and 'tokenizer' in self.cfg:
            self.cfg.tokenizer.type = 'huggingface_restricted' if restricted else 'huggingface'
            self.cfg.tokenizer.hf_model = str(hf_model)

        logging.info(
            f"Initialized {'restricted ' if restricted else ''}HuggingFace (LLM) tokenizer '{hf_model}' "
            f"with {self.tokenizer.vocab_size} tokens for baseline RNN-T/TDT (no factorization)."
        )

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return []
