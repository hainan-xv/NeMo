# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Training entrypoint for the Chunkwise-Aligner BASELINE (external alignment; BPE).

This is the published "Chunkwise Aligners for Streaming Speech Recognition"
(arXiv:2605.11422) baseline that the alignment-free Chunked Aligner is compared
against. The model is architecturally identical to the Chunked Aligner (same
encoder + RNN-T prediction net + joint + greedy chunked decoding); the ONLY
difference is the training objective: a FROZEN external CTC model force-aligns
each label to a frame, the label->chunk assignment is fixed from that alignment,
and the trainee maximizes the probability of that single path.

Requirements:
  * A frozen external CTC model sharing this model's tokenizer, provided via
    ``model.external_aligner.model_path`` (a local .nemo) or
    ``model.external_aligner.pretrained_name``.

Two setup modes are supported (mirroring ``speech_to_text_chunked_aligner_bpe.py``):

1. From-scratch / config-driven:
   Provide ``model.tokenizer.dir`` (+ encoder/preprocessor in the YAML) and the
   external aligner.

2. Warm-started from a base TDT/RNN-T model: pass ``base_model_name`` or
   ``base_model_path``; the preprocessor / encoder / spec_augment sub-configs and
   the BPE tokenizer are derived from the base model and the encoder weights are
   warm-started (encoder-only).

Example:
    python speech_to_text_chunkwise_aligner_bpe.py \
        --config-path=../conf/aligner --config-name=chunkwise_aligner_encoder_bpe \
        model.external_aligner.pretrained_name=stt_en_fastconformer_ctc_large \
        model.chunked_aligner.chunk_size=12 \
        model.tokenizer.dir=... \
        model.train_ds.manifest_filepath=... model.validation_ds.manifest_filepath=...
"""

import os
import tempfile

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

# Encoder-side architecture sub-configs copied verbatim from the base model so
# that an encoder-only warm start is guaranteed to line up.
_ARCH_KEYS = ["preprocessor", "encoder", "spec_augment"]


def _load_base_model(cfg):
    """Restore the base model from a local .nemo path or a pretrained name."""
    base_path = cfg.get("base_model_path", None)
    base_name = cfg.get("base_model_name", None)
    if base_path:
        logging.info(f"Restoring base model from local file: {base_path}")
        return EncDecRNNTBPEModel.restore_from(restore_path=base_path, map_location="cpu")
    if base_name:
        logging.info(f"Restoring base model from pretrained name: {base_name}")
        return EncDecRNNTBPEModel.from_pretrained(model_name=base_name, map_location="cpu")
    return None


def _export_tokenizer(base_model, out_dir: str) -> str:
    """Export the base model's SentencePiece tokenizer to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    sp = base_model.tokenizer.tokenizer  # underlying SentencePieceProcessor

    model_file = os.path.join(out_dir, "tokenizer.model")
    with open(model_file, "wb") as f:
        f.write(sp.serialized_model_proto())

    n = sp.get_piece_size()
    with open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf-8") as fv, open(
        os.path.join(out_dir, "tokenizer.vocab"), "w", encoding="utf-8"
    ) as fsv:
        for i in range(n):
            piece = sp.id_to_piece(i)
            fv.write(piece + "\n")
            try:
                score = sp.get_score(i)
            except Exception:
                score = 0.0
            fsv.write(f"{piece}\t{score}\n")

    logging.info(f"Exported base tokenizer ({n} pieces) to {out_dir}")
    return out_dir


@hydra_runner(config_path="../conf/aligner", config_name="chunkwise_aligner_encoder_bpe")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    base_model = _load_base_model(cfg)

    if base_model is not None:
        base_cfg = base_model.cfg

        # 1) Inject encoder-side architecture sub-configs from the base model.
        arch = {key: OmegaConf.to_container(base_cfg[key], resolve=True) for key in _ARCH_KEYS if key in base_cfg}
        with open_dict(cfg.model):
            for key, val in arch.items():
                cfg.model[key] = val

        # 2) Export + point at the base tokenizer (unless a real tokenizer dir is given).
        tok = cfg.model.get("tokenizer", None)
        have_tok_dir = (
            tok is not None
            and not OmegaConf.is_missing(tok, "dir")
            and tok.get("dir", None) not in (None, "")
        )
        if not have_tok_dir:
            tok_dir = _export_tokenizer(
                base_model, cfg.get("tokenizer_out_dir", None) or tempfile.mkdtemp(prefix="chunkwise_aligner_tok_")
            )
            with open_dict(cfg.model):
                cfg.model.tokenizer = {"dir": tok_dir, "type": "bpe"}

    # 2b) Apply explicit encoder overrides ON TOP of the (possibly base-injected) encoder config.
    enc_overrides = cfg.get("encoder_overrides", None)
    if enc_overrides is not None and "encoder" in cfg.model:
        with open_dict(cfg.model):
            cfg.model.encoder = OmegaConf.merge(cfg.model.encoder, enc_overrides)
        logging.info(
            "Applied encoder_overrides on top of the encoder config: "
            f"{OmegaConf.to_container(enc_overrides, resolve=True)}"
        )

    # 3) Build trainer / exp_manager.
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # 4) Instantiate the standard RNNT-BPE model with the Chunkwise-Aligner baseline enabled.
    asr_model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)

    # 5) Optional encoder-only warm start from the base model.
    if base_model is not None and cfg.get("init_encoder_from_base", True):
        missing, unexpected = asr_model.encoder.load_state_dict(base_model.encoder.state_dict(), strict=False)
        logging.info(f"Loaded base encoder weights (missing={len(missing)}, unexpected={len(unexpected)}).")

    # Also support the standard NeMo partial-restore hooks (init_from_*).
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    if base_model is not None:
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
