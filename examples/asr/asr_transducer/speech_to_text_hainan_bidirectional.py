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

"""Training entrypoint for a bidirectional HAINAN model.

HAINAN (Hybrid-Autoregressive Inference Transducer) extends TDT with stochastic predictor masking:
during training the prediction-network output is randomly zeroed at a subset of text indices, so the
joint also learns to predict from the encoder alone (enabling non-/semi-autoregressive inference).
Crucially, HAINAN adds NO new parameters over a TDT model, so a bidirectional HAINAN has the EXACT
same architecture as the bidirectional TDT. We therefore warm-start the ENTIRE model (shared encoder +
forward AND backward (prediction-net, joint) pairs) from a previously trained bidirectional TDT
``.nemo`` and only flip masking on -> the model just needs to learn masked-predictor robustness, which
is far faster than training from a plain TDT.

Example:
    python speech_to_text_hainan_bidirectional.py \
        --config-path=../conf/hainan_bidirectional --config-name=hainan_bidirectional \
        base_model_path=/results/.../Bidirectional-TDT-averaged.nemo \
        model.hainan_predictor_mask_prob=0.5 \
        model.train_ds.input_cfg=... model.validation_ds.manifest_filepath=...
"""

import os
import tempfile

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.bidirectional_tdt_models import EncDecBidirectionalTDTBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

# Architecture sub-configs copied verbatim from the base (bidirectional TDT) model. The forward and
# backward branches share these (decoder/joint), so deriving them once reconstructs both branches.
# `tokenizer` is handled separately (its files are copied out of the base model) because the base
# config stores them as `nemo:` artifact references that only resolve during a `.nemo` restore.
_ARCH_KEYS = [
    "preprocessor",
    "encoder",
    "decoder",
    "joint",
    "spec_augment",
    "model_defaults",
    "loss",
    "decoding",
]


def _load_base_model(cfg):
    """Restore the base bidirectional TDT model from a local .nemo path (or pretrained name)."""
    base_path = cfg.get("base_model_path", None)
    base_name = cfg.get("base_model_name", None)
    if base_path:
        logging.info(f"Restoring base bidirectional TDT model from local file: {base_path}")
        return ASRModel.restore_from(restore_path=base_path, map_location="cpu")
    if base_name:
        logging.info(f"Restoring base model from pretrained name: {base_name}")
        return ASRModel.from_pretrained(model_name=base_name, map_location="cpu")
    raise ValueError(
        "Provide `base_model_path` pointing at a bidirectional TDT .nemo (the model you just trained)."
    )


def _build_tokenizer_cfg_from_base(base_model):
    """Build a tokenizer config that reuses the base model's SentencePiece tokenizer.

    The base model's ``cfg.tokenizer`` stores ``nemo:<hash>`` artifact references that only resolve
    while restoring from a ``.nemo``, and the resolved temp files are cleaned up once ``restore_from``
    returns. So we reconstruct the SPE model from the (fully loaded) in-memory SentencePiece processor
    and regenerate a ``vocab.txt`` from its pieces, writing both into a stable directory.
    """
    base_type = str(base_model.cfg.tokenizer.get("type", "bpe")).lower()
    if base_type != "bpe":
        raise ValueError(f"Expected a `bpe` (SentencePiece) tokenizer in the base model, got `{base_type}`.")

    tokenizer_wrapper = getattr(base_model, "tokenizer", None)
    spm = getattr(tokenizer_wrapper, "tokenizer", None)  # underlying sentencepiece.SentencePieceProcessor
    if spm is None or not hasattr(spm, "serialized_model_proto"):
        raise RuntimeError(
            "Base model does not expose a SentencePiece processor; cannot reuse its tokenizer "
            f"(tokenizer type={type(tokenizer_wrapper).__name__})."
        )

    tok_dir = tempfile.mkdtemp(prefix="hainan_bidirectional_tok_")

    model_path = os.path.join(tok_dir, "tokenizer.model")
    with open(model_path, "wb") as f:
        f.write(spm.serialized_model_proto())

    vocab_path = os.path.join(tok_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for i in range(spm.get_piece_size()):
            f.write(f"{spm.id_to_piece(i)}\n")

    logging.info(
        f"Reconstructed base SPE tokenizer into {tok_dir} "
        f"(model_path={model_path}, vocab_size={spm.get_piece_size()})."
    )
    return {"dir": tok_dir, "type": "bpe", "model_path": model_path, "vocab_path": vocab_path}


@hydra_runner(config_path="../conf/hainan_bidirectional", config_name="hainan_bidirectional")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    base_model = _load_base_model(cfg)
    base_cfg = base_model.cfg

    # 1) Inject architecture sub-configs (+ TDT loss / decoding) from the base model, plus the base
    #    SentencePiece tokenizer (copied to a stable dir).
    arch = {key: OmegaConf.to_container(base_cfg[key], resolve=True) for key in _ARCH_KEYS if key in base_cfg}
    tokenizer_cfg = _build_tokenizer_cfg_from_base(base_model)

    with open_dict(cfg.model):
        for key, val in arch.items():
            cfg.model[key] = val
        cfg.model.tokenizer = tokenizer_cfg

        # The forward and backward losses are computed explicitly (not via the fused joint path).
        cfg.model.joint.fuse_loss_wer = False

    # 2) Build trainer / exp_manager.
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # 3) Instantiate the bidirectional HAINAN model (same class as bidirectional TDT; masking is enabled
    #    via model.hainan_predictor_mask_prob).
    model = EncDecBidirectionalTDTBPEModel(cfg=cfg.model, trainer=trainer)
    logging.info(
        f"HAINAN predictor masking probability: {model.hainan_predictor_mask_prob} "
        f"(0.0 == plain bidirectional TDT)."
    )

    # 4) Warm-start the WHOLE model from the base bidirectional TDT (identical architecture -> encoder +
    #    forward AND backward branches all transfer). strict=False tolerates any non-parameter buffer
    #    differences (e.g. decoding helpers).
    if cfg.get("init_from_base", True):
        missing, unexpected = model.load_state_dict(base_model.state_dict(), strict=False)
        logging.info(
            f"Warm-started FULL bidirectional model from base bidirectional TDT "
            f"(missing={len(missing)}, unexpected={len(unexpected)}). "
            f"Both forward and backward branches were transferred; HAINAN adds no new parameters."
        )
        if missing:
            logging.info(f"  Missing keys (kept at init): {missing}")
        if unexpected:
            logging.info(f"  Unexpected keys (ignored): {unexpected}")

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer.fit(model)

    if hasattr(cfg.model, "test_ds") and cfg.model.test_ds.manifest_filepath is not None:
        if model.prepare_test(trainer):
            trainer.test(model)


if __name__ == "__main__":
    main()
