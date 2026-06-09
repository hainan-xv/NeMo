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

"""Training entrypoint for the 3-stream (spelling + capitalization + punctuation) TDT BPE model.

To avoid hand-matching the pretrained encoder architecture, the model's architecture sub-configs
(preprocessor / encoder / decoder / joint / spec_augment / model_defaults) and the BPE tokenizer
are *derived at runtime* from a base TDT model (e.g. ``nvidia/parakeet-tdt-0.6b-v2``). The encoder
weights of that base model are then loaded into our model (encoder-only warm start); the prediction
network, joint and the new capitalization + punctuation streams are trained from scratch.

Example:
    python speech_to_text_multistream_cap_punct_tdt_bpe.py \
        --config-path=../conf/multistream_cap_punct_tdt --config-name=multistream_cap_punct_tdt_bpe \
        base_model_path=/pretrained/parakeet-tdt-0.6b-v2.nemo \
        model.train_ds.input_cfg=... model.validation_ds.manifest_filepath=...
"""

import os
import tempfile

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.multistream_cap_punct_tdt_bpe_models import EncDecMultiStreamCapPunctTDTBPEModel
from nemo.collections.asr.parts.utils.multistream_cap_punct_factorization import DEFAULT_PUNCT_MARKS
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

# architecture sub-configs copied verbatim from the base model. `decoding` is also copied so the
# base RNN-T __init__ can build its (unused) decoding object; our model drives WER via its own
# greedy multistream decoder instead.
_ARCH_KEYS = ["preprocessor", "encoder", "decoder", "joint", "spec_augment", "model_defaults", "decoding"]


def _load_base_model(cfg):
    """Restore the base TDT model from a local .nemo path or a pretrained name."""
    base_path = cfg.get("base_model_path", None)
    base_name = cfg.get("base_model_name", None)
    if base_path:
        logging.info(f"Restoring base TDT model from local file: {base_path}")
        return ASRModel.restore_from(restore_path=base_path, map_location="cpu")
    if base_name:
        logging.info(f"Restoring base TDT model from pretrained name: {base_name}")
        return ASRModel.from_pretrained(model_name=base_name, map_location="cpu")
    raise ValueError("Provide either `base_model_path` (local .nemo) or `base_model_name` (pretrained).")


def _export_tokenizer(base_model, out_dir: str) -> str:
    """Export the base model's SentencePiece tokenizer to `out_dir`."""
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


@hydra_runner(config_path="../conf/multistream_cap_punct_tdt", config_name="multistream_cap_punct_tdt_bpe")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    base_model = _load_base_model(cfg)
    base_cfg = base_model.cfg

    # 1) Inject architecture sub-configs + base loss from the base model.
    arch = {key: OmegaConf.to_container(base_cfg[key], resolve=True) for key in _ARCH_KEYS if key in base_cfg}
    if "loss" in base_cfg:
        arch["loss"] = OmegaConf.to_container(base_cfg["loss"], resolve=True)

    # base TDT durations (live under loss.tdt_kwargs of the base model)
    durations = None
    loss_cfg = base_cfg.get("loss", None)
    if loss_cfg is not None and loss_cfg.get("tdt_kwargs", None) is not None:
        durations = loss_cfg.tdt_kwargs.get("durations", None)
    if durations is not None:
        durations = list(durations)

    md = arch.setdefault("model_defaults", {})
    # allow CLI/yaml to override; otherwise fall back to base/defaults
    cli_md = cfg.model.get("model_defaults", None)
    cli_durations = cli_md.get("tdt_durations", None) if cli_md is not None else None
    cli_num_cap = cli_md.get("num_cap", None) if cli_md is not None else None
    cli_punct_marks = cli_md.get("punct_marks", None) if cli_md is not None else None
    md["tdt_durations"] = list(cli_durations) if cli_durations is not None else (durations or [0, 1, 2, 3, 4])
    md["num_cap"] = int(cli_num_cap) if cli_num_cap is not None else 4
    md["punct_marks"] = list(cli_punct_marks) if cli_punct_marks is not None else list(DEFAULT_PUNCT_MARKS)

    # Capture any CLI/yaml overrides under model.joint (e.g. masking_prob) BEFORE we overwrite
    # cfg.model.joint wholesale with the base model's joint config below, then re-apply them.
    cli_joint = cfg.model.get("joint", None)
    cli_joint = OmegaConf.to_container(cli_joint, resolve=True) if cli_joint is not None else None

    with open_dict(cfg.model):
        for key, val in arch.items():
            cfg.model[key] = val
        # re-apply CLI joint overrides on top of the base joint config
        if cli_joint:
            for k, v in cli_joint.items():
                cfg.model.joint[k] = v

    # 2) Export + point at the base tokenizer.
    tok_dir = _export_tokenizer(base_model, cfg.get("tokenizer_out_dir", tempfile.mkdtemp(prefix="ms_cp_tdt_tok_")))
    with open_dict(cfg.model):
        cfg.model.tokenizer = {"dir": tok_dir, "type": "bpe"}

    # 3) Build trainer / exp_manager.
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # 4) Instantiate our model.
    model = EncDecMultiStreamCapPunctTDTBPEModel(cfg=cfg.model, trainer=trainer)

    # 5) Encoder-only warm start from the base model.
    if cfg.get("init_encoder_from_base", True):
        missing, unexpected = model.encoder.load_state_dict(base_model.encoder.state_dict(), strict=False)
        logging.info(f"Loaded base encoder weights (missing={len(missing)}, unexpected={len(unexpected)}).")

    del base_model
    torch.cuda.empty_cache()

    trainer.fit(model)

    if hasattr(cfg.model, "test_ds") and cfg.model.test_ds.manifest_filepath is not None:
        if model.prepare_test(trainer):
            trainer.test(model)


if __name__ == "__main__":
    main()
