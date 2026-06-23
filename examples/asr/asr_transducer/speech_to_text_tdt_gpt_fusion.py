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

"""Training entrypoint for a jointly trained GPT-LM + TDT model with detached log-linear fusion.

The architecture sub-configs (preprocessor / encoder / decoder / joint / spec_augment /
model_defaults / loss / decoding / tokenizer) are derived AT RUNTIME from a base TDT model
(e.g. ``nvidia/parakeet-tdt-0.6b-v2``). The prediction-network and joint ``_target_`` are swapped to
the fusion-aware modules, and a GPT LM config is injected into the decoder. Because the SPE-1024
vocabulary is unchanged, the encoder, prediction network and joint are all warm-started from the base
model; the GPT LM is warm-started from a HuggingFace GPT-2 backbone (blocks only).

Example:
    python speech_to_text_tdt_gpt_fusion.py \
        --config-path=../conf/tdt_gpt_fusion --config-name=tdt_gpt_fusion \
        base_model_name=nvidia/parakeet-tdt-0.6b-v2 \
        model.joint.lm_fusion_alpha=1.0 model.lm_loss_weight=1.0 \
        model.train_ds.input_cfg=... model.validation_ds.manifest_filepath=...
"""

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.tdt_gpt_fusion_models import EncDecTDTGPTFusionModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

# Architecture sub-configs copied verbatim from the base TDT model. `loss` (TDT durations/sigma),
# `decoding` (TDT greedy) and `tokenizer` (SPE-1024) are copied too so the standard TDT machinery and
# vocabulary are reused as-is.
_ARCH_KEYS = [
    "preprocessor",
    "encoder",
    "decoder",
    "joint",
    "spec_augment",
    "model_defaults",
    "loss",
    "decoding",
    "tokenizer",
]

_FUSION_DECODER_TARGET = "nemo.collections.asr.modules.tdt_gpt_fusion.TDTGPTFusionDecoder"
_FUSION_JOINT_TARGET = "nemo.collections.asr.modules.tdt_gpt_fusion.TDTGPTFusionJoint"


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


@hydra_runner(config_path="../conf/tdt_gpt_fusion", config_name="tdt_gpt_fusion")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    base_model = _load_base_model(cfg)
    base_cfg = base_model.cfg

    # Capture the fusion-specific overrides from our YAML BEFORE the base arch configs overwrite them.
    fusion_gpt_lm = OmegaConf.to_container(cfg.model.decoder.gpt_lm, resolve=True)
    fusion_alpha = float(cfg.model.joint.get("lm_fusion_alpha", 1.0))

    # 1) Inject architecture sub-configs (+ TDT loss / decoding / tokenizer) from the base model.
    arch = {key: OmegaConf.to_container(base_cfg[key], resolve=True) for key in _ARCH_KEYS if key in base_cfg}

    with open_dict(cfg.model):
        for key, val in arch.items():
            cfg.model[key] = val

        # 2) Swap prediction net + joint to the fusion-aware modules and re-apply fusion fields.
        cfg.model.decoder._target_ = _FUSION_DECODER_TARGET
        cfg.model.decoder.gpt_lm = fusion_gpt_lm
        cfg.model.joint._target_ = _FUSION_JOINT_TARGET
        cfg.model.joint.lm_fusion_alpha = fusion_alpha
        # The fusion model uses the non-fused joint path (1024-class vocab fits) so the joint can add
        # the LM term; make sure the (copied) base joint config does not request fused loss/wer.
        cfg.model.joint.fuse_loss_wer = False

        # 3) Greedy batched TDT decoding with CUDA graphs OFF (a neural LM step is not graph-friendly).
        if cfg.model.get("decoding", None) is None:
            cfg.model.decoding = {}
        cfg.model.decoding.strategy = "greedy_batch"
        if cfg.model.decoding.get("greedy", None) is None:
            cfg.model.decoding.greedy = {}
        cfg.model.decoding.greedy.use_cuda_graph_decoder = False

    # 4) Build trainer / exp_manager.
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # 5) Instantiate the fusion model (builds the fusion decoder/joint + GPT LM from cfg).
    model = EncDecTDTGPTFusionModel(cfg=cfg.model, trainer=trainer)

    # 6) Warm start encoder + prediction net (LSTM) + joint from the base TDT (vocab matches). The
    #    GPT LM (decoder.lm.*) is left as initialized (gpt2 blocks + fresh embedding/head).
    if cfg.get("init_tdt_from_base", True):
        enc_missing, enc_unexpected = model.encoder.load_state_dict(base_model.encoder.state_dict(), strict=False)
        dec_missing, dec_unexpected = model.decoder.load_state_dict(base_model.decoder.state_dict(), strict=False)
        joint_missing, joint_unexpected = model.joint.load_state_dict(base_model.joint.state_dict(), strict=False)
        logging.info(
            f"Warm-started from base TDT: "
            f"encoder(missing={len(enc_missing)}, unexpected={len(enc_unexpected)}), "
            f"decoder(missing={len(dec_missing)}, unexpected={len(dec_unexpected)}), "
            f"joint(missing={len(joint_missing)}, unexpected={len(joint_unexpected)}). "
            f"Decoder 'missing' keys are expected to be the GPT LM (decoder.lm.*)."
        )

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer.fit(model)

    if hasattr(cfg.model, "test_ds") and cfg.model.test_ds.manifest_filepath is not None:
        if model.prepare_test(trainer):
            trainer.test(model)


if __name__ == "__main__":
    main()
