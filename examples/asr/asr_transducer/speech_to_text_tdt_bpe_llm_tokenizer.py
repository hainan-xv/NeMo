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

"""Training entrypoint for a BASELINE TDT BPE model that uses an LLM (HuggingFace) tokenizer.

Same idea as the multistream entrypoint, but with NO factorization: text is treated as a plain
sequence of tokens from a HuggingFace tokenizer (e.g. the Qwen LLM tokenizer). The architecture
sub-configs (preprocessor / encoder / decoder / joint / spec_augment / model_defaults / loss /
decoding) are derived AT RUNTIME from a base TDT model (e.g. ``nvidia/parakeet-tdt-0.6b-v2``). The
encoder weights of that base model are warm-started into our model; the prediction network and
joint are trained from scratch, sized to the (large) LLM vocabulary.

Example:
    python speech_to_text_tdt_bpe_llm_tokenizer.py \
        --config-path=../conf/tdt_llm_tokenizer --config-name=tdt_bpe_llm_tokenizer \
        base_model_name=nvidia/parakeet-tdt-0.6b-v2 \
        model.tokenizer.hf_model=Qwen/Qwen3-1.7B \
        model.train_ds.input_cfg=... model.validation_ds.manifest_filepath=...
"""

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.rnnt_bpe_hf_tokenizer_models import EncDecRNNTBPEHFTokenizerModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

# Architecture sub-configs copied verbatim from the base model. `loss` (TDT durations/sigma) and
# `decoding` (TDT greedy) are copied too so the standard RNN-T/TDT machinery is reused as-is. The
# `tokenizer` is intentionally NOT copied -- we substitute the LLM tokenizer instead.
_ARCH_KEYS = ["preprocessor", "encoder", "decoder", "joint", "spec_augment", "model_defaults", "loss", "decoding"]


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


@hydra_runner(config_path="../conf/tdt_llm_tokenizer", config_name="tdt_bpe_llm_tokenizer")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    base_model = _load_base_model(cfg)
    base_cfg = base_model.cfg

    # 1) Inject architecture sub-configs (+ TDT loss / decoding) from the base model. The
    #    prediction-net / joint vocab sizes are recomputed from the (HF/LLM) tokenizer inside
    #    EncDecRNNTBPEModel.__init__, so the 1024-class base joint/decoder get resized automatically.
    arch = {key: OmegaConf.to_container(base_cfg[key], resolve=True) for key in _ARCH_KEYS if key in base_cfg}

    # Capture CLI/yaml overrides under model.joint (e.g. fuse_loss_wer / fused_batch_size, which
    # matter a lot with a ~150k vocab) BEFORE we overwrite cfg.model.joint with the base joint, then
    # re-apply them on top.
    cli_joint = cfg.model.get("joint", None)
    cli_joint = OmegaConf.to_container(cli_joint, resolve=True) if cli_joint is not None else None

    with open_dict(cfg.model):
        for key, val in arch.items():
            cfg.model[key] = val
        if cli_joint:
            for k, v in cli_joint.items():
                cfg.model.joint[k] = v

    # 2) Build trainer / exp_manager.
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # 3) Instantiate the baseline RNN-T/TDT model with the LLM tokenizer.
    model = EncDecRNNTBPEHFTokenizerModel(cfg=cfg.model, trainer=trainer)

    # 4) Encoder-only warm start from the base model.
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
