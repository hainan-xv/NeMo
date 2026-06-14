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

"""Training entrypoint for a CHAT (Chunk-wise Attention Transducer) BPE model that uses an LLM
(HuggingFace) tokenizer as its vocabulary.

To avoid a bespoke config file, this REUSES the generic ``conf/asr_finetune/speech_to_text_finetune.yaml``
for the data / optim / trainer / exp_manager / tokenizer sections and synthesizes the CHAT-specific
architecture in code (so the launch script only carries CLI overrides):

  * The acoustic front end (``preprocessor`` / ``encoder`` / ``spec_augment``) is derived from a base
    model and the encoder is warm-started (encoder-only). ``encoder_overrides`` is merged on top so a
    fixed non-causal attention context (e.g. ``[64, 64]``) actually takes effect.
  * The JOINT is :class:`nemo.collections.asr.modules.RNNTAttJoint` (cross-attention over chunks of
    encoder output), with ``chunk_size`` from the CLI. Prediction net + joint are standard RNN-T,
    trained from scratch and sized to the (LLM / restricted-LLM) tokenizer vocabulary.
  * The tokenizer comes from ``model.tokenizer`` (set ``type=huggingface`` for the full Qwen vocab or
    ``type=huggingface_restricted`` + ``restrict_vocab_file`` for the English-pruned compact vocab).

Example:
    python speech_to_text_chat_bpe_llm_tokenizer.py \
        --config-path=../conf/asr_finetune --config-name=speech_to_text_finetune \
        init_from_pretrained_model=nvidia/parakeet-tdt-0.6b-v2 \
        model.tokenizer.type=huggingface_restricted \
        ++model.tokenizer.hf_model=Qwen/Qwen3-1.7B \
        ++model.tokenizer.restrict_vocab_file=/results/qwen3_english_spe_kept_ids.json \
        ++model.joint.chunk_size=16 \
        ++encoder_overrides.att_context_size=[64,64] \
        model.train_ds.manifest_filepath=... model.validation_ds.manifest_filepath=...
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

# Acoustic front end copied verbatim from the base model so an encoder-only warm start lines up.
_ARCH_KEYS = ["preprocessor", "encoder", "spec_augment"]


def _load_base_model(cfg):
    """Restore the base model from a local .nemo path or a pretrained name (finetune-style keys)."""
    base_path = cfg.get("base_model_path", None) or cfg.get("init_from_nemo_model", None)
    base_name = cfg.get("base_model_name", None) or cfg.get("init_from_pretrained_model", None)
    if base_path:
        logging.info(f"Restoring base model from local file: {base_path}")
        return ASRModel.restore_from(restore_path=base_path, map_location="cpu")
    if base_name:
        logging.info(f"Restoring base model from pretrained name: {base_name}")
        return ASRModel.from_pretrained(model_name=base_name, map_location="cpu")
    raise ValueError(
        "Provide a base model via `init_from_pretrained_model` / `base_model_name` (pretrained) "
        "or `init_from_nemo_model` / `base_model_path` (local .nemo)."
    )


def _build_chat_arch(cfg_model, enc_hidden: int):
    """Synthesize the CHAT architecture sub-configs (model_defaults / decoder / joint / loss /
    decoding) in code, honoring any values already present under ``cfg_model`` (e.g. a CLI
    ``++model.joint.chunk_size=16``)."""
    pred_hidden = 640
    joint_hidden = 640
    chunk_size = 16  # default; overridable via ++model.joint.chunk_size

    existing_joint = cfg_model.get("joint", None)
    if existing_joint is not None and existing_joint.get("chunk_size", None) is not None:
        chunk_size = int(existing_joint.get("chunk_size"))

    model_defaults = {"enc_hidden": int(enc_hidden), "pred_hidden": pred_hidden, "joint_hidden": joint_hidden}

    decoder = {
        "_target_": "nemo.collections.asr.modules.RNNTDecoder",
        "normalization_mode": None,
        "random_state_sampling": False,
        "blank_as_pad": True,
        "prednet": {"pred_hidden": pred_hidden, "pred_rnn_layers": 1, "t_max": None, "dropout": 0.2},
    }

    joint = {
        "_target_": "nemo.collections.asr.modules.RNNTAttJoint",
        "log_softmax": None,
        "preserve_memory": False,
        "chunk_size": chunk_size,
        "fuse_loss_wer": False,  # not supported for the CHAT cross-attention joint
        "fused_batch_size": None,
        "jointnet": {
            "encoder_hidden": int(enc_hidden),
            "pred_hidden": pred_hidden,
            "joint_hidden": joint_hidden,
            "activation": "relu",
            "dropout": 0.2,
        },
    }

    loss = {"loss_name": "default", "warprnnt_numba_kwargs": {"fastemit_lambda": 0.0, "clamp": -1.0}}

    decoding = {
        "strategy": "greedy_batch",
        "greedy": {"max_symbols": 10},
        "beam": {
            "beam_size": 2,
            "return_best_hypothesis": False,
            "score_norm": True,
            "tsd_max_sym_exp": 50,
            "alsd_max_target_len": 2.0,
        },
    }

    with open_dict(cfg_model):
        cfg_model.model_defaults = model_defaults
        cfg_model.decoder = decoder
        cfg_model.joint = joint
        cfg_model.loss = loss
        cfg_model.decoding = decoding
        cfg_model.compute_eval_loss = cfg_model.get("compute_eval_loss", True)
        cfg_model.log_prediction = cfg_model.get("log_prediction", True)


@hydra_runner(config_path="../conf/asr_finetune", config_name="speech_to_text_finetune")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    base_model = _load_base_model(cfg)
    base_cfg = base_model.cfg

    # 1) Inject the acoustic front end from the base model.
    arch = {key: OmegaConf.to_container(base_cfg[key], resolve=True) for key in _ARCH_KEYS if key in base_cfg}
    with open_dict(cfg.model):
        for key, val in arch.items():
            cfg.model[key] = val

    # 2) Merge explicit encoder overrides on top (step 1 replaces cfg.model.encoder wholesale).
    enc_overrides = cfg.get("encoder_overrides", None)
    if enc_overrides is not None and "encoder" in cfg.model:
        with open_dict(cfg.model):
            cfg.model.encoder = OmegaConf.merge(cfg.model.encoder, enc_overrides)
        logging.info(
            "Applied encoder_overrides on top of the encoder config: "
            f"{OmegaConf.to_container(enc_overrides, resolve=True)}"
        )

    # 3) Synthesize the CHAT architecture (joint=RNNTAttJoint, standard RNN-T loss / greedy decoding).
    _build_chat_arch(cfg.model, enc_hidden=int(cfg.model.encoder.d_model))

    # 4) Build trainer / exp_manager.
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # 5) Instantiate the CHAT RNN-T model (RNNTAttJoint) with the (restricted) LLM tokenizer.
    model = EncDecRNNTBPEHFTokenizerModel(cfg=cfg.model, trainer=trainer)

    # 6) Encoder-only warm start from the base model (strict=False tolerates the att-context change).
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
