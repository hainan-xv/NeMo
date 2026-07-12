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
from omegaconf import DictConfig, ListConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

from nemo.collections.speechlm2.parts.pretrained import move_embedding
from nemo.utils import logging


def maybe_install_lora(model):
    """Add LoRA adapters to a model, using HuggingFace PEFT library.

    Two convenience keys (not part of PEFT's ``LoraConfig``) are supported under
    ``model.cfg.lora`` and translated into PEFT's ``layers_to_transform`` /
    ``layers_pattern`` before the config is built:

    - ``last_layer_only: bool`` — restrict LoRA to the final decoder layer only.
    - ``last_n_layers: int`` — restrict LoRA to the last N decoder layers.

    These are especially useful for the two-stream (last-layer fusion) model,
    where only the final layer performs cross-modal attention, so LoRA capacity
    is best spent there. If neither key is set, behavior is unchanged (LoRA on
    all layers matched by ``target_modules``).
    """
    if "lora" in model.cfg:
        assert hasattr(model, "cfg") and isinstance(model.cfg, DictConfig)
        assert hasattr(model, "llm") and isinstance(model.llm, PreTrainedModel)
        assert "prevent_freeze_params" in model.cfg and isinstance(model.cfg.prevent_freeze_params, (list, ListConfig))

        lora_kwargs = OmegaConf.to_container(model.cfg.lora, resolve=True)
        # Custom convenience keys -> PEFT layers_to_transform / layers_pattern.
        last_layer_only = bool(lora_kwargs.pop("last_layer_only", False))
        last_n_layers = lora_kwargs.pop("last_n_layers", None)
        if last_layer_only:
            last_n_layers = 1
        if last_n_layers:
            n_layers = int(model.llm.config.num_hidden_layers)
            n = max(1, min(int(last_n_layers), n_layers))
            layer_idxs = list(range(n_layers - n, n_layers))
            lora_kwargs["layers_to_transform"] = layer_idxs
            lora_kwargs.setdefault("layers_pattern", "layers")
            logging.info(
                f"LoRA restricted to decoder layer(s) {layer_idxs} of {n_layers} "
                f"(layers_pattern={lora_kwargs['layers_pattern']})"
            )
        model.lora_config = LoraConfig(**lora_kwargs)
        # PEFT inspects get_input_embeddings() while wrapping the model, so temporarily
        # restore the embedding layer that SALM keeps outside the LLM for FSDP/TP.
        with move_embedding(model):
            model.llm = get_peft_model(model.llm, model.lora_config)
        model.cfg.prevent_freeze_params.append(r"^.+\.lora_.+$")
        logging.info(f"LoRA adapter installed: {model.lora_config}")
