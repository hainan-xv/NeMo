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

"""
Train a CHAT (Chunk-wise Attention Transducer) BPE model with an alignment-guided
cross-entropy loss (EncDecRNNTBPEModelChatCE) instead of the alignment-free RNNT loss.

Requires:
    - model.joint._target_ = nemo.collections.asr.modules.RNNTAttJoint (CHAT joint)
    - a positive model.joint.chunk_size
    - training data with word alignments in cut.custom["alignments"] (Granary v2),
      loaded through lhotse with model.train_ds.use_chat_ce_dataset=true

See oci_chat/chat_ce.sh for a full launch example.
"""

import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models.chat_ce_models import EncDecRNNTBPEModelChatCE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


@hydra_runner(
    config_path="../conf/fastconformer/cache_aware_streaming",
    config_name="fastconformer_chat_transducer_bpe_streaming",
)
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecRNNTBPEModelChatCE(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
