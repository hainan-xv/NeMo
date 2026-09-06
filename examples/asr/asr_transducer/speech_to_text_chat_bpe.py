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
"""Train a CHAT (chunk-wise attention) transducer.

Identical to ``speech_to_text_rnnt_bpe.py`` except for the model class, which
adds one config field::

    model.loss_type=rnnt              # marginalise over every alignment (default)
    model.loss_type=forced_alignment  # condition on ONE alignment from the cuts

Both settings share the architecture, the 1,024-piece vocabulary and the greedy
chunk-synchronous decoder, so the two runs are directly comparable and a
checkpoint from either can initialise the other.

```sh
python speech_to_text_chat_bpe.py \
    --config-path=../conf/fastconformer/cache_aware_streaming \
    --config-name=nemotron_chat_transducer_granary2 \
    model.loss_type=forced_alignment \
    model.tokenizer.dir=<tokenizer dir> \
    trainer.devices=8
```
"""

import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecCHATBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


@hydra_runner(
    config_path="../conf/fastconformer/cache_aware_streaming", config_name="nemotron_chat_transducer_granary2"
)
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecCHATBPEModel(cfg=cfg.model, trainer=trainer)

    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
