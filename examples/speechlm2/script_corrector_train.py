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
"""Train a SpeechLM to verify/correct a frozen CHAT transducer.

    python examples/speechlm2/script_corrector_train.py \
        --config-path=examples/speechlm2/conf \
        --config-name=streaming_stt_granary2_lora_script_corrector \
        ++model.chat_nemo=/path/to/chat.nemo

THE DATALOADER MUST RETURN CUTS. The corrector derives the reference chunk
partition from the aligner's WORD TIMINGS, which live on the cut -- so the plain
tensor-only loader cannot drive this at all. That is why the loader is built here
rather than through the model's ordinary setup path.
"""

import copy
import os

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.speechlm2.models.script_corrector_model import ScriptCorrectorModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

# Keys that must NEVER be written by a warm start. ``perception.encoder`` IS
# ``chat.encoder`` -- the same module object, shared deliberately -- so loading
# the donor SCRIPT arm's encoder weights would overwrite the FROZEN CHAT encoder
# in place and silently corrupt the very model this run is supposed to verify.
# The corruption would be invisible: shapes match, the load succeeds, and CHAT
# just quietly stops being the model we measured.
SHARED_PREFIXES = ("perception.encoder.", "chat.")


def _init_from_ckpt(model, path: str) -> None:
    """Load WEIGHTS ONLY from a training checkpoint, discarding optimizer state.

    This is an INITIALISATION, not a resume: the step counter, LR schedule and
    optimizer moments are left fresh, so the corrector starts its own schedule
    from step 0 with the donor SCRIPT arm's parameters. The warm start is
    deliberately PARTIAL -- the encoder comes from CHAT, not from the donor.

    A checkpoint matching NOTHING raises rather than no-ops: a silent miss looks
    exactly like a successful warm start while training from scratch, which is
    precisely what this run did before the key was wired up at all.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    src = ckpt.get("state_dict", ckpt)
    tgt = model.state_dict()

    loaded, skipped, missing, shared = {}, [], [], 0
    for k, v in src.items():
        if k.startswith(SHARED_PREFIXES):
            shared += 1
            continue
        if k in tgt and tgt[k].shape == v.shape:
            loaded[k] = v
        elif k in tgt:
            skipped.append(f"{k} {tuple(v.shape)} != {tuple(tgt[k].shape)}")
        else:
            missing.append(k)

    if not loaded:
        raise ValueError(
            f"init_from_ckpt matched ZERO parameters from {path}. "
            f"{len(skipped)} shape mismatches, {len(missing)} keys absent, {shared} shared-encoder keys held back. "
            "Refusing to train from scratch under the guise of a warm start."
        )

    model.load_state_dict(loaded, strict=False)
    logging.info(
        "init_from_ckpt: loaded %d/%d tensors from %s (%d shape-mismatched, %d unknown, %d shared-encoder held back)",
        len(loaded),
        len(tgt),
        path,
        len(skipped),
        len(missing),
        shared,
    )
    for line in skipped[:10]:
        logging.warning("init_from_ckpt: shape mismatch, left at init: %s", line)


@hydra_runner(config_path="conf", config_name="streaming_stt_granary2_lora_script_corrector")
def train(cfg):
    # At module scope this ran on import, which made the module unimportable on a
    # CPU-only box and so untestable.
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    model = ScriptCorrectorModel(OmegaConf.to_container(cfg.model, resolve=True))

    # The launcher passes ++init_from_ckpt; before this existed the key was
    # accepted by Hydra and then read by nobody, so every corrector run so far
    # trained cold.
    init_ckpt = cfg.get("init_from_ckpt", None)
    if init_ckpt:
        _init_from_ckpt(model, str(init_ckpt))

    ds_cfg = cfg.data.train_ds
    with open_dict(ds_cfg):
        ds_cfg.use_lhotse = True
        # Bucketing keys on PACKED length for the generator objective; the
        # corrector's cost is driven by chunk count, and mixing the two would
        # size batches for a budget this model does not have.
        ds_cfg.use_bucketing = bool(ds_cfg.get("use_bucketing", False))
    loader = get_lhotse_dataloader_from_config(
        ds_cfg,
        global_rank=trainer.global_rank,
        world_size=trainer.world_size,
        dataset=LhotseSpeechToTextBpeDataset(tokenizer=model.tokenizer, return_cuts=True),
        tokenizer=model.tokenizer,
    )

    # Validation needs cuts for exactly the same reason training does: the
    # reference chunk partition comes from the aligner's word timings. Built the
    # same way, so val examples are constructed identically to train ones --
    # otherwise val_* would describe a different task from the one being trained.
    val_loader = None
    val_cfg = cfg.data.get("validation_ds")
    if val_cfg is not None:
        vc = copy.deepcopy(val_cfg)
        with open_dict(vc):
            vc.use_lhotse = True
            vc.shuffle = False
            # SCRIPT's validation_ds is NESTED -- validation_ds.datasets.<name>.
            # manifest_filepath -- because that model validates on several named
            # sets. The lhotse loader wants a FLAT manifest_filepath and fails
            # with "You must specify either: manifest_filepath, cuts_path, or
            # shar_path", which the try/except below then turns into a silent
            # training-only run. Flatten to the first entry.
            if "datasets" in vc and vc.get("manifest_filepath") is None:
                first = next(iter(vc.datasets.values()))
                vc.manifest_filepath = first.manifest_filepath
                logging.info("validation: flattened datasets -> %s", vc.manifest_filepath)
                del vc["datasets"]
        try:
            val_loader = get_lhotse_dataloader_from_config(
                vc,
                global_rank=trainer.global_rank,
                world_size=trainer.world_size,
                dataset=LhotseSpeechToTextBpeDataset(tokenizer=model.tokenizer, return_cuts=True),
                tokenizer=model.tokenizer,
            )
        except Exception as e:
            # Loud, and on stdout: the previous run degraded to training-only with
            # this warning buried in stderr, so val_* was simply absent from wandb
            # with no visible reason.
            msg = f"NO VALIDATION LOADER ({e}); training WITHOUT val_* metrics"
            print("=" * 78 + f"\n!! {msg}\n" + "=" * 78, flush=True)
            logging.warning(msg)

    trainer.fit(model, train_dataloaders=loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    train()
