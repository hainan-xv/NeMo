# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import os
import re
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from _weakref import proxy
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint, _is_local_file_protocol
from lightning.pytorch.trainer import call
from lightning.pytorch.utilities import rank_zero_info

from nemo.collections.common.callbacks import EMA
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.model_utils import ckpt_to_dir, inject_model_parallel_rank, uninject_model_parallel_rank
from nemo.utils.msc_utils import import_multistorageclient, is_multistorageclient_url


class NeMoModelCheckpoint(ModelCheckpoint):
    """Light wrapper around Lightning's ModelCheckpoint to force a saved checkpoint on train_end.
    Extends Lightning's on_save_checkpoint func to save the .nemo file. Saves the .nemo file based
    on the best checkpoint saved (according to the monitor value).
    Also contains func to save the EMA copy of the model.
    """

    UNFINISHED_CHECKPOINT_SUFFIX = "-unfinished"

    def __init__(
        self,
        always_save_nemo: bool = False,
        save_nemo_on_train_end: bool = True,
        save_best_model: bool = False,
        postfix: str = ".nemo",
        n_resume: bool = False,
        model_parallel_size: int = None,
        async_save: bool = False,  # controls only finalize callbacks
        save_last_n_optim_states: int = -1,
        **kwargs,
    ):
        # Parse and store "extended" parameters: save_best model and postfix.
        self.always_save_nemo = always_save_nemo
        self.save_nemo_on_train_end = save_nemo_on_train_end
        self.save_best_model = save_best_model
        self.save_last_n_optim_states = save_last_n_optim_states
        if self.save_best_model and not self.save_nemo_on_train_end:
            logging.warning(
                (
                    "Found save_best_model is True and save_nemo_on_train_end is False. "
                    "Set save_nemo_on_train_end to True to automatically save the best model."
                )
            )
        self.postfix = postfix
        self.previous_best_path = ""
        self.model_parallel_size = model_parallel_size
        self.async_save = async_save
        self.async_finalize_cb = None
        # Checkpoints which removal is deferred until async save is done.
        # Each element of `deferred_ckpts_to_remove` is a growing list
        # that `self._remove_checkpoint` adds to. Once `self._save_checkpoint`
        # is called, the last element is frozen and a new element is added.
        self.deferred_ckpts_to_remove: List[List[str]] = []

        # Number of back-to-back checkpoint writes that stalled/failed. Reset on
        # every successful write; used by the time-boxed save path to stop the
        # run cleanly when the shared filesystem is wedged (see
        # `_bounded_save_checkpoint`).
        self._consecutive_ckpt_write_failures = 0
        # Background threads ferrying node-local checkpoints to the shared FS.
        self._pending_ferries: List = []
        # `.ferrytmp` paths a *currently running* ferry is actively writing. Used to
        # tell live temp files apart from dead orphans when sweeping cruft (see
        # `_sweep_ferry_orphans`); guarded by a lock as ferries run on bg threads.
        import threading as _threading

        self._active_ferry_tmps: set = set()
        self._ferry_lock = _threading.Lock()

        # Node-local staging + background ferry is enabled (see `_bounded_save_checkpoint`).
        # When staging, a newly saved `-last.ckpt` is not yet durable on the shared FS by
        # the time PyTorch-Lightning would delete the *previous* `-last.ckpt`. Eagerly
        # deleting the old one is what caused the durable `-last` to regress to a stale
        # step whenever the new ferry stalled or was killed at the wall-clock limit. So in
        # staging mode we keep the previous `-last` and instead prune to the newest
        # `NEMO_CKPT_KEEP_LAST_N` finished `-last` checkpoints *after* a ferry lands.
        # Default 2 == one durable + one fallback: enough to survive a stalled/killed
        # ferry of the newest `-last` without hoarding many multi-GB copies.
        self._staging_enabled = bool(os.environ.get("NEMO_CKPT_STAGING_DIR", "").strip())
        try:
            self._keep_last_n = max(1, int(os.environ.get("NEMO_CKPT_KEEP_LAST_N", "2")))
        except ValueError:
            self._keep_last_n = 2

        # `prefix` is deprecated
        if 'prefix' in kwargs:
            self.prefix = kwargs.pop('prefix')
        else:
            self.prefix = ""

        # Call the parent class constructor with the remaining kwargs.
        super().__init__(**kwargs)

        if self.save_top_k != -1 and n_resume:
            logging.debug("Checking previous runs")
            self.nemo_topk_check_previous_run()

    def nemo_topk_check_previous_run(self):
        """
        Check if there are previous runs.
        """
        try:
            self.best_k_models
            self.kth_best_model_path
            self.best_model_score
            self.best_model_path
        except AttributeError:
            raise AttributeError("Lightning's ModelCheckpoint was updated. NeMoModelCheckpoint will need an update.")
        self.best_k_models = {}
        self.kth_best_model_path = ""
        self.best_model_score = None
        self.best_model_path = ""

        checkpoints = list(path for path in self._saved_checkpoint_paths if not self._is_ema_filepath(path))
        for checkpoint in checkpoints:
            if 'mp_rank' in str(checkpoint) or 'tp_rank' in str(checkpoint):
                checkpoint = uninject_model_parallel_rank(checkpoint)
            checkpoint = str(checkpoint)
            # second case is for distributed checkpoints, since they are a directory there's no extension
            if checkpoint[-10:] == '-last.ckpt' or checkpoint[-5:] == '-last':
                continue
            index = checkpoint.find(self.monitor) + len(self.monitor) + 1  # Find monitor in str + 1 for '='
            if index != len(self.monitor):
                match = re.search('[A-z]', checkpoint[index:])
                if match:
                    value = checkpoint[index : index + match.start() - 1]  # -1 due to separator hypen
                    self.best_k_models[checkpoint] = float(value)
        if len(self.best_k_models) < 1:
            return  # No saved checkpoints yet

        _reverse = False if self.mode == "min" else True

        best_k_models = sorted(self.best_k_models, key=self.best_k_models.get, reverse=_reverse)

        # This section should be ok as rank zero will delete all excess checkpoints, since all other ranks are
        # instantiated after rank zero. models_to_delete should be 0 for all other ranks.
        if self.model_parallel_size is not None:
            # check for distributed checkpoint
            if checkpoints[0].is_dir():
                models_to_delete = len(best_k_models) - self.save_top_k
            else:
                models_to_delete = len(best_k_models) - self.model_parallel_size * self.save_top_k
        else:
            models_to_delete = len(best_k_models) - self.save_top_k

        models_to_delete = max(0, models_to_delete)
        logging.debug(f'Number of models to delete: {models_to_delete}')

        # If EMA enabled, delete the additional EMA weights
        ema_enabled = self._has_ema_ckpts(self._saved_checkpoint_paths)

        for _ in range(models_to_delete):
            model = best_k_models.pop(-1)
            self.best_k_models.pop(model)
            self._del_model_without_trainer(model)
            if ema_enabled and self._fs.exists(self._ema_format_filepath(model)):
                self._del_model_without_trainer(self._ema_format_filepath(model))
            logging.debug(f"Removed checkpoint: {model}")

        self.kth_best_model_path = best_k_models[-1]
        self.best_model_path = best_k_models[0]
        self.best_model_score = self.best_k_models[self.best_model_path]

    def _remove_invalid_entries_from_topk(self):
        # Removes invalid (incomplete or not existing) checkpoints from topk checkpoints.
        # This might be needed if the checkpointing was abruptly terminated.
        def __is_ckpt_ok(ckpt_path: str) -> bool:
            exists = (
                os.path.isfile(ckpt_path)
                or os.path.isfile(inject_model_parallel_rank(ckpt_path))
                or os.path.isdir(ckpt_path.removesuffix('.ckpt'))
            )
            return exists and not self.is_checkpoint_unfinished(ckpt_path)

        self.best_k_models = {k: v for k, v in self.best_k_models.items() if __is_ckpt_ok(k)}
        if len(self.best_k_models) > 0:
            reverse_arr = self.mode != "min"
            best_k_models_arr = sorted(self.best_k_models, key=self.best_k_models.get, reverse=reverse_arr)
            self.kth_best_model_path = best_k_models_arr[-1]
            self.kth_value = self.best_k_models[self.kth_best_model_path]
            self.best_model_path = best_k_models_arr[0]
            self.best_model_score = self.best_k_models[self.best_model_path]
        else:
            self.kth_best_model_path = ""
            self.kth_value = None
            self.best_model_path = ""
            self.best_model_score = None

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state dict.
        """
        super().load_state_dict(state_dict)
        self._remove_invalid_entries_from_topk()

    def setup(self, trainer, pl_module, stage: str) -> None:
        """
        Setup the checkpoint.
        """
        if is_global_rank_zero():
            logging.debug("Removing unfinished checkpoints if any...")
            NeMoModelCheckpoint._remove_unfinished_checkpoints(self.dirpath)
            # Sweep orphaned ferry temp files. A `*.ferrytmp` is a partial copy left
            # behind when a background ferry (see `_bounded_save_checkpoint`) was killed
            # mid-write (e.g. job hit the wall-clock limit). At setup() no ferry is
            # running yet, so any `*.ferrytmp` present is a stale orphan and safe to drop
            # -- otherwise they pile up (GBs each) across requeues.
            if self._staging_enabled and self.dirpath:
                import glob as _glob

                for _stale in _glob.glob(os.path.join(str(self.dirpath), "*.ferrytmp")):
                    try:
                        os.remove(_stale)
                        logging.info(f"[checkpoint] removed orphaned ferry temp file {_stale}.")
                    except OSError as _e:
                        logging.warning(f"[checkpoint] could not remove {_stale}: {_e!r}")
        # Ensure that all ranks continue with unfinished checkpoints removed
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        super().setup(trainer, pl_module, stage)
        # When using S3 checkpointing, only Rank 0 has the checkpoint and model path set in exp_manager.
        # Sync the values across all ranks to ensure consistency.
        path = trainer.strategy.broadcast(trainer.ckpt_path)
        trainer.ckpt_path = path

        self.last_model_path = trainer.strategy.broadcast(self.last_model_path)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Save the checkpoint.
        """
        output = super().on_save_checkpoint(trainer, pl_module, checkpoint)
        if not self.always_save_nemo:
            return output
        # Load the best model and then re-save it
        app_state = AppState()
        if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
            logging.warning('always_save_nemo will slow down training for model_parallel > 1.')
        # since we are creating tarfile artifacts we need to update .nemo path
        app_state.model_restore_path = self._format_nemo_checkpoint_name()
        if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
            maybe_injected_best_model_path = inject_model_parallel_rank(self.best_model_path)
        else:
            maybe_injected_best_model_path = self.best_model_path

        if self.save_best_model:
            if not os.path.exists(maybe_injected_best_model_path):
                return

            if self.best_model_path == self.previous_best_path:
                logging.debug('Best model has not changed, skipping save.')
                return output

            self.previous_best_path = self.best_model_path
            old_state_dict = deepcopy(pl_module.state_dict())
            checkpoint = torch.load(maybe_injected_best_model_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            # get a new instanace of the model
            pl_module.load_state_dict(checkpoint, strict=True)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            backup_path = self._backup_existing_nemo_ckpt(trainer)
            pl_module.save_to(save_path=app_state.model_restore_path)
            logging.info(f"New best .nemo model saved to: {app_state.model_restore_path}")
            pl_module.load_state_dict(old_state_dict, strict=True)
        else:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            backup_path = self._backup_existing_nemo_ckpt(trainer)
            pl_module.save_to(save_path=app_state.model_restore_path)
            logging.info(f"New .nemo model saved to: {app_state.model_restore_path}")
        if backup_path is not None and is_global_rank_zero():
            logging.info(f'Removing old .nemo backup {backup_path}')
            get_filesystem(backup_path).rm(backup_path)
        return output

    def on_train_end(self, trainer, pl_module):
        """
        Save the checkpoint on train end.
        """
        if trainer.fast_dev_run:
            return None

        # check if we need to save a last checkpoint manually as validation isn't always run based on the interval
        if self.save_last and trainer.val_check_interval != 0:
            should_save_last_checkpoint = False
            if isinstance(trainer.val_check_interval, float) and trainer.val_check_interval % trainer.global_step != 0:
                should_save_last_checkpoint = True
            if isinstance(trainer.val_check_interval, int) and trainer.global_step % trainer.val_check_interval != 0:
                should_save_last_checkpoint = True
            if should_save_last_checkpoint:
                monitor_candidates = self._monitor_candidates(trainer)
                if self.last_model_path == self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST):
                    logging.debug(f'Last checkpoint {self.last_model_path} already saved')
                else:
                    super()._save_last_checkpoint(trainer, monitor_candidates)
        # Call parent on_train_end() to save the -last checkpoint
        super().on_train_end(trainer, pl_module)

        # Load the best model and then re-save it
        if self.save_best_model:
            # wait for all processes
            trainer.strategy.barrier("SaveBestCheckpointConnector.resume_end")
            if self.best_model_path == "":
                logging.warning(
                    f"{self} was told to save the best checkpoint at the end of training, but no saved checkpoints "
                    "were found. Saving latest model instead."
                )
            else:
                if os.path.isdir(self.best_model_path.split('.ckpt')[0]):
                    self.best_model_path = self.best_model_path.split('.ckpt')[0]
                self.best_model_path = trainer.strategy.broadcast(self.best_model_path)
                trainer._checkpoint_connector.restore(self.best_model_path)

        if self.save_nemo_on_train_end:
            backup_path = self._backup_existing_nemo_ckpt(trainer)
            pl_module.save_to(save_path=self._format_nemo_checkpoint_name())
            if backup_path is not None and is_global_rank_zero():
                logging.info(f'Removing old .nemo backup {backup_path}')
                get_filesystem(backup_path).rm(backup_path)

    def _backup_existing_nemo_ckpt(self, trainer) -> Optional[str]:
        """Search for an available name with version infix and rename existing checkpoint.

        NOTE: this behavior is slightly different from regular checkpoints.
        PTL creates new regular checkpoint with the first available name.
        Here, for backward compatibility, we create .nemo checkpoint as before
        and create a backup under the first available name.

        Args:
            trainer (Trainer): trainer instance.

        Returns:
            Path to the backup checkpoint or None, if no backup was created
        """
        base_path = self._format_nemo_checkpoint_name()
        available_path = base_path
        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while self.file_exists(available_path, trainer, check_dist_ckpt=False):
                available_path = self._format_nemo_checkpoint_name(version_cnt)
                version_cnt += 1
        if available_path == base_path:
            # no existing ckpt, no need to backup
            return None
        if trainer.is_global_zero:
            logging.info(f'{base_path} already exists, moving existing checkpoint to {available_path}')
            if is_multistorageclient_url(base_path):
                # TODO: multistorageclient doesn't have "rename" function, therefore no-op but we should
                # refactor this once multistorageclient have rename function supported.
                pass
            else:
                shutil.move(base_path, available_path)
        trainer.strategy.barrier()
        return available_path

    def _format_nemo_checkpoint_name(self, ver: Optional[int] = None) -> str:
        version_infix = '' if ver is None else f'{self.CHECKPOINT_JOIN_CHAR}v{ver}'
        if is_multistorageclient_url(self.dirpath):
            return f"{self.dirpath}/{self.prefix + version_infix + self.postfix}"
        return os.path.abspath(
            os.path.expanduser(os.path.join(self.dirpath, self.prefix + version_infix + self.postfix))
        )

    def _del_model_without_trainer(self, filepath: str) -> None:

        filepath = Path(filepath)

        # check if filepath is a distributed a checkpoint
        if ckpt_to_dir(filepath).is_dir():
            if is_global_rank_zero():
                try:
                    dist_ckpt = ckpt_to_dir(filepath)
                    shutil.rmtree(dist_ckpt, ignore_errors=True)
                    logging.info(f"Removed distributed checkpoint: {dist_ckpt}")
                except:
                    logging.info(f"Tried to remove distributed checkpoint: {dist_ckpt} but failed.")

        else:
            app_state = AppState()

            # legacy model parallel checkpoint
            if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
                # filepath needs to be updated to include mp_rank
                filepath = inject_model_parallel_rank(filepath)

            # each model parallel rank needs to remove its model
            if is_global_rank_zero() or (
                app_state.model_parallel_size is not None and app_state.data_parallel_rank == 0
            ):
                try:
                    self._fs.rm(filepath)
                    logging.info(f"Removed checkpoint: {filepath}")
                except:
                    logging.info(f"Tried to remove checkpoint: {filepath} but failed.")

    def _ema_callback(self, trainer: 'lightning.pytorch.Trainer') -> Optional[EMA]:  # noqa: F821
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback

    def _drop_optimizer_states(self, trainer, filepath: Union[str, Path], storage_options: Optional[Any]) -> None:
        # Get list of saved checkpoints
        checkpoints = self._get_checkpoints_list(filepath)
        suffix = "-no-optim"

        # Drop optimizer states
        checkpoint_index = len(checkpoints) - self.save_last_n_optim_states - 1
        if len(checkpoints) > self.save_last_n_optim_states:
            checkpoint_path = checkpoints[checkpoint_index]

            logging.info(f"Loading '{checkpoint_path}' checkpoint to drop optimizer states...")
            checkpoint = trainer.strategy.load_checkpoint(checkpoint_path=checkpoint_path, load_optimizer_states=False)

            # Load related state dict
            self._load_current_state_dict(trainer, checkpoint)

            # Save the checkpoint without optimizer states
            if storage_options is None:
                storage_options = dict(include_optimizer=False)
            else:
                storage_options["include_optimizer"] = False

            trainer.save_checkpoint(
                f"{checkpoint_path}{suffix}.ckpt", self.save_weights_only, storage_options=storage_options
            )

            # Remove the checkpoint version with optimizer states
            if is_global_rank_zero():
                trainer.strategy.remove_checkpoint(checkpoint_path)
                shutil.move(f"{checkpoint_path}{suffix}", checkpoint_path)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # Load the correct state_dict for current checkpoint.
            # Temporary solution.
            checkpoint = trainer.strategy.load_checkpoint(
                checkpoint_path=ckpt_to_dir(filepath), load_optimizer_states=False
            )
            self._load_current_state_dict(trainer, checkpoint)

            logging.info(f"Successfully dropped optimizer states for '{checkpoint_path}' checkpoint.")

    def _get_checkpoints_list(self, filepath: Union[str, Path]) -> List[str]:
        # Get a checkpoints directory
        checkpoints_dir = os.path.dirname(filepath)

        # Get a list of saved checkpoints
        checkpoints = [
            d
            for d in os.listdir(checkpoints_dir)
            if os.path.isdir(os.path.join(checkpoints_dir, d)) and '-last' not in d
        ]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-step=')[1].split('-')[0]))
        checkpoints = [os.path.join(checkpoints_dir, checkpoint) for checkpoint in checkpoints]

        return checkpoints

    def _load_current_state_dict(self, trainer, checkpoint) -> None:
        # Temporary solution for loading the correct state dict
        # when dropping optimizer states "on the fly" during training.

        # TODO @dimapihtar @mikolajblaz: provide a more elegant solution at the mcore level.

        call._call_lightning_module_hook(trainer, "on_load_checkpoint", checkpoint)

        # Load model state_dict
        trainer.strategy.load_model_state_dict(
            checkpoint,
            strict=trainer.lightning_module.strict_loading,
        )

    @staticmethod
    def format_checkpoint_unfinished_marker_path(checkpoint_path: Union[Path, str]) -> Path:
        """Format the path to the unfinished checkpoint marker file.

        If the marker file exists, corresponding checkpoint is considered unfinished/incomplete.
        NOTE: Marker path for the EMA checkpoint part is the same as for the original checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file or dir.
              Does not need to exist.

        Returns:
            Path to the unfinished checkpoint marker file.
        """
        marker_filepath = str(uninject_model_parallel_rank(checkpoint_path))
        marker_filepath = marker_filepath.removesuffix(".nemo")
        marker_filepath = marker_filepath.removesuffix(".ckpt")
        marker_filepath = marker_filepath.removesuffix("-EMA")
        return Path(marker_filepath + NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX)

    @staticmethod
    def is_checkpoint_unfinished(checkpoint_path: Union[Path, str]) -> bool:
        """Check if the checkpoint is unfinished.

        Args:
            checkpoint_path: Path to the checkpoint file or dir.
              Does not need to exist.

        Returns:
            True if the checkpoint is unfinished, False otherwise.
        """
        return NeMoModelCheckpoint.format_checkpoint_unfinished_marker_path(checkpoint_path).exists()

    @staticmethod
    def set_checkpoint_unfinished_marker(checkpoint_path: Union[Path, str], barrier_after=False) -> None:
        """Marks given checkpoint as unfinished.

        Args:
            checkpoint_filepath: Path to the checkpoint file or dir.
              Does not need to exist.
            barrier_after: Synchronize ranks after writing the marker file.
              Defaults to False.
        """
        if is_global_rank_zero():
            marker_path = NeMoModelCheckpoint.format_checkpoint_unfinished_marker_path(checkpoint_path)
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.touch()
        if barrier_after and torch.distributed.is_initialized():
            torch.distributed.barrier()

    @staticmethod
    def remove_checkpoint_unfinished_marker(checkpoint_path: Union[Path, str], barrier_before=False) -> None:
        """Clear unfinished marker for given checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file or dir.
              Does not need to exist.
            barrier_before: Synchronize ranks before removing the marker file.
              Defaults to False.
        """
        try:
            if barrier_before and torch.distributed.is_initialized():
                torch.distributed.barrier()
            if is_global_rank_zero():
                marker_path = NeMoModelCheckpoint.format_checkpoint_unfinished_marker_path(checkpoint_path)
                if marker_path.exists():
                    marker_path.unlink()
        except:
            return

    def file_exists(
        self, filepath: str, trainer: "lightning.pytorch.Trainer", check_dist_ckpt: bool = True  # noqa: F821
    ) -> bool:
        """Checks if a file or a file without a suffix (distributed checkpoint) exists."""
        if is_multistorageclient_url(filepath):
            exists = self._fs.exists(filepath)
        else:
            exists = self._fs.exists(filepath) or (check_dist_ckpt and self._fs.exists(ckpt_to_dir(filepath)))

        return trainer.strategy.broadcast(exists)

    def _save_checkpoint(self, trainer: 'lightning.pytorch.Trainer', filepath: str) -> None:  # noqa: F821
        # barrier_after=True, so all ranks continue after the unfinished checkpoint marker is placed.
        # if anything goes wrong during checkpointing, we should be able to detect that data is incomplete.
        self.set_checkpoint_unfinished_marker(filepath, barrier_after=True)
        ema_callback = self._ema_callback(trainer)
        if ema_callback is not None:
            if self.async_save:
                raise ValueError('async_save with EMA not supported')
            with ema_callback.save_original_optimizer_state(trainer):
                super()._save_checkpoint(trainer, filepath)

            # save EMA copy of the model as well.
            with ema_callback.save_ema_model(trainer):
                filepath = self._ema_format_filepath(filepath)
                if self.verbose:
                    rank_zero_info(f"Saving EMA weights to separate checkpoint {filepath}")
                super()._save_checkpoint(trainer, filepath)
            self.remove_checkpoint_unfinished_marker(filepath, barrier_before=True)
        else:
            # Async save passed the finalization function to checkpoint_io,
            # sync save calls the finalization function immediately after save.
            finalize_fn = self._get_finalize_save_checkpoint_callback(trainer, filepath, trainer.global_step)
            if self.async_save:
                checkpoint_io = trainer.strategy.checkpoint_io
                if not isinstance(checkpoint_io, AsyncFinalizableCheckpointIO):
                    raise ValueError('Async save requires async compatible CheckpointIO')
                storage_options = dict(finalize_fn=finalize_fn)
                # Each upcoming ckpt removal request will be executed as part of this save finalization
                self.deferred_ckpts_to_remove.append([])
                logging.info(
                    f'Checkpoint save for step {trainer.global_step} started at {datetime.now().isoformat()}.'
                )
                trainer.save_checkpoint(filepath, self.save_weights_only, storage_options=storage_options)
                logging.info(f'Scheduled async checkpoint save for {filepath}')
            else:
                storage_options = None
                # Time-boxed synchronous save: on a flaky shared filesystem a
                # stalled write on rank 0 would otherwise wedge the post-save DDP
                # barrier until the NCCL watchdog kills the whole job. Instead we
                # bound the write with a timeout, skip it on failure (keeping the
                # -unfinished marker so resume ignores the partial file) while all
                # ranks still step through the same barriers, and stop the run
                # cleanly after too many consecutive failures.
                self._bounded_save_checkpoint(trainer, filepath, finalize_fn)

        if self.save_last_n_optim_states >= 0 and '-last' in filepath:
            self._drop_optimizer_states(trainer, filepath, storage_options)

    def _bounded_save_checkpoint(self, trainer, filepath, finalize_fn) -> None:
        """Checkpoint save hardened against a flaky shared filesystem.

        On e.g. Lustre a write can stall for a long time (or forever). Because
        rank 0's stalled write parks it in uninterruptible I/O while the other
        ranks reach the post-save barrier, the NCCL watchdog eventually tears the
        whole job down. Two hardening layers, selected by env vars:

        1. Staging + background ferry (``NEMO_CKPT_STAGING_DIR`` set, recommended):
           rank 0 writes the checkpoint to *node-local* storage (fast, reliable),
           then a background thread ferries it to the real shared-FS path via a
           temp file + atomic rename, clearing the ``-unfinished`` marker only once
           the shared-FS copy is complete. The flaky write is entirely off the
           training critical path — training only ever blocks on the fast local
           write.

        2. Time-boxed direct write (``NEMO_CKPT_STAGING_DIR`` unset): rank 0's
           write to the shared FS runs in a worker thread bounded by
           ``NEMO_CKPT_WRITE_TIMEOUT_S``; on timeout we abandon it and continue,
           leaving the ``-unfinished`` marker so resume skips the partial file.

        In both layers every rank issues the SAME sequence of barriers regardless
        of outcome (so DDP never desyncs), and after ``NEMO_CKPT_MAX_CONSEC_FAILS``
        consecutive failures (default 4 == one epoch of evals here) — or that many
        checkpoints piling up un-ferried — we stop the run cleanly.

        NOTE: we bypass ``trainer.save_checkpoint`` because in Lightning 2.4 it
        ends with ``strategy.barrier("Trainer.save_checkpoint")``; issuing that
        NCCL barrier from a worker thread would be unsafe. The barriers stay on
        the main thread; only the disk write is threaded.
        """
        import threading
        import time

        staging_dir = os.environ.get("NEMO_CKPT_STAGING_DIR", "").strip()
        timeout_s = float(os.environ.get("NEMO_CKPT_WRITE_TIMEOUT_S", "90"))
        max_consec = int(os.environ.get("NEMO_CKPT_MAX_CONSEC_FAILS", "4"))

        # Build the payload on the main thread (state_dict is local for DDP).
        checkpoint = trainer._checkpoint_connector.dump_checkpoint(self.save_weights_only)

        def _barrier():
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

        def _threaded_write(target_path, tag):
            """Run the rank-0 disk write in a worker thread; return True on failure."""
            res = {"exc": None}

            def _w():
                try:
                    trainer.strategy.save_checkpoint(checkpoint, target_path, storage_options=None)
                except BaseException as e:  # noqa: BLE001 - keep the worker from crashing the process
                    res["exc"] = e

            t = threading.Thread(target=_w, name=f"nemo-ckpt-{tag}", daemon=True)
            t.start()
            t.join(timeout_s if timeout_s > 0 else None)
            if t.is_alive():
                logging.error(
                    f'[checkpoint] step {trainer.global_step}: {tag} write did not finish within '
                    f'{timeout_s:.0f}s ({target_path}); abandoning and continuing.'
                )
                return True
            if res["exc"] is not None:
                logging.error(
                    f'[checkpoint] step {trainer.global_step}: {tag} write failed ({target_path}): '
                    f'{repr(res["exc"])}. Continuing.'
                )
                return True
            return False

        # ---- Layer 1: stage locally, ferry to the shared FS in the background ----
        if staging_dir:
            failed = False
            if trainer.is_global_zero:
                run_tag = os.environ.get("SLURM_JOB_ID", "nojob")
                local_run_dir = os.path.join(staging_dir, "nemo_ckpt_staging", run_tag)
                local_path = os.path.join(local_run_dir, os.path.basename(filepath))
                try:
                    os.makedirs(local_run_dir, exist_ok=True)
                except Exception as e:  # noqa: BLE001
                    logging.error(f'[checkpoint] cannot create staging dir {local_run_dir}: {e!r}')
                logging.info(
                    f'Checkpoint save for step {trainer.global_step} started at {datetime.now().isoformat()} '
                    f'(staging to {local_path}, then background ferry to {filepath}).'
                )
                failed = _threaded_write(local_path, "local")

            # Barriers identical on all ranks (see NOTE): match the sync path's
            # strategy.barrier + finalize barrier.
            trainer.strategy.barrier("Trainer.save_checkpoint")
            _barrier()

            if trainer.is_global_zero:
                if not failed:
                    self._consecutive_ckpt_write_failures = 0
                    # Optimistic bookkeeping; resume only ever sees the atomically
                    # renamed (complete) file, so this is safe.
                    self._last_global_step_saved = trainer.global_step
                    self._last_checkpoint_saved = filepath
                    self._launch_ckpt_ferry(trainer, local_path, filepath, trainer.global_step)
                else:
                    # A node-local write failing is rare and does NOT block training,
                    # so just skip this checkpoint and continue. Crucially we must NOT
                    # touch trainer.should_stop here: this branch only runs on rank 0,
                    # and a one-sided stop desyncs DDP (the exact bug that wedged the
                    # job at the ferry-backlog limit).
                    logging.error(
                        f'[checkpoint] step {trainer.global_step}: node-local staging write failed; '
                        f'skipping this checkpoint and continuing.'
                    )
            return

        # ---- Layer 2: time-boxed direct write to the shared FS ----
        logging.info(
            f'Checkpoint save for step {trainer.global_step} started at {datetime.now().isoformat()}'
            + (f' (write timeout {timeout_s:.0f}s).' if timeout_s > 0 else '.')
        )
        failed = _threaded_write(filepath, "direct")

        # Only rank 0 actually writes, so `failed` is set only there. Reduce it
        # (MAX) so EVERY rank makes the same success/failure decision below — a
        # one-sided trainer.should_stop desyncs DDP and wedges the job.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
            flag = torch.tensor([1 if failed else 0], device=dev)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            failed = bool(flag.item())

        # Barrier #1: the one Lightning's trainer.save_checkpoint normally emits.
        trainer.strategy.barrier("Trainer.save_checkpoint")

        if not failed:
            # Success: normal finalizer runs its own barrier and clears the marker.
            finalize_fn()
            self._consecutive_ckpt_write_failures = 0
            return

        # Failure: match the success path's finalize barrier so the collective
        # sequence is identical on every rank, but keep the -unfinished marker.
        _barrier()
        self._register_ckpt_failure(trainer, max_consec)

    def _register_ckpt_failure(self, trainer, max_consec) -> None:
        """Track consecutive checkpoint-write failures; stop cleanly past the limit."""
        self._consecutive_ckpt_write_failures = getattr(self, "_consecutive_ckpt_write_failures", 0) + 1
        logging.error(
            f'[checkpoint] consecutive write failures: {self._consecutive_ckpt_write_failures}/{max_consec}.'
        )
        if self._consecutive_ckpt_write_failures >= max_consec:
            logging.error(
                f'[checkpoint] {self._consecutive_ckpt_write_failures} consecutive checkpoint-write failures '
                f'(>= {max_consec}; ~a full epoch of evals). The shared filesystem looks wedged — stopping '
                f'training cleanly.'
            )
            # Avoid a final .nemo export that would just stall on the same FS.
            self.save_nemo_on_train_end = False
            trainer.should_stop = True

    def _launch_ckpt_ferry(self, trainer, local_path, dest_path, global_step) -> None:
        """Start a background thread copying a node-local checkpoint to the shared FS.

        A stalled shared-FS write parks its ferry thread in uninterruptible I/O, so
        such threads can pile up when the FS is flaky. We NEVER stop training for
        this — a pending ferry does not block the training loop, and stopping here
        would (a) throw away a healthy run over a transient FS blip and (b) risk a
        one-sided trainer.should_stop (this runs on rank 0 only) that desyncs DDP.

        Instead we bound the number of in-flight ferries: once too many are
        outstanding we skip persisting new checkpoints, dropping their node-local
        copy to reclaim space, until the backlog drains. At worst we miss a few
        checkpoints during a shared-FS outage; training keeps going.
        """
        import threading

        max_pending = int(os.environ.get("NEMO_CKPT_MAX_PENDING_FERRIES", "6"))

        # Opportunistically sweep dead ferry cruft before launching a new one. Cheap
        # rank-0 main-thread file ops; only touches temp files no live ferry owns.
        self._sweep_ferry_orphans()

        # Drop references to ferries that have already completed.
        self._pending_ferries = [f for f in getattr(self, "_pending_ferries", []) if f.is_alive()]
        if len(self._pending_ferries) >= max_pending:
            logging.warning(
                f'[checkpoint] {len(self._pending_ferries)} ferries still in flight (>= {max_pending}); the '
                f'shared FS looks backed up. Skipping persistence of step {global_step} and dropping its '
                f'node-local copy ({local_path}). Training continues.'
            )
            try:
                os.remove(local_path)
            except OSError:
                pass
            return

        # Register this ferry's temp file as live so a concurrent sweep won't yank it
        # out from under the copy (the thread clears it again when it finishes).
        with self._ferry_lock:
            self._active_ferry_tmps.add(dest_path + ".ferrytmp")

        t = threading.Thread(
            target=self._ferry_checkpoint_to_shared_fs,
            args=(local_path, dest_path, global_step),
            name=f"nemo-ckpt-ferry-{global_step}",
            daemon=True,
        )
        t.start()
        self._pending_ferries.append(t)

    def _ferry_checkpoint_to_shared_fs(self, local_path, dest_path, global_step) -> None:
        """Copy a node-local checkpoint to the shared FS (temp file + atomic rename).

        Runs in a background thread, so a stalled shared-FS write here never blocks
        training or the DDP collectives. The ``-unfinished`` marker is cleared only
        after the atomic rename, so resume always sees either a complete file or
        none.
        """
        import shutil
        import time

        retries = int(os.environ.get("NEMO_CKPT_FERRY_RETRIES", "240"))
        tmp_path = dest_path + ".ferrytmp"
        try:
            for attempt in range(1, retries + 1):
                try:
                    shutil.copyfile(local_path, tmp_path)
                    os.replace(tmp_path, dest_path)  # atomic on the same filesystem
                    # dest is now complete; clear the -unfinished marker (rank-0, no barrier).
                    self.remove_checkpoint_unfinished_marker(dest_path)
                    try:
                        os.remove(local_path)
                    except OSError:
                        pass
                    logging.info(f'[checkpoint] ferried step {global_step} to {dest_path} (attempt {attempt}).')
                    # Now that a fresh `-last.ckpt` is durable on the shared FS, prune older
                    # `-last` checkpoints (kept intact until now so a stalled/killed ferry
                    # could not leave us without a durable last). Rank-0 file ops only.
                    if str(dest_path).endswith("-last.ckpt"):
                        self._prune_last_checkpoints(self._keep_last_n)
                    return
                except BaseException as e:  # noqa: BLE001
                    logging.warning(
                        f'[checkpoint] ferry attempt {attempt}/{retries} for step {global_step} ({dest_path}) '
                        f'failed: {e!r}; retrying.'
                    )
                    time.sleep(min(30.0, 5.0 * attempt))
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            logging.error(
                f'[checkpoint] gave up ferrying step {global_step} to {dest_path} after {retries} attempts; '
                f'node-local copy left at {local_path}.'
            )
        finally:
            # This ferry no longer owns its temp file, whether it landed, gave up, or
            # errored out -- so a later sweep is free to reclaim any leftover.
            with self._ferry_lock:
                self._active_ferry_tmps.discard(tmp_path)

    def _prune_last_checkpoints(self, keep: int) -> None:
        """Keep only the newest ``keep`` finished ``*-last.ckpt`` files (by step).

        Used in staging mode where `_should_remove_checkpoint` intentionally does not
        delete the previous `-last` (so a durable last always survives even if a ferry
        stalls). This runs from the background ferry thread after a successful landing,
        so it does plain rank-0 file removals with NO DDP barriers (the barrier-based
        `_remove_checkpoint` must never be called off the main thread).
        """
        import glob as _glob
        import re as _re

        try:
            if not is_global_rank_zero() or not self.dirpath:
                return
            last_files = _glob.glob(os.path.join(str(self.dirpath), "*-last.ckpt"))
            # Only consider finished checkpoints (no -unfinished marker).
            last_files = [p for p in last_files if not self.is_checkpoint_unfinished(p)]
            if len(last_files) <= keep:
                return

            def _step(p):
                m = _re.search(r"step=(\d+)", os.path.basename(p))
                return int(m.group(1)) if m else -1

            last_files.sort(key=_step, reverse=True)
            for stale in last_files[keep:]:
                for path in (stale, self._ema_format_filepath(stale)):
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                            logging.info(f'[checkpoint] pruned old last checkpoint {path}.')
                    except OSError as e:
                        logging.warning(f'[checkpoint] could not prune {path}: {e!r}')
        except BaseException as e:  # noqa: BLE001
            logging.warning(f'[checkpoint] pruning old -last checkpoints failed: {e!r}')

    def _sweep_ferry_orphans(self) -> None:
        """Reclaim background-ferry cruft that no live ferry owns.

        A ``<ckpt>.ferrytmp`` is a partial copy written by a background ferry; it is
        renamed onto the final path atomically on success and deleted on give-up. But
        a ferry for an evicted top-k checkpoint, or one whose owning thread died, can
        leave its temp file (and a now-stale ``-unfinished`` marker) behind for the
        rest of the run, quietly wasting GBs. ``setup()`` clears these only on the next
        (re)start, which never happens inside a single long job -- so we also sweep them
        opportunistically here (from ``_launch_ckpt_ferry``, rank-0 main thread, no DDP
        barriers). Temp files a *currently running* ferry still owns are left untouched:
        unlinking one mid-copy would just make that ferry recreate it, and it may still
        land successfully.
        """
        import glob as _glob

        if not self._staging_enabled or not is_global_rank_zero() or not self.dirpath:
            return
        try:
            with self._ferry_lock:
                active = set(self._active_ferry_tmps)
            for tmp in _glob.glob(os.path.join(str(self.dirpath), "*.ferrytmp")):
                if tmp in active:
                    continue  # a live ferry is still writing this one
                try:
                    os.remove(tmp)
                    logging.info(f'[checkpoint] swept orphaned ferry temp {tmp}.')
                except OSError as e:
                    logging.warning(f'[checkpoint] could not sweep {tmp}: {e!r}')
                    continue
                # If the final checkpoint never landed, its unfinished marker is stale.
                dest = tmp[: -len(".ferrytmp")]
                if not os.path.exists(dest):
                    marker = self.format_checkpoint_unfinished_marker_path(dest)
                    try:
                        if os.path.exists(marker):
                            os.remove(marker)
                            logging.info(f'[checkpoint] swept stale unfinished marker {marker}.')
                    except OSError:
                        pass
        except BaseException as e:  # noqa: BLE001
            logging.warning(f'[checkpoint] ferry-orphan sweep failed: {e!r}')

    def _get_finalize_save_checkpoint_callback(
        self, trainer: 'lightning.pytorch.Trainer', filepath: str, global_step: int  # noqa: F821
    ):
        """Creates a callback that can be used to finalize async (and sync) ckpt saves."""

        def _cb():
            logging.debug(f'Finalize callback called for step {global_step}, filepath {filepath}')
            self._last_global_step_saved = global_step
            self._last_checkpoint_saved = filepath

            # notify loggers
            if trainer.is_global_zero:
                for logger in trainer.loggers:
                    logger.after_save_checkpoint(proxy(self))

            # barrier_before=True, so all ranks synchronize before removing the unfinished checkpoint marker
            # we don't want to remove the marker until all checkpointing is done.
            self.remove_checkpoint_unfinished_marker(filepath, barrier_before=True)

            if not self.async_save:
                return

            logging.info(
                f'Async checkpoint save for step {global_step} ({filepath}) finalized successfully at {datetime.now().isoformat()}.'
            )

            # Remove checkpoints marked for removal by `self._remove_checkpoint`
            # For each finalization there is exactly one entry in self.deferred_ckpts_to_remove
            assert self.deferred_ckpts_to_remove
            ckpts_to_remove = self.deferred_ckpts_to_remove.pop(0)
            logging.debug(f'Checkpoints to remove: {ckpts_to_remove}')
            for ckpt_to_remove in ckpts_to_remove:
                self._remove_checkpoint(trainer, ckpt_to_remove, override_async=True)

        return _cb

    def _remove_checkpoint(
        self, trainer: "lightning.pytorch.Trainer", filepath: str, override_async=False  # noqa: F821
    ) -> None:
        """Performs checkpoint removal or deferred removal.

        With async save, `self._remove_checkpoint` is called before the checkpoint
        is actually finished so we can't remove it. Instead we add it to
        `self.deferred_ckpts_to_remove` for future removal.
        """
        if self.async_save and not override_async:
            # Register checkpoint removal in the last (active) checkpoint removal list
            self.deferred_ckpts_to_remove[-1].append(filepath)
            return
        # barrier_after=True, so all ranks continue after the unfinished checkpoint marker is placed.
        # if anything goes wrong during removal, we should be able to detect that data is incomplete.
        self.set_checkpoint_unfinished_marker(filepath, barrier_after=True)
        super()._remove_checkpoint(trainer, filepath)
        ema_callback = self._ema_callback(trainer)
        if ema_callback is not None:
            # remove EMA copy of the state dict as well.
            filepath = self._ema_format_filepath(filepath)
            super()._remove_checkpoint(trainer, filepath)
        # barrier_before=True, so all ranks synchronize before removing the unfinished checkpoint marker
        # we don't want to remove the marker until the checkpoint is actually removed.
        self.remove_checkpoint_unfinished_marker(filepath, barrier_before=True)

    def _ema_format_filepath(self, filepath: str) -> str:
        return filepath.replace(self.FILE_EXTENSION, f'-EMA{self.FILE_EXTENSION}')

    def _has_ema_ckpts(self, checkpoints: Iterable[Path]) -> bool:
        return any(self._is_ema_filepath(checkpoint_path) for checkpoint_path in checkpoints)

    def _is_ema_filepath(self, filepath: Union[Path, str]) -> bool:
        return str(filepath).endswith(f'-EMA{self.FILE_EXTENSION}')

    @property
    def _saved_checkpoint_paths(self) -> Iterable[Path]:
        # distributed checkpoints are directories so we check for them here
        # we filter out unfinished checkpoints, these should be deleted during next cleanup

        if is_multistorageclient_url(self.dirpath):
            msc = import_multistorageclient()
            return msc.glob(f"{self.dirpath}/*.ckpt")
        else:
            dist_checkpoints = [d for d in Path(self.dirpath).glob("*") if d.is_dir()]
        if dist_checkpoints:
            return filter(lambda p: not self.is_checkpoint_unfinished(p), dist_checkpoints)
        else:
            checkpoint_files = [f for f in Path(self.dirpath).rglob("*.ckpt")]
            return filter(lambda p: not self.is_checkpoint_unfinished(p), checkpoint_files)

    @staticmethod
    def _remove_unfinished_checkpoints(checkpoint_dir: Union[Path, str]) -> None:

        # Delete unfinished checkpoints from the filesystems.
        # "Unfinished marker" files are removed as well.

        if not is_global_rank_zero():
            raise AssertionError("_remove_unfinished_checkpoints should run only on rank 0")

        if is_multistorageclient_url(checkpoint_dir):
            msc = import_multistorageclient()
            existing_marker_filepaths = msc.glob(
                f"{checkpoint_dir}*{NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}"
            )
            fs = get_filesystem(checkpoint_dir)
            for ckpt_filepath in existing_marker_filepaths:
                fs.rm(ckpt_filepath)
        else:
            checkpoint_dir = Path(checkpoint_dir)

            existing_marker_filepaths = {
                f.resolve()
                for f in checkpoint_dir.glob(f"*{NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}")
                if f.is_file()
            }

            checkpoint_filepaths = {f.resolve() for f in checkpoint_dir.rglob("*.ckpt") if f.is_file()}
            for ckpt_filepath in checkpoint_filepaths:
                possible_marker_path = NeMoModelCheckpoint.format_checkpoint_unfinished_marker_path(ckpt_filepath)
                if possible_marker_path in existing_marker_filepaths:
                    logging.warning(f'Removing unfinished checkpoint: {ckpt_filepath}')
                    os.remove(ckpt_filepath)

            # some directories might be distributed checkpoints, we remove these if they have a unfinished marker
            all_dirpaths = {d.resolve() for d in checkpoint_dir.glob("*") if d.is_dir()}
            for ckpt_dirpath in all_dirpaths:
                possible_marker_path = NeMoModelCheckpoint.format_checkpoint_unfinished_marker_path(ckpt_dirpath)
                if possible_marker_path in existing_marker_filepaths:
                    logging.warning(f'Removing unfinished dist checkpoint: {ckpt_dirpath}')
                    shutil.rmtree(ckpt_dirpath)

            # delete markers
            for marker_path in existing_marker_filepaths:
                os.remove(marker_path)

    def _should_remove_checkpoint(self, trainer: "pl.Trainer", previous: str, current: str) -> bool:  # noqa: F821
        """Checks if the previous checkpoint should be deleted.
        A checkpoint won't be deleted if any of the cases apply:
        - The previous checkpoint is the same as the current checkpoint (means the old was already overwritten by new)
        - The previous checkpoint is not in the current checkpoint directory and the filesystem is local
        - The previous checkpoint is the checkpoint the Trainer resumed from and the filesystem is local
            and the resumed from checkpoint is not the last checkpoint
        """
        if previous == current:
            return False
        # Staging mode: never eagerly delete the previous `-last.ckpt`. The new one is
        # only staged node-locally at this point and is ferried to the shared FS in the
        # background; if we deleted the old (durable) `-last` now and the ferry later
        # stalled/was killed, the newest durable `-last` would regress to a stale step.
        # Old `-last` checkpoints are pruned (keeping the newest N) only after a ferry
        # actually lands -- see `_prune_last_checkpoints`.
        if (
            self._staging_enabled
            and str(current).endswith("-last.ckpt")
            and str(previous).endswith("-last.ckpt")
        ):
            return False
        if not _is_local_file_protocol(previous):
            return True
        previous = Path(previous).absolute()
        resume_path = Path(trainer.ckpt_path).absolute() if trainer.ckpt_path is not None else None

        if resume_path is not None and previous == resume_path:
            if str(current).endswith("-last.ckpt") and resume_path.name.endswith("-last.ckpt"):
                # delete the previous `-last.ckpt` checkpoint when current saved checkpoint is also `-last.ckpt`,
                # if they're in the same directory
                pass
            else:
                return False
        if self.dirpath is None:
            raise ValueError(f"{self.__class__}.dirpath is None.")
        dirpath = Path(self.dirpath).absolute()
        return dirpath in previous.parents
