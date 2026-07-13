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

import torch

from nemo.collections.asr.data.audio_to_text_chat_ce import LhotseSpeechToTextChatCEDataset
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.utils import logging


class EncDecRNNTBPEModelChatCE(EncDecRNNTBPEModel):
    """
    CHAT (Chunk-wise Attention Transducer) model trained with an alignment-guided
    cross-entropy loss instead of the alignment-free RNNT loss.

    Each target token is assigned (in the data pipeline) to the encoder chunk that
    contains its word's ending timestamp; the loss then evaluates the RNNTAttJoint
    only along that single forced monotonic path (token + blank steps) and applies
    cross-entropy -- avoiding the [B, T, U, V] joint tensor and forward-backward.

    Only training uses the alignment CE + the alignment-carrying dataset; validation
    and inference are unchanged (standard CHAT greedy transducer decoding), so the
    validation set does not need alignments.
    """

    def _chat_ce_reduction(self) -> str:
        cfg = self.cfg.get("chat_ce", None)
        if cfg is not None and cfg.get("reduction", None) is not None:
            return cfg.get("reduction")
        return self.cfg.get("rnnt_reduction", "mean_volume")

    def _setup_dataloader_from_config(self, config):
        # Training data must carry per-token chunk assignments derived from word
        # alignments; validation/test/transcription use the standard dataset.
        if config.get("use_lhotse") and config.get("use_chat_ce_dataset", False):
            window_stride = self.cfg.preprocessor.get("window_stride", 0.01)
            subsampling_factor = self.cfg.encoder.get("subsampling_factor", 8)
            frame_length_in_secs = float(window_stride) * float(subsampling_factor)
            chat_ce_cfg = self.cfg.get("chat_ce", None)
            num_delay_frames = 0 if chat_ce_cfg is None else int(chat_ce_cfg.get("num_delay_frames", 0))
            chunk_size = int(getattr(self.joint, "chunk_size", -1))
            if chunk_size <= 0:
                raise ValueError(
                    "EncDecRNNTBPEModelChatCE requires a positive joint.chunk_size; "
                    "set model.joint.chunk_size explicitly."
                )
            logging.info(
                f"ChatCE training dataloader: chunk_size={chunk_size}, "
                f"frame_length_in_secs={frame_length_in_secs}, num_delay_frames={num_delay_frames}"
            )
            dataset = LhotseSpeechToTextChatCEDataset(
                tokenizer=self.tokenizer,
                chunk_size=chunk_size,
                frame_length_in_secs=frame_length_in_secs,
                num_delay_frames=num_delay_frames,
            )
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=dataset,
                tokenizer=self.tokenizer,
            )
        return super()._setup_dataloader_from_config(config)

    def training_step(self, batch, batch_nb):
        # ChatCE training batch: (audio, audio_len, tokens, token_len, token_chunk_idx)
        signal, signal_len, transcript, transcript_len, token_chunk_idx = batch

        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        # Prediction network over the full target sequence (with SOS) -> [B, D, U+1]
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        loss_value = self.joint.chat_ce_loss(
            encoder_outputs=encoded,
            decoder_outputs=decoder,
            encoder_lengths=encoded_len,
            targets=transcript,
            target_lengths=transcript_len,
            token_chunk_idx=token_chunk_idx,
            reduction=self._chat_ce_reduction(),
        )
        loss_value = self.add_auxiliary_losses(loss_value)

        tensorboard_logs = {
            'train_loss': loss_value,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }

        # Periodic free-running greedy WER on the training batch (same decode as val).
        if (sample_id + 1) % log_every_n_steps == 0:
            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            _, scores, words = self.wer.compute()
            self.wer.reset()
            tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        self.log_dict(tensorboard_logs)
        return {'loss': loss_value}
