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

"""SCRIPT and CHAT trained jointly on ONE shared encoder.

The two families already agree on everything that matters for sharing: the same
ConformerEncoder donor, the same Qwen3 vocabulary, the same ``forced_alignment``
machinery deriving word->chunk assignments from the same lhotse cuts, and the
same 80 ms encoder frame. They differ only in the HEAD mounted on the encoder:

- SCRIPT  -- projector + Qwen3 LLM, scoring ``p(words_k | text_history, audio_k)``
- CHAT    -- RNNTDecoder + RNNTAttJoint, scoring a banded transducer lattice

so a single encoder forward can feed both, and the encoder receives gradient
from both objectives::

    L = L_script + chat_loss_weight * L_chat

LATENCY IS SHARED, NOT MERELY MATCHED. SCRIPT draws ``chunk_size`` per batch;
that same draw is pushed into ``joint.chunk_size`` before the CHAT loss runs and
into the encoder's look-ahead (``right = chunk - 1``) by the inherited
``_set_encoder_att_context``. Both heads therefore see the identical acoustic
window on every step, which is the whole point of training them together.

WHY THE REAL ``EncDecCHATBPEModel`` IS MOUNTED rather than its loss methods
copied onto this class: ``_banded_loss`` reaches a dozen sibling helpers and
~17 attributes set up by CHAT's own ``__init__`` (band geometry, the alignment
config, the tokenizer's word-prefix convention, the mismatch counters). Copying
them would fork that setup and let the two drift apart silently -- exactly the
failure the joint model exists to rule out. Mounting the real class keeps CHAT's
semantics authoritative; we simply never call its ``forward``.
"""

from collections import defaultdict
from typing import List, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.models.chat_bpe_models import EncDecCHATBPEModel
from nemo.collections.speechlm2.data.script_dataset import ScriptBatch, ScriptSTTDataset
from nemo.collections.speechlm2.models.script_model import ScriptSTTModel
from nemo.collections.speechlm2.parts.metrics.wer import WER
from nemo.collections.speechlm2.parts.script_prompt import apply_text_style
from nemo.utils import logging


class JointChatScriptModel(ScriptSTTModel):
    """:class:`ScriptSTTModel` with a CHAT transducer head on the same encoder."""

    def __init__(
        self,
        cfg: dict,
        forced_aligner=None,
        data_cfg=None,
        val_data_cfg=None,
        dataset_cls=ScriptSTTDataset,
    ) -> None:
        super().__init__(
            cfg,
            forced_aligner=forced_aligner,
            data_cfg=data_cfg,
            val_data_cfg=val_data_cfg,
            dataset_cls=dataset_cls,
        )

        chat_cfg = cfg.get("chat", None) if hasattr(cfg, "get") else None
        if chat_cfg is None:
            raise ValueError(
                "JointChatScriptModel requires a `model.chat` config block holding the CHAT "
                "head (tokenizer / decoder / joint / loss_type / forced_alignment)."
            )
        if not isinstance(chat_cfg, DictConfig):
            chat_cfg = OmegaConf.create(chat_cfg)

        self.chat_loss_weight = float(cfg.get("chat_loss_weight", 1.0))

        # ModelPT only builds dataloaders when these are present; the joint model
        # drives data from the SCRIPT side, so refuse a config that would make the
        # CHAT submodule open its own (silently doubling the input pipeline).
        for key in ("train_ds", "validation_ds", "test_ds"):
            if chat_cfg.get(key, None) is not None:
                raise ValueError(
                    f"model.chat.{key} must be unset: the CHAT head consumes the SCRIPT "
                    "dataloader's batches, and a second data config would build a second, "
                    "differently-shuffled input pipeline."
                )

        self.chat = EncDecCHATBPEModel(chat_cfg, trainer=None)

        # Drop the CHAT submodule's own audio front-end. It is a byte-identical
        # second copy of the shared one; left in place it would be saved in every
        # checkpoint, counted by the optimizer, and -- because it never runs --
        # flagged by DDP as an unused parameter. The shared `self.perception`
        # supplies `encoded` instead (see `_chat_loss`).
        self.chat.encoder = None
        self.chat.preprocessor = None
        self.chat.spec_augmentation = None

        # Capture the SHARED encoder's output as it is produced. A forward hook
        # rather than a plumbed return value because `self.perception(...)` is
        # reached from several call paths in the SCRIPT step (indexed fill,
        # cumsum fill, grouped-branch recompute); hooking the encoder itself
        # covers all of them and cannot fall out of sync with a future one.
        self._cap_enc: Optional[torch.Tensor] = None
        self._cap_enc_len: Optional[torch.Tensor] = None
        self.perception.encoder.register_forward_hook(self._capture_encoder_output)

        # CHAT hypotheses accumulate exactly as SCRIPT's do, so that epoch end can
        # score BOTH with the same WER class over the same references. Using CHAT's
        # own RNNT WER metric instead would reintroduce the normaliser mismatch that
        # once made CHAT read ~0.15 against SCRIPT's ~0.087 on one manifest.
        self._partial_wer_refs_chat = defaultdict(list)
        self._partial_wer_hyps_chat = defaultdict(list)
        # Detached copy of the step's frames, kept for the training-WER decode in
        # on_train_batch_end (by which time backward has freed the graph).
        self._train_enc = None
        self._train_enc_len = None

    def _capture_encoder_output(self, _module, _args, output) -> None:
        """Stash the Conformer's ``(B, D, T)`` output and its lengths."""
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            self._cap_enc, self._cap_enc_len = output[0], output[1]

    def _chat_loss(self, batch: ScriptBatch) -> Optional[torch.Tensor]:
        """Banded transducer loss on the frames the SCRIPT step just encoded."""
        enc, enc_len = self._cap_enc, self._cap_enc_len
        if enc is None or enc_len is None:
            raise RuntimeError(
                "No encoder output was captured for this step. The CHAT head scores the "
                "SAME frames the SCRIPT head consumed, so a missing capture means the two "
                "losses would disagree about the audio rather than merely about the model."
            )
        if batch.cuts is None:
            raise ValueError("batch.cuts is required: the CHAT banded loss reads word alignments from them.")

        # Same latency as the SCRIPT branch this step. `_set_encoder_att_context`
        # has already pinned the encoder's look-ahead to `chunk - 1`; this pins
        # the transducer's chunking to the identical grid.
        self.chat.joint.chunk_size = int(batch.chunk_size)
        return self.chat._banded_loss(enc, enc_len, batch.cuts)

    def _training_step_inner(self, batch: ScriptBatch, batch_idx: int):
        # DISARM DDP's REDUCER, UNCONDITIONALLY, BEFORE THE FORWARD.
        #
        # ScriptSTTModel.backward() always calls _allreduce_grads() under DDP -- a
        # single flat, rank-invariant reduction over every trainable parameter. But
        # it only disarms DDP's own bucket reducer on the BANDED path, so with
        # loss_type='forced' both mechanisms run: DDP fires per-bucket all-reduces
        # during backward (arming happens inside DDP.forward, so the no_sync() around
        # backward is too late -- see _disarm_ddp_reducer's docstring) and the manual
        # reduction follows.
        #
        # SCRIPT alone tolerated that. Appending the 203M CHAT head does not: it
        # changes DDP's bucket boundaries and the order buckets become ready, so the
        # two collective streams can be issued in different relative orders on
        # different ranks. Job 21236435 ran 1371 healthy steps and then every rank
        # stalled on the SAME WorkNCCL(ALLREDUCE, NumelIn=151670) -- 151670 being
        # exactly this head's vocabulary, blank included.
        #
        # Must precede the forward, for the reason the docstring gives.
        self._disarm_ddp_reducer()

        # Clear first: a stale tensor from the previous step would be graph-free
        # (backward already ran) and would fail with a confusing autograd error
        # rather than pointing at the missing capture.
        self._cap_enc = self._cap_enc_len = None

        out = super()._training_step_inner(batch, batch_idx)
        script_loss = out["loss"]

        chat_loss = self._chat_loss(batch)
        loss = script_loss + self.chat_loss_weight * chat_loss

        # Only when the WER decode is actually due; holding these every step would
        # pin a (B, D, T) tensor for the whole run for nothing.
        every = int(getattr(self.core_cfg, "train_wer_every_n_steps", 0) or 0)
        if every > 0 and (self.global_step + 1) % every == 0:
            self._train_enc = self._cap_enc.detach()
            self._train_enc_len = self._cap_enc_len.detach()
        else:
            self._train_enc = self._train_enc_len = None

        self.log_dict(
            {
                "train_loss_script": script_loss.detach(),
                "train_loss_chat": chat_loss.detach(),
                "train_loss_joint": loss.detach(),
            },
            on_step=True,
        )
        out["loss"] = loss
        return out

    # ------------------------------------------------------------------
    # CHAT-side WER (training and validation)
    # ------------------------------------------------------------------

    def _chat_decode(self, enc, enc_len, chunk: int) -> List[str]:
        """Greedy CHAT decode of ``(B, D, T)`` encoder frames into strings."""
        self.chat.joint.chunk_size = int(chunk)
        with torch.no_grad():
            out = self.chat.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=enc, encoded_lengths=enc_len, return_hypotheses=False
            )
        hyps = out[0] if isinstance(out, tuple) else out
        return [(getattr(h, "text", None) or "") if not isinstance(h, str) else h for h in hyps]

    def _encode_for_chat(self, batch, chunk: int):
        """Encoder frames at ``chunk``'s look-ahead, as ``(B, D, T)`` + lengths.

        Re-runs the shared encoder rather than reusing whatever the SCRIPT decode
        left behind: generate() re-tunes the look-ahead as it walks the chunks, so
        the captured frames are not guaranteed to be the ones a CHAT decode at
        this operating point should see.
        """
        self._set_encoder_att_context(int(chunk))
        self._cap_enc = self._cap_enc_len = None
        with torch.no_grad():
            self.perception(input_signal=batch.audios, input_signal_length=batch.audio_lens)
        return self._cap_enc, self._cap_enc_len

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        # Same cadence and same bounded sample as SCRIPT's training_batch_wer, and
        # on the SAME frames this step already encoded -- so the two numbers differ
        # only by the head that decoded them.
        every = int(getattr(self.core_cfg, "train_wer_every_n_steps", 0) or 0)
        if every <= 0 or (self.global_step + 1) % every != 0:
            return
        enc, enc_len = self._train_enc, self._train_enc_len
        if enc is None or enc_len is None or batch.cuts is None:
            return
        try:
            n = max(int(getattr(self.core_cfg, "train_wer_max_utts", 4) or 4), 1)
            hyps = self._chat_decode(enc[:n], enc_len[:n], int(batch.chunk_size))
            refs = [self._cut_text(c) for c in list(batch.cuts)[:n]]
            metric = WER(normalize=True, verbose=False)
            metric.update("train", refs=refs, hyps=hyps)
            for name, value in metric.compute().items():
                if name == "wer":
                    self.log("training_batch_wer_chat", value.to(self.device), on_step=True)
        except Exception as e:  # never let a metric kill a training step
            logging.warning("training_batch_wer_chat skipped: %r", e)

    @staticmethod
    def _cut_text(cut) -> str:
        return " ".join(s.text for s in (cut.supervisions or []) if s.text)

    def _eval_step(self, batch, name: str, batch_idx: int = 0) -> None:
        super()._eval_step(batch, name, batch_idx)
        if batch.cuts is None:
            return
        try:
            chunk = int(self.val_chunk_size)
            enc, enc_len = self._encode_for_chat(batch, chunk)
            if enc is None:
                return
            # The SAME references SCRIPT is scored against, restyled identically --
            # otherwise the two heads are measured against different strings and the
            # comparison this model exists to support is meaningless.
            refs = list(batch.text)
            if self.core_cfg.prompt_control:
                refs = [
                    apply_text_style(r, self.core_cfg.val_capitalization, self.core_cfg.val_punctuation) for r in refs
                ]
            hyps = self._chat_decode(enc, enc_len, chunk)
            self._partial_wer_refs_chat[name].extend(refs)
            self._partial_wer_hyps_chat[name].extend(hyps)
            if batch_idx % self.core_cfg.log_every_n_steps == 0 and refs and hyps:
                logging.info("[%s] CHAT decode batch %d\n  ref: `%s`\n  hyp: `%s`", name, batch_idx, refs[0], hyps[0])
        except Exception as e:
            logging.warning("val_wer_chat skipped for batch %d: %r", batch_idx, e)

    def on_validation_epoch_end(self) -> None:
        # SCRIPT's metrics first; it clears its own buffers at the end.
        local = {
            name: {"refs": self._partial_wer_refs_chat[name], "hyps": self._partial_wer_hyps_chat[name]}
            for name in self._partial_wer_refs_chat
        }
        super().on_validation_epoch_end()

        if torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        else:
            gathered = [local]

        wer = WER(normalize=True, verbose=False)
        wer_verbatim = WER(normalize=False, verbose=False)
        has_data = False
        for rank_data in gathered:
            for name, values in rank_data.items():
                has_data = has_data or bool(values["refs"])
                wer.update(name, refs=values["refs"], hyps=values["hyps"])
                wer_verbatim.update(name, refs=values["refs"], hyps=values["hyps"])
        if has_data:
            for metric_name, metric_value in wer.compute().items():
                log_name = "val_wer_chat" if metric_name == "wer" else f"val_{metric_name}_chat"
                self.log(log_name, metric_value.to(self.device), on_epoch=True, sync_dist=False)
            for metric_name, metric_value in wer_verbatim.compute().items():
                log_name = "val_wer_chat_verbatim" if metric_name == "wer" else f"val_{metric_name}_chat_verbatim"
                self.log(log_name, metric_value.to(self.device), on_epoch=True, sync_dist=False)

        self._partial_wer_refs_chat.clear()
        self._partial_wer_hyps_chat.clear()
