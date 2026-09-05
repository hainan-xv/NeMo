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
"""CHAT transducer trained on a forced alignment.

THE EXPERIMENT. A larger text vocabulary clearly helps the SpeechLM: swapping
Qwen's 151,936 pieces for the ASR encoder's 1,024 cost ~0.14 WER on the
leaderboard. Whether a TRANSDUCER benefits the same way is unknown, and normally
untestable: the RNN-T loss marginalises over all alignments and needs a
[B, T, U, V] tensor, which at V=151,936 does not fit.

Fixing the alignment removes the marginalisation. Each word is emitted at the
chunk holding its last token -- the same assignment the SpeechLM trains on -- so
the loss is plain cross-entropy along a single path of U + T steps, scored by
``RNNTAttJoint.joint_on_path``. Memory then grows with the path, not with
T * U * V, and the two vocabularies become directly comparable.

WHY A FIXED ALIGNMENT IS NOT A HANDICAP. Marginalising over alignments exists to
LEARN an alignment you do not have; given a good one, conditioning on it is hard
EM rather than soft EM. The SpeechLM is trained exactly this way -- same forced
alignment, same delay, same chunk assignment -- and reaches 5.96 macro against
the RNN-T baseline's 5.82, so the recipe demonstrably works at this scale.

Crucially, DECODING MIRRORS TRAINING: RNNTAttJoint's CHAT greedy decode walks
chunks and emits tokens within a chunk until a blank, with the prediction state
advancing only on real tokens -- precisely the path built here. There is no
search over alignments at inference, so there is no train/test mismatch of the
kind that would otherwise punish single-path training.

What IS fixed by construction is emission latency: a word is emitted at its
aligned chunk plus the configured delay, rather than wherever the model finds
convenient. That is the same deliberate trade the SpeechLM makes to keep
streaming latency controllable.
"""

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.modules import RNNTAttJoint, RNNTDecoder
from nemo.collections.speechlm2.parts.metrics.wer import WER
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers
from nemo.collections.speechlm2.parts.pretrained import setup_perception
from nemo.collections.speechlm2.parts.utils import to_dataclass
from nemo.utils import logging


@dataclass
class ChatSTTModelConfig:
    """Config for the forced-alignment CHAT transducer.

    Attributes:
        pretrained_asr: ``.nemo`` whose encoder (and, for the small-vocabulary
            arm, whose tokenizer) is used.
        chunk_size: encoder frames per chunk. Fixed, not sampled: the dataset's
            chunk indices and the joint's own chunking must agree exactly.
        vocab_size: number of text classes, EXCLUDING blank. Blank is class
            ``vocab_size`` (NeMo's convention).
        text_vocab_from_asr: use the ASR encoder's SentencePiece vocabulary
            (~1,024) instead of the LLM tokenizer named by ``pretrained_llm``.
            This is the knob the experiment turns.
    """

    pretrained_asr: str
    chunk_size: int = 14
    vocab_size: int = 1024
    pretrained_llm: str = ""
    text_vocab_from_asr: bool = True
    pred_hidden: int = 640
    pred_rnn_layers: int = 2
    joint_hidden: int = 640
    att_context_size: Optional[list] = None
    load_asr_weights: bool = True
    freeze_speech_encoder: bool = False
    audio_pad_to: int = 0
    # Extra PREVIOUS chunks the joint may attend to when emitting for a chunk.
    # 0 = standard CHAT (the joint sees only the chunk it emits for). 1 gives a
    # 28-frame window on a 14-frame emission grid -- the transducer analogue of
    # SCRIPT's win28 -- so a word straddling a boundary has its onset visible.
    # Emission granularity and latency are unchanged; no look-ahead is added.
    joint_history_chunks: int = 0
    # Cap on tokens emitted per chunk at decode time.
    #
    # 14 = chunk_size, which is what has ACTUALLY been in force for every run so
    # far: the CHAT greedy decode used to derive this from the frame count of
    # the chunk slice and ignore whatever was configured here. Now that the
    # configured value is honoured, keep it at the emission grid so the running
    # arms are unaffected and the win28 window does not inflate it.
    #
    # Headroom check: at ~1.9 tokens/word the 1k vocabulary needs ~6 tokens for a
    # typical 1.12s chunk and ~10 for a dense one, so 14 does not truncate; Qwen
    # (~1.16) needs far fewer. The cap therefore cannot bias the vocabulary
    # comparison, while still bounding a repetition loop.
    max_symbols: int = 14


class ChatSTTModel(LightningModule):
    """Encoder + prediction network + chunk-attention joint, forced-alignment CE."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)
        self.core_cfg: ChatSTTModelConfig = to_dataclass(ChatSTTModelConfig, cfg)

        self.chunk_size = int(self.core_cfg.chunk_size)
        self.max_symbols = int(self.core_cfg.max_symbols)
        self._decoding = None
        # Pin the encoder's look-ahead to this chunk size so a frame never
        # depends on audio past its own chunk boundary -- the same constraint the
        # SpeechLM encoder runs under, which is what keeps the comparison honest.
        att = self.core_cfg.att_context_size or [70, self.chunk_size - 1]
        self.perception = setup_perception(
            cfg=self.cfg,
            output_dim=self.core_cfg.joint_hidden,
            pretrained_asr=self.core_cfg.pretrained_asr,
            pretrained_weights=self.core_cfg.load_asr_weights,
            audio_pad_to=self.core_cfg.audio_pad_to or None,
            att_context_size=att,
        )

        V = int(self.core_cfg.vocab_size)
        self.blank_id = V  # blank is the extra class at the end
        self.decoder = RNNTDecoder(
            prednet={
                "pred_hidden": self.core_cfg.pred_hidden,
                "pred_rnn_layers": self.core_cfg.pred_rnn_layers,
                "dropout": 0.1,
            },
            vocab_size=V,
        )
        self.joint = RNNTAttJoint(
            jointnet={
                "encoder_hidden": self.core_cfg.joint_hidden,
                "pred_hidden": self.core_cfg.pred_hidden,
                "joint_hidden": self.core_cfg.joint_hidden,
                "activation": "relu",
                "dropout": 0.1,
            },
            num_classes=V,
            chunk_size=self.chunk_size,
            history_chunks=int(self.core_cfg.joint_history_chunks),
        )
        logging.info(
            f"ChatSTTModel: vocab={V} (+1 blank), chunk_size={self.chunk_size}, "
            f"att_context_size={att}, joint_hidden={self.core_cfg.joint_hidden}, "
            f"joint_history_chunks={self.core_cfg.joint_history_chunks} "
            f"(joint attends to {(int(self.core_cfg.joint_history_chunks) + 1) * self.chunk_size} frames)"
        )

        if self.cfg.get("init_rnnt_from_asr", True) and self.core_cfg.pretrained_asr:
            self._init_from_pretrained_rnnt(self.core_cfg.pretrained_asr)

    def _init_from_pretrained_rnnt(self, nemo_path: str) -> None:
        """Warm-start the VOCABULARY-INDEPENDENT parts of the pretrained RNN-T.

        Loading only the encoder wastes most of a checkpoint that is otherwise
        shape-compatible, and worse, actively damages the part we do load. The
        pretrained model maps its 1024-d encoder into the joint's 640-d space
        with ``joint.enc``; we replaced that learned projection with a RANDOM
        ``perception.proj`` of exactly the same shape, so the encoder's output
        arrives at the joint scrambled. "Initialised from a good checkpoint" was
        therefore only half true, and it is the most likely reason convergence
        looked slower than a normal RNN-T fine-tune.

        WHAT IS COPIED, AND WHY ONLY THIS. Only weights whose meaning does not
        depend on the vocabulary:

          pretrained joint.enc  (640x1024) -> perception.proj   ... encoder into joint space
          our joint.enc         (640x640)  -> IDENTITY          ... so the composition is a no-op
          pretrained joint.pred (640x640)  -> joint.pred        ... prediction state into joint space
          pretrained LSTM       (2 x 640)  -> decoder LSTM      ... the language model itself

        The embedding table and the output layer are deliberately left RANDOM in
        both arms even though arm 1's vocabulary matches the donor exactly.
        Copying them would hand arm 1 a pretrained softmax and embedding while
        arm 2 (Qwen) structurally cannot have them -- which would confound the
        one variable the two arms exist to isolate. A vocabulary-size comparison
        where one side starts from a matching pretrained output layer measures
        the initialisation, not the vocabulary.

        Q/K/V are new to the attention joint and have no donor.
        """
        import tarfile
        import tempfile

        try:
            with tarfile.open(nemo_path, "r:") as tf:
                member = next((m for m in tf.getmembers() if m.name.endswith("model_weights.ckpt")), None)
                if member is None:
                    logging.warning(f"{nemo_path} has no model_weights.ckpt; skipping RNN-T warm start.")
                    return
                with tempfile.TemporaryDirectory() as td:
                    tf.extract(member, td)
                    sd = torch.load(os.path.join(td, member.name), map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001
            logging.warning(f"could not read pretrained RNN-T weights from {nemo_path}: {e!r}")
            return

        copied, skipped = [], []

        def _copy(dst: torch.Tensor, src_key: str, label: str) -> None:
            src = sd.get(src_key)
            if src is None:
                skipped.append(f"{label} (no {src_key} in donor)")
            elif tuple(src.shape) != tuple(dst.shape):
                skipped.append(f"{label} (shape {tuple(src.shape)} != {tuple(dst.shape)})")
            else:
                with torch.no_grad():
                    dst.copy_(src.to(dst.dtype))
                copied.append(label)

        # 1. The encoder -> joint-space projection, the piece whose loss hurt most.
        proj = getattr(self.perception, "proj", None)
        if isinstance(proj, torch.nn.Linear):
            _copy(proj.weight, "joint.enc.weight", "perception.proj.weight <- joint.enc")
            _copy(proj.bias, "joint.enc.bias", "perception.proj.bias <- joint.enc")
            # Our joint.enc now sees an already-projected 640-d input, so start it
            # at identity: the composition then reproduces the donor's mapping
            # exactly instead of passing it through a second random matrix.
            enc_lin = getattr(self.joint, "enc", None)
            if isinstance(enc_lin, torch.nn.Linear) and enc_lin.in_features == enc_lin.out_features:
                with torch.no_grad():
                    enc_lin.weight.copy_(torch.eye(enc_lin.out_features, dtype=enc_lin.weight.dtype))
                    enc_lin.bias.zero_()
                copied.append("joint.enc <- identity")
        else:
            skipped.append("perception.proj (absent; encoder dim already matches joint_hidden)")

        # 2. Prediction state -> joint space.
        pred_lin = getattr(self.joint, "pred", None)
        if isinstance(pred_lin, torch.nn.Linear):
            _copy(pred_lin.weight, "joint.pred.weight", "joint.pred.weight")
            _copy(pred_lin.bias, "joint.pred.bias", "joint.pred.bias")

        # 3. The prediction network's LSTM -- the donor's language model. Its
        #    embedding is NOT copied (see the docstring).
        lstm = None
        for name, mod in self.decoder.named_modules():
            if isinstance(mod, torch.nn.LSTM):
                lstm = mod
                break
        if lstm is None:
            skipped.append("prediction LSTM (not found)")
        else:
            for layer in range(lstm.num_layers):
                for w in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    attr = f"{w}_l{layer}"
                    dst = getattr(lstm, attr, None)
                    if dst is None:
                        skipped.append(f"lstm.{attr} (absent)")
                        continue
                    _copy(dst, f"decoder.prediction.dec_rnn.lstm.{attr}", f"lstm.{attr}")

        logging.info(
            "ChatSTTModel RNN-T warm start from %s\n  copied (%d): %s\n  left random: %s\n"
            "  embedding + output layer are intentionally random in BOTH arms so the\n"
            "  vocabulary comparison is not confounded by a matching pretrained softmax.",
            nemo_path,
            len(copied),
            ", ".join(copied) or "nothing",
            ", ".join(skipped + ["joint.Q/K/V (new to the attention joint)"]),
        )
        if not copied:
            logging.warning(
                "RNN-T warm start copied NOTHING -- the model is training its decoder, joint and "
                "encoder projection from scratch. Check that pretrained_asr is an RNN-T .nemo."
            )

    def _encode(self, audios, audio_lens):
        """Encoder output as (B, T, D), with the orientation ASSERTED, not guessed.

        The previous heuristic compared shape[1] to max(enc_len), which silently
        picks the wrong axis whenever the feature dim happens to equal the frame
        count -- and a mis-oriented tensor does not fail here, it fails deep
        inside the joint's reshape with an unreadable size error.
        """
        enc, enc_len = self.perception(input_signal=audios, input_signal_length=audio_lens)
        D = int(self.core_cfg.joint_hidden)
        if enc.shape[-1] != D and enc.shape[1] == D:
            enc = enc.transpose(1, 2)  # (B, D, T) -> (B, T, D)
        if enc.shape[-1] != D:
            raise RuntimeError(
                f"encoder output {tuple(enc.shape)} has neither axis equal to joint_hidden={D}; "
                "the joint cannot reshape it. Check perception's output_dim."
            )
        return enc, enc_len

    def forward_loss(self, batch) -> torch.Tensor:
        """Cross-entropy along the forced path."""
        enc, enc_len = self._encode(batch.audios, batch.audio_lens)

        # (B, D, U+1) -> (B, U+1, D); the decoder prepends its own SOS.
        g, _, _ = self.decoder(targets=batch.pred_input, target_length=batch.pred_lens)
        g = g.transpose(1, 2)

        # The dataset estimates frames from DURATION (ceil(secs / frame_length))
        # while the joint chunks the ACTUAL encoder output, and subsampling
        # boundary effects make those differ by a frame at the tail -- which
        # flips the final chunk. A +/-1 discrepancy is benign: the surplus chunk
        # simply carries no target. A larger one means the two sides disagree
        # about chunk_size or framing, which is a real misconfiguration.
        n_joint = torch.div(enc_len + self.chunk_size - 1, self.chunk_size, rounding_mode="floor")
        n_data = batch.n_chunks.to(n_joint.device)
        drift = (n_joint - n_data).abs()
        if bool((drift > 1).any()):
            bad = (drift > 1).nonzero(as_tuple=True)[0][:5].tolist()
            raise RuntimeError(
                "chunk-count mismatch beyond tail rounding: "
                f"joint={n_joint[bad].tolist()} vs dataset={n_data[bad].tolist()} (utterances {bad}). "
                "The two must use the same chunk_size and frame_length_in_secs."
            )

        # Drop any step whose chunk the encoder did not produce, so t_idx can
        # never index past the joint's chunk axis.
        keep = batch.t_idx < n_joint[batch.b_idx]
        b_idx, t_idx, u_idx = batch.b_idx[keep], batch.t_idx[keep], batch.u_idx[keep]
        labels = batch.labels[keep]

        logits = self.joint.joint_on_path(enc, g, b_idx, t_idx, u_idx, enc_len)

        if labels.numel() == 0:
            # cross_entropy over an EMPTY path returns nan (mean of nothing), and
            # that nan reaches the weights and kills the run. A batch can end up
            # empty when every utterance in it has no alignable words -- rare,
            # but with 64 ranks drawing batches it happens, and it took down job
            # 13104496 at step 3 while the loss was still at its initial value.
            #
            # Return a graph-connected zero instead: backward runs on every rank
            # (so DDP stays in lockstep) and contributes no gradient.
            logging.warning(f"empty forced-alignment path at step {self.global_step}; contributing zero loss.")
            return logits.sum() * 0.0 + enc.sum() * 0.0

        return F.cross_entropy(logits.float(), labels)

    @property
    def decoding(self):
        """Lazily-built :class:`RNNTDecoding` for greedy CHAT decoding.

        Inference needs no new algorithm: ``rnnt_decoder_predictions_tensor``
        already detects ``chunk_encoder_for_decoding`` on the joint, chunks the
        encoder output and forwards ``chunk_frame_lengths``, and the CHAT greedy
        loop walks chunks emitting until a blank -- the same procedure this model
        is trained on. All that was missing was the decoding object itself.
        """
        if getattr(self, "_decoding", None) is None:
            from omegaconf import OmegaConf

            from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecoding

            # A plain id list: WER is scored downstream against detokenised text,
            # so the vocabulary only has to be the right SIZE here.
            vocab = [str(i) for i in range(int(self.core_cfg.vocab_size))]
            # strategy="greedy", NOT "greedy_batch": the CHAT-aware decode
            # (_greedy_decode_chat) lives in GreedyRNNTInfer, the non-batched
            # class. greedy_batch routes to the label-looping computer, which
            # reshapes the encoder output on assumptions CHAT's chunked layout
            # does not satisfy and dies inside project_encoder. Slower, since it
            # walks utterances one at a time, but validation is small and this is
            # the path whose emission rule matches training.
            cfg = OmegaConf.create({"strategy": "greedy", "greedy": {"max_symbols": self.max_symbols}})
            self._decoding = RNNTDecoding(decoding_cfg=cfg, decoder=self.decoder, joint=self.joint, vocabulary=vocab)
        return self._decoding

    @torch.no_grad()
    def transcribe_ids(self, audios: torch.Tensor, audio_lens: torch.Tensor):
        """Greedy CHAT decode -> per-utterance token id lists.

        Ids rather than text: the tokenizer lives on the dataset side, and the
        two vocabulary arms detokenise differently, so the caller converts.
        """
        was_training = self.training
        self.eval()
        try:
            enc, enc_len = self._encode(audios, audio_lens)
            # rnnt_decoder_predictions_tensor expects (B, D, T).
            hyps = self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=enc.transpose(1, 2), encoded_lengths=enc_len, return_hypotheses=True
            )
            if isinstance(hyps, tuple):
                hyps = hyps[0]
            out = []
            for h in hyps:
                y = h.y_sequence
                out.append(y.tolist() if torch.is_tensor(y) else list(y))
            return out
        finally:
            self.train(was_training)

    def training_step(self, batch, batch_idx):
        loss = self.forward_loss(batch)

        # A non-finite loss must not be logged (it would poison the metric) and
        # marks the step for neutralisation. The real check is on GRADIENTS, in
        # on_before_optimizer_step -- see there.
        finite = bool(torch.isfinite(loss.detach()).item())
        self._loss_nonfinite = not finite
        if finite:
            self.log("train_loss", loss, prog_bar=True, sync_dist=True)
            self._maybe_log_train_decode(batch)
        return loss

    def on_before_optimizer_step(self, optimizer):
        """Zero the gradients of any step that is not finite, on every rank.

        WHY GRADIENTS AND NOT JUST THE LOSS. Job 13105689 died with a FINITE
        loss of 6.95 at step 7 and non-finite weights by step 8, with warmup
        still holding the LR at ~2.8e-6 -- so this was never an LR problem. A
        finite loss can still produce a non-finite gradient, and gradient
        clipping then turns one bad value into a global catastrophe: a nan grad
        anywhere makes the total norm nan, so the clip coefficient is nan, so
        EVERY parameter's gradient becomes nan. Checking only the loss, as the
        previous version did, cannot see any of this.

        WHY NOT RETURN None FROM training_step. Lightning rejects that outright
        under distributed training, which killed job 13104496. Zeroing here
        keeps every rank in the backward pass and DDP's all-reduce in lockstep.

        The decision is ALL-REDUCED: DDP has already averaged gradients across
        ranks, so one rank's nan is present everywhere, and every rank must
        therefore make the same call.
        """
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        dev = grads[0].device if grads else self.device
        bad = torch.zeros((), device=dev)
        if grads:
            total = torch.stack([n.float() for n in torch._foreach_norm(grads)]).sum()
            if not torch.isfinite(total):
                bad.fill_(1.0)
        if getattr(self, "_loss_nonfinite", False):
            bad.fill_(1.0)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)

        if not bool(bad.item()):
            self._nonfinite_steps = 0
            return

        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()
        self._nonfinite_steps = getattr(self, "_nonfinite_steps", 0) + 1
        logging.warning(
            f"non-finite loss or gradient at step {self.global_step} "
            f"({self._nonfinite_steps} in a row); gradients zeroed, update skipped."
        )
        # A one-off bad batch is survivable. A RUN of them means the weights are
        # already non-finite, so every future step will be too -- fail loudly
        # rather than burn the rest of the allocation, as job 13096766 did for
        # 3.5 hours on 8 nodes while Slurm reported COMPLETED.
        if self._nonfinite_steps >= int(self.cfg.get("max_nonfinite_steps", 25)):
            raise RuntimeError(
                f"loss or gradients have been non-finite for {self._nonfinite_steps} consecutive steps; "
                "the weights are almost certainly nan. Zeroing gradients cannot recover from that -- "
                "the run must be restarted from a good checkpoint."
            )

    def _maybe_log_train_decode(self, batch) -> None:
        """Periodically print ref / forced-alignment target / greedy hyp.

        WHY, when val_wer already exists: val_wer is one number every few
        thousand steps, and while it sits near 1.0 -- as it does for the first
        tens of thousands of steps, when the model is still emitting mostly
        blanks -- it cannot distinguish "learning slowly but correctly" from
        "learning something wrong". The decoded strings can: a hypothesis that
        is empty, or stuck on one token, or a fluent transcript of the wrong
        audio, each look completely different long before WER separates them.

        The TARGET line is the reason to do this on TRAINING data specifically.
        It is the forced-alignment path the loss is actually computed against,
        reconstructed from the batch's own labels -- so if the alignment,
        tokenisation or chunk assignment were wrong, this line would show it
        directly, while a validation decode (which has no alignment) could not.
        A target that does not read like the reference means the model is
        learning exactly what it was told to, and the fault is upstream.

        Rank 0 only. That is safe under DDP, where parameters are replicated and
        a no_grad forward runs no collectives -- but it would DEADLOCK under
        FSDP, where the forward all-gathers shards and every rank must
        participate. If this model ever moves to FSDP, decode on all ranks and
        print on one.
        """
        every = int(self.cfg.get("log_train_decode_every_n_steps", 0) or 0)
        if every <= 0 or self.global_step <= 0 or self.global_step % every != 0:
            return
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        # Once per global step, not once per micro-batch under accumulation.
        if getattr(self, "_last_decode_step", None) == self.global_step:
            return
        self._last_decode_step = self.global_step

        n = max(1, int(self.cfg.get("log_train_decode_examples", 2) or 2))
        n = min(n, int(batch.audios.shape[0]))
        try:
            lens = batch.audio_lens[:n]
            ids = self.transcribe_ids(batch.audios[:n, : int(lens.max())], lens)
            refs = list(getattr(batch, "text", []) or [])
            lines = []
            for i in range(n):
                ref = refs[i] if i < len(refs) else "<no reference in batch>"
                lines.append(
                    f"\n  [{i}] ref: `{ref}`"
                    f"\n      tgt: `{self._target_text(batch, i)}`"
                    f"\n      hyp: `{self._detokenize(ids[i])}`"
                )
            logging.info("train decode @ step %d%s", self.global_step, "".join(lines))
        except Exception as e:  # noqa: BLE001
            # Diagnostics must never take down a training run that is otherwise
            # healthy -- losing the printout is an annoyance, losing the run is
            # hours of GPU time.
            logging.warning(f"train decode logging failed at step {self.global_step}: {e!r}")

    def _target_text(self, batch, i: int) -> str:
        """The forced-alignment target for utterance i, as text.

        Blanks are dropped: they are the per-chunk terminators, not content, so
        including them would make every target unreadable. What remains is
        exactly the token sequence the loss pushes the model to emit.
        """
        sel = batch.b_idx == i
        if not bool(sel.any()):
            return ""
        labels = batch.labels[sel]
        toks = [int(t) for t in labels[labels != self.blank_id].tolist()]
        return self._detokenize(toks)

    # ------------------------------------------------------------------
    # Validation: autoregressive decoding -> WER. No loss.
    # ------------------------------------------------------------------
    def on_validation_epoch_start(self) -> None:
        self._partial_wer_refs = defaultdict(list)
        self._partial_wer_hyps = defaultdict(list)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Decode-only WER, exactly as the SpeechLM validates.

        No validation LOSS: the training objective is the forced-alignment path,
        and the validation set is a plain manifest with no word alignments, so
        there is no chunk assignment to score against. WER needs none of that --
        only audio and reference text -- and it measures the thing we actually
        care about.
        """
        if isinstance(batch, dict) and not hasattr(batch, "audios"):
            for name, sub in batch.items():
                if sub is not None:
                    self._eval_step(sub, name, batch_idx)
        else:
            self._eval_step(batch, "val", batch_idx)

    def _eval_step(self, batch, name: str, batch_idx: int = 0) -> None:
        refs = list(getattr(batch, "text", []) or [])
        if not refs:
            # Without references WER is undefined, and silently logging 0 or
            # skipping would look like a healthy validation pass.
            raise RuntimeError(
                f"validation batch '{name}' carries no reference text; ChatAlignedBatch.text must be "
                "populated for decode-only WER."
            )
        ids = self.transcribe_ids(batch.audios, batch.audio_lens)
        hyps = [self._detokenize(seq) for seq in ids]
        self._partial_wer_refs[name].extend(refs)
        self._partial_wer_hyps[name].extend(hyps)
        if batch_idx == 0 and refs and hyps:
            logging.info("[%s] decode batch %d\n  ref: `%s`\n  hyp: `%s`", name, batch_idx, refs[0], hyps[0])

    def _detokenize(self, ids) -> str:
        """Ids -> text via whatever tokenizer this arm was built with.

        The tokenizer lives on the dataset, so it is attached by the training
        script; without it WER would silently score ids against words.
        """
        tok = getattr(self, "tokenizer", None)
        if tok is None:
            raise RuntimeError(
                "ChatSTTModel has no tokenizer attached; set model.tokenizer before validation "
                "or WER would compare token ids against reference text."
            )
        return tok.ids_to_text(list(ids)) if ids else ""

    def on_validation_epoch_end(self) -> None:
        # Gather decoded strings and compute a true corpus WER: averaging
        # rank-local WERs would be wrong when ranks see different word counts.
        local = {
            n: {"refs": self._partial_wer_refs[n], "hyps": self._partial_wer_hyps[n]} for n in self._partial_wer_refs
        }
        if torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        else:
            gathered = [local]

        wer = WER(normalize=True, verbose=False)
        has_data = False
        for rank_data in gathered:
            for n, v in rank_data.items():
                has_data = has_data or bool(v["refs"])
                wer.update(n, refs=v["refs"], hyps=v["hyps"])
        if has_data:
            for k, v in wer.compute().items():
                self.log("val_wer" if k == "wer" else f"val_{k}", v.to(self.device), on_epoch=True, sync_dist=False)
        self._partial_wer_refs.clear()
        self._partial_wer_hyps.clear()

    def configure_optimizers(self):
        """Per-group learning rates: pretrained encoder low, from-scratch parts high.

        A single learning rate has to serve two very different populations here.
        The 609M encoder is pretrained and only needs fine-tuning, while the
        prediction network, the joint, and the joint's Q/K/V are trained from
        SCRATCH. Set the rate low enough to be safe for the encoder and the new
        parts crawl; set it high enough for the new parts and the encoder is at
        risk early.

        How far off the single rate was: the donor RNN-T -- the model this very
        encoder comes from -- trained with NoamAnnealing(lr=3.0, d_model=1024,
        warmup=8000), whose peak is 1.05e-3. We were running a flat 1e-4, i.e.
        10.5x below peak and 4-8x below it for the whole of training, for
        components that are no better initialised than the donor's were.

        ``lr_multipliers`` maps a regex over parameter names to a factor on the
        base LR, so the encoder can stay at its current effective rate while the
        from-scratch parts get the rate the architecture actually wants. The
        scheduler is unaffected: PyTorch records each group's initial LR as its
        base and scales all groups together, so the ratio holds throughout.
        """
        mults = self.cfg.get("lr_multipliers", None)
        if not mults:
            return configure_optimizers(self)

        import re

        # freeze_and_subset yields bare parameters, but grouping needs the NAMES,
        # so apply the same freeze rules here rather than calling it.
        freeze = [re.compile(x) for x in (self.cfg.get("freeze_params", []) or [])]
        keep = [re.compile(x) for x in (self.cfg.get("prevent_freeze_params", []) or [])]
        named = []
        for name, p in self.named_parameters():
            if any(k.match(name) for k in keep):
                named.append((name, p))
                continue
            if any(f.match(name) for f in freeze):
                p.requires_grad = False
                continue
            named.append((name, p))
        base_lr = float(self.cfg.optimizer.lr)
        buckets: dict[float, list] = {}
        assigned: dict[float, list] = {}
        for name, p in named:
            mult = 1.0
            for pattern, m in mults.items():
                if re.search(pattern, name):
                    mult = float(m)
                    break
            buckets.setdefault(mult, []).append(p)
            assigned.setdefault(mult, []).append(name)

        groups = [{"params": ps, "lr": base_lr * mult} for mult, ps in sorted(buckets.items())]
        opt_cfg = OmegaConf.to_container(self.cfg.optimizer, resolve=True)
        target = opt_cfg.pop("_target_")
        opt_cfg.pop("lr", None)
        mod, cls = target.rsplit(".", 1)
        import importlib

        optimizer = getattr(importlib.import_module(mod), cls)(groups, lr=base_lr, **opt_cfg)

        for mult, names in sorted(assigned.items()):
            n_par = sum(p.numel() for p in buckets[mult])
            logging.info(
                f"  lr group x{mult:g} -> {base_lr * mult:.2e}: {len(names)} tensors, {n_par/1e6:.1f}M params "
                f"(e.g. {names[0]})"
            )

        ans = {"optimizer": optimizer}
        if "lr_scheduler" in self.cfg:
            from nemo.collections.speechlm2.parts.optim_setup import safe_instantiate

            sched = safe_instantiate(self.cfg.lr_scheduler, optimizer)
            ans["lr_scheduler"] = {"scheduler": sched, "interval": "step", "frequency": 1}
        return ans


__all__ = ["ChatSTTModel", "ChatSTTModelConfig"]
