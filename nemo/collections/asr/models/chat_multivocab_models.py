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

"""CHAT with SEVERAL vocabularies sharing one encoder.

One encoder, N (tokenizer, decoder, joint) heads -- 1k/2k/4k by default. Each
training batch samples ONE head and is scored entirely with it; validation is
pinned to head 0 so ``val_wer`` is a single comparable curve across the run
rather than a mixture whose value depends on which head happened to be drawn.

WHY THIS CAN WORK AT ALL. The vocabulary sweep found 1k/4k/8k statistically
indistinguishable (4.81/4.83/4.82 macro-7) while the encoder is ~0.6B of the
~0.65B parameters and the vocabulary-shaped tensors are a rounding error. So the
heads are cheap, and if the encoder representation is genuinely vocabulary-
agnostic then one encoder should serve all three at no cost -- which is the
hypothesis this model exists to test.

THE TOKENIZER IS NOT JUST AN OUTPUT LAYER HERE, which is what makes this more
than an extra nn.Linear. The banded loss builds its lattice from PER-CHUNK TOKEN
COUNTS (``_chunk_tokens`` -> ``build_band_index``), so switching head changes the
targets, the band geometry and the number of lattice nodes, not only the softmax
width. Every one of those is rebuilt per batch from the active head.
"""

import os
import random
from typing import Any, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.models.chat_bpe_models import EncDecCHATBPEModel
from nemo.utils import logging

__all__ = ["EncDecMultiVocabCHATBPEModel"]


class _Head:
    """One vocabulary's worth of model: everything downstream of the encoder."""

    __slots__ = (
        "tokenizer",
        "decoder",
        "joint",
        "decoding",
        "wer",
        "loss",
        "cfg_decoder",
        "cfg_joint",
        "cfg_tokenizer",
        "name",
    )

    def __init__(self, model, name: str):
        self.name = name
        self.tokenizer = model.tokenizer
        self.decoder = model.decoder
        self.joint = model.joint
        self.decoding = model.decoding
        self.wer = model.wer
        self.loss = model.loss
        # Snapshots so the ACTIVE head's shapes are what save_to writes. Without
        # them cfg.decoder/cfg.joint would describe whichever head was built
        # last, and a restore would construct the wrong-sized output layer.
        self.cfg_decoder = OmegaConf.to_container(model.cfg.decoder, resolve=False)
        self.cfg_joint = OmegaConf.to_container(model.cfg.joint, resolve=False)
        # The tokenizer dir belongs to the head too. change_vocabulary rewrites
        # cfg.tokenizer as it builds each head, so without capturing it here the
        # saved config keeps whichever head was built LAST.
        self.cfg_tokenizer = OmegaConf.to_container(model.cfg.tokenizer, resolve=False)


class EncDecMultiVocabCHATBPEModel(EncDecCHATBPEModel):
    """CHAT with one encoder and several vocabulary heads.

    Config::

        model:
          tokenizer:            # MUST be head 0 -- the dataloader and validation use it
            dir: .../v1024
            type: bpe
          multivocab:
            tokenizer_dirs: [.../v1024, .../v2048, .../v4096]
            sample_weights: [1, 1, 1]     # optional, defaults to uniform
            val_head: 0                   # optional, defaults to 0
    """

    def __init__(self, cfg: DictConfig, trainer=None):
        mv = cfg.get("multivocab", None)
        dirs = [str(d) for d in mv["tokenizer_dirs"]] if (mv and mv.get("tokenizer_dirs", None)) else None

        # Head 0 is special: the LhotseSpeechToTextBpeDataset is constructed with
        # self.tokenizer during setup_training_data (called from inside
        # ModelPT.__init__), so whatever cfg.tokenizer.dir points at is what
        # every batch's `transcript` tensor is encoded with. Pinning it to
        # dirs[0] keeps validation on one vocabulary for the whole run.
        if dirs and str(cfg.tokenizer.dir) != dirs[0]:
            raise ValueError(
                f"model.tokenizer.dir ({cfg.tokenizer.dir}) must equal multivocab.tokenizer_dirs[0] ({dirs[0]}); "
                "head 0 feeds the dataloader and validation."
            )

        super().__init__(cfg, trainer=trainer)

        if not dirs:
            # SINGLE-HEAD FALLBACK, and it is load-bearing rather than defensive.
            # init_from_nemo_model restores the DONOR .nemo by calling
            # from_config_dict with THIS class and the donor's own config -- which
            # has no multivocab section, because the donor is an ordinary
            # single-vocabulary CHAT model. Requiring multivocab here made warm
            # starting impossible: the run died in 97 s at
            # from_config_dict -> imported_cls(cfg=donor_cfg). Degrading to one
            # head is correct for that path, which only wants encoder weights.
            dirs = [str(cfg.tokenizer.dir)]
            logging.info("CHAT multi-vocab: no multivocab section (donor restore?); running single-head")

        self._heads: List[_Head] = [_Head(self, dirs[0])]
        ref_joint = self._heads[0].joint
        for d in dirs[1:]:
            # change_vocabulary rebuilds tokenizer/decoder/joint/decoding/wer/loss
            # in place. The previous head's modules survive because _heads still
            # references them, so its `del self.joint` only drops the attribute.
            self.change_vocabulary(d, "bpe")
            self._inherit_chunk_geometry(ref_joint, self.joint)
            self._heads.append(_Head(self, d))

        # HARD CHECK, because the failure this guards is silent. chunk_size is
        # inferred by the model from the encoder's att_context_size and set as a
        # RUNTIME attribute on head 0's joint; it is not written back to the
        # config, so a joint rebuilt from to_config_dict() gets the -1 default.
        # Heads 1 and 2 then have window_width 0 and train against a degenerate
        # emission grid -- observed, with the loss still falling, until the WER
        # decode finally raised in chunk_concat_audio 11 minutes in. A wrong
        # chunk_size does not fail fast, so it is asserted rather than trusted.
        sizes = [int(h.joint.chunk_size) for h in self._heads]
        if len(set(sizes)) != 1 or sizes[0] <= 0:
            raise ValueError(f"heads disagree on joint.chunk_size or it is unset: {sizes}")

        # Register every head's parameters. named_parameters() de-duplicates by
        # identity, so the active head being reachable as BOTH self.decoder and
        # heads_decoders[i] does not double-count it in the optimizer.
        #
        # ONLY WHEN THERE IS MORE THAN ONE HEAD. With a single head self.decoder
        # and self.joint already register everything, and the ModuleLists would
        # be pure duplication -- but worse, they add heads_decoders.0.* /
        # heads_joints.0.* keys that a plain CHAT checkpoint does not have. That
        # is not hypothetical: init_from_nemo_model builds this class from the
        # DONOR config (single head) and then calls load_state_dict(strict=True)
        # with the donor's weights, which failed with 21 "Missing key(s)" and
        # took the run down at 3m16. Skipping the lists on that path makes the
        # instance structurally identical to the plain model the donor is.
        if len(self._heads) > 1:
            self.heads_decoders = torch.nn.ModuleList([h.decoder for h in self._heads])
            self.heads_joints = torch.nn.ModuleList([h.joint for h in self._heads])
            self.heads_wers = torch.nn.ModuleList([h.wer for h in self._heads])

        w = mv.get("sample_weights", None) if mv else None
        self._head_weights = [float(x) for x in w] if w else [1.0] * len(self._heads)
        if len(self._head_weights) != len(self._heads):
            raise ValueError(f"sample_weights has {len(self._head_weights)} entries for {len(self._heads)} heads")
        self._val_head = int(mv.get("val_head", 0)) if mv else 0
        self._head_rng: Optional[random.Random] = None
        self._active_head = -1
        self._head_counts = [0] * len(self._heads)

        self._select_head(0)
        # change_vocabulary left cfg AND the tokenizer artifacts describing the
        # LAST head built; put both back.
        self._sync_cfg_to_head(0)
        self._register_head_tokenizer(0)
        logging.info(
            "CHAT multi-vocab: %d heads %s, sample weights %s, validation pinned to head %d (%s)",
            len(self._heads),
            [f"{h.name.rstrip('/').split('/')[-1]}:{h.joint.num_classes_with_blank - 1}" for h in self._heads],
            self._head_weights,
            self._val_head,
            self._heads[self._val_head].name.rstrip("/").split("/")[-1],
        )

    # ----------------------------------------------------------------- heads
    @staticmethod
    def _inherit_chunk_geometry(ref, new) -> None:
        """Copy the joint's INFERRED geometry, which to_config_dict() does not carry.

        These are the attributes CHAT derives at construction rather than reads
        from the config. Everything vocabulary-shaped must differ between heads;
        everything about the emission grid must not.
        """
        for attr in ("chunk_size", "history_chunks", "frame_trim"):
            if hasattr(ref, attr):
                setattr(new, attr, getattr(ref, attr))

    def _select_head(self, i: int) -> None:
        """Make head ``i`` the one every inherited code path sees."""
        if i == self._active_head:
            return
        h = self._heads[i]
        self.tokenizer = h.tokenizer
        self.decoder = h.decoder
        self.joint = h.joint
        self.decoding = h.decoding
        self.wer = h.wer
        self.loss = h.loss
        # Match the model's mode at the moment of selection: a head bound in
        # while the model is training must not still be in eval from a decode.
        h.decoder.train(self.training)
        h.joint.train(self.training)
        self._active_head = i

    def _sync_cfg_to_head(self, i: int) -> None:
        """Point cfg.decoder/cfg.joint at head ``i``'s shapes.

        NOT done on every switch. Rebuilding two OmegaConf nodes per batch cost
        a measured 23% of step time against a single-vocab arm -- for a config
        nothing reads mid-step. Checkpoints are written after validation, which
        pins head 0, so syncing at construction and before a save is sufficient
        and the .nemo still describes the head it contains.
        """
        h = self._heads[i]
        with open_dict(self.cfg):
            self.cfg.decoder = OmegaConf.create(h.cfg_decoder)
            self.cfg.joint = OmegaConf.create(h.cfg_joint)
            # MUST include the tokenizer. Saved with head 2's dir but head 0's
            # weights, the .nemo failed its own constructor check on restore --
            # every decode in job 19136746 died before producing a single WER.
            self.cfg.tokenizer = OmegaConf.create(h.cfg_tokenizer)

    def _sample_head(self) -> int:
        if self._head_rng is None:
            # Seeded per rank so the heads are drawn independently across DDP
            # ranks; a shared seed would make every rank pick the same head and
            # turn N heads into N/world_size effective samples per step.
            self._head_rng = random.Random(1234 + int(self.global_rank))
        total = sum(self._head_weights)
        r = self._head_rng.uniform(0.0, total)
        acc = 0.0
        for i, w in enumerate(self._head_weights):
            acc += w
            if r <= acc:
                return i
        return len(self._heads) - 1

    def _retokenize_batch(self, cuts, device):
        """Targets for the ACTIVE head, from the cuts' text.

        The dataloader encodes `transcript` with head 0's tokenizer once, at
        setup. Feeding that to another head's WER would score head 1's output
        against head 0's token ids -- not merely noisy, meaningless. So the
        reference is re-encoded from text whenever a non-zero head is active.
        """
        ids = []
        for cut in cuts:
            text = cut.supervisions[0].text if cut.supervisions else ""
            ids.append(self.tokenizer.text_to_ids(text) if text else [])
        n = max((len(x) for x in ids), default=0)
        out = torch.zeros((len(ids), max(n, 1)), dtype=torch.long, device=device)
        lens = torch.zeros(len(ids), dtype=torch.long, device=device)
        for b, x in enumerate(ids):
            if x:
                out[b, : len(x)] = torch.tensor(x, dtype=torch.long, device=device)
            lens[b] = len(x)
        return out, lens

    def train(self, mode: bool = True):
        """Put EVERY head in ``mode``, not just whichever the tree walk reaches.

        Observed: after a validation epoch, Lightning's model.train() left the
        previously-active head's decoder and joint in eval while the others came
        back to train -- even though all three are reachable from modules() and
        the ModuleList holds them. The next training step then ran that head's
        LSTM forward in eval and died in backward with "cudnn RNN backward can
        only be called in training mode" (grid job 19118743, reproduced locally).

        The precise mechanism is not identified; the invariant is simple enough
        to assert directly rather than depend on it emerging.
        """
        super().train(mode)
        for h in getattr(self, "_heads", ()):
            h.decoder.train(mode)
            h.joint.train(mode)
        return self

    def state_dict(self, *args, **kwargs):
        """Always emit head 0 under ``decoder.*`` / ``joint.*``.

        The active head is reachable BOTH as self.decoder and as
        heads_decoders[i], so those keys describe whichever head happened to be
        sampled when the state was captured. Scheduled checkpoints are written
        after validation, which pins head 0 -- but a TERMINATION checkpoint is
        not, and one written mid-training with the 2k head active produced
        decoder.prediction.embed.weight of [2049, 640]. Resuming then failed with
        a size mismatch against the 1k head the fresh model builds.

        Pinning head 0 for the duration of the capture makes the alias
        deterministic, so a checkpoint is always readable by a fresh model.
        """
        prev = self._active_head
        if prev != 0:
            self._select_head(0)
        try:
            return super().state_dict(*args, **kwargs)
        finally:
            if prev != 0:
                self._select_head(prev)

    def _register_head_tokenizer(self, i: int) -> None:
        """Point the tokenizer.* ARTIFACTS at head ``i``'s files.

        change_vocabulary calls register_artifact('tokenizer.model_path', ...)
        for every head, and each call overwrites the same key -- so a .nemo saved
        after building three heads bundles the LAST head's tokenizer. On restore
        NeMo rewrites cfg.tokenizer.model_path to that bundled file, and head 0
        is then constructed with head 2's vocabulary: _heads[0] came back with
        96 pieces instead of 48, and every eval decode died on a size mismatch.

        Syncing cfg.tokenizer alone is not enough, because the artifact registry
        is what decides which FILE ends up inside the .nemo.
        """
        d = self._heads[i].name
        for key, fname in (("tokenizer.model_path", "tokenizer.model"), ("tokenizer.vocab_path", "vocab.txt")):
            path = os.path.join(d, fname)
            if os.path.isfile(path):
                self.register_artifact(key, path)

    # ---------------------------------------------------- joint decoding

    def enable_joint_decoding(
        self, weights=None, beam: int = 4, max_candidates: int = 16, strategy: str = "greedy"
    ) -> None:
        """Decode with ALL heads at once rather than the selected head alone.

        Hooks the ordinary transcribe() pipeline, so batching, audio loading and
        manifest writing are untouched -- only the step that turns encoder
        output into text changes. ``weights=[1, 0, 0]`` falls back to head 0.
        """
        from nemo.collections.asr.parts.submodules.multivocab_joint_decoding import MultiVocabChunkJointDecoder

        self._joint_decoder = MultiVocabChunkJointDecoder(
            self, weights=weights, beam=beam, max_candidates=max_candidates, strategy=strategy
        )
        logging.info(
            f"CHAT multi-vocab JOINT decoding: {len(self._heads)} heads "
            f"{[h.name.rsplit('/', 1)[-1] for h in self._heads]}, weights "
            f"{[h.weight for h in self._joint_decoder.heads]}, strategy {strategy}, "
            f"max_symbols {self._joint_decoder.max_symbols}"
        )

    def disable_joint_decoding(self) -> None:
        self._joint_decoder = None

    def _transcribe_output_processing(self, outputs, trcfg) -> Any:
        """Substitute the joint decode for the active head's greedy decode."""
        jd = getattr(self, "_joint_decoder", None)
        if jd is None:
            return super()._transcribe_output_processing(outputs, trcfg)

        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

        encoded = outputs.pop("encoded")
        encoded_len = outputs.pop("encoded_len")
        results = jd.decode(encoded, encoded_len)
        del encoded, encoded_len
        # y_sequence is left empty on purpose: the winning TEXT has no single
        # token sequence, it has one per head. Everything downstream (the
        # leaderboard scorer included) reads .text.
        return [
            Hypothesis(score=float(sum(r.chunk_scores)), y_sequence=[], text=r.text, timestamp=[]) for r in results
        ]

    def save_to(self, save_path: str):
        """Always write head 0's config, weights and tokenizer.

        state_dict() already pins head 0 for the weights; the config and the
        tokenizer artifacts have to agree with it or the .nemo cannot rebuild
        itself.
        """
        prev = self._active_head
        self._select_head(0)
        self._sync_cfg_to_head(0)
        self._register_head_tokenizer(0)
        try:
            return super().save_to(save_path)
        finally:
            if prev != 0:
                self._select_head(prev)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Read ``decoder.*`` / ``joint.*`` as head 0, matching state_dict().

        Those keys are an ALIAS of whichever head is bound, so save and load must
        agree on which one they mean. state_dict() pins head 0; without the same
        pin here a restore whose instance happens to be on another head fails
        with a size mismatch -- observed restoring a .nemo whose config and
        weights were both head 0 (48 pieces) into an instance presenting head 2
        (96), which broke every multi-vocab eval decode.
        """
        prev = getattr(self, "_active_head", 0)
        if prev != 0:
            self._select_head(0)
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        finally:
            if prev != 0:
                self._select_head(prev)

    # ------------------------------------------------------------- lightning
    def training_step(self, batch, batch_nb):
        i = self._sample_head()
        self._select_head(i)
        self._head_counts[i] += 1

        signal, signal_len, transcript, transcript_len, cuts = batch
        if i != 0:
            transcript, transcript_len = self._retokenize_batch(cuts, signal.device)
            batch = (signal, signal_len, transcript, transcript_len, cuts)

        out = super().training_step(batch, batch_nb)

        # Which head produced this step's train_loss / training_batch_wer, and
        # the running mix. A skewed mix (one head starved) is otherwise invisible
        # and would quietly make the comparison between heads unfair.
        self.log("train_head", float(i), prog_bar=True)
        total = max(sum(self._head_counts), 1)
        for k, c in enumerate(self._head_counts):
            self.log(f"train_head_frac_{k}", c / total)
        return out

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        # Pinned, so val_wer is one curve for one vocabulary across the whole
        # run. The dataloader's transcripts are head 0's ids, which is exactly
        # what this head expects -- no re-encoding needed or wanted.
        self._select_head(self._val_head)
        return super().validation_pass(batch, batch_idx, dataloader_idx)

    def on_validation_epoch_start(self):
        self._select_head(self._val_head)
        # Checkpoints are written at the end of validation, so this is the point
        # where cfg must describe the live head.
        self._sync_cfg_to_head(self._val_head)
        return super().on_validation_epoch_start()
