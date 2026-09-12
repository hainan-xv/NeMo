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
"""CHAT transducer with a selectable training objective.

ONE model, two losses. ``EncDecCHATBPEModel`` is an ``EncDecRNNTBPEModel`` --
same ``RNNTAttJoint``, same ``RNNTDecoder``, same 1,024-piece SentencePiece
vocabulary plus blank, same greedy chunk-synchronous decoding -- with a single
switch::

    model.loss_type: rnnt              # marginalise over every alignment
    model.loss_type: forced_alignment  # condition on ONE alignment

This replaces a separate ChatSTTModel class. Two classes for two losses meant
two vocabularies (the old one carried three unused SCRIPT delimiters, giving
1,027 text classes against the standard 1,024), two decode paths to keep in
sync, and a checkpoint from one that could not be loaded by the other. Folding
the objective into a config field makes the two arms differ in exactly the thing
under study and nothing else, and lets either arm initialise from the other.

WHY A FORCED ALIGNMENT AT ALL. The RNN-T loss sums over every way of
interleaving labels and blanks, which needs a ``[B, T, U, V+1]`` tensor. Scoring
one path instead needs only ``[U+T, V+1]``: the cost stops scaling with the
lattice, which is what makes a large vocabulary trainable. The bet is that a
good alignment is nearly as good a target as the marginal -- that is what the
two ``loss_type`` settings measure against each other.

The forced path is built from the word timings the Granary cuts already carry in
``cut.custom["alignments"]``, so it needs no new dataset class -- only
``return_cuts=True`` on the ordinary Lhotse BPE dataset.
"""

import os
import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from lhotse.dataset.collation import collate_vectors
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.losses.banded_rnnt import BandedLattice, banded_rnnt_loss, build_lattices
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.utils.chat_alignment import assign_words_to_chunks, build_forced_path, chunk_texts
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.mixins import AccessMixin
from nemo.utils import logging

__all__ = ["EncDecCHATBPEModel"]

LOSS_TYPES = ("rnnt", "forced_alignment", "banded")


class _RNNTAutoTokenizer(AutoTokenizer):
    """AutoTokenizer with the leniency the RNN-T decoding path assumes.

    ``RNNTBPEDecoding.decode_tokens_to_str`` passes token IDS to
    ``tokens_to_text``. ``SentencePieceTokenizer`` accepts ids or strings, so
    that works for a SentencePiece vocabulary; ``AutoTokenizer`` accepts only
    strings. Worse, the transducer's BLANK sits at ``vocab_size`` -- outside the
    vocabulary -- so converting it yields ``None`` and the HuggingFace decoder
    fails with:

        TypeError: argument 'tokens': 'NoneType' object cannot be converted to 'PyString'

    Accepting ids and dropping out-of-vocabulary ones restores the contract the
    decoding path relies on, without touching shared ASR code.
    """

    def tokens_to_text(self, tokens, remove_special_tokens=False):
        # Drop None FIRST. This is reached two ways: with raw ids, and with the
        # output of decode_ids_to_tokens, which already maps an out-of-vocabulary
        # id (the blank) to None. Deciding ids-vs-strings from items[0] before
        # filtering crashes whenever a hypothesis BEGINS with a blank.
        items = [t for t in list(tokens) if t is not None]
        if not items:
            return ""
        if not isinstance(items[0], str):
            n = self.vocab_size
            items = self.tokenizer.convert_ids_to_tokens([int(t) for t in items if 0 <= int(t) < n])
            items = [t for t in items if t is not None]
            if not items:
                return ""
        return super().tokens_to_text(items, remove_special_tokens=remove_special_tokens)


class EncDecCHATBPEModel(EncDecRNNTBPEModel):
    """CHAT transducer trainable with either the marginalised or a forced loss."""

    def __init__(self, cfg: DictConfig, trainer=None):
        # EVERY attribute the data-setup path reads must be assigned BEFORE
        # super().__init__(): ModelPT.__init__ calls setup_training_data() from
        # inside it, so anything set afterwards does not exist yet and the model
        # dies with AttributeError at construction.
        self.loss_type = str(cfg.get("loss_type", "rnnt"))
        if self.loss_type not in LOSS_TYPES:
            raise ValueError(f"model.loss_type must be one of {LOSS_TYPES}, got {self.loss_type!r}")

        fa = cfg.get("forced_alignment", {}) or {}
        self.num_delay_frames = int(fa.get("num_delay_frames", 0))
        self.recover_history_words = int(fa.get("recover_history_words", 0))
        # Seconds of audio per ENCODER frame: 10 ms hop x 8x subsampling. Only
        # used to turn word end times into frame indices; a wrong value silently
        # shifts every word to the wrong chunk, so it is derived from the
        # configured preprocessor rather than hard-coded.
        self.frame_length_in_secs = float(
            fa.get("frame_length_in_secs", None) or cfg.preprocessor.window_stride * cfg.encoder.subsampling_factor
        )

        # FLEXIBLE DELAY. When > 0, each batch draws d ~ U{0..max_delay_frames}
        # and (a) shifts the alignment by d, (b) hides the last d frames of every
        # chunk from the joint. The two are exact complements, so the model emits
        # a chunk's words using exactly the audio that has arrived -- which makes
        # d a LATENCY knob: at d frames it commits d frames before the chunk
        # completes. Training over the range yields one model usable at any
        # latency in it, instead of one model per latency.
        # How many chunks a word may drift from where the aligner put it, when
        # loss_type is "banded". 0 reproduces the forced loss exactly.
        self.band_chunks = int(fa.get("band_chunks", 1))
        self.max_delay_frames = int(fa.get("max_delay_frames", 0) or 0)
        # Latency to decode at. Half the range by default: the middle of what the
        # model was trained on rather than either extreme.
        infer = fa.get("inference_delay_frames", None)
        self.inference_delay_frames = int(infer) if infer is not None else self.max_delay_frames // 2
        self._delay_rng: Optional[random.Random] = None

        self._ws_ids: Optional[frozenset] = None
        # Set only while the TRAINING loader is being built: the forced loss
        # needs the cuts (for their alignments), validation does not and is
        # scored by the ordinary WER path.
        self._want_cuts = False

        super().__init__(cfg=cfg, trainer=trainer)

        # Restore the real get_vocab now that the config is built (see
        # _setup_tokenizer); leaving the placeholder would corrupt anything that
        # later asks the tokenizer for its vocabulary.
        _patch = getattr(self, "_hf_vocab_patch", None)
        if _patch is not None:
            _patch[0].get_vocab = _patch[1]
            self._hf_vocab_patch = None

        if self.loss_type in ("forced_alignment", "banded"):
            logging.info(
                f"CHAT {self.loss_type} loss (band_chunks={self.band_chunks}): delay={self.num_delay_frames} frames, "
                f"recover_history_words={self.recover_history_words}, "
                f"frame_length={self.frame_length_in_secs:.4f}s, chunk_size={self.joint.chunk_size}"
            )

    # ----------------------------------------------------------- tokenizer

    def _setup_tokenizer(self, tokenizer_cfg):
        """Accept ``type: huggingface`` in addition to the ASR collection's bpe/wpe.

        The forced-alignment and banded losses exist because a large vocabulary
        makes the full RNN-T lattice unaffordable, so being able to attach one is
        the point. NeMo's own ``AutoTokenizer`` already satisfies everything the
        RNN-T BPE model asks of a tokenizer -- including ``.tokenizer.get_vocab()``,
        which is how the joint sizes itself -- so this only has to construct it
        and register the files so ``save_to`` produces a self-contained .nemo.
        """
        if str(tokenizer_cfg.get("type", "")).lower() not in ("huggingface", "hf"):
            return super()._setup_tokenizer(tokenizer_cfg)

        tok_dir = tokenizer_cfg["dir"]
        # Downstream branches test tokenizer_type against 'agg'; a subword
        # tokenizer behaves like the bpe path everywhere else.
        self.tokenizer_type = "bpe"
        self.tokenizer_dir = tok_dir
        self.tokenizer_cfg = OmegaConf.to_container(tokenizer_cfg) if hasattr(tokenizer_cfg, "keys") else {}
        self.tokenizer = _RNNTAutoTokenizer(pretrained_model_name=tok_dir)

        # EncDecRNNTBPEModel copies the vocabulary into cfg.labels and then into
        # cfg.joint.vocabulary. OmegaConf gives several strings special meaning,
        # and an LLM vocabulary contains them as ordinary pieces:
        #     "${..."  -> parsed as an interpolation   (36 Qwen pieces)
        #     "???"    -> parsed as MISSING            (1 Qwen piece)
        # Each one aborts model construction, and escaping does not survive the
        # second copy. Rather than chase OmegaConf's special values one at a
        # time, do not put the strings in the config at all.
        #
        # Nothing needs them. cfg.labels is used ONLY for its length, to size the
        # decoder embedding and the joint output layer, and joint.vocabulary is
        # stored but never read -- decode_ids_to_str goes through the tokenizer.
        # So the config gets a placeholder list of the right length while the
        # tokenizer keeps the real vocabulary.
        #
        # The patch is reverted as soon as the parent constructor has run, so
        # nothing else ever sees the placeholder.
        _hf = self.tokenizer.tokenizer
        _raw_get_vocab = _hf.get_vocab
        _n = len(_raw_get_vocab())

        def _placeholder_vocab():
            return {f"<piece_{i}>": i for i in range(_n)}

        _hf.get_vocab = _placeholder_vocab
        self._hf_vocab_patch = (_hf, _raw_get_vocab)

        # Bundle the vocabulary into the .nemo. Without this, restore_from would
        # need the original lustre path to still exist, which it will not on
        # another machine.
        for fname in (
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            path = os.path.join(tok_dir, fname)
            if os.path.isfile(path):
                self.register_artifact(f"tokenizer.{fname}", path)
        logging.info(f"HuggingFace tokenizer from {tok_dir}: {self.tokenizer.vocab_size} pieces")

    # ------------------------------------------------------------------ data

    def setup_training_data(self, train_data_config):
        self._want_cuts = self.loss_type in ("forced_alignment", "banded")
        try:
            super().setup_training_data(train_data_config)
        finally:
            self._want_cuts = False

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if self._want_cuts and config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextBpeDataset(tokenizer=self.tokenizer, return_cuts=True),
                tokenizer=self.tokenizer,
            )
        return super()._setup_dataloader_from_config(config)

    # ------------------------------------------------------- inference delay

    def _pin_inference_delay(self) -> None:
        """Fix the trim for a decode pass.

        Without this the joint keeps whatever d the LAST TRAINING BATCH drew, so
        val_wer would be measured at a random latency each time and would not
        describe any single operating point. Outside training the attribute
        defaults to 0, which is a different operating point again -- so both
        paths have to be pinned explicitly.
        """
        if self.max_delay_frames > 0:
            self.joint.frame_trim = self.inference_delay_frames

    def on_validation_epoch_start(self):
        self._pin_inference_delay()
        return super().on_validation_epoch_start()

    def on_test_epoch_start(self):
        self._pin_inference_delay()
        return super().on_test_epoch_start()

    # ------------------------------------------------------- flexible delay

    def _sample_delay(self) -> int:
        """One delay for the whole batch, drawn independently per rank.

        Seeding by global rank matters: with a shared seed all 64 ranks would
        draw the SAME d every step, so each step would see one latency across
        the entire global batch instead of a spread over the range.
        """
        if self.max_delay_frames <= 0:
            return self.num_delay_frames
        if self._delay_rng is None:
            self._delay_rng = random.Random(1234 + int(self.global_rank))
        return self._delay_rng.randint(0, self.max_delay_frames)

    def _append_flush_chunk(self, encoded, encoded_len):
        """One all-zero chunk on the end, so trimmed-off words still get emitted.

        With d > 0 the last d frames of the final chunk are hidden, so any word
        ending in them has no chunk left to be emitted from and would otherwise
        be folded backwards into a chunk that cannot see it. An extra chunk gives
        those words somewhere to go; it carries no audio of its own, but with
        history_chunks >= 1 it still attends to the real frames behind it, which
        is what it needs to flush them.
        """
        b, d_model, _ = encoded.shape
        pad = encoded.new_zeros(b, d_model, self.joint.chunk_size)
        return torch.cat([encoded, pad], dim=2), encoded_len + self.joint.chunk_size

    # -------------------------------------------------------- forced path

    def _word_start_ids(self) -> frozenset:
        """Token ids that begin a word (U+2581 SentencePiece, U+0120 byte BPE).

        Training extends a chunk backward over whole words and retract-style
        decoding rolls back over whole words; both must agree on where a word
        starts, or the model would be asked to recover from states it never saw.
        """
        if self._ws_ids is None:
            ids = set()
            for i in range(self.tokenizer.vocab_size):
                try:
                    got = self.tokenizer.ids_to_tokens([i])
                except Exception:  # noqa: BLE001 -- a gap in the vocab is not fatal
                    continue
                piece = got[0] if got else None
                if isinstance(piece, str) and (piece.startswith("▁") or piece.startswith("Ġ")):
                    ids.add(i)
            self._ws_ids = frozenset(ids)
        return self._ws_ids

    def _chunk_tokens(self, cut, n_chunks: int, delay: Optional[int] = None) -> List[List[int]]:
        """Tokens each chunk is responsible for, one list per chunk.

        The text comes from the ORIGINAL transcript, sliced by where each
        aligned word sits in it -- not from the aligner's word forms, which have
        punctuation stripped. Building targets from the bare forms trains a PnC
        model that can never emit punctuation, and inflates a verbatim WER
        against a punctuated reference on every sentence.
        """
        aligned = (cut.custom or {}).get("alignments", []) or []
        words = [w["text"] for w in aligned]
        transcript = " ".join(s.text for s in cut.supervisions if s.text) if cut.supervisions else ""

        groups = assign_words_to_chunks(
            [w["end_time"] for w in aligned],
            n_chunks,
            self.joint.chunk_size,
            self.frame_length_in_secs,
            self.num_delay_frames if delay is None else delay,
        )
        texts = self._chunk_texts_for_tokenizer(groups, words, transcript)
        return [self.tokenizer.text_to_ids(t) if t else [] for t in texts]

    @property
    def _tokenizer_supplies_word_prefix(self) -> bool:
        """Does this tokenizer mark a word start without being given a space?

        SentencePiece does: its dummy prefix makes ``text_to_ids("word")`` begin
        with ``▁word``, so a chunk tokenized on its own still says "word start".
        A byte-level BPE like Qwen's does NOT -- ``"word"`` encodes as ``word``
        and only ``" word"`` encodes as ``Ġword``, a DIFFERENT id.
        """
        cached = getattr(self, "_tok_word_prefix", None)
        if cached is None:
            ids = self.tokenizer.text_to_ids("word")
            pieces = self.tokenizer.ids_to_tokens(list(ids)) if ids else []
            cached = bool(pieces) and str(pieces[0]).startswith(("▁", "Ġ"))
            self._tok_word_prefix = cached
        return cached

    def _chunk_texts_for_tokenizer(self, groups, words, transcript) -> List[str]:
        """Chunk texts, spaced so that tokenizing them SEPARATELY gives the same
        ids as tokenizing the whole transcript at once.

        This is the difference between the 1,024-piece arm working and the
        151.7k Qwen arm producing 'the bestselling singleby a Germanartist'.
        chunk_texts strips whitespace and leaves the word-boundary marker to the
        tokenizer's dummy prefix, which SentencePiece supplies and a byte-level
        BPE does not. Without it every chunk after the first trains the model on
        the NO-SPACE variant of its opening word, so the words are individually
        right and run together when decoded -- and no amount of training fixes
        it, because the targets themselves are wrong.

        The first chunk carrying text keeps its bare form: it opens the
        utterance, where the full-sentence tokenization has no preceding space
        either. For SentencePiece nothing changes, which is what keeps the
        explicit space from becoming the junk standalone piece chunk_texts warns
        about.
        """
        texts = chunk_texts(groups, words, transcript)
        if self._tokenizer_supplies_word_prefix:
            return texts
        out, started = [], False
        for t in texts:
            if not t:
                out.append(t)
                continue
            out.append(" " + t if started else t)
            started = True
        return out

    def _build_batch_path(self, cuts, n_chunks: torch.Tensor, device, delay: Optional[int] = None):
        """Assemble (b, t, u, labels) and the prediction-network input."""
        blank = self.joint.num_classes_with_blank - 1
        ws = self._word_start_ids() if self.recover_history_words > 0 else None

        b_all, t_all, u_all, lab_all, preds = [], [], [], [], []
        for b, cut in enumerate(cuts):
            chunks = self._chunk_tokens(cut, int(n_chunks[b]), delay)
            starts = [[i for i, t in enumerate(c) if i == 0 or t in ws] for c in chunks] if ws else None
            t_idx, u_idx, labels = build_forced_path(chunks, blank, self.recover_history_words, starts)
            b_all += [b] * len(t_idx)
            t_all += t_idx
            u_all += u_idx
            lab_all += labels
            preds.append(torch.tensor([tok for c in chunks for tok in c], dtype=torch.long))

        pred_lens = torch.tensor([p.numel() for p in preds], dtype=torch.long)
        pred_input = collate_vectors(preds, padding_value=0).to(device)
        as_t = lambda x: torch.tensor(x, dtype=torch.long, device=device)  # noqa: E731
        return as_t(b_all), as_t(t_all), as_t(u_all), as_t(lab_all), pred_input, pred_lens.to(device)

    def _banded_loss(self, encoded, encoded_len, cuts) -> torch.Tensor:
        """RNN-T forward summed over paths within ``band_chunks`` of the alignment.

        Uses the same chunk assignment as the forced loss -- the alignment is the
        band's centre -- but scores the NODES of a lattice rather than the steps
        of one path, and sums their paths instead of taking the product along a
        single one. ``joint_on_path`` already evaluates the joint at arbitrary
        ``(b, t, u)`` triples, so the expensive part is shared unchanged.
        """
        chunk_size = self.joint.chunk_size
        n_chunks = torch.div(encoded_len + chunk_size - 1, chunk_size, rounding_mode="floor").cpu()

        chunks_per_utt = [self._chunk_tokens(cut, int(n_chunks[b])) for b, cut in enumerate(cuts)]
        per_utt, num_chunks, target_lens = build_lattices(chunks_per_utt, self.band_chunks)
        if sum(target_lens) == 0:
            logging.warning(f"empty banded lattice at step {self.global_step}; contributing zero loss")
            return encoded.sum() * 0.0
        if min(num_chunks) == 0:
            # BandedLattice reads index[(b, num_chunks[b] - 1, ...)] for its final
            # node, so a zero-chunk utterance looks up chunk -1 and raises
            # KeyError rather than producing a loss. Reachable whenever an
            # utterance encodes to no frames at all.
            logging.warning(
                f"banded batch at step {self.global_step} has an utterance with no chunks "
                f"(num_chunks={num_chunks}); contributing zero loss"
            )
            return encoded.sum() * 0.0

        lattice = BandedLattice(per_utt, num_chunks, target_lens)
        b_idx, t_idx, u_idx = lattice.index_tensors(encoded.device)

        # Targets, and the prediction network run over them. u indexes emitted
        # labels, exactly as in the forced path.
        u_max = max(target_lens)
        targets = torch.zeros((len(chunks_per_utt), max(u_max, 1)), dtype=torch.long, device=encoded.device)
        for b, chunks in enumerate(chunks_per_utt):
            flat = [tok for c in chunks for tok in c]
            if flat:
                targets[b, : len(flat)] = torch.tensor(flat, dtype=torch.long, device=encoded.device)
        pred_lens = torch.tensor(target_lens, dtype=torch.long, device=encoded.device)

        g, _, _ = self.decoder(targets=targets, target_length=pred_lens)
        g = g.transpose(1, 2)

        logits = self.joint.joint_on_path(encoded.transpose(1, 2), g, b_idx, t_idx, u_idx, encoded_len)
        log_probs = logits.float().log_softmax(-1)

        blank = self.joint.num_classes_with_blank - 1
        nll = banded_rnnt_loss(log_probs, lattice, targets, blank)
        # mean_volume: per target token, matching the rnnt arm's reduction so the
        # two losses are on the same scale.
        loss = nll.sum() / max(int(pred_lens.sum()), 1)

        if not torch.isfinite(loss):
            return self._skip_nonfinite_batch(loss, encoded, g, logits, nll, encoded_len, lattice, target_lens)
        return loss

    def _skip_nonfinite_batch(self, loss, encoded, g, logits, nll, encoded_len, lattice, target_lens):
        """Drop a batch whose loss is not finite, and say where it went wrong.

        WHY THIS EXISTS. A single non-finite loss produces NaN gradients, and
        gradient clipping does not repair a NaN -- it propagates into every
        weight on the next step and the run emits NaN forever. That is not a
        hypothetical: the first Qwen-vocabulary banded run trained cleanly for
        ~400 steps (loss 15.5 -> 12.8), hit one bad batch, and then logged NaN
        for 5,143 consecutive steps before anyone noticed. Skipping the batch
        costs one update; not skipping it costs the entire job.

        The stage report matters because the failure is otherwise SILENT: the
        loss arithmetic is well conditioned (random-structure sweeps stay finite,
        and log_softmax in float32 cannot underflow to -inf), so the interesting
        question is whether `encoded`, the prediction network, or the joint went
        bad first -- which only the batch that fails can answer.
        """
        stage = "loss reduction (all inputs finite)"
        for name, tensor in (("encoder output", encoded), ("prediction network", g), ("joint logits", logits)):
            if not torch.isfinite(tensor).all():
                stage = name
                break
        bad = (~torch.isfinite(nll)).nonzero().flatten().tolist()
        # Surface the skip as a METRIC, not only as a log line. The guard returns
        # a loss of exactly 0.0, which on a progress bar is indistinguishable
        # from a model that has learned the task perfectly -- and that is how the
        # first skipped run was read. A train_batches_skipped that climbs with
        # every step says plainly that nothing is being learned.
        self._batches_skipped = getattr(self, "_batches_skipped", 0) + 1
        self.log("train_batches_skipped", float(self._batches_skipped), prog_bar=True)
        logging.error(
            "non-finite banded loss at step %s: first non-finite stage = %s; "
            "batch=%d utts, encoded_len min/max=%d/%d, lattice nodes=%d, "
            "target_lens=%s, non-finite utterances=%s. Skipping this batch.",
            self.global_step,
            stage,
            encoded.shape[0],
            int(encoded_len.min()),
            int(encoded_len.max()),
            lattice.num_nodes,
            target_lens,
            bad,
        )
        # Zero loss that still TOUCHES every parameter, so DDP sees a gradient
        # for each one and does not abort on an unused parameter.
        #
        # Built from the PARAMETERS, not from the poisoned activations. Routing
        # it through `logits` instead looks equivalent -- nan_to_num(logits) is
        # finite, and multiplying by zero should zero the gradient -- but the
        # backward of the ops FEEDING logits still multiplies by their saved
        # non-finite activations, so 0 * inf = NaN reaches the weights. The loss
        # then looks contained while the gradients are exactly as poisonous as
        # before.
        #
        # Each parameter is reduced to a SCALAR and made finite before anything
        # multiplies it. Multiplying by zero last does not work on its own --
        # NaN * 0 is NaN -- and whatever went wrong upstream may well have
        # reached the weights already, so the guard cannot assume they are clean.
        # nan_to_num on a scalar per parameter costs nothing next to a copy of
        # all 812M of them, and its derivative is zero exactly where the value
        # was not finite.
        zero = sum(torch.nan_to_num(p.sum()) for p in self.parameters() if p.requires_grad)
        return zero * 0.0

    def _forced_alignment_loss(self, encoded, encoded_len, cuts) -> torch.Tensor:
        # ONE delay per batch. It has to reach BOTH the alignment shift and the
        # joint's window trim: they are complements, and applying either alone
        # would train the model to emit words it cannot hear, or to ignore audio
        # it has been given.
        delay = self._sample_delay()
        self.joint.frame_trim = delay
        if delay > 0:
            encoded, encoded_len = self._append_flush_chunk(encoded, encoded_len)

        # Chunk counts come from the ACTUAL encoder output, not from the
        # duration, so the two sides cannot disagree about the tail chunk.
        chunk_size = self.joint.chunk_size
        n_chunks = torch.div(encoded_len + chunk_size - 1, chunk_size, rounding_mode="floor")

        b_idx, t_idx, u_idx, labels, pred_input, pred_lens = self._build_batch_path(
            cuts, n_chunks.cpu(), encoded.device, delay
        )

        if labels.numel() == 0:
            # cross_entropy over an empty path is nan, and that nan reaches the
            # weights. A batch can legitimately have no alignable words; return a
            # graph-connected zero so backward still runs on every rank and DDP
            # stays in lockstep.
            logging.warning(f"empty forced-alignment path at step {self.global_step}; contributing zero loss")
            return encoded.sum() * 0.0

        g, _, _ = self.decoder(targets=pred_input, target_length=pred_lens)
        g = g.transpose(1, 2)  # (B, D, U+1) -> (B, U+1, D); the decoder prepends its own SOS

        # forward() hands back [B, D, T], but joint_on_path chunks along the TIME
        # axis and would otherwise slice the feature axis into "chunks" -- which
        # reshapes to a plausible-looking tensor rather than failing loudly.
        logits = self.joint.joint_on_path(encoded.transpose(1, 2), g, b_idx, t_idx, u_idx, encoded_len)
        return F.cross_entropy(logits.float(), labels)

    # ------------------------------------------------------------- training

    def training_step(self, batch, batch_nb):
        """Only the LOSS differs between the arms; everything logged is shared.

        The rnnt arm defers to the parent untouched. The forced arm mirrors the
        parent step for step -- same access-registry handling, same metric names,
        same logging cadence, and in particular the SAME ``training_batch_wer``,
        computed by the same greedy decode against the same reference tokens.
        The two curves therefore mean the same thing and can be read on one plot;
        a WER that appeared for one arm and not the other would make exactly the
        comparison this model exists to support impossible.
        """
        if self.loss_type == "rnnt":
            # The window trim applies to the marginalised loss too, and means the
            # same thing: the last d frames of each chunk have not arrived yet.
            # There is no alignment to shift here -- the RNN-T loss chooses its
            # own emission points, so hiding the frames is the whole mechanism.
            #
            # No flush chunk is needed either. Mid-utterance the trimmed frames
            # simply reappear in the next chunk's window; only the final chunk's
            # trailing d frames are never seen, and with pad_extra_duration 0.5 s
            # (~6 frames) those are silence padding for any d <= 4.
            if self.max_delay_frames > 0:
                self.joint.frame_trim = self._sample_delay()
            return super().training_step(batch, batch_nb)

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len, cuts = batch
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        if self.loss_type == "banded":
            loss_value = self.add_auxiliary_losses(self._banded_loss(encoded, encoded_len, cuts))
        else:
            loss_value = self.add_auxiliary_losses(self._forced_alignment_loss(encoded, encoded_len, cuts))

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # Logged every step and on the progress bar, unlike the rest: a forced
        # loss that collapses to ~0 within a few steps means the targets came
        # through empty, and that should be visible in the job's own log rather
        # than only in wandb.
        self.log('train_loss', loss_value, prog_bar=True)

        if (sample_id + 1) % log_every_n_steps == 0:
            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            _, scores, words = self.wer.compute()
            self.wer.reset()
            self.log_dict(
                {
                    'learning_rate': self._optimizer.param_groups[0]['lr'],
                    'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
                    'training_batch_wer': scores.float() / words,
                }
            )

        return {'loss': loss_value}

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []
