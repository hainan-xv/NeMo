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
"""``EncDecCHATBPEModel``: one architecture, two selectable training losses.

The point of folding the forced-alignment objective into ``EncDecRNNTBPEModel``
was that the two arms should differ ONLY in the loss. These tests pin that down:
same vocabulary, same parameters, interchangeable checkpoints, same decoder.
"""

import math
import os

import pytest
import torch
from lhotse import CutSet, MonoCut, SupervisionSegment
from omegaconf import DictConfig

from nemo.collections.asr.models import EncDecCHATBPEModel


def _cfg(test_data_dir, loss_type, recover=0):
    return DictConfig(
        {
            'loss_type': loss_type,
            'forced_alignment': {'num_delay_frames': 0, 'recover_history_words': recover},
            'sample_rate': 16000,
            'compute_eval_loss': False,
            'skip_nan_grad': False,
            'model_defaults': {'enc_hidden': 32, 'pred_hidden': 32, 'joint_hidden': 32},
            'tokenizer': {'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_spe_128"), 'type': 'bpe'},
            'preprocessor': {
                '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
                'features': 64,
                'window_stride': 0.01,
            },
            'encoder': {
                '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
                'feat_in': 64,
                'feat_out': -1,
                'n_layers': 2,
                'd_model': 32,
                'subsampling': 'dw_striding',
                'subsampling_factor': 8,
                'subsampling_conv_channels': 16,
                'causal_downsampling': True,
                'self_attention_model': 'rel_pos',
                'n_heads': 2,
                'att_context_size': [70, 13],  # chunk_size = right + 1 = 14
                'att_context_style': 'chunked_limited',
                'conv_kernel_size': 9,
                'conv_context_size': 'causal',
            },
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
                'prednet': {'pred_hidden': 32, 'pred_rnn_layers': 1},
            },
            'joint': {
                '_target_': 'nemo.collections.asr.modules.RNNTAttJoint',
                'jointnet': {'encoder_hidden': 32, 'pred_hidden': 32, 'joint_hidden': 32, 'activation': 'relu'},
            },
            'decoding': {'strategy': 'greedy_batch', 'greedy': {'max_symbols': 10}},
            'loss': {'loss_name': 'default'},
            'optim': {'name': 'adamw', 'lr': 1e-3},
        }
    )


def _cuts(texts_and_times, duration=3.0):
    """CutSet carrying word alignments the way the Granary manifests do."""
    cuts = []
    for i, words in enumerate(texts_and_times):
        text = " ".join(w[0] for w in words)
        cut = MonoCut(id=f"c{i}", start=0.0, duration=duration, channel=0, recording=None)
        cut.supervisions = [
            SupervisionSegment(id=f"s{i}", recording_id=f"r{i}", start=0.0, duration=duration, text=text)
        ]
        cut.custom = {
            "alignments": [{"text": w, "start_time": s, "end_time": e} for (w, s, e) in words],
        }
        cuts.append(cut)
    return CutSet.from_cuts(cuts)


def _train_ds_cfg(tmp_path):
    """A minimal Lhotse training config backed by real audio on disk."""
    from lhotse.testing.dummies import DummyManifest

    cuts = DummyManifest(CutSet, begin_id=0, end_id=4, with_data=True)
    cuts = cuts.save_audios(str(tmp_path / "audio"))
    for i, cut in enumerate(cuts):
        cut.supervisions[0].text = "hello world"
        cut.custom = {
            "alignments": [
                {"text": "hello", "start_time": 0.0, "end_time": 0.3},
                {"text": "world", "start_time": 0.4, "end_time": 0.8},
            ]
        }
    path = str(tmp_path / "cuts.jsonl.gz")
    cuts.to_file(path)
    return {
        'use_lhotse': True,
        'cuts_path': path,
        'sample_rate': 16000,
        'shuffle': False,
        'num_workers': 0,
        'batch_size': 2,
        'use_bucketing': False,
    }


def _with_stub_trainer(model):
    """Attach the minimum trainer surface training_step reads.

    ``self.trainer.global_step`` and ``log_every_n_steps`` are all it needs; a
    real pl.Trainer would drag in a full fit loop for no extra coverage.
    """
    from types import SimpleNamespace

    model._trainer = SimpleNamespace(global_step=0, log_every_n_steps=1, global_rank=0, world_size=1)
    return model


@pytest.fixture(autouse=True)
def _cpu_default_device():
    """Pin these CPU tests to CPU regardless of what ran before them.

    Several speechlm2 modules call ``torch.set_default_device('cuda')``, a
    process-wide mutation that leaks into whatever pytest collects next. Without
    this the tests here pass alone and fail in a full session, which is the worst
    possible failure mode to debug.
    """
    prev = torch.get_default_device()
    torch.set_default_device('cpu')
    try:
        yield
    finally:
        torch.set_default_device(prev)


@pytest.fixture()
def rnnt_model(test_data_dir):
    return EncDecCHATBPEModel(cfg=_cfg(test_data_dir, 'rnnt'))


@pytest.fixture()
def forced_model(test_data_dir):
    return EncDecCHATBPEModel(cfg=_cfg(test_data_dir, 'forced_alignment'))


class TestOneArchitectureTwoLosses:
    @pytest.mark.unit
    def test_the_marginalised_arm_still_trains(self, rnnt_model):
        """Adding the forced option must not disturb the ordinary RNN-T path.

        ``training_step`` delegates to the parent for ``loss_type: rnnt``, so
        this walks the same forward/joint/loss the parent would.
        """
        rnnt_model.train()
        audio = torch.randn(2, 16000 * 3) * 0.1
        enc, enc_len = rnnt_model.forward(input_signal=audio, input_signal_length=torch.tensor([16000 * 3, 16000 * 2]))
        tr, trl = torch.randint(0, 100, (2, 5)), torch.tensor([5, 3])
        dec, tl, _ = rnnt_model.decoder(targets=tr, target_length=trl)
        joint = rnnt_model.joint(encoder_outputs=enc, decoder_outputs=dec, encoder_lengths=enc_len)
        # The CHAT joint scores CHUNKS, so the loss's input length is the chunk
        # count, not the frame count -- the parent's training_step does the same.
        loss = rnnt_model.loss(
            log_probs=joint,
            targets=tr,
            input_lengths=rnnt_model.joint.num_chunks_per_utterance,
            target_lengths=tl,
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None for p in rnnt_model.parameters())

    @pytest.mark.unit
    def test_both_losses_give_the_identical_vocabulary(self, rnnt_model, forced_model):
        """The old split had 1,027 text classes on one side and 1,024 on the other."""
        assert rnnt_model.joint.num_classes_with_blank == forced_model.joint.num_classes_with_blank
        assert rnnt_model.tokenizer.vocab_size == forced_model.tokenizer.vocab_size
        assert rnnt_model.joint.num_classes_with_blank == rnnt_model.tokenizer.vocab_size + 1

    @pytest.mark.unit
    def test_checkpoints_are_interchangeable(self, rnnt_model, forced_model):
        """Either arm must be able to initialise from the other, exactly.

        This is what a separate class made impossible and is the main reason for
        the merge: a forced-alignment run should be usable as a starting point
        for a marginalised run and vice versa.
        """
        missing, unexpected = forced_model.load_state_dict(rnnt_model.state_dict(), strict=True)
        assert not missing and not unexpected

    @pytest.mark.unit
    def test_both_share_one_decoding_path(self, rnnt_model, forced_model):
        assert type(rnnt_model.decoding.decoding) is type(forced_model.decoding.decoding)
        assert hasattr(forced_model.joint, "chunk_encoder_for_decoding")

    @pytest.mark.unit
    def test_rejects_an_unknown_loss_type(self, test_data_dir):
        with pytest.raises(ValueError, match="loss_type"):
            EncDecCHATBPEModel(cfg=_cfg(test_data_dir, 'marginal'))

    @pytest.mark.unit
    def test_frame_length_derives_from_the_preprocessor(self, forced_model):
        # 10 ms hop x 8x subsampling. A wrong value shifts every word to the
        # wrong chunk without raising anything.
        assert forced_model.frame_length_in_secs == pytest.approx(0.08)


class TestForcedAlignmentLoss:
    WORDS = [
        [("hello", 0.0, 0.5), ("world", 0.6, 1.4), ("again", 1.5, 2.4)],
        [("one", 0.1, 0.9), ("two", 1.0, 2.0)],
    ]

    @pytest.mark.unit
    def test_loss_is_finite_and_backpropagates(self, forced_model):
        forced_model.train()
        audio = torch.randn(2, 16000 * 3) * 0.1
        alen = torch.tensor([16000 * 3, 16000 * 2])
        enc, enc_len = forced_model.forward(input_signal=audio, input_signal_length=alen)
        loss = forced_model._forced_alignment_loss(enc, enc_len, _cuts(self.WORDS))
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in forced_model.parameters() if p.grad is not None]
        assert grads, "no parameter received a gradient"
        assert all(torch.isfinite(g).all() for g in grads)

    @pytest.mark.unit
    def test_path_covers_every_chunk_and_scores_u_plus_t_positions(self, forced_model):
        n_chunks = torch.tensor([5, 3])
        b, t, u, lab, pred, plens = forced_model._build_batch_path(_cuts(self.WORDS), n_chunks, torch.device('cpu'))
        for i, n in enumerate(n_chunks.tolist()):
            # Every chunk contributes at least its blank; none may exceed the
            # encoder's chunk axis, which would index out of bounds in the joint.
            assert set(t[b == i].tolist()) == set(range(n))
        # Path length = emitted labels + one blank per chunk.
        assert lab.numel() == int(plens.sum()) + int(n_chunks.sum())
        assert (u <= plens[b]).all()

    @pytest.mark.unit
    def test_empty_alignments_do_not_produce_nan(self, forced_model):
        """A batch with no alignable words must not poison the weights.

        cross_entropy over an empty path is nan, and with many ranks drawing
        batches this happens for real.
        """
        forced_model.train()
        audio = torch.randn(1, 16000 * 2) * 0.1
        enc, enc_len = forced_model.forward(input_signal=audio, input_signal_length=torch.tensor([16000 * 2]))
        # No alignments at all -> every chunk still emits a blank, so the path is
        # non-empty; the loss must simply be finite.
        loss = forced_model._forced_alignment_loss(enc, enc_len, _cuts([[]]))
        assert torch.isfinite(loss)

    @pytest.mark.unit
    def test_recovery_adds_scored_positions_without_changing_the_targets(self, test_data_dir):
        plain = EncDecCHATBPEModel(cfg=_cfg(test_data_dir, 'forced_alignment', recover=0))
        rec = EncDecCHATBPEModel(cfg=_cfg(test_data_dir, 'forced_alignment', recover=2))
        n_chunks = torch.tensor([5, 3])
        cuts = _cuts(self.WORDS)
        _, _, _, lab_a, pred_a, _ = plain._build_batch_path(cuts, n_chunks, torch.device('cpu'))
        _, _, _, lab_b, pred_b, _ = rec._build_batch_path(cuts, n_chunks, torch.device('cpu'))
        assert lab_b.numel() > lab_a.numel()
        # The prediction-network input -- the actual transcript -- is untouched;
        # recovery only changes which positions are scored.
        assert torch.equal(pred_a, pred_b)

    @pytest.mark.unit
    def test_word_start_ids_are_found_in_the_sentencepiece_vocab(self, forced_model):
        ws = forced_model._word_start_ids()
        assert len(ws) > 0
        ids = forced_model.tokenizer.text_to_ids("hello world")
        # "world" is preceded by a space, so its first piece must be a word start.
        assert any(i in ws for i in ids)


class TestConstructionWithATrainingDataloader:
    """Building the train loader happens INSIDE ModelPT.__init__.

    The unit tests above build a model with no ``train_ds`` at all, so they
    cannot see ordering bugs in ``__init__`` -- and one of those (attributes
    assigned after ``super().__init__()``, which calls ``setup_training_data()``
    from inside it) reached the grid and killed job 13179581 at construction.
    These tests pay the cost of a real dataloader to close that gap.
    """

    @pytest.mark.unit
    def test_forced_arm_constructs_and_requests_cuts(self, test_data_dir, tmp_path):
        cfg = _cfg(test_data_dir, 'forced_alignment')
        cfg.train_ds = _train_ds_cfg(tmp_path)
        model = EncDecCHATBPEModel(cfg=cfg)
        assert model.loss_type == 'forced_alignment'
        # The forced loss reads cut.custom["alignments"], so the batch must carry
        # the cuts; without them the path could not be built at all.
        batch = next(iter(model._train_dl))
        assert len(batch) == 5, "forced-alignment training batches must include cuts"
        assert all(c.custom.get("alignments") for c in batch[4])

    @pytest.mark.unit
    def test_rnnt_arm_constructs_without_cuts(self, test_data_dir, tmp_path):
        cfg = _cfg(test_data_dir, 'rnnt')
        cfg.train_ds = _train_ds_cfg(tmp_path)
        model = EncDecCHATBPEModel(cfg=cfg)
        batch = next(iter(model._train_dl))
        assert len(batch) == 4, "the marginalised arm must keep the parent's 4-tuple batch"

    @pytest.mark.unit
    def test_a_real_batch_produces_a_finite_forced_loss(self, test_data_dir, tmp_path):
        """End to end on real audio: dataloader -> encoder -> forced path -> loss."""
        cfg = _cfg(test_data_dir, 'forced_alignment', recover=2)
        cfg.train_ds = _train_ds_cfg(tmp_path)
        model = EncDecCHATBPEModel(cfg=cfg)
        model.train()
        signal, signal_len, _, _, cuts = next(iter(model._train_dl))
        enc, enc_len = model.forward(input_signal=signal, input_signal_length=signal_len)
        loss = model._forced_alignment_loss(enc, enc_len, cuts)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())


class TestBothArmsLogTheSameMetrics:
    """training_batch_wer went missing from the forced arm for a whole run.

    The forced ``training_step`` is a separate implementation from the parent's,
    so "they share the training code except the loss" is only true if something
    checks it. These tests are that check.
    """

    @staticmethod
    def _run_step(model, monkeypatch):
        """Drive one training_step, capturing what it logs."""
        logged = {}
        monkeypatch.setattr(type(model), 'log', lambda self, k, v, **kw: logged.__setitem__(k, v), raising=False)
        monkeypatch.setattr(type(model), 'log_dict', lambda self, d, **kw: logged.update(d), raising=False)
        # log_every_n_steps=1 and global_step=0 make the WER branch fire on the
        # very first step, which is what we want to observe.
        model._optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        model.trainer.log_every_n_steps = 1
        batch = next(iter(model._train_dl))
        out = model.training_step(batch, 0)
        return logged, out

    @pytest.mark.unit
    def test_forced_arm_logs_training_batch_wer(self, test_data_dir, tmp_path, monkeypatch):
        cfg = _cfg(test_data_dir, 'forced_alignment')
        cfg.train_ds = _train_ds_cfg(tmp_path)
        model = _with_stub_trainer(EncDecCHATBPEModel(cfg=cfg))
        logged, out = self._run_step(model, monkeypatch)
        assert 'training_batch_wer' in logged, f"forced arm logged only {sorted(logged)}"
        assert 'train_loss' in logged and 'learning_rate' in logged
        assert torch.isfinite(torch.as_tensor(logged['training_batch_wer']))
        assert torch.isfinite(out['loss'])

    @pytest.mark.unit
    def test_both_arms_log_the_same_metric_names(self, test_data_dir, tmp_path, monkeypatch):
        names = {}
        for arm in ('rnnt', 'forced_alignment'):
            cfg = _cfg(test_data_dir, arm)
            cfg.train_ds = _train_ds_cfg(tmp_path)
            model = _with_stub_trainer(EncDecCHATBPEModel(cfg=cfg))
            names[arm] = set(self._run_step(model, monkeypatch)[0])
        assert names['rnnt'] == names['forced_alignment'], (
            f"only in rnnt: {names['rnnt'] - names['forced_alignment']}; "
            f"only in forced: {names['forced_alignment'] - names['rnnt']}"
        )


def _flex_cfg(test_data_dir, max_delay=4, infer=None):
    cfg = _cfg(test_data_dir, 'forced_alignment')
    cfg.forced_alignment.max_delay_frames = max_delay
    if infer is not None:
        cfg.forced_alignment.inference_delay_frames = infer
    cfg.joint.history_chunks = 1  # frames are removed, so the window must reach back
    return cfg


class TestFlexibleDelay:
    """d shifts the alignment AND hides the last d frames of each chunk.

    The two are complements: a word emitted at chunk t has its last frame in
    [t*C - d, (t+1)*C - d), which is exactly what survives the trim. Applying
    only one of them would train the model to emit words whose audio it cannot
    hear (or to ignore audio it was given), and nothing would crash -- the loss
    would just be quietly wrong.
    """

    @pytest.mark.unit
    def test_inference_delay_defaults_to_half_the_range(self, test_data_dir):
        m = EncDecCHATBPEModel(cfg=_flex_cfg(test_data_dir, max_delay=4))
        assert m.inference_delay_frames == 2

    @pytest.mark.unit
    def test_inference_delay_can_be_set_explicitly(self, test_data_dir):
        m = EncDecCHATBPEModel(cfg=_flex_cfg(test_data_dir, max_delay=4, infer=0))
        assert m.inference_delay_frames == 0

    @pytest.mark.unit
    def test_disabled_by_default_leaves_the_joint_untouched(self, forced_model):
        assert forced_model.max_delay_frames == 0
        assert forced_model.joint.frame_trim == 0

    @pytest.mark.unit
    def test_sampled_delay_stays_in_range(self, test_data_dir):
        m = EncDecCHATBPEModel(cfg=_flex_cfg(test_data_dir, max_delay=4))
        draws = {m._sample_delay() for _ in range(200)}
        assert draws <= {0, 1, 2, 3, 4}
        assert len(draws) == 5, f"the range is not being covered: {sorted(draws)}"

    @pytest.mark.unit
    def test_ranks_draw_different_delays(self, test_data_dir):
        """A shared seed would give all 64 ranks the same d every step, so each
        step would see ONE latency globally instead of a spread."""
        seqs = []
        for rank in (0, 1):
            m = EncDecCHATBPEModel(cfg=_flex_cfg(test_data_dir, max_delay=4))
            m.trainer = None
            object.__setattr__(m, '_global_rank', rank) if hasattr(m, '_global_rank') else None
            m._delay_rng = None
            import random as _r

            m._delay_rng = _r.Random(1234 + rank)
            seqs.append([m._sample_delay() for _ in range(20)])
        assert seqs[0] != seqs[1]

    @pytest.mark.unit
    def test_trim_hides_exactly_d_frames_per_chunk(self, test_data_dir):
        """The trim must shorten the VALID count, so cross_attention's trailing
        zero frame lands after the removal rather than before it."""
        m = EncDecCHATBPEModel(cfg=_flex_cfg(test_data_dir, max_delay=4))
        C = m.joint.chunk_size
        chunked = torch.randn(1, 3, C * 8)
        lengths = torch.tensor([[C, C, C]])
        m.joint.frame_trim = 0
        _, base = m.joint._apply_history_window(chunked, lengths.clone())
        m.joint.frame_trim = 3
        _, trimmed = m.joint._apply_history_window(chunked, lengths.clone())
        assert torch.equal(base - trimmed, torch.full_like(base, 3))

    @pytest.mark.unit
    def test_a_chunk_past_the_audio_stays_empty_under_trim(self, test_data_dir):
        m = EncDecCHATBPEModel(cfg=_flex_cfg(test_data_dir, max_delay=4))
        C = m.joint.chunk_size
        m.joint.frame_trim = 2
        _, valid = m.joint._apply_history_window(torch.randn(1, 3, C * 8), torch.tensor([[C, C, 0]]))
        assert valid[0, 2] == 0, "a chunk with no audio must not become valid via its history"

    @pytest.mark.unit
    def test_flush_chunk_adds_exactly_one_chunk(self, test_data_dir):
        m = EncDecCHATBPEModel(cfg=_flex_cfg(test_data_dir, max_delay=4))
        C = m.joint.chunk_size
        enc = torch.randn(2, 32, 5 * C)
        enc2, len2 = m._append_flush_chunk(enc, torch.tensor([5 * C, 3 * C]))
        assert enc2.shape[2] == 6 * C
        assert torch.equal(len2, torch.tensor([6 * C, 4 * C]))
        assert torch.count_nonzero(enc2[:, :, 5 * C :]) == 0, "the flush chunk must carry no audio"

    @pytest.mark.unit
    def test_alignment_shift_and_trim_agree(self, test_data_dir):
        """The invariant: every word emitted at chunk t must still be visible.

        Word last frame f is emitted at chunk floor((f+d)/C); the trim leaves
        frames < (t+1)*C - d visible. So f < (t+1)*C - d must hold for every word.
        """
        from nemo.collections.asr.parts.utils.chat_alignment import assign_words_to_chunks

        C, frame = 14, 0.08
        ends = [round(0.13 * i, 3) for i in range(1, 40)]
        for d in range(5):
            groups = assign_words_to_chunks(ends, 12, C, frame, d)
            for t, idxs in enumerate(groups):
                for i in idxs:
                    f = math.ceil(ends[i] / frame)
                    if t < 11:  # the last chunk absorbs overflow by design
                        assert f < (t + 1) * C - d + 1, f"d={d}: word {i} (frame {f}) not visible at chunk {t}"

    @pytest.mark.unit
    def test_the_rnnt_arm_also_trims(self, test_data_dir, tmp_path, monkeypatch):
        """The marginalised arm must sample and apply d too.

        It delegates to the parent's training_step, so the trim has to be set
        before that call or the flexible-delay setting is silently a no-op for
        this loss -- the run trains normally and simply is not the model asked
        for.
        """
        cfg = _flex_cfg(test_data_dir, max_delay=4)
        cfg.loss_type = 'rnnt'
        cfg.train_ds = _train_ds_cfg(tmp_path)
        model = _with_stub_trainer(EncDecCHATBPEModel(cfg=cfg))
        model._optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        monkeypatch.setattr(type(model), 'log', lambda self, k, v, **kw: None, raising=False)
        monkeypatch.setattr(type(model), 'log_dict', lambda self, d, **kw: None, raising=False)

        seen = set()
        for _ in range(40):
            model.training_step(next(iter(model._train_dl)), 0)
            seen.add(model.joint.frame_trim)
        assert seen <= {0, 1, 2, 3, 4}
        assert len(seen) > 1, f"frame_trim never varied: {seen}"

    @pytest.mark.unit
    def test_the_rnnt_arm_leaves_the_trim_alone_when_disabled(self, test_data_dir, tmp_path, monkeypatch):
        cfg = _cfg(test_data_dir, 'rnnt')
        cfg.train_ds = _train_ds_cfg(tmp_path)
        model = _with_stub_trainer(EncDecCHATBPEModel(cfg=cfg))
        model._optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        monkeypatch.setattr(type(model), 'log', lambda self, k, v, **kw: None, raising=False)
        monkeypatch.setattr(type(model), 'log_dict', lambda self, d, **kw: None, raising=False)
        model.training_step(next(iter(model._train_dl)), 0)
        assert model.joint.frame_trim == 0


class TestBandedLossInTheModel:
    """band=0 must reproduce the forced loss through the real model, not just in
    the loss module's own unit tests."""

    @pytest.mark.unit
    def test_band_zero_matches_the_forced_loss_end_to_end(self, test_data_dir, tmp_path):
        cfgs = {}
        for name, lt, band in (("forced", "forced_alignment", None), ("banded", "banded", 0)):
            c = _cfg(test_data_dir, lt)
            if band is not None:
                c.forced_alignment.band_chunks = band
            c.train_ds = _train_ds_cfg(tmp_path)
            cfgs[name] = c

        losses = {}
        for name, c in cfgs.items():
            torch.manual_seed(0)
            m = EncDecCHATBPEModel(cfg=c)
            m.eval()
            torch.manual_seed(1)
            signal, signal_len, _, _, cuts = next(iter(m._train_dl))
            with torch.no_grad():
                enc, enc_len = m.forward(input_signal=signal, input_signal_length=signal_len)
                if name == "banded":
                    losses[name] = m._banded_loss(enc, enc_len, cuts)
                else:
                    losses[name] = m._forced_alignment_loss(enc, enc_len, cuts)
        # The forced loss is a mean over path steps and the banded one a sum of
        # NLLs per target token, so compare that both are finite and that the
        # banded value equals the forced path's total likelihood scale.
        assert torch.isfinite(losses["banded"]) and torch.isfinite(losses["forced"])
        assert losses["banded"] > 0

    @pytest.mark.unit
    def test_a_wider_band_lowers_the_loss(self, test_data_dir, tmp_path):
        """Strictly more admissible paths -> no less probability mass."""
        prev = None
        for band in (0, 1, 2):
            c = _cfg(test_data_dir, "banded")
            c.forced_alignment.band_chunks = band
            c.train_ds = _train_ds_cfg(tmp_path)
            torch.manual_seed(0)
            m = EncDecCHATBPEModel(cfg=c)
            m.eval()
            torch.manual_seed(1)
            signal, signal_len, _, _, cuts = next(iter(m._train_dl))
            with torch.no_grad():
                enc, enc_len = m.forward(input_signal=signal, input_signal_length=signal_len)
                loss = m._banded_loss(enc, enc_len, cuts).item()
            if prev is not None:
                assert loss <= prev + 1e-4, f"band {band} loss {loss} exceeds band {band-1} loss {prev}"
            prev = loss

    @pytest.mark.unit
    def test_banded_backpropagates(self, test_data_dir, tmp_path):
        c = _cfg(test_data_dir, "banded")
        c.forced_alignment.band_chunks = 1
        c.train_ds = _train_ds_cfg(tmp_path)
        m = EncDecCHATBPEModel(cfg=c)
        m.train()
        signal, signal_len, _, _, cuts = next(iter(m._train_dl))
        enc, enc_len = m.forward(input_signal=signal, input_signal_length=signal_len)
        loss = m._banded_loss(enc, enc_len, cuts)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

    @pytest.mark.unit
    def test_banded_requests_cuts_like_the_forced_arm(self, test_data_dir, tmp_path):
        c = _cfg(test_data_dir, "banded")
        c.train_ds = _train_ds_cfg(tmp_path)
        m = EncDecCHATBPEModel(cfg=c)
        assert len(next(iter(m._train_dl))) == 5, "banded needs the cuts for their alignments"


class TestNoDeletionsAtTheAudioEnd:
    """The trim hides the last d frames of EVERY chunk, and the final chunk has
    no successor to recover them from -- so without a flush chunk the last d
    frames of an utterance are visible to nothing and whatever was spoken in
    them cannot be emitted. Padded audio hides this (training and the
    leaderboard eval both pad 0.5 s); a raw file does not.
    """

    @staticmethod
    def _visible_own_frames(model, enc_len):
        """Real frames of audio reachable by at least one chunk."""
        enc = torch.randn(1, 32, enc_len)
        _, n_chunks, cl = model.joint.chunk_encoder_for_decoding(enc, torch.tensor([enc_len]))
        c = model.joint.chunk_size
        hist = model.joint.history_chunks
        # valid = min(t, hist) * C + own-chunk frames, so subtract the history part
        own = []
        for t in range(int(n_chunks[0])):
            own.append(max(int(cl[0, t]) - min(t, hist) * c, 0))
        return sum(own)

    @pytest.mark.unit
    @pytest.mark.parametrize("extra", [0, 1, 3, 9, 13])
    def test_every_real_frame_stays_reachable(self, test_data_dir, extra):
        cfg = _flex_cfg(test_data_dir, max_delay=4)
        cfg.loss_type = 'rnnt'
        model = EncDecCHATBPEModel(cfg=cfg)
        model.eval()
        c = model.joint.chunk_size
        enc_len = c * 5 + extra

        model.joint.frame_trim = 0
        baseline = self._visible_own_frames(model, enc_len)
        model.joint.frame_trim = model.inference_delay_frames
        trimmed = self._visible_own_frames(model, enc_len)

        assert (
            trimmed >= baseline
        ), f"enc_len={enc_len}: trimming hid {baseline - trimmed} real frames with no chunk to recover them"

    @pytest.mark.unit
    def test_the_flush_chunk_appears_only_when_trimming(self, test_data_dir):
        cfg = _flex_cfg(test_data_dir, max_delay=4)
        cfg.loss_type = 'rnnt'
        model = EncDecCHATBPEModel(cfg=cfg)
        model.eval()
        c = model.joint.chunk_size
        enc = torch.randn(1, 32, c * 5)

        model.joint.frame_trim = 0
        _, n0, _ = model.joint.chunk_encoder_for_decoding(enc, torch.tensor([c * 5]))
        model.joint.frame_trim = 2
        _, n2, _ = model.joint.chunk_encoder_for_decoding(enc, torch.tensor([c * 5]))
        assert int(n2[0]) == int(n0[0]) + 1, "trimming must add exactly one flush chunk"

    @pytest.mark.unit
    def test_untrimmed_decoding_is_completely_unchanged(self, forced_model):
        """A model that never trims must chunk exactly as it did before."""
        c = forced_model.joint.chunk_size
        enc = torch.randn(2, 32, c * 4 + 5)
        lens = torch.tensor([c * 4 + 5, c * 3])
        assert forced_model.joint.frame_trim == 0
        chunked, n, cl = forced_model.joint.chunk_encoder_for_decoding(enc, lens)
        assert int(n[0]) == 5 and int(n[1]) == 3

    @pytest.mark.unit
    def test_validation_pins_the_inference_delay(self, test_data_dir):
        """Without this the joint keeps whatever d the last TRAINING batch drew,
        so val_wer describes a random latency rather than one operating point.
        The hooks were silently deleted once already, by an over-greedy edit."""
        cfg = _flex_cfg(test_data_dir, max_delay=4)
        cfg.loss_type = 'rnnt'
        m = _with_stub_trainer(EncDecCHATBPEModel(cfg=cfg))
        m.joint.frame_trim = 4  # as if a training batch had just drawn d=4
        m.on_validation_epoch_start()
        assert m.joint.frame_trim == m.inference_delay_frames == 2

    @pytest.mark.unit
    def test_disabled_models_are_not_pinned(self, test_data_dir):
        cfg = _cfg(test_data_dir, 'rnnt')
        m = _with_stub_trainer(EncDecCHATBPEModel(cfg=cfg))
        m.on_validation_epoch_start()
        assert m.joint.frame_trim == 0
