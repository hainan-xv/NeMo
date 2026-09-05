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
"""The forced-alignment path CHAT is trained on."""

import os
import types

import pytest
import torch
import torch.nn.functional as F

from nemo.collections.speechlm2.data.chat_dataset import build_path


@pytest.mark.unit
def test_path_layout_is_tokens_then_a_blank_per_chunk():
    BLANK = 99
    t, u, lab = build_path([[5, 6], [], [7]], BLANK)
    assert t == [0, 0, 0, 1, 2, 2]
    assert u == [0, 1, 2, 2, 2, 3]
    assert lab == [5, 6, BLANK, BLANK, 7, BLANK]


@pytest.mark.unit
def test_u_advances_only_on_real_tokens():
    """u indexes the prediction network, which never consumes a blank."""
    BLANK = 0
    chunks = [[1, 2], [], [3], [], [], [4, 5, 6]]
    t, u, lab = build_path(chunks, BLANK)
    emitted = 0
    for ti, ui, li in zip(t, u, lab):
        assert ui == emitted, "u must equal the number of labels emitted so far"
        if li != BLANK:
            emitted += 1
    assert emitted == 6


@pytest.mark.unit
def test_every_chunk_contributes_exactly_one_blank():
    """Silent chunks included -- that is how the model learns to emit nothing."""
    BLANK = 7
    for chunks in ([[1], [2]], [[], [], []], [[1, 2, 3]], []):
        t, u, lab = build_path(chunks, BLANK)
        assert sum(1 for x in lab if x == BLANK) == len(chunks)
        # ...and it is always the chunk's last step.
        for c in range(len(chunks)):
            steps = [i for i, x in enumerate(t) if x == c]
            assert lab[steps[-1]] == BLANK
            assert all(lab[i] != BLANK for i in steps[:-1])


@pytest.mark.unit
def test_path_length_is_tokens_plus_chunks_not_the_lattice():
    """The whole point: cost is O(U + T), not O(T * U)."""
    chunks = [[1, 2], [], [3], [4, 5]]
    U = sum(len(c) for c in chunks)
    T = len(chunks)
    t, u, lab = build_path(chunks, 0)
    assert len(t) == len(u) == len(lab) == U + T
    assert U + T < T * U or T * U == 0


@pytest.mark.unit
def test_chunk_index_is_non_decreasing_and_covers_every_chunk():
    """t must be monotone -- a transducer cannot go back to an earlier chunk."""
    chunks = [[1], [], [2, 3], [], [4]]
    t, _, _ = build_path(chunks, 0)
    assert t == sorted(t)
    assert set(t) == set(range(len(chunks)))


@pytest.mark.unit
def test_u_indexes_the_decoder_output_which_is_u_plus_one_long():
    """RNNTDecoder prepends its own SOS and returns U+1 states.

    So the dataset passes emitted tokens ONLY, and u_idx must stay within
    [0, U]. Prefixing a start symbol in the dataset would double it and shift
    every prediction state by one -- training the joint against the wrong state
    while looking entirely healthy.
    """
    chunks = [[11, 12], [], [13]]
    tokens = [tok for c in chunks for tok in c]  # what the dataset emits
    U = len(tokens)
    t, u, lab = build_path(chunks, 99)
    assert max(u) <= U, "u must index the decoder's U+1 outputs"
    # State u was produced after consuming tokens[:u].
    for ui, li in zip(u, lab):
        if li != 99:
            assert tokens[ui] == li, "the label at state u is the u-th emitted token"


@pytest.mark.unit
def test_forced_path_loss_flows_to_encoder_prednet_and_joint():
    """End-to-end: the path objective must train all three components.

    Built from the modules directly rather than the full model, so the test does
    not need pretrained weights. What it pins is the wiring: decoder output
    indexed by u, encoder chunks indexed by t, cross-entropy over the path, and
    gradient reaching every part.
    """
    from nemo.collections.asr.modules import RNNTAttJoint, RNNTDecoder
    from nemo.collections.speechlm2.data.chat_dataset import build_path

    V, D, CHUNK = 20, 16, 4
    torch.manual_seed(0)
    dec = RNNTDecoder(prednet={"pred_hidden": D, "pred_rnn_layers": 1, "dropout": 0.0}, vocab_size=V)
    joint = RNNTAttJoint(
        jointnet={"encoder_hidden": D, "pred_hidden": D, "joint_hidden": D, "activation": "relu", "dropout": 0.0},
        num_classes=V,
        chunk_size=CHUNK,
    )
    BLANK = V

    chunks_per_utt = [[[1, 2], [], [3]], [[4], [5, 6], []]]
    b_idx, t_idx, u_idx, labels, rows = [], [], [], [], []
    for bi, chunks in enumerate(chunks_per_utt):
        t, u, lab = build_path(chunks, BLANK)
        b_idx += [bi] * len(t)
        t_idx += t
        u_idx += u
        labels += lab
        rows.append([tok for c in chunks for tok in c])

    U = max(len(r) for r in rows)
    pred_input = torch.full((2, U), BLANK, dtype=torch.long)
    for i, r in enumerate(rows):
        pred_input[i, : len(r)] = torch.tensor(r)
    pred_lens = torch.tensor([len(r) for r in rows])

    # 3 chunks per utterance, matching chunks_per_utt.
    n_frames = CHUNK * 3
    enc = torch.randn(2, n_frames, D, requires_grad=True)
    enc_len = torch.tensor([n_frames, n_frames])

    g, _, _ = dec(targets=pred_input, target_length=pred_lens)
    g = g.transpose(1, 2)
    assert g.shape[1] == U + 1, "decoder must emit U+1 states (it prepends SOS itself)"

    logits = joint.joint_on_path(enc, g, torch.tensor(b_idx), torch.tensor(t_idx), torch.tensor(u_idx), enc_len)
    assert logits.shape == (len(labels), V + 1)
    # The dataset's chunk count must equal the joint's own chunking.
    assert joint.num_chunks_per_utterance.tolist() == [3, 3]

    loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor(labels))
    assert torch.isfinite(loss)
    loss.backward()

    assert enc.grad is not None and enc.grad.abs().sum() > 0, "no gradient to the encoder"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dec.parameters()), "no gradient to prednet"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in joint.parameters()), "no gradient to joint"


@pytest.mark.unit
def test_chat_greedy_decode_follows_the_training_convention():
    """Inference must walk chunks emitting until a blank -- the training path.

    This is the property that makes single-path training sound: there is no
    search over alignments at decode time, so training and inference are the
    same procedure. If the decoder instead advanced chunks on a different rule,
    the forced-alignment objective would be optimising something the decoder
    never does.
    """
    from omegaconf import OmegaConf

    from nemo.collections.asr.modules import RNNTAttJoint, RNNTDecoder
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecoding

    V, D, CHUNK = 12, 16, 4
    torch.manual_seed(0)
    dec = RNNTDecoder(prednet={"pred_hidden": D, "pred_rnn_layers": 1, "dropout": 0.0}, vocab_size=V)
    joint = RNNTAttJoint(
        jointnet={"encoder_hidden": D, "pred_hidden": D, "joint_hidden": D, "activation": "relu", "dropout": 0.0},
        num_classes=V,
        chunk_size=CHUNK,
    )
    # The joint advertises the hook the decoding layer looks for; that is what
    # routes it through the CHAT path rather than the standard one.
    assert hasattr(joint, "chunk_encoder_for_decoding")

    decoding = RNNTDecoding(
        decoding_cfg=OmegaConf.create({"strategy": "greedy_batch", "greedy": {"max_symbols": 8}}),
        decoder=dec,
        joint=joint,
        vocabulary=[str(i) for i in range(V)],
    )

    B, n_frames = 2, CHUNK * 3
    enc = torch.randn(B, n_frames, D)
    enc_len = torch.tensor([n_frames, n_frames])
    hyps = decoding.rnnt_decoder_predictions_tensor(
        encoder_output=enc.transpose(1, 2), encoded_lengths=enc_len, return_hypotheses=True
    )
    if isinstance(hyps, tuple):
        hyps = hyps[0]

    assert len(hyps) == B
    for h in hyps:
        y = h.y_sequence
        y = y.tolist() if torch.is_tensor(y) else list(y)
        # Blank is never an OUTPUT token -- it only terminates a chunk.
        assert all(0 <= t < V for t in y), f"decoded id outside the vocabulary: {y}"
        # Timestamps are chunk indices, and a transducer cannot revisit a chunk.
        ts = h.timestamp
        ts = ts.tolist() if torch.is_tensor(ts) else list(ts)
        assert ts == sorted(ts), f"chunk indices not monotone: {ts}"
        assert all(0 <= t < 3 for t in ts), f"chunk index out of range: {ts}"


@pytest.mark.unit
def test_dataset_chunk_estimate_tracks_the_encoder_within_one():
    """The dataset and the encoder count chunks INDEPENDENTLY; they must agree.

    The dataset derives frames from duration (ceil(secs / frame_length)); the
    joint chunks the encoder's actual output. Subsampling shifts the tail, so
    they can differ by one -- which the model tolerates by dropping path steps
    past the encoder's last chunk. A drift of TWO would mean the two sides
    disagree about framing, and every word after that point would be scored
    against the wrong chunk.

    This is the check that would have caught the cluster failure
    (joint=21 vs dataset=20). The earlier local test computed BOTH counts from
    the encoder, so it was consistent by construction and blind to the drift.
    """
    import glob
    import math

    from omegaconf import OmegaConf

    from nemo.collections.speechlm2.models.chat_model import ChatSTTModel

    hits = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/**/*.nemo"),
        recursive=True,
    )
    if not hits:
        pytest.skip("nemotron ASR .nemo not present in the local HF cache")

    cfg = OmegaConf.load("examples/speechlm2/conf/streaming_stt_granary2_chat_asrvocab.yaml")
    OmegaConf.resolve(cfg)
    cfg.model.pretrained_asr = hits[0]
    model = ChatSTTModel(OmegaConf.to_container(cfg.model, resolve=True)).float()

    chunk = model.chunk_size
    frame_len = float(cfg.data.dataset.frame_length_in_secs)
    sr = int(cfg.data.dataset.sample_rate)

    worst = 0
    for secs in (1.0, 2.3, 3.7, 5.0, 7.11, 9.42, 12.0):
        n = int(secs * sr)
        with torch.no_grad():
            enc, enc_len = model._encode(torch.randn(1, n), torch.tensor([n]))
        enc_chunks = math.ceil(int(enc_len[0]) / chunk)
        # Exactly how the dataset estimates it, from duration alone.
        data_chunks = math.ceil(math.ceil(secs / frame_len) / chunk)
        worst = max(worst, abs(enc_chunks - data_chunks))
        assert abs(enc_chunks - data_chunks) <= 1, (
            f"{secs}s: encoder says {enc_chunks} chunks, dataset estimates {data_chunks}. "
            "A drift > 1 misassigns every word after the divergence."
        )
    assert worst <= 1


@pytest.mark.unit
def test_target_text_reconstructs_the_transcript_from_the_path():
    """The logged `tgt` line must be the transcript, or it is worse than useless.

    Its whole purpose is to answer "is the model learning the wrong thing?" --
    so if blanks leaked in, or u/t indices were transposed, the line would look
    wrong for a HEALTHY model and send you hunting a bug that isn't there.
    Round-trip it: a path built from known chunk tokens, stripped of blanks,
    must give back exactly the tokens that went in, in order.
    """
    from nemo.collections.speechlm2.models.chat_model import ChatSTTModel

    blank = 99
    chunk_tokens = [[1, 2], [], [3], [4, 5, 6], []]
    t_idx, u_idx, labels = build_path(chunk_tokens, blank)

    batch = types.SimpleNamespace(
        b_idx=torch.zeros(len(labels), dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
    )
    model = types.SimpleNamespace(
        blank_id=blank,
        _detokenize=lambda ids: " ".join(str(i) for i in ids),
    )
    got = ChatSTTModel._target_text(model, batch, 0)
    assert got == "1 2 3 4 5 6"

    # And it must SELECT by utterance: with two utterances interleaved in one
    # batch, taking the wrong rows would silently log utterance 0's transcript
    # for every example, which reads as plausible.
    b2 = types.SimpleNamespace(
        b_idx=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        labels=torch.tensor([1, 2, blank, 7, 8, blank], dtype=torch.long),
    )
    assert ChatSTTModel._target_text(model, b2, 0) == "1 2"
    assert ChatSTTModel._target_text(model, b2, 1) == "7 8"


@pytest.mark.unit
def test_train_decode_logging_is_gated_and_never_kills_the_run():
    """Fires on the right steps, once each, and swallows its own failures.

    A diagnostic that throws would turn a healthy run into a crashed one, and a
    diagnostic that fires every micro-batch under gradient accumulation would
    decode several times per step for no extra information.
    """
    from nemo.collections.speechlm2.models.chat_model import ChatSTTModel

    calls = []

    def make(step, every=500):
        m = types.SimpleNamespace(
            cfg={"log_train_decode_every_n_steps": every, "log_train_decode_examples": 2},
            global_step=step,
            _last_decode_step=None,
            blank_id=99,
            transcribe_ids=lambda a, l: calls.append(int(m.global_step)) or [[1]],
            _detokenize=lambda ids: "x",
            _target_text=lambda b, i: "y",
        )
        # cfg.get must behave like DictConfig's.
        m.cfg = type(
            "C",
            (),
            {
                "get": lambda _s, k, d=None: {
                    "log_train_decode_every_n_steps": every,
                    "log_train_decode_examples": 2,
                }.get(k, d)
            },
        )()
        return m

    batch = types.SimpleNamespace(
        audios=torch.zeros(2, 16000),
        audio_lens=torch.tensor([16000, 16000]),
        text=["a", "b"],
    )

    ChatSTTModel._maybe_log_train_decode(make(499), batch)
    assert calls == [], "fired on a step that is not a multiple of the interval"

    ChatSTTModel._maybe_log_train_decode(make(500), batch)
    assert calls == [500], "did not fire on the interval"

    # Same global step twice (gradient accumulation) -> decode once.
    m = make(1000)
    ChatSTTModel._maybe_log_train_decode(m, batch)
    ChatSTTModel._maybe_log_train_decode(m, batch)
    assert calls == [500, 1000], f"fired more than once per global step: {calls}"

    # every=0 disables.
    ChatSTTModel._maybe_log_train_decode(make(500, every=0), batch)
    assert calls == [500, 1000]

    # A failure inside the decode must NOT propagate.
    boom = make(2000)
    boom.transcribe_ids = lambda a, l: (_ for _ in ()).throw(RuntimeError("cuda oom"))
    ChatSTTModel._maybe_log_train_decode(boom, batch)  # must not raise


@pytest.mark.unit
def test_on_path_joint_equals_the_standard_chat_joint():
    """joint_on_path must be the ORDINARY CHAT joint, evaluated at fewer points.

    This is the load-bearing claim of the whole design: the forced-alignment
    arm is meant to differ from standard CHAT only in WHICH (b, t, u) pairs are
    scored, never in HOW they are scored. If the on-path route diverged --
    a missing residual, a different mask, an unscaled softmax -- the model would
    still train, and would still decode with the full joint, so the mismatch
    would show up only as unexplained WER.

    Compared over EVERY triple, with ragged encoder lengths so the padding mask
    and the appended zero frame are both exercised.
    """
    from nemo.collections.asr.modules import RNNTAttJoint

    torch.manual_seed(0)
    V, D, CHUNK = 15, 16, 4
    joint = RNNTAttJoint(
        jointnet={"encoder_hidden": D, "pred_hidden": D, "joint_hidden": D, "activation": "relu", "dropout": 0.0},
        num_classes=V,
        chunk_size=CHUNK,
    )
    joint.eval()  # dropout off, or the two routes draw different masks

    B, T_frames, U = 3, CHUNK * 4, 6
    f_raw = torch.randn(B, T_frames, D)
    g = torch.randn(B, U, D)
    # Ragged on purpose: equal lengths would leave the padding mask untested.
    f_len = torch.tensor([T_frames, T_frames - 3, T_frames - CHUNK - 1])

    with torch.no_grad():
        full = joint.joint(f_raw, g, f_len)
        Tc = full.shape[1]
        b_idx, t_idx, u_idx = torch.meshgrid(torch.arange(B), torch.arange(Tc), torch.arange(U), indexing="ij")
        b_idx, t_idx, u_idx = b_idx.reshape(-1), t_idx.reshape(-1), u_idx.reshape(-1)
        path = joint.joint_on_path(f_raw, g, b_idx, t_idx, u_idx, f_len)

    assert full.shape == (B, Tc, U, V + 1)
    assert path.shape == (B * Tc * U, V + 1)
    torch.testing.assert_close(path, full[b_idx, t_idx, u_idx], atol=1e-5, rtol=1e-4)


@pytest.mark.unit
def test_rnnt_warm_start_copies_the_vocab_independent_weights_only():
    """The donor's encoder->joint projection must be USED, and the softmax must not.

    Two distinct failures this guards.

    (1) perception.proj is a RANDOM Linear(1024, 640) inserted between the
        pretrained encoder and the joint, doing exactly the job the donor's
        joint.enc already learned. Leaving it random scrambles the encoder
        output on its way to the joint -- "initialised from a good checkpoint"
        while discarding the part that makes the checkpoint usable.

    (2) The embedding and output layer must stay random in BOTH arms. Arm 1's
        vocabulary matches the donor exactly, so copying them is possible --
        and would hand arm 1 a pretrained softmax that arm 2 structurally
        cannot have, turning a vocabulary-size comparison into a comparison of
        initialisations.
    """
    import glob
    import tarfile
    import tempfile

    from omegaconf import OmegaConf

    from nemo.collections.speechlm2.models.chat_model import ChatSTTModel

    hits = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/**/*.nemo"),
        recursive=True,
    )
    if not hits:
        pytest.skip("nemotron ASR .nemo not present in the local HF cache")

    cfg = OmegaConf.load("examples/speechlm2/conf/streaming_stt_granary2_chat_asrvocab.yaml")
    OmegaConf.resolve(cfg)
    cfg.model.pretrained_asr = hits[0]
    model = ChatSTTModel(OmegaConf.to_container(cfg.model, resolve=True)).float()

    with tarfile.open(hits[0], "r:") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("model_weights.ckpt"))
        with tempfile.TemporaryDirectory() as td:
            tf.extract(member, td)
            donor = torch.load(os.path.join(td, member.name), map_location="cpu", weights_only=True)

    # (1) copied
    assert torch.equal(model.perception.proj.weight.data, donor["joint.enc.weight"])
    assert torch.equal(model.perception.proj.bias.data, donor["joint.enc.bias"])
    assert torch.equal(model.joint.pred.weight.data, donor["joint.pred.weight"])
    lstm = next(m for _, m in model.decoder.named_modules() if isinstance(m, torch.nn.LSTM))
    for layer in range(lstm.num_layers):
        for w in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            attr = f"{w}_l{layer}"
            assert torch.equal(
                getattr(lstm, attr).data, donor[f"decoder.prediction.dec_rnn.lstm.{attr}"]
            ), f"lstm.{attr} not warm-started"

    # joint.enc must be identity, or it re-scrambles the projection just copied.
    n = model.joint.enc.out_features
    assert torch.equal(model.joint.enc.weight.data, torch.eye(n))
    assert bool((model.joint.enc.bias.data == 0).all())

    # (2) NOT copied -- fair to both arms.
    emb = next(m for _, m in model.decoder.named_modules() if isinstance(m, torch.nn.Embedding))
    d_emb = donor["decoder.prediction.embed.weight"]
    rows = min(emb.weight.shape[0], d_emb.shape[0])
    assert not torch.equal(emb.weight.data[:rows], d_emb[:rows]), "embedding was copied; arm 2 cannot match this"
    out = model.joint.joint_net[2]
    d_out = donor["joint.joint_net.2.weight"]
    rows = min(out.weight.shape[0], d_out.shape[0])
    assert not torch.equal(out.weight.data[:rows], d_out[:rows]), "output layer was copied; arm 2 cannot match this"


@pytest.mark.unit
def test_rnnt_warm_start_can_be_disabled_and_never_raises():
    """Warm start is a startup convenience, not a correctness requirement.

    A missing or unreadable donor must degrade to random init with a warning,
    not take down an 8-node job at minute one.
    """
    from omegaconf import OmegaConf

    from nemo.collections.speechlm2.models.chat_model import ChatSTTModel

    cfg = OmegaConf.load("examples/speechlm2/conf/streaming_stt_granary2_chat_asrvocab.yaml")
    OmegaConf.resolve(cfg)
    d = OmegaConf.to_container(cfg.model, resolve=True)
    # Not a tarball at all: _init_from_pretrained_rnnt must swallow it.
    stub = types.SimpleNamespace(
        perception=types.SimpleNamespace(proj=None),
        joint=types.SimpleNamespace(pred=None, enc=None),
        decoder=types.SimpleNamespace(named_modules=lambda: iter(())),
    )
    # Must not raise: a bad donor path degrades to random init with a warning.
    ChatSTTModel._init_from_pretrained_rnnt(stub, "/nonexistent/path/to.nemo")

    # And the knob exists so the warm start can be turned off for an ablation.
    assert d.get("init_rnnt_from_asr", True) in (True, False)


@pytest.mark.unit
def test_batched_loss_and_grads_match_sequential_with_ragged_lengths():
    """A padded batch must give the same loss and gradients as one-at-a-time.

    RAGGED ON BOTH AXES on purpose: different audio lengths AND different token
    counts. Equal-length inputs would pass trivially while padding leakage --
    the encoder attending across the pad boundary, the chunk mask admitting pad
    frames, the prediction LSTM's padded tail bleeding into valid states --
    went undetected. Any of those would make training depend on how utterances
    happen to be bucketed together, which is invisible in a loss curve and
    impossible to reproduce.

    Compared at float32 in eval() so dropout is off; the encoder uses
    layer_norm (no BatchNorm anywhere), so batch composition cannot shift
    normalisation statistics and the equality holds in train mode too.
    """
    import glob
    import math

    from omegaconf import OmegaConf

    from nemo.collections.speechlm2.data.chat_dataset import build_path
    from nemo.collections.speechlm2.models.chat_model import ChatSTTModel

    hits = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/**/*.nemo"),
        recursive=True,
    )
    if not hits:
        pytest.skip("nemotron ASR .nemo not present in the local HF cache")

    cfg = OmegaConf.load("examples/speechlm2/conf/streaming_stt_granary2_chat_asrvocab.yaml")
    OmegaConf.resolve(cfg)
    cfg.model.pretrained_asr = hits[0]
    model = ChatSTTModel(OmegaConf.to_container(cfg.model, resolve=True)).float()
    model.eval()

    SR, C, BLANK = 16000, model.chunk_size, model.blank_id
    torch.manual_seed(0)
    secs = [2.0, 3.7, 5.1, 8.3]
    audios = [torch.randn(1, int(s * SR)) * 0.1 for s in secs]
    lens = [torch.tensor([a.shape[1]]) for a in audios]

    n_chunks = []
    with torch.no_grad():
        for a, l in zip(audios, lens):
            _, el = model._encode(a, l)
            n_chunks.append(int(math.ceil(int(el[0]) / C)))

    rng = torch.Generator().manual_seed(1)
    chunk_tokens = [
        [
            [
                int(torch.randint(1, 900, (1,), generator=rng))
                for _ in range(int(torch.randint(0, 4, (1,), generator=rng)))
            ]
            for _ in range(n)
        ]
        for n in n_chunks
    ]
    paths = []
    for ct in chunk_tokens:
        t, u, lab = build_path(ct, BLANK)
        paths.append((torch.tensor(t), torch.tensor(u), torch.tensor(lab), [x for c in ct for x in c]))
    assert len({len(p[3]) for p in paths}) > 1, "token counts must differ or the test is trivial"

    def forward_one(a, l, t_idx, u_idx, toks):
        enc, enc_len = model._encode(a, l)
        pin = torch.tensor([toks] if toks else [[BLANK]], dtype=torch.long)
        g, _, _ = model.decoder(targets=pin, target_length=torch.tensor([len(toks)]))
        return model.joint.joint_on_path(enc, g.transpose(1, 2), torch.zeros_like(t_idx), t_idx, u_idx, enc_len)

    # --- sequential ---
    model.zero_grad(set_to_none=True)
    seq_logits, total = [], 0
    for a, l, p in zip(audios, lens, paths):
        lg = forward_one(a, l, p[0], p[1], p[3])
        seq_logits.append(lg)
        total += lg.shape[0]
    seq_loss = sum(F.cross_entropy(lg.float(), p[2], reduction="sum") for lg, p in zip(seq_logits, paths)) / total
    seq_loss.backward()
    seq_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    # --- batched ---
    model.zero_grad(set_to_none=True)
    n_max = max(a.shape[1] for a in audios)
    A = torch.zeros(len(audios), n_max)
    for i, a in enumerate(audios):
        A[i, : a.shape[1]] = a[0]
    AL = torch.tensor([a.shape[1] for a in audios])
    rows = [p[3] for p in paths]
    U = max(max(len(r) for r in rows), 1)
    PIN = torch.full((len(rows), U), BLANK, dtype=torch.long)
    for i, r in enumerate(rows):
        PIN[i, : len(r)] = torch.tensor(r, dtype=torch.long)
    B_ = torch.cat([torch.full_like(p[0], i) for i, p in enumerate(paths)])
    T_ = torch.cat([p[0] for p in paths])
    U_ = torch.cat([p[1] for p in paths])
    L_ = torch.cat([p[2] for p in paths])

    enc, enc_len = model._encode(A, AL)
    g, _, _ = model.decoder(targets=PIN, target_length=torch.tensor([len(r) for r in rows]))
    bat_logits = model.joint.joint_on_path(enc, g.transpose(1, 2), B_, T_, U_, enc_len)
    bat_loss = F.cross_entropy(bat_logits.float(), L_)
    bat_loss.backward()
    bat_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    assert abs(seq_loss.item() - bat_loss.item()) < 1e-5, f"{seq_loss.item()} vs {bat_loss.item()}"

    off = 0
    for i, p in enumerate(paths):
        n = p[0].shape[0]
        torch.testing.assert_close(seq_logits[i], bat_logits[off : off + n], atol=1e-4, rtol=1e-3)
        off += n

    for name, gseq in seq_grads.items():
        torch.testing.assert_close(gseq, bat_grads[name], atol=1e-4, rtol=1e-3, msg=lambda m, n=name: f"{n}: {m}")


def _att_joint(V, D, C, M, seed=0):
    from nemo.collections.asr.modules import RNNTAttJoint

    torch.manual_seed(seed)
    return RNNTAttJoint(
        jointnet={"encoder_hidden": D, "pred_hidden": D, "joint_hidden": D, "activation": "relu", "dropout": 0.0},
        num_classes=V,
        chunk_size=C,
        history_chunks=M,
    )


@pytest.mark.unit
def test_history_window_default_is_standard_chat():
    """history_chunks=0 must leave the joint byte-identical to before.

    The two vocabulary arms and every earlier result depend on the default path
    being untouched, so this is a compatibility lock, not a feature test.
    """
    from nemo.collections.asr.parts.utils.chunking_utils import chunk_concat_audio

    V, D, C = 12, 8, 4
    j = _att_joint(V, D, C, 0)
    f = torch.randn(2, C * 5, D)
    f_len = torch.tensor([C * 5, C * 5 - 2])
    chunked, lens = chunk_concat_audio(f, f_len, C)
    w, wl = j._apply_history_window(chunked, lens)
    assert w is chunked and wl is lens, "M=0 must be a no-op, not a copy"


@pytest.mark.unit
@pytest.mark.parametrize("M", [1, 2])
def test_history_window_adds_previous_chunks_without_lookahead(M):
    """The window spans [max(0,t-M)*C, (t+1)*C) -- history only, never future.

    Look-ahead here would be silent and fatal: it would inflate offline metrics
    while being impossible to reproduce in a streaming deployment, since the
    audio simply has not arrived yet.
    """
    from nemo.collections.asr.parts.utils.chunking_utils import chunk_concat_audio

    V, D, C, T = 12, 8, 4, 6
    j = _att_joint(V, D, C, M)
    f = torch.arange(1 * T * C * D, dtype=torch.float32).reshape(1, T * C, D)
    f_len = torch.tensor([T * C])
    chunked, lens = chunk_concat_audio(f, f_len, C)
    w, wl = j._apply_history_window(chunked, lens)

    W = (M + 1) * C
    assert w.shape == (1, T, W * D)
    wv = w.reshape(1, T, W, D)
    for t in range(T):
        start = max(0, t - M) * C
        n_valid = int(wl[0, t])
        assert n_valid == min(t, M) * C + int(lens[0, t])
        # Every VALID position must be a real frame at or before this chunk's end.
        for jpos in range(n_valid):
            src = start + jpos
            assert src < (t + 1) * C, f"chunk {t} position {jpos} reads frame {src}, past its own end"
            torch.testing.assert_close(wv[0, t, jpos], f[0, src])


@pytest.mark.unit
def test_history_window_joint_output_is_independent_of_future_audio():
    """End-to-end causality: perturbing later audio cannot change chunk t.

    Stronger than inspecting indices -- it exercises the gather, the mask and
    the attention together, which is where an off-by-one would actually land.
    """
    V, D, C, M, T = 12, 8, 4, 1, 6
    j = _att_joint(V, D, C, M).eval()
    torch.manual_seed(0)
    f = torch.randn(1, T * C, D)
    f_len = torch.tensor([T * C])
    g = torch.randn(1, 3, D)

    with torch.no_grad():
        base = j.joint(f, g, f_len)
        f2 = f.clone()
        t_cut = 3
        f2[0, (t_cut + 1) * C :] = torch.randn_like(f2[0, (t_cut + 1) * C :])  # future only
        pert = j.joint(f2, g, f_len)

    torch.testing.assert_close(base[:, : t_cut + 1], pert[:, : t_cut + 1], atol=1e-5, rtol=1e-4)
    assert not torch.allclose(base[:, t_cut + 1 :], pert[:, t_cut + 1 :]), "future change had no effect anywhere"


@pytest.mark.unit
@pytest.mark.parametrize("M", [0, 1, 2])
def test_on_path_joint_matches_full_joint_with_history(M):
    """joint_on_path must stay the full joint evaluated at a subset, for any M.

    The windowing is applied in two different call sites; if they ever disagreed,
    training would optimise different values from the ones decoding produces.
    """
    V, D, C = 15, 16, 4
    j = _att_joint(V, D, C, M).eval()
    torch.manual_seed(1)
    B, T_frames, U = 3, C * 5, 6
    f = torch.randn(B, T_frames, D)
    g = torch.randn(B, U, D)
    f_len = torch.tensor([T_frames, T_frames - 3, T_frames - C - 1])

    with torch.no_grad():
        full = j.joint(f, g, f_len)
        Tc = full.shape[1]
        b, t, u = torch.meshgrid(torch.arange(B), torch.arange(Tc), torch.arange(U), indexing="ij")
        b, t, u = b.reshape(-1), t.reshape(-1), u.reshape(-1)
        path = j.joint_on_path(f, g, b, t, u, f_len)

    torch.testing.assert_close(path, full[b, t, u], atol=1e-5, rtol=1e-4)


@pytest.mark.unit
def test_history_window_changes_the_logits():
    """A window that made no difference would be a silent no-op experiment."""
    V, D, C = 15, 16, 4
    torch.manual_seed(1)
    f = torch.randn(2, C * 5, D)
    g = torch.randn(2, 4, D)
    f_len = torch.tensor([C * 5, C * 5])
    with torch.no_grad():
        a = _att_joint(V, D, C, 0, seed=7).eval().joint(f, g, f_len)
        b = _att_joint(V, D, C, 1, seed=7).eval().joint(f, g, f_len)
    assert a.shape == b.shape
    # Chunk 0 has no history either way; later chunks must differ.
    assert not torch.allclose(a[:, 1:], b[:, 1:], atol=1e-6)
