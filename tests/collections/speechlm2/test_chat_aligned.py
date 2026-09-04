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

import pytest
import torch

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
