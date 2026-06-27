# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Tests for the Chunkwise-Aligner baseline: the single fixed-path loss and the
frozen external CTC aligner's token->chunk bucketing / feasibility skipping."""

import math

import pytest
import torch

from nemo.collections.asr.losses.chunked_aligner_pytorch import (
    ChunkwiseAlignerLoss,
    chunked_aligner_loss_bruteforce,
    chunkwise_aligner_single_path_logprob,
)


def _counts_to_token_chunk_ids(counts_per_sample, U_max):
    """List[List[int]] per-chunk counts -> [B, U_max] token chunk ids (-1 pad)."""
    B = len(counts_per_sample)
    out = torch.full((B, U_max), -1, dtype=torch.long)
    for b, counts in enumerate(counts_per_sample):
        u = 0
        for c, k in enumerate(counts):
            for _ in range(k):
                out[b, u] = c
                u += 1
    return out


def _random_feasible_counts(T_b, U_b, C, rng):
    """Random per-chunk token counts with sum == U_b and counts[c] <= frames_in_chunk."""
    n_chunks = (T_b + C - 1) // C
    frames_here = [min(C, T_b - c * C) for c in range(n_chunks)]
    if sum(frames_here) < U_b:
        return None  # infeasible to host U_b tokens at all
    counts = [0] * n_chunks
    remaining = U_b
    # Greedy random fill respecting per-chunk capacity.
    order = list(range(n_chunks))
    rng.shuffle(order)
    for c in order:
        if remaining == 0:
            break
        cap = frames_here[c]
        take = rng.randint(0, min(cap, remaining))
        counts[c] = take
        remaining -= take
    # Place any leftover wherever capacity remains.
    c = 0
    while remaining > 0 and c < n_chunks:
        room = frames_here[c] - counts[c]
        add = min(room, remaining)
        counts[c] += add
        remaining -= add
        c += 1
    if remaining != 0:
        return None
    return counts


@pytest.mark.unit
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4])
def test_loss_matches_single_path_reference(chunk_size):
    """ChunkwiseAlignerLoss (reduction='none') == -reference single-path logprob."""
    import random

    rng = random.Random(1234 + chunk_size)
    torch.manual_seed(7 + chunk_size)

    B, T, V = 4, 10, 6
    blank = V - 1

    act_lens = torch.tensor([10, 8, 6, 5])[:B]
    label_lens = torch.tensor([3, 2, 2, 1])[:B]
    U_max = int(label_lens.max())

    acts = torch.randn(B, T, U_max + 1, V)
    labels = torch.randint(0, V - 1, (B, U_max))

    counts_per_sample = []
    n_chunks_max = (int(act_lens.max()) + chunk_size - 1) // chunk_size
    chunk_counts = torch.zeros(B, n_chunks_max, dtype=torch.long)
    for b in range(B):
        counts = _random_feasible_counts(int(act_lens[b]), int(label_lens[b]), chunk_size, rng)
        if counts is None:
            pytest.skip("randomly generated an infeasible segmentation")
        counts_per_sample.append(counts)
        for c, k in enumerate(counts):
            chunk_counts[b, c] = k

    token_chunk_ids = _counts_to_token_chunk_ids(counts_per_sample, U_max)

    ref = chunkwise_aligner_single_path_logprob(
        acts, labels, act_lens, label_lens, chunk_counts, blank=blank, chunk_size=chunk_size
    )

    loss = ChunkwiseAlignerLoss(blank=blank, chunk_size=chunk_size, reduction='none')
    per_sample = loss(acts, labels, act_lens, label_lens, token_chunk_ids)

    assert torch.allclose(per_sample, -ref, atol=1e-4), f"{per_sample} vs {-ref}"


@pytest.mark.unit
def test_single_path_is_a_term_of_full_sum():
    """The fixed-path logprob must be <= the full-sum logprob (it's one of its terms)."""
    torch.manual_seed(0)
    B, T, V = 2, 8, 5
    chunk_size = 3
    blank = V - 1
    act_lens = torch.tensor([8, 6])
    label_lens = torch.tensor([2, 2])
    U_max = 2
    acts = torch.randn(B, T, U_max + 1, V)
    labels = torch.randint(0, V - 1, (B, U_max))

    n_chunks_max = (int(act_lens.max()) + chunk_size - 1) // chunk_size
    # Assign both tokens to chunk 0 (a valid, simple segmentation).
    chunk_counts = torch.zeros(B, n_chunks_max, dtype=torch.long)
    chunk_counts[:, 0] = 2

    single = chunkwise_aligner_single_path_logprob(
        acts, labels, act_lens, label_lens, chunk_counts, blank=blank, chunk_size=chunk_size
    )
    full = chunked_aligner_loss_bruteforce(acts, labels, act_lens, label_lens, blank=blank, chunk_size=chunk_size)

    assert torch.all(single <= full + 1e-4)


@pytest.mark.unit
def test_infeasible_assignment_is_skipped():
    """An overflowing assignment (more tokens than frames in a chunk) is excluded."""
    torch.manual_seed(3)
    B, T, V = 2, 4, 5
    chunk_size = 1  # each chunk hosts at most 1 token
    blank = V - 1
    act_lens = torch.tensor([4, 4])
    label_lens = torch.tensor([2, 2])
    U_max = 2
    acts = torch.randn(B, T, U_max + 1, V)
    labels = torch.randint(0, V - 1, (B, U_max))

    # Sample 0: feasible (one token per chunk). Sample 1: both tokens in chunk 0 -> overflow.
    token_chunk_ids = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)

    loss = ChunkwiseAlignerLoss(blank=blank, chunk_size=chunk_size, reduction='none')
    per_sample = loss(acts, labels, act_lens, label_lens, token_chunk_ids)

    assert per_sample[1].item() == 0.0  # infeasible -> zero contribution
    assert per_sample[0].item() != 0.0  # feasible -> real loss

    # mean_volume must only divide by the valid sample's label count.
    mv = ChunkwiseAlignerLoss(blank=blank, chunk_size=chunk_size, reduction='mean_volume')
    val = mv(acts, labels, act_lens, label_lens, token_chunk_ids)
    expected = per_sample[0] / float(label_lens[0])
    assert torch.allclose(val, expected, atol=1e-5)


@pytest.mark.unit
def test_valid_mask_excludes_samples():
    torch.manual_seed(5)
    B, T, V = 2, 6, 5
    chunk_size = 2
    blank = V - 1
    act_lens = torch.tensor([6, 6])
    label_lens = torch.tensor([2, 2])
    acts = torch.randn(B, T, 3, V)
    labels = torch.randint(0, V - 1, (B, 2))
    token_chunk_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

    loss = ChunkwiseAlignerLoss(blank=blank, chunk_size=chunk_size, reduction='none')
    valid_mask = torch.tensor([True, False])
    per_sample = loss(acts, labels, act_lens, label_lens, token_chunk_ids, valid_mask=valid_mask)
    assert per_sample[1].item() == 0.0


@pytest.mark.unit
def test_loss_is_differentiable():
    torch.manual_seed(9)
    B, T, V = 2, 6, 5
    chunk_size = 2
    blank = V - 1
    act_lens = torch.tensor([6, 4])
    label_lens = torch.tensor([2, 1])
    acts = torch.randn(B, T, 3, V, requires_grad=True)
    labels = torch.randint(0, V - 1, (B, 2))
    token_chunk_ids = torch.tensor([[0, 1], [0, -1]], dtype=torch.long)

    loss = ChunkwiseAlignerLoss(blank=blank, chunk_size=chunk_size, reduction='mean_volume')
    val = loss(acts, labels, act_lens, label_lens, token_chunk_ids)
    val.backward()
    assert acts.grad is not None
    assert torch.isfinite(acts.grad).all()


# ---------------------------------------------------------------------------
# External CTC aligner: token->chunk bucketing + feasibility, with the CTC
# log-probs stubbed so the forced alignment is deterministic.
# ---------------------------------------------------------------------------


def _make_aligner_with_stub(per_frame_labels, V_ext):
    """Build an ExternalCTCForcedAligner without loading a model, stubbing CTC log-probs.

    ``per_frame_labels`` is [B, T] of the dominant (near one-hot) token id per
    external frame, so :func:`viterbi_decoding` produces a known alignment.
    """
    from nemo.collections.asr.parts.submodules.external_ctc_aligner import ExternalCTCForcedAligner

    aligner = ExternalCTCForcedAligner.__new__(ExternalCTCForcedAligner)
    aligner._device = torch.device("cpu")
    aligner.viterbi_device = "cpu"

    B, T = per_frame_labels.shape
    log_probs = torch.full((B, T, V_ext), -20.0)
    for b in range(B):
        for t in range(T):
            log_probs[b, t, int(per_frame_labels[b, t])] = 0.0
    log_probs = torch.log_softmax(log_probs, dim=-1)
    enc_len = torch.full((B,), T, dtype=torch.long)

    aligner._ctc_log_probs = lambda input_signal, input_signal_length: (log_probs, enc_len)
    return aligner


@pytest.mark.unit
def test_external_aligner_buckets_tokens_into_chunks():
    V_ext = 5  # ids 0..3 real, 4 = blank
    # Per-frame dominant ids: token 0 (id=1) at frames 0-1, token 1 (id=2) at frames 2-3.
    per_frame = torch.tensor([[1, 1, 2, 2]])
    aligner = _make_aligner_with_stub(per_frame, V_ext)

    labels = torch.tensor([[1, 2]])
    label_lens = torch.tensor([2])
    target_frames = torch.tensor([4])  # trainee frames == external frames here

    token_chunk_ids, valid_mask = aligner.align_to_chunks(
        input_signal=torch.zeros(1, 16000),
        input_signal_length=torch.tensor([16000]),
        labels=labels,
        label_lens=label_lens,
        target_frame_lengths=target_frames,
        chunk_size=2,
    )
    assert bool(valid_mask[0])
    # frame 0 -> chunk 0 ; frame 2 -> chunk 1
    assert token_chunk_ids[0, 0].item() == 0
    assert token_chunk_ids[0, 1].item() == 1


@pytest.mark.unit
def test_external_aligner_flags_overflow_infeasible():
    V_ext = 5
    # Both tokens collapse into the first two frames; with chunk_size=4 and only
    # T=4 trainee frames there is one chunk that would need to host both tokens
    # at frames 0 and 1 -> feasible. To force overflow, shrink trainee frames so
    # T < U is impossible; instead force both tokens into the same single-frame chunk.
    per_frame = torch.tensor([[1, 2, 4, 4]])  # token0 at f0, token1 at f1
    aligner = _make_aligner_with_stub(per_frame, V_ext)

    labels = torch.tensor([[1, 2]])
    label_lens = torch.tensor([2])
    target_frames = torch.tensor([2])  # only 2 trainee frames, chunk_size=2 -> 1 chunk of 2 frames

    # chunk_size larger than the 2 frames -> single chunk capacity 2, both tokens fit -> feasible.
    ids_ok, valid_ok = aligner.align_to_chunks(
        input_signal=torch.zeros(1, 16000),
        input_signal_length=torch.tensor([16000]),
        labels=labels,
        label_lens=label_lens,
        target_frame_lengths=target_frames,
        chunk_size=2,
    )
    assert bool(valid_ok[0])

    # T < U -> infeasible (1 trainee frame, 2 tokens).
    ids_bad, valid_bad = aligner.align_to_chunks(
        input_signal=torch.zeros(1, 16000),
        input_signal_length=torch.tensor([16000]),
        labels=labels,
        label_lens=label_lens,
        target_frame_lengths=torch.tensor([1]),
        chunk_size=2,
    )
    assert not bool(valid_bad[0])
    assert torch.all(ids_bad[0] == -1)


# ---------------------------------------------------------------------------
# End-to-end model wiring: build a chunkwise_aligner EncDecRNNTModel with the
# external aligner stubbed (no real CTC model loaded), then exercise the loss +
# greedy decoding paths.
# ---------------------------------------------------------------------------

# fmt: off
_LABELS = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'"]
# fmt: on


class _StubAligner:
    """Stand-in for ExternalCTCForcedAligner: assigns one token per chunk (left-packed)."""

    def __init__(self, *args, **kwargs):
        pass

    def to(self, *args, **kwargs):
        return self

    def align_to_chunks(self, input_signal, input_signal_length, labels, label_lens, target_frame_lengths, chunk_size):
        B, U = labels.shape
        device = labels.device
        token_chunk_ids = torch.full((B, U), -1, dtype=torch.long, device=device)
        valid = torch.ones(B, dtype=torch.bool, device=device)
        for b in range(B):
            U_b = int(label_lens[b])
            T_b = int(target_frame_lengths[b])
            n_chunks = (T_b + chunk_size - 1) // chunk_size
            if U_b > n_chunks or T_b < U_b:
                valid[b] = False
                continue
            for u in range(U_b):
                token_chunk_ids[b, u] = u  # one token per chunk
        return token_chunk_ids, valid


def _build_chunkwise_model(chunk_size=2):
    from omegaconf import DictConfig, ListConfig

    import nemo.collections.asr.parts.submodules.external_ctc_aligner as ext_mod
    from nemo.collections.asr.models import EncDecRNNTModel

    # Stub the loader so no real CTC model is restored.
    ext_mod.ExternalCTCForcedAligner = _StubAligner

    model_defaults = {'enc_hidden': 128, 'pred_hidden': 64, 'joint_hidden': 64}
    preprocessor = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'features': 64}
    encoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASREncoder',
        'feat_in': 64,
        'activation': 'relu',
        'conv_mask': True,
        'jasper': [
            {
                'filters': model_defaults['enc_hidden'],
                'repeat': 1,
                'kernel': [1],
                'stride': [1],
                'dilation': [1],
                'dropout': 0.0,
                'residual': False,
                'separable': True,
                'se': True,
                'se_context_size': -1,
            }
        ],
    }
    decoder = {
        '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
        'prednet': {'pred_hidden': model_defaults['pred_hidden'], 'pred_rnn_layers': 1},
    }
    joint = {
        '_target_': 'nemo.collections.asr.modules.RNNTJoint',
        'fuse_loss_wer': False,
        'jointnet': {'joint_hidden': model_defaults['joint_hidden'], 'activation': 'relu'},
    }
    cfg = DictConfig(
        {
            'labels': ListConfig(_LABELS),
            'loss_type': 'chunkwise_aligner',
            'compute_eval_loss': True,
            'external_aligner': DictConfig({'pretrained_name': 'dummy', 'model_path': None}),
            'chunked_aligner': DictConfig({'chunk_size': chunk_size, 'reduction': 'mean_volume'}),
            'preprocessor': DictConfig(preprocessor),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'joint': DictConfig(joint),
            'decoding': DictConfig({'max_symbols': None}),
        }
    )
    return EncDecRNNTModel(cfg=cfg)


@pytest.mark.unit
def test_model_builds_and_uses_chunked_decoding():
    from nemo.collections.asr.parts.submodules.chunked_aligner_decoding import ChunkedAlignerDecoding
    from nemo.collections.asr.losses.chunked_aligner_pytorch import ChunkwiseAlignerLoss as _CWL

    model = _build_chunkwise_model(chunk_size=2)
    assert model.loss_type == 'chunkwise_aligner'
    assert isinstance(model.loss, _CWL)
    assert isinstance(model.decoding, ChunkedAlignerDecoding)
    assert isinstance(model._external_aligner, _StubAligner)


@pytest.mark.unit
def test_model_chunkwise_loss_and_alignment_paths():
    model = _build_chunkwise_model(chunk_size=2)
    model.train()
    B, audio_len = 3, 8000
    signal = torch.randn(B, audio_len)
    signal_len = torch.full((B,), audio_len, dtype=torch.long)
    transcript = torch.randint(0, len(_LABELS), (B, 4))
    transcript_len = torch.tensor([3, 2, 4])

    encoded, encoded_len = model.forward(input_signal=signal, input_signal_length=signal_len)

    token_chunk_ids, valid_mask, n_disc, n_tot = model._external_chunk_assignment(
        signal, signal_len, transcript, transcript_len, encoded_len
    )
    assert token_chunk_ids.shape[0] == B
    assert valid_mask.shape[0] == B
    assert n_tot == B

    loss = model._chunkwise_aligner_loss(
        encoded, encoded_len, transcript, transcript_len, token_chunk_ids, valid_mask
    )
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


@pytest.mark.unit
def test_model_chunkwise_greedy_decode_shapes():
    model = _build_chunkwise_model(chunk_size=2)
    model.eval()
    B, audio_len = 2, 8000
    signal = torch.randn(B, audio_len)
    signal_len = torch.full((B,), audio_len, dtype=torch.long)
    with torch.no_grad():
        encoded, encoded_len = model.forward(input_signal=signal, input_signal_length=signal_len)
        texts, token_ids = model.decoding.decode_encoder_output(encoded, encoded_len)
    assert len(texts) == B
    assert len(token_ids) == B
    assert all(isinstance(t, str) for t in texts)
