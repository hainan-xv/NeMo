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
    ChatExternalAlignerCELoss,
    ChunkwiseAlignerLoss,
    chat_external_aligner_single_path_logprob,
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

    def align_to_chunks(
        self,
        input_signal,
        input_signal_length,
        labels,
        label_lens,
        target_frame_lengths,
        chunk_size,
        enforce_left_packing=True,
    ):
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


# ---------------------------------------------------------------------------
# Word-level (Qwen) external aligner: sub-word -> word grouping + word -> chunk
# bucketing, with the heavy aligner call stubbed so no model / package is needed.
# ---------------------------------------------------------------------------


class _FakeBPETokenizer:
    """Minimal SentencePiece-style tokenizer: pieces use the '▁' word marker."""

    def __init__(self, id2piece):
        self.id2piece = id2piece

    def ids_to_tokens(self, ids):
        return [self.id2piece[int(i)] for i in ids]

    def ids_to_text(self, ids):
        s = ''.join(self.id2piece[int(i)] for i in ids)
        return s.replace('\u2581', ' ').strip()


class _Item:
    def __init__(self, text, start_time, end_time=None):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time if end_time is not None else start_time


def _make_word_aligner(id2piece, run_aligner):
    from nemo.collections.asr.parts.submodules.external_word_aligner import QwenWordForcedAligner

    aligner = QwenWordForcedAligner(tokenizer=_FakeBPETokenizer(id2piece), sample_rate=16000)
    aligner._run_aligner = run_aligner  # bypass the real Qwen model / package
    return aligner


@pytest.mark.unit
def test_group_token_ids_into_words_conventions():
    from nemo.collections.asr.parts.submodules.external_word_aligner import group_token_ids_into_words

    # SentencePiece: '▁' marks word starts.
    assert group_token_ids_into_words(['\u2581HE', 'LLO', '\u2581WORLD']) == [[0, 1], [2]]
    # Byte-level BPE: 'Ġ' marks word starts (first piece is always a start).
    assert group_token_ids_into_words(['He', '\u0120wor', 'ld']) == [[0], [1, 2]]
    # WordPiece: '##' marks continuations.
    assert group_token_ids_into_words(['he', '##llo', 'world']) == [[0, 1], [2]]


@pytest.mark.unit
def test_word_aligner_buckets_subwords_into_word_chunks():
    # "HELLO WORLD" -> ids [1,2,3], pieces ['▁HE','LLO','▁WORLD'] -> 2 words.
    id2piece = {1: '\u2581HE', 2: 'LLO', 3: '\u2581WORLD'}

    # 1 sample, T_tr=8 frames, chunk_size=4 -> 2 chunks; audio = 1.0s.
    # word0 starts at 0.0s -> frame 0 -> chunk 0; word1 at 0.6s -> frame 4 -> chunk 1.
    def run_aligner(audios, texts):
        assert texts == ['HELLO WORLD']
        return [[_Item('HELLO', 0.0), _Item('WORLD', 0.6)]]

    aligner = _make_word_aligner(id2piece, run_aligner)
    labels = torch.tensor([[1, 2, 3]])
    label_lens = torch.tensor([3])
    target_frames = torch.tensor([8])

    token_chunk_ids, valid_mask = aligner.align_to_chunks(
        input_signal=torch.zeros(1, 16000),
        input_signal_length=torch.tensor([16000]),
        labels=labels,
        label_lens=label_lens,
        target_frame_lengths=target_frames,
        chunk_size=4,
    )
    assert bool(valid_mask[0])
    # All sub-words of "HELLO" share chunk 0; "WORLD" -> chunk 1.
    assert token_chunk_ids[0].tolist() == [0, 0, 1]


@pytest.mark.unit
def test_word_aligner_word_count_mismatch_is_discarded():
    id2piece = {1: '\u2581HE', 2: 'LLO', 3: '\u2581WORLD'}

    # Aligner returns only ONE word but the transcript has two -> discard.
    def run_aligner(audios, texts):
        return [[_Item('HELLO', 0.0)]]

    aligner = _make_word_aligner(id2piece, run_aligner)
    token_chunk_ids, valid_mask = aligner.align_to_chunks(
        input_signal=torch.zeros(1, 16000),
        input_signal_length=torch.tensor([16000]),
        labels=torch.tensor([[1, 2, 3]]),
        label_lens=torch.tensor([3]),
        target_frame_lengths=torch.tensor([8]),
        chunk_size=4,
    )
    assert not bool(valid_mask[0])
    assert torch.all(token_chunk_ids[0] == -1)


@pytest.mark.unit
def test_word_aligner_flags_T_less_than_U():
    id2piece = {1: '\u2581HE', 2: 'LLO', 3: '\u2581WORLD'}

    def run_aligner(audios, texts):
        return [[_Item('HELLO', 0.0), _Item('WORLD', 0.6)]]

    aligner = _make_word_aligner(id2piece, run_aligner)
    # 3 tokens but only 2 trainee frames -> infeasible.
    token_chunk_ids, valid_mask = aligner.align_to_chunks(
        input_signal=torch.zeros(1, 16000),
        input_signal_length=torch.tensor([16000]),
        labels=torch.tensor([[1, 2, 3]]),
        label_lens=torch.tensor([3]),
        target_frame_lengths=torch.tensor([2]),
        chunk_size=4,
    )
    assert not bool(valid_mask[0])


@pytest.mark.unit
def test_word_aligner_failure_discards_batch_but_does_not_raise():
    id2piece = {1: '\u2581HE', 2: 'LLO', 3: '\u2581WORLD'}

    def run_aligner(audios, texts):
        raise RuntimeError("simulated aligner failure")

    aligner = _make_word_aligner(id2piece, run_aligner)
    token_chunk_ids, valid_mask = aligner.align_to_chunks(
        input_signal=torch.zeros(2, 16000),
        input_signal_length=torch.tensor([16000, 16000]),
        labels=torch.tensor([[1, 2, 3], [1, 3, -1]]),
        label_lens=torch.tensor([3, 2]),
        target_frame_lengths=torch.tensor([8, 8]),
        chunk_size=4,
    )
    # Failure must be swallowed (training continues) and all samples discarded.
    assert not bool(valid_mask.any())
    assert torch.all(token_chunk_ids == -1)


class _StubWordAligner:
    """Stand-in for QwenWordForcedAligner: records construction, no model load."""

    last = None

    def __init__(
        self,
        tokenizer,
        model_name_or_path="Qwen/Qwen3-ForcedAligner-0.6B",
        language="English",
        dtype="bfloat16",
        device=None,
        sample_rate=16000,
    ):
        self.tokenizer = tokenizer
        self.model_name_or_path = model_name_or_path
        self.language = language
        self.sample_rate = sample_rate
        _StubWordAligner.last = self

    def to(self, *args, **kwargs):
        return self

    def align_to_chunks(self, labels, label_lens, target_frame_lengths, chunk_size, **kwargs):
        B, U = labels.shape
        return torch.full((B, U), -1, dtype=torch.long), torch.zeros(B, dtype=torch.bool)


@pytest.mark.unit
def test_model_qwen_backend_wires_word_aligner(monkeypatch):
    """external_aligner.backend='qwen' must build the word-level aligner with the trainee tokenizer."""
    from omegaconf import open_dict

    import nemo.collections.asr.parts.submodules.external_word_aligner as wmod

    monkeypatch.setattr(wmod, 'QwenWordForcedAligner', _StubWordAligner)

    model = _build_chunkwise_model(chunk_size=2)
    model.tokenizer = _FakeBPETokenizer({1: '\u2581a', 2: 'b'})
    with open_dict(model.cfg.external_aligner):
        model.cfg.external_aligner.backend = 'qwen'
        model.cfg.external_aligner.model_name = 'fake/qwen-aligner'
        model.cfg.external_aligner.language = 'English'

    model._setup_chunkwise_aligner_loss_and_decoding()

    assert isinstance(model._external_aligner, _StubWordAligner)
    assert model._external_aligner.tokenizer is model.tokenizer
    assert model._external_aligner.model_name_or_path == 'fake/qwen-aligner'


@pytest.mark.unit
def test_qwen_backend_requires_tokenizer(monkeypatch):
    """The word-level backend must error clearly when there is no sub-word tokenizer."""
    from omegaconf import open_dict

    import nemo.collections.asr.parts.submodules.external_word_aligner as wmod

    monkeypatch.setattr(wmod, 'QwenWordForcedAligner', _StubWordAligner)

    model = _build_chunkwise_model(chunk_size=2)  # char model -> no tokenizer
    assert getattr(model, 'tokenizer', None) is None
    with open_dict(model.cfg.external_aligner):
        model.cfg.external_aligner.backend = 'qwen'

    with pytest.raises(ValueError):
        model._setup_chunkwise_aligner_loss_and_decoding()


# ---------------------------------------------------------------------------
# Offline (precomputed) word-level aligner: reads word-start times from disk and
# does the same word -> chunk bucketing without any model / qwen_asr.
# ---------------------------------------------------------------------------


def _write_alignments(tmp_path, mapping, name='aligns.json'):
    import json

    p = tmp_path / name
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(mapping, f)
    return str(p)


@pytest.mark.unit
def test_precomputed_aligner_buckets_subwords(tmp_path):
    from nemo.collections.asr.parts.submodules.external_word_aligner import PrecomputedWordForcedAligner

    id2piece = {1: '\u2581HE', 2: 'LLO', 3: '\u2581WORLD'}
    # Path key normalizes to file_id '1089-0' (basename sans ext).
    path = _write_alignments(tmp_path, {'/data/x/1089-0.flac': [0.0, 0.6]})

    aligner = PrecomputedWordForcedAligner(tokenizer=_FakeBPETokenizer(id2piece), alignments_path=path)
    token_chunk_ids, valid_mask = aligner.align_to_chunks(
        file_ids=['/somewhere/1089-0.wav'],  # different dir/ext -> same file_id
        labels=torch.tensor([[1, 2, 3]]),
        label_lens=torch.tensor([3]),
        target_frame_lengths=torch.tensor([8]),
        audio_durations=torch.tensor([1.0]),
        chunk_size=4,
    )
    assert bool(valid_mask[0])
    assert token_chunk_ids[0].tolist() == [0, 0, 1]


@pytest.mark.unit
def test_precomputed_aligner_missing_key_discarded(tmp_path):
    from nemo.collections.asr.parts.submodules.external_word_aligner import PrecomputedWordForcedAligner

    id2piece = {1: '\u2581HE', 2: 'LLO', 3: '\u2581WORLD'}
    path = _write_alignments(tmp_path, {'present': [0.0, 0.6]})
    aligner = PrecomputedWordForcedAligner(tokenizer=_FakeBPETokenizer(id2piece), alignments_path=path)

    token_chunk_ids, valid_mask = aligner.align_to_chunks(
        file_ids=['absent'],
        labels=torch.tensor([[1, 2, 3]]),
        label_lens=torch.tensor([3]),
        target_frame_lengths=torch.tensor([8]),
        audio_durations=torch.tensor([1.0]),
        chunk_size=4,
    )
    assert not bool(valid_mask[0])
    assert torch.all(token_chunk_ids[0] == -1)


@pytest.mark.unit
def test_precomputed_loader_accepts_multiple_formats(tmp_path):
    from nemo.collections.asr.parts.submodules.external_word_aligner import _load_word_start_alignments

    mapping = {
        'a': [0.0, 0.5],                                   # bare list of starts (no ends)
        'b': {'word_starts': [0.1, 0.7]},                 # word_starts key
        'c': {'words': [{'start': 0.0, 'end': 0.4}, {'start_time': 0.9, 'end_time': 1.0}]},  # word dicts
        'd': {'starts': [0.0, 0.6], 'ends': [0.5, 0.9]},  # preferred starts/ends format
    }
    path = _write_alignments(tmp_path, mapping)
    loaded = _load_word_start_alignments(path)
    # The loader now returns (starts, ends|None) tuples.
    assert loaded['a'] == ([0.0, 0.5], None)
    assert loaded['b'] == ([0.1, 0.7], None)
    assert loaded['c'] == ([0.0, 0.9], [0.4, 1.0])
    assert loaded['d'] == ([0.0, 0.6], [0.5, 0.9])


@pytest.mark.unit
def test_precomputed_loader_merges_directory(tmp_path):
    from nemo.collections.asr.parts.submodules.external_word_aligner import _load_word_start_alignments

    d = tmp_path / "shards"
    d.mkdir()
    _write_alignments(d, {'a': [0.0]}, name='shard_0000.json')
    _write_alignments(d, {'b': [0.1]}, name='shard_0001.json')
    loaded = _load_word_start_alignments(str(d))
    assert set(loaded.keys()) == {'a', 'b'}


@pytest.mark.unit
def test_model_precomputed_backend_wires_aligner(tmp_path):
    """external_aligner.backend='precomputed' must build the offline aligner."""
    from omegaconf import open_dict

    from nemo.collections.asr.parts.submodules.external_word_aligner import PrecomputedWordForcedAligner

    path = _write_alignments(tmp_path, {'x': [0.0]})

    model = _build_chunkwise_model(chunk_size=2)
    model.tokenizer = _FakeBPETokenizer({1: '\u2581a', 2: 'b'})
    with open_dict(model.cfg.external_aligner):
        model.cfg.external_aligner.backend = 'precomputed'
        model.cfg.external_aligner.alignments_path = path

    model._setup_chunkwise_aligner_loss_and_decoding()

    assert isinstance(model._external_aligner, PrecomputedWordForcedAligner)
    assert model._external_aligner_is_precomputed is True


@pytest.mark.unit
def test_precomputed_backend_requires_alignments_path():
    from omegaconf import open_dict

    model = _build_chunkwise_model(chunk_size=2)
    model.tokenizer = _FakeBPETokenizer({1: '\u2581a', 2: 'b'})
    with open_dict(model.cfg.external_aligner):
        model.cfg.external_aligner.backend = 'precomputed'
        model.cfg.external_aligner.alignments_path = None

    with pytest.raises(ValueError):
        model._setup_chunkwise_aligner_loss_and_decoding()


@pytest.mark.unit
def test_word_aligner_monotonic_clamp_on_regression():
    # Two words whose raw timestamps regress (word1 earlier than word0). The
    # assignment must stay non-decreasing across words.
    id2piece = {1: '\u2581A', 2: '\u2581B'}

    def run_aligner(audios, texts):
        return [[_Item('A', 0.6), _Item('B', 0.1)]]  # regressing start times

    aligner = _make_word_aligner(id2piece, run_aligner)
    token_chunk_ids, valid_mask = aligner.align_to_chunks(
        input_signal=torch.zeros(1, 16000),
        input_signal_length=torch.tensor([16000]),
        labels=torch.tensor([[1, 2]]),
        label_lens=torch.tensor([2]),
        target_frame_lengths=torch.tensor([8]),
        chunk_size=4,
    )
    assert bool(valid_mask[0])
    ids = token_chunk_ids[0].tolist()
    assert ids[0] <= ids[1]  # non-decreasing


# ===========================================================================
# CHAT external-alignment cross-entropy baseline (loss_type='chat_aligner')
# ===========================================================================
# The CHAT joint's activations are CHUNK-indexed (B, n_chunks, U+1, V): the chunk
# axis IS the RNN-T "time" axis. The CE loss scores the single token->chunk path,
# one blank per chunk, with NO left-packing constraint (a chunk may host any
# number of tokens).


def _random_monotonic_chunk_ids(U_b, n_chunks, rng):
    """Random non-decreasing chunk ids in [0, n_chunks) of length U_b."""
    ids = sorted(rng.randint(0, n_chunks - 1) for _ in range(U_b))
    return ids


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_chat_ce_loss_matches_single_path_reference(seed):
    """ChatExternalAlignerCELoss(reduction='none') == -reference single-path logprob."""
    import random

    rng = random.Random(100 + seed)
    B, n_chunks, U_max, V = 4, 5, 6, 9
    blank = V - 1

    acts = torch.randn(B, n_chunks, U_max + 1, V)
    labels = torch.randint(0, blank, (B, U_max))
    act_lens = torch.tensor([rng.randint(2, n_chunks) for _ in range(B)])
    label_lens = torch.tensor([rng.randint(1, U_max) for _ in range(B)])

    token_chunk_ids = torch.full((B, U_max), -1, dtype=torch.long)
    for b in range(B):
        ids = _random_monotonic_chunk_ids(int(label_lens[b]), int(act_lens[b]), rng)
        for u, c in enumerate(ids):
            token_chunk_ids[b, u] = c

    loss = ChatExternalAlignerCELoss(blank=blank, reduction='none')
    per_sample = loss(acts, labels, act_lens, label_lens, token_chunk_ids)
    ref = chat_external_aligner_single_path_logprob(acts, labels, act_lens, label_lens, token_chunk_ids, blank)

    for b in range(B):
        assert math.isclose(float(per_sample[b]), float(-ref[b]), rel_tol=1e-4, abs_tol=1e-4)


@pytest.mark.unit
def test_chat_ce_loss_allows_many_tokens_per_chunk():
    """Unlike the additive-joint baseline, a CHAT chunk may host > chunk_size tokens."""
    B, n_chunks, U, V = 1, 2, 5, 7
    blank = V - 1
    acts = torch.randn(B, n_chunks, U + 1, V)
    labels = torch.randint(0, blank, (B, U))
    act_lens = torch.tensor([n_chunks])
    label_lens = torch.tensor([U])
    # All 5 tokens packed into chunk 0 (would be infeasible for a 2-frame chunk
    # under left-packing, but fine for CHAT cross-attention pooling).
    token_chunk_ids = torch.tensor([[0, 0, 0, 0, 0]])

    loss = ChatExternalAlignerCELoss(blank=blank, reduction='none')
    val = loss(acts, labels, act_lens, label_lens, token_chunk_ids)
    ref = chat_external_aligner_single_path_logprob(acts, labels, act_lens, label_lens, token_chunk_ids, blank)
    assert torch.isfinite(val[0])
    assert math.isclose(float(val[0]), float(-ref[0]), rel_tol=1e-4, abs_tol=1e-4)


@pytest.mark.unit
def test_chat_ce_loss_invalid_assignment_masked():
    """Out-of-range / non-monotonic chunk ids -> infeasible -> masked out (mean_volume)."""
    B, n_chunks, U, V = 2, 3, 3, 6
    blank = V - 1
    acts = torch.randn(B, n_chunks, U + 1, V)
    labels = torch.randint(0, blank, (B, U))
    act_lens = torch.tensor([n_chunks, n_chunks])
    label_lens = torch.tensor([U, U])
    # Sample 0: valid monotonic. Sample 1: non-monotonic (1 -> 0) -> infeasible.
    token_chunk_ids = torch.tensor([[0, 1, 2], [1, 0, 2]])

    ref = chat_external_aligner_single_path_logprob(acts, labels, act_lens, label_lens, token_chunk_ids, blank)
    assert torch.isfinite(ref[0]) and float(ref[1]) <= -1e8  # NEG_INF for sample 1

    loss = ChatExternalAlignerCELoss(blank=blank, reduction='mean_volume')
    val = loss(acts, labels, act_lens, label_lens, token_chunk_ids)
    # Only sample 0's tokens count: mean_volume == sample0 NLL / U.
    expected = float(-ref[0]) / U
    assert math.isclose(float(val), expected, rel_tol=1e-4, abs_tol=1e-4)


@pytest.mark.unit
def test_map_word_starts_relaxed_left_packing():
    """enforce_left_packing=False allows T<U and >chunk_size tokens per chunk."""
    from nemo.collections.asr.parts.submodules.external_word_aligner import map_word_starts_to_token_chunks

    # 1 word covering 3 tokens, T_tr=2 frames, chunk_size=2 -> 1 chunk. Under
    # left-packing this is infeasible (T<U and counts>frames); relaxed it is fine.
    word_groups = [[0, 1, 2]]
    word_starts = [0.0]
    strict = map_word_starts_to_token_chunks(word_groups, word_starts, 1.0, 2, 3, 2, enforce_left_packing=True)
    relaxed = map_word_starts_to_token_chunks(word_groups, word_starts, 1.0, 2, 3, 2, enforce_left_packing=False)
    assert strict is None
    assert relaxed == {0: 0, 1: 0, 2: 0}


@pytest.mark.unit
def test_map_word_starts_end_anchor():
    """anchor='end' buckets a word by the end of its last sub-word, not its onset."""
    from nemo.collections.asr.parts.submodules.external_word_aligner import map_word_starts_to_token_chunks

    # 2 words; audio_dur=1s, T_tr=10 frames, chunk_size=2 -> 5 chunks.
    # word 0: tokens [0,1] start 0.0s (frame 0 -> chunk 0), end 0.5s (frame 5 -> chunk 2)
    # word 1: token  [2]   start 0.6s (frame 6 -> chunk 3), end 0.9s (frame 9 -> chunk 4)
    word_groups = [[0, 1], [2]]
    starts = [0.0, 0.6]
    ends = [0.5, 0.9]

    by_start = map_word_starts_to_token_chunks(
        word_groups, starts, 1.0, 10, 3, 2, enforce_left_packing=False, word_ends_sec=ends, anchor='start'
    )
    by_end = map_word_starts_to_token_chunks(
        word_groups, starts, 1.0, 10, 3, 2, enforce_left_packing=False, word_ends_sec=ends, anchor='end'
    )
    assert by_start == {0: 0, 1: 0, 2: 3}
    assert by_end == {0: 2, 1: 2, 2: 4}

    # anchor='end' with no end times provided -> falls back to start anchoring.
    fallback = map_word_starts_to_token_chunks(
        word_groups, starts, 1.0, 10, 3, 2, enforce_left_packing=False, word_ends_sec=None, anchor='end'
    )
    assert fallback == by_start


# ---------------------------------------------------------------------------
# End-to-end CHAT external-aligner model wiring (RNNTAttJoint + CE loss).
# ---------------------------------------------------------------------------


def _build_chat_aligner_model(chunk_size=2):
    from omegaconf import DictConfig, ListConfig

    import nemo.collections.asr.parts.submodules.external_ctc_aligner as ext_mod
    from nemo.collections.asr.models import EncDecRNNTModel

    # Stub the CTC loader so no real model is restored (backend defaults to 'ctc').
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
    # CHAT cross-attention joint; chunk_size set explicitly so the encoder need not
    # use chunked_limited attention in this unit test.
    joint = {
        '_target_': 'nemo.collections.asr.modules.RNNTAttJoint',
        'fuse_loss_wer': False,
        'chunk_size': chunk_size,
        'jointnet': {'joint_hidden': model_defaults['joint_hidden'], 'activation': 'relu'},
    }
    decoding = {
        'strategy': 'greedy_batch',
        'greedy': {'max_symbols': 5},
        'beam': {'beam_size': 1, 'return_best_hypothesis': True},
    }
    cfg = DictConfig(
        {
            'labels': ListConfig(_LABELS),
            'loss_type': 'chat_aligner',
            'compute_eval_loss': True,
            'external_aligner': DictConfig(
                {'backend': 'ctc', 'pretrained_name': 'dummy', 'model_path': None, 'reduction': 'mean_volume'}
            ),
            'preprocessor': DictConfig(preprocessor),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'joint': DictConfig(joint),
            'decoding': DictConfig(decoding),
        }
    )
    return EncDecRNNTModel(cfg=cfg)


@pytest.mark.unit
def test_model_chat_aligner_wires_ce_loss_and_rnnt_decoding():
    from nemo.collections.asr.metrics.wer import WER
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecoding

    model = _build_chat_aligner_model(chunk_size=2)
    assert model.loss_type == 'chat_aligner'
    assert isinstance(model.loss, ChatExternalAlignerCELoss)
    # CHAT reuses the standard RNN-T decoding + WER objects.
    assert isinstance(model.decoding, RNNTDecoding)
    assert isinstance(model.wer, WER)
    # CHAT pools the whole chunk -> no left-packing constraint on the aligner.
    assert model._aligner_enforce_left_packing is False
    # chunk_size threaded from the joint for the external aligner bucketing.
    assert int(model.chunk_size) == 2
    assert isinstance(model._external_aligner, _StubAligner)


@pytest.mark.unit
def test_model_chat_aligner_loss_runs_and_backprops():
    model = _build_chat_aligner_model(chunk_size=2)
    model.train()
    B, audio_len = 3, 8000
    signal = torch.randn(B, audio_len)
    signal_len = torch.full((B,), audio_len, dtype=torch.long)
    transcript = torch.randint(0, len(_LABELS), (B, 4))
    transcript_len = torch.tensor([3, 2, 4])

    encoded, encoded_len = model.forward(input_signal=signal, input_signal_length=signal_len)
    decoder_out, target_len, _ = model.decoder(targets=transcript, target_length=transcript_len)
    joint = model.joint(encoder_outputs=encoded, decoder_outputs=decoder_out, encoder_lengths=encoded_len)
    num_chunks = model.joint.num_chunks_per_utterance
    assert num_chunks is not None

    # Build a valid monotonic token->chunk assignment within each sample's chunks.
    token_chunk_ids = torch.full_like(transcript, -1)
    valid_mask = torch.ones(B, dtype=torch.bool)
    for b in range(B):
        n_b = int(num_chunks[b])
        for u in range(int(transcript_len[b])):
            token_chunk_ids[b, u] = min(u, n_b - 1)

    loss = model._chat_aligner_loss(joint, transcript, target_len, num_chunks, token_chunk_ids, valid_mask)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
