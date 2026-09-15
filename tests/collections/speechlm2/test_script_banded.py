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
"""Properties of SCRIPT's banded forward algorithm.

These mirror tests/collections/asr/test_banded_rnnt.py, with one deliberate
difference: CHAT's ``test_band_zero_matches_the_forced_loss_end_to_end`` asserts
only that both losses are finite, so the band-0 equivalence it is named for is
not actually pinned anywhere at the model level. Here it IS asserted, as an exact
equality against the summed cross-entropy of the single path.
"""

import itertools

import pytest
import torch

from nemo.collections.speechlm2.parts.script_banded import NEG_INF, banded_forward, span_scores


@pytest.fixture(autouse=True)
def _cpu_default_device():
    """Pin the default device to CPU for this module.

    Several modules in this suite call ``torch.set_default_device('cuda')`` at
    import time and never restore it (test_salm.py, test_duplex_stt.py and
    others; test_salm_automodel.py is the one that does restore, and says why).
    It is a process-wide mutation, so whether these tests see a CPU default
    depends on collection order -- run alone they pass, run after those modules
    every ``torch.randn(..., generator=<cpu generator>)`` here raises
    "Expected a 'cuda' device type for generator but found 'cpu'".

    This DP is pure tensor arithmetic and is deliberately exercised on CPU in
    float64, where the brute-force comparison is exact.
    """
    prev = torch.tensor([]).device
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(prev)


def _brute_force(span_logprob, cut, cut_valid, n_chunks, n_tokens):
    """Enumerate every in-band partition explicitly and log-sum-exp the paths.

    Deliberately written as the definition rather than as an optimisation: nested
    python loops over all reachable (chunk, cut) sequences. Only usable on toy
    sizes, which is the point -- it shares no code with the DP.
    """
    b, t_max, c, kp1 = span_logprob.shape
    out = []
    for i in range(b):
        n_t, n_u = int(n_chunks[i]), int(n_tokens[i])
        paths = []

        def walk(t, u, acc):
            if t == n_t:
                if u == n_u:
                    paths.append(acc)
                return
            for j in range(c):
                if not bool(cut_valid[i, t, j]):
                    continue
                if int(cut[i, t, j]) != u:
                    continue
                for k in range(kp1):
                    if u + k > n_u:
                        break
                    walk(t + 1, u + k, acc + float(span_logprob[i, t, j, k]))

        walk(0, 0, 0.0)
        if not paths:
            out.append(-NEG_INF)
        else:
            out.append(-float(torch.logsumexp(torch.tensor(paths, dtype=torch.float64), dim=0)))
    return torch.tensor(out, dtype=torch.float64)


def _toy(band, n_t=3, n_u=4, kp1=3, seed=0):
    """A tiny problem whose candidate cuts are the aligner cut +/- ``band``.

    The aligner path is one token per chunk, so chunk ``t`` nominally starts at
    ``u = t``; the band widens that to ``[t - band, t + band]`` clipped to range.

    Scores are drawn ONCE into a table indexed by the ABSOLUTE cut ``u``, then
    gathered into candidate slots. Drawing them per-slot instead would give the
    same path a different score at each band -- the comparison the monotonicity
    test makes would then be meaningless, and it would pass or fail at random.
    """
    g = torch.Generator().manual_seed(seed)
    table = torch.randn(n_t, n_u + 1, kp1, generator=g, dtype=torch.float64).log_softmax(-1)

    cands = []
    for t in range(n_t):
        lo, hi = max(0, t - band), min(n_u, t + band)
        cands.append(list(range(lo, hi + 1)))
    c = max(len(x) for x in cands)

    cut = torch.zeros(1, n_t, c, dtype=torch.long)
    cut_valid = torch.zeros(1, n_t, c, dtype=torch.bool)
    span_logprob = torch.full((1, n_t, c, kp1), NEG_INF, dtype=torch.float64)
    for t, xs in enumerate(cands):
        for j, u in enumerate(xs):
            cut[0, t, j] = u
            cut_valid[0, t, j] = True
            span_logprob[0, t, j] = table[t, u]

    return (
        span_logprob,
        cut,
        cut_valid,
        torch.tensor([n_t]),
        torch.tensor([n_u]),
    )


@pytest.mark.unit
@pytest.mark.parametrize("band", [0, 1, 2])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_dp_equals_explicit_path_enumeration(band, seed):
    """The DP is only worth having if it computes the definition."""
    args = _toy(band, seed=seed)
    got = banded_forward(*args)
    want = _brute_force(*args)
    assert torch.allclose(got.double(), want, atol=1e-9), f"{got.item()} != {want.item()}"


@pytest.mark.unit
def test_band_zero_is_exactly_the_single_path_cross_entropy():
    """band=0 must reproduce the forced loss EXACTLY, not merely be finite.

    With one candidate cut per chunk there is a single surviving partition, so the
    marginal collapses to that path's joint log-probability -- the sum of the
    per-chunk cross-entropies the forced objective already computes.
    """
    span_logprob, cut, cut_valid, n_chunks, n_tokens = _toy(band=0, n_t=3, n_u=3, kp1=2)
    nll = banded_forward(span_logprob, cut, cut_valid, n_chunks, n_tokens)

    # The only path: every chunk starts where the aligner put it and emits one token.
    forced = sum(float(span_logprob[0, t, 0, 1]) for t in range(3))
    assert torch.allclose(nll.double(), torch.tensor([-forced], dtype=torch.float64), atol=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_wider_band_never_increases_the_loss(seed):
    """Widening the band only ever adds paths to the sum."""
    losses = []
    for band in (0, 1, 2):
        losses.append(float(banded_forward(*_toy(band, seed=seed))))
    assert losses[0] >= losses[1] - 1e-9
    assert losses[1] >= losses[2] - 1e-9


@pytest.mark.unit
def test_span_scores_is_a_running_prefix_sum():
    """One branch forward must score every span length that starts at its cut."""
    tok = torch.tensor([[[[-0.1, -0.2, -0.3]]]])
    stop = torch.tensor([[[[-1.0, -2.0, -3.0, -4.0]]]])
    got = span_scores(tok, stop)[0, 0, 0]
    assert torch.allclose(
        got,
        torch.tensor([-1.0, -0.1 - 2.0, -0.1 - 0.2 - 3.0, -0.1 - 0.2 - 0.3 - 4.0]),
        atol=1e-6,
    )


@pytest.mark.unit
def test_span_scores_rejects_a_mismatched_stop_shape():
    with pytest.raises(ValueError, match="stop_logprob must be"):
        span_scores(torch.zeros(1, 1, 1, 3), torch.zeros(1, 1, 1, 3))


@pytest.mark.unit
def test_gradients_flow_and_stay_finite():
    """The finite sentinel exists so a fully-masked group cannot produce NaN."""
    span_logprob, cut, cut_valid, n_chunks, n_tokens = _toy(band=1, seed=7)
    span_logprob = span_logprob.float().requires_grad_(True)
    banded_forward(span_logprob, cut, cut_valid, n_chunks, n_tokens).sum().backward()
    assert span_logprob.grad is not None
    assert torch.isfinite(span_logprob.grad).all()


@pytest.mark.unit
def test_an_unreachable_target_does_not_produce_nan():
    """No in-band partition can consume the tokens -> a large finite loss, not NaN."""
    span_logprob, cut, cut_valid, n_chunks, _ = _toy(band=0, n_t=2, n_u=2, kp1=2)
    unreachable = torch.tensor([9])  # far more tokens than any path can emit
    nll = banded_forward(span_logprob, cut, cut_valid, n_chunks, unreachable)
    assert torch.isfinite(nll).all()
    assert float(nll) > 1e20


@pytest.mark.unit
def test_shorter_utterances_in_a_batch_are_not_advanced_past_their_end():
    """Padding chunks must carry alpha forward, not keep consuming tokens."""
    a = _toy(band=1, n_t=2, n_u=2, kp1=3, seed=5)
    b = _toy(band=1, n_t=2, n_u=2, kp1=3, seed=5)

    # Pad example `a` with one extra chunk that it does not own.
    span = torch.cat([a[0], torch.zeros_like(a[0][:, :1])], dim=1)
    cut = torch.cat([a[1], torch.zeros_like(a[1][:, :1])], dim=1)
    valid = torch.cat([a[2], torch.ones_like(a[2][:, :1])], dim=1)

    padded = banded_forward(span, cut, valid, torch.tensor([2]), a[4])
    plain = banded_forward(*b)
    assert torch.allclose(padded, plain, atol=1e-9)


@pytest.mark.unit
def test_batching_matches_running_each_example_alone():
    ex = [_toy(band=1, n_t=3, n_u=4, kp1=3, seed=s) for s in (11, 12)]
    alone = torch.cat([banded_forward(*e) for e in ex])
    together = banded_forward(
        torch.cat([e[0] for e in ex]),
        torch.cat([e[1] for e in ex]),
        torch.cat([e[2] for e in ex]),
        torch.cat([e[3] for e in ex]),
        torch.cat([e[4] for e in ex]),
    )
    assert torch.allclose(alone, together, atol=1e-9)


@pytest.mark.unit
def test_every_in_band_partition_is_reachable_and_counted():
    """A sanity check on the toy builder itself, not just the DP.

    If the band never actually admitted more than one path, the monotonicity test
    above would pass vacuously.
    """
    _, cut, cut_valid, n_chunks, n_tokens = _toy(band=1, n_t=3, n_u=4, kp1=3)
    n_t, n_u, kp1 = int(n_chunks[0]), int(n_tokens[0]), 3
    seen = set()
    for combo in itertools.product(range(kp1), repeat=n_t):
        u = 0
        ok = True
        for t, k in enumerate(combo):
            starts = {int(cut[0, t, j]) for j in range(cut.shape[2]) if bool(cut_valid[0, t, j])}
            if u not in starts:
                ok = False
                break
            u += k
        if ok and u == n_u:
            seen.add(combo)
    assert len(seen) > 1


# ---------------------------------------------------------------------------
# The banded 2-D layout builder
# ---------------------------------------------------------------------------

_VS, _VE, _EOT, _PAD = 900, 901, 902, 0


def _chunks():
    """Four chunks over a 6-token transcript, one of them silent."""
    from nemo.collections.speechlm2.parts.script import ChunkSpec

    return [
        ChunkSpec(audio_len=2, target_ids=[10, 11]),
        ChunkSpec(audio_len=2, target_ids=[12]),
        ChunkSpec(audio_len=2, target_ids=[]),
        ChunkSpec(audio_len=2, target_ids=[13, 14, 15]),
    ]


def _build(band, word_starts=(0, 1, 2, 3, 4, 5), instruction=(7, 8)):
    from nemo.collections.speechlm2.parts.script import build_packed_banded_example

    return build_packed_banded_example(
        instruction_ids=list(instruction),
        chunks=_chunks(),
        word_starts=list(word_starts),
        band_words=band,
        vision_start_id=_VS,
        vision_end_id=_VE,
        eot_id=_EOT,
    )


@pytest.mark.unit
def test_band_zero_layout_matches_the_forced_builder():
    """band=0 must lay out exactly what the existing forced builder lays out.

    The spine and the per-segment history prefixes are the contract between the
    two losses; if they drift, a band-0 run is not the control it claims to be.
    """
    from nemo.collections.speechlm2.parts.script import build_packed_chunk_example

    banded = _build(band=0)
    forced = build_packed_chunk_example(
        instruction_ids=[7, 8],
        chunks=_chunks(),
        vision_start_id=_VS,
        vision_end_id=_VE,
        eot_id=_EOT,
    )

    P = banded.spine_len
    assert P == forced.spine_len
    assert torch.equal(banded.input_ids[:P], forced.input_ids[:P])
    assert banded.n_cand == 1
    assert banded.n_chunks == 4 and banded.n_tokens == 6
    # One segment per chunk, each with the aligner's own history prefix.
    for t in range(4):
        seg = (banded.seg_ids == t + 1).nonzero(as_tuple=True)[0]
        f_seg = (forced.seg_ids == t + 1).nonzero(as_tuple=True)[0]
        assert int(banded.prefix_len[seg[0]]) == int(forced.prefix_len[f_seg[0]])


@pytest.mark.unit
def test_band_zero_cuts_are_the_aligner_cuts():
    banded = _build(band=0)
    # chunk targets are 2, 1, 0, 3 tokens long -> cuts at 0, 2, 3, 3
    assert banded.cut.squeeze(-1).tolist() == [0, 2, 3, 3]
    assert banded.cut_valid.all()


@pytest.mark.unit
def test_a_wider_band_only_adds_candidates_and_keeps_the_aligner_cut():
    narrow, wide = _build(band=0), _build(band=1)
    assert wide.n_cand > narrow.n_cand
    for t in range(4):
        wide_cuts = {int(wide.cut[t, j]) for j in range(wide.n_cand) if bool(wide.cut_valid[t, j])}
        assert int(narrow.cut[t, 0]) in wide_cuts


@pytest.mark.unit
def test_the_band_adds_only_word_starts():
    """A cut the BAND introduces may only separate whole words.

    With word starts at 0/2/4 (plus the transcript end at 6), no other position
    may be offered. This is what CHAT's band does not guarantee: its
    ``band_nodes`` sees only per-chunk token counts, so a band-1 lattice there
    admits cuts inside a word and scores the model for emitting half of one.
    """
    aligner = {0, 2, 3, 3}  # this fixture's own chunk boundaries
    b = _build(band=2, word_starts=(0, 2, 4))
    allowed = {0, 2, 4, 6} | aligner
    for t in range(4):
        for j in range(b.n_cand):
            if bool(b.cut_valid[t, j]):
                assert int(b.cut[t, j]) in allowed, f"cut {int(b.cut[t, j])} is neither a word start nor the aligner's"


@pytest.mark.unit
def test_the_aligner_cut_survives_even_when_it_is_not_a_word_start():
    """The forced path must stay inside the band, always.

    Dropping a cut the aligner actually chose would mean band=0 is not the forced
    loss and that widening the band could REMOVE a path -- breaking the
    monotonicity the loss depends on. So the aligner's own cut is force-included
    whatever the word-start table says.
    """
    b = _build(band=2, word_starts=(0, 2, 4))  # 3 is an aligner cut but not a word start
    for t, u in enumerate([0, 2, 3, 3]):
        cuts = {int(b.cut[t, j]) for j in range(b.n_cand) if bool(b.cut_valid[t, j])}
        assert u in cuts


@pytest.mark.unit
def test_prefix_len_tracks_the_candidate_cut():
    """prefix_len is the ONLY thing that distinguishes candidates of a chunk.

    That is exactly why the flat layout needs no mask change: build_script_mask
    already reads each segment's history as ``kp < prefix_len[q]``.
    """
    b = _build(band=1)
    m = 2  # len(instruction)
    for t in range(b.n_chunks):
        for j in range(b.n_cand):
            seg = t * b.n_cand + j + 1
            pos = (b.seg_ids == seg).nonzero(as_tuple=True)[0]
            assert int(b.prefix_len[pos[0]]) == int(b.cut[t, j]) + m


@pytest.mark.unit
def test_every_segment_is_well_formed_including_padding_candidates():
    """Padding candidates still need a valid segment; cut_valid keeps them out."""
    b = _build(band=1)
    n_seg = int(b.seg_ids.max())
    assert n_seg == b.n_chunks * b.n_cand
    for seg in range(1, n_seg + 1):
        pos = (b.seg_ids == seg).nonzero(as_tuple=True)[0]
        assert pos.numel() > 0
        assert int(b.input_ids[pos[0]]) == _VS
        assert _VE in b.input_ids[pos].tolist()


@pytest.mark.unit
def test_branch_ve_abs_points_at_the_vision_end_token():
    """The loss reads K+1 scores from here; a wrong index is silent and fatal."""
    b = _build(band=1)
    for n, a in enumerate(b.branch_ve_abs.tolist()):
        assert int(b.input_ids[a]) == _VE, f"segment {n} ve_abs={a} is not <ve>"


@pytest.mark.unit
def test_span_valid_stops_at_the_end_of_the_transcript():
    b = _build(band=1)
    for t in range(b.n_chunks):
        for j in range(b.n_cand):
            if not bool(b.cut_valid[t, j]):
                continue
            u = int(b.cut[t, j])
            for k in range(b.span_valid.shape[-1]):
                if u + k > b.n_tokens:
                    assert not bool(b.span_valid[t, j, k])


@pytest.mark.unit
def test_branch_targets_are_the_transcript_read_from_the_cut():
    """Each candidate is teacher-forced on the transcript starting at ITS cut."""
    b = _build(band=1)
    from nemo.collections.speechlm2.data.streaming_stt_dataset import IGNORE_INDEX

    transcript = b.input_ids[2 : b.spine_len].tolist()
    for t in range(b.n_chunks):
        for j in range(b.n_cand):
            if not bool(b.cut_valid[t, j]):
                continue
            u = int(b.cut[t, j])
            seg = t * b.n_cand + j + 1
            pos = (b.seg_ids == seg).nonzero(as_tuple=True)[0]
            row = b.target_ids[pos].tolist()
            supervised = [x for x in row if x != IGNORE_INDEX]
            assert supervised == transcript[u : u + len(supervised)]


# ---------------------------------------------------------------------------
# Config defaults and the shipped recipe
# ---------------------------------------------------------------------------

_BANDED_YAML = "examples/speechlm2/conf/streaming_stt_granary2_lora_script_banded1.yaml"


@pytest.mark.unit
def test_loss_type_defaults_to_forced_on_both_sides():
    """Default off, so a requeue under an existing EXP_NAME keeps its objective."""
    from nemo.collections.speechlm2.data.script_dataset import ScriptSTTDataConfig

    cfg = ScriptSTTDataConfig(sample_rate=16000, frame_length_in_secs=0.08, chunk_size=14)
    assert cfg.loss_type == "forced"
    assert cfg.target_construction == "legacy"


@pytest.mark.unit
def test_the_shipped_banded_recipe_satisfies_every_guard():
    """The config the launch script runs must satisfy what the model enforces.

    Each of these makes the (chunk, cut) dynamic program invalid rather than
    merely different, and the model raises on all of them -- but it raises on the
    GRID, after an 8-node allocation has been granted. Checking the shipped YAML
    here turns that into a test failure.
    """
    import os

    from omegaconf import OmegaConf

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    path = os.path.join(root, _BANDED_YAML)
    if not os.path.isfile(path):
        pytest.skip(f"{_BANDED_YAML} not present")
    cfg = OmegaConf.load(path)

    assert cfg.model.loss_type == "banded"
    assert cfg.model.target_construction == "partition"
    # FLAT, not 2-D: the 2-D path measured 5.11 s/step against flat's 0.55 s/step
    # for the identical objective, and the model refuses twod_layout=true.
    assert cfg.model.twod_layout is False
    assert int(cfg.model.band_words) >= 1
    assert not cfg.model.get("gate_in_history", False)
    assert not cfg.model.get("read_write", False)


@pytest.mark.unit
def test_the_shipped_banded_recipe_mirrors_every_paired_key():
    """script_train.py refuses a model/dataset mismatch; the recipe must interpolate.

    If only one side is banded, the batch and the loss disagree about what a
    branch even is -- the dataset would emit forced rows and the DP would read
    candidate cuts that are not there.
    """
    import os

    from omegaconf import OmegaConf

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    path = os.path.join(root, _BANDED_YAML)
    if not os.path.isfile(path):
        pytest.skip(f"{_BANDED_YAML} not present")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    ds = raw["data"]["dataset"]

    for key in ("target_construction", "loss_type", "band_words", "twod_layout"):
        assert ds.get(key) == "${model.%s}" % key, f"data.dataset.{key} must interpolate model.{key}"

    resolved = OmegaConf.load(path)
    for key in ("target_construction", "loss_type", "band_words", "twod_layout"):
        assert resolved.data.dataset[key] == resolved.model[key]


@pytest.mark.unit
def test_row_and_column_indexing_equals_the_naive_slice():
    """The memory-lean gather in _branch_span_logprobs must be exact.

    It indexes rows and columns together so the result is (n*k1,). The obvious
    alternative -- select the rows, then gather the column -- builds an
    (n, k1, vocab) intermediate to read one value out of each row. At vocab
    151936 that intermediate is hundreds of megabytes per micro-batch, recomputed
    in the checkpointed backward. This pins the two as numerically identical, so
    the optimisation cannot silently change the loss.
    """
    from nemo.collections.speechlm2.data.streaming_stt_dataset import IGNORE_INDEX

    torch.manual_seed(0)
    n, b_w, v, k1 = 3, 11, 17, 4
    logits = torch.randn(n, b_w, v, dtype=torch.float64)
    ve = torch.tensor([1, 4, 2])
    idx = (ve.unsqueeze(1) + torch.arange(k1).unsqueeze(0)).clamp(max=b_w - 1)
    tgt = torch.randint(0, v, (n, k1))
    tgt[0, 2] = IGNORE_INDEX

    flat = logits.reshape(-1, v)
    row = (torch.arange(n).unsqueeze(1) * b_w + idx).reshape(-1)

    lean = flat[row, tgt.clamp(min=0).reshape(-1)].view(n, k1)
    naive = flat[row].view(n, k1, v).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)

    assert torch.equal(lean, naive)


@pytest.mark.unit
def test_logsumexp_result_upcast_matches_a_fully_upcast_tensor():
    """Upcasting only the (n, b) logsumexp result, not the (n, b, vocab) logits.

    torch.logsumexp accumulates in fp32 for a bf16 input, so the cheap form has to
    agree with the expensive one to bf16's own resolution.
    """
    torch.manual_seed(1)
    logits = torch.randn(2, 5, 512).to(torch.bfloat16)

    cheap = torch.logsumexp(logits, dim=-1).float()
    expensive = torch.logsumexp(logits.float(), dim=-1)

    torch.testing.assert_close(cheap, expensive, atol=5e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# Cross-family metric comparability
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_script_logs_train_loss_not_bare_loss():
    """The LOGGED key is train_loss, matching the ASR collection.

    CHAT logs 'train_loss'; SCRIPT used to log a bare 'loss', so a wandb panel
    could not carry both families. The key RETURNED from training_step must stay
    'loss' -- Lightning reads it to drive the optimizer -- so this pins the two
    apart, which is exactly the distinction easy to get wrong when renaming.
    """
    import inspect

    from nemo.collections.speechlm2.models import script_model

    src = inspect.getsource(script_model)
    assert '"train_loss": loss' in src, "the logged key should be train_loss"
    assert 'return {"loss": loss}' in src, "the RETURNED key must remain 'loss' for Lightning"
    assert '"loss": loss,\n' not in src, "a bare logged 'loss' key has come back"


@pytest.mark.unit
def test_script_and_chat_agree_on_metric_names():
    """One wandb panel has to carry both families, so the names must match.

    Pins the shared vocabulary rather than each model's full set: CHAT logs extra
    ASR-specific keys and SCRIPT logs extra streaming ones, but these four have to
    mean the same thing on both sides.
    """
    import inspect

    from nemo.collections.asr.models import chat_bpe_models
    from nemo.collections.speechlm2.models import script_model

    chat = inspect.getsource(chat_bpe_models)
    script = inspect.getsource(script_model)

    for name in ("train_loss", "training_batch_wer", "learning_rate"):
        assert name in chat, f"CHAT no longer logs {name}"
        assert name in script, f"SCRIPT no longer logs {name}"


@pytest.mark.unit
def test_both_families_expose_a_comparable_val_wer():
    """val_wer alone is NOT comparable across families, so both extras must exist.

    CHAT's val_wer is verbatim (ASR convention); SCRIPT's is Whisper-normalised
    (speechlm2 convention). On the same manifest that gap read ~0.15 vs ~0.087 and
    was mistaken for a quality difference. Neither native metric may change -- both
    are checkpoint monitors -- so each family gains the OTHER normalisation:
    val_wer_norm on CHAT, val_wer_verbatim on SCRIPT.
    """
    import inspect

    from nemo.collections.asr.models import chat_bpe_models
    from nemo.collections.speechlm2.models import script_model

    assert "val_wer_norm" in inspect.getsource(chat_bpe_models), "CHAT lost its normalised val WER"
    assert "val_wer_verbatim" in inspect.getsource(script_model), "SCRIPT lost its verbatim val WER"


@pytest.mark.unit
def test_normalised_wer_uses_the_shared_edit_distance():
    """Both sides must use nemo's word_error_rate, or the numbers diverge anyway.

    Computing the same metric with a different edit-distance implementation would
    reintroduce the incomparability it exists to remove -- a subtle version of the
    original bug rather than a fix for it.
    """
    import inspect

    from nemo.collections.asr.models import chat_bpe_models

    chat = inspect.getsource(chat_bpe_models)
    assert "word_error_rate_detail" in chat
    assert "editdistance" not in chat, "editdistance is not a dependency of this repo"

    from nemo.collections.speechlm2.parts.metrics import wer as speechlm_wer

    assert "word_error_rate" in inspect.getsource(speechlm_wer)


@pytest.mark.unit
def test_training_wer_is_bounded_and_disableable():
    """A metric must not be able to dominate or kill a run."""
    import dataclasses

    from nemo.collections.speechlm2.models.script_model import ScriptSTTModelConfig

    # Read the DECLARED defaults: the config has required fields, so it cannot be
    # instantiated bare.
    defaults = {f.name: f.default for f in dataclasses.fields(ScriptSTTModelConfig)}
    assert defaults["train_wer_every_n_steps"] >= 100, "too frequent: this is a real decode, not a cheap score"
    assert 1 <= defaults["train_wer_max_utts"] <= 16, "unbounded sample would make the cost batch-dependent"

    import inspect

    from nemo.collections.speechlm2.models.script_model import ScriptSTTModel

    src = inspect.getsource(ScriptSTTModel._maybe_log_training_wer)
    assert "except Exception" in src, "a metric failure must never take down training"
    assert "no_grad" in src, "the decode must not build a graph"
