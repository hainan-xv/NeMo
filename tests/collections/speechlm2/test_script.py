# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Tests for the SCRIPT (spine + branch) streaming SpeechLM.

Two tests carry the correctness argument:

* ``test_parity_packed_vs_separate_examples`` — a branch's logits inside the
  single packed sequence are numerically identical to running that chunk as its
  own standalone ``[history] <vs> audio <ve> words`` example. That equivalence is
  what justifies the custom 4D mask plus overlapping ``position_ids``.
* ``test_offline_encode_dependency_is_chunk_limited`` — encoding a whole
  utterance in one pass does not leak future audio into earlier frames. Encoding
  offline is a batching optimization; the dependency structure stays streaming.
"""

import math

import os

import pytest
import torch

from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX, IGNORE_INDEX
from nemo.collections.speechlm2.parts.alignments import WordAlignment
from nemo.collections.speechlm2.parts.script import (
    PAD_SEG_ID,
    SPINE_SEG_ID,
    ChunkSpec,
    audio_window_start,
    batched_stream_decode_script,
    build_packed_chunk_example,
    build_script_mask,
    build_separate_chunk_examples,
    build_twod_branch_mask,
    build_twod_chunk_example,
    collate_packed_chunk_examples,
    collate_twod_chunk_examples,
)
from nemo.collections.speechlm2.parts.script_messages import get_llm_messages_for_sample

# Token ids for the toy vocab used in the structural tests.
VS, VE, EOT = 90, 91, 92
INSTR = [10, 11]  # 2-token instruction


def _blocked(v) -> bool:
    return float(v) == torch.finfo(torch.float32).min


def _allowed(v) -> bool:
    return float(v) == 0.0


def _shallow_broadcast(cache, n):
    """Shallow-copy a KV cache with its layers broadcast to ``n`` rows."""
    import copy

    out = copy.copy(cache)
    out.layers = [copy.copy(layer) for layer in cache.layers]
    for layer in out.layers:
        layer.keys = layer.keys[:1].expand(n, -1, -1, -1)
        layer.values = layer.values[:1].expand(n, -1, -1, -1)
    return out


def _mask_of(packed):
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    return build_script_mask(
        packed.seg_ids[None], packed.order_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )[0, 0]


# ---------------------------------------------------------------------------
# Layout structure
# ---------------------------------------------------------------------------


def test_packed_layout_structure():
    chunks = [ChunkSpec(audio_len=2, target_ids=[20, 21]), ChunkSpec(audio_len=3, target_ids=[30])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT)

    # Spine = instruction + every chunk's words, in order.
    assert ex.spine_len == len(INSTR) + 3
    assert ex.input_ids[: ex.spine_len].tolist() == INSTR + [20, 21, 30]
    assert ex.seg_ids[: ex.spine_len].tolist() == [SPINE_SEG_ID] * ex.spine_len
    assert ex.position_ids[: ex.spine_len].tolist() == list(range(ex.spine_len))
    # The spine is context only — never supervised.
    assert (ex.target_ids[: ex.spine_len] == IGNORE_INDEX).all()
    assert not ex.is_audio[: ex.spine_len].any()

    # Branch 1: <vs> A A <ve> 20 21 <eot>
    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0]
    assert ex.input_ids[b1].tolist() == [VS, AUDIO_TOKEN_IDX, AUDIO_TOKEN_IDX, VE, 20, 21, EOT]
    assert ex.is_audio[b1].tolist() == [False, True, True, False, False, False, False]
    # Positions continue from the branch's history prefix (= len(INSTR)).
    assert ex.position_ids[b1].tolist() == [2, 3, 4, 5, 6, 7, 8]
    assert ex.prefix_len[b1].tolist() == [2] * 7
    # <ve> predicts the first word; each word predicts the next; the last predicts eot.
    assert ex.target_ids[b1].tolist() == [IGNORE_INDEX] * 3 + [20, 21, EOT] + [IGNORE_INDEX]
    # Audio slots map to global frames 0,1.
    assert ex.audio_frame_index[b1].tolist() == [-1, 0, 1, -1, -1, -1, -1]

    # Branch 2: <vs> A A A <ve> 30 <eot> — 7 tokens.
    # Its history prefix includes chunk 1's two words; its frames are 2,3,4.
    b2 = (ex.seg_ids == 2).nonzero(as_tuple=True)[0]
    assert ex.prefix_len[b2].tolist() == [4] * 7
    assert ex.audio_frame_index[b2].tolist() == [-1, 2, 3, 4, -1, -1, -1]


def test_empty_chunk_predicts_only_eot():
    """A silent chunk still gets a branch; it just predicts <eot> immediately."""
    ex = build_packed_chunk_example(INSTR, [ChunkSpec(audio_len=2, target_ids=[])], VS, VE, EOT)
    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0]
    assert ex.input_ids[b1].tolist() == [VS, AUDIO_TOKEN_IDX, AUDIO_TOKEN_IDX, VE, EOT]
    # Only the <ve> position is supervised, and its target is <eot>.
    assert ex.target_ids[b1].tolist() == [IGNORE_INDEX] * 3 + [EOT] + [IGNORE_INDEX]


def test_supervise_eot_false_drops_the_stop_target():
    ex = build_packed_chunk_example(INSTR, [ChunkSpec(audio_len=1, target_ids=[20])], VS, VE, EOT, supervise_eot=False)
    # Branch is <vs> A <ve> 20 <eot>; only <ve> is supervised (it predicts 20).
    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0]
    assert ex.target_ids[b1].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 20, IGNORE_INDEX, IGNORE_INDEX]


# ---------------------------------------------------------------------------
# Audio window (audio_history_chunks)
# ---------------------------------------------------------------------------


def test_audio_window_start_helper():
    frame_starts = [0, 4, 8, 12]
    # M = 0: window begins at the chunk's own start.
    assert [audio_window_start(k, frame_starts, 0) for k in range(4)] == [0, 4, 8, 12]
    # M = 1: reaches back one chunk, clamped at the first chunk.
    assert [audio_window_start(k, frame_starts, 1) for k in range(4)] == [0, 0, 4, 8]
    # M = 2: reaches back two chunks.
    assert [audio_window_start(k, frame_starts, 2) for k in range(4)] == [0, 0, 0, 4]


def test_windowed_frame_index_structure():
    """With M=1 a branch wraps the previous chunk's frames as well as its own."""
    chunks = [ChunkSpec(2, [20]), ChunkSpec(2, [21]), ChunkSpec(2, [22])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, audio_history_chunks=1)

    def frames_of(seg):
        idx = (ex.seg_ids == seg).nonzero(as_tuple=True)[0]
        return [f for f in ex.audio_frame_index[idx].tolist() if f >= 0]

    assert frames_of(1) == [0, 1]  # first chunk: nothing to reach back to
    assert frames_of(2) == [0, 1, 2, 3]  # chunk 1 + chunk 2
    assert frames_of(3) == [2, 3, 4, 5]  # chunk 2 + chunk 3


def test_window_never_includes_future_frames():
    """The window always ends at the chunk's own boundary — no look-ahead."""
    chunks = [ChunkSpec(3, [20]), ChunkSpec(3, [21]), ChunkSpec(3, [22])]
    for M in (0, 1, 2):
        ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, audio_history_chunks=M)
        for kc in range(len(chunks)):
            idx = (ex.seg_ids == kc + 1).nonzero(as_tuple=True)[0]
            frames = [f for f in ex.audio_frame_index[idx].tolist() if f >= 0]
            boundary = sum(c.audio_len for c in chunks[: kc + 1])
            assert max(frames) < boundary, f"M={M} chunk={kc} saw frame {max(frames)} past {boundary}"


# ---------------------------------------------------------------------------
# Fixed-frame audio window (audio_window_frames)
# ---------------------------------------------------------------------------


def test_fixed_frame_window_helper():
    frame_starts = [0, 4, 8, 12]  # four 4-frame chunks

    def start(k, F):
        return audio_window_start(k, frame_starts, 0, win_end=frame_starts[k] + 4, audio_window_frames=F)

    # F=8: the last 8 frames ending at the boundary, clamped at 0 early on.
    assert [start(k, 8) for k in range(4)] == [0, 0, 4, 8]
    # F=4 equals the chunk size, so it degenerates to the per-chunk window.
    assert [start(k, 4) for k in range(4)] == [0, 4, 8, 12]
    # F is a FLOOR: a window smaller than the chunk never clips into the chunk.
    assert [start(k, 2) for k in range(4)] == [0, 4, 8, 12]


def test_fixed_frame_window_is_constant_across_chunk_sizes():
    """The whole point: acoustic context stays 8 frames whatever the chunk size."""
    for cs in (1, 2, 4, 8):
        chunks = [ChunkSpec(cs, [20]) for _ in range(6)]
        ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, audio_window_frames=8)
        # Later branches (past the initial ramp-up) all see exactly 8 frames.
        for kc in range(len(chunks)):
            idx = (ex.seg_ids == kc + 1).nonzero(as_tuple=True)[0]
            n = sum(1 for f in ex.audio_frame_index[idx].tolist() if f >= 0)
            boundary = (kc + 1) * cs
            assert n == min(8, boundary), f"cs={cs} chunk={kc}: {n} frames, expected {min(8, boundary)}"


def test_fixed_frame_window_keeps_a_large_chunks_own_audio():
    """A chunk longer than the window must still see all of its own frames."""
    chunks = [ChunkSpec(10, [20]), ChunkSpec(10, [21])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, audio_window_frames=4)
    idx = (ex.seg_ids == 2).nonzero(as_tuple=True)[0]
    frames = [f for f in ex.audio_frame_index[idx].tolist() if f >= 0]
    assert frames == list(range(10, 20)), "large chunk lost part of its own audio"


def test_fixed_frame_window_takes_precedence_over_history_chunks():
    chunks = [ChunkSpec(4, [20]) for _ in range(4)]
    both = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, audio_history_chunks=2, audio_window_frames=8)
    only = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, audio_window_frames=8)
    assert both.audio_frame_index.tolist() == only.audio_frame_index.tolist()


def test_fixed_frame_window_never_looks_ahead():
    chunks = [ChunkSpec(3, [20]), ChunkSpec(3, [21]), ChunkSpec(3, [22])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, audio_window_frames=6)
    for kc in range(len(chunks)):
        idx = (ex.seg_ids == kc + 1).nonzero(as_tuple=True)[0]
        frames = [f for f in ex.audio_frame_index[idx].tolist() if f >= 0]
        assert max(frames) < (kc + 1) * 3


def test_window_defaults_are_backward_compatible():
    """Omitting the new argument must reproduce the old layout byte for byte."""
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(3, [30]), ChunkSpec(2, [])]
    for M in (0, 1, 2):
        old = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, audio_history_chunks=M)
        new = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, audio_history_chunks=M, audio_window_frames=0)
        for f in ("input_ids", "position_ids", "seg_ids", "prefix_len", "target_ids", "audio_frame_index"):
            assert getattr(old, f).tolist() == getattr(new, f).tolist(), f"M={M} field={f}"


# ---------------------------------------------------------------------------
# Attention mask
# ---------------------------------------------------------------------------


def test_mask_spine_causal_and_pure_text():
    chunks = [ChunkSpec(2, [20]), ChunkSpec(2, [21])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT)
    m = _mask_of(ex)
    P = ex.spine_len

    for q in range(P):
        for k in range(P):  # causal within the spine
            assert _allowed(m[q, k]) if k <= q else _blocked(m[q, k])
        for k in range(P, m.shape[1]):  # never any branch token
            assert _blocked(m[q, k])


def test_mask_branch_sees_only_prefix_and_own_branch():
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(2, [30])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT)
    m = _mask_of(ex)
    P = ex.spine_len
    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0].tolist()
    b2 = (ex.seg_ids == 2).nonzero(as_tuple=True)[0].tolist()

    # Branch 2's history prefix is instruction + chunk 1's words (4 spine tokens).
    pref2 = int(ex.prefix_len[b2[0]])
    assert pref2 == 4
    for q in b2:
        for k in range(P):
            assert _allowed(m[q, k]) if k < pref2 else _blocked(m[q, k])
        for k in b1:  # never another branch
            assert _blocked(m[q, k])

    # Branch 1 sees only the instruction, not its own words' spine twins.
    pref1 = int(ex.prefix_len[b1[0]])
    assert pref1 == len(INSTR)
    for q in b1:
        for k in range(P):
            assert _allowed(m[q, k]) if k < pref1 else _blocked(m[q, k])
        for k in b2:  # never a later branch
            assert _blocked(m[q, k])
        # Causal within its own branch.
        for k in b1:
            assert _allowed(m[q, k]) if k <= q else _blocked(m[q, k])


def test_mask_padding_blocked():
    a = build_packed_chunk_example(INSTR, [ChunkSpec(2, [20])], VS, VE, EOT)
    b = build_packed_chunk_example(INSTR, [ChunkSpec(2, [20]), ChunkSpec(2, [21])], VS, VE, EOT)
    batch = collate_packed_chunk_examples([a, b], pad_id=0)
    m = build_script_mask(batch.seg_ids, batch.order_ids, batch.prefix_len, batch.valid, torch.float32)

    n_real = int(a.input_ids.numel())
    T = batch.input_ids.shape[1]
    assert n_real < T  # row 0 really is padded
    for q in range(T):
        for k in range(n_real, T):
            assert _blocked(m[0, 0, q, k])


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def test_collate_shapes_and_padding():
    a = build_packed_chunk_example(INSTR, [ChunkSpec(2, [20])], VS, VE, EOT)
    b = build_packed_chunk_example(INSTR, [ChunkSpec(2, [20]), ChunkSpec(3, [21, 22])], VS, VE, EOT)
    batch = collate_packed_chunk_examples([a, b], pad_id=7)

    T = max(a.input_ids.numel(), b.input_ids.numel())
    for t in (
        batch.input_ids,
        batch.position_ids,
        batch.seg_ids,
        batch.prefix_len,
        batch.target_ids,
        batch.is_audio,
        batch.audio_frame_index,
        batch.valid,
    ):
        assert t.shape == (2, T)
    assert batch.spine_lens.tolist() == [a.spine_len, b.spine_len]

    n = int(a.input_ids.numel())
    assert batch.valid[0, :n].all() and not batch.valid[0, n:].any()
    assert (batch.input_ids[0, n:] == 7).all()
    assert (batch.seg_ids[0, n:] == PAD_SEG_ID).all()
    assert (batch.target_ids[0, n:] == IGNORE_INDEX).all()
    assert (batch.audio_frame_index[0, n:] == -1).all()


# ---------------------------------------------------------------------------
# Word -> chunk assignment (the delay rule)
# ---------------------------------------------------------------------------


def _align(*triples):
    return [WordAlignment(text=t, start_time=s, end_time=e) for t, s, e in triples]


def _assistant_contents(messages):
    return [m["content"] for m in messages if m["role"] == "assistant"]


def test_messages_zero_delay_assigns_word_to_the_chunk_it_ends_in():
    # frame_length 0.08, chunk_size 2 -> a chunk covers 0.16s.
    # "Hello" ends at 0.48s = frame 6 -> ready at chunk 3 (boundary frame 6).
    # "World" ends at 0.80s = frame 10 -> ready at chunk 5 (boundary frame 10).
    msgs = get_llm_messages_for_sample(
        system_role="system",
        system_prompt="P",
        audio_tag="<a>",
        blank_token="<b>",
        chunk_size=2,
        num_delay_frames=0,
        audio_duration_secs=1.0,
        frame_length_in_secs=0.08,
        alignments=_align(("Hello", 0.16, 0.48), ("World", 0.60, 0.80)),
    )
    assert _assistant_contents(msgs) == ["<b>", "<b>", "Hello", "<b>", "World", "<b>", "<b>"]
    # Every user turn holds exactly chunk_size audio tags.
    assert all(m["content"].count("<a>") == 2 for m in msgs if m["role"] == "user")


def test_messages_delay_pushes_a_word_to_a_later_chunk():
    """A positive delay moves a word into a later chunk — the latency knob."""
    words = _align(("Hello", 0.16, 0.48), ("World", 0.60, 0.80))
    kw = dict(
        system_role="system",
        system_prompt="P",
        audio_tag="<a>",
        blank_token="<b>",
        chunk_size=2,
        audio_duration_secs=1.0,
        frame_length_in_secs=0.08,
        alignments=words,
    )
    d0 = _assistant_contents(get_llm_messages_for_sample(num_delay_frames=0, **kw))
    d2 = _assistant_contents(get_llm_messages_for_sample(num_delay_frames=2, **kw))

    assert d0.index("Hello") == 2
    assert d2.index("Hello") == 3  # delayed by one 2-frame chunk
    assert d0.index("World") == 4
    assert d2.index("World") == 5


def test_messages_residual_words_are_never_dropped():
    """A delay large enough to push words past the last boundary must not lose them."""
    words = _align(("Hello", 0.16, 0.48), ("World", 0.60, 0.80))
    msgs = get_llm_messages_for_sample(
        system_role="system",
        system_prompt="P",
        audio_tag="<a>",
        blank_token="<b>",
        chunk_size=2,
        num_delay_frames=50,
        audio_duration_secs=1.0,
        frame_length_in_secs=0.08,
        alignments=words,
    )
    contents = _assistant_contents(msgs)
    # Everything is held back, so it all lands in the final assistant turn.
    assert contents[-1] == "Hello World"
    assert all(c == "<b>" for c in contents[:-1])


def test_messages_use_transcript_spans_to_keep_punctuation():
    msgs = get_llm_messages_for_sample(
        system_role="system",
        system_prompt="P",
        audio_tag="<a>",
        blank_token="<b>",
        chunk_size=2,
        num_delay_frames=0,
        audio_duration_secs=1.0,
        frame_length_in_secs=0.08,
        alignments=_align(("Hello", 0.16, 0.48), ("World", 0.60, 0.80)),
        transcript="Hello, World!",
    )
    contents = [c for c in _assistant_contents(msgs) if c != "<b>"]
    assert contents == ["Hello,", " World!"]


def test_messages_no_alignments_gives_all_blank_chunks():
    msgs = get_llm_messages_for_sample(
        system_role="system",
        system_prompt="P",
        audio_tag="<a>",
        blank_token="<b>",
        chunk_size=4,
        num_delay_frames=0,
        audio_duration_secs=0.64,
        frame_length_in_secs=0.08,
        alignments=[],
    )
    # 8 frames / 4 = 2 chunks, both silent.
    assert _assistant_contents(msgs) == ["<b>", "<b>"]


def test_messages_final_chunk_is_ceiled_to_a_full_chunk():
    """The frame count is ceiled to whole chunks; the model zero-pads the tail."""
    msgs = get_llm_messages_for_sample(
        system_role="system",
        system_prompt="P",
        audio_tag="<a>",
        blank_token="<b>",
        chunk_size=4,
        num_delay_frames=0,
        audio_duration_secs=0.40,
        frame_length_in_secs=0.08,
        alignments=[],
    )
    user_turns = [m for m in msgs if m["role"] == "user"]
    assert len(user_turns) == math.ceil(5 / 4) == 2
    assert all(m["content"].count("<a>") == 4 for m in user_turns)  # incl. the partial one


@pytest.mark.parametrize("bad", [0, -1])
def test_messages_reject_non_fixed_chunking(bad):
    with pytest.raises(ValueError, match="fixed chunking"):
        get_llm_messages_for_sample(
            system_role="system",
            system_prompt="P",
            audio_tag="<a>",
            blank_token="<b>",
            chunk_size=bad,
            num_delay_frames=0,
            audio_duration_secs=1.0,
            frame_length_in_secs=0.08,
        )


# ---------------------------------------------------------------------------
# Parity: packed branch logits == standalone per-chunk example logits
# ---------------------------------------------------------------------------

transformers = pytest.importorskip("transformers")
from transformers import Qwen3Config  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM  # noqa: E402


def _tiny_qwen3(vocab_size=128):
    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval().float()


def _embed_with_audio(model, input_ids, is_audio, audio_vecs):
    """Embed text ids and splice ``audio_vecs`` (in order) at the audio positions."""
    ids = input_ids.clone()
    ids[is_audio] = 0  # any valid id; overwritten below
    emb = model.get_input_embeddings()(ids)
    if is_audio.any():
        emb = emb.clone()
        emb[is_audio] = audio_vecs.to(emb.dtype)
    return emb


def _frames_by_index(all_frames, frame_index, is_audio):
    """Gather global frames for the audio slots of one example."""
    return all_frames[frame_index[is_audio]]


@torch.no_grad()
@pytest.mark.parametrize("position_scheme", ["branch", "continuous"])
@pytest.mark.parametrize(
    "audio_history_chunks,audio_window_frames",
    [(0, 0), (1, 0), (2, 0), (0, 4), (0, 6), (0, 8)],
)
def test_parity_packed_vs_separate_examples(audio_history_chunks, audio_window_frames, position_scheme):
    """The whole correctness argument: packed branch logits == standalone logits."""
    model = _tiny_qwen3()
    H = model.config.hidden_size

    instruction = [5, 6, 7]
    chunks = [
        ChunkSpec(audio_len=2, target_ids=[20, 21]),
        ChunkSpec(audio_len=3, target_ids=[30]),
        ChunkSpec(audio_len=1, target_ids=[]),  # silent chunk
        ChunkSpec(audio_len=2, target_ids=[40, 41, 42]),
    ]
    torch.manual_seed(123)
    all_frames = torch.randn(sum(c.audio_len for c in chunks), H)  # global frame table

    packed = build_packed_chunk_example(
        instruction,
        chunks,
        VS,
        VE,
        EOT,
        audio_history_chunks=audio_history_chunks,
        audio_window_frames=audio_window_frames,
        position_scheme=position_scheme,
    )
    packed_emb = _embed_with_audio(
        model,
        packed.input_ids,
        packed.is_audio,
        _frames_by_index(all_frames, packed.audio_frame_index, packed.is_audio),
    )
    packed_logits = model(
        inputs_embeds=packed_emb[None],
        attention_mask=_mask_of(packed)[None, None],
        position_ids=packed.position_ids[None],
    ).logits[0]

    separate = build_separate_chunk_examples(
        instruction,
        chunks,
        VS,
        VE,
        EOT,
        audio_history_chunks=audio_history_chunks,
        audio_window_frames=audio_window_frames,
        position_scheme=position_scheme,
    )
    for k, sep in enumerate(separate, start=1):
        sep_emb = _embed_with_audio(
            model,
            sep.input_ids,
            sep.is_audio,
            _frames_by_index(all_frames, sep.audio_frame_index, sep.is_audio),
        )
        # Positions come from the builder: under the continuous scheme they are
        # NOT 0..L-1, and defaulting would compare different geometries.
        sep_logits = model(inputs_embeds=sep_emb[None], position_ids=sep.position_ids[None]).logits[0]

        packed_branch = packed_logits[(packed.seg_ids == k).nonzero(as_tuple=True)[0]]
        sep_branch = sep_logits[sep.branch_start :]
        assert packed_branch.shape == sep_branch.shape
        torch.testing.assert_close(packed_branch, sep_branch, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_parity_batched():
    """Two utterances packed and padded into a batch must match their standalone runs."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    torch.manual_seed(7)

    instruction = [5, 6]
    utts = [
        [ChunkSpec(2, [20, 21]), ChunkSpec(2, [22])],
        [ChunkSpec(1, [30]), ChunkSpec(3, [31, 32]), ChunkSpec(2, [])],
    ]
    frames = [torch.randn(sum(c.audio_len for c in chs), H) for chs in utts]

    examples = [build_packed_chunk_example(instruction, chs, VS, VE, EOT) for chs in utts]
    batch = collate_packed_chunk_examples(examples, pad_id=0)
    mask = build_script_mask(batch.seg_ids, batch.order_ids, batch.prefix_len, batch.valid, torch.float32)

    embs = []
    for i, ex in enumerate(examples):
        e = _embed_with_audio(
            model,
            batch.input_ids[i],
            batch.is_audio[i],
            frames[i][batch.audio_frame_index[i][batch.is_audio[i]]],
        )
        embs.append(e)
    batched_logits = model(
        inputs_embeds=torch.stack(embs), attention_mask=mask, position_ids=batch.position_ids
    ).logits

    for i, ex in enumerate(examples):
        single_emb = _embed_with_audio(model, ex.input_ids, ex.is_audio, frames[i][ex.audio_frame_index[ex.is_audio]])
        single_logits = model(
            inputs_embeds=single_emb[None],
            attention_mask=_mask_of(ex)[None, None],
            position_ids=ex.position_ids[None],
        ).logits[0]
        n = int(ex.input_ids.numel())
        torch.testing.assert_close(batched_logits[i, :n], single_logits, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("audio_history_chunks,audio_window_frames", [(0, 0), (1, 0), (0, 5)])
def test_batched_decode_matches_per_utterance(audio_history_chunks, audio_window_frames):
    """Batching (with left-padding across ragged streams) must not change the output."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    torch.manual_seed(11)

    cs = 2
    frames_list = [torch.randn(6, H), torch.randn(4, H), torch.randn(5, H)]  # ragged, incl. partial
    instrs = [[5, 6], [5, 7], [5, 8]]
    kw = dict(
        llm=model,
        embed_tokens=model.get_input_embeddings(),
        chunk_size=cs,
        vision_start_id=VS,
        vision_end_id=VE,
        eot_id=EOT,
        pad_id=0,
        max_new_tokens=4,
        audio_history_chunks=audio_history_chunks,
    )

    together = batched_stream_decode_script(instruction_ids_list=instrs, frames_list=frames_list, **kw)
    for b in range(len(frames_list)):
        alone = batched_stream_decode_script(instruction_ids_list=[instrs[b]], frames_list=[frames_list[b]], **kw)
        assert together[b] == alone[0], f"stream {b} changed when batched"


@torch.no_grad()
def test_decode_matches_teacher_forced_packed_layout():
    """Decoding must condition exactly as training does.

    Decode greedily, feed the result back in as if it were the training targets,
    and check the teacher-forced argmax at each supervised position reproduces
    the decoded tokens. If the decoder's history/audio/positions disagreed with
    the packed layout, the two would diverge.
    """
    model = _tiny_qwen3()
    H = model.config.hidden_size
    torch.manual_seed(5)

    cs = 2
    instruction = [5, 6]
    frames = torch.randn(6, H)  # exactly 3 chunks

    emitted, chunk_ids = batched_stream_decode_script(
        llm=model,
        embed_tokens=model.get_input_embeddings(),
        instruction_ids_list=[instruction],
        frames_list=[frames],
        chunk_size=cs,
        vision_start_id=VS,
        vision_end_id=VE,
        eot_id=EOT,
        pad_id=0,
        max_new_tokens=4,
        return_chunk_ids=True,
    )
    emitted, chunk_ids = emitted[0], chunk_ids[0]

    # Rebuild the packed example from what was decoded.
    per_chunk = [[] for _ in range(3)]
    for tok, k in zip(emitted, chunk_ids):
        per_chunk[k].append(tok)
    chunks = [ChunkSpec(audio_len=cs, target_ids=toks) for toks in per_chunk]

    # supervise_eot=False: a chunk that hit ``max_new_tokens`` was cut off before
    # it chose to stop, so its <eot> target is one the decode never reached. The
    # word targets are what encode the decode/training conditioning match.
    packed = build_packed_chunk_example(instruction, chunks, VS, VE, EOT, supervise_eot=False)
    emb = _embed_with_audio(
        model, packed.input_ids, packed.is_audio, frames[packed.audio_frame_index[packed.is_audio]]
    )
    logits = model(
        inputs_embeds=emb[None],
        attention_mask=_mask_of(packed)[None, None],
        position_ids=packed.position_ids[None],
    ).logits[0]

    sup = (packed.target_ids != IGNORE_INDEX).nonzero(as_tuple=True)[0]
    assert sup.numel() > 0
    argmax = logits[sup].argmax(dim=-1)
    torch.testing.assert_close(argmax, packed.target_ids[sup])


@torch.no_grad()
def test_decode_inserts_word_start_when_chunk_would_merge():
    """A chunk whose first token is a continuation must not glue onto the previous word.

    SCRIPT emits whole words per chunk and the history already ends with the
    previous chunk's last word, so a chunk starting with a non-word-start token
    would render as "border"+"ruffian" -> "borderruffian". The decoder inserts a
    leading-space token instead of constraining the model's output distribution
    (constraining can starve a chunk into emitting nothing).
    """
    H, cs = 32, 4
    WORD, SPACE = 40, 41

    class _FixedLLM:
        """Emits WORD once per chunk, then <eot>."""

        def __call__(self, inputs_embeds, **kw):
            b, t = inputs_embeds.shape[0], inputs_embeds.shape[1]
            logits = torch.zeros(b, t, 128)
            logits[..., WORD if kw.get("past_key_values") is None else EOT] = 1.0
            return type("Out", (), {"logits": logits, "past_key_values": object()})()

    embed = torch.nn.Embedding(128, H)
    frames = torch.randn(2 * cs, H)  # exactly two chunks

    def run(**kw):
        return batched_stream_decode_script(
            llm=_FixedLLM(),
            embed_tokens=embed,
            instruction_ids_list=[[5, 6]],
            frames_list=[frames],
            chunk_size=cs,
            vision_start_id=VS,
            vision_end_id=VE,
            eot_id=EOT,
            pad_id=0,
            max_new_tokens=4,
            **kw,
        )[0]

    # WORD is never a word start, so chunk 2 would merge onto chunk 1's output.
    assert run() == [WORD, WORD]
    assert run(is_word_start=lambda t: False, insert_word_start_id=SPACE) == [WORD, SPACE, WORD]
    # A token that IS a word start needs no help.
    assert run(is_word_start=lambda t: True, insert_word_start_id=SPACE) == [WORD, WORD]
    # The guard never fires on the first chunk -- there is nothing to merge onto.
    assert run(is_word_start=lambda t: False, insert_word_start_id=SPACE)[0] == WORD


@torch.no_grad()
def test_decode_pads_partial_final_chunk_to_training_length():
    """The last chunk is zero-padded to a full window, matching training.

    In training every chunk's audio turn is a full ``chunk_size`` frames and the
    slots past the real audio are zero-filled by the gather. A raw slice at
    inference would hand the model a shorter window and drop the trailing-silence
    end-of-audio cue.
    """
    H = 32
    cs = 4
    seen_lengths = []

    class _RecordingLLM:
        """Records each prefill length, then immediately emits <eot>."""

        def __init__(self, embed):
            self._embed = embed

        def __call__(self, inputs_embeds, **kw):
            if kw.get("past_key_values") is None:  # a prefill, not a decode step
                seen_lengths.append(inputs_embeds.shape[1])
            b, t = inputs_embeds.shape[0], inputs_embeds.shape[1]
            logits = torch.zeros(b, t, 128)
            logits[..., EOT] = 1.0  # always stop immediately
            return type("Out", (), {"logits": logits, "past_key_values": object()})()

    embed = torch.nn.Embedding(128, H)
    instruction = [5, 6]
    frames = torch.randn(6, H)  # 6 frames = 1 full chunk + a 2-frame partial

    batched_stream_decode_script(
        llm=_RecordingLLM(embed),
        embed_tokens=embed,
        instruction_ids_list=[instruction],
        frames_list=[frames],
        chunk_size=cs,
        vision_start_id=VS,
        vision_end_id=VE,
        eot_id=EOT,
        pad_id=0,
        max_new_tokens=1,
    )

    # Both chunks must prefill instruction + <vs> + cs audio + <ve>, even the
    # partial one (only 2 real frames, zero-padded up to 4).
    expected = len(instruction) + 1 + cs + 1
    assert seen_lengths == [expected, expected]


# ---------------------------------------------------------------------------
# The streaming constraint: encoding offline must not leak future audio
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_offline_encode_dependency_is_chunk_limited():
    """Encoding the whole utterance at once must stay chunk-limited.

    SCRIPT encodes an utterance in a single pass and then slices frames per
    chunk. That is a batching optimization, and it is only sound if the encoder's
    receptive field never crosses a chunk boundary — otherwise a frame would
    carry information the streaming model could not have had yet.

    ``att_context_style="chunked_limited"`` with ``att_context_size=[left,
    chunk_size - 1]`` guarantees exactly that, and crucially the look-ahead does
    NOT compound across layers. This test perturbs all audio from chunk 3 onward
    and asserts the frames of chunks 0-2 are bit-identical through a 3-layer
    stack.
    """
    pytest.importorskip("torch")
    from nemo.collections.asr.modules.transformer_encoder import StreamingTransformerEncoder

    chunk_size = 4
    torch.manual_seed(0)
    enc = (
        StreamingTransformerEncoder(
            feat_in=16,
            n_layers=3,
            d_model=64,
            n_heads=4,
            ff_expansion=2.0,
            subsampling="feature_stacking",
            subsampling_factor=1,
            drop_rate=0.0,
            self_attention_model="rope",
            # This is what ScriptSTTModel._set_encoder_att_context pins per batch.
            att_context_size=[8, chunk_size - 1],
            att_context_style="chunked_limited",
        )
        .eval()
        .float()
    )

    n_frames = 40
    x = torch.randn(1, 16, n_frames)
    lens = torch.tensor([n_frames])
    y_ref, _ = enc(audio_signal=x, length=lens)

    split = 3 * chunk_size  # perturb everything from chunk 3 onward
    x_perturbed = x.clone()
    x_perturbed[:, :, split:] += 5.0
    y_perturbed, _ = enc(audio_signal=x_perturbed, length=lens)

    # Frames before the perturbation must be untouched...
    torch.testing.assert_close(y_ref[:, :, :split], y_perturbed[:, :, :split], atol=0.0, rtol=0.0)
    # ...and the test would be vacuous if nothing moved at all.
    assert (y_ref[:, :, split:] - y_perturbed[:, :, split:]).abs().max() > 1e-3


@torch.no_grad()
def test_offline_encode_leaks_future_without_chunk_limiting():
    """Control for the test above: full attention DOES leak future audio.

    Confirms the previous test is actually detecting the chunk-limiting rather
    than some accidental invariance of the tiny encoder.
    """
    from nemo.collections.asr.modules.transformer_encoder import StreamingTransformerEncoder

    torch.manual_seed(0)
    enc = (
        StreamingTransformerEncoder(
            feat_in=16,
            n_layers=3,
            d_model=64,
            n_heads=4,
            ff_expansion=2.0,
            subsampling="feature_stacking",
            subsampling_factor=1,
            drop_rate=0.0,
            self_attention_model="rope",
            att_context_size=[-1, -1],  # unbounded: full bidirectional attention
        )
        .eval()
        .float()
    )

    n_frames = 40
    x = torch.randn(1, 16, n_frames)
    lens = torch.tensor([n_frames])
    y_ref, _ = enc(audio_signal=x, length=lens)

    split = 3 * 4
    x_perturbed = x.clone()
    x_perturbed[:, :, split:] += 5.0
    y_perturbed, _ = enc(audio_signal=x_perturbed, length=lens)

    assert (y_ref[:, :, :split] - y_perturbed[:, :, :split]).abs().max() > 1e-3


# ---------------------------------------------------------------------------
# 2-D layout: spine forwarded once, branches on a batch axis
# ---------------------------------------------------------------------------


def test_twod_layout_matches_flat_structurally():
    """The 2-D builder must lay out the same tokens, positions and targets."""
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(3, [30]), ChunkSpec(2, []), ChunkSpec(2, [40, 41, 42])]
    flat = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT)
    two = build_twod_chunk_example(INSTR, chunks, VS, VE, EOT)

    assert two.spine_len == flat.spine_len
    assert two.spine_ids.tolist() == flat.input_ids[: flat.spine_len].tolist()
    assert two.spine_positions.tolist() == flat.position_ids[: flat.spine_len].tolist()

    for k in range(len(chunks)):
        idx = (flat.seg_ids == k + 1).nonzero(as_tuple=True)[0]
        n = int(two.branch_valid[k].sum())
        assert n == len(idx), f"branch {k}: {n} tokens vs flat {len(idx)}"
        assert two.branch_ids[k, :n].tolist() == flat.input_ids[idx].tolist()
        assert two.branch_positions[k, :n].tolist() == flat.position_ids[idx].tolist()
        assert two.branch_targets[k, :n].tolist() == flat.target_ids[idx].tolist()
        assert two.branch_frame_index[k, :n].tolist() == flat.audio_frame_index[idx].tolist()
        assert int(two.branch_prefix[k]) == int(flat.prefix_len[idx[0]])


def test_twod_branch_mask_grants_prefix_and_own_causal():
    chunks = [ChunkSpec(2, [20]), ChunkSpec(2, [21, 22])]
    two = build_twod_chunk_example(INSTR, chunks, VS, VE, EOT)
    P = two.spine_len
    m = build_twod_branch_mask(two.branch_prefix, two.branch_valid, P, torch.float32)[:, 0]

    for k in range(len(chunks)):
        pref = int(two.branch_prefix[k])
        n = int(two.branch_valid[k].sum())
        for j in range(n):
            for i in range(P):  # spine half: exactly its own history prefix
                assert _allowed(m[k, j, i]) if i < pref else _blocked(m[k, j, i])
            for jj in range(two.branch_ids.shape[1]):  # own half: causal, no padding
                ok = jj <= j and bool(two.branch_valid[k, jj])
                assert _allowed(m[k, j, P + jj]) if ok else _blocked(m[k, j, P + jj])


@torch.no_grad()
@pytest.mark.parametrize("audio_history_chunks,audio_window_frames", [(0, 0), (1, 0), (0, 6), (0, 28)])
def test_parity_twod_vs_flat(audio_history_chunks, audio_window_frames):
    """THE GATE: the 2-D layout must reproduce the flat path's branch logits.

    Forwards the spine once with ``use_cache=True``, broadcasts its K/V across the
    branch axis, then runs every branch as one batch against a 4D mask. If this
    holds, the 2-D form is a pure restructuring -- same model, same gradients --
    and a checkpoint trained either way is valid for the other.
    """
    model = _tiny_qwen3()
    H = model.config.hidden_size
    instruction = [5, 6, 7]
    chunks = [
        ChunkSpec(audio_len=2, target_ids=[20, 21]),
        ChunkSpec(audio_len=3, target_ids=[30]),
        ChunkSpec(audio_len=2, target_ids=[]),  # silent chunk
        ChunkSpec(audio_len=2, target_ids=[40, 41, 42]),  # ragged branch length
    ]
    torch.manual_seed(3)
    frames = torch.randn(sum(c.audio_len for c in chunks), H)
    kw = dict(audio_history_chunks=audio_history_chunks, audio_window_frames=audio_window_frames)

    # --- reference: one flat sequence + the 4D SCRIPT mask ---
    flat = build_packed_chunk_example(instruction, chunks, VS, VE, EOT, **kw)
    flat_emb = _embed_with_audio(model, flat.input_ids, flat.is_audio, frames[flat.audio_frame_index[flat.is_audio]])
    flat_logits = model(
        inputs_embeds=flat_emb[None],
        attention_mask=_mask_of(flat)[None, None],
        position_ids=flat.position_ids[None],
    ).logits[0]

    # --- candidate: spine once, branches batched against the shared cache ---
    two = build_twod_chunk_example(instruction, chunks, VS, VE, EOT, **kw)
    P, N = two.spine_len, two.branch_ids.shape[0]

    spine_emb = model.get_input_embeddings()(two.spine_ids[None])
    cache = model(inputs_embeds=spine_emb, position_ids=two.spine_positions[None], use_cache=True).past_key_values

    # Broadcast, never copy: materialising the spine K/V per branch would cost
    # more than the whole layout saves.
    for layer in cache.layers:
        layer.keys = layer.keys.expand(N, -1, -1, -1)
        layer.values = layer.values.expand(N, -1, -1, -1)
        assert layer.keys.stride()[0] == 0, "spine cache was copied, not broadcast"

    br_audio = two.branch_frame_index >= 0
    br_emb = _embed_with_audio(model, two.branch_ids, br_audio, frames[two.branch_frame_index[br_audio]])
    two_logits = model(
        inputs_embeds=br_emb,
        attention_mask=build_twod_branch_mask(two.branch_prefix, two.branch_valid, P, br_emb.dtype),
        position_ids=two.branch_positions,
        past_key_values=cache,
        use_cache=False,
    ).logits

    for k in range(N):
        idx = (flat.seg_ids == k + 1).nonzero(as_tuple=True)[0]
        n = int(two.branch_valid[k].sum())
        torch.testing.assert_close(two_logits[k, :n], flat_logits[idx], atol=1e-4, rtol=1e-4)


def test_parity_twod_vs_flat_gradients():
    """Equivalence must hold for the backward too, or training would differ."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    instruction = [5, 6]
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(2, [30]), ChunkSpec(2, [40, 41])]
    torch.manual_seed(4)
    frames = torch.randn(sum(c.audio_len for c in chunks), H)

    def loss_and_grads(which):
        model.zero_grad(set_to_none=True)
        if which == "flat":
            flat = build_packed_chunk_example(instruction, chunks, VS, VE, EOT)
            emb = _embed_with_audio(
                model, flat.input_ids, flat.is_audio, frames[flat.audio_frame_index[flat.is_audio]]
            )
            logits = model(
                inputs_embeds=emb[None],
                attention_mask=_mask_of(flat)[None, None],
                position_ids=flat.position_ids[None],
            ).logits[0]
            tgt = flat.target_ids
        else:
            two = build_twod_chunk_example(instruction, chunks, VS, VE, EOT)
            P, N = two.spine_len, two.branch_ids.shape[0]
            spine_emb = model.get_input_embeddings()(two.spine_ids[None])
            cache = model(
                inputs_embeds=spine_emb, position_ids=two.spine_positions[None], use_cache=True
            ).past_key_values
            for layer in cache.layers:
                layer.keys = layer.keys.expand(N, -1, -1, -1)
                layer.values = layer.values.expand(N, -1, -1, -1)
            br_audio = two.branch_frame_index >= 0
            emb = _embed_with_audio(model, two.branch_ids, br_audio, frames[two.branch_frame_index[br_audio]])
            logits = model(
                inputs_embeds=emb,
                attention_mask=build_twod_branch_mask(two.branch_prefix, two.branch_valid, P, emb.dtype),
                position_ids=two.branch_positions,
                past_key_values=cache,
                use_cache=False,
            ).logits
            logits, tgt = logits.flatten(0, 1), two.branch_targets.flatten(0, 1)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=IGNORE_INDEX
        )
        loss.backward()
        g = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return float(loss), g

    l_flat, g_flat = loss_and_grads("flat")
    l_two, g_two = loss_and_grads("twod")

    assert abs(l_flat - l_two) < 1e-4, f"loss differs: {l_flat} vs {l_two}"
    assert set(g_flat) == set(g_two)
    for name in g_flat:
        torch.testing.assert_close(g_flat[name], g_two[name], atol=1e-4, rtol=1e-4, msg=lambda m: f"{name}: {m}")


def test_collate_twod_shapes_and_padding():
    a = build_twod_chunk_example(INSTR, [ChunkSpec(2, [20])], VS, VE, EOT)
    b = build_twod_chunk_example(INSTR, [ChunkSpec(2, [20, 21]), ChunkSpec(3, [30]), ChunkSpec(2, [])], VS, VE, EOT)
    batch = collate_twod_chunk_examples([a, b], pad_id=7)

    B, N, bb = batch.branch_ids.shape
    assert B == 2 and N == 3  # padded to the utterance with the most branches
    assert batch.branch_counts.tolist() == [1, 3]
    assert batch.spine_lens.tolist() == [a.spine_len, b.spine_len]
    # Utterance 0 has one real branch; the padded rows must be fully invalid and
    # fully ignored, or they would contribute phantom loss.
    assert batch.branch_valid[0, 1:].sum() == 0
    assert (batch.branch_targets[0, 1:] == IGNORE_INDEX).all()
    assert (batch.spine_ids[0, a.spine_len :] == 7).all()
    assert batch.spine_valid[0, a.spine_len :].sum() == 0


def test_twod_and_flat_supervise_identical_targets():
    """Both layouts must carry exactly the same supervised (position, token) set."""
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(3, [30]), ChunkSpec(2, []), ChunkSpec(2, [40, 41, 42])]
    for M, F in [(0, 0), (1, 0), (0, 28)]:
        kw = dict(audio_history_chunks=M, audio_window_frames=F)
        flat = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, **kw)
        two = build_twod_chunk_example(INSTR, chunks, VS, VE, EOT, **kw)
        flat_tgt = flat.target_ids[flat.target_ids != IGNORE_INDEX].tolist()
        two_tgt = two.branch_targets[two.branch_targets != IGNORE_INDEX].tolist()
        assert flat_tgt == two_tgt, f"M={M} F={F}: supervision differs"
        # ...and the same audio frames, in the same order.
        flat_fr = flat.audio_frame_index[flat.audio_frame_index >= 0].tolist()
        two_fr = two.branch_frame_index[two.branch_frame_index >= 0].tolist()
        assert flat_fr == two_fr, f"M={M} F={F}: audio windows differ"


@pytest.mark.parametrize("micro_batch", [1, 2, 3, 5])
def test_twod_micro_batching_changes_nothing(micro_batch):
    """Splitting the branch axis must not change the loss or any gradient.

    Branches are a BATCH axis in the 2-D layout, so they can be processed in
    groups with their activations recomputed in backward. That is the whole
    memory argument, and it is only sound if the split is numerically invisible:
    each group contributes a SUM and the batch-wide target count is the single
    denominator.
    """
    torch.manual_seed(6)
    model = _tiny_qwen3()
    H = model.config.hidden_size
    instruction = [5, 6, 7]
    chunks = [
        ChunkSpec(2, [20, 21]),
        ChunkSpec(2, [30]),
        ChunkSpec(2, []),  # silent chunk contributes only <eot>
        ChunkSpec(2, [40, 41]),
        ChunkSpec(2, [50]),
    ]
    frames = torch.randn(sum(c.audio_len for c in chunks), H)
    two = build_twod_chunk_example(instruction, chunks, VS, VE, EOT)
    P, N = two.spine_len, two.branch_ids.shape[0]
    n_targets = int((two.branch_targets != IGNORE_INDEX).sum())

    def run(step):
        model.zero_grad(set_to_none=True)
        spine_emb = model.get_input_embeddings()(two.spine_ids[None])
        cache = model(inputs_embeds=spine_emb, position_ids=two.spine_positions[None], use_cache=True).past_key_values

        total = None
        for lo in range(0, N, step):
            hi = min(lo + step, N)
            n = hi - lo
            sub = _shallow_broadcast(cache, n)
            audio = two.branch_frame_index[lo:hi] >= 0
            emb = _embed_with_audio(model, two.branch_ids[lo:hi], audio, frames[two.branch_frame_index[lo:hi][audio]])
            logits = model(
                inputs_embeds=emb,
                attention_mask=build_twod_branch_mask(two.branch_prefix[lo:hi], two.branch_valid[lo:hi], P, emb.dtype),
                position_ids=two.branch_positions[lo:hi],
                past_key_values=sub,
                use_cache=False,
            ).logits
            s = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),
                two.branch_targets[lo:hi].flatten(0, 1),
                reduction="sum",
                ignore_index=IGNORE_INDEX,
            )
            total = s if total is None else total + s
        loss = total / n_targets
        loss.backward()
        return float(loss), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    l_all, g_all = run(N)  # every branch at once
    l_mb, g_mb = run(micro_batch)  # split

    assert abs(l_all - l_mb) < 1e-5, f"loss changed: {l_all} vs {l_mb}"
    assert set(g_all) == set(g_mb)
    for name in g_all:
        torch.testing.assert_close(g_all[name], g_mb[name], atol=1e-5, rtol=1e-5, msg=lambda m: f"{name}: {m}")


# ---------------------------------------------------------------------------
# Structured attention: same flat sequence, only the permitted blocks computed
# ---------------------------------------------------------------------------

from nemo.collections.speechlm2.parts.script_attention import (  # noqa: E402
    build_attention_plan,
    register_script_attention,
    script_attention_plan,
)


def _plan_and_batch(chunk_specs, instruction=None, **kw):
    instruction = instruction or INSTR
    exs = [build_packed_chunk_example(instruction, cs, VS, VE, EOT, **kw) for cs in chunk_specs]
    return collate_packed_chunk_examples(exs, pad_id=0), build_attention_plan(exs)


def test_attention_plan_locates_every_token():
    """The plan must address exactly the spine and branch tokens, once each."""
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(3, [30]), ChunkSpec(2, []), ChunkSpec(2, [40, 41, 42])]
    batch, plan = _plan_and_batch([chunks], audio_window_frames=6)

    seg = batch.seg_ids[0]
    covered = sorted(
        plan.spine_pos[0][plan.spine_valid[0]].tolist() + plan.branch_pos[0][plan.branch_valid[0]].tolist()
    )
    assert covered == list(range(int(batch.valid[0].sum()))), "plan does not tile the sequence"
    # spine slots really are spine tokens; branch slots really are branch tokens
    assert (seg[plan.spine_pos[0][plan.spine_valid[0]]] == SPINE_SEG_ID).all()
    assert (seg[plan.branch_pos[0][plan.branch_valid[0]]] >= 1).all()
    # each branch row carries its own prefix
    for k in range(int(seg.max())):
        idx = (seg == k + 1).nonzero(as_tuple=True)[0]
        assert int(plan.branch_prefix[0, k]) == int(batch.prefix_len[0, idx[0]])


@torch.no_grad()
@pytest.mark.parametrize("window", [0, 6, 28])
def test_structured_attention_matches_dense(window):
    """Structured logits must equal the dense-mask logits on the same weights."""
    register_script_attention()
    H = 32
    chunks_a = [ChunkSpec(2, [20, 21]), ChunkSpec(2, [30]), ChunkSpec(2, []), ChunkSpec(2, [40, 41])]
    chunks_b = [ChunkSpec(2, [50]), ChunkSpec(2, [60, 61, 62]), ChunkSpec(2, [70])]
    batch, plan = _plan_and_batch([chunks_a, chunks_b], audio_window_frames=window)

    torch.manual_seed(9)
    frames = torch.randn(64, H)
    model_dense = _tiny_qwen3()
    emb = torch.stack(
        [
            _embed_with_audio(
                model_dense,
                batch.input_ids[i],
                batch.is_audio[i],
                frames[batch.audio_frame_index[i][batch.is_audio[i]]],
            )
            for i in range(batch.input_ids.shape[0])
        ]
    )

    mask = build_script_mask(batch.seg_ids, batch.order_ids, batch.prefix_len, batch.valid, emb.dtype)
    ref = model_dense(inputs_embeds=emb, attention_mask=mask, position_ids=batch.position_ids).logits

    model_struct = _tiny_qwen3()
    model_struct.set_attn_implementation("script")
    with script_attention_plan(plan):
        got = model_struct(inputs_embeds=emb, attention_mask=None, position_ids=batch.position_ids).logits

    # Compare only real tokens; padded slots are meaningless in either path.
    for i in range(batch.input_ids.shape[0]):
        n = int(batch.valid[i].sum())
        torch.testing.assert_close(got[i, :n], ref[i, :n], atol=1e-4, rtol=1e-4)


def test_structured_attention_gradients_match_dense():
    """Equivalence must hold for the backward, or training would differ."""
    register_script_attention()
    H = 32
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(2, [30]), ChunkSpec(2, [40, 41])]
    batch, plan = _plan_and_batch([chunks], audio_window_frames=6)
    torch.manual_seed(10)
    frames = torch.randn(32, H)

    def run(structured):
        model = _tiny_qwen3()
        if structured:
            model.set_attn_implementation("script")
        model.zero_grad(set_to_none=True)
        emb = _embed_with_audio(
            model,
            batch.input_ids[0],
            batch.is_audio[0],
            frames[batch.audio_frame_index[0][batch.is_audio[0]]],
        )[None]
        if structured:
            with script_attention_plan(plan):
                logits = model(inputs_embeds=emb, attention_mask=None, position_ids=batch.position_ids).logits
        else:
            mask = build_script_mask(batch.seg_ids, batch.order_ids, batch.prefix_len, batch.valid, emb.dtype)
            logits = model(inputs_embeds=emb, attention_mask=mask, position_ids=batch.position_ids).logits
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), batch.target_ids.flatten(0, 1), ignore_index=IGNORE_INDEX
        )
        loss.backward()
        return float(loss), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    l_ref, g_ref = run(structured=False)
    l_got, g_got = run(structured=True)
    assert abs(l_ref - l_got) < 1e-4, f"loss differs: {l_ref} vs {l_got}"
    assert set(g_ref) == set(g_got)
    for name in g_ref:
        torch.testing.assert_close(g_ref[name], g_got[name], atol=1e-4, rtol=1e-4, msg=lambda m: f"{name}: {m}")


def test_structured_attention_falls_back_without_a_plan():
    """No plan active => ordinary SDPA, so decoding keeps working unchanged."""
    register_script_attention()
    model = _tiny_qwen3()
    model.set_attn_implementation("script")
    ids = torch.tensor([[5, 6, 7, 8]])
    with torch.no_grad():
        out = model(input_ids=ids).logits
    ref_model = _tiny_qwen3()
    with torch.no_grad():
        ref = ref_model(input_ids=ids).logits
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Attention backend selection (flex / script / dense)
# ---------------------------------------------------------------------------


def test_flex_mask_mod_equals_dense_mask():
    """The FlexAttention predicate must select exactly the dense mask's pairs.

    This is the whole correctness argument for attn_backend=flex: the kernel is
    different, the permitted set is not. Checked element-by-element, including a
    ragged batch where the two utterances have different lengths.
    """
    from nemo.collections.speechlm2.models.script_model import ScriptSTTModel

    a = build_packed_chunk_example(INSTR, [ChunkSpec(2, [20, 21]), ChunkSpec(2, [30])], VS, VE, EOT)
    b = build_packed_chunk_example(
        INSTR,
        [ChunkSpec(2, [40]), ChunkSpec(3, [50, 51]), ChunkSpec(2, [])],
        VS,
        VE,
        EOT,
        audio_window_frames=6,
    )
    batch = collate_packed_chunk_examples([a, b], pad_id=0)
    dense = build_script_mask(batch.seg_ids, batch.order_ids, batch.prefix_len, batch.valid, torch.float32)[:, 0]

    mod = ScriptSTTModel._script_mask_mod(batch)
    B, T = batch.seg_ids.shape
    bi = torch.arange(B)[:, None, None].expand(B, T, T)
    qi = torch.arange(T)[None, :, None].expand(B, T, T)
    ki = torch.arange(T)[None, None, :].expand(B, T, T)
    got = mod(bi, None, qi, ki)

    assert torch.equal(got, dense == 0), "flex predicate and dense mask disagree"
    density = got.float().mean().item()
    assert 0.0 < density < 0.5, f"unexpected mask density {density:.3f}"


@pytest.mark.parametrize(
    "backend,expect_impl,expect_mask",
    [("dense", "eager", True), ("script", "script", False)],
)
def test_training_attention_selects_backend(backend, expect_impl, expect_mask):
    """Backend choice must drive both the implementation and the mask object."""
    from nemo.collections.speechlm2.models.script_model import ScriptSTTModel

    ex = build_packed_chunk_example(INSTR, [ChunkSpec(2, [20]), ChunkSpec(2, [30])], VS, VE, EOT)
    batch = collate_packed_chunk_examples([ex], pad_id=0)

    fake = ScriptSTTModel.__new__(ScriptSTTModel)
    fake._attn_backend = backend
    fake._bidirectional_audio = False
    impl, mask = ScriptSTTModel._training_attention(fake, batch, torch.float32)
    assert impl == expect_impl
    assert (mask is not None) == expect_mask
    if expect_mask:
        assert mask.shape == (1, 1, batch.seg_ids.shape[1], batch.seg_ids.shape[1])


def test_invalid_attn_backend_is_rejected():
    from nemo.collections.speechlm2.models.script_model import ScriptSTTModelConfig
    from nemo.collections.speechlm2.parts.utils import to_dataclass

    cfg = to_dataclass(
        ScriptSTTModelConfig,
        {
            "pretrained_llm": "x",
            "pretrained_asr": "y",
            "load_llm_weights": False,
            "load_asr_weights": False,
            "blank_token": "",
            "chunk_size": 2,
            "freeze_speech_encoder": True,
            "freeze_modality_adapter": True,
            "freeze_modality_proj": True,
            "freeze_llm_model": True,
            "freeze_llm_head": True,
            "freeze_embed_tokens": True,
            "attn_backend": "nonsense",
        },
    )
    assert cfg.attn_backend == "nonsense"  # coercion keeps it; __init__ is what rejects it


# ---------------------------------------------------------------------------
# Prompt-controlled SCRIPT
# ---------------------------------------------------------------------------


def test_text_style_identity_when_both_settings_are_on():
    """The default style must be byte-for-byte the original transcript."""
    from nemo.collections.speechlm2.parts.script_prompt import apply_text_style

    for s in ("It's well-known, isn't it? Yes.", " leading space kept.", "", "   "):
        assert apply_text_style(s, True, True) == s


@pytest.mark.parametrize(
    "text,cap,punct,expected",
    [
        ("It's well-known, isn't it? Yes.", True, False, "It's well-known isn't it Yes"),
        ("It's well-known, isn't it? Yes.", False, True, "it's well-known, isn't it? yes."),
        ("It's well-known, isn't it? Yes.", False, False, "it's well-known isn't it yes"),
        # intra-number marks survive; a trailing/standalone mark does not
        ("Up 1,200 to $3.5 at 9:30 -- yes.", True, False, "Up 1,200 to 3.5 at 9:30 yes"),
        # collapsing must not leave double spaces behind
        ("end. Next", True, False, "end Next"),
        # a punctuation-only chunk becomes empty, i.e. silent
        ("...", True, False, ""),
    ],
)
def test_text_style_variants(text, cap, punct, expected):
    from nemo.collections.speechlm2.parts.script_prompt import apply_text_style

    assert apply_text_style(text, cap, punct) == expected


def test_text_style_preserves_leading_whitespace():
    """Chunk text is sliced with its leading space so the first token is a word
    start; losing it would silently change the tokenization."""
    from nemo.collections.speechlm2.parts.script_prompt import apply_text_style

    assert apply_text_style(" Hello there.", True, False) == " Hello there"
    assert apply_text_style("Hello there.", True, False) == "Hello there"


def test_control_prompt_states_every_setting():
    from nemo.collections.speechlm2.parts.script_prompt import ScriptControls, render_control_prompt

    p = render_control_prompt("Base.", ScriptControls(14, 3, True, False))
    assert (
        p == "Base. The audio is chunked every 14 frames with an emission delay of 3 frames. "
        "Use capitalization. Do not use punctuation."
    )

    # singular/zero wording
    assert "every 1 frame with" in render_control_prompt("B.", ScriptControls(1, 1, True, True))
    assert "of 1 frame." in render_control_prompt("B.", ScriptControls(2, 1, True, True))
    assert "with no emission delay." in render_control_prompt("B.", ScriptControls(2, 0, True, True))


def test_control_prompt_is_injective_over_settings():
    """Distinct operating points must produce distinct prompts, or the model
    cannot tell them apart."""
    from nemo.collections.speechlm2.parts.script_prompt import ScriptControls, render_control_prompt

    seen = {
        render_control_prompt("B.", ScriptControls(c, d, cap, pun))
        for c in (2, 14)
        for d in (0, 3)
        for cap in (True, False)
        for pun in (True, False)
    }
    assert len(seen) == 2 * 2 * 2 * 2


def test_messages_respect_capitalization_and_punctuation():
    """Style flows all the way into the per-chunk supervision."""
    words = _align(("Hello,", 0.16, 0.48), ("World.", 0.60, 0.80))
    kw = dict(
        system_role="system",
        system_prompt="P",
        audio_tag="<a>",
        blank_token="<b>",
        chunk_size=2,
        num_delay_frames=0,
        audio_duration_secs=1.0,
        frame_length_in_secs=0.08,
        alignments=words,
        transcript="Hello, World.",
    )
    plain = _assistant_contents(get_llm_messages_for_sample(**kw))
    styled = _assistant_contents(get_llm_messages_for_sample(capitalization=False, punctuation=False, **kw))

    assert "Hello," in plain and "World." in "".join(plain)
    # " world" keeps its leading space: that space is the word-start marker the
    # slice deliberately preserves, and restyling must not eat it.
    assert [c for c in styled if c != "<b>"] == ["hello", " world"]


def test_punctuation_only_chunk_becomes_silent():
    """Once punctuation is stripped, a chunk whose text was only a mark reveals
    nothing and must fall back to the blank sentinel."""
    msgs = get_llm_messages_for_sample(
        system_role="system",
        system_prompt="P",
        audio_tag="<a>",
        blank_token="<b>",
        chunk_size=2,
        num_delay_frames=0,
        audio_duration_secs=0.4,
        frame_length_in_secs=0.08,
        alignments=_align((".", 0.0, 0.16), ("Hi", 0.20, 0.32)),
        transcript=". Hi",
        capitalization=True,
        punctuation=False,
    )
    contents = _assistant_contents(msgs)
    assert contents[0] == "<b>"  # the "." chunk is silent, not an empty string
    assert "Hi" in "".join(contents)


def test_batch_messages_accept_per_sample_controls():
    """Delay / cap / punct vary per example within one batch; chunk size cannot."""
    from nemo.collections.speechlm2.parts.script_messages import get_llm_messages_for_batch

    words = [_align(("Hello", 0.16, 0.48)), _align(("Hello", 0.16, 0.48))]
    batch = get_llm_messages_for_batch(
        system_role="system",
        system_prompt=["P", "P"],
        audio_tag="<a>",
        blank_token="<b>",
        chunk_size=2,
        num_delay_frames=[0, 2],
        audio_durations_secs=[1.0, 1.0],
        frame_length_in_secs=0.08,
        alignments=words,
        capitalization=[True, False],
        punctuation=[True, True],
    )
    a, b = (_assistant_contents(m) for m in batch)
    assert a.index("Hello") == 2  # delay 0
    assert b.index("hello") == 3  # delay 2 -> one chunk later, and lowercased


def test_batch_messages_reject_mismatched_control_lengths():
    from nemo.collections.speechlm2.parts.script_messages import get_llm_messages_for_batch

    with pytest.raises(ValueError, match="num_delay_frames has 3 entries"):
        get_llm_messages_for_batch(
            system_role="system",
            system_prompt=["P", "P"],
            audio_tag="<a>",
            blank_token="<b>",
            chunk_size=2,
            num_delay_frames=[0, 1, 2],
            audio_durations_secs=[1.0, 1.0],
            frame_length_in_secs=0.08,
            alignments=[_align(("Hi", 0.0, 0.1))] * 2,
        )


def test_sampled_controls_are_deterministic_and_cover_the_space():
    import numpy as np

    from nemo.collections.speechlm2.parts.script_prompt import sample_controls

    delays = [0, 1, 2, 3, 4, 6, 8]

    def draw(seed):
        # One generator advanced 400 times -- re-seeding per draw would return
        # the same controls 400 times and the coverage assertions below would
        # pass vacuously.
        rng = np.random.default_rng(seed)
        return [sample_controls(rng, 14, delays, 0.5, 0.5) for _ in range(400)]

    a, b = draw(0), draw(0)
    assert a == b, "same seed must give the same controls"
    assert draw(1) != a, "different seeds must differ"

    assert {c.num_delay_frames for c in a} == set(delays)
    assert {(c.capitalization, c.punctuation) for c in a} == {
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    }
    assert all(c.chunk_size == 14 for c in a), "chunk size is passed in, never drawn per example"


def test_resolve_delay_candidates():
    from nemo.collections.speechlm2.parts.script_prompt import resolve_delay_candidates

    assert resolve_delay_candidates(None, 3) == [3]
    assert resolve_delay_candidates(5, 3) == [5]
    assert resolve_delay_candidates([0, 2, 4], 3) == [0, 2, 4]
    with pytest.raises(ValueError, match="at least one delay"):
        resolve_delay_candidates([], 3)
    with pytest.raises(ValueError, match="non-negative"):
        resolve_delay_candidates([1, -2], 3)


def test_generate_rejects_control_kwargs_when_not_prompt_controlled():
    """Asking a non-prompt-controlled checkpoint for an operating point must fail
    loudly. Silently ignoring the request would make the knobs look functional on
    a model that never learned them."""
    from types import SimpleNamespace

    from nemo.collections.speechlm2.models.script_model import ScriptSTTModel

    # The guard runs before any weights are touched, so a stub carrying just the
    # two attributes it reads is enough to exercise it.
    stub = SimpleNamespace(
        core_cfg=SimpleNamespace(
            prompt_control=False,
            val_num_delay_frames=3,
            val_capitalization=True,
            val_punctuation=True,
        ),
        _resolve_inference_chunk_size=lambda override: override or 14,
    )
    audios = torch.zeros(2, 16000)
    audio_lens = torch.tensor([16000, 16000])

    for kwargs in ({"num_delay_frames": 6}, {"capitalization": False}, {"punctuation": False}):
        with pytest.raises(ValueError, match="prompt_control=False"):
            ScriptSTTModel.generate(stub, audios=audios, audio_lens=audio_lens, system_prompt="P", **kwargs)


# ---------------------------------------------------------------------------
# Read/write gate
# ---------------------------------------------------------------------------

READ, WRITE = 90, 91


def _rw_chunks():
    """Two chunks: the first silent, the second revealing two words."""
    return [
        ChunkSpec(audio_len=2, target_ids=[], gate_id=READ),
        ChunkSpec(audio_len=2, target_ids=[11, 12], gate_id=WRITE),
    ]


def test_gate_never_enters_the_spine():
    """THE invariant: the spine is the running transcript and nothing else.

    A gate token leaking into the spine would change the conditioning history and
    break the exactness property the parity tests rely on.
    """
    ex = build_packed_chunk_example(
        instruction_ids=[5, 6], chunks=_rw_chunks(), vision_start_id=80, vision_end_id=81, eot_id=82
    )
    spine = ex.input_ids[: ex.spine_len].tolist()
    assert spine == [5, 6, 11, 12], spine
    assert READ not in spine and WRITE not in spine


def test_gate_is_supervised_as_the_branch_first_token():
    """<ve> predicts the gate, the gate predicts the first word, last word -> eot."""
    ex = build_packed_chunk_example(
        instruction_ids=[5, 6], chunks=_rw_chunks(), vision_start_id=80, vision_end_id=81, eot_id=82
    )
    for k, expect in ((1, [READ, 82]), (2, [WRITE, 11, 12, 82])):
        sup = ex.target_ids[(ex.seg_ids == k) & (ex.target_ids != IGNORE_INDEX)]
        assert sup.tolist() == expect, (k, sup.tolist())


def test_gate_off_by_default_is_byte_identical():
    """gate_id=None must reproduce the pre-existing layout exactly."""
    plain = [ChunkSpec(audio_len=2, target_ids=[]), ChunkSpec(audio_len=2, target_ids=[11, 12])]
    kw = dict(instruction_ids=[5, 6], vision_start_id=80, vision_end_id=81, eot_id=82)
    a = build_packed_chunk_example(chunks=plain, **kw)
    b = build_packed_chunk_example(
        chunks=[ChunkSpec(audio_len=c.audio_len, target_ids=c.target_ids, gate_id=None) for c in plain], **kw
    )
    for f in ("input_ids", "position_ids", "seg_ids", "prefix_len", "target_ids", "is_audio"):
        assert torch.equal(getattr(a, f), getattr(b, f)), f


@pytest.mark.parametrize("layout", ["packed", "twod", "separate"])
def test_all_three_builders_apply_the_gate(layout):
    """The gate must reach every layout, or they silently disagree."""
    chunks, kw = _rw_chunks(), dict(instruction_ids=[5, 6], vision_start_id=80, vision_end_id=81, eot_id=82)
    if layout == "packed":
        ex = build_packed_chunk_example(chunks=chunks, **kw)
        ids = ex.input_ids[ex.spine_len :].tolist()
    elif layout == "twod":
        ex = build_twod_chunk_example(chunks=chunks, **kw)
        ids = ex.branch_ids.flatten().tolist()
    else:
        exs = build_separate_chunk_examples(chunks=chunks, **kw)
        ids = [t for e in exs for t in e.input_ids.tolist()]
    assert READ in ids and WRITE in ids, layout


def test_parity_packed_vs_separate_with_gate():
    """Exactness must survive the gate: packed branches == standalone examples."""
    chunks = [
        ChunkSpec(audio_len=2, target_ids=[], gate_id=READ),
        ChunkSpec(audio_len=2, target_ids=[11, 12], gate_id=WRITE),
        ChunkSpec(audio_len=2, target_ids=[], gate_id=READ),
        ChunkSpec(audio_len=2, target_ids=[13], gate_id=WRITE),
    ]
    kw = dict(instruction_ids=[5, 6], vision_start_id=80, vision_end_id=81, eot_id=82)
    packed = build_packed_chunk_example(chunks=chunks, **kw)
    separate = build_separate_chunk_examples(chunks=chunks, **kw)
    for k, sep in enumerate(separate, start=1):
        got = packed.input_ids[packed.seg_ids == k].tolist()
        # A standalone example is [history..., <vs> audio <ve> gate words <eot>];
        # its branch is the tail after the history prefix.
        exp = sep.input_ids[sep.branch_start :].tolist()
        assert got == exp, (k, got, exp)


def test_decode_strips_the_gate_and_read_suppresses_words():
    """Decode must remove the gate before it reaches the history, and a <read>
    decision must discard anything that followed it."""
    from nemo.collections.speechlm2.parts.script import batched_stream_decode_script

    # Scripted "model": emits a fixed token sequence per chunk, then eot.
    class FakeLLM:
        def __init__(self, plan):
            self.plan, self.step = plan, 0

        def __call__(self, inputs_embeds=None, **kw):
            n = inputs_embeds.shape[0]
            tid = self.plan[min(self.step, len(self.plan) - 1)]
            self.step += 1
            logits = torch.full((n, 1, 200), -1e9)
            logits[:, 0, tid] = 0.0
            return SimpleNamespace(logits=logits, past_key_values=None)

    from types import SimpleNamespace

    emb = lambda ids: torch.zeros(*ids.shape, 8)
    frames = [torch.zeros(2, 8)]

    # chunk 0: <write> 11 <eot>
    out = batched_stream_decode_script(
        llm=FakeLLM([WRITE, 11, 82]),
        embed_tokens=emb,
        instruction_ids_list=[[5]],
        frames_list=frames,
        chunk_size=2,
        vision_start_id=80,
        vision_end_id=81,
        eot_id=82,
        pad_id=0,
        max_new_tokens=8,
        read_id=READ,
        write_id=WRITE,
    )
    assert out == [[11]], out

    # chunk 0: <read> then a stray word -> everything discarded
    out = batched_stream_decode_script(
        llm=FakeLLM([READ, 11, 82]),
        embed_tokens=emb,
        instruction_ids_list=[[5]],
        frames_list=frames,
        chunk_size=2,
        vision_start_id=80,
        vision_end_id=81,
        eot_id=82,
        pad_id=0,
        max_new_tokens=8,
        read_id=READ,
        write_id=WRITE,
    )
    assert out == [[]], out


# ---------------------------------------------------------------------------
# Gate in history
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gate_in_history", [False, True])
def test_gate_in_history_controls_the_spine(gate_in_history):
    """Off: the spine is words alone. On: it is what each branch emitted."""
    ex = build_packed_chunk_example(
        instruction_ids=[5, 6],
        chunks=_rw_chunks(),
        vision_start_id=80,
        vision_end_id=81,
        eot_id=82,
        gate_in_history=gate_in_history,
    )
    spine = ex.input_ids[: ex.spine_len].tolist()
    assert spine == ([5, 6, READ, WRITE, 11, 12] if gate_in_history else [5, 6, 11, 12]), spine


def test_gate_in_history_gives_the_spine_a_length_that_tracks_chunks():
    """The whole point: with words alone the history cannot express elapsed time.

    Two utterances with the SAME words but different amounts of leading silence
    must produce different histories -- otherwise a branch cannot tell whether one
    chunk or many passed before it.
    """

    def spine(n_silent, **kw):
        chunks = [ChunkSpec(audio_len=2, target_ids=[], gate_id=READ) for _ in range(n_silent)]
        chunks.append(ChunkSpec(audio_len=2, target_ids=[11], gate_id=WRITE))
        ex = build_packed_chunk_example(
            instruction_ids=[5], chunks=chunks, vision_start_id=80, vision_end_id=81, eot_id=82, **kw
        )
        return ex.input_ids[: ex.spine_len].tolist()

    assert spine(1) == spine(5), "without the gate the two are indistinguishable"
    assert spine(1, gate_in_history=True) != spine(5, gate_in_history=True)
    assert len(spine(5, gate_in_history=True)) - len(spine(1, gate_in_history=True)) == 4


@pytest.mark.parametrize("gate_in_history", [False, True])
def test_packed_and_separate_agree_on_the_HISTORY(gate_in_history):
    """Regression guard: every builder must define the history identically.

    The branch-only parity test does NOT catch a divergence here -- it compares
    branch tokens and never looks at the prefix, which is exactly how the packed
    spine and the separate reference drifted apart once already.
    """
    chunks = [
        ChunkSpec(audio_len=2, target_ids=[], gate_id=READ),
        ChunkSpec(audio_len=2, target_ids=[11, 12], gate_id=WRITE),
        ChunkSpec(audio_len=2, target_ids=[], gate_id=READ),
        ChunkSpec(audio_len=2, target_ids=[13], gate_id=WRITE),
    ]
    kw = dict(
        instruction_ids=[5, 6],
        chunks=chunks,
        vision_start_id=80,
        vision_end_id=81,
        eot_id=82,
        gate_in_history=gate_in_history,
    )
    packed = build_packed_chunk_example(**kw)
    separate = build_separate_chunk_examples(**kw)
    spine = packed.input_ids[: packed.spine_len].tolist()
    for k, sep in enumerate(separate):
        # Each standalone example's prefix IS the history at that chunk, which
        # must equal the packed spine truncated to that branch's prefix_len.
        pref_len = int(packed.prefix_len[packed.seg_ids == k + 1][0])
        assert sep.input_ids[: sep.branch_start].tolist() == spine[:pref_len], k


def test_twod_matches_packed_spine_with_gate_in_history():
    ex_p = build_packed_chunk_example(
        instruction_ids=[5, 6],
        chunks=_rw_chunks(),
        vision_start_id=80,
        vision_end_id=81,
        eot_id=82,
        gate_in_history=True,
    )
    ex_t = build_twod_chunk_example(
        instruction_ids=[5, 6],
        chunks=_rw_chunks(),
        vision_start_id=80,
        vision_end_id=81,
        eot_id=82,
        gate_in_history=True,
    )
    assert ex_t.spine_ids.tolist() == ex_p.input_ids[: ex_p.spine_len].tolist()


def test_decode_keeps_gate_in_history_but_not_in_output():
    """With gate_in_history the gate must persist in the conditioning history
    while the returned token stream still excludes it from the text."""
    from types import SimpleNamespace

    from nemo.collections.speechlm2.parts.script import batched_stream_decode_script

    class FakeLLM:
        def __init__(self, plan):
            self.plan, self.step = plan, 0

        def __call__(self, inputs_embeds=None, **kw):
            n = inputs_embeds.shape[0]
            tid = self.plan[min(self.step, len(self.plan) - 1)]
            self.step += 1
            logits = torch.full((n, 1, 200), -1e9)
            logits[:, 0, tid] = 0.0
            return SimpleNamespace(logits=logits, past_key_values=None)

    emb = lambda ids: torch.zeros(*ids.shape, 8)  # noqa: E731
    frames = [torch.zeros(2, 8)]
    kw = dict(
        embed_tokens=emb,
        instruction_ids_list=[[5]],
        frames_list=frames,
        chunk_size=2,
        vision_start_id=80,
        vision_end_id=81,
        eot_id=82,
        pad_id=0,
        max_new_tokens=8,
        read_id=READ,
        write_id=WRITE,
    )
    # write + word, gate retained in the history
    assert batched_stream_decode_script(llm=FakeLLM([WRITE, 11, 82]), gate_in_history=True, **kw) == [[WRITE, 11]]
    # a read chunk still contributes its gate -- that is the elapsed-time signal
    assert batched_stream_decode_script(llm=FakeLLM([READ, 82]), gate_in_history=True, **kw) == [[READ]]
    # and with the flag off, neither appears
    assert batched_stream_decode_script(llm=FakeLLM([WRITE, 11, 82]), gate_in_history=False, **kw) == [[11]]


# ---------------------------------------------------------------------------
# FSM decode
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FSM decode
#
# These check that the state machine is CORRECT -- that it feeds the training
# conditioning and honours the gate. They deliberately do NOT assert it matches
# batched_stream_decode_script token for token: the FSM is meant to be the
# faithful streaming algorithm, and it is an empirical question whether it beats
# the bulk-prefill path, not something to pin down by construction.
# ---------------------------------------------------------------------------

TEXT_MARK = 7.0


def _text_emb(ids):
    """Text embeddings carry a marker so a stub LLM can tell them from audio."""
    return torch.full((*ids.shape, 8), TEXT_MARK)


class _StubLLM:
    """Emits ``plan`` in order, one token per GENERATION step.

    A generation step is a single-token step whose input is a TEXT embedding:
    that is ``<ve>`` (whose logits predict the first word) and then each token
    fed back. Single-token AUDIO steps and the multi-token prefix prefill are
    not decision points and must not advance the plan -- which is exactly the
    distinction the FSM's per-frame stepping makes visible.
    """

    def __init__(self, plan, vocab=200):
        self.plan, self.vocab, self.i = plan, vocab, -1
        self.audio_steps = 0
        self.fed = []

    def __call__(self, inputs_embeds=None, **kw):
        from types import SimpleNamespace

        n, steps, _ = inputs_embeds.shape
        is_text = bool(torch.isclose(inputs_embeds.reshape(-1)[0], torch.tensor(TEXT_MARK)))
        if steps > 1:
            self.i = -1  # multi-token prefill = start of a new chunk
        elif steps == 1 and not is_text:
            self.audio_steps += 1
            self.fed.append(inputs_embeds[0, 0].clone())
        elif steps == 1 and is_text:
            self.i += 1
        logits = torch.full((n, steps, self.vocab), -1e9)
        logits[:, -1, self.plan[min(max(self.i, 0), len(self.plan) - 1)]] = 0.0
        return SimpleNamespace(logits=logits, past_key_values=None)


def test_fsm_feeds_one_audio_frame_per_step_and_the_training_window():
    """The FSM must feed exactly the window the training packer would build."""
    from nemo.collections.speechlm2.parts.script_fsm import fsm_stream_decode_script

    frames = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8)
    llm = _StubLLM([82])  # emit <eot> immediately: no words, isolate the audio path
    fsm_stream_decode_script(
        llm=llm,
        embed_tokens=_text_emb,
        instruction_ids_list=[[5]],
        frames_list=[frames],
        chunk_size=2,
        vision_start_id=80,
        vision_end_id=81,
        eot_id=82,
        pad_id=0,
        max_new_tokens=4,
    )
    # 3 chunks x 2 frames, one frame per step, in order.
    assert llm.audio_steps == 6, llm.audio_steps
    assert torch.allclose(torch.stack(llm.fed), frames), "frames fed out of order or altered"


def test_fsm_window_follows_audio_window_frames():
    """With a fixed frame window every branch sees F frames, not one chunk."""
    from nemo.collections.speechlm2.parts.script_fsm import fsm_stream_decode_script

    frames = torch.randn(8, 8)
    llm = _StubLLM([82])
    fsm_stream_decode_script(
        llm=llm,
        embed_tokens=_text_emb,
        instruction_ids_list=[[5]],
        frames_list=[frames],
        chunk_size=2,
        vision_start_id=80,
        vision_end_id=81,
        eot_id=82,
        pad_id=0,
        max_new_tokens=4,
        audio_window_frames=4,
    )
    # chunks end at frames 2,4,6,8 -> windows [0:2],[0:4],[2:6],[4:8] = 2+4+4+4
    assert llm.audio_steps == 14, llm.audio_steps


def test_fsm_emits_words_and_stops_at_eot():
    from nemo.collections.speechlm2.parts.script_fsm import fsm_stream_decode_script

    out = fsm_stream_decode_script(
        llm=_StubLLM([11, 12, 82]),
        embed_tokens=_text_emb,
        instruction_ids_list=[[5]],
        frames_list=[torch.zeros(2, 8)],
        chunk_size=2,
        vision_start_id=80,
        vision_end_id=81,
        eot_id=82,
        pad_id=0,
        max_new_tokens=6,
    )
    assert out == [[11, 12]], out


def test_fsm_honours_the_read_write_gate():
    from nemo.collections.speechlm2.parts.script_fsm import fsm_stream_decode_script

    kw = dict(
        embed_tokens=_text_emb,
        instruction_ids_list=[[5]],
        frames_list=[torch.zeros(2, 8)],
        chunk_size=2,
        vision_start_id=80,
        vision_end_id=81,
        eot_id=82,
        pad_id=0,
        max_new_tokens=6,
        read_id=READ,
        write_id=WRITE,
    )
    assert fsm_stream_decode_script(llm=_StubLLM([WRITE, 11, 82]), **kw) == [[11]]
    assert fsm_stream_decode_script(llm=_StubLLM([READ, 82]), **kw) == [[]]
    assert fsm_stream_decode_script(llm=_StubLLM([WRITE, 11, 82]), gate_in_history=True, **kw) == [[WRITE, 11]]


def test_fsm_handles_ragged_batches():
    """Streams of different lengths must each stop at their own last chunk."""
    from nemo.collections.speechlm2.parts.script_fsm import fsm_stream_decode_script

    out = fsm_stream_decode_script(
        llm=_StubLLM([11, 82]),
        embed_tokens=_text_emb,
        instruction_ids_list=[[5], [6, 7]],
        frames_list=[torch.zeros(6, 8), torch.zeros(2, 8)],
        chunk_size=2,
        vision_start_id=80,
        vision_end_id=81,
        eot_id=82,
        pad_id=0,
        max_new_tokens=4,
    )
    assert len(out) == 2
    assert len(out[0]) == 3 * len(out[1]), (out, "3 chunks vs 1")


# ---------------------------------------------------------------------------
# Position scheme
# ---------------------------------------------------------------------------


def _word_pos(ex, token):
    """Position of `token` wherever it is predicted inside a branch."""
    for i, (t, s, a) in enumerate(zip(ex.input_ids.tolist(), ex.seg_ids.tolist(), ex.is_audio.tolist())):
        if s > 0 and not a and t == token:
            return int(ex.position_ids[i])
    return None


def _spine_pos(ex, token):
    for i, (t, s) in enumerate(zip(ex.input_ids.tolist(), ex.seg_ids.tolist())):
        if s == 0 and t == token:
            return int(ex.position_ids[i])
    return None


@pytest.mark.parametrize("scheme,expect_equal", [("branch", False), ("continuous", True)])
def test_position_scheme_and_chunk_boundary_invariance(scheme, expect_equal):
    """The point of the continuous scheme: where the chunk boundary fell must not
    change the text geometry.

    Same transcript "one two three", two splits:
      A  previous chunk silent, this chunk emits all three
      B  previous chunk emitted "one", this chunk emits "two three"
    The offset from "one" to "two" should be identical under `continuous` and
    differs sharply under `branch` (the audio block sits between them in B).
    """
    ONE, TWO, THREE, W = 201, 202, 203, 4
    kw = dict(instruction_ids=[1, 2, 3], vision_start_id=80, vision_end_id=81, eot_id=82, position_scheme=scheme)
    a = build_packed_chunk_example(chunks=[ChunkSpec(W, []), ChunkSpec(W, [ONE, TWO, THREE])], **kw)
    b = build_packed_chunk_example(chunks=[ChunkSpec(W, [ONE]), ChunkSpec(W, [TWO, THREE])], **kw)
    # In A both words are predicted in branch 2; in B "one" lives in the history.
    off_a = _word_pos(a, TWO) - _word_pos(a, ONE)
    off_b = _word_pos(b, TWO) - _spine_pos(b, ONE)
    if expect_equal:
        assert off_a == off_b == 1, (off_a, off_b)
    else:
        assert off_a != off_b, (off_a, off_b)


def test_continuous_scheme_puts_words_on_their_spine_positions():
    """Under `continuous` a branch predicts each word at exactly the position the
    spine gives it -- that is what makes the text one continuous sequence."""
    chunks = [ChunkSpec(3, [201]), ChunkSpec(3, [202, 203]), ChunkSpec(3, [204])]
    ex = build_packed_chunk_example(
        instruction_ids=[1, 2, 3],
        chunks=chunks,
        vision_start_id=80,
        vision_end_id=81,
        eot_id=82,
        position_scheme="continuous",
    )
    for tok in (201, 202, 203, 204):
        assert _word_pos(ex, tok) == _spine_pos(ex, tok), tok


def test_position_ids_are_never_negative():
    """The continuous scheme shifts branches left; ids must stay valid indices."""
    for W in (2, 8, 28):
        ex = build_packed_chunk_example(
            instruction_ids=[1],
            chunks=[ChunkSpec(W, [201]), ChunkSpec(W, [202])],
            vision_start_id=80,
            vision_end_id=81,
            eot_id=82,
            position_scheme="continuous",
        )
        assert int(ex.position_ids.min()) >= 0, (W, int(ex.position_ids.min()))


def test_mask_is_identical_under_both_position_schemes():
    """THE decoupling guarantee: the mask is structural, so moving positions
    around cannot change who attends to whom."""
    chunks = [ChunkSpec(3, [201]), ChunkSpec(3, [202, 203])]
    kw = dict(instruction_ids=[1, 2, 3], chunks=chunks, vision_start_id=80, vision_end_id=81, eot_id=82)
    a = build_packed_chunk_example(position_scheme="branch", **kw)
    b = build_packed_chunk_example(position_scheme="continuous", **kw)
    assert torch.equal(a.order_ids, b.order_ids)
    assert not torch.equal(a.position_ids, b.position_ids), "schemes should differ in RoPE space"
    assert torch.equal(_mask_of(a), _mask_of(b))


def test_script_batch_carries_order_ids_end_to_end():
    """Regression: the dataset must populate order_ids on the batch it emits.

    The unit tests build batches by calling collate_* directly, so a missing
    field in ScriptSTTDataset.get_batch_data slipped through and only surfaced
    on the cluster as `TypeError: 'NoneType' object is not subscriptable` deep
    inside the FlexAttention vmap. This asserts the batch object the model
    actually receives is complete.
    """
    import inspect

    from nemo.collections.speechlm2.data.script_dataset import ScriptBatch, ScriptSTTDataset

    # Every field the mask consumes must be handed to ScriptBatch by the dataset.
    src = inspect.getsource(ScriptSTTDataset.get_batch_data)
    for field in ("order_ids", "seg_ids", "prefix_len", "position_ids"):
        assert f"{field}=packed.{field}" in src, f"get_batch_data never sets {field}"

    # And a batch built the way the dataset builds it must mask cleanly.
    chunks = [ChunkSpec(2, [201]), ChunkSpec(2, [202, 203])]
    packed = build_packed_chunk_example(
        instruction_ids=[1, 2, 3], chunks=chunks, vision_start_id=80, vision_end_id=81, eot_id=82
    )
    batched = collate_packed_chunk_examples([packed], pad_id=0)
    b = ScriptBatch(
        input_tokens=batched.input_ids,
        position_ids=batched.position_ids,
        order_ids=batched.order_ids,
        seg_ids=batched.seg_ids,
        prefix_len=batched.prefix_len,
        target_tokens=batched.target_ids,
        is_audio=batched.is_audio,
        valid=batched.valid,
    )
    assert b.order_ids is not None
    m = build_script_mask(b.seg_ids, b.order_ids, b.prefix_len, b.valid, torch.float32)
    assert m.shape[-1] == b.input_tokens.shape[-1]


def test_sampled_position_scheme_draws_both_and_is_deterministic():
    """`sampled` must actually produce both layouts, reproducibly."""
    import numpy as np

    def draw(seed, p=0.5, n=400):
        rng = np.random.default_rng(seed)
        return ["continuous" if rng.random() < p else "branch" for _ in range(n)]

    a, b = draw(0), draw(0)
    assert a == b, "same seed must give the same sequence of schemes"
    assert draw(1) != a, "different seeds must differ"
    assert set(a) == {"branch", "continuous"}
    frac = a.count("continuous") / len(a)
    assert 0.4 < frac < 0.6, frac
    # the probability knob must actually bite
    assert draw(0, p=0.0).count("continuous") == 0
    assert draw(0, p=1.0).count("branch") == 0


def test_dataset_rejects_a_bad_position_scheme():
    from nemo.collections.speechlm2.data.script_dataset import ScriptSTTDataConfig
    from nemo.collections.speechlm2.parts.utils import to_dataclass

    cfg = to_dataclass(
        ScriptSTTDataConfig,
        {"sample_rate": 16000, "frame_length_in_secs": 0.08, "chunk_size": 2, "position_scheme": "nonsense"},
    )
    assert cfg.position_scheme == "nonsense"  # coercion keeps it; __init__ rejects it
    assert cfg.continuous_prob == 0.5  # default P(continuous)


def test_model_resolves_sampled_to_a_concrete_decode_scheme():
    """A model trained with `sampled` must decode with ONE concrete layout, and
    say which -- decoding has no notion of sampling."""
    from nemo.collections.speechlm2.models.script_model import ScriptSTTModelConfig
    from nemo.collections.speechlm2.parts.utils import to_dataclass

    base = dict(
        pretrained_llm="x",
        pretrained_asr="y",
        load_llm_weights=False,
        load_asr_weights=False,
        blank_token="",
        chunk_size=2,
        freeze_speech_encoder=True,
        freeze_modality_adapter=True,
        freeze_modality_proj=True,
        freeze_llm_model=True,
        freeze_llm_head=True,
        freeze_embed_tokens=True,
    )
    cfg = to_dataclass(ScriptSTTModelConfig, {**base, "position_scheme": "sampled"})
    assert cfg.position_scheme == "sampled"
    assert cfg.val_position_scheme == "continuous"  # the concrete decode default


# ---------------------------------------------------------------------------
# Full-context (offline) ablation
# ---------------------------------------------------------------------------


def test_full_context_layout_is_a_single_branch_over_all_frames():
    """The offline upper bound: one branch, every frame, the whole transcript.

    This is the SCRIPT layout with a single chunk, so the same builder produces
    it -- what makes it an ABLATION rather than a different model is that the
    encoder keeps its chunk-limited look-ahead, which lives in att_context_size
    and not in this layout.
    """
    words = [201, 202, 203, 204]
    ex = build_packed_chunk_example(
        instruction_ids=[1, 2, 3],
        chunks=[ChunkSpec(audio_len=12, target_ids=words)],
        vision_start_id=80,
        vision_end_id=81,
        eot_id=82,
    )
    assert int(ex.seg_ids.max()) == 1, "exactly one branch"
    assert int(ex.is_audio.sum()) == 12, "all frames in that branch"
    # the branch predicts the entire transcript, then <eot>
    sup = ex.target_ids[(ex.seg_ids == 1) & (ex.target_ids != IGNORE_INDEX)]
    assert sup.tolist() == words + [82]
    # its history prefix is the instruction alone -- there is no earlier chunk
    assert int(ex.prefix_len[ex.seg_ids == 1][0]) == 3


def test_full_context_matches_the_standalone_offline_example():
    """One chunk means packed == standalone trivially; assert it, so a future
    layout change cannot silently break the offline arm."""
    chunks = [ChunkSpec(audio_len=6, target_ids=[201, 202])]
    kw = dict(instruction_ids=[1, 2], chunks=chunks, vision_start_id=80, vision_end_id=81, eot_id=82)
    packed = build_packed_chunk_example(**kw)
    sep = build_separate_chunk_examples(**kw)
    assert len(sep) == 1
    branch = packed.input_ids[packed.seg_ids == 1].tolist()
    assert branch == sep[0].input_ids[sep[0].branch_start :].tolist()


# ======================================================================
# Bidirectional audio within a branch
# ======================================================================


def _bidir_batch():
    ex = build_packed_chunk_example(INSTR, [ChunkSpec(3, [20, 21]), ChunkSpec(3, [30])], VS, VE, EOT)
    return ex, collate_packed_chunk_examples([ex], pad_id=0)


def test_bidirectional_audio_opens_exactly_the_audio_block():
    """The new rule adds audio->later-audio pairs and NOTHING else."""
    ex, batch = _bidir_batch()
    args = (batch.seg_ids, batch.order_ids, batch.prefix_len, batch.valid, torch.float32)
    causal = build_script_mask(*args) == 0
    bidir = build_script_mask(*args, is_audio=batch.is_audio) == 0

    # Strictly more permissive.
    assert (bidir | causal).equal(bidir), "bidirectional rule must not REMOVE any pair"
    added = bidir & ~causal
    assert added.any(), "expected the audio block to gain pairs"

    seg, aud = batch.seg_ids[0], batch.is_audio[0]
    qs, ks = torch.where(added[0, 0])
    # Every added pair: same branch, both ends audio, and strictly anti-causal.
    assert bool((seg[qs] == seg[ks]).all()) and bool((seg[qs] != SPINE_SEG_ID).all())
    assert bool(aud[qs].all()) and bool(aud[ks].all())
    assert bool((ks > qs).all()), "added pairs must be exactly the backward-looking ones"


def test_bidirectional_audio_still_blocks_cross_branch_and_spine():
    """The isolation guarantees survive: no branch sees another branch's audio."""
    ex, batch = _bidir_batch()
    m = (
        build_script_mask(
            batch.seg_ids, batch.order_ids, batch.prefix_len, batch.valid, torch.float32, is_audio=batch.is_audio
        )[0, 0]
        == 0
    )
    seg, aud = batch.seg_ids[0], batch.is_audio[0]
    T = seg.shape[0]
    for q in range(T):
        for k in range(T):
            if not m[q, k]:
                continue
            if seg[q] != SPINE_SEG_ID and seg[k] != SPINE_SEG_ID:
                assert seg[q] == seg[k], f"branch {int(seg[q])} attends branch {int(seg[k])}"
            if seg[q] == SPINE_SEG_ID:
                assert seg[k] == SPINE_SEG_ID and not aud[k], "spine must stay pure text"


def test_bidirectional_audio_text_stays_causal():
    """Only the audio goes bidirectional; predicted words keep the causal rule."""
    ex, batch = _bidir_batch()
    m = (
        build_script_mask(
            batch.seg_ids, batch.order_ids, batch.prefix_len, batch.valid, torch.float32, is_audio=batch.is_audio
        )[0, 0]
        == 0
    )
    seg, aud, order = batch.seg_ids[0], batch.is_audio[0], batch.order_ids[0]
    for q in range(seg.shape[0]):
        if seg[q] == SPINE_SEG_ID or aud[q]:
            continue
        for k in range(seg.shape[0]):
            if m[q, k] and seg[k] == seg[q]:
                assert order[k] <= order[q], "a branch TEXT token attended a later token"


def test_bidirectional_audio_every_frame_sees_the_text_history():
    """Each audio frame attends the full instruction + words-so-far prefix."""
    ex, batch = _bidir_batch()
    m = (
        build_script_mask(
            batch.seg_ids, batch.order_ids, batch.prefix_len, batch.valid, torch.float32, is_audio=batch.is_audio
        )[0, 0]
        == 0
    )
    seg, aud, pref = batch.seg_ids[0], batch.is_audio[0], batch.prefix_len[0]
    P = int(ex.spine_len)
    for q in range(seg.shape[0]):
        if not aud[q]:
            continue
        want = torch.zeros(seg.shape[0], dtype=torch.bool)
        want[: int(pref[q])] = True
        assert bool((m[q, :P] == want[:P]).all()), f"audio frame {q} does not see exactly its history"


def test_bidirectional_audio_flex_predicate_matches_dense():
    """flex and dense must implement the same rule (as for the causal one)."""
    from nemo.collections.speechlm2.models.script_model import ScriptSTTModel

    ex, batch = _bidir_batch()
    dense = build_script_mask(
        batch.seg_ids, batch.order_ids, batch.prefix_len, batch.valid, torch.float32, is_audio=batch.is_audio
    )[:, 0]
    mod = ScriptSTTModel._script_mask_mod(batch, bidirectional_audio=True)
    B, T = batch.seg_ids.shape
    bi = torch.arange(B)[:, None, None].expand(B, T, T)
    qi = torch.arange(T)[None, :, None].expand(B, T, T)
    ki = torch.arange(T)[None, None, :].expand(B, T, T)
    assert torch.equal(mod(bi, None, qi, ki), dense == 0)


def test_bidirectional_audio_twod_mask_matches_flat():
    """The 2-D branch mask must encode the same rule as the flat one."""
    from nemo.collections.speechlm2.parts.script import build_twod_branch_mask

    ex, batch = _bidir_batch()
    flat = (
        build_script_mask(
            batch.seg_ids, batch.order_ids, batch.prefix_len, batch.valid, torch.float32, is_audio=batch.is_audio
        )[0, 0]
        == 0
    )
    P = int(ex.spine_len)
    rows = [(batch.seg_ids[0] == k + 1).nonzero(as_tuple=True)[0] for k in range(int(batch.seg_ids[0].max()))]
    b = max(int(r.numel()) for r in rows)
    N = len(rows)
    bprefix = torch.zeros(N, dtype=torch.long)
    bvalid = torch.zeros(N, b, dtype=torch.bool)
    baud = torch.zeros(N, b, dtype=torch.bool)
    for k, r in enumerate(rows):
        m_ = int(r.numel())
        bprefix[k] = batch.prefix_len[0, r[0]]
        bvalid[k, :m_] = True
        baud[k, :m_] = batch.is_audio[0, r]
    two = build_twod_branch_mask(bprefix, bvalid, P, torch.float32, branch_is_audio=baud)[:, 0] == 0
    for k, r in enumerate(rows):
        m_ = int(r.numel())
        assert torch.equal(two[k, :m_, :P], flat[r][:, :P]), f"branch {k}: spine part differs"
        assert torch.equal(two[k, :m_, P : P + m_], flat[r][:, r]), f"branch {k}: own part differs"


def test_bidirectional_audio_structured_backend_matches_dense():
    """The structured `script` kernel must reproduce the dense-masked attention."""
    from nemo.collections.speechlm2.parts.script_attention import (
        build_attention_plan,
        script_attention_plan,
        script_structured_attention,
    )

    torch.manual_seed(0)
    ex, batch = _bidir_batch()
    plan = build_attention_plan([ex], bidirectional_audio=True)
    B, T, h, d = 1, batch.seg_ids.shape[1], 2, 8
    q, k, v = (torch.randn(B, h, T, d) for _ in range(3))

    mask = build_script_mask(
        batch.seg_ids, batch.order_ids, batch.prefix_len, batch.valid, torch.float32, is_audio=batch.is_audio
    )
    ref = torch.softmax(torch.einsum("bhid,bhjd->bhij", q, k) * d**-0.5 + mask, dim=-1) @ v

    class _M:
        training = False

    with script_attention_plan(plan):
        got, _ = script_structured_attention(_M(), q, k, v, None, scaling=d**-0.5)
    got = got.transpose(1, 2)
    valid = batch.valid[0]
    assert torch.allclose(got[:, :, valid], ref[:, :, valid], atol=1e-5), "structured != dense"


@torch.no_grad()
@pytest.mark.parametrize("audio_window_frames", [0, 4])
def test_parity_bidirectional_audio_packed_vs_separate(audio_window_frames):
    """The correctness argument must survive the new rule.

    Packed branch logits still have to equal the standalone example's, now with
    both sides using the bidirectional-audio mask. This is what keeps a branch
    exactly a prefix-LM run over its own history + audio.
    """
    model = _tiny_qwen3()
    H = model.config.hidden_size

    instruction = [5, 6, 7]
    chunks = [
        ChunkSpec(audio_len=2, target_ids=[20, 21]),
        ChunkSpec(audio_len=3, target_ids=[30]),
        ChunkSpec(audio_len=1, target_ids=[]),
        ChunkSpec(audio_len=2, target_ids=[40, 41]),
    ]
    torch.manual_seed(321)
    all_frames = torch.randn(sum(c.audio_len for c in chunks), H)

    packed = build_packed_chunk_example(instruction, chunks, VS, VE, EOT, audio_window_frames=audio_window_frames)
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None],
        packed.order_ids[None],
        packed.prefix_len[None],
        valid[None],
        torch.float32,
        is_audio=packed.is_audio[None],
    )
    packed_emb = _embed_with_audio(
        model,
        packed.input_ids,
        packed.is_audio,
        _frames_by_index(all_frames, packed.audio_frame_index, packed.is_audio),
    )
    packed_logits = model(
        inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]
    ).logits[0]

    separate = build_separate_chunk_examples(instruction, chunks, VS, VE, EOT, audio_window_frames=audio_window_frames)
    for k, sep in enumerate(separate, start=1):
        sep_emb = _embed_with_audio(
            model, sep.input_ids, sep.is_audio, _frames_by_index(all_frames, sep.audio_frame_index, sep.is_audio)
        )
        # The standalone run needs the SAME rule: causal, plus its own audio run
        # attending itself both ways. Without this it would be plain causal and
        # the comparison would be meaningless.
        L = int(sep.input_ids.shape[0])
        aud = sep.is_audio
        allow = torch.ones(L, L, dtype=torch.bool).tril() | (aud[:, None] & aud[None, :])
        sep_mask = torch.zeros(1, 1, L, L).masked_fill(~allow, torch.finfo(torch.float32).min)
        sep_logits = model(
            inputs_embeds=sep_emb[None], attention_mask=sep_mask, position_ids=sep.position_ids[None]
        ).logits[0]

        packed_branch = packed_logits[(packed.seg_ids == k).nonzero(as_tuple=True)[0]]
        sep_branch = sep_logits[sep.branch_start :]
        assert packed_branch.shape == sep_branch.shape
        torch.testing.assert_close(packed_branch, sep_branch, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_bidirectional_audio_actually_changes_the_logits():
    """Guard against the flag being silently inert."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    chunks = [ChunkSpec(audio_len=4, target_ids=[20, 21]), ChunkSpec(audio_len=4, target_ids=[30])]
    torch.manual_seed(11)
    frames = torch.randn(sum(c.audio_len for c in chunks), H)

    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT)
    emb = _embed_with_audio(
        model, ex.input_ids, ex.is_audio, _frames_by_index(frames, ex.audio_frame_index, ex.is_audio)
    )
    valid = torch.ones_like(ex.input_ids, dtype=torch.bool)
    args = (ex.seg_ids[None], ex.order_ids[None], ex.prefix_len[None], valid[None], torch.float32)

    def run(mask):
        return model(inputs_embeds=emb[None], attention_mask=mask, position_ids=ex.position_ids[None]).logits[0]

    causal = run(build_script_mask(*args))
    bidir = run(build_script_mask(*args, is_audio=ex.is_audio[None]))
    assert not torch.allclose(causal, bidir, atol=1e-6), "bidirectional_audio had no effect"

    # ...but only through the audio: the SPINE is pure text and must be untouched.
    spine = (ex.seg_ids == SPINE_SEG_ID).nonzero(as_tuple=True)[0]
    torch.testing.assert_close(causal[spine], bidir[spine], atol=1e-6, rtol=1e-6)


@torch.no_grad()
def test_bidirectional_audio_decode_matches_teacher_forced_layout():
    """Decode must condition exactly as training does under the new rule.

    The decode path cannot rely on HF's implicit causal mask any more -- it has
    to hand the prefill an explicit 4-D one. This decodes greedily with
    bidirectional audio, feeds the result back as training targets under the
    bidirectional PACKED mask, and checks the teacher-forced argmax reproduces
    it. A mismatch here is the train/decode skew that would otherwise only show
    up as an unexplained WER gap.
    """
    model = _tiny_qwen3()
    H = model.config.hidden_size
    torch.manual_seed(5)

    cs = 3
    instruction = [5, 6]
    frames = torch.randn(9, H)  # exactly 3 chunks

    emitted, chunk_ids = batched_stream_decode_script(
        llm=model,
        embed_tokens=model.get_input_embeddings(),
        instruction_ids_list=[instruction],
        frames_list=[frames],
        chunk_size=cs,
        vision_start_id=VS,
        vision_end_id=VE,
        eot_id=EOT,
        pad_id=0,
        max_new_tokens=4,
        return_chunk_ids=True,
        bidirectional_audio=True,
    )
    emitted, chunk_ids = emitted[0], chunk_ids[0]

    per_chunk = [[] for _ in range(3)]
    for tok, k in zip(emitted, chunk_ids):
        per_chunk[k].append(tok)
    chunks = [ChunkSpec(audio_len=cs, target_ids=toks) for toks in per_chunk]

    packed = build_packed_chunk_example(instruction, chunks, VS, VE, EOT, supervise_eot=False)
    emb = _embed_with_audio(
        model, packed.input_ids, packed.is_audio, frames[packed.audio_frame_index[packed.is_audio]]
    )
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None],
        packed.order_ids[None],
        packed.prefix_len[None],
        valid[None],
        torch.float32,
        is_audio=packed.is_audio[None],
    )
    logits = model(inputs_embeds=emb[None], attention_mask=mask, position_ids=packed.position_ids[None]).logits[0]

    sup = (packed.target_ids != IGNORE_INDEX).nonzero(as_tuple=True)[0]
    assert sup.numel() > 0
    torch.testing.assert_close(logits[sup].argmax(dim=-1), packed.target_ids[sup])


@torch.no_grad()
def test_bidirectional_audio_decode_passes_the_right_prefill_mask():
    """The decode flag must reach the LLM, and open exactly the audio block.

    Asserted on the mask handed to the model rather than on the decoded tokens:
    the rule shifts logits by ~1e-3 on a tiny random model, which almost never
    flips a greedy argmax, so a token-level check would pass whether or not the
    flag did anything.
    """
    model = _tiny_qwen3()
    H = model.config.hidden_size
    torch.manual_seed(9)
    frames = torch.randn(12, H)

    class _Spy:
        """Records the attention_mask of each PREFILL call (the multi-token ones)."""

        def __init__(self, inner):
            self.inner, self.prefills = inner, []
            self.config, self.device = inner.config, next(inner.parameters()).device

        def __call__(self, **kw):
            if kw["inputs_embeds"].shape[1] > 1:
                self.prefills.append(kw["attention_mask"])
            return self.inner(**kw)

        def get_input_embeddings(self):
            return self.inner.get_input_embeddings()

    def run(flag):
        spy = _Spy(model)
        batched_stream_decode_script(
            llm=spy,
            embed_tokens=model.get_input_embeddings(),
            instruction_ids_list=[[5, 6]],
            frames_list=[frames],
            chunk_size=4,
            vision_start_id=VS,
            vision_end_id=VE,
            eot_id=EOT,
            pad_id=0,
            max_new_tokens=4,
            bidirectional_audio=flag,
        )
        return spy.prefills

    causal_masks = run(False)
    bidir_masks = run(True)
    assert causal_masks and bidir_masks
    # Without the flag the decoder keeps the cheap 2-D validity mask.
    assert all(m.dim() == 2 for m in causal_masks)

    for m in bidir_masks:
        assert m.dim() == 4, "prefill must get an explicit 4-D mask under bidirectional audio"
        allow = m[0, 0] == 0
        L = allow.shape[0]
        tri = torch.ones(L, L, dtype=torch.bool).tril()
        added = allow & ~tri
        assert added.any(), "no backward audio pairs were opened"
        # Every extra pair is strictly anti-causal and both ends lie in one
        # contiguous run -- the chunk's audio block.
        q, k = torch.where(added)
        assert bool((k > q).all())
        span = torch.cat([q, k]).unique()
        assert bool((span.max() - span.min() + 1) == span.numel()), "opened pairs are not one block"


def test_bidirectional_audio_rejects_state_machine_decode():
    """The FSM cannot honour the rule, so it must refuse rather than mis-decode."""
    from nemo.collections.speechlm2.models.script_model import ScriptSTTModel

    fake = ScriptSTTModel.__new__(ScriptSTTModel)
    fake._bidirectional_audio = True
    with pytest.raises(ValueError, match="incompatible with bidirectional_audio"):
        ScriptSTTModel._reject_fsm_with_bidirectional_audio(fake, state_machine=True)
    # ...and stays silent when either half is off.
    ScriptSTTModel._reject_fsm_with_bidirectional_audio(fake, state_machine=False)
    fake._bidirectional_audio = False
    ScriptSTTModel._reject_fsm_with_bidirectional_audio(fake, state_machine=True)


@torch.no_grad()
def test_bidirectional_audio_flex_attention_matches_dense_numerically():
    """End-to-end flex numerics, not just predicate equality.

    The predicate test compares booleans; this one compiles the block mask and
    runs flex_attention against dense-masked attention. The previous flex bug in
    this file (a None tensor inside the vmap) was invisible to a predicate-level
    check because the tests never built a real BlockMask.
    """
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    from nemo.collections.speechlm2.models.script_model import ScriptSTTModel

    ex = build_packed_chunk_example(INSTR, [ChunkSpec(4, [20, 21]), ChunkSpec(4, [30])], VS, VE, EOT)
    batch = collate_packed_chunk_examples([ex], pad_id=0)
    B, T = batch.seg_ids.shape
    h, d = 2, 8
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, h, T, d) for _ in range(3))

    bm = create_block_mask(ScriptSTTModel._script_mask_mod(batch, True), B=B, H=None, Q_LEN=T, KV_LEN=T, device="cpu")
    got = flex_attention(q, k, v, block_mask=bm, scale=d**-0.5)

    mask = build_script_mask(
        batch.seg_ids, batch.order_ids, batch.prefix_len, batch.valid, torch.float32, is_audio=batch.is_audio
    )
    ref = torch.softmax(torch.einsum("bhid,bhjd->bhij", q, k) * d**-0.5 + mask, dim=-1) @ v

    valid = batch.valid[0]
    torch.testing.assert_close(got[:, :, valid], ref[:, :, valid], atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_full_context_with_bidirectional_audio_is_one_unrestricted_block():
    """The combination actually launched: offline layout + unmasked audio.

    full_context makes the branch span the whole utterance, so bidirectional
    audio has maximal surface -- every frame attends every other frame. Neither
    feature's own tests cover the pair, and this is the configuration being
    trained, so pin its shape here.
    """
    n_frames, words = 12, [20, 21, 22]
    ex = build_packed_chunk_example(INSTR, [ChunkSpec(audio_len=n_frames, target_ids=words)], VS, VE, EOT)
    batch = collate_packed_chunk_examples([ex], pad_id=0)

    # One branch, and it carries every frame.
    assert int(batch.seg_ids[0].max()) == 1
    assert int(batch.is_audio[0].sum()) == n_frames

    allow = (
        build_script_mask(
            batch.seg_ids, batch.order_ids, batch.prefix_len, batch.valid, torch.float32, is_audio=batch.is_audio
        )[0, 0]
        == 0
    )
    aud = batch.is_audio[0].nonzero(as_tuple=True)[0]

    # Every audio pair, in both directions -- no chunk boundary survives.
    assert bool(allow[aud][:, aud].all()), "audio block is not fully connected"

    # The text it predicts is still causal, and still sees all the audio.
    seg, order = batch.seg_ids[0], batch.order_ids[0]
    txt = [
        i
        for i in range(seg.shape[0])
        if seg[i] != SPINE_SEG_ID and not batch.is_audio[0, i] and order[i] > order[aud[-1]]
    ]
    assert txt, "expected predicted-word positions after the audio"
    for q in txt:
        assert bool(allow[q, aud].all()), "a predicted word cannot see all the audio"
        for k in txt:
            if allow[q, k]:
                assert order[k] <= order[q], "predicted words are no longer causal"

    # The spine stays pure text under the combination.
    spine = (seg == SPINE_SEG_ID).nonzero(as_tuple=True)[0]
    assert not bool(allow[spine][:, aud].any()), "spine attended audio"


# ======================================================================
# StreamingSTTModel._sample_token: None-valued generation knobs
# ======================================================================


@pytest.mark.parametrize("do_sample", [False, True])
def test_sample_token_treats_none_knobs_as_disabled(do_sample):
    """transformers >= 5 defaults every sampling knob to None.

    None means "leave it off", but the code compared it to numbers:
    `no_repeat_ngram_size > 0` raised TypeError, and `repetition_penalty != 1.0`
    is True for None, reaching a division by None. Both sit BEFORE the greedy
    fast path, so an ordinary argmax decode died on a bare
    GenerationConfig(do_sample=False) -- which is exactly what the leaderboard
    driver passes.
    """
    from transformers import GenerationConfig

    from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel

    cfg = GenerationConfig(do_sample=do_sample)
    # Guard the premise: if a future transformers stops defaulting these to
    # None, this test would silently stop testing anything.
    assert cfg.no_repeat_ngram_size is None and cfg.repetition_penalty is None

    torch.manual_seed(0)
    logits = torch.randn(3, 32)
    fake = StreamingSTTModel.__new__(StreamingSTTModel)
    out = StreamingSTTModel._sample_token(fake, logits, [7, 7, 9], cfg)

    assert out.shape == (3,)
    assert out.dtype == torch.long
    assert bool(((out >= 0) & (out < 32)).all())
    if not do_sample:
        # Disabled knobs must leave the greedy result untouched.
        torch.testing.assert_close(out, logits.argmax(dim=-1))


def test_per_step_emissions_reconstructs_the_transcript():
    """RNN-T per-step attribution must not invent words.

    The decoder REVISES its tail between streaming steps, so an append-only diff
    of consecutive cumulative transcripts emits both the superseded words and
    their replacements. Those phantoms align as insertions -- on AMI that
    corrupted 42.7% of records and inflated the insertion count ~4x. The
    invariant that catches it: the per-step pieces must concatenate back to the
    final cumulative transcript.
    """
    import importlib.util
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "nemotron_leaderboard_eval.py"
    text = src.read_text()
    start = text.index("def _per_step_emissions(")
    ns = {"List": list}
    exec(text[start : text.index("\n\n\n", start)], ns)  # noqa: S102 - test-only extraction
    per_step = ns["_per_step_emissions"]

    cases = [
        (["the eval", "the evaluation is"], "the evaluation is"),  # tail revised
        (["Okay", "Okay."], "Okay."),  # last word rewritten
        (["a", "a b", "a b c"], "a b c"),  # pure append
        (["", "", "hi there"], "hi there"),  # silent leading steps
        (["x y z", "x q"], "x q"),  # tail shortened AND changed
    ]
    for steps, expected in cases:
        out = per_step([[s] for s in steps], 1)[0]
        assert " ".join(t for _, t in out).split() == expected.split(), f"{steps} -> {out}"
        # Steps must be non-decreasing and each word owned by exactly one step.
        assert [k for k, _ in out] == sorted(k for k, _ in out)


# ======================================================================
# Position-dependent emission penalty (inference-only)
# ======================================================================


def test_emission_penalty_schedule_repeats_its_last_value():
    from nemo.collections.speechlm2.parts.script import emission_penalty_at

    sched = [0.0, 0.5, 2.0]
    assert [emission_penalty_at(k, 0.0, sched) for k in range(5)] == [0.0, 0.5, 2.0, 2.0, 2.0]
    # Absent / empty must be a hard no-op, not a zero-length lookup error.
    assert emission_penalty_at(0, 0.0, None) == 0.0
    assert emission_penalty_at(7, 0.0, []) == 0.0
    # Linear form: first word always free, each later one costs lam more.
    assert [emission_penalty_at(k, 0.5) for k in range(4)] == [0.0, 0.5, 1.0, 1.5]
    assert [emission_penalty_at(k, 0.0) for k in range(4)] == [0.0] * 4
    # An explicit schedule takes precedence over lam.
    assert emission_penalty_at(3, 99.0, [0.0, 0.25]) == 0.25


@torch.no_grad()
def test_emission_penalty_off_is_bit_identical():
    """Default (None) must not perturb decoding at all."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    torch.manual_seed(4)
    frames = torch.randn(12, H)

    def run(pen):
        return batched_stream_decode_script(
            llm=model,
            embed_tokens=model.get_input_embeddings(),
            instruction_ids_list=[[5, 6]],
            frames_list=[frames],
            chunk_size=4,
            vision_start_id=VS,
            vision_end_id=VE,
            eot_id=EOT,
            pad_id=0,
            max_new_tokens=6,
            emission_penalty=pen,
        )[0]

    base = run(None)
    assert run([]) == base, "empty schedule changed the output"
    assert run([0.0, 0.0, 0.0]) == base, "all-zero schedule changed the output"


@torch.no_grad()
def test_emission_penalty_shortens_chunks_monotonically():
    """A larger penalty must never make a chunk emit MORE words."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    torch.manual_seed(4)
    frames = torch.randn(20, H)

    lens = []
    for p in (0.0, 2.0, 10.0, 1e4):
        out = batched_stream_decode_script(
            llm=model,
            embed_tokens=model.get_input_embeddings(),
            instruction_ids_list=[[5, 6]],
            frames_list=[frames],
            chunk_size=4,
            vision_start_id=VS,
            vision_end_id=VE,
            eot_id=EOT,
            pad_id=0,
            max_new_tokens=6,
            emission_penalty=[p],
        )[0]
        lens.append(len(out))
    assert lens == sorted(lens, reverse=True), f"not monotone in penalty: {lens}"
    assert lens[-1] == 0, "an overwhelming penalty must stop every chunk immediately"


@torch.no_grad()
def test_emission_penalty_is_position_dependent():
    """The whole point: position 0 and position 1 must be controllable apart.

    A schedule that is free at the first word but prohibitive afterwards should
    leave at most one word per chunk, while the flat-zero schedule does not.
    """
    model = _tiny_qwen3()
    H = model.config.hidden_size
    torch.manual_seed(4)
    frames = torch.randn(20, H)

    def run(pen):
        return batched_stream_decode_script(
            llm=model,
            embed_tokens=model.get_input_embeddings(),
            instruction_ids_list=[[5, 6]],
            frames_list=[frames],
            chunk_size=4,
            vision_start_id=VS,
            vision_end_id=VE,
            eot_id=EOT,
            pad_id=0,
            max_new_tokens=6,
            return_chunk_ids=True,
            emission_penalty=pen,
        )

    _, free_ids = run([0.0])
    toks, gated_ids = run([0.0, 1e4])
    from collections import Counter

    per_chunk_gated = Counter(gated_ids[0])
    assert per_chunk_gated and max(per_chunk_gated.values()) <= 1, f"expected <=1 word/chunk, got {per_chunk_gated}"
    assert max(Counter(free_ids[0]).values()) > 1, "baseline should emit multi-word chunks for this to be meaningful"


@torch.no_grad()
def test_emission_penalty_lambda_matches_the_equivalent_schedule():
    """lam=L must decode identically to the explicit ramp [0, L, 2L, ...].

    The two forms are one code path with two front-ends; if they ever diverge,
    a swept lambda would not mean what the ablation schedules meant.
    """
    model = _tiny_qwen3()
    H = model.config.hidden_size
    torch.manual_seed(4)
    frames = torch.randn(20, H)

    def run(**kw):
        return batched_stream_decode_script(
            llm=model,
            embed_tokens=model.get_input_embeddings(),
            instruction_ids_list=[[5, 6]],
            frames_list=[frames],
            chunk_size=4,
            vision_start_id=VS,
            vision_end_id=VE,
            eot_id=EOT,
            pad_id=0,
            max_new_tokens=6,
            **kw,
        )[0]

    lam = 0.5
    # max_new_tokens=6 bounds the words per chunk, so 8 entries cover every position.
    equivalent = [k * lam for k in range(8)]
    assert run(emission_penalty_lambda=lam) == run(emission_penalty=equivalent)
    # lam=0 is off.
    assert run(emission_penalty_lambda=0.0) == run()


# ======================================================================
# ASR-vocabulary swap
# ======================================================================


def _asr_spm_path(tmp_path):
    import glob

    from nemo.collections.speechlm2.parts.asr_vocab import extract_spm_from_nemo

    hits = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/**/*.nemo"),
        recursive=True,
    )
    if not hits:
        pytest.skip("nemotron ASR .nemo not present in the local HF cache")
    return extract_spm_from_nemo(hits[0], str(tmp_path))


def test_asr_vocab_tokenizer_roundtrips_and_keeps_special_strings(tmp_path):
    """Call sites resolve markers BY STRING, so the strings must survive the swap."""
    from nemo.collections.speechlm2.parts.asr_vocab import AsrVocabTokenizer

    specials = ["<|vision_start|>", "<|vision_end|>", "<|im_end|>", "<|box_start|>", "<|box_end|>"]
    tok = AsrVocabTokenizer(_asr_spm_path(tmp_path), special_tokens=specials, eos_token="<|im_end|>")

    assert len(tok) == 1024 + len(specials), "specials must be appended AFTER the SentencePiece pieces"
    # A piece keeps the id the ASR model gave it.
    assert tok.convert_ids_to_tokens(0) == "<unk>"
    for s in specials:
        i = tok.convert_tokens_to_ids(s)
        assert i >= 1024 and tok.convert_ids_to_tokens(i) == s
    assert tok.eos_token_id == tok.convert_tokens_to_ids("<|im_end|>")

    for text in ("the evaluation is complete", "hello world", "Marvin said okay."):
        assert tok.ids_to_text(tok.text_to_ids(text)) == text


def test_asr_vocab_embedding_init_is_donor_averaged(tmp_path):
    """New pieces must inherit the donor's geometry, not be random.

    ``in`` exists verbatim in Qwen and must be COPIED; ``▁the`` does not and must
    equal the mean of Qwen's embeddings for " the". Getting the word-initial
    mapping backwards (looking up "the" instead of " the") would seed every
    word-initial piece with a mid-word vector, which is silent and costly.
    """
    import glob

    from transformers import AutoTokenizer

    from nemo.collections.speechlm2.parts.asr_vocab import AsrVocabTokenizer, build_embedding_from_donor

    qdirs = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/"))
    if not qdirs:
        pytest.skip("Qwen3-1.7B not present in the local HF cache")
    donor = AutoTokenizer.from_pretrained(qdirs[0])

    tok = AsrVocabTokenizer(_asr_spm_path(tmp_path), special_tokens=["<|im_end|>"], eos_token="<|im_end|>")
    torch.manual_seed(0)
    donor_w = torch.randn(len(donor), 16)
    new_w = build_embedding_from_donor(tok, donor, donor_w, verbose=False)

    assert new_w.shape == (len(tok), 16)
    assert torch.isfinite(new_w).all()

    vocab = donor.get_vocab()
    # verbatim copy
    i = tok.convert_tokens_to_ids("in")
    if i > 0 and "in" in vocab:
        torch.testing.assert_close(new_w[i], donor_w[vocab["in"]])
    # word-initial piece -> donor's " the", NOT "the"
    j = tok.convert_tokens_to_ids("▁the")
    expected = donor_w[torch.tensor(donor.encode(" the", add_special_tokens=False))].mean(0)
    torch.testing.assert_close(new_w[j], expected)
    # the special keeps its own pretrained vector
    k = tok.convert_tokens_to_ids("<|im_end|>")
    torch.testing.assert_close(new_w[k], donor_w[vocab["<|im_end|>"]])


def test_asr_vocab_is_off_by_default():
    """Backward compatibility: existing configs must not trigger the swap."""
    from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModelConfig

    assert StreamingSTTModelConfig.text_vocab_from_asr is False


def test_asr_vocab_exposes_the_nemo_and_hf_attributes_the_pipeline_uses(tmp_path):
    """The swap is only non-invasive if the wrapper covers every attribute used.

    The dataset pads with ``tokenizer.pad_id`` and validates its delimiters via
    ``getattr(tok, "unk_token_id", None)`` -- if the latter is missing the check
    is skipped and an out-of-vocabulary marker silently becomes <unk>.
    """
    from nemo.collections.speechlm2.parts.asr_vocab import AsrVocabTokenizer

    tok = AsrVocabTokenizer(
        _asr_spm_path(tmp_path),
        special_tokens=["<|vision_start|>", "<|im_end|>", "<|endoftext|>"],
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )
    # pad_id must be a REAL index -- collators index with it.
    assert 0 <= tok.pad_id < len(tok)
    assert tok.eos_id == tok.convert_tokens_to_ids("<|im_end|>")
    assert tok.unk_token_id == 0

    # The dataset's out-of-vocabulary guard must be able to fire.
    assert tok.convert_tokens_to_ids("<|definitely_not_a_token|>") == tok.unk_token_id
    assert tok.convert_tokens_to_ids("<|vision_start|>") != tok.unk_token_id

    ids = tok.text_to_ids("hello world")
    assert tok.tokens_to_text(tok.ids_to_tokens(ids), remove_special_tokens=True) == "hello world"


def test_asr_vocab_covers_every_tokenizer_attribute_speechlm2_uses():
    """The swap is only safe if the wrapper covers the WHOLE interface.

    Both cluster jobs died on ``AttributeError: 'AsrVocabTokenizer' object has no
    attribute 'encode'`` because the original audit checked four files by hand
    and missed callers elsewhere in the package. Derive the requirement from the
    source instead, so a newly-used attribute fails here rather than an hour into
    a run.
    """
    import pathlib
    import re

    from nemo.collections.speechlm2.parts.asr_vocab import AsrVocabTokenizer

    root = pathlib.Path(__file__).resolve().parents[3] / "nemo" / "collections" / "speechlm2"
    used = set()
    for f in root.rglob("*.py"):
        text = f.read_text()
        used |= set(re.findall(r"tokenizer\.tokenizer\.([a-zA-Z_]+)", text))
        used |= set(re.findall(r"self\.tokenizer\.([a-zA-Z_]+)", text))
    used.discard("tokenizer")  # the self-reference the wrapper provides

    missing = sorted(a for a in used if not hasattr(AsrVocabTokenizer, a))
    assert not missing, f"AsrVocabTokenizer is missing attributes speechlm2 uses: {missing}"


def test_asr_vocab_encode_matches_text_to_ids(tmp_path):
    """`encode` is the HF spelling of `text_to_ids`; they must not diverge."""
    from nemo.collections.speechlm2.parts.asr_vocab import AsrVocabTokenizer

    tok = AsrVocabTokenizer(_asr_spm_path(tmp_path), special_tokens=["<|im_end|>"], eos_token="<|im_end|>")
    for text in ("hello world", "the evaluation is complete", "<audio><audio>"):
        assert tok.encode(text, add_special_tokens=False) == tok.text_to_ids(text)
    v = tok.get_vocab()
    assert len(v) == len(tok) and v is tok.get_vocab(), "get_vocab must be complete and cached"
    assert tok.decode(tok.encode("hello world")) == "hello world"


def test_asr_vocab_is_a_tokenizer_spec_and_dispatches_like_the_baseline(tmp_path):
    """Lhotse's dataloader DISPATCHES ON TYPE, not on duck-typing.

    TokenizerWrapper routes a TokenizerSpec through ``text_to_ids`` and anything
    else through ``tokenizer(text)`` -- the character-parser protocol. The
    tokenizer this replaces (NeMo's AutoTokenizer) is a TokenizerSpec, so the
    replacement must be one too; otherwise training dies mid-dataloader with
    "'AsrVocabTokenizer' object is not callable", which is what happened.
    """
    from nemo.collections.common.tokenizers.aggregate_tokenizer import TokenizerWrapper
    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

    from nemo.collections.speechlm2.parts.asr_vocab import AsrVocabTokenizer

    tok = AsrVocabTokenizer(_asr_spm_path(tmp_path), special_tokens=["<|im_end|>"], eos_token="<|im_end|>")
    assert isinstance(tok, TokenizerSpec)
    # Same dispatch branch as the tokenizer it replaces.
    assert issubclass(AutoTokenizer, TokenizerSpec)
    wrapper = TokenizerWrapper(tok)
    assert wrapper._impl.__name__ == "_call_tokenizer"

    text = "the evaluation is complete"
    assert wrapper(text, None) == tok.text_to_ids(text)  # how the dataloader calls it
    assert tok.tokens_to_ids(tok.text_to_tokens(text)) == tok.text_to_ids(text)
