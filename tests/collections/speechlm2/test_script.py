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
    collate_packed_chunk_examples,
)
from nemo.collections.speechlm2.parts.script_messages import get_llm_messages_for_sample

# Token ids for the toy vocab used in the structural tests.
VS, VE, EOT = 90, 91, 92
INSTR = [10, 11]  # 2-token instruction


def _blocked(v) -> bool:
    return float(v) == torch.finfo(torch.float32).min


def _allowed(v) -> bool:
    return float(v) == 0.0


def _mask_of(packed):
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    return build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
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
    m = build_script_mask(batch.seg_ids, batch.position_ids, batch.prefix_len, batch.valid, torch.float32)

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
@pytest.mark.parametrize("audio_history_chunks", [0, 1, 2])
def test_parity_packed_vs_separate_examples(audio_history_chunks):
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

    packed = build_packed_chunk_example(instruction, chunks, VS, VE, EOT, audio_history_chunks=audio_history_chunks)
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
        instruction, chunks, VS, VE, EOT, audio_history_chunks=audio_history_chunks
    )
    for k, sep in enumerate(separate, start=1):
        sep_emb = _embed_with_audio(
            model,
            sep.input_ids,
            sep.is_audio,
            _frames_by_index(all_frames, sep.audio_frame_index, sep.is_audio),
        )
        sep_logits = model(inputs_embeds=sep_emb[None]).logits[0]  # plain causal, positions 0..L-1

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
    mask = build_script_mask(batch.seg_ids, batch.position_ids, batch.prefix_len, batch.valid, torch.float32)

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
@pytest.mark.parametrize("audio_history_chunks", [0, 1])
def test_batched_decode_matches_per_utterance(audio_history_chunks):
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
