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
"""Tests for the SCRIPT (spine + branch) packed layout.

The centrepiece is ``test_parity_packed_vs_separate_examples``: it proves that a
branch's logits in the single packed sequence are numerically identical to
running that chunk as its own standalone ``[history] <vs> audio <ve> words``
example. That equivalence is the whole correctness argument for the custom 4D
mask + overlapping ``position_ids``.
"""

from types import SimpleNamespace

import pytest
import torch

from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX, IGNORE_INDEX
from nemo.collections.speechlm2.parts.script import (
    ChunkSpec,
    _audio_window_start,
    batched_stream_decode_redecode,
    batched_stream_decode_script,
    batched_stream_decode_script_last_layer,
    build_script_mask,
    build_packed_chunk_example,
    build_packed_redecode_example,
    build_separate_chunk_examples,
    build_separate_redecode_examples,
    collate_packed_chunk_examples,
    run_script_layers_split,
    stream_decode_script,
)

# Token ids for the toy vocab used in the structural tests.
VS, VE, EOT = 90, 91, 92
INSTR = [10, 11]  # 2-token instruction


def _blocked(v) -> bool:
    return float(v) == torch.finfo(torch.float32).min


def _allowed(v) -> bool:
    return float(v) == 0.0


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_packed_layout_structure():
    # 2 chunks: chunk1 reveals [20,21] with 2 frames; chunk2 reveals [30] with 3 frames.
    chunks = [ChunkSpec(audio_len=2, target_ids=[20, 21]), ChunkSpec(audio_len=3, target_ids=[30])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT)

    # Spine = instruction + all words = [10,11, 20,21, 30]  (P=5)
    assert ex.spine_len == 5
    assert ex.input_ids[:5].tolist() == [10, 11, 20, 21, 30]
    assert ex.position_ids[:5].tolist() == [0, 1, 2, 3, 4]
    assert ex.seg_ids[:5].tolist() == [0, 0, 0, 0, 0]
    assert ex.target_ids[:5].tolist() == [IGNORE_INDEX] * 5  # spine never supervised
    assert ex.is_audio[:5].tolist() == [False] * 5

    # Branch 1: <vs> A A <ve> 20 21 <eot>  ; prefix_len = 2 (just the instruction)
    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0]
    assert ex.input_ids[b1].tolist() == [VS, AUDIO_TOKEN_IDX, AUDIO_TOKEN_IDX, VE, 20, 21, EOT]
    assert ex.is_audio[b1].tolist() == [False, True, True, False, False, False, False]
    # positions start at prefix_len=2 and are contiguous (7 branch tokens)
    assert ex.position_ids[b1].tolist() == [2, 3, 4, 5, 6, 7, 8]
    assert ex.position_ids[b1].tolist() == list(range(2, 2 + 7))
    assert (ex.prefix_len[b1] == 2).all()
    # targets: <ve> predicts 20, "20" predicts 21, "21" predicts eot; rest ignore
    tb1 = ex.target_ids[b1].tolist()
    assert tb1 == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 20, 21, EOT, IGNORE_INDEX]

    # Branch 2: <vs> A A A <ve> 30 <eot> ; prefix_len = 4 (instruction + "20 21")
    b2 = (ex.seg_ids == 2).nonzero(as_tuple=True)[0]
    assert ex.input_ids[b2].tolist() == [VS, AUDIO_TOKEN_IDX, AUDIO_TOKEN_IDX, AUDIO_TOKEN_IDX, VE, 30, EOT]
    assert (ex.prefix_len[b2] == 4).all()
    assert ex.position_ids[b2].tolist() == list(range(4, 4 + 7))
    tb2 = ex.target_ids[b2].tolist()
    assert tb2 == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 30, EOT, IGNORE_INDEX]


def test_empty_chunk_predicts_only_eot():
    # A silent chunk (no words) should just predict eot right after <ve>.
    chunks = [ChunkSpec(audio_len=2, target_ids=[])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT)
    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0]
    assert ex.input_ids[b1].tolist() == [VS, AUDIO_TOKEN_IDX, AUDIO_TOKEN_IDX, VE, EOT]
    # <ve> position predicts eot; nothing else supervised
    assert ex.target_ids[b1].tolist() == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, EOT, IGNORE_INDEX]


FLUSH = 89  # toy-vocab <flush> control-token id (distinct from VS/VE/EOT)


def test_packed_flush_token_layout():
    # A flush chunk inserts <flush> right after <ve> (before the words). It is an
    # INPUT signal: <ve> is NOT taught to emit <flush>; the <flush> position predicts
    # the first word instead.
    chunks = [ChunkSpec(audio_len=2, target_ids=[20, 21], flush=True)]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, flush_id=FLUSH)
    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0]
    # branch: <vs> A A <ve> <flush> 20 21 <eot>
    assert ex.input_ids[b1].tolist() == [VS, AUDIO_TOKEN_IDX, AUDIO_TOKEN_IDX, VE, FLUSH, 20, 21, EOT]
    assert ex.is_audio[b1].tolist() == [False, True, True, False, False, False, False, False]
    # positions stay contiguous from prefix_len=2 (8 branch tokens now)
    assert ex.position_ids[b1].tolist() == list(range(2, 2 + 8))
    # targets: <ve> IGNORED (never emit <flush>); <flush> predicts 20; 20->21; 21->eot.
    assert ex.target_ids[b1].tolist() == [
        IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 20, 21, EOT, IGNORE_INDEX
    ]


def test_packed_flush_empty_chunk_predicts_eot():
    # An EMPTY flush chunk (nothing pending) still gets a <flush> token; that token
    # predicts <eot> so the model learns "flush with nothing pending -> emit nothing".
    chunks = [ChunkSpec(audio_len=2, target_ids=[], flush=True)]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, flush_id=FLUSH)
    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0]
    assert ex.input_ids[b1].tolist() == [VS, AUDIO_TOKEN_IDX, AUDIO_TOKEN_IDX, VE, FLUSH, EOT]
    assert ex.target_ids[b1].tolist() == [
        IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, EOT, IGNORE_INDEX
    ]


def test_packed_flush_off_when_chunk_not_flagged():
    # Passing flush_id but leaving ch.flush False must reproduce the original layout
    # (backward compatible -- no stray control token).
    chunks = [ChunkSpec(audio_len=2, target_ids=[20, 21], flush=False)]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, flush_id=FLUSH)
    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0]
    assert FLUSH not in ex.input_ids[b1].tolist()
    assert ex.input_ids[b1].tolist() == [VS, AUDIO_TOKEN_IDX, AUDIO_TOKEN_IDX, VE, 20, 21, EOT]


def test_packed_flush_rejects_contiguous_positions():
    chunks = [ChunkSpec(audio_len=2, target_ids=[20], flush=True)]
    with pytest.raises(ValueError):
        build_packed_chunk_example(
            INSTR, chunks, VS, VE, EOT, flush_id=FLUSH, contiguous_text_positions=True
        )


# ---------------------------------------------------------------------------
# Mask tests
# ---------------------------------------------------------------------------


def test_mask_spine_causal_and_pure_text():
    chunks = [ChunkSpec(audio_len=2, target_ids=[20, 21]), ChunkSpec(audio_len=3, target_ids=[30])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT)
    valid = torch.ones_like(ex.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        ex.seg_ids[None], ex.position_ids[None], ex.prefix_len[None], valid[None], torch.float32
    )[0, 0]
    P = ex.spine_len

    # Spine causal among itself.
    assert _allowed(mask[3, 0]) and _allowed(mask[3, 3]) and _blocked(mask[0, 3])
    # Spine query never attends any branch/audio key (all branch keys are >= P).
    for q in range(P):
        for j in range(P, ex.input_ids.numel()):
            assert _blocked(mask[q, j])


def test_mask_branch_sees_only_prefix_and_own_branch():
    chunks = [ChunkSpec(audio_len=2, target_ids=[20, 21]), ChunkSpec(audio_len=3, target_ids=[30])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT)
    valid = torch.ones_like(ex.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        ex.seg_ids[None], ex.position_ids[None], ex.prefix_len[None], valid[None], torch.float32
    )[0, 0]

    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0].tolist()
    b2 = (ex.seg_ids == 2).nonzero(as_tuple=True)[0].tolist()

    # Branch 2 word "30" (a query) sees: instruction (spine 0,1) + "20 21" (spine 2,3)
    # = its history prefix of length 4 -> spine indices 0..3 allowed, spine 4 ("30") blocked.
    q30 = b2[5]  # <vs>,A,A,A,<ve>,30,<eot> -> index 5 is "30"
    for j in range(4):
        assert _allowed(mask[q30, j]), f"branch2 should see prefix spine token {j}"
    assert _blocked(mask[q30, 4])  # spine "30" (its own word in the spine) is NOT history

    # Branch 2 must NOT see branch 1 at all (different chunk / its audio + words).
    for j in b1:
        assert _blocked(mask[q30, j])

    # Branch 2 sees its own audio + earlier branch tokens (causal), not future.
    assert _allowed(mask[q30, b2[1]])  # its own audio frame
    assert _allowed(mask[q30, b2[4]])  # its own <ve>
    assert _blocked(mask[q30, b2[6]])  # its own eot (future)

    # Branch 1 audio query sees instruction prefix (len 2) but not spine words.
    qa1 = b1[1]  # first audio frame of branch 1
    assert _allowed(mask[qa1, 0]) and _allowed(mask[qa1, 1])
    assert _blocked(mask[qa1, 2])  # spine word "20" is not in branch1's prefix (len 2)


def test_mask_padding_blocked():
    chunks = [ChunkSpec(audio_len=1, target_ids=[20])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT)
    T = ex.input_ids.numel()
    valid = torch.ones(T + 2, dtype=torch.bool)
    valid[T:] = False  # two padding keys
    seg = torch.cat([ex.seg_ids, torch.full((2,), -1)])
    pos = torch.cat([ex.position_ids, torch.zeros(2, dtype=torch.long)])
    pref = torch.cat([ex.prefix_len, torch.zeros(2, dtype=torch.long)])
    mask = build_script_mask(seg[None], pos[None], pref[None], valid[None], torch.float32)[0, 0]
    for q in range(T + 2):
        assert _blocked(mask[q, T]) and _blocked(mask[q, T + 1])


class _FakeTok:
    """Deterministic char-code tokenizer for parsing tests."""

    def text_to_ids(self, s: str):
        s = s.strip()
        return [ord(c) for c in s] if s else []


def _make_dataset_stub(blank_token: str):
    from nemo.collections.speechlm2.data.script_dataset import ScriptSTTDataset

    ds = object.__new__(ScriptSTTDataset)  # bypass heavy __init__
    ds.tokenizer = _FakeTok()
    ds.cfg = SimpleNamespace(audio_tag="<audio>", blank_token=blank_token)
    return ds


def test_messages_to_chunks_noblank():
    ds = _make_dataset_stub(blank_token="")
    messages = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "<audio><audio>"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "<audio><audio>"},
        {"role": "assistant", "content": ""},  # silent chunk (no-blank)
        {"role": "user", "content": "<audio>"},
        {"role": "assistant", "content": " world"},
    ]
    chunks = ds._messages_to_chunks(messages)
    assert [c.audio_len for c in chunks] == [2, 2, 1]
    assert chunks[0].target_ids == [ord(c) for c in "hello"]
    assert chunks[1].target_ids == []  # blank -> empty
    assert chunks[2].target_ids == [ord(c) for c in "world"]


def test_messages_to_chunks_explicit_blank_sentinel():
    ds = _make_dataset_stub(blank_token="<blank>")
    messages = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "<audio><audio>"},
        {"role": "assistant", "content": "<blank>"},  # blank sentinel -> empty
        {"role": "user", "content": "<audio>"},
        {"role": "assistant", "content": "hi"},
    ]
    chunks = ds._messages_to_chunks(messages)
    assert [c.audio_len for c in chunks] == [2, 1]
    assert chunks[0].target_ids == []
    assert chunks[1].target_ids == [ord(c) for c in "hi"]


def test_messages_to_chunks_reads_flush_flag():
    # The per-turn "flush" tag on an assistant message must propagate to ChunkSpec.flush.
    ds = _make_dataset_stub(blank_token="")
    messages = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "<audio><audio>"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "<audio>"},
        {"role": "assistant", "content": " world", "flush": True},
    ]
    chunks = ds._messages_to_chunks(messages)
    assert chunks[0].flush is False
    assert chunks[1].flush is True


def test_flush_final_chunk_emits_stranded_tail_word():
    # With a large delay, the last word is stranded past the final chunk boundary
    # (ready = end + delay > num_frames). enable_flush makes the FINAL chunk a flush
    # turn (delay ignored) that still emits it, and tags exactly that turn.
    from nemo.collections.speechlm2.parts.alignments import WordAlignment
    from nemo.collections.speechlm2.parts.script_messages import get_llm_messages_for_sample

    aligns = [
        WordAlignment(text="hello", start_time=0.16, end_time=0.48),  # end frame 6
        WordAlignment(text="world", start_time=0.60, end_time=0.80),  # end frame 10
    ]
    kw = dict(
        system_role="system",
        system_prompt="p",
        audio_tag="<a>",
        blank_token="",
        chunk_size=2,
        audio_duration_secs=1.12,  # -> 14 frames -> 7 chunks (boundaries 2..14)
        frame_length_in_secs=0.08,
        alignments=aligns,
        transcript="hello world",
    )
    # delay 6: "world" ready = 10 + 6 = 16 > 14 (stranded). flush_prob=0 -> only the
    # final chunk flushes.
    msgs = get_llm_messages_for_sample(num_delay_frames=6, enable_flush=True, flush_prob=0.0, **kw)
    assistant = [m for m in msgs if m["role"] == "assistant"]
    assert assistant[-1].get("flush") is True  # final chunk is the flush turn
    assert sum(1 for m in assistant if m.get("flush")) == 1  # only the final chunk
    text = " ".join(m["content"] for m in assistant if m["content"].strip())
    assert "hello" in text and "world" in text  # nothing dropped

    # Without flush the same last word is only saved by the residual dump and NO turn
    # is tagged -> the model gets no explicit end-of-audio signal.
    msgs_off = get_llm_messages_for_sample(num_delay_frames=6, enable_flush=False, **kw)
    assert not any(m.get("flush") for m in msgs_off if m["role"] == "assistant")


def _cc_dataset_class():
    from nemo.collections.speechlm2.data.script_dataset import ScriptSTTDataset

    return ScriptSTTDataset


def test_delay_prompt_sampling_pairs_and_covers_all():
    import numpy as np

    ds = object.__new__(_cc_dataset_class())  # bypass heavy __init__
    ds._delay_prompts = [{"delay": 0, "prompt": "A"}, {"delay": 2, "prompt": "B"}, {"delay": 4, "prompt": "C"}]
    valid = {(0, "A"), (2, "B"), (4, "C")}
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        got = ds._sample_delay_prompt(rng)
        assert got in valid  # delay stays paired with its own prompt
        seen.add(got)
    assert seen == valid  # all three settings get sampled


def test_delay_prompt_disabled_returns_none():
    import numpy as np

    ds = object.__new__(_cc_dataset_class())
    ds._delay_prompts = None
    assert ds._sample_delay_prompt(np.random.default_rng(0)) is None


class _CharTok:
    """Char-code tokenizer with round-tripping ids_to_text (for prefix tests)."""

    def text_to_ids(self, s):
        return [ord(c) for c in s]

    def ids_to_text(self, ids):
        return "".join(chr(int(i)) for i in ids)


def test_prefix_word_ids_is_a_strict_char_prefix():
    import numpy as np

    ds = object.__new__(_cc_dataset_class())
    ds.tokenizer = _CharTok()
    rng = np.random.default_rng(0)
    full = [ord(c) for c in " two"]  # leading space + "two"
    for _ in range(20):
        got = ds._prefix_word_ids(full, rng)
        assert got is not None
        assert got == full[: len(got)]  # genuine char/token prefix (keeps the leading space)
        assert got != full  # strictly shorter (a truncation)
    # single-char core or empty -> no proper prefix
    assert ds._prefix_word_ids([ord(" "), ord("a")], rng) is None
    assert ds._prefix_word_ids([], rng) is None


def test_sample_prefix_corrupt_eligibility():
    import numpy as np

    ds = object.__new__(_cc_dataset_class())
    ds.tokenizer = _CharTok()
    ds._sc_prefix_prob = 1.0  # always corrupt eligible chunks
    one = [ord(c) for c in " one"]
    two = [ord(c) for c in " two"]
    chunks = [
        ChunkSpec(2, one, last_word_ids=one),
        ChunkSpec(2, [], last_word_ids=[]),  # silent chunk (no output)
        ChunkSpec(2, two, last_word_ids=two),
    ]
    corrupt = ds._sample_prefix_corrupt(chunks, np.random.default_rng(0))
    assert corrupt[0] is None  # no previous chunk
    assert corrupt[1] is not None and corrupt[1] == one[: len(corrupt[1])]  # prefix of chunk0's " one"
    assert corrupt[2] is None  # previous chunk (1) was silent -> skip


# ---------------------------------------------------------------------------
# Exact-delay + text-representation prompting
# ---------------------------------------------------------------------------


def _promptctl_stub(vary=True, blank=""):
    cls = _cc_dataset_class()
    ds = object.__new__(cls)
    ds.cfg = SimpleNamespace(blank_token=blank)
    ds._vary_text_repr = vary
    ds._text_repr_keep = set("'")
    ds._prompt_template = cls._DEFAULT_PROMPT_TEMPLATE
    ds._format_clauses = dict(cls._DEFAULT_FORMAT_CLAUSES)
    ds._exact_delay = True
    ds._exact_max_delay = 4
    return ds


def test_exact_delay_prompt_render():
    ds = _promptctl_stub()
    p = ds._build_exact_prompt(3, True, True)
    assert "delay of 3 frames" in p
    assert "normal capitalization and punctuation" in p
    p2 = ds._build_exact_prompt(0, False, False)
    assert "delay of 0 frames" in p2
    assert "lowercase with no punctuation" in p2
    # No dangling placeholders.
    assert "{" not in p and "}" not in p


def test_exact_delay_prompt_render_without_text_repr():
    ds = _promptctl_stub(vary=False)
    p = ds._build_exact_prompt(2, True, True)
    assert "delay of 2 frames" in p
    assert "{" not in p and "}" not in p  # {format_clause} resolves to empty


def test_text_repr_transform_all_four_combos():
    ds = _promptctl_stub()
    content = " Hello, World's best!"
    assert ds._apply_text_repr(content, True, True) == content  # cap+punct: unchanged
    assert ds._apply_text_repr(content, False, True) == " hello, world's best!"  # lowercase, keep punct
    assert ds._apply_text_repr(content, True, False) == " Hello World's best"  # strip punct, keep apostrophe+case
    assert ds._apply_text_repr(content, False, False) == " hello world's best"  # lowercase + strip punct


def test_text_repr_preserves_leading_space_and_blank():
    ds = _promptctl_stub(blank="<blank>")
    # blank / empty sentinels untouched.
    assert ds._apply_text_repr("<blank>", False, False) == "<blank>"
    assert ds._apply_text_repr("", False, False) == ""
    # no leading space in -> none out.
    assert ds._apply_text_repr("Cat.", True, False) == "Cat"
    # punctuation-only content collapses to '' (treated as silent downstream).
    assert ds._apply_text_repr(" ...", True, False) == ""


def test_strip_punct_collapses_and_keeps_apostrophe():
    ds = _promptctl_stub()
    assert ds._strip_punct("well - known") == "well known"  # hyphen removed, spaces collapsed
    assert ds._strip_punct("don't stop!") == "don't stop"  # apostrophe kept, bang removed


def test_append_format_clause():
    ds = _promptctl_stub()
    base = "Do the thing."
    out = ds._append_format_clause(base, False, False)
    assert out == "Do the thing. Write the text in all lowercase with no punctuation."


def test_sample_text_repr_disabled_makes_no_draw():
    import numpy as np

    ds = _promptctl_stub(vary=False)
    rng = np.random.default_rng(0)
    state_before = rng.bit_generator.state
    assert ds._sample_text_repr(rng) == (True, True)
    assert rng.bit_generator.state == state_before  # no RNG consumption when disabled


def test_sample_text_repr_covers_all_combos():
    import numpy as np

    ds = _promptctl_stub()
    rng = np.random.default_rng(0)
    seen = {ds._sample_text_repr(rng) for _ in range(500)}
    assert seen == {(True, True), (True, False), (False, True), (False, False)}


def test_collate_shapes_and_padding():
    ex1 = build_packed_chunk_example(INSTR, [ChunkSpec(2, [20, 21])], VS, VE, EOT)
    ex2 = build_packed_chunk_example(INSTR, [ChunkSpec(1, [30]), ChunkSpec(2, [31])], VS, VE, EOT)
    batch = collate_packed_chunk_examples([ex1, ex2], pad_id=0)
    T = max(ex1.input_ids.numel(), ex2.input_ids.numel())
    assert batch.input_ids.shape == (2, T)
    assert batch.valid[0, ex1.input_ids.numel():].sum() == 0  # padded region invalid
    assert batch.valid[0, : ex1.input_ids.numel()].all()
    assert batch.seg_ids[0, ex1.input_ids.numel():].tolist() == [-1] * (T - ex1.input_ids.numel())


# ---------------------------------------------------------------------------
# Parity test: packed branch logits == standalone per-chunk example logits
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


def _embed_with_audio(model, input_ids: torch.Tensor, is_audio: torch.Tensor, audio_vecs: torch.Tensor):
    """Embed text ids and splice ``audio_vecs`` (ordered) at audio positions."""
    ids = input_ids.clone()
    ids[is_audio] = 0  # any valid id; overwritten below
    emb = model.get_input_embeddings()(ids)  # (L, H)
    if is_audio.any():
        emb = emb.clone()
        emb[is_audio] = audio_vecs.to(emb.dtype)
    return emb


@torch.no_grad()
def test_parity_packed_vs_separate_examples():
    model = _tiny_qwen3()
    H = model.config.hidden_size

    instruction = [5, 6, 7]  # 3-token instruction
    chunks = [
        ChunkSpec(audio_len=2, target_ids=[20, 21]),
        ChunkSpec(audio_len=3, target_ids=[30]),
        ChunkSpec(audio_len=1, target_ids=[]),  # silent chunk
        ChunkSpec(audio_len=2, target_ids=[40, 41, 42]),
    ]

    # One random frame-embedding per chunk; reused identically in both paths.
    torch.manual_seed(123)
    audio_frames = [torch.randn(ch.audio_len, H) for ch in chunks]

    # --- packed forward ---
    packed = build_packed_chunk_example(instruction, chunks, VS, VE, EOT)
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    packed_audio = torch.cat(audio_frames, dim=0)  # order = chunk1..chunkN frames
    packed_emb = _embed_with_audio(model, packed.input_ids, packed.is_audio, packed_audio)
    packed_logits = model(
        inputs_embeds=packed_emb[None],
        attention_mask=mask,
        position_ids=packed.position_ids[None],
    ).logits[0]

    # --- separate forwards (oracle) ---
    separate = build_separate_chunk_examples(instruction, chunks, VS, VE, EOT)

    for k, (sep, frames) in enumerate(zip(separate, audio_frames), start=1):
        sep_emb = _embed_with_audio(model, sep.input_ids, sep.is_audio, frames)
        sep_logits = model(inputs_embeds=sep_emb[None]).logits[0]  # standard causal, positions 0..L-1

        packed_branch_idx = (packed.seg_ids == k).nonzero(as_tuple=True)[0]
        packed_branch_logits = packed_logits[packed_branch_idx]
        sep_branch_logits = sep_logits[sep.branch_start :]

        assert packed_branch_logits.shape == sep_branch_logits.shape
        torch.testing.assert_close(packed_branch_logits, sep_branch_logits, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_parity_batched():
    """Two utterances packed+padded into a batch must match their standalone runs."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    torch.manual_seed(7)

    utts = [
        [ChunkSpec(2, [20, 21]), ChunkSpec(2, [22])],
        [ChunkSpec(1, [30]), ChunkSpec(3, [31, 32]), ChunkSpec(2, [])],
    ]
    instruction = [5, 6]

    examples = [build_packed_chunk_example(instruction, chs, VS, VE, EOT) for chs in utts]
    batch = collate_packed_chunk_examples(examples, pad_id=0)
    mask = build_script_mask(
        batch.seg_ids, batch.position_ids, batch.prefix_len, batch.valid, torch.float32
    )

    # Build batched embeds with per-utterance audio frames.
    audio_by_utt = [[torch.randn(ch.audio_len, H) for ch in chs] for chs in utts]
    B, T = batch.input_ids.shape
    emb = torch.zeros(B, T, H)
    for i, (ex, frames) in enumerate(zip(examples, audio_by_utt)):
        L = ex.input_ids.numel()
        packed_audio = torch.cat(frames, dim=0)
        emb[i, :L] = _embed_with_audio(model, ex.input_ids, ex.is_audio, packed_audio)

    packed_logits = model(inputs_embeds=emb, attention_mask=mask, position_ids=batch.position_ids).logits

    # Oracle: each chunk of each utterance as a standalone example.
    for i, chs in enumerate(utts):
        separate = build_separate_chunk_examples(instruction, chs, VS, VE, EOT)
        for k, (sep, frames) in enumerate(zip(separate, audio_by_utt[i]), start=1):
            sep_emb = _embed_with_audio(model, sep.input_ids, sep.is_audio, frames)
            sep_logits = model(inputs_embeds=sep_emb[None]).logits[0]
            idx = (batch.seg_ids[i] == k).nonzero(as_tuple=True)[0]
            torch.testing.assert_close(
                packed_logits[i][idx], sep_logits[sep.branch_start :], atol=1e-4, rtol=1e-4
            )


@torch.no_grad()
def test_stream_decode_matches_forced_packed():
    """Greedy streaming decode (spine KV + audio eviction) must equal the argmax
    of a teacher-forced packed forward of the emitted tokens.

    This validates the inference loop against the training layout: same audio,
    same positions, same conditioning -> identical greedy choices.
    """
    model = _tiny_qwen3()
    H = model.config.hidden_size
    embed = model.get_input_embeddings()

    instruction = [5, 6, 7]
    chunk_size = 2
    n_frames = 6  # -> 3 chunks of 2 frames
    torch.manual_seed(321)
    frames = torch.randn(n_frames, H)

    max_new = 4
    emitted = stream_decode_script(
        llm=model,
        embed_tokens=embed,
        instruction_ids=instruction,
        frames=frames,
        chunk_size=chunk_size,
        vision_start_id=VS,
        vision_end_id=VE,
        eot_id=EOT,
        max_new_tokens=max_new,
    )
    assert len(emitted) == 3  # one emitted-word list per chunk

    # Rebuild the packed example with the emitted words as targets and verify that
    # a teacher-forced packed forward's argmax reproduces exactly the greedy stream.
    chunks = [ChunkSpec(audio_len=chunk_size, target_ids=emitted[k]) for k in range(3)]
    packed = build_packed_chunk_example(instruction, chunks, VS, VE, EOT)
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    packed_emb = _embed_with_audio(model, packed.input_ids, packed.is_audio, frames)
    logits = model(
        inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]
    ).logits[0]
    pred = logits.argmax(dim=-1)

    # Per branch, the supervised positions predict [w_0, ..., w_{u-1}, eot]. The
    # greedy stream produced w_0..w_{u-1}, so the first u supervised positions must
    # match. Only when the stream terminated ON eot (u < max_new_tokens) does the
    # (u+1)-th supervised position predict eot; if it stopped at max_new_tokens we
    # skip that position (the stream never chose eot there).
    supervised = packed.target_ids != IGNORE_INDEX
    for k, words in enumerate(emitted, start=1):
        idx_k = ((packed.seg_ids == k) & supervised).nonzero(as_tuple=True)[0]
        u = len(words)
        assert pred[idx_k[:u]].tolist() == words, f"chunk {k}: stream words != packed argmax"
        if u < max_new:  # terminated on eot -> that decision must reproduce
            assert int(pred[idx_k[u]]) == EOT, f"chunk {k}: expected eot at terminating position"


@torch.no_grad()
def test_batched_flush_decode_matches_forced_packed():
    """With flush_id set, the batched decoder appends <flush> on each stream's FINAL
    chunk; its greedy output must equal the argmax of a teacher-forced packed forward
    whose final chunk is a flush ChunkSpec. Validates train/inference parity for the
    flush path (same audio, positions, conditioning -> identical greedy choices)."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    embed = model.get_input_embeddings()
    FLUSH_TID = 88

    instruction = [5, 6, 7]
    chunk_size = 2
    n_frames = 6  # -> 3 chunks; the 3rd is the final (flush) chunk
    n_chunks = 3
    torch.manual_seed(321)
    frames = torch.randn(n_frames, H)

    max_new = 4
    emitted, chunk_ids = batched_stream_decode_script(
        llm=model,
        embed_tokens=embed,
        instruction_ids_list=[instruction],
        frames_list=[frames],
        chunk_size=chunk_size,
        vision_start_id=VS,
        vision_end_id=VE,
        eot_id=EOT,
        pad_id=0,
        max_new_tokens=max_new,
        return_chunk_ids=True,
        flush_id=FLUSH_TID,
        flush_final=True,
    )
    emitted0, chunk_ids0 = emitted[0], chunk_ids[0]

    # Regroup emitted tokens by their chunk id, then rebuild the packed example with
    # the FINAL chunk marked flush (matching what the decoder fed).
    per_chunk = [[] for _ in range(n_chunks)]
    for tok, k in zip(emitted0, chunk_ids0):
        per_chunk[k].append(tok)
    chunks = [
        ChunkSpec(audio_len=chunk_size, target_ids=per_chunk[k], flush=(k == n_chunks - 1))
        for k in range(n_chunks)
    ]
    packed = build_packed_chunk_example(instruction, chunks, VS, VE, EOT, flush_id=FLUSH_TID)
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    packed_emb = _embed_with_audio(model, packed.input_ids, packed.is_audio, frames)
    logits = model(
        inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]
    ).logits[0]
    pred = logits.argmax(dim=-1)

    # The final chunk's branch must contain the <flush> token as an input.
    b_final = (packed.seg_ids == n_chunks).nonzero(as_tuple=True)[0]
    assert FLUSH_TID in packed.input_ids[b_final].tolist()

    supervised = packed.target_ids != IGNORE_INDEX
    for k in range(n_chunks):
        words = per_chunk[k]
        idx_k = ((packed.seg_ids == k + 1) & supervised).nonzero(as_tuple=True)[0]
        u = len(words)
        assert pred[idx_k[:u]].tolist() == words, f"chunk {k}: stream words != packed argmax"
        if u < max_new:
            assert int(pred[idx_k[u]]) == EOT, f"chunk {k}: expected eot at terminating position"


@torch.no_grad()
def test_batched_flush_decode_pads_partial_final_chunk_to_training_length():
    """Regression: when T_enc is NOT a multiple of chunk_size the FINAL chunk is
    partial. Training always feeds ``audio_tag * chunk_size`` per chunk and
    zero-pads the frames past the real audio (win_end > T_enc) via the gather in
    the packed forward, so the final branch has a full ``chunk_size`` audio window
    ending in trailing silence. The batched decoder must reproduce that by
    zero-padding the partial final chunk; otherwise it feeds fewer audio tokens
    (and loses the end-of-audio cue), stranding delay-held tail words at high
    delay. Verified by matching a teacher-forced packed forward whose final chunk
    is a flush ChunkSpec (frames zero-padded to the full window)."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    embed = model.get_input_embeddings()
    FLUSH_TID = 88

    instruction = [5, 6, 7]
    chunk_size = 2
    n_frames = 5  # NOT a multiple of 2 -> 3 chunks, final chunk is partial (1 frame)
    n_chunks = 3
    torch.manual_seed(321)
    frames = torch.randn(n_frames, H)
    # Training's zero-padded view: the final chunk's OOB frame (index 5) is a zero
    # row, so the packed forward conditions on [f0..f4, 0] (6 audio slots).
    frames_padded = torch.cat([frames, frames.new_zeros(n_chunks * chunk_size - n_frames, H)], dim=0)

    max_new = 4
    emitted, chunk_ids = batched_stream_decode_script(
        llm=model,
        embed_tokens=embed,
        instruction_ids_list=[instruction],
        frames_list=[frames],  # raw, partial final chunk -> decoder must pad it
        chunk_size=chunk_size,
        vision_start_id=VS,
        vision_end_id=VE,
        eot_id=EOT,
        pad_id=0,
        max_new_tokens=max_new,
        return_chunk_ids=True,
        flush_id=FLUSH_TID,
        flush_final=True,
    )
    emitted0, chunk_ids0 = emitted[0], chunk_ids[0]

    per_chunk = [[] for _ in range(n_chunks)]
    for tok, k in zip(emitted0, chunk_ids0):
        per_chunk[k].append(tok)
    chunks = [
        ChunkSpec(audio_len=chunk_size, target_ids=per_chunk[k], flush=(k == n_chunks - 1))
        for k in range(n_chunks)
    ]
    packed = build_packed_chunk_example(instruction, chunks, VS, VE, EOT, flush_id=FLUSH_TID)
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    # Packed conditions on the ZERO-padded frames (matching the decoder's padding).
    packed_emb = _embed_with_audio(model, packed.input_ids, packed.is_audio, frames_padded)
    logits = model(
        inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]
    ).logits[0]
    pred = logits.argmax(dim=-1)

    supervised = packed.target_ids != IGNORE_INDEX
    for k in range(n_chunks):
        words = per_chunk[k]
        idx_k = ((packed.seg_ids == k + 1) & supervised).nonzero(as_tuple=True)[0]
        u = len(words)
        assert pred[idx_k[:u]].tolist() == words, f"chunk {k}: stream words != packed argmax (partial-chunk pad)"
        if u < max_new:
            assert int(pred[idx_k[u]]) == EOT, f"chunk {k}: expected eot at terminating position"


@torch.no_grad()
def test_stream_decode_next_chunk_frames_matches_frames_arg():
    """The streaming state-machine path (``next_chunk_frames`` callable) must
    produce byte-identical emissions to the default ``frames``-arg path.

    The callable here reconstructs each chunk's window from a GROWING buffer that
    is filled in arbitrary-sized pieces (mimicking an incremental encoder that
    returns a variable number of frames per step, then re-chunks by chunk_size),
    guaranteeing the frames->chunk mapping is identical whether audio is encoded
    offline up front or incrementally by the streaming state machine.
    """
    model = _tiny_qwen3()
    H = model.config.hidden_size
    embed = model.get_input_embeddings()

    instruction = [5, 6, 7]
    chunk_size = 2
    n_frames = 7  # 4 chunks; last is partial (1 frame)
    torch.manual_seed(321)
    frames = torch.randn(n_frames, H)

    common = dict(
        llm=model,
        embed_tokens=embed,
        instruction_ids=instruction,
        chunk_size=chunk_size,
        vision_start_id=VS,
        vision_end_id=VE,
        eot_id=EOT,
        max_new_tokens=4,
    )
    baseline = stream_decode_script(frames=frames, **common)

    # Growing buffer filled in arbitrary "encoder step" sizes summing to n_frames.
    buf: list = []
    fed = [0]
    steps = iter([1, 3, 1, 2])

    def _feed_one_step() -> bool:
        if fed[0] >= n_frames:
            return False
        try:
            step = next(steps)
        except StopIteration:
            step = chunk_size
        end = min(fed[0] + step, n_frames)
        for r in range(fed[0], end):
            buf.append(frames[r])
        fed[0] = end
        return True

    def next_chunk_frames(k: int):
        win_end = (k + 1) * chunk_size
        while len(buf) < win_end and fed[0] < n_frames:
            _feed_one_step()
        if k * chunk_size >= len(buf):
            return None
        return torch.stack(buf[k * chunk_size : win_end], dim=0)

    got = stream_decode_script(frames=None, next_chunk_frames=next_chunk_frames, device=frames.device, **common)
    assert got == baseline


@torch.no_grad()
def test_stream_decode_next_chunk_frames_windowed_matches_frames_arg():
    """Callable path parity with an audio-history window (audio_history_chunks>0),
    where a branch's window spans multiple chunks back. The callable serves the
    same window via ``_audio_window_start`` over a fully pre-filled buffer."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    embed = model.get_input_embeddings()

    instruction = [5, 6]
    chunk_size = 2
    n_frames = 8  # 4 chunks
    M = 1  # one chunk of audio history
    torch.manual_seed(99)
    frames = torch.randn(n_frames, H)

    common = dict(
        llm=model,
        embed_tokens=embed,
        instruction_ids=instruction,
        chunk_size=chunk_size,
        vision_start_id=VS,
        vision_end_id=VE,
        eot_id=EOT,
        max_new_tokens=4,
        audio_history_chunks=M,
    )
    baseline = stream_decode_script(frames=frames, **common)

    def next_chunk_frames(k: int):
        if k * chunk_size >= n_frames:
            return None
        win_end = (k + 1) * chunk_size
        win_start = _audio_window_start(k * chunk_size, win_end, max(0, k - M) * chunk_size, 0)
        return frames[win_start:win_end]

    got = stream_decode_script(frames=None, next_chunk_frames=next_chunk_frames, device=frames.device, **common)
    assert got == baseline


def _embed_by_frame_index(model, input_ids: torch.Tensor, audio_frame_index: torch.Tensor, global_frames: torch.Tensor):
    """Embed text ids and fill each audio slot by its GLOBAL frame index (gather).

    Mirrors ScriptSTTModel._build_input_embeds_indexed, used for the
    windowed (audio_history_chunks>0) parity where frames are reused across branches.
    """
    ids = input_ids.clone()
    is_audio = input_ids == AUDIO_TOKEN_IDX
    ids[is_audio] = 0
    emb = model.get_input_embeddings()(ids)  # (L, H)
    if is_audio.any():
        emb = emb.clone()
        emb[is_audio] = global_frames[audio_frame_index[is_audio]].to(emb.dtype)
    return emb


# ---------------------------------------------------------------------------
# Audio history window (audio_history_chunks > 0)
# ---------------------------------------------------------------------------


def test_windowed_frame_index_structure():
    # chunk1: 2 frames [20,21]; chunk2: 3 frames [30]; chunk3: 2 frames [40].
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(3, [30]), ChunkSpec(2, [40])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, audio_history_chunks=1)
    # Global frame layout: chunk1=frames[0,1]; chunk2=frames[2,3,4]; chunk3=frames[5,6].
    # Branch 2's window = chunks 1..2 -> global frames [0,1,2,3,4].
    b2 = (ex.seg_ids == 2).nonzero(as_tuple=True)[0]
    fidx_b2 = ex.audio_frame_index[b2][ex.is_audio[b2]].tolist()
    assert fidx_b2 == [0, 1, 2, 3, 4]
    # Branch 1 (initial) has no previous chunk -> window is just chunk 1 (no pad).
    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0]
    fidx_b1 = ex.audio_frame_index[b1][ex.is_audio[b1]].tolist()
    assert fidx_b1 == [0, 1]
    # Branch 3's window = chunks 2..3 -> global frames [2,3,4,5,6].
    b3 = (ex.seg_ids == 3).nonzero(as_tuple=True)[0]
    fidx_b3 = ex.audio_frame_index[b3][ex.is_audio[b3]].tolist()
    assert fidx_b3 == [2, 3, 4, 5, 6]
    # Non-audio positions carry -1.
    assert int(ex.audio_frame_index[ex.seg_ids == 0].max()) == -1


@torch.no_grad()
def test_parity_windowed_packed_vs_separate():
    """With an audio history window, packed branch logits must still equal the
    standalone example that uses the SAME windowed audio."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    instruction = [5, 6, 7]
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(3, [30]), ChunkSpec(2, [40, 41])]
    M = 1
    total_frames = sum(c.audio_len for c in chunks)
    torch.manual_seed(2024)
    global_frames = torch.randn(total_frames, H)

    packed = build_packed_chunk_example(instruction, chunks, VS, VE, EOT, audio_history_chunks=M)
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    packed_emb = _embed_by_frame_index(model, packed.input_ids, packed.audio_frame_index, global_frames)
    packed_logits = model(
        inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]
    ).logits[0]

    separate = build_separate_chunk_examples(instruction, chunks, VS, VE, EOT, audio_history_chunks=M)
    for k, sep in enumerate(separate, start=1):
        sep_emb = _embed_by_frame_index(model, sep.input_ids, sep.audio_frame_index, global_frames)
        sep_logits = model(inputs_embeds=sep_emb[None]).logits[0]
        idx = (packed.seg_ids == k).nonzero(as_tuple=True)[0]
        torch.testing.assert_close(packed_logits[idx], sep_logits[sep.branch_start :], atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Fixed-frame audio window (audio_window_frames > 0)
# ---------------------------------------------------------------------------


def test_fixed_frame_window_structure():
    # 4 chunks of 3 frames each; a fixed 7-frame window ending at each boundary.
    # Global frames: chunk0=[0,1,2], chunk1=[3,4,5], chunk2=[6,7,8], chunk3=[9,10,11].
    chunks = [ChunkSpec(3, [20]), ChunkSpec(3, [21]), ChunkSpec(3, [22]), ChunkSpec(3, [23])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, audio_window_frames=7)

    def win(seg):
        b = (ex.seg_ids == seg).nonzero(as_tuple=True)[0]
        return ex.audio_frame_index[b][ex.is_audio[b]].tolist()

    assert win(1) == [0, 1, 2]  # only 3 frames available (clamped at start)
    assert win(2) == [0, 1, 2, 3, 4, 5]  # 6 available (< 7)
    assert win(3) == [2, 3, 4, 5, 6, 7, 8]  # exactly 7, ending at frame 9
    assert win(4) == [5, 6, 7, 8, 9, 10, 11]  # exactly 7, ending at frame 12


def test_fixed_frame_window_keeps_large_chunk():
    # A chunk LARGER than the window keeps ALL its frames (the frame count is a floor).
    chunks = [ChunkSpec(10, [20, 21])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, audio_window_frames=7)
    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0]
    assert ex.audio_frame_index[b1][ex.is_audio[b1]].tolist() == list(range(10))


@torch.no_grad()
def test_parity_fixed_frame_window_packed_vs_separate():
    """Packed branch logits must match the standalone example when the audio
    window is sized by a fixed frame count (audio_window_frames)."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    instruction = [5, 6, 7]
    chunks = [ChunkSpec(3, [20, 21]), ChunkSpec(3, [30]), ChunkSpec(3, [40, 41]), ChunkSpec(3, [50])]
    W = 7
    total_frames = sum(c.audio_len for c in chunks)
    torch.manual_seed(2025)
    global_frames = torch.randn(total_frames, H)

    packed = build_packed_chunk_example(instruction, chunks, VS, VE, EOT, audio_window_frames=W)
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    packed_emb = _embed_by_frame_index(model, packed.input_ids, packed.audio_frame_index, global_frames)
    packed_logits = model(
        inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]
    ).logits[0]

    separate = build_separate_chunk_examples(instruction, chunks, VS, VE, EOT, audio_window_frames=W)
    for k, sep in enumerate(separate, start=1):
        sep_emb = _embed_by_frame_index(model, sep.input_ids, sep.audio_frame_index, global_frames)
        sep_logits = model(inputs_embeds=sep_emb[None]).logits[0]
        idx = (packed.seg_ids == k).nonzero(as_tuple=True)[0]
        torch.testing.assert_close(packed_logits[idx], sep_logits[sep.branch_start :], atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_batched_decode_matches_per_utterance_fixed_frame():
    """Batched == per-utterance decode with a fixed-frame audio window."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    embed = model.get_input_embeddings()
    pad_id = 0
    chunk_size = 3
    max_new = 4
    W = 7

    torch.manual_seed(444)
    instrs = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]
    frames_list = [torch.randn(9, H), torch.randn(3, H), torch.randn(12, H)]

    ref = []
    for instr, frames in zip(instrs, frames_list):
        per_chunk = stream_decode_script(
            llm=model, embed_tokens=embed, instruction_ids=instr, frames=frames,
            chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT,
            max_new_tokens=max_new, audio_window_frames=W,
        )
        ref.append([t for chunk in per_chunk for t in chunk])

    got = batched_stream_decode_script(
        llm=model, embed_tokens=embed, instruction_ids_list=instrs, frames_list=frames_list,
        chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=pad_id,
        max_new_tokens=max_new, audio_window_frames=W,
    )
    assert got == ref, f"fixed-frame batched decode diverged:\n batched={got}\n per-utt={ref}"


# ---------------------------------------------------------------------------
# Delay-sized LEFT audio context (audio_left_context_frames > 0)
#
# Each branch window is extended left by ``audio_left_context_frames`` frames of
# pre-chunk history: window = [k*cs - left, (k+1)*cs) = cs + left frames (clamped
# at 0), so a word held back by the emission delay keeps its own audio in the
# chunk that emits it.
# ---------------------------------------------------------------------------


def test_left_context_window_helper():
    # Pure-function check of _audio_window_start's left extension (M=0, W=0).
    cs, left = 3, 2
    # chunk k=1: base start = 3, extended left by 2 -> 1.
    assert _audio_window_start(1 * cs, 2 * cs, 1 * cs, 0, left) == 1
    # chunk k=0 clamps at 0 (no negative frames).
    assert _audio_window_start(0, cs, 0, 0, left) == 0
    # left=0 is a no-op (unchanged base behaviour).
    assert _audio_window_start(2 * cs, 3 * cs, 2 * cs, 0, 0) == 2 * cs


def test_left_context_window_structure():
    # 4 chunks of 3 frames each; prepend left=2 frames of pre-chunk history.
    # Global frames: chunk0=[0,1,2], chunk1=[3,4,5], chunk2=[6,7,8], chunk3=[9,10,11].
    chunks = [ChunkSpec(3, [20]), ChunkSpec(3, [21]), ChunkSpec(3, [22]), ChunkSpec(3, [23])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, audio_left_context_frames=2)

    def win(seg):
        b = (ex.seg_ids == seg).nonzero(as_tuple=True)[0]
        return ex.audio_frame_index[b][ex.is_audio[b]].tolist()

    assert win(1) == [0, 1, 2]  # clamped at start (no negative frames)
    assert win(2) == [1, 2, 3, 4, 5]  # 2 left frames + own 3
    assert win(3) == [4, 5, 6, 7, 8]
    assert win(4) == [7, 8, 9, 10, 11]


def test_left_context_scales_with_chunk_size():
    # The left slab is a FIXED frame count; total context = chunk_size + left.
    chunks = [ChunkSpec(5, [20]), ChunkSpec(5, [21])]  # cs=5
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, audio_left_context_frames=3)
    b2 = (ex.seg_ids == 2).nonzero(as_tuple=True)[0]
    # chunk1 = frames [5,6,7,8,9]; extended left by 3 -> starts at frame 2 => 8 frames total.
    assert ex.audio_frame_index[b2][ex.is_audio[b2]].tolist() == [2, 3, 4, 5, 6, 7, 8, 9]


@torch.no_grad()
def test_parity_left_context_packed_vs_separate():
    """Packed branch logits must match the standalone example when each window is
    extended left by ``audio_left_context_frames`` frames."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    instruction = [5, 6, 7]
    chunks = [ChunkSpec(3, [20, 21]), ChunkSpec(3, [30]), ChunkSpec(3, [40, 41]), ChunkSpec(3, [50])]
    left = 4
    total_frames = sum(c.audio_len for c in chunks)
    torch.manual_seed(2026)
    global_frames = torch.randn(total_frames, H)

    packed = build_packed_chunk_example(instruction, chunks, VS, VE, EOT, audio_left_context_frames=left)
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    packed_emb = _embed_by_frame_index(model, packed.input_ids, packed.audio_frame_index, global_frames)
    packed_logits = model(
        inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]
    ).logits[0]

    separate = build_separate_chunk_examples(instruction, chunks, VS, VE, EOT, audio_left_context_frames=left)
    for k, sep in enumerate(separate, start=1):
        sep_emb = _embed_by_frame_index(model, sep.input_ids, sep.audio_frame_index, global_frames)
        sep_logits = model(inputs_embeds=sep_emb[None]).logits[0]
        idx = (packed.seg_ids == k).nonzero(as_tuple=True)[0]
        torch.testing.assert_close(packed_logits[idx], sep_logits[sep.branch_start :], atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_batched_decode_matches_per_utterance_left_context():
    """Batched == per-utterance decode with a delay-sized left audio context."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    embed = model.get_input_embeddings()
    pad_id = 0
    chunk_size = 3
    max_new = 4
    left = 4

    torch.manual_seed(555)
    instrs = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]
    frames_list = [torch.randn(9, H), torch.randn(3, H), torch.randn(12, H)]

    ref = []
    for instr, frames in zip(instrs, frames_list):
        per_chunk = stream_decode_script(
            llm=model, embed_tokens=embed, instruction_ids=instr, frames=frames,
            chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT,
            max_new_tokens=max_new, audio_left_context_frames=left,
        )
        ref.append([t for chunk in per_chunk for t in chunk])

    got = batched_stream_decode_script(
        llm=model, embed_tokens=embed, instruction_ids_list=instrs, frames_list=frames_list,
        chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=pad_id,
        max_new_tokens=max_new, audio_left_context_frames=left,
    )
    assert got == ref, f"left-context batched decode diverged:\n batched={got}\n per-utt={ref}"


# ---------------------------------------------------------------------------
# History-word recovery
# ---------------------------------------------------------------------------


def test_recovery_layout():
    # chunk1 "one two" (last word "two"=[21]); chunk2 "three"=[30].
    chunks = [ChunkSpec(2, [20, 21], last_word_ids=[21]), ChunkSpec(2, [30], last_word_ids=[30])]
    ex = build_packed_chunk_example(
        INSTR, chunks, VS, VE, EOT, audio_history_chunks=1, recover_prev=[False, True]
    )
    # Spine = INSTR(2) + [20,21] + [30]; prefix for chunk2 normally = 2+2 = 4.
    # Recovery drops chunk1's last word (1 token) -> branch2 prefix_len = 3.
    b2 = (ex.seg_ids == 2).nonzero(as_tuple=True)[0]
    assert (ex.prefix_len[b2] == 3).all()
    # Branch2 target predicts [recovered "two"(21), "three"(30), eot].
    supervised = ex.target_ids != IGNORE_INDEX
    b2_targets = ex.target_ids[b2][supervised[b2]].tolist()
    assert b2_targets == [21, 30, EOT]
    # Branch1 is untouched (prefix 2, predicts "one two" + eot).
    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0]
    assert (ex.prefix_len[b1] == 2).all()
    assert ex.target_ids[b1][supervised[b1]].tolist() == [20, 21, EOT]


@torch.no_grad()
def test_parity_windowed_recovery_packed_vs_separate():
    model = _tiny_qwen3()
    H = model.config.hidden_size
    instruction = [5, 6]
    chunks = [ChunkSpec(2, [20, 21], last_word_ids=[21]), ChunkSpec(3, [30, 31], last_word_ids=[31])]
    recover = [False, True]
    M = 1
    total_frames = sum(c.audio_len for c in chunks)
    torch.manual_seed(77)
    global_frames = torch.randn(total_frames, H)

    packed = build_packed_chunk_example(
        instruction, chunks, VS, VE, EOT, audio_history_chunks=M, recover_prev=recover
    )
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    packed_emb = _embed_by_frame_index(model, packed.input_ids, packed.audio_frame_index, global_frames)
    packed_logits = model(
        inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]
    ).logits[0]

    separate = build_separate_chunk_examples(
        instruction, chunks, VS, VE, EOT, audio_history_chunks=M, recover_prev=recover
    )
    for k, sep in enumerate(separate, start=1):
        sep_emb = _embed_by_frame_index(model, sep.input_ids, sep.audio_frame_index, global_frames)
        sep_logits = model(inputs_embeds=sep_emb[None]).logits[0]
        idx = (packed.seg_ids == k).nonzero(as_tuple=True)[0]
        torch.testing.assert_close(packed_logits[idx], sep_logits[sep.branch_start :], atol=1e-4, rtol=1e-4)


def test_recovery_no_window_layout():
    """Recovery with NO audio window (audio_history_chunks=0): the recovering
    branch drops the previous chunk's last word from history and prepends it to
    its target, but its audio is ONLY its own chunk (no prior-chunk frames)."""
    # chunk1 "one two" (last word "two"=[21]); chunk2 "three"=[30].
    chunks = [ChunkSpec(2, [20, 21], last_word_ids=[21]), ChunkSpec(2, [30], last_word_ids=[30])]
    ex = build_packed_chunk_example(
        INSTR, chunks, VS, VE, EOT, audio_history_chunks=0, recover_prev=[False, True]
    )
    b2 = (ex.seg_ids == 2).nonzero(as_tuple=True)[0]
    # prefix for chunk2 normally = INSTR(2)+[20,21] = 4; recovery drops "two"(1) -> 3.
    assert (ex.prefix_len[b2] == 3).all()
    # audio window is ONLY chunk2's 2 frames (no previous chunk pulled in).
    assert int(ex.is_audio[b2].sum()) == 2
    # target predicts [recovered "two"(21), "three"(30), eot].
    supervised = ex.target_ids != IGNORE_INDEX
    assert ex.target_ids[b2][supervised[b2]].tolist() == [21, 30, EOT]
    # branch1 untouched.
    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0]
    assert (ex.prefix_len[b1] == 2).all()
    assert ex.target_ids[b1][supervised[b1]].tolist() == [20, 21, EOT]


@torch.no_grad()
def test_parity_recovery_no_window_packed_vs_separate():
    """Recovery with audio_history_chunks=0 must still match the standalone example
    (which drops the same word and sees only its own chunk's audio)."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    instruction = [5, 6]
    chunks = [ChunkSpec(2, [20, 21], last_word_ids=[21]), ChunkSpec(3, [30, 31], last_word_ids=[31])]
    recover = [False, True]
    M = 0
    total_frames = sum(c.audio_len for c in chunks)
    torch.manual_seed(88)
    global_frames = torch.randn(total_frames, H)

    packed = build_packed_chunk_example(
        instruction, chunks, VS, VE, EOT, audio_history_chunks=M, recover_prev=recover
    )
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    packed_emb = _embed_by_frame_index(model, packed.input_ids, packed.audio_frame_index, global_frames)
    packed_logits = model(
        inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]
    ).logits[0]

    separate = build_separate_chunk_examples(
        instruction, chunks, VS, VE, EOT, audio_history_chunks=M, recover_prev=recover
    )
    for k, sep in enumerate(separate, start=1):
        sep_emb = _embed_by_frame_index(model, sep.input_ids, sep.audio_frame_index, global_frames)
        sep_logits = model(inputs_embeds=sep_emb[None]).logits[0]
        idx = (packed.seg_ids == k).nonzero(as_tuple=True)[0]
        torch.testing.assert_close(packed_logits[idx], sep_logits[sep.branch_start :], atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Self-correction (delete-last-word)
# ---------------------------------------------------------------------------

DEL = 93  # toy "delete last word" token id


def test_self_correction_layout():
    # chunk1 "one two" (last word "two"=[21]); chunk2 "three"=[30]. The model
    # mis-committed "two" as W'=[99]; branch2 must delete it and re-emit 21.
    chunks = [ChunkSpec(2, [20, 21], last_word_ids=[21]), ChunkSpec(2, [30], last_word_ids=[30])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, corrupt_prev=[None, [99]], delete_id=DEL)

    b2 = (ex.seg_ids == 2).nonzero(as_tuple=True)[0]
    # [99] <vs> A A <ve> <DEL> 21 30 <eot>
    assert ex.input_ids[b2].tolist() == [99, VS, AUDIO_TOKEN_IDX, AUDIO_TOKEN_IDX, VE, DEL, 21, 30, EOT]
    assert ex.is_audio[b2].tolist() == [False, False, True, True, False, False, False, False, False]
    # prefix excludes the correct "21" (spine pos 3): branch2 attends spine 0,1,2 only.
    assert (ex.prefix_len[b2] == 3).all()
    # W' at the history-tail position (3), then the rest contiguous.
    assert ex.position_ids[b2].tolist() == [3, 4, 5, 6, 7, 8, 9, 10, 11]
    # target: delete the wrong word, re-emit "two"(21), then "three"(30), then eot.
    supervised = ex.target_ids != IGNORE_INDEX
    assert ex.target_ids[b2][supervised[b2]].tolist() == [DEL, 21, 30, EOT]

    # branch 1 (no corruption) is untouched.
    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0]
    assert ex.input_ids[b1].tolist() == [VS, AUDIO_TOKEN_IDX, AUDIO_TOKEN_IDX, VE, 20, 21, EOT]
    assert (ex.prefix_len[b1] == 2).all()


def test_self_correction_chunk_scope_layout():
    # Whole-chunk scope: chunk1 "one two"=[20,21]; chunk2 "three"=[30]. The model
    # mis-committed the WHOLE chunk1 as W'=[98,99]; branch2 must delete the whole
    # previous chunk and re-emit both true tokens [20,21], then chunk2's [30].
    chunks = [ChunkSpec(2, [20, 21], last_word_ids=[21]), ChunkSpec(2, [30], last_word_ids=[30])]
    ex = build_packed_chunk_example(
        INSTR, chunks, VS, VE, EOT, corrupt_prev=[None, [98, 99]], delete_id=DEL, correction_scope="chunk"
    )

    b2 = (ex.seg_ids == 2).nonzero(as_tuple=True)[0]
    # [98 99] <vs> A A <ve> <DEL> 20 21 30 <eot>
    assert ex.input_ids[b2].tolist() == [98, 99, VS, AUDIO_TOKEN_IDX, AUDIO_TOKEN_IDX, VE, DEL, 20, 21, 30, EOT]
    # prefix excludes the WHOLE correct chunk1 (spine positions 2,3): branch2
    # attends only the instruction (spine positions 0,1).
    assert (ex.prefix_len[b2] == 2).all()
    # W' (2 tokens) sits on the history-tail positions (2,3), then the rest contiguous.
    assert ex.position_ids[b2].tolist() == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # target: delete, re-emit the whole true chunk1 [20,21], then chunk2 [30], then eot.
    supervised = ex.target_ids != IGNORE_INDEX
    assert ex.target_ids[b2][supervised[b2]].tolist() == [DEL, 20, 21, 30, EOT]


@torch.no_grad()
def test_delete_aware_decode_pops_last_chunk():
    """With correction_scope='chunk', a leading <del> pops ALL tokens committed in
    the previous chunk (not just its last word)."""
    A, Bt = 20, 21
    V, H = 128, 8
    # chunk0 prefill emits A then B then eot (a 2-token chunk); chunk1 emits
    # <del> (pop the whole previous chunk) then C, eot.
    C = 22
    llm = _ScriptedLLM([A, Bt, EOT, DEL, C, EOT], V)
    embed = lambda ids: torch.zeros(*ids.shape, H)  # noqa: E731
    frames = torch.zeros(4, H)  # 2 chunks of size 2
    emitted = batched_stream_decode_script(
        llm=llm, embed_tokens=embed, instruction_ids_list=[[5, 6]], frames_list=[frames],
        chunk_size=2, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=0,
        max_new_tokens=4, delete_id=DEL, is_word_start=lambda t: True, correction_scope="chunk",
    )
    # chunk0 commits [A, B]; chunk1's leading <del> pops the WHOLE chunk -> [], then C.
    assert emitted == [[C]]


@torch.no_grad()
def test_parity_self_correction_chunk_scope_packed_vs_separate():
    """Whole-chunk self-correction: packed branch logits must equal the standalone
    example with the whole wrong chunk W' as history tail and target <DEL> w_prev_chunk w_k."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    instruction = [5, 6, 7]
    chunks = [
        ChunkSpec(2, [20, 21], last_word_ids=[21]),
        ChunkSpec(3, [30, 31], last_word_ids=[31]),
        ChunkSpec(2, [40], last_word_ids=[40]),
    ]
    corrupt = [None, [98, 99], None]  # chunk2 corrects the whole mis-committed chunk1 -> [98,99]
    torch.manual_seed(7)
    audio_frames = [torch.randn(c.audio_len, H) for c in chunks]

    packed = build_packed_chunk_example(
        instruction, chunks, VS, VE, EOT, corrupt_prev=corrupt, delete_id=DEL, correction_scope="chunk"
    )
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    packed_emb = _embed_with_audio(model, packed.input_ids, packed.is_audio, torch.cat(audio_frames, dim=0))
    packed_logits = model(
        inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]
    ).logits[0]

    separate = build_separate_chunk_examples(
        instruction, chunks, VS, VE, EOT, corrupt_prev=corrupt, delete_id=DEL, correction_scope="chunk"
    )
    for k, (sep, frames) in enumerate(zip(separate, audio_frames), start=1):
        sep_emb = _embed_with_audio(model, sep.input_ids, sep.is_audio, frames)
        sep_logits = model(inputs_embeds=sep_emb[None], position_ids=sep.position_ids[None]).logits[0]
        idx = (packed.seg_ids == k).nonzero(as_tuple=True)[0]
        torch.testing.assert_close(packed_logits[idx], sep_logits[sep.branch_start :], atol=1e-4, rtol=1e-4)


class _ScriptedLLM:
    """A stand-in LLM whose argmax follows a fixed per-call token script, so we can
    drive the decoder deterministically (used to test <del> word-popping)."""

    def __init__(self, script, vocab_size):
        self.script = list(script)
        self.i = 0
        self.V = vocab_size

    def __call__(self, inputs_embeds, attention_mask=None, position_ids=None,
                 past_key_values=None, use_cache=False, return_dict=True):
        na, L, _ = inputs_embeds.shape
        logits = torch.zeros(na, L, self.V)
        tok = self.script[self.i]
        self.i += 1
        logits[:, -1, tok] = 100.0  # force argmax at the last position

        class _Out:
            pass

        o = _Out()
        o.logits = logits
        o.past_key_values = past_key_values if past_key_values is not None else object()
        return o


@torch.no_grad()
def test_delete_aware_decode_pops_last_word():
    """A leading <del> in a chunk pops the previous chunk's last committed word."""
    A = 20
    V, H = 128, 8
    # scripted argmax per llm call: chunk0 -> [A (prefill), eot]; chunk1 -> [<del>, A, eot]
    llm = _ScriptedLLM([A, EOT, DEL, A, EOT], V)
    embed = lambda ids: torch.zeros(*ids.shape, H)  # noqa: E731
    frames = torch.zeros(4, H)  # 2 chunks of size 2
    emitted = batched_stream_decode_script(
        llm=llm, embed_tokens=embed, instruction_ids_list=[[5, 6]], frames_list=[frames],
        chunk_size=2, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=0,
        max_new_tokens=4, delete_id=DEL, is_word_start=lambda t: True,
    )
    # chunk0 commits [A]; chunk1's leading <del> pops it, then re-emits A -> [A].
    assert emitted == [[A]]


@torch.no_grad()
def test_delete_aware_decode_returns_raw_stream():
    """return_raw yields the literal emission stream WITH <del> kept, while the
    corrected `emitted` has the popped word removed."""
    A = 20
    V, H = 128, 8
    llm = _ScriptedLLM([A, EOT, DEL, A, EOT], V)
    embed = lambda ids: torch.zeros(*ids.shape, H)  # noqa: E731
    frames = torch.zeros(4, H)  # 2 chunks of size 2
    emitted, raw = batched_stream_decode_script(
        llm=llm, embed_tokens=embed, instruction_ids_list=[[5, 6]], frames_list=[frames],
        chunk_size=2, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=0,
        max_new_tokens=4, delete_id=DEL, is_word_start=lambda t: True, return_raw=True,
    )
    assert emitted == [[A]]  # corrected: <del> popped the first A, re-emitted A
    assert raw == [[A, DEL, A]]  # raw: chunk0 "A", chunk1 "<del> A"


@torch.no_grad()
def test_delete_id_none_is_unchanged():
    """Without delete_id the decode is byte-identical to the original behavior."""
    A, Bt = 20, 21
    V, H = 128, 8
    llm = _ScriptedLLM([A, EOT, Bt, EOT], V)
    embed = lambda ids: torch.zeros(*ids.shape, H)  # noqa: E731
    frames = torch.zeros(4, H)
    emitted = batched_stream_decode_script(
        llm=llm, embed_tokens=embed, instruction_ids_list=[[5, 6]], frames_list=[frames],
        chunk_size=2, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=0,
        max_new_tokens=4,
    )
    assert emitted == [[A, Bt]]


@torch.no_grad()
def test_parity_self_correction_packed_vs_separate():
    """A self-correcting branch's packed logits must equal the standalone example
    with the same wrong word W' shown as history tail and target <DEL> w_prev w_k."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    instruction = [5, 6, 7]
    chunks = [
        ChunkSpec(2, [20, 21], last_word_ids=[21]),
        ChunkSpec(3, [30, 31], last_word_ids=[31]),
        ChunkSpec(2, [40], last_word_ids=[40]),
    ]
    corrupt = [None, [99], None]  # chunk2 corrupts the prev last word "21" -> 99
    torch.manual_seed(5)
    audio_frames = [torch.randn(c.audio_len, H) for c in chunks]

    packed = build_packed_chunk_example(instruction, chunks, VS, VE, EOT, corrupt_prev=corrupt, delete_id=DEL)
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    packed_emb = _embed_with_audio(model, packed.input_ids, packed.is_audio, torch.cat(audio_frames, dim=0))
    packed_logits = model(
        inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]
    ).logits[0]

    separate = build_separate_chunk_examples(instruction, chunks, VS, VE, EOT, corrupt_prev=corrupt, delete_id=DEL)
    for k, (sep, frames) in enumerate(zip(separate, audio_frames), start=1):
        sep_emb = _embed_with_audio(model, sep.input_ids, sep.is_audio, frames)
        sep_logits = model(inputs_embeds=sep_emb[None], position_ids=sep.position_ids[None]).logits[0]
        idx = (packed.seg_ids == k).nonzero(as_tuple=True)[0]
        torch.testing.assert_close(packed_logits[idx], sep_logits[sep.branch_start :], atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_parity_self_correction_windowed_packed_vs_separate():
    """Self-correction + a 1-chunk audio history window (M=1, i.e. 2 chunks per
    branch): the corrected branch now also sees the previous chunk's audio (where the
    mis-committed word was spoken). Packed logits must still match the standalone."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    instruction = [5, 6, 7]
    chunks = [
        ChunkSpec(2, [20, 21], last_word_ids=[21]),
        ChunkSpec(3, [30, 31], last_word_ids=[31]),
        ChunkSpec(2, [40], last_word_ids=[40]),
    ]
    corrupt = [None, [99], None]  # chunk2 corrupts the prev last word "21" -> 99
    M = 1
    total_frames = sum(c.audio_len for c in chunks)
    torch.manual_seed(11)
    global_frames = torch.randn(total_frames, H)

    packed = build_packed_chunk_example(
        instruction, chunks, VS, VE, EOT, audio_history_chunks=M, corrupt_prev=corrupt, delete_id=DEL
    )
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    packed_emb = _embed_by_frame_index(model, packed.input_ids, packed.audio_frame_index, global_frames)
    packed_logits = model(
        inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]
    ).logits[0]

    separate = build_separate_chunk_examples(
        instruction, chunks, VS, VE, EOT, audio_history_chunks=M, corrupt_prev=corrupt, delete_id=DEL
    )
    for k, sep in enumerate(separate, start=1):
        sep_emb = _embed_by_frame_index(model, sep.input_ids, sep.audio_frame_index, global_frames)
        sep_logits = model(inputs_embeds=sep_emb[None], position_ids=sep.position_ids[None]).logits[0]
        idx = (packed.seg_ids == k).nonzero(as_tuple=True)[0]
        torch.testing.assert_close(packed_logits[idx], sep_logits[sep.branch_start :], atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_batched_decode_matches_per_utterance():
    """Batched chunk-synchronous decode must equal per-utterance stream decode.

    Different utterances have different #chunks, chunk lengths, and (greedy) word
    counts, exercising left-padding, per-stream position_ids, and finished-stream
    masking. Left-padding + per-row cumsum positions must reproduce each stream's
    standalone greedy output exactly.
    """
    model = _tiny_qwen3()
    H = model.config.hidden_size
    embed = model.get_input_embeddings()
    pad_id = 0
    chunk_size = 2
    max_new = 4

    torch.manual_seed(999)
    instrs = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]
    # Different total frame counts -> different #chunks (3, 1, 4 chunks).
    frames_list = [torch.randn(6, H), torch.randn(2, H), torch.randn(7, H)]

    # Per-utterance reference.
    ref = []
    for instr, frames in zip(instrs, frames_list):
        per_chunk = stream_decode_script(
            llm=model, embed_tokens=embed, instruction_ids=instr, frames=frames,
            chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT,
            max_new_tokens=max_new,
        )
        ref.append([t for chunk in per_chunk for t in chunk])

    # Batched.
    got = batched_stream_decode_script(
        llm=model, embed_tokens=embed, instruction_ids_list=instrs, frames_list=frames_list,
        chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=pad_id,
        max_new_tokens=max_new,
    )

    assert got == ref, f"batched decode diverged from per-utterance:\n batched={got}\n per-utt={ref}"


@torch.no_grad()
def test_batched_decode_matches_per_utterance_windowed():
    """Same batched==per-utterance equivalence, but with an audio history window
    (audio_history_chunks=1) so each branch's audio spans two chunks."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    embed = model.get_input_embeddings()
    pad_id = 0
    chunk_size = 2
    max_new = 4
    M = 1

    torch.manual_seed(555)
    instrs = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]
    frames_list = [torch.randn(6, H), torch.randn(2, H), torch.randn(7, H)]

    ref = []
    for instr, frames in zip(instrs, frames_list):
        per_chunk = stream_decode_script(
            llm=model, embed_tokens=embed, instruction_ids=instr, frames=frames,
            chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT,
            max_new_tokens=max_new, audio_history_chunks=M,
        )
        ref.append([t for chunk in per_chunk for t in chunk])

    got = batched_stream_decode_script(
        llm=model, embed_tokens=embed, instruction_ids_list=instrs, frames_list=frames_list,
        chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=pad_id,
        max_new_tokens=max_new, audio_history_chunks=M,
    )
    assert got == ref, f"windowed batched decode diverged:\n batched={got}\n per-utt={ref}"


# ---------------------------------------------------------------------------
# Contiguous-text positions ("Option A"): words placed contiguous with the
# history, audio prelude overlaid on the history's tail positions.
# ---------------------------------------------------------------------------

# A 6-token instruction leaves room so early-chunk preludes don't clamp.
LONG_INSTR = [10, 11, 12, 13, 14, 15]


def test_contiguous_positions_layout():
    chunks = [ChunkSpec(audio_len=2, target_ids=[20, 21]), ChunkSpec(audio_len=3, target_ids=[30])]
    ex = build_packed_chunk_example(LONG_INSTR, chunks, VS, VE, EOT, contiguous_text_positions=True)

    # Spine is pure text and unchanged: INSTR(6) + [20,21] + [30] at positions 0..8.
    assert ex.spine_len == 9
    assert ex.position_ids[:9].tolist() == list(range(9))

    # Branch 1: <vs> A A <ve> 20 21 <eot>, pref=6.
    #   prelude (<vs>,A,A,<ve>) overlaid at [2,3,4,5]; words+eot contiguous at [6,7,8].
    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0]
    assert ex.input_ids[b1].tolist() == [VS, AUDIO_TOKEN_IDX, AUDIO_TOKEN_IDX, VE, 20, 21, EOT]
    assert ex.position_ids[b1].tolist() == [2, 3, 4, 5, 6, 7, 8]
    # word "20" sits one past the history's last position (5) AND equals its spine twin.
    assert int(ex.position_ids[b1][4]) == 6
    assert int(ex.position_ids[6]) == 6  # spine "20" is also at position 6

    # Branch 2: <vs> A A A <ve> 30 <eot>, pref=8 -> prelude [3,4,5,6,7], words [8,9].
    b2 = (ex.seg_ids == 2).nonzero(as_tuple=True)[0]
    assert ex.position_ids[b2].tolist() == [3, 4, 5, 6, 7, 8, 9]
    assert int(ex.position_ids[b2][5]) == 8  # word "30" == its spine twin position (8)


def test_contiguous_positions_clamp_short_history_is_nonnegative():
    # First chunk with a big audio window vs a 2-token instruction: the prelude
    # would need positions < 0, so it clamps at 0 (benign) and never goes negative.
    chunks = [ChunkSpec(audio_len=5, target_ids=[20])]
    ex = build_packed_chunk_example([10, 11], chunks, VS, VE, EOT, contiguous_text_positions=True)
    b1 = (ex.seg_ids == 1).nonzero(as_tuple=True)[0]
    # pref=2, window_len=5 -> prelude=[0,0,0,0,0,0,1], words+eot=[2,3].
    assert ex.position_ids[b1].tolist() == [0, 0, 0, 0, 0, 0, 1, 2, 3]
    assert int(ex.position_ids[b1].min()) >= 0
    # Words are still contiguous with the history (first word at pref=2).
    assert int(ex.position_ids[b1][7]) == 2


@torch.no_grad()
def test_parity_contiguous_packed_vs_separate():
    """Contiguous-position packed branch logits must equal the standalone example
    run with the SAME (overlaid) position_ids. Instruction is long enough that no
    prelude clamps, so position-causal (packed) == order-causal (standalone)."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    instruction = [5, 6, 7, 8, 9, 1, 2, 3]  # len 8
    chunks = [
        ChunkSpec(audio_len=2, target_ids=[20, 21]),
        ChunkSpec(audio_len=3, target_ids=[30]),
        ChunkSpec(audio_len=1, target_ids=[]),  # silent chunk
        ChunkSpec(audio_len=2, target_ids=[40, 41, 42]),
    ]
    torch.manual_seed(123)
    audio_frames = [torch.randn(ch.audio_len, H) for ch in chunks]

    packed = build_packed_chunk_example(instruction, chunks, VS, VE, EOT, contiguous_text_positions=True)
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    packed_audio = torch.cat(audio_frames, dim=0)
    packed_emb = _embed_with_audio(model, packed.input_ids, packed.is_audio, packed_audio)
    packed_logits = model(
        inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]
    ).logits[0]

    separate = build_separate_chunk_examples(instruction, chunks, VS, VE, EOT, contiguous_text_positions=True)
    for k, (sep, frames) in enumerate(zip(separate, audio_frames), start=1):
        sep_emb = _embed_with_audio(model, sep.input_ids, sep.is_audio, frames)
        # Must pass the overlaid positions (NOT the HF default 0..L-1).
        sep_logits = model(inputs_embeds=sep_emb[None], position_ids=sep.position_ids[None]).logits[0]
        idx = (packed.seg_ids == k).nonzero(as_tuple=True)[0]
        torch.testing.assert_close(packed_logits[idx], sep_logits[sep.branch_start :], atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_parity_windowed_contiguous_packed_vs_separate():
    """Contiguous positions + an audio history window together."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    instruction = [5, 6, 7, 8, 9, 1, 2, 3]  # len 8 (no clamp for these windows)
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(3, [30]), ChunkSpec(2, [40, 41])]
    M = 1
    total_frames = sum(c.audio_len for c in chunks)
    torch.manual_seed(2024)
    global_frames = torch.randn(total_frames, H)

    packed = build_packed_chunk_example(
        instruction, chunks, VS, VE, EOT, audio_history_chunks=M, contiguous_text_positions=True
    )
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    packed_emb = _embed_by_frame_index(model, packed.input_ids, packed.audio_frame_index, global_frames)
    packed_logits = model(
        inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]
    ).logits[0]

    separate = build_separate_chunk_examples(
        instruction, chunks, VS, VE, EOT, audio_history_chunks=M, contiguous_text_positions=True
    )
    for k, sep in enumerate(separate, start=1):
        sep_emb = _embed_by_frame_index(model, sep.input_ids, sep.audio_frame_index, global_frames)
        sep_logits = model(inputs_embeds=sep_emb[None], position_ids=sep.position_ids[None]).logits[0]
        idx = (packed.seg_ids == k).nonzero(as_tuple=True)[0]
        torch.testing.assert_close(packed_logits[idx], sep_logits[sep.branch_start :], atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_stream_decode_contiguous_matches_forced_packed():
    """Greedy contiguous-position streaming decode must equal the argmax of a
    teacher-forced contiguous-position packed forward of the emitted tokens."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    embed = model.get_input_embeddings()

    instruction = [5, 6, 7, 8, 9, 1]  # len 6 -> pref >= c+2=4 always (no clamp)
    chunk_size = 2
    n_frames = 6  # 3 chunks
    torch.manual_seed(321)
    frames = torch.randn(n_frames, H)

    max_new = 4
    emitted = stream_decode_script(
        llm=model, embed_tokens=embed, instruction_ids=instruction, frames=frames,
        chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT,
        max_new_tokens=max_new, contiguous_text_positions=True,
    )
    assert len(emitted) == 3

    chunks = [ChunkSpec(audio_len=chunk_size, target_ids=emitted[k]) for k in range(3)]
    packed = build_packed_chunk_example(instruction, chunks, VS, VE, EOT, contiguous_text_positions=True)
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    packed_emb = _embed_with_audio(model, packed.input_ids, packed.is_audio, frames)
    logits = model(
        inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]
    ).logits[0]
    pred = logits.argmax(dim=-1)

    supervised = packed.target_ids != IGNORE_INDEX
    for k, words in enumerate(emitted, start=1):
        idx_k = ((packed.seg_ids == k) & supervised).nonzero(as_tuple=True)[0]
        u = len(words)
        assert pred[idx_k[:u]].tolist() == words, f"chunk {k}: stream words != packed argmax"
        if u < max_new:
            assert int(pred[idx_k[u]]) == EOT, f"chunk {k}: expected eot at terminating position"


# ---------------------------------------------------------------------------
# Last-layer restricted history (script_last_layer_history_tokens)
# ---------------------------------------------------------------------------


def _core(model):
    """(layers, norm, rotary_emb, lm_head) for a tiny Qwen3ForCausalLM."""
    return model.model.layers, model.model.norm, model.model.rotary_emb, model.lm_head


def test_history_window_large_is_noop():
    """A history_window >= the longest history reproduces the unrestricted mask."""
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(3, [30]), ChunkSpec(2, [40, 41])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT)
    valid = torch.ones_like(ex.input_ids, dtype=torch.bool)
    full = build_script_mask(ex.seg_ids[None], ex.position_ids[None], ex.prefix_len[None], valid[None], torch.float32)
    wide = build_script_mask(
        ex.seg_ids[None], ex.position_ids[None], ex.prefix_len[None], valid[None], torch.float32, history_window=1000
    )
    assert torch.equal(full, wide)


def test_history_window_blocks_old_history_for_branch_only():
    """With a small window a branch query loses OLD history tokens but keeps the
    last N + its own audio/tokens; spine queries stay fully causal."""
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(3, [30])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT)
    valid = torch.ones_like(ex.input_ids, dtype=torch.bool)
    N = 1
    mask = build_script_mask(
        ex.seg_ids[None], ex.position_ids[None], ex.prefix_len[None], valid[None], torch.float32, history_window=N
    )[0, 0]

    # Branch 2's word "30": prefix_len = 4 (INSTR + "20 21"); last N=1 history token
    # is spine pos 3 ("21"). So spine 3 allowed; spine 0,1,2 blocked.
    b2 = (ex.seg_ids == 2).nonzero(as_tuple=True)[0].tolist()
    q30 = b2[5]
    assert _allowed(mask[q30, 3])  # last history token kept
    for j in range(3):
        assert _blocked(mask[q30, j])  # older history dropped
    # Own audio + <ve> still attended.
    assert _allowed(mask[q30, b2[1]]) and _allowed(mask[q30, b2[4]])

    # Spine query stays fully causal (window only restricts BRANCH queries).
    full = build_script_mask(
        ex.seg_ids[None], ex.position_ids[None], ex.prefix_len[None], valid[None], torch.float32
    )[0, 0]
    P = ex.spine_len
    assert torch.equal(mask[:P, :P], full[:P, :P])


@torch.no_grad()
def test_run_script_layers_split_noop_equals_model():
    """Manual per-layer driver with mask_top == mask_lower == the standard forward.

    Validates the manual layer-stack driver reproduces HF's own forward exactly,
    for every choice of how many top layers are split off.
    """
    model = _tiny_qwen3()
    H = model.config.hidden_size
    instruction = [5, 6, 7]
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(3, [30]), ChunkSpec(2, [40, 41, 42])]
    torch.manual_seed(123)
    audio_frames = [torch.randn(ch.audio_len, H) for ch in chunks]

    packed = build_packed_chunk_example(instruction, chunks, VS, VE, EOT)
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    emb = _embed_with_audio(model, packed.input_ids, packed.is_audio, torch.cat(audio_frames, dim=0))[None]
    ref = model(inputs_embeds=emb, attention_mask=mask, position_ids=packed.position_ids[None]).logits

    layers, norm, rotary_emb, lm_head = _core(model)
    for k in range(1, model.config.num_hidden_layers):
        got = run_script_layers_split(
            layers=layers, norm=norm, rotary_emb=rotary_emb, lm_head=lm_head,
            inputs_embeds=emb, position_ids=packed.position_ids[None],
            mask_lower=mask, mask_top=mask, num_top_layers=k,
        )
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_last_layer_restricted_decode_matches_forced_packed():
    """Greedy last-layer-restricted decode == argmax of a teacher-forced restricted
    packed forward of the emitted tokens (train/inference parity under restriction).
    """
    model = _tiny_qwen3()
    H = model.config.hidden_size
    embed = model.get_input_embeddings()
    layers, norm, rotary_emb, lm_head = _core(model)

    instruction = [5, 6, 7]
    chunk_size = 2
    n_frames = 8  # 4 chunks
    N = 2  # keep last 2 history tokens at the top layer
    K = 1  # restrict only the final layer
    torch.manual_seed(321)
    frames = torch.randn(n_frames, H)

    max_new = 4
    emitted = batched_stream_decode_script_last_layer(
        layers=layers, norm=norm, rotary_emb=rotary_emb, lm_head=lm_head,
        embed_tokens=embed, instruction_ids_list=[instruction], frames_list=[frames],
        chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=0,
        num_top_layers=K, history_tokens=N, max_new_tokens=max_new,
    )
    got = emitted[0]

    # Rebuild the packed example with the emitted words per chunk and teacher-force a
    # restricted packed forward; its argmax at supervised positions must match.
    num_chunks = -(-n_frames // chunk_size)
    per_chunk = [[] for _ in range(num_chunks)]
    # Re-run the per-chunk split of the flat emission via a second decode that also
    # returns chunk ids, so we can group tokens by chunk.
    emitted2, chunk_ids = batched_stream_decode_script_last_layer(
        layers=layers, norm=norm, rotary_emb=rotary_emb, lm_head=lm_head,
        embed_tokens=embed, instruction_ids_list=[instruction], frames_list=[frames],
        chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=0,
        num_top_layers=K, history_tokens=N, max_new_tokens=max_new, return_chunk_ids=True,
    )
    assert emitted2 == emitted
    for tok, cid in zip(emitted2[0], chunk_ids[0]):
        per_chunk[cid].append(tok)

    chunks = [ChunkSpec(audio_len=chunk_size, target_ids=per_chunk[c]) for c in range(num_chunks)]
    packed = build_packed_chunk_example(instruction, chunks, VS, VE, EOT)
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask_full = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    mask_restricted = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32,
        history_window=N,
    )
    emb = _embed_with_audio(model, packed.input_ids, packed.is_audio, frames)[None]
    logits = run_script_layers_split(
        layers=layers, norm=norm, rotary_emb=rotary_emb, lm_head=lm_head,
        inputs_embeds=emb, position_ids=packed.position_ids[None],
        mask_lower=mask_full, mask_top=mask_restricted, num_top_layers=K,
    )[0]
    pred = logits.argmax(dim=-1)

    supervised = packed.target_ids != IGNORE_INDEX
    for c in range(num_chunks):
        idx_k = ((packed.seg_ids == c + 1) & supervised).nonzero(as_tuple=True)[0]
        u = len(per_chunk[c])
        assert pred[idx_k[:u]].tolist() == per_chunk[c], f"chunk {c}: decode words != restricted packed argmax"
        if u < max_new:
            assert int(pred[idx_k[u]]) == EOT, f"chunk {c}: expected eot at terminating position"


@torch.no_grad()
def test_last_layer_decode_large_window_matches_baseline():
    """With history_tokens >= any history the restriction is a no-op, so the
    two-cache manual decode must reproduce the standard batched decode token-for-token."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    embed = model.get_input_embeddings()
    layers, norm, rotary_emb, lm_head = _core(model)
    chunk_size = 2
    max_new = 4

    torch.manual_seed(999)
    instrs = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]
    frames_list = [torch.randn(6, H), torch.randn(2, H), torch.randn(7, H)]

    baseline = batched_stream_decode_script(
        llm=model, embed_tokens=embed, instruction_ids_list=instrs, frames_list=frames_list,
        chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=0,
        max_new_tokens=max_new,
    )
    for K in (1, 2):
        got = batched_stream_decode_script_last_layer(
            layers=layers, norm=norm, rotary_emb=rotary_emb, lm_head=lm_head,
            embed_tokens=embed, instruction_ids_list=instrs, frames_list=frames_list,
            chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=0,
            num_top_layers=K, history_tokens=10_000, max_new_tokens=max_new,
        )
        assert got == baseline, f"no-op restricted decode (K={K}) diverged:\n got={got}\n base={baseline}"


@torch.no_grad()
def test_batched_decode_contiguous_matches_per_utterance():
    """Batched == per-utterance decode also holds under contiguous positions
    (both decoders assign identical positions + order-causal KV attention)."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    embed = model.get_input_embeddings()
    pad_id = 0
    chunk_size = 2
    max_new = 4

    torch.manual_seed(999)
    instrs = [[5, 6, 7, 8], [8, 9, 1, 2, 3], [10, 11, 12, 13]]
    frames_list = [torch.randn(6, H), torch.randn(2, H), torch.randn(7, H)]

    ref = []
    for instr, frames in zip(instrs, frames_list):
        per_chunk = stream_decode_script(
            llm=model, embed_tokens=embed, instruction_ids=instr, frames=frames,
            chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT,
            max_new_tokens=max_new, contiguous_text_positions=True,
        )
        ref.append([t for chunk in per_chunk for t in chunk])

    got = batched_stream_decode_script(
        llm=model, embed_tokens=embed, instruction_ids_list=instrs, frames_list=frames_list,
        chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=pad_id,
        max_new_tokens=max_new, contiguous_text_positions=True,
    )
    assert got == ref, f"contiguous batched decode diverged:\n batched={got}\n per-utt={ref}"


# ---------------------------------------------------------------------------
# Windowed re-decoding (redecode)
# ---------------------------------------------------------------------------


def test_redecode_layout_structure():
    # 3 chunks of 2 frames each; window M=1 (N=2 chunks), depth R=1.
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(2, [30]), ChunkSpec(2, [40])]
    ex = build_packed_redecode_example(INSTR, chunks, VS, VE, EOT, audio_history_chunks=1, redecode_depth=1)

    # Spine = INSTR(2) + [20,21] + [30] + [40] = 6 tokens.
    assert ex.spine_len == 6
    assert ex.input_ids[:6].tolist() == [10, 11, 20, 21, 30, 40]
    assert ex.seg_ids[:6].tolist() == [0] * 6

    # Levels: c0 -> j=0,1 ; c1 -> j=0,1 ; c2 -> j=0  => 5 branches in (c,j) order.
    assert int(ex.seg_ids.max()) == 5

    # seg 1 = (c=0, j=0): base branch, window = chunk0 frames [0,1], prefix_len=2 (instruction).
    b = (ex.seg_ids == 1).nonzero(as_tuple=True)[0]
    assert ex.input_ids[b].tolist() == [VS, AUDIO_TOKEN_IDX, AUDIO_TOKEN_IDX, VE, 20, 21, EOT]
    assert (ex.prefix_len[b] == 2).all()
    assert ex.audio_frame_index[b][ex.is_audio[b]].tolist() == [0, 1]
    assert ex.target_ids[b].tolist() == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 20, 21, EOT, IGNORE_INDEX]

    # seg 2 = (c=0, j=1): +1 chunk lookahead -> window = chunks 0..1 frames [0,1,2,3];
    # history unchanged (prefix_len=2); still predicts chunk 0's words [20,21].
    b = (ex.seg_ids == 2).nonzero(as_tuple=True)[0]
    assert ex.input_ids[b].tolist() == [VS] + [AUDIO_TOKEN_IDX] * 4 + [VE, 20, 21, EOT]
    assert (ex.prefix_len[b] == 2).all()
    assert ex.audio_frame_index[b][ex.is_audio[b]].tolist() == [0, 1, 2, 3]
    # 9 branch tokens: <vs> + 4 audio + <ve> + 2 words + <eot>, positions pref..pref+8.
    assert ex.position_ids[b].tolist() == list(range(2, 2 + 9))
    sup = ex.target_ids[b] != IGNORE_INDEX
    assert ex.target_ids[b][sup].tolist() == [20, 21, EOT]

    # seg 4 = (c=1, j=1): history = instr + chunk0 words (prefix_len=4); window = chunks 1..2
    # frames [2,3,4,5]; predicts chunk 1's word [30].
    b = (ex.seg_ids == 4).nonzero(as_tuple=True)[0]
    assert (ex.prefix_len[b] == 4).all()
    assert ex.audio_frame_index[b][ex.is_audio[b]].tolist() == [2, 3, 4, 5]
    sup = ex.target_ids[b] != IGNORE_INDEX
    assert ex.target_ids[b][sup].tolist() == [30, EOT]

    # seg 5 = (c=2, j=0): last chunk, no lookahead available; prefix_len=5, window chunks 1..2.
    b = (ex.seg_ids == 5).nonzero(as_tuple=True)[0]
    assert (ex.prefix_len[b] == 5).all()
    assert ex.audio_frame_index[b][ex.is_audio[b]].tolist() == [2, 3, 4, 5]


def test_redecode_include_mode_subsamples_lookahead_branches():
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(2, [30]), ChunkSpec(2, [40])]
    # Drop every j>=1 branch -> only the 3 base (j=0) branches remain, and the result
    # must be byte-identical to the base windowed builder (audio_history_chunks=1).
    ex = build_packed_redecode_example(
        INSTR, chunks, VS, VE, EOT, audio_history_chunks=1, redecode_depth=1,
        include_mode=lambda c, j: False,
    )
    base = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT, audio_history_chunks=1)
    assert int(ex.seg_ids.max()) == 3
    assert ex.input_ids.tolist() == base.input_ids.tolist()
    assert ex.seg_ids.tolist() == base.seg_ids.tolist()
    assert ex.position_ids.tolist() == base.position_ids.tolist()
    assert ex.prefix_len.tolist() == base.prefix_len.tolist()
    assert ex.target_ids.tolist() == base.target_ids.tolist()
    assert ex.audio_frame_index.tolist() == base.audio_frame_index.tolist()


@torch.no_grad()
def test_parity_redecode_packed_vs_separate():
    """Every (c, j) branch's packed logits must equal its standalone example."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    instruction = [5, 6, 7]
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(3, [30]), ChunkSpec(2, [40, 41]), ChunkSpec(2, [50])]
    M, R = 2, 2
    total_frames = sum(c.audio_len for c in chunks)
    torch.manual_seed(2026)
    global_frames = torch.randn(total_frames, H)

    packed = build_packed_redecode_example(
        instruction, chunks, VS, VE, EOT, audio_history_chunks=M, redecode_depth=R
    )
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    packed_emb = _embed_by_frame_index(model, packed.input_ids, packed.audio_frame_index, global_frames)
    packed_logits = model(
        inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]
    ).logits[0]

    separate = build_separate_redecode_examples(
        instruction, chunks, VS, VE, EOT, audio_history_chunks=M, redecode_depth=R
    )
    # examples[i] lines up with packed branch segment i+1.
    for k, sep in enumerate(separate, start=1):
        sep_emb = _embed_by_frame_index(model, sep.input_ids, sep.audio_frame_index, global_frames)
        sep_logits = model(inputs_embeds=sep_emb[None]).logits[0]
        idx = (packed.seg_ids == k).nonzero(as_tuple=True)[0]
        assert idx.numel() == sep.input_ids.numel() - sep.branch_start
        torch.testing.assert_close(packed_logits[idx], sep_logits[sep.branch_start :], atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_redecode_batched_matches_per_utterance():
    """Batched redecode decode == decoding each utterance alone (padding-safe)."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    embed = model.get_input_embeddings()
    pad_id = 0
    chunk_size = 2
    max_new = 4
    M, R = 2, 2

    torch.manual_seed(909)
    instrs = [[5, 6, 7], [8, 9], [10, 11, 12]]
    # 5, 3, 6 chunks respectively (so the lock lag / trailing-chunk handling is exercised).
    frames_list = [torch.randn(10, H), torch.randn(6, H), torch.randn(12, H)]

    ref = []
    for instr, frames in zip(instrs, frames_list):
        one = batched_stream_decode_redecode(
            llm=model, embed_tokens=embed, instruction_ids_list=[instr], frames_list=[frames],
            chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=pad_id,
            audio_history_chunks=M, redecode_depth=R, max_new_tokens=max_new,
        )
        ref.append(one[0])

    got = batched_stream_decode_redecode(
        llm=model, embed_tokens=embed, instruction_ids_list=instrs, frames_list=frames_list,
        chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=pad_id,
        audio_history_chunks=M, redecode_depth=R, max_new_tokens=max_new,
    )
    assert got == ref, f"batched redecode decode diverged:\n batched={got}\n per-utt={ref}"


@torch.no_grad()
def test_redecode_noncorrective_depth0():
    """redecode_depth=0 is the NON-CORRECTIVE run: each chunk is decoded exactly
    once at j=0 (on its own running j=0 history + the M-chunk window ending at that
    chunk) and appended -- no lookahead, no re-decoding of past chunks. It must
    match a manual j=0-only replay, and every token's lock step is its own chunk.

    Note this is DISTINCT from the provisional (j=0) stream of a corrective (R>=1)
    run, whose previews condition on already partially-corrected history.
    """
    model = _tiny_qwen3()
    H = model.config.hidden_size
    embed = model.get_input_embeddings()
    instruction = [5, 6, 7]
    chunk_size = 2
    M = 2
    n_chunks = 5
    torch.manual_seed(313)
    frames = torch.randn(n_chunks * chunk_size, H)
    max_new = 4

    emitted, lock_steps = batched_stream_decode_redecode(
        llm=model, embed_tokens=embed, instruction_ids_list=[instruction], frames_list=[frames],
        chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=0,
        audio_history_chunks=M, redecode_depth=0, max_new_tokens=max_new, return_chunk_ids=True,
    )
    emitted, lock_steps = emitted[0], lock_steps[0]

    # Manual j=0-only replay: history is the running (un-corrected) j=0 outputs.
    committed = []
    for c in range(n_chunks):
        win_start = max(0, c - M) * chunk_size
        win_end = (c + 1) * chunk_size
        fr = frames[win_start:win_end]
        hist = [t for w in committed for t in w]
        committed.append(_decode_one(model, embed, instruction, hist, fr, chunk_size, max_new))

    expected = [t for w in committed for t in w]
    assert emitted == expected, f"non-corrective (R=0) decode diverged:\n got={emitted}\n exp={expected}"
    # min(c + R, last) with R=0 -> every token locks at its own chunk index.
    exp_locks = [c for c, w in enumerate(committed) for _ in w]
    assert lock_steps == exp_locks


@torch.no_grad()
def test_redecode_decode_matches_forced_packed():
    """Greedy redecode decode must equal the argmax of a teacher-forced packed
    forward built from the decoded per-chunk tokens (decode <-> training layout).

    For each chunk c, the LOCKED value is decoded at lookahead j = min(R, last-c)
    from clean history; rebuilding the packed redecode example with the decoded
    tokens as targets and taking the argmax at that (c, j) branch must reproduce it.
    """
    model = _tiny_qwen3()
    H = model.config.hidden_size
    embed = model.get_input_embeddings()
    instruction = [5, 6, 7]
    chunk_size = 2
    M, R = 2, 1
    n_chunks = 4
    torch.manual_seed(4242)
    frames = torch.randn(n_chunks * chunk_size, H)
    max_new = 4

    # Decode a single stream; committed[c] is recovered by re-running the reference
    # loop here so we can segment per chunk (the public API returns a flat stream).
    emitted = batched_stream_decode_redecode(
        llm=model, embed_tokens=embed, instruction_ids_list=[instruction], frames_list=[frames],
        chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=0,
        audio_history_chunks=M, redecode_depth=R, max_new_tokens=max_new,
    )[0]

    # Recover per-chunk committed tokens with a tiny single-stream replay.
    committed = [None] * n_chunks
    for t in range(n_chunks):
        for j in range(min(R, t), -1, -1):
            c = t - j
            win_start = max(0, t - M) * chunk_size
            win_end = (t + 1) * chunk_size
            fr = frames[win_start:win_end]
            hist = [tok for cc in range(c) for tok in (committed[cc] or [])]
            words = _decode_one(model, embed, instruction, hist, fr, chunk_size, max_new)
            committed[c] = words
    assert [tok for c in range(n_chunks) for tok in (committed[c] or [])] == emitted

    # Now teacher-force the packed example built from the decoded chunks and check
    # the locked branch's argmax reproduces each chunk's committed tokens.
    chunks = [ChunkSpec(chunk_size, list(committed[c] or [])) for c in range(n_chunks)]
    packed = build_packed_redecode_example(
        instruction, chunks, VS, VE, EOT, audio_history_chunks=M, redecode_depth=R
    )
    valid = torch.ones_like(packed.input_ids, dtype=torch.bool)
    mask = build_script_mask(
        packed.seg_ids[None], packed.position_ids[None], packed.prefix_len[None], valid[None], torch.float32
    )
    packed_emb = _embed_by_frame_index(model, packed.input_ids, packed.audio_frame_index, frames)
    logits = model(inputs_embeds=packed_emb[None], attention_mask=mask, position_ids=packed.position_ids[None]).logits[0]

    # Map each (c, j) to its segment id in (c, j) order to find the locked branch.
    seg = 0
    seg_of = {}
    for c in range(n_chunks):
        for j in range(0, min(R, n_chunks - 1 - c) + 1):
            seg += 1
            seg_of[(c, j)] = seg
    for c in range(n_chunks):
        j_lock = min(R, n_chunks - 1 - c)
        idx = (packed.seg_ids == seg_of[(c, j_lock)]).nonzero(as_tuple=True)[0]
        sup = (packed.target_ids[idx] != IGNORE_INDEX)
        n_words = len(committed[c] or [])
        if n_words == 0:
            continue
        # supervised positions predict [w_0..w_{n-1}, eot]; first n argmax == the words.
        pred = logits[idx][sup].argmax(dim=-1).tolist()
        assert pred[:n_words] == list(committed[c])


def _decode_one(model, embed, instruction, hist, fr, chunk_size, max_new):
    """Single-chunk greedy decode reference (no padding/batching)."""
    device = fr.device
    ids = list(instruction) + list(hist) + [VS] + [AUDIO_TOKEN_IDX] * int(fr.shape[0]) + [VE]
    ids_t = torch.tensor(ids, dtype=torch.long, device=device)
    is_audio = ids_t == AUDIO_TOKEN_IDX
    emb = _embed_with_audio(model, ids_t, is_audio, fr)
    out = model(inputs_embeds=emb[None], use_cache=True, return_dict=True)
    logits = out.logits[:, -1]
    cache = out.past_key_values
    cur = torch.tensor([[len(ids)]], device=device)
    words = []
    for _ in range(max_new):
        nxt = int(logits.argmax(dim=-1).item())
        if nxt == EOT:
            break
        words.append(nxt)
        temb = model.get_input_embeddings()(torch.tensor([[nxt]], device=device))
        out = model(inputs_embeds=temb, position_ids=cur, past_key_values=cache, use_cache=True, return_dict=True)
        cache = out.past_key_values
        cur = cur + 1
        logits = out.logits[:, -1]
    return words


# ---------------------------------------------------------------------------
# Shared-audio layout (encoder frames laid ONCE as a self-contained causal track;
# each branch attends its window through the mask -> packed length independent of
# the audio window size). Mirrors the coverage every other layout has: a
# structural test, packed-vs-separate parity (base / fixed-frame / chunk windows),
# and a batched==per-utterance decode equivalence.
# ---------------------------------------------------------------------------

from nemo.collections.speechlm2.parts.shared_audio_chunk import (  # noqa: E402
    AUDIO_SEG_ID,
    SPINE_SEG_ID,
    batched_shared_audio_decode,
    build_separate_shared_audio_examples,
    build_shared_audio_chunk_example,
    build_shared_audio_chunk_mask,
)


def _shared_audio_mask(ex, dtype=torch.float32):
    """4D shared-audio mask for a single (unbatched) example."""
    valid = torch.ones_like(ex.input_ids, dtype=torch.bool)
    return build_shared_audio_chunk_mask(
        ex.seg_ids[None],
        ex.position_ids[None],
        ex.prefix_len[None],
        ex.win_start[None],
        ex.win_end[None],
        ex.audio_frame_index[None],
        valid[None],
        dtype,
    )


def test_shared_audio_layout_structure():
    # chunk0: 2 frames [20,21]; chunk1: 3 frames [30]; chunk2: 2 frames [] (silent).
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(3, [30]), ChunkSpec(2, [])]
    ex = build_shared_audio_chunk_example(INSTR, chunks, VE, EOT, audio_history_chunks=1)

    P = len(INSTR) + 2 + 1 + 0  # instruction + all chunk words
    F = 2 + 3 + 2  # total frames laid ONCE
    # Spine (pure text) first, then the single shared audio track.
    assert ex.spine_len == P
    assert ex.audio_len == F
    assert ex.seg_ids[:P].tolist() == [SPINE_SEG_ID] * P
    assert ex.seg_ids[P : P + F].tolist() == [AUDIO_SEG_ID] * F
    # Audio track carries the frame index 0..F-1 (doubles as the embed gather idx),
    # laid contiguously exactly once (no per-branch copies).
    assert ex.audio_frame_index[P : P + F].tolist() == list(range(F))
    assert bool(ex.is_audio[P : P + F].all())
    assert int(ex.is_audio[:P].sum()) == 0

    # Branch windows (frame bounds [ws, we)) with audio_history_chunks=1:
    #   chunk0 -> [0,2); chunk1 -> chunks 0..1 -> [0,5); chunk2 -> chunks 1..2 -> [2,7).
    def win(seg):
        b = (ex.seg_ids == seg).nonzero(as_tuple=True)[0]
        return int(ex.win_start[b][0]), int(ex.win_end[b][0])

    assert win(1) == (0, 2)
    assert win(2) == (0, 5)
    assert win(3) == (2, 7)
    # A branch is anchor <ve> + words + eot (no per-branch <vs>/audio copy).
    b2 = (ex.seg_ids == 2).nonzero(as_tuple=True)[0]
    assert ex.input_ids[b2].tolist() == [VE, 30, EOT]
    # Silent chunk 2: anchor + eot only, and the anchor predicts eot directly.
    b3 = (ex.seg_ids == 3).nonzero(as_tuple=True)[0]
    assert ex.input_ids[b3].tolist() == [VE, EOT]
    assert ex.target_ids[b3].tolist() == [EOT, IGNORE_INDEX]


@torch.no_grad()
def _run_shared_audio_parity(chunks, instruction, *, audio_window_frames=0, audio_history_chunks=0, seed=0):
    """Every branch's logits in the packed shared-audio sequence must equal the
    standalone ``[history][audio 0..we][<ve> w_k <eot>]`` example run under the
    same shared-audio mask (the parity claim for the shared-audio optimization)."""
    model = _tiny_qwen3()
    H = model.config.hidden_size
    total_frames = sum(c.audio_len for c in chunks)
    torch.manual_seed(seed)
    global_frames = torch.randn(total_frames, H)

    packed = build_shared_audio_chunk_example(
        instruction,
        chunks,
        VE,
        EOT,
        audio_window_frames=audio_window_frames,
        audio_history_chunks=audio_history_chunks,
    )
    packed_emb = _embed_by_frame_index(model, packed.input_ids, packed.audio_frame_index, global_frames)
    packed_logits = model(
        inputs_embeds=packed_emb[None],
        attention_mask=_shared_audio_mask(packed),
        position_ids=packed.position_ids[None],
    ).logits[0]

    separate = build_separate_shared_audio_examples(
        instruction,
        chunks,
        VE,
        EOT,
        audio_window_frames=audio_window_frames,
        audio_history_chunks=audio_history_chunks,
    )
    for k, sep in enumerate(separate, start=1):
        sep_emb = _embed_by_frame_index(model, sep.input_ids, sep.audio_frame_index, global_frames)
        sep_logits = model(
            inputs_embeds=sep_emb[None],
            attention_mask=_shared_audio_mask(sep),
            position_ids=sep.position_ids[None],
        ).logits[0]
        idx = (packed.seg_ids == k).nonzero(as_tuple=True)[0]
        assert packed_logits[idx].shape == sep_logits[sep.branch_start :].shape
        torch.testing.assert_close(
            packed_logits[idx], sep_logits[sep.branch_start :], atol=1e-4, rtol=1e-4
        )


def test_parity_shared_audio_packed_vs_separate():
    # Base window (each branch sees only its own chunk's frames).
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(3, [30]), ChunkSpec(1, []), ChunkSpec(2, [40, 41, 42])]
    _run_shared_audio_parity(chunks, [5, 6, 7], seed=123)


def test_parity_shared_audio_fixed_frame_window():
    # Fixed-frame window: each branch sees the last W=5 frames ending at its boundary.
    chunks = [ChunkSpec(3, [20]), ChunkSpec(3, [21, 22]), ChunkSpec(3, [23]), ChunkSpec(3, [24, 25])]
    _run_shared_audio_parity(chunks, [5, 6, 7], audio_window_frames=5, seed=2024)


def test_parity_shared_audio_chunk_window():
    # Chunk-based window: each branch sees the previous M=1 chunk plus its own.
    chunks = [ChunkSpec(2, [20, 21]), ChunkSpec(3, [30]), ChunkSpec(2, [40, 41])]
    _run_shared_audio_parity(chunks, [5, 6, 7], audio_history_chunks=1, seed=77)


@torch.no_grad()
def test_batched_shared_audio_decode_matches_per_utterance():
    """Batched chunk-synchronous shared-audio decode must equal decoding each
    utterance on its own (exercises left-padding, per-row positions/window bounds,
    and finished-stream masking across a ragged batch)."""
    model = _tiny_qwen3()
    embed = model.get_input_embeddings()
    H = model.config.hidden_size
    chunk_size = 2
    max_new = 4
    W = 3  # fixed-frame window

    torch.manual_seed(999)
    instrs = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]
    # Different total frame counts -> different #chunks (3, 1, 4).
    frames_list = [torch.randn(6, H), torch.randn(2, H), torch.randn(7, H)]

    ref = [
        batched_shared_audio_decode(
            llm=model,
            embed_tokens=embed,
            instruction_ids_list=[instr],
            frames_list=[fr],
            chunk_size=chunk_size,
            vision_end_id=VE,
            eot_id=EOT,
            pad_id=0,
            max_new_tokens=max_new,
            audio_window_frames=W,
        )[0]
        for instr, fr in zip(instrs, frames_list)
    ]

    got = batched_shared_audio_decode(
        llm=model,
        embed_tokens=embed,
        instruction_ids_list=instrs,
        frames_list=frames_list,
        chunk_size=chunk_size,
        vision_end_id=VE,
        eot_id=EOT,
        pad_id=0,
        max_new_tokens=max_new,
        audio_window_frames=W,
    )

    assert got == ref, f"batched shared-audio decode diverged from per-utterance:\n batched={got}\n per-utt={ref}"
