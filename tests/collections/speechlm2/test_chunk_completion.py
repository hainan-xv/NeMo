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
"""Tests for the chunk-completion (spine + branch) packed layout.

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
from nemo.collections.speechlm2.parts.chunk_completion import (
    ChunkSpec,
    batched_stream_decode_chunk_completion,
    build_chunk_completion_mask,
    build_packed_chunk_example,
    build_separate_chunk_examples,
    collate_packed_chunk_examples,
    stream_decode_chunk_completion,
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


# ---------------------------------------------------------------------------
# Mask tests
# ---------------------------------------------------------------------------


def test_mask_spine_causal_and_pure_text():
    chunks = [ChunkSpec(audio_len=2, target_ids=[20, 21]), ChunkSpec(audio_len=3, target_ids=[30])]
    ex = build_packed_chunk_example(INSTR, chunks, VS, VE, EOT)
    valid = torch.ones_like(ex.input_ids, dtype=torch.bool)
    mask = build_chunk_completion_mask(
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
    mask = build_chunk_completion_mask(
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
    mask = build_chunk_completion_mask(seg[None], pos[None], pref[None], valid[None], torch.float32)[0, 0]
    for q in range(T + 2):
        assert _blocked(mask[q, T]) and _blocked(mask[q, T + 1])


class _FakeTok:
    """Deterministic char-code tokenizer for parsing tests."""

    def text_to_ids(self, s: str):
        s = s.strip()
        return [ord(c) for c in s] if s else []


def _make_dataset_stub(blank_token: str):
    from nemo.collections.speechlm2.data.chunk_completion_dataset import ChunkCompletionSTTDataset

    ds = object.__new__(ChunkCompletionSTTDataset)  # bypass heavy __init__
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


def _cc_dataset_class():
    from nemo.collections.speechlm2.data.chunk_completion_dataset import ChunkCompletionSTTDataset

    return ChunkCompletionSTTDataset


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
    mask = build_chunk_completion_mask(
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
    mask = build_chunk_completion_mask(
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
    emitted = stream_decode_chunk_completion(
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
    mask = build_chunk_completion_mask(
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


def _embed_by_frame_index(model, input_ids: torch.Tensor, audio_frame_index: torch.Tensor, global_frames: torch.Tensor):
    """Embed text ids and fill each audio slot by its GLOBAL frame index (gather).

    Mirrors ChunkCompletionSTTModel._build_input_embeds_indexed, used for the
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
    mask = build_chunk_completion_mask(
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
    mask = build_chunk_completion_mask(
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
        per_chunk = stream_decode_chunk_completion(
            llm=model, embed_tokens=embed, instruction_ids=instr, frames=frames,
            chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT,
            max_new_tokens=max_new,
        )
        ref.append([t for chunk in per_chunk for t in chunk])

    # Batched.
    got = batched_stream_decode_chunk_completion(
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
        per_chunk = stream_decode_chunk_completion(
            llm=model, embed_tokens=embed, instruction_ids=instr, frames=frames,
            chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT,
            max_new_tokens=max_new, audio_history_chunks=M,
        )
        ref.append([t for chunk in per_chunk for t in chunk])

    got = batched_stream_decode_chunk_completion(
        llm=model, embed_tokens=embed, instruction_ids_list=instrs, frames_list=frames_list,
        chunk_size=chunk_size, vision_start_id=VS, vision_end_id=VE, eot_id=EOT, pad_id=pad_id,
        max_new_tokens=max_new, audio_history_chunks=M,
    )
    assert got == ref, f"windowed batched decode diverged:\n batched={got}\n per-utt={ref}"
