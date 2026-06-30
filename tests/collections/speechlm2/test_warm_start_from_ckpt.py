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

"""Tests for warm_start_from_ckpt (vocab-resize-aware weight init).

This is the helper the StreamingSTT training script uses to warm-start a new run
(e.g. one that just added the <flush> special token, growing the vocab by one
row) from an already-trained checkpoint. The key guarantee is that overlapping
vocab rows are copied while any newly added rows keep their initialization.
"""

import torch
import torch.nn as nn

from nemo.collections.speechlm2.parts.pretrained import warm_start_from_ckpt


class _Tiny(nn.Module):
    """Embedding + tied-ish head + a non-vocab tensor, mirroring the shapes that
    matter for warm-start (vocab-dim-0 tensors vs. everything else)."""

    def __init__(self, vocab: int, hidden: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.proj = nn.Linear(hidden, hidden)


def test_warm_start_resizes_vocab_and_preserves_new_rows(tmp_path):
    # "Old" trained checkpoint with a smaller vocab.
    old = _Tiny(vocab=10)
    # Make rows easily identifiable.
    with torch.no_grad():
        old.embed_tokens.weight.copy_(torch.arange(10).float().unsqueeze(1).repeat(1, 4))
    ckpt_path = tmp_path / "old.ckpt"
    torch.save({"state_dict": old.state_dict()}, ckpt_path)

    # "New" model: vocab grew by one (the <flush> row). Mark the new row sentinel.
    new = _Tiny(vocab=11)
    with torch.no_grad():
        new.embed_tokens.weight.fill_(-999.0)  # so we can detect what got copied
    SENTINEL = new.embed_tokens.weight[10].clone()  # the would-be <flush> row

    stats = warm_start_from_ckpt(new, str(ckpt_path))

    # Overlapping rows [0:10] copied from the checkpoint ...
    assert torch.allclose(new.embed_tokens.weight[:10], old.embed_tokens.weight)
    # ... and the new row 10 kept its (fresh) init, NOT overwritten.
    assert torch.allclose(new.embed_tokens.weight[10], SENTINEL)
    # lm_head (also vocab-dim-0) was resized + copied too.
    assert torch.allclose(new.lm_head.weight[:10], old.lm_head.weight)
    # Two resized tensors: embed_tokens.weight and lm_head.weight.
    assert stats["resized"] == 2
    assert stats["skipped"] == 0


def test_warm_start_exact_shapes_full_copy(tmp_path):
    old = _Tiny(vocab=8)
    ckpt_path = tmp_path / "old.ckpt"
    torch.save({"state_dict": old.state_dict()}, ckpt_path)

    new = _Tiny(vocab=8)
    stats = warm_start_from_ckpt(new, str(ckpt_path))

    for k in old.state_dict():
        assert torch.allclose(new.state_dict()[k], old.state_dict()[k])
    assert stats["resized"] == 0
    assert stats["skipped"] == 0


def test_warm_start_skips_incompatible_and_keeps_missing(tmp_path):
    old = _Tiny(vocab=8, hidden=4)
    sd = old.state_dict()
    # A genuinely incompatible tensor (different non-vocab dim) must be skipped.
    sd["proj.weight"] = torch.zeros(6, 6)
    # An extra checkpoint key absent from the model must be ignored.
    sd["does_not_exist"] = torch.zeros(3)
    # Drop a model key from the checkpoint -> it must keep its init (counts as missing).
    del sd["proj.bias"]
    ckpt_path = tmp_path / "old.ckpt"
    torch.save({"state_dict": sd}, ckpt_path)

    new = _Tiny(vocab=8, hidden=4)
    init_proj_bias = new.proj.bias.detach().clone()
    stats = warm_start_from_ckpt(new, str(ckpt_path))

    # proj.weight shape-incompatible -> skipped (model keeps its own init).
    assert stats["skipped"] >= 1
    # proj.bias absent from ckpt -> kept at init.
    assert torch.allclose(new.proj.bias, init_proj_bias)
    # The unused/extra checkpoint key is accounted for.
    assert stats["ckpt_unused"] >= 1
