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
"""Path-restricted CHAT joint: identical values, without the [B,T,U,V] tensor.

The full joint returns [B, T, U, V+1], which grows as T*U*V and is what makes a
large vocabulary untrainable for a transducer. With the alignment fixed by forced
alignment only the pairs on the path are needed, so the path variant must
reproduce exactly those entries -- otherwise the cheap training objective is
optimising something different from what the model computes at decode time.
"""

import pytest
import torch

from nemo.collections.asr.modules import RNNTAttJoint


def _joint(vocab=11, d=16, heads=4):
    torch.manual_seed(0)
    j = RNNTAttJoint(
        jointnet={
            "encoder_hidden": d,
            "pred_hidden": d,
            "joint_hidden": d,
            "activation": "relu",
            "dropout": 0.0,
        },
        num_classes=vocab,
        chunk_size=4,
    )
    return j.eval()


@pytest.mark.unit
@pytest.mark.parametrize("with_sizes", [True, False])
def test_joint_on_path_matches_the_full_joint(with_sizes):
    B, T, C, U, D = 2, 3, 4, 5, 16
    j = _joint(d=D)
    torch.manual_seed(1)
    f_chunked = torch.randn(B, T, C, D)
    g = torch.randn(B, U, D)
    sizes = torch.tensor([[4, 4, 2], [4, 1, 0]]) if with_sizes else None

    with torch.no_grad():
        full = j.cross_attention(f_chunked, g, sizes)  # [B, T, U, D]
        b = torch.tensor([0, 0, 1, 1, 1])
        t = torch.tensor([0, 2, 0, 1, 2])
        u = torch.tensor([0, 4, 1, 3, 2])
        path = j.cross_attention_on_path(f_chunked, g, sizes, b, t, u)  # [N, D]

    assert path.shape == (5, D)
    torch.testing.assert_close(path, full[b, t, u], atol=1e-5, rtol=1e-5)


@pytest.mark.unit
def test_joint_on_path_covers_every_pair():
    """Exhaustively: every (b,t,u) must match, not just a lucky sample."""
    B, T, C, U, D = 2, 3, 4, 4, 16
    j = _joint(d=D)
    torch.manual_seed(2)
    f_chunked = torch.randn(B, T, C, D)
    g = torch.randn(B, U, D)
    sizes = torch.tensor([[4, 3, 1], [2, 4, 0]])

    b, t, u = torch.meshgrid(torch.arange(B), torch.arange(T), torch.arange(U), indexing="ij")
    b, t, u = b.reshape(-1), t.reshape(-1), u.reshape(-1)
    with torch.no_grad():
        full = j.cross_attention(f_chunked, g, sizes)
        path = j.cross_attention_on_path(f_chunked, g, sizes, b, t, u)
    torch.testing.assert_close(path, full[b, t, u], atol=1e-5, rtol=1e-5)


@pytest.mark.unit
def test_joint_on_path_is_asymptotically_cheaper():
    """The point of the exercise: cost must scale with the PATH, not T*U*V."""
    B, T_frames, U, D, V = 1, 40, 30, 16, 4096
    j = _joint(vocab=V, d=D)
    torch.manual_seed(3)
    f = torch.randn(B, T_frames, D)
    g = torch.randn(B, U, D)
    f_len = torch.tensor([T_frames])

    with torch.no_grad():
        full = j.joint(f, g, f_len)  # [B, T_chunks, U, V+1]
        n_chunks = full.shape[1]
        # A forced-alignment path visits each (t, u) once: U labels + one blank per chunk.
        n_path = U + n_chunks
        b = torch.zeros(n_path, dtype=torch.long)
        t = torch.clamp(torch.arange(n_path) * n_chunks // n_path, max=n_chunks - 1)
        u = torch.clamp(torch.arange(n_path), max=U - 1)
        path = j.joint_on_path(f, g, b, t, u, f_len)

    assert path.shape == (n_path, V + 1)
    torch.testing.assert_close(path, full[b, t, u], atol=1e-4, rtol=1e-4)
    # The tensor that is never built.
    assert full.numel() // path.numel() > 5, "path variant should be far smaller here"
