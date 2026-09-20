#!/usr/bin/env python3
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

"""Time the banded RNN-T loss: reference PyTorch vs the CUDA kernels.

Measures the LOSS ONLY -- forward plus backward -- not a training step, because
a training step also pays for the encoder, prediction network and joint, which
are identical on both paths. The end-to-end effect is therefore smaller than
the number printed here, and the script prints what fraction of a step the loss
would have to be for a given end-to-end gain.

Shapes are chosen to match the running spe1k arm: chunk_size 14 at 80 ms is
1.12 s per chunk, and the bucket batch sizes run from 152 (shortest) to 16
(longest), so a short-utterance batch has many utterances and few chunks and a
long one is the reverse.
"""

import argparse
import time

import numpy as np
import torch

from nemo.collections.asr.losses.banded_rnnt import BandedLattice, banded_rnnt_loss, build_lattices
from nemo.collections.asr.parts.numba.banded_rnnt.banded_rnnt_numba import (
    banded_rnnt_loss_cuda,
    build_band_index,
    kernel_is_usable,
)

# (batch, chunks, label) -- the bucket extremes and a midpoint.
SHAPES = [(152, 9, "short utts, big batch"), (40, 27, "mid"), (16, 54, "long utts, small batch")]


def make_case(B, T, vocab, rng):
    """Per-utterance chunk token-id lists, ~5 tokens per 1.12 s chunk at 1k vocab."""
    return [
        [[int(rng.integers(0, vocab - 1)) for _ in range(int(rng.poisson(5)))] for _ in range(T)] for _ in range(B)
    ]


def timeit(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=1025, help="V+1, matching the 1k SentencePiece arm")
    ap.add_argument("--band", type=int, default=1)
    ap.add_argument("--band-side", default="both")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    device = torch.device("cuda")
    blank = args.vocab - 1
    rng = np.random.default_rng(0)

    print(f"vocab={args.vocab} band={args.band} side={args.band_side} iters={args.iters}")
    print(f"{'shape':<26} {'nodes':>8} {'ref ms':>9} {'kernel ms':>10} {'speedup':>8}  {'loss match':>10}")
    for B, T, label in SHAPES:
        chunks = make_case(B, T, args.vocab, rng)
        per_utt, nc, tl = build_lattices(chunks, args.band, args.band_side)
        lat = BandedLattice(per_utt, nc, tl)
        band = build_band_index(chunks, args.band, args.band_side)
        ok, why = kernel_is_usable(band, device)
        if not ok:
            print(f"{label:<26} {'-':>8} {'-':>9} {'-':>10} {'-':>8}  SKIP: {why}")
            continue

        u_max = max(max(tl), 1)
        targets = torch.zeros((B, u_max), dtype=torch.long, device=device)
        for b, cs in enumerate(chunks):
            flat = [t for c in cs for t in c]
            if flat:
                targets[b, : len(flat)] = torch.tensor(flat, dtype=torch.long, device=device)
        base = torch.randn(lat.num_nodes, args.vocab, device=device)

        def run_ref():
            x = base.clone().requires_grad_(True)
            banded_rnnt_loss(x.log_softmax(-1), lat, targets, blank).sum().backward()
            return x.grad

        def run_kernel():
            x = base.clone().requires_grad_(True)
            banded_rnnt_loss_cuda(x.log_softmax(-1), band, targets, blank).sum().backward()
            return x.grad

        # Equality first: a fast wrong answer is worse than a slow right one.
        xr = base.clone().requires_grad_(True)
        lr = banded_rnnt_loss(xr.log_softmax(-1), lat, targets, blank)
        xk = base.clone().requires_grad_(True)
        lk = banded_rnnt_loss_cuda(xk.log_softmax(-1), band, targets, blank)
        match = torch.allclose(lr, lk, rtol=1e-4, atol=1e-4)

        t_ref = timeit(run_ref, args.iters) * 1e3
        t_ker = timeit(run_kernel, args.iters) * 1e3
        print(
            f"{label:<26} {lat.num_nodes:>8} {t_ref:>9.2f} {t_ker:>10.2f} {t_ref / t_ker:>7.2f}x  "
            f"{'OK' if match else 'MISMATCH':>10}"
        )


if __name__ == "__main__":
    main()
