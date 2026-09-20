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

"""Host side of the banded RNN-T loss: band index, kernel launch, autograd.

See ``banded_rnnt_kernel.py`` for why the kernels exist. This module supplies
the two things the kernels need and the reference implementation did badly:

  1. ``BandIndex`` -- the band's geometry built with numpy in O(B) numpy calls
     rather than O(number of nodes) Python dict operations. The reference built
     a ``dict`` keyed on ``(b, t, u)`` tuples and then made a second full pass
     to find each node's two predecessors; the kernels need neither, because a
     predecessor is just an index arithmetic step plus a band bounds check.

  2. A ``torch.autograd.Function`` that returns an ANALYTIC gradient from the
     alpha/beta recursions, instead of letting autograd tape T+U out-of-place
     ``index_put`` operations and walk back through them.
"""

from typing import List, Sequence, Tuple

import numpy as np
import torch

from nemo.collections.asr.parts.numba.banded_rnnt.banded_rnnt_kernel import (
    NEG_INF,
    compute_alphas_kernel,
    compute_betas_kernel,
    compute_grad_kernel,
)

# CUDA caps a block at 1024 threads and the kernels use one thread per label
# position, so a longer target than this cannot be scored by the kernel and the
# caller must fall back. Checked rather than assumed: silently launching with
# too many threads is a CudaAPIError at best and a wrong answer at worst.
MAX_THREADS_PER_BLOCK = 1024


class BandIndex:
    """Geometry of the banded lattice for a batch.

    ``b_idx``/``t_idx``/``u_idx`` enumerate the band's nodes in the SAME order
    the reference ``BandedLattice`` used -- b-major, then t ascending, then u
    ascending -- because the caller evaluates the joint at exactly these triples
    and the returned ``[N, V+1]`` must line up.
    """

    def __init__(self, band_lo: np.ndarray, band_hi: np.ndarray, num_chunks: np.ndarray, target_lens: np.ndarray):
        self.band_lo = band_lo  # [B, maxT]
        self.band_hi = band_hi  # [B, maxT]
        self.num_chunks = num_chunks  # [B]
        self.target_lens = target_lens  # [B]
        self.maxT = int(band_lo.shape[1])
        self.maxU1 = int(target_lens.max()) + 1 if len(target_lens) else 1

        B = len(num_chunks)
        bs, ts, us = [], [], []
        for b in range(B):
            T = int(num_chunks[b])
            if T == 0:
                continue
            lo = band_lo[b, :T]
            hi = band_hi[b, :T]
            widths = (hi - lo + 1).astype(np.int64)
            t_rep = np.repeat(np.arange(T, dtype=np.int64), widths)
            # u runs lo[t]..hi[t] inclusive for each t, concatenated.
            starts = np.repeat(lo.astype(np.int64), widths)
            within = np.arange(widths.sum(), dtype=np.int64) - np.repeat(
                np.concatenate(([0], np.cumsum(widths)[:-1])), widths
            )
            us.append(starts + within)
            ts.append(t_rep)
            bs.append(np.full(widths.sum(), b, dtype=np.int64))
        self.b_idx = np.concatenate(bs) if bs else np.zeros(0, dtype=np.int64)
        self.t_idx = np.concatenate(ts) if ts else np.zeros(0, dtype=np.int64)
        self.u_idx = np.concatenate(us) if us else np.zeros(0, dtype=np.int64)
        self.num_nodes = int(self.b_idx.shape[0])

    def index_tensors(self, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        as_t = lambda x: torch.from_numpy(x).to(device, non_blocking=True)  # noqa: E731
        return as_t(self.b_idx), as_t(self.t_idx), as_t(self.u_idx)

    def fits_in_one_block(self) -> bool:
        return self.maxU1 <= MAX_THREADS_PER_BLOCK


def build_band_index(
    chunk_tokens_per_utt: Sequence[Sequence[Sequence[int]]], band: int, band_side: str = "both"
) -> BandIndex:
    """Band geometry from per-chunk token counts, mirroring ``band_nodes``.

    At chunk ``t`` the forced path occupies ``u`` in ``[S(t), S(t+1)]`` with
    ``S`` the cumulative token count; widening by ``band`` chunks gives
    ``[S(t-band), S(t+band+1)]``. Both bounds are monotone non-decreasing in
    ``t``, which is what makes the lattice a true band and the kernels' bounds
    check a pair of comparisons.
    """
    if band_side not in ("both", "later", "earlier"):
        raise ValueError(f"band_side must be 'both', 'later' or 'earlier', got {band_side!r}")

    B = len(chunk_tokens_per_utt)
    num_chunks = np.array([len(c) for c in chunk_tokens_per_utt], dtype=np.int64)
    maxT = int(num_chunks.max()) if B and num_chunks.max() > 0 else 1
    band_lo = np.zeros((B, maxT), dtype=np.int32)
    band_hi = np.zeros((B, maxT), dtype=np.int32)
    target_lens = np.zeros(B, dtype=np.int64)

    for b, chunks in enumerate(chunk_tokens_per_utt):
        T = len(chunks)
        if T == 0:
            continue
        counts = np.array([len(c) for c in chunks], dtype=np.int64)
        cum = np.concatenate(([0], np.cumsum(counts)))  # S, length T+1
        target_lens[b] = int(cum[-1])
        t = np.arange(T, dtype=np.int64)
        if band_side in ("both", "later"):
            lo = cum[np.maximum(0, t - band)]
        else:
            lo = cum[t]
        if band_side in ("both", "earlier"):
            hi = cum[np.minimum(T - 1, t + band) + 1]
        else:
            hi = cum[t + 1]
        band_lo[b, :T] = lo
        band_hi[b, :T] = hi

    return BandIndex(band_lo, band_hi, num_chunks, target_lens)


class _BandedRNNTFunction(torch.autograd.Function):
    """NLL over the band, with an analytic gradient from alpha/beta."""

    @staticmethod
    def forward(ctx, log_probs, targets, band: BandIndex, blank_id: int):
        device = log_probs.device
        B = len(band.num_chunks)
        maxT, maxU1 = band.maxT, band.maxU1

        b_t, t_t, u_t = band.index_tensors(device)

        # The two log-probabilities each node can use: blank, and the single
        # label leaving it. At u == U there is no label, so clamp the gather and
        # mask -- indexing out of bounds would be a silent wrap, not an error.
        blank_flat = log_probs[:, blank_id]
        if targets.shape[1] > 0:
            u_clamped = u_t.clamp(max=targets.shape[1] - 1)
            label_flat = log_probs.gather(1, targets[b_t, u_clamped].unsqueeze(1)).squeeze(1)
            label_flat = torch.where(u_t < targets.shape[1], label_flat, torch.full_like(label_flat, NEG_INF))
        else:
            label_flat = torch.full_like(blank_flat, NEG_INF)

        flat_idx = (b_t * maxT + t_t) * maxU1 + u_t
        dense_shape = (B * maxT * maxU1,)
        blank_lp = torch.full(dense_shape, NEG_INF, device=device, dtype=torch.float32)
        label_lp = torch.full(dense_shape, NEG_INF, device=device, dtype=torch.float32)
        blank_lp.scatter_(0, flat_idx, blank_flat.float())
        label_lp.scatter_(0, flat_idx, label_flat.float())

        alphas = torch.empty(dense_shape, device=device, dtype=torch.float32)
        betas = torch.empty(dense_shape, device=device, dtype=torch.float32)
        llf = torch.zeros(B, device=device, dtype=torch.float32)
        llb = torch.zeros(B, device=device, dtype=torch.float32)

        tlen = torch.from_numpy(band.num_chunks.astype(np.int32)).to(device)
        ulen = torch.from_numpy(band.target_lens.astype(np.int32)).to(device)
        blo = torch.from_numpy(band.band_lo.reshape(-1)).to(device)
        bhi = torch.from_numpy(band.band_hi.reshape(-1)).to(device)

        threads = maxU1
        compute_alphas_kernel[B, threads](blank_lp, label_lp, alphas, llf, tlen, ulen, blo, bhi, maxT, maxU1)
        compute_betas_kernel[B, threads](blank_lp, label_lp, betas, llb, tlen, ulen, blo, bhi, maxT, maxU1)

        ctx.save_for_backward(blank_lp, label_lp, alphas, betas, llf, tlen, ulen, blo, bhi, flat_idx, b_t, u_t, targets)
        ctx.dims = (B, maxT, maxU1, blank_id, log_probs.shape[0], log_probs.shape[1])
        return -llf

    @staticmethod
    def backward(ctx, grad_out):
        (blank_lp, label_lp, alphas, betas, llf, tlen, ulen, blo, bhi, flat_idx, b_t, u_t, targets) = ctx.saved_tensors
        B, maxT, maxU1, blank_id, N, V1 = ctx.dims
        device = blank_lp.device

        grad_blank = torch.zeros_like(blank_lp)
        grad_label = torch.zeros_like(label_lp)
        total = B * maxT * maxU1
        threads = 256
        blocks = (total + threads - 1) // threads
        compute_grad_kernel[blocks, threads](
            grad_blank, grad_label, blank_lp, label_lp, alphas, betas, llf, tlen, ulen, blo, bhi, maxT, maxU1
        )

        # Back to the packed [N, V+1] the caller differentiates: only the blank
        # column and the one label column at each node are nonzero.
        gb = grad_blank.gather(0, flat_idx)
        gl = grad_label.gather(0, flat_idx)
        grad = torch.zeros((N, V1), device=device, dtype=torch.float32)
        grad[:, blank_id] = gb
        if targets.shape[1] > 0:
            u_clamped = u_t.clamp(max=targets.shape[1] - 1)
            lab = targets[b_t, u_clamped]
            valid = (u_t < targets.shape[1]).float()
            grad.scatter_add_(1, lab.unsqueeze(1), (gl * valid).unsqueeze(1))

        # grad_out is per UTTERANCE [B]; grad is per NODE [N, V+1]. Each node
        # belongs to exactly one utterance, so the upstream scale has to be
        # gathered through b_idx -- broadcasting [B] against [N, V+1] is a shape
        # error when N != B, and would be a SILENT mis-scaling if they happened
        # to match.
        scale = grad_out[b_t].unsqueeze(1)
        return grad * scale, None, None, None


def banded_rnnt_loss_cuda(log_probs: torch.Tensor, band: BandIndex, targets: torch.Tensor, blank_id: int):
    """Per-utterance NLL over the band, computed with the CUDA kernels."""
    return _BandedRNNTFunction.apply(log_probs, targets, band, blank_id)


def kernel_is_usable(band: BandIndex, device) -> Tuple[bool, str]:
    """Whether the kernel path can run this batch, and why not when it cannot."""
    if device.type != "cuda":
        return False, "not on CUDA"
    if not band.fits_in_one_block():
        return False, f"target length {band.maxU1 - 1} exceeds {MAX_THREADS_PER_BLOCK - 1} labels per block"
    if band.num_nodes == 0:
        return False, "empty band"
    if int(band.num_chunks.min()) == 0:
        return False, "an utterance has no chunks"
    return True, ""
