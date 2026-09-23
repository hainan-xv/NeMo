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

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch

from nemo.collections.asr.parts.numba.banded_rnnt.banded_rnnt_kernel import (
    NEG_INF,
    compute_alphas_kernel,
    compute_alphas_token_band_kernel,
    compute_betas_kernel,
    compute_betas_token_band_kernel,
    compute_grad_kernel,
    compute_grad_token_band_kernel,
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

        ctx.save_for_backward(
            blank_lp, label_lp, alphas, betas, llf, tlen, ulen, blo, bhi, flat_idx, b_t, u_t, targets
        )
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


@dataclass
class TokenBandIndex:
    """Band expressed TOKEN-CENTRICALLY: W candidate chunks per token row.

    The chunk-centric ``BandIndex`` stores, per chunk t, a contiguous range of u.
    This stores the transpose: per row u, the ``W = 2*band+1`` chunks it may sit
    at. The two describe the SAME node set --
    ``u in [S(t-band), S(t+band+1)) <=> a(u) in [t-band, t+band]`` -- but the
    token-centric form has a FIXED width, so the joint evaluates it as one
    regular [B, U+1, W, ...] tensor instead of a flat list of N arbitrary
    ``(b, t, u)`` triples that must be gathered per node.

    U + 1 ROWS, not U. Row u is "u tokens already emitted", so row U is the
    terminal row where the final blank is taken; it carries no token of its own
    and is pinned to the last chunk.

    Fields:
        base:  [B, U+1] int32. Node (u, o) sits at chunk ``t = base[u] + o`` with
               ``o in [0, W)``. Already includes the band's low offset, so the
               kernels never need to know ``band_side``.
        shift: [B, U+1] int32, ``base[u] - base[u-1] >= 0``. The recursion needs
               it: the blank predecessor of (u, o) is (u, o-1) and the EMIT
               predecessor is (u-1, o + shift[u]), since both name chunk t.
        valid: [B, U+1, W] bool, false where the chunk falls outside [0, T_b).
        target_lens: [B] int64, U_b.   num_chunks: [B] int64, T_b.
    """

    base: np.ndarray
    shift: np.ndarray
    valid: np.ndarray
    target_lens: np.ndarray
    num_chunks: np.ndarray
    band: int

    @property
    def width(self) -> int:
        """W, the number of candidate chunks per row.

        NOT ``2*band+1`` in general: ``band_side`` is one-sided for the 'later'
        and 'earlier' arms (the default here is 'later'), which gives W = band+1.
        Reading it off ``valid`` keeps this in step with what was actually built.
        """
        return int(self.valid.shape[-1])


def build_token_band(
    chunk_tokens_per_utt: Sequence[Sequence[Sequence[int]]], band: int, band_side: str = "both"
) -> TokenBandIndex:
    """Token-centric band geometry, EXACTLY the transpose of ``build_band_index``.

    Derived from the chunk-centric bounds rather than from a single alignment
    index a(u), for two reasons the first version got wrong:

    * The kernels test ``band_lo[t] <= u <= band_hi[t]`` -- INCLUSIVE at the top
      -- so the band reaches one row further than the half-open reading of
      ``[S(t-band), S(t+1))`` suggests. Chunk-initial rows therefore get one
      extra chunk at the low end.
    * With an EMPTY chunk, S(c) == S(c+1), and "the chunk containing token u" is
      no longer the only chunk u can sit at. a(u) cannot express that; the
      bounds can.

    So each row u is placed by
        A(u) = min{k : S(k) >= u}      Z(u) = max{k : S(k) <= u}
    giving ``t in [A(u) - hi_b - 1, Z(u) + lo_b]``, which is the chunk-centric
    condition ``S(t - lo_b) <= u <= S(t + hi_b + 1)`` solved for t.
    """
    if band_side not in ("both", "later", "earlier"):
        raise ValueError(f"band_side must be 'both', 'later' or 'earlier', got {band_side!r}")

    # lo_b / hi_b mirror build_band_index's choice of cum[] bounds.
    lo_b = band if band_side in ("both", "later") else 0
    hi_b = band if band_side in ("both", "earlier") else 0

    B = len(chunk_tokens_per_utt)
    num_chunks = np.array([len(c) for c in chunk_tokens_per_utt], dtype=np.int64)
    cums, target_lens = [], []
    for chunks in chunk_tokens_per_utt:
        counts = np.array([len(c) for c in chunks], dtype=np.int64)
        cum = np.concatenate(([0], np.cumsum(counts)))  # S, length T+1
        cums.append(cum)
        target_lens.append(int(cum[-1]))
    target_lens = np.array(target_lens, dtype=np.int64)
    Urows = int(target_lens.max()) + 1 if B else 1

    lows, highs = np.zeros((B, Urows), dtype=np.int64), np.zeros((B, Urows), dtype=np.int64)
    for b, cum in enumerate(cums):
        n = int(target_lens[b]) + 1  # rows 0..U
        u = np.arange(n, dtype=np.int64)
        A = np.searchsorted(cum, u, side="left")
        Z = np.searchsorted(cum, u, side="right") - 1
        lows[b, :n] = A - hi_b - 1
        highs[b, :n] = Z + lo_b

    span = int((highs - lows).max()) + 1 if B else 1
    W = max(span, 1)

    base = lows.astype(np.int32)
    shift = np.zeros((B, Urows), dtype=np.int32)
    shift[:, 1:] = base[:, 1:] - base[:, :-1]

    o_ix = np.arange(W, dtype=np.int64)[None, None, :]
    t = lows[:, :, None] + o_ix  # [B, Urows, W]
    u_ix = np.arange(Urows)[None, :, None]
    valid = (
        (t >= 0) & (t < num_chunks[:, None, None]) & (t <= highs[:, :, None]) & (u_ix <= target_lens[:, None, None])
    )

    return TokenBandIndex(
        base=base, shift=shift, valid=valid, target_lens=target_lens, num_chunks=num_chunks, band=band
    )


class _TokenBandRNNTFunction(torch.autograd.Function):
    """NLL over the token-centric band, analytic gradient from alpha/beta.

    Simpler than ``_BandedRNNTFunction`` because the band is already a DENSE
    ``[B, U+1, W, V+1]`` tensor: the chunk-centric version has to scatter N
    scattered nodes into a flat ``[B*T*U1]`` buffer and gather them back, and
    none of that exists here.
    """

    @staticmethod
    def forward(ctx, log_probs, targets, band: "TokenBandIndex", blank_id: int):
        device = log_probs.device
        B, U1, W, V1 = log_probs.shape

        valid = torch.from_numpy(band.valid).to(device)
        ulen_t = torch.from_numpy(band.target_lens.astype(np.int64)).to(device)

        blank_lp = log_probs[..., blank_id].float()
        if targets.shape[1] > 0:
            pad = torch.zeros(B, U1 - targets.shape[1], dtype=targets.dtype, device=device)
            tgt_full = torch.cat([targets, pad], dim=1)  # [B, U1]; row U1-1 unused
            gather_ix = tgt_full[:, :, None, None].expand(B, U1, W, 1)
            label_lp = log_probs.gather(3, gather_ix).squeeze(3).float()
        else:
            tgt_full = torch.zeros(B, U1, dtype=torch.long, device=device)
            label_lp = torch.full_like(blank_lp, NEG_INF)

        # A row at or past U has no outgoing label, and a node outside the band
        # has no probability at all.
        rows = torch.arange(U1, device=device)[None, :, None]
        label_lp = torch.where(rows < ulen_t[:, None, None], label_lp, torch.full_like(label_lp, NEG_INF))
        blank_lp = torch.where(valid, blank_lp, torch.full_like(blank_lp, NEG_INF))
        label_lp = torch.where(valid, label_lp, torch.full_like(label_lp, NEG_INF))

        blank_lp = blank_lp.reshape(-1).contiguous()
        label_lp = label_lp.reshape(-1).contiguous()
        alphas = torch.empty_like(blank_lp)
        betas = torch.empty_like(blank_lp)
        llf = torch.zeros(B, device=device, dtype=torch.float32)
        llb = torch.zeros(B, device=device, dtype=torch.float32)

        tlen = torch.from_numpy(band.num_chunks.astype(np.int32)).to(device)
        ulen = torch.from_numpy(band.target_lens.astype(np.int32)).to(device)
        base = torch.from_numpy(band.base.reshape(-1)).to(device)
        shift = torch.from_numpy(band.shift.reshape(-1)).to(device)

        compute_alphas_token_band_kernel[B, U1](blank_lp, label_lp, alphas, llf, tlen, ulen, base, shift, U1, W)
        compute_betas_token_band_kernel[B, U1](blank_lp, label_lp, betas, llb, tlen, ulen, base, shift, U1, W)

        ctx.save_for_backward(blank_lp, label_lp, alphas, betas, llf, tlen, ulen, base, shift, tgt_full)
        ctx.dims = (B, U1, W, V1, blank_id)
        return -llf

    @staticmethod
    def backward(ctx, grad_out):
        blank_lp, label_lp, alphas, betas, llf, tlen, ulen, base, shift, tgt_full = ctx.saved_tensors
        B, U1, W, V1, blank_id = ctx.dims
        device = blank_lp.device

        grad_blank = torch.zeros_like(blank_lp)
        grad_label = torch.zeros_like(label_lp)
        total = B * U1 * W
        threads = 256
        blocks = (total + threads - 1) // threads
        compute_grad_token_band_kernel[blocks, threads](
            grad_blank, grad_label, blank_lp, label_lp, alphas, betas, llf, tlen, ulen, base, shift, U1, W
        )

        grad = torch.zeros(B, U1, W, V1, device=device, dtype=torch.float32)
        grad[..., blank_id] = grad_blank.view(B, U1, W)
        grad.scatter_add_(3, tgt_full[:, :, None, None].expand(B, U1, W, 1), grad_label.view(B, U1, W, 1))

        # grad_out is per UTTERANCE [B]; grad is per NODE.
        return grad * grad_out.view(B, 1, 1, 1), None, None, None


def token_band_rnnt_loss(log_probs: torch.Tensor, targets: torch.Tensor, band: "TokenBandIndex", blank_id: int):
    """NLL per utterance over the token-centric band.

    Args:
        log_probs: [B, U+1, W, V+1] from ``RNNTAttJoint.joint_on_token_band``.
        targets:   [B, U] token ids.
        band:      from :func:`build_token_band`.
    """
    return _TokenBandRNNTFunction.apply(log_probs, targets, band, blank_id)


class _ChunkBandRNNTFunction(torch.autograd.Function):
    """NLL over the band laid out CHUNK-MAJOR: logits are [B, T, W, V+1].

    Reuses ``compute_alphas/betas/grad_kernel`` unchanged. Those kernels read a
    dense [B, T, U+1] array of blank/label log-probs, and that array is built
    here with a scatter of B*T*W scalars -- NOT by gathering an [N, V+1] block.
    The [N, V+1] gather is what the triple-list path pays; skipping it is most of
    why this layout is cheaper.
    """

    @staticmethod
    def forward(ctx, log_probs, targets, band: BandIndex, blank_id: int, u_idx, valid):
        device = log_probs.device
        B, T, W, V1 = log_probs.shape
        maxT, maxU1 = band.maxT, band.maxU1

        u_cl = u_idx.clamp(0, max(targets.shape[1] - 1, 0))
        blank_bt = log_probs[..., blank_id]
        if targets.shape[1] > 0:
            lab = targets.gather(1, u_cl.reshape(B, -1)).view(B, T, W)  # TOKEN IDS, not positions
            label_bt = log_probs.gather(3, lab.unsqueeze(-1)).squeeze(-1)
            label_bt = torch.where(u_idx < targets.shape[1], label_bt, torch.full_like(label_bt, NEG_INF))
        else:
            lab = torch.zeros(B, T, W, dtype=torch.long, device=device)
            label_bt = torch.full_like(blank_bt, NEG_INF)
        blank_bt = torch.where(valid, blank_bt, torch.full_like(blank_bt, NEG_INF))
        label_bt = torch.where(valid, label_bt, torch.full_like(label_bt, NEG_INF))

        t_ar = torch.arange(T, device=device).view(1, T, 1).expand(B, T, W)
        b_ar = torch.arange(B, device=device).view(B, 1, 1).expand(B, T, W)
        # Invalid slots go to a SCRATCH index one past the end, never to 0.
        # Sending them to 0 makes every masked node scatter NEG_INF onto the
        # start node (b=0, t=0, u=0) and the lattice dies -- the loss comes back
        # as +1e28 rather than as an error.
        n_dense = B * maxT * maxU1
        flat_idx = ((b_ar * maxT + t_ar) * maxU1 + u_idx.clamp(0, maxU1 - 1)).reshape(-1)
        scratch = torch.full_like(flat_idx, n_dense)
        flat_idx = torch.where(valid.reshape(-1), flat_idx, scratch)

        blank_lp = torch.full((n_dense + 1,), NEG_INF, device=device, dtype=torch.float32)
        label_lp = torch.full((n_dense + 1,), NEG_INF, device=device, dtype=torch.float32)
        blank_lp.scatter_(0, flat_idx, blank_bt.reshape(-1).float())
        label_lp.scatter_(0, flat_idx, label_bt.reshape(-1).float())
        blank_lp = blank_lp[:n_dense].contiguous()
        label_lp = label_lp[:n_dense].contiguous()
        dense_shape = (n_dense,)

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

        ctx.save_for_backward(blank_lp, label_lp, alphas, betas, llf, tlen, ulen, blo, bhi, flat_idx, valid, lab)
        ctx.dims = (B, T, W, V1, maxT, maxU1, blank_id)
        return -llf

    @staticmethod
    def backward(ctx, grad_out):
        (blank_lp, label_lp, alphas, betas, llf, tlen, ulen, blo, bhi, flat_idx, valid, lab) = ctx.saved_tensors
        B, T, W, V1, maxT, maxU1, blank_id = ctx.dims
        device = blank_lp.device

        grad_blank = torch.zeros_like(blank_lp)
        grad_label = torch.zeros_like(label_lp)
        total = B * maxT * maxU1
        threads = 256
        blocks = (total + threads - 1) // threads
        compute_grad_kernel[blocks, threads](
            grad_blank, grad_label, blank_lp, label_lp, alphas, betas, llf, tlen, ulen, blo, bhi, maxT, maxU1
        )

        grad_blank = torch.cat([grad_blank, grad_blank.new_zeros(1)])
        grad_label = torch.cat([grad_label, grad_label.new_zeros(1)])
        gb = grad_blank.gather(0, flat_idx).view(B, T, W) * valid
        gl = grad_label.gather(0, flat_idx).view(B, T, W) * valid

        grad = torch.zeros(B, T, W, V1, device=device, dtype=torch.float32)
        grad[..., blank_id] = gb
        grad.scatter_add_(3, lab.unsqueeze(-1), gl.unsqueeze(-1))
        return grad * grad_out.view(B, 1, 1, 1), None, None, None, None, None


def chunk_band_rnnt_loss(log_probs, targets, band: BandIndex, blank_id: int, u_idx, valid):
    """NLL per utterance over the chunk-major band. ``log_probs`` is
    [B, T, W, V+1] from ``RNNTAttJoint.joint_on_chunk_band``."""
    return _ChunkBandRNNTFunction.apply(log_probs, targets, band, blank_id, u_idx, valid)
