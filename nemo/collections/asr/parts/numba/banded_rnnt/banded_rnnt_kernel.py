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

"""CUDA kernels for the banded RNN-T loss.

WHY THIS EXISTS. The reference implementation in
``nemo/collections/asr/losses/banded_rnnt.py`` scores far FEWER lattice nodes
than full RNN-T -- ~3U+T against T*U -- and was still measurably SLOWER in
training: 1.73 it/s against 2.40 it/s for warprnnt_numba on an otherwise
identical arm (same 1k vocabulary, same batch, same nodes, same encoder). Node
count was never the bottleneck. The reference pays, per training step:

  * one host->device copy AND an implicit sync per anti-diagonal, because it
    built ``torch.tensor(diag, device=cuda)`` INSIDE the recursion loop, and
    there are T+U diagonals;
  * four more H2D copies of Python lists for the index vectors;
  * an out-of-place ``index_put`` allocating a fresh length-N tensor per
    diagonal;
  * a pure-Python lattice build with a dict keyed on ``(b, t, u)`` tuples.

So it ran T+U sequential steps of ~6 tiny launch-bound kernels, against one
fused kernel for warprnnt_numba. These kernels remove that entirely.

STRUCTURE mirrors ``cuda_utils/gpu_rnnt_kernel.py`` deliberately: one block per
utterance, one thread per label position u, and the anti-diagonal wavefront
``t = n - u`` with a ``syncthreads`` per step -- alpha(t,u) depends only on
(t-1,u) and (t,u-1), both of which have smaller t+u.

THE ONE DIFFERENCE is the band mask. Nodes outside ``[band_lo[b,t],
band_hi[b,t]]`` are never written and stay at -inf, so reading a predecessor
that falls outside the band contributes nothing and needs no special case.

WHAT IS AND IS NOT DENSE. ``alphas``/``betas`` are dense ``[B, T, U+1]``, which
is a few MB and not what the band exists to avoid; the thing that must never be
materialised is ``[B, T, U, V+1]``, and it is not -- the caller evaluates the
joint at band nodes only and passes just the two log-probabilities each node
needs (blank, and the one label leaving it).
"""

import math

from numba import cuda

# Matches banded_rnnt.NEG_INF. A finite sentinel rather than -inf so that
# -inf + -inf stays representable and no NaN appears in the recursion.
NEG_INF = -1e30


# NOTE: `math` is imported at MODULE level on purpose. numba's CUDA target
# compiles the function bytecode directly and rejects IMPORT_NAME, so an
# `import math` inside a device function fails at first launch with
# UnsupportedBytecodeError -- not at definition time, which makes it look like a
# kernel bug rather than an import placement one.


@cuda.jit(device=True, inline=True)
def _log1p_exp(x):
    """log(1 + exp(x)) for x <= 0, without underflowing for very negative x."""
    if x < -30.0:
        return 0.0
    return math.log(1.0 + math.exp(x))


@cuda.jit(device=True, inline=True)
def _log_sum_exp(a, b):
    """log(exp(a) + exp(b)), stable, and -inf-safe for the band's padding."""
    if a <= NEG_INF:
        return b
    if b <= NEG_INF:
        return a
    if a > b:
        return a + _log1p_exp(b - a)
    return b + _log1p_exp(a - b)


@cuda.jit()
def compute_alphas_kernel(
    blank_lp,  # [B, T, U1] log P(blank | t, u), -inf outside the band
    label_lp,  # [B, T, U1] log P(target[u] | t, u), -inf outside the band or at u == U
    alphas,  # [B, T, U1] OUT
    ll,  # [B] OUT, log-likelihood of the forward pass
    tlen,  # [B] number of CHUNKS
    ulen,  # [B] number of labels
    band_lo,  # [B, T] first u in the band at chunk t
    band_hi,  # [B, T] last  u in the band at chunk t
    maxT: int,
    maxU1: int,
):
    """Forward variable over the banded lattice. One block per utterance."""
    b = cuda.blockIdx.x
    u = cuda.threadIdx.x

    T = tlen[b]
    U = ulen[b]
    off = b * maxT * maxU1
    boff = b * maxT

    # Every node starts unreachable; the band mask is then implicit.
    for t in range(maxT):
        alphas[off + t * maxU1 + u] = NEG_INF
    cuda.syncthreads()

    if u == 0:
        alphas[off] = 0.0
    cuda.syncthreads()

    # Anti-diagonal wavefront. Max t+u is (T-1)+U, hence the T+U bound.
    for n in range(1, T + U):
        t = n - u
        if 0 <= t < T and band_lo[boff + t] <= u <= band_hi[boff + t]:
            acc = NEG_INF
            # Arrive by emitting blank at (t-1, u).
            if t > 0 and band_lo[boff + t - 1] <= u <= band_hi[boff + t - 1]:
                acc = alphas[off + (t - 1) * maxU1 + u] + blank_lp[off + (t - 1) * maxU1 + u]
            # Arrive by emitting target[u-1] at (t, u-1).
            if u > 0 and band_lo[boff + t] <= u - 1 <= band_hi[boff + t]:
                acc = _log_sum_exp(acc, alphas[off + t * maxU1 + u - 1] + label_lp[off + t * maxU1 + u - 1])
            alphas[off + t * maxU1 + u] = acc
        cuda.syncthreads()

    # A path ends having consumed every chunk and emitted every label; standard
    # RNN-T termination then takes one final blank from (T-1, U).
    if u == 0:
        ll[b] = alphas[off + (T - 1) * maxU1 + U] + blank_lp[off + (T - 1) * maxU1 + U]


@cuda.jit()
def compute_betas_kernel(
    blank_lp,
    label_lp,
    betas,  # [B, T, U1] OUT
    ll,  # [B] OUT, log-likelihood of the backward pass (equals the forward one)
    tlen,
    ulen,
    band_lo,
    band_hi,
    maxT: int,
    maxU1: int,
):
    """Backward variable. Same shape and wavefront as the forward pass, reversed."""
    b = cuda.blockIdx.x
    u = cuda.threadIdx.x

    T = tlen[b]
    U = ulen[b]
    off = b * maxT * maxU1
    boff = b * maxT

    for t in range(maxT):
        betas[off + t * maxU1 + u] = NEG_INF
    cuda.syncthreads()

    if u == 0:
        betas[off + (T - 1) * maxU1 + U] = blank_lp[off + (T - 1) * maxU1 + U]
    cuda.syncthreads()

    for n in range(T + U - 2, -1, -1):
        t = n - u
        if 0 <= t < T and band_lo[boff + t] <= u <= band_hi[boff + t] and not (t == T - 1 and u == U):
            acc = NEG_INF
            # Leave by blank into (t+1, u).
            if t + 1 < T and band_lo[boff + t + 1] <= u <= band_hi[boff + t + 1]:
                acc = betas[off + (t + 1) * maxU1 + u] + blank_lp[off + t * maxU1 + u]
            # Leave by target[u] into (t, u+1).
            if u + 1 <= U and band_lo[boff + t] <= u + 1 <= band_hi[boff + t]:
                acc = _log_sum_exp(acc, betas[off + t * maxU1 + u + 1] + label_lp[off + t * maxU1 + u])
            betas[off + t * maxU1 + u] = acc
        cuda.syncthreads()

    if u == 0:
        ll[b] = betas[off]


@cuda.jit()
def compute_grad_kernel(
    grad_blank,  # [B, T, U1] OUT, dL/d log P(blank | t, u)
    grad_label,  # [B, T, U1] OUT, dL/d log P(target[u] | t, u)
    blank_lp,
    label_lp,
    alphas,
    betas,
    ll,  # [B] forward log-likelihood
    tlen,
    ulen,
    band_lo,
    band_hi,
    maxT: int,
    maxU1: int,
):
    """Gradient of the NLL w.r.t. the two log-probabilities at each band node.

    The loss is ``-ll``, and ``d ll / d log P(x | t,u) = exp(alpha(t,u) +
    log P(x | t,u) + beta(next) - ll)``, so the gradient of the LOSS is the
    negative of that. Only the blank and the single outgoing label have nonzero
    gradient at a node, which is why the caller can scatter these two vectors
    into an otherwise-zero ``[N, V+1]`` tensor.
    """
    idx = cuda.grid(1)
    total = cuda.gridsize(1)

    while idx < blank_lp.shape[0]:
        b = idx // (maxT * maxU1)
        rem = idx - b * maxT * maxU1
        t = rem // maxU1
        u = rem - t * maxU1

        T = tlen[b]
        U = ulen[b]
        boff = b * maxT

        gb = 0.0
        gl = 0.0
        if t < T and u <= U and band_lo[boff + t] <= u <= band_hi[boff + t]:
            a = alphas[idx]
            if a > NEG_INF:
                denom = ll[b]
                # Blank: into (t+1, u), or the terminating blank at (T-1, U).
                if t == T - 1 and u == U:
                    gb = -math.exp(a + blank_lp[idx] - denom)
                elif t + 1 < T and band_lo[boff + t + 1] <= u <= band_hi[boff + t + 1]:
                    nb = betas[b * maxT * maxU1 + (t + 1) * maxU1 + u]
                    if nb > NEG_INF:
                        gb = -math.exp(a + blank_lp[idx] + nb - denom)
                # Label: into (t, u+1).
                if u + 1 <= U and band_lo[boff + t] <= u + 1 <= band_hi[boff + t]:
                    nl = betas[b * maxT * maxU1 + t * maxU1 + u + 1]
                    if nl > NEG_INF:
                        gl = -math.exp(a + label_lp[idx] + nl - denom)

        grad_blank[idx] = gb
        grad_label[idx] = gl
        idx += total


# ---------------------------------------------------------------------------
# TOKEN-CENTRIC BAND
#
# The kernels above index the band chunk-centrically: per chunk t, a contiguous
# [band_lo, band_hi) range of u, marched along anti-diagonals t + u. These index
# it token-centrically instead: node (u, o) sits at chunk t = align[u] + o with
# o in [0, W), W = 2*band+1 fixed.
#
# The recursion is CLEANER in these coordinates. For node (u, o):
#     blank predecessor  (t-1, u)   = (u,   o-1)
#     emit  predecessor  (t, u-1)   = (u-1, o + shift[u])     shift[u] = a(u)-a(u-1)
# because both name the same chunk. So the only extra state a thread needs is a
# per-token integer shift, instead of a pair of band bounds per chunk.
#
# The wavefront variable is n = t + u = align[u] + o + u, NOT u + o: both
# predecessors sit at n-1 only under the former (with shift > 0, u + o puts the
# emit predecessor on the SAME diagonal, which would read a value not yet
# written).
# ---------------------------------------------------------------------------


@cuda.jit()
def compute_alphas_token_band_kernel(blank_lp, label_lp, alphas, llf, tlen, ulen, base, shift, maxU1: int, W: int):
    """Forward pass on the token-centric band. One block per utterance, one
    thread per token row; the wavefront is ``n = t + u``."""
    b = cuda.blockIdx.x
    u = cuda.threadIdx.x
    T = tlen[b]
    U = ulen[b]
    off = b * maxU1 * W
    boff = b * maxU1

    if u <= U:
        for o in range(W):
            alphas[off + u * W + o] = NEG_INF
    cuda.syncthreads()

    if u == 0:
        # Start state (t=0, u=0): o such that base[0] + o == 0.
        o0 = -base[boff]
        if 0 <= o0 < W:
            alphas[off + o0] = 0.0
    cuda.syncthreads()

    nmax = (T - 1) + U
    for n in range(1, nmax + 1):
        if u <= U:
            o = n - base[boff + u] - u
            t = base[boff + u] + o
            if 0 <= o < W and 0 <= t < T:
                val = NEG_INF
                if o > 0:  # blank from (u, o-1)
                    pv = alphas[off + u * W + o - 1]
                    if pv > NEG_INF:
                        val = pv + blank_lp[off + u * W + o - 1]
                if u > 0:  # emit from (u-1, o + shift[u]) -- same chunk t
                    op = o + shift[boff + u]
                    if 0 <= op < W:
                        pv = alphas[off + (u - 1) * W + op]
                        if pv > NEG_INF:
                            cand = pv + label_lp[off + (u - 1) * W + op]
                            val = cand if val == NEG_INF else _log_sum_exp(val, cand)
                if val > NEG_INF:
                    alphas[off + u * W + o] = val
        cuda.syncthreads()

    if u == 0:
        o_end = (T - 1) - base[boff + U]
        if 0 <= o_end < W and alphas[off + U * W + o_end] > NEG_INF:
            llf[b] = alphas[off + U * W + o_end] + blank_lp[off + U * W + o_end]
        else:
            # The band does not cover the terminal node; the caller drops this
            # utterance rather than propagating an inf.
            llf[b] = NEG_INF


@cuda.jit()
def compute_betas_token_band_kernel(blank_lp, label_lp, betas, llb, tlen, ulen, base, shift, maxU1: int, W: int):
    """Backward pass, the mirror of the forward: successors are (u, o+1) for
    blank and (u+1, o - shift[u+1]) for emit."""
    b = cuda.blockIdx.x
    u = cuda.threadIdx.x
    T = tlen[b]
    U = ulen[b]
    off = b * maxU1 * W
    boff = b * maxU1

    if u <= U:
        for o in range(W):
            betas[off + u * W + o] = NEG_INF
    cuda.syncthreads()

    if u == 0:
        o_end = (T - 1) - base[boff + U]
        if 0 <= o_end < W:
            betas[off + U * W + o_end] = blank_lp[off + U * W + o_end]
    cuda.syncthreads()

    nmax = (T - 1) + U
    for n in range(nmax - 1, -1, -1):
        if u <= U:
            o = n - base[boff + u] - u
            t = base[boff + u] + o
            if 0 <= o < W and 0 <= t < T:
                val = NEG_INF
                if o + 1 < W:  # blank into (u, o+1)
                    nv = betas[off + u * W + o + 1]
                    if nv > NEG_INF:
                        val = blank_lp[off + u * W + o] + nv
                if u < U:  # emit into (u+1, o - shift[u+1])
                    on = o - shift[boff + u + 1]
                    if 0 <= on < W:
                        nv = betas[off + (u + 1) * W + on]
                        if nv > NEG_INF:
                            cand = label_lp[off + u * W + o] + nv
                            val = cand if val == NEG_INF else _log_sum_exp(val, cand)
                if val > NEG_INF:
                    betas[off + u * W + o] = val
        cuda.syncthreads()

    if u == 0:
        o0 = -base[boff]
        llb[b] = betas[off + o0] if 0 <= o0 < W else NEG_INF


@cuda.jit()
def compute_grad_token_band_kernel(
    grad_blank,  # [B, U+1, W] OUT
    grad_label,  # [B, U+1, W] OUT
    blank_lp,
    label_lp,
    alphas,
    betas,
    ll,
    tlen,
    ulen,
    base,
    shift,
    maxU1: int,
    W: int,
):
    """Same identity as ``compute_grad_kernel``: the gradient of the LOSS w.r.t.
    a node's log-probability is ``-exp(alpha + logp + beta(next) - ll)``, nonzero
    only for the blank and the single outgoing label."""
    idx = cuda.grid(1)
    total = cuda.gridsize(1)

    while idx < blank_lp.shape[0]:
        b = idx // (maxU1 * W)
        rem = idx - b * maxU1 * W
        u = rem // W
        o = rem - u * W

        T = tlen[b]
        U = ulen[b]
        boff = b * maxU1
        off = b * maxU1 * W

        gb = 0.0
        gl = 0.0
        if u <= U:
            t = base[boff + u] + o
            if 0 <= t < T:
                a = alphas[idx]
                if a > NEG_INF:
                    denom = ll[b]
                    # Blank: into (u, o+1), or the terminating blank at (T-1, U).
                    if t == T - 1 and u == U:
                        gb = -math.exp(a + blank_lp[idx] - denom)
                    elif o + 1 < W:
                        nb = betas[off + u * W + o + 1]
                        if nb > NEG_INF:
                            gb = -math.exp(a + blank_lp[idx] + nb - denom)
                    # Label: into (u+1, o - shift[u+1]).
                    if u < U:
                        on = o - shift[boff + u + 1]
                        if 0 <= on < W:
                            nl = betas[off + (u + 1) * W + on]
                            if nl > NEG_INF:
                                gl = -math.exp(a + label_lp[idx] + nl - denom)

        grad_blank[idx] = gb
        grad_label[idx] = gl
        idx += total
