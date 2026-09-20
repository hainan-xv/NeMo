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
