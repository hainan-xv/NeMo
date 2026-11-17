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

"""CUDA kernels for the two-head confidence RNN-T lattice.

The lattice recursion is ordinary RNN-T; what differs is only which quantity
sits on each arc. Composition of the two heads is elementwise and therefore
stays in PyTorch (differentiable, cheap); these kernels consume the two GATHERED
columns the recursion actually needs:

    blank_lp[b, t, u]  = log P_eff(blank | t, u)
    label_lp[b, t, u]  = log P_eff(y_{u+1} | t, u)

both (B, T, U+1), rather than the full (B, T, U+1, V+1). That is the same shape
trick the banded kernel uses, and it keeps the expensive part -- the O(T*U)
sequential wavefront -- in numba while leaving the O(T*U*V) elementwise work to
cuBLAS-backed torch ops.

Gradients are returned with respect to these two columns; the chain rule back to
the token and confidence heads is handled by autograd outside.
"""

import math

from nemo.core.utils import numba_utils

__all__ = [
    "compute_alphas_kernel",
    "compute_betas_kernel",
    "compute_grad_kernel",
]

try:
    from numba import cuda

    HAVE_NUMBA = True
except (ImportError, ModuleNotFoundError):
    HAVE_NUMBA = False
    cuda = None

NEG_INF = -1e30


if HAVE_NUMBA:

    @cuda.jit(device=True, inline=True)
    def _log_sum_exp(a, b):
        """Numerically stable log(exp(a) + exp(b)), NEG_INF-safe."""
        if a <= NEG_INF:
            return b
        if b <= NEG_INF:
            return a
        if a > b:
            return a + math.log1p(math.exp(b - a))
        return b + math.log1p(math.exp(a - b))

    @cuda.jit()
    def compute_alphas_kernel(blank_lp, label_lp, alphas, llForward, tlen, ulen, maxT: int, maxU1: int):
        """Forward variable over the full RNN-T lattice.

        One block per batch element, one thread per u. Threads advance together
        along anti-diagonals so that both predecessors of (t, u) -- namely
        (t-1, u) and (t, u-1) -- are already final when it is computed.
        """
        b = cuda.blockIdx.x
        u = cuda.threadIdx.x
        T = tlen[b]
        U1 = ulen[b] + 1

        if u < maxU1:
            alphas[b, 0, u] = NEG_INF
        cuda.syncthreads()
        if u == 0:
            alphas[b, 0, 0] = 0.0
        cuda.syncthreads()

        # Diagonal d = t + u. d = 0 is the origin, already seeded.
        for d in range(1, maxT + maxU1 - 1):
            t = d - u
            if 0 <= t < T and u < U1:
                acc = NEG_INF
                if t > 0:
                    acc = _log_sum_exp(acc, alphas[b, t - 1, u] + blank_lp[b, t - 1, u])
                if u > 0:
                    acc = _log_sum_exp(acc, alphas[b, t, u - 1] + label_lp[b, t, u - 1])
                if t == 0 and u == 0:
                    acc = 0.0
                alphas[b, t, u] = acc
            cuda.syncthreads()

        # A complete path consumes every chunk and emits every label, then takes
        # one terminating blank out of (T-1, U).
        if u == 0:
            llForward[b] = alphas[b, T - 1, U1 - 1] + blank_lp[b, T - 1, U1 - 1]

    @cuda.jit()
    def compute_betas_kernel(blank_lp, label_lp, betas, llBackward, tlen, ulen, maxT: int, maxU1: int):
        """Backward variable. Mirror image of the alpha wavefront."""
        b = cuda.blockIdx.x
        u = cuda.threadIdx.x
        T = tlen[b]
        U1 = ulen[b] + 1

        if u < maxU1:
            betas[b, T - 1, u] = NEG_INF
        cuda.syncthreads()
        if u == 0:
            betas[b, T - 1, U1 - 1] = blank_lp[b, T - 1, U1 - 1]
        cuda.syncthreads()

        for d in range(1, maxT + maxU1 - 1):
            t = (T - 1) - (d - ((U1 - 1) - u))
            uu = u
            if 0 <= t < T and uu < U1 and not (t == T - 1 and uu == U1 - 1):
                acc = NEG_INF
                if t + 1 < T:
                    acc = _log_sum_exp(acc, betas[b, t + 1, uu] + blank_lp[b, t, uu])
                if uu + 1 < U1:
                    acc = _log_sum_exp(acc, betas[b, t, uu + 1] + label_lp[b, t, uu])
                betas[b, t, uu] = acc
            cuda.syncthreads()

        if u == 0:
            llBackward[b] = betas[b, 0, 0]

    @cuda.jit()
    def compute_grad_kernel(
        grad_blank, grad_label, blank_lp, label_lp, alphas, betas, logll, tlen, ulen, maxT: int, maxU1: int
    ):
        """d(NLL)/d(log P_eff) on each arc.

        For an arc a with probability p_a, the posterior occupancy of that arc is
        exp(alpha(src) + log p_a + beta(dst) - logP(Y|X)); the gradient of the
        NEGATIVE log-likelihood with respect to log p_a is minus that.
        """
        idx = cuda.grid(1)
        total = cuda.gridsize(1)
        n = blank_lp.shape[0] * maxT * maxU1

        while idx < n:
            u = idx % maxU1
            t = (idx // maxU1) % maxT
            b = idx // (maxU1 * maxT)

            T = tlen[b]
            U1 = ulen[b] + 1
            if t < T and u < U1:
                ll = logll[b]
                # blank arc (t, u) -> (t+1, u); the arc out of the final node is
                # the terminating blank and has beta == 0 by definition.
                if t + 1 < T:
                    grad_blank[b, t, u] = -math.exp(alphas[b, t, u] + blank_lp[b, t, u] + betas[b, t + 1, u] - ll)
                elif t == T - 1 and u == U1 - 1:
                    grad_blank[b, t, u] = -math.exp(alphas[b, t, u] + blank_lp[b, t, u] - ll)
                else:
                    grad_blank[b, t, u] = 0.0
                # label arc (t, u) -> (t, u+1)
                if u + 1 < U1:
                    grad_label[b, t, u] = -math.exp(alphas[b, t, u] + label_lp[b, t, u] + betas[b, t, u + 1] - ll)
                else:
                    grad_label[b, t, u] = 0.0
            elif t < maxT and u < maxU1:
                grad_blank[b, t, u] = 0.0
                grad_label[b, t, u] = 0.0

            idx += total

else:  # pragma: no cover - numba absent

    def _unavailable(*args, **kwargs):
        raise RuntimeError("numba.cuda is not available; the confidence RNN-T CUDA path cannot run")

    compute_alphas_kernel = compute_betas_kernel = compute_grad_kernel = _unavailable
