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

"""CUDA-backed two-head confidence RNN-T loss.

Split of responsibilities:

* the elementwise head composition and the gather of the two needed columns stay
  in PyTorch, so autograd routes gradients back to BOTH heads for free;
* the sequential O(T*U) lattice recursion runs in numba.

``confidence_rnnt_loss_cuda`` is therefore a drop-in for the pure-PyTorch
reference in ``nemo.collections.asr.losses.confidence_rnnt`` and is checked
against it in tests/collections/asr/test_confidence_rnnt_numba.py.
"""

from typing import Optional, Tuple

import torch

from nemo.collections.asr.parts.numba.confidence_rnnt.confidence_rnnt_kernel import (
    HAVE_NUMBA,
    compute_alphas_kernel,
    compute_betas_kernel,
    compute_grad_kernel,
)

__all__ = ["confidence_rnnt_loss_cuda", "kernel_is_usable"]

if HAVE_NUMBA:
    from numba import cuda


def kernel_is_usable(log_py: torch.Tensor) -> Tuple[bool, str]:
    """Whether the CUDA path can run for this input."""
    if not HAVE_NUMBA:
        return False, "numba.cuda is not available"
    if not log_py.is_cuda:
        return False, "inputs are not on a CUDA device"
    maxU1 = log_py.shape[2]
    if maxU1 > 1024:
        return False, f"U+1={maxU1} exceeds the 1024-thread block limit"
    return True, ""


def _compose_gathered(log_py: torch.Tensor, log_pc: torch.Tensor, targets: torch.Tensor, blank_id: int):
    """Compose the two heads DIRECTLY into the two columns the lattice needs.

    Returns ``blank_lp`` and ``label_lp``, both (B, T, U+1):

        blank_lp = logaddexp(log P_c(1) + log P_y(blank),  log P_c(0))
        label_lp =           log P_c(1) + log P_y(y_{u+1})

    Deliberately never materialises the full (B, T, U+1, V+1) effective
    distribution. Doing so costs two extra tensors of that size -- one for the
    add, one for a defensive clone -- which at a 151k vocabulary is ~11 GiB each
    and OOMs in backward on an 80 GiB card. Only these two columns are ever read
    by the recursion, so the rest is pure waste.
    """
    B, T, U1, _ = log_py.shape
    log_c1 = log_pc[..., 1]  # (B, T, U+1)
    log_c0 = log_pc[..., 0]

    blank_lp = torch.logaddexp(log_py[..., blank_id] + log_c1, log_c0).contiguous()

    U = U1 - 1
    if U > 0:
        idx = targets[:, :U].view(B, 1, U, 1).expand(B, T, U, 1)
        label_py = log_py[:, :, :U, :].gather(-1, idx).squeeze(-1)
        label_lp = label_py + log_c1[:, :, :U]
    else:
        label_lp = log_py.new_zeros((B, T, 0))
    # u == U has no token u+1; the recursion never reads it, but the kernel
    # indexes the full width so it must be finite.
    pad = log_py.new_zeros((B, T, 1))
    label_lp = torch.cat([label_lp, pad], dim=2).contiguous()
    return blank_lp, label_lp


class _ConfidenceRNNTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, blank_lp, label_lp, input_lens, target_lens):
        B, T, U1 = blank_lp.shape
        device = blank_lp.device

        work_dtype = torch.float32
        blank32 = blank_lp.detach().to(work_dtype).contiguous()
        label32 = label_lp.detach().to(work_dtype).contiguous()

        alphas = torch.empty((B, T, U1), device=device, dtype=work_dtype)
        betas = torch.empty((B, T, U1), device=device, dtype=work_dtype)
        llf = torch.empty((B,), device=device, dtype=work_dtype)
        llb = torch.empty((B,), device=device, dtype=work_dtype)

        tlen = input_lens.to(device=device, dtype=torch.int32).contiguous()
        ulen = target_lens.to(device=device, dtype=torch.int32).contiguous()

        stream = cuda.external_stream(torch.cuda.current_stream(device).cuda_stream)
        compute_alphas_kernel[B, U1, stream](
            cuda.as_cuda_array(blank32),
            cuda.as_cuda_array(label32),
            cuda.as_cuda_array(alphas),
            cuda.as_cuda_array(llf),
            cuda.as_cuda_array(tlen),
            cuda.as_cuda_array(ulen),
            T,
            U1,
        )
        compute_betas_kernel[B, U1, stream](
            cuda.as_cuda_array(blank32),
            cuda.as_cuda_array(label32),
            cuda.as_cuda_array(betas),
            cuda.as_cuda_array(llb),
            cuda.as_cuda_array(tlen),
            cuda.as_cuda_array(ulen),
            T,
            U1,
        )

        grad_blank = torch.zeros_like(blank32)
        grad_label = torch.zeros_like(label32)
        n = B * T * U1
        threads = 256
        blocks = max(1, (n + threads - 1) // threads)
        compute_grad_kernel[blocks, threads, stream](
            cuda.as_cuda_array(grad_blank),
            cuda.as_cuda_array(grad_label),
            cuda.as_cuda_array(blank32),
            cuda.as_cuda_array(label32),
            cuda.as_cuda_array(alphas),
            cuda.as_cuda_array(betas),
            cuda.as_cuda_array(llf),
            cuda.as_cuda_array(tlen),
            cuda.as_cuda_array(ulen),
            T,
            U1,
        )
        torch.cuda.current_stream(device).synchronize()

        ctx.save_for_backward(grad_blank.to(blank_lp.dtype), grad_label.to(label_lp.dtype))
        return (-llf).to(blank_lp.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        gb, gl = ctx.saved_tensors
        scale = grad_out.view(-1, 1, 1)
        return gb * scale, gl * scale, None, None


def confidence_rnnt_loss_cuda(
    log_py: torch.Tensor,
    log_pc: torch.Tensor,
    targets: torch.Tensor,
    input_lens: torch.Tensor,
    target_lens: torch.Tensor,
    blank_id: int,
    reduction: Optional[str] = "mean",
) -> torch.Tensor:
    """CUDA counterpart of ``losses.confidence_rnnt.confidence_rnnt_loss``."""
    ok, why = kernel_is_usable(log_py)
    if not ok:
        raise RuntimeError(f"confidence RNN-T CUDA path unusable: {why}")

    blank_lp, label_lp = _compose_gathered(log_py, log_pc, targets, blank_id)
    nll = _ConfidenceRNNTFunction.apply(blank_lp, label_lp, input_lens, target_lens)

    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll
