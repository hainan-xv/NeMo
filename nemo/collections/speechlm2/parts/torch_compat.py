# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Compatibility imports for composable FSDP across PyTorch releases."""

try:
    from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method
except ImportError:
    # PyTorch 2.5 keeps composable FSDP in the private namespace. These symbols
    # moved to torch.distributed.fsdp in PyTorch 2.6.
    from torch.distributed._composable.fsdp import fully_shard, register_fsdp_forward_method

__all__ = ["fully_shard", "register_fsdp_forward_method"]
