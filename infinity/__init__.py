"""Infinity: Single-GPU training toolkit with CPU-backed parameters."""

from .model.cpu_master import CPUMasterModel
from .config.training import CPUMasterConfig
from .data.datasets import MetaMathDataset, collate_fn
from .ops.attention import FlashAttentionLayer

__version__ = "0.1.0"

__all__ = [
    "CPUMasterModel",
    "CPUMasterConfig",
    "MetaMathDataset",
    "collate_fn",
    "FlashAttentionLayer",
]
