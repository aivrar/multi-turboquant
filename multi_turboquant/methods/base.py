# SPDX-License-Identifier: MIT
"""Abstract base protocol for KV cache compression methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from ..config import CacheMethod, MethodFamily


@dataclass(frozen=True)
class MethodInfo:
    """Static metadata about a compression method."""
    method: CacheMethod
    family: MethodFamily
    bits: float
    requires_calibration: bool
    supports_asymmetric: bool
    supported_head_dims: tuple[int, ...]
    transform_name: str
    description: str
    fma_count: int          # FMAs per encode for head_dim=128
    param_count: int        # learnable/fixed parameters


@dataclass
class CompressedKV:
    """Container for compressed KV cache data."""
    data: torch.Tensor           # packed byte tensor
    head_dim: int                # original head dimension
    method: CacheMethod          # method used for compression
    metadata: dict[str, Any] | None = None  # method-specific metadata


class CompressionMethod(ABC):
    """Abstract base class for KV cache compression methods.

    All methods implement encode/decode for KV vectors, plus metadata
    about their capabilities and requirements.

    Implementations must be stateless with respect to the compression
    itself — all state (rotation matrices, codebooks, etc.) is derived
    from configuration and cached internally.
    """

    @abstractmethod
    def info(self) -> MethodInfo:
        """Return static metadata about this method."""
        ...

    @abstractmethod
    def encode(
        self,
        x: torch.Tensor,
        *,
        head_indices: torch.Tensor | None = None,
        layer_idx: int = 0,
    ) -> CompressedKV:
        """Compress KV vectors.

        Args:
            x: Input tensor of shape [..., num_heads, head_dim] in float16/32.
            head_indices: Optional per-head group indices for methods that
                need them (e.g., TurboQuant outlier masks).
            layer_idx: Layer index for methods with per-layer calibration.

        Returns:
            CompressedKV with packed byte data.
        """
        ...

    @abstractmethod
    def decode(
        self,
        compressed: CompressedKV,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Decompress KV vectors back to original space.

        Args:
            compressed: CompressedKV from encode().
            dtype: Target output dtype.

        Returns:
            Reconstructed tensor of shape [..., num_heads, head_dim].
        """
        ...

    @abstractmethod
    def packed_dim(self, head_dim: int) -> int:
        """Return the packed byte dimension for a given head_dim.

        This is the last dimension size of the compressed data tensor.
        """
        ...

    def validate_head_dim(self, head_dim: int) -> bool:
        """Check if this method supports the given head dimension."""
        return head_dim in self.info().supported_head_dims

    def compression_ratio(self) -> float:
        """Return the compression ratio vs FP16."""
        return 16.0 / self.info().bits

    def supports_triton(self) -> bool:
        """Whether Triton kernels are available for this method."""
        return False

    def supports_cuda(self) -> bool:
        """Whether compiled CUDA kernels are available."""
        return False

    def supports_metal(self) -> bool:
        """Whether Metal kernels are available (Apple Silicon)."""
        return False
