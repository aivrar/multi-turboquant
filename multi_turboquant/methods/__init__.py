# SPDX-License-Identifier: MIT
"""KV cache compression method implementations.

Each method implements the CompressionMethod protocol from base.py.
Methods are auto-registered via the @register_method decorator.
"""

from .base import CompressionMethod, CompressedKV, MethodInfo
from .turboquant import TurboQuantMethod
from .tcq import TCQMethod
from .isoquant import IsoQuantMethod
from .planarquant import PlanarQuantMethod
from .triattention import TriAttentionMethod

__all__ = [
    "CompressionMethod",
    "CompressedKV",
    "MethodInfo",
    "TurboQuantMethod",
    "TCQMethod",
    "IsoQuantMethod",
    "PlanarQuantMethod",
    "TriAttentionMethod",
]
