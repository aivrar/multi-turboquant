# SPDX-License-Identifier: MIT
"""Method registry — discover and instantiate compression methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import CacheMethod, MethodFamily, METHOD_FAMILIES

if TYPE_CHECKING:
    from .methods.base import CompressionMethod, MethodInfo


_REGISTRY: dict[CacheMethod, type["CompressionMethod"]] = {}


def register_method(method: CacheMethod):
    """Decorator to register a CompressionMethod class for a cache type."""
    def decorator(cls: type["CompressionMethod"]):
        if method in _REGISTRY:
            raise ValueError(
                f"Method {method.value} already registered to "
                f"{_REGISTRY[method].__name__}"
            )
        _REGISTRY[method] = cls
        return cls
    return decorator


def get_method(method: CacheMethod, **kwargs) -> "CompressionMethod":
    """Instantiate a registered compression method.

    Args:
        method: The cache method to instantiate.
        **kwargs: Method-specific initialization arguments
            (e.g., metadata_path for TurboQuant).

    Raises:
        KeyError: If the method is not registered.
    """
    if method not in _REGISTRY:
        raise KeyError(
            f"Method {method.value} is not registered. "
            f"Available: {', '.join(m.value for m in _REGISTRY)}"
        )
    return _REGISTRY[method](**kwargs)


def list_methods() -> list["MethodInfo"]:
    """Return MethodInfo for all registered methods."""
    infos = []
    for method in CacheMethod:
        if method in _REGISTRY:
            instance = _REGISTRY[method]()
            infos.append(instance.info())
    return infos


def list_methods_by_family(family: MethodFamily) -> list[CacheMethod]:
    """Return all registered methods in a given family."""
    return [
        method for method, cls in _REGISTRY.items()
        if METHOD_FAMILIES.get(method) == family
    ]


def is_registered(method: CacheMethod) -> bool:
    """Check if a method is registered."""
    return method in _REGISTRY


def registered_methods() -> list[CacheMethod]:
    """Return all registered cache methods."""
    return list(_REGISTRY.keys())
