# SPDX-License-Identifier: MIT
"""Repository-wide compatibility matrix for optional optimizations.

The matrix describes whether two catalog entries may share one execution
profile.  It is deliberately independent of package installation: installing
two projects never upgrades an unreviewed pairing to a supported composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Iterable

from .core import (
    IntegrationMode,
    OptimizationDescriptor,
    OptimizationMaturity,
)


class CompositionDisposition(str, Enum):
    """How two optimization capabilities may be exposed together."""

    DIRECT = "direct"
    CONDITIONAL = "conditional"
    ROUTED = "routed"
    PROHIBITED = "prohibited"


@dataclass(frozen=True)
class CompositionRule:
    """A symmetric, reviewable decision for one unordered pair."""

    left: str
    right: str
    disposition: CompositionDisposition
    reason: str
    requirements: tuple[str, ...] = ()
    validation_gates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        left = self.left.strip().lower()
        right = self.right.strip().lower()
        if not left or not right or left == right:
            raise ValueError("Composition rules require two distinct optimization IDs")
        if not self.reason.strip():
            raise ValueError("Composition rules require a non-empty reason")
        object.__setattr__(self, "left", min(left, right))
        object.__setattr__(self, "right", max(left, right))

    @property
    def pair(self) -> tuple[str, str]:
        return self.left, self.right

    def to_dict(self) -> dict:
        return {
            "left": self.left,
            "right": self.right,
            "disposition": self.disposition.value,
            "reason": self.reason,
            "requirements": list(self.requirements),
            "validation_gates": list(self.validation_gates),
        }


class CompositionMatrix:
    """Complete set of symmetric rules for a fixed catalog."""

    def __init__(self, optimization_ids: Iterable[str], rules: Iterable[CompositionRule]):
        ids = tuple(sorted(set(optimization_ids)))
        if any(not item or item.lower() != item for item in ids):
            raise ValueError("Composition matrix IDs must be non-empty lowercase strings")
        mapping: dict[tuple[str, str], CompositionRule] = {}
        for rule in rules:
            if rule.left not in ids or rule.right not in ids:
                raise ValueError(f"Rule {rule.pair!r} references an unknown optimization")
            if rule.pair in mapping:
                raise ValueError(f"Duplicate composition rule for {rule.pair!r}")
            mapping[rule.pair] = rule

        expected = set(combinations(ids, 2))
        missing = expected - set(mapping)
        extra = set(mapping) - expected
        if missing or extra:
            raise ValueError(
                f"Composition matrix is incomplete (missing={len(missing)}, extra={len(extra)})"
            )
        self._ids = ids
        self._rules = mapping

    @property
    def optimization_ids(self) -> tuple[str, ...]:
        return self._ids

    @property
    def rules(self) -> tuple[CompositionRule, ...]:
        return tuple(self._rules[pair] for pair in sorted(self._rules))

    def rule(self, left: str, right: str) -> CompositionRule:
        normalized = tuple(sorted((left.strip().lower(), right.strip().lower())))
        if len(set(normalized)) != 2:
            raise ValueError("Composition lookup requires two distinct optimization IDs")
        try:
            return self._rules[normalized]
        except KeyError as exc:
            raise KeyError(f"Unknown composition pair {normalized!r}") from exc

    def to_dict(self) -> dict:
        return {
            "optimization_ids": list(self.optimization_ids),
            "pair_count": len(self._rules),
            "rules": [rule.to_dict() for rule in self.rules],
        }


def _rule(
    left: str,
    right: str,
    disposition: CompositionDisposition,
    reason: str,
    *,
    requirements: tuple[str, ...] = (),
    validation_gates: tuple[str, ...] = (),
) -> CompositionRule:
    return CompositionRule(
        left,
        right,
        disposition,
        reason,
        requirements,
        validation_gates,
    )


# Rules here record source-specific facts that cannot be inferred from the
# generic domain and engine metadata.  All other pairs use the fail-closed
# classifier below.
_CURATED_RULES = {
    rule.pair: rule
    for rule in (
        _rule(
            "fastdms",
            "flashattention",
            CompositionDisposition.DIRECT,
            "FastDMS declares FlashAttention as a runtime dependency.",
            requirements=("A DMS-trained checkpoint supported by FastDMS",),
            validation_gates=("FastDMS compact-cache correctness and soak tests",),
        ),
        _rule(
            "flashattention",
            "jetspec",
            CompositionDisposition.DIRECT,
            "JetSpec exposes FlashAttention 2 in its reviewed reference path.",
            requirements=("A draft head matching the exact target model",),
            validation_gates=("Greedy target-output parity and acceptance-length measurement",),
        ),
        _rule(
            "flashattention",
            "jetlong",
            CompositionDisposition.DIRECT,
            "Jet-Long documents a non-fused FlashAttention path.",
            requirements=("The exact supported Qwen3 model and Jet-Long dependency profile",),
            validation_gates=("Base-window regression and long-context evaluation",),
        ),
        _rule(
            "flashattention",
            "triattention",
            CompositionDisposition.DIRECT,
            "Official TriAttention documents FlashAttention as its accelerated attention dependency.",
            requirements=("Model-matched TriAttention calibration statistics",),
            validation_gates=("Full-attention quality and KV-budget regression",),
        ),
        _rule(
            "lmcache",
            "minference",
            CompositionDisposition.CONDITIONAL,
            "Both can target vLLM, but sparse prefill and reusable KV storage need one exact-version profile.",
            requirements=(
                "One mutually supported vLLM/Torch/Triton version",
                "A standard LMCache-supported KV representation",
            ),
            validation_gates=(
                "Cache-hit correctness with MInference enabled",
                "TTFT comparison for cache hits and misses",
            ),
        ),
        _rule(
            "lmcache",
            "proxima",
            CompositionDisposition.CONDITIONAL,
            "STAR-KV uses a custom low-rank paged layout that has no reviewed LMCache serializer.",
            requirements=(
                "A STAR-KV-aware LMCache serializer and connector",
                "An exact vLLM version supported by both integrations",
            ),
            validation_gates=("Round-trip cache identity and mixed-request concurrency soak",),
        ),
        _rule(
            "lmcache",
            "triattention",
            CompositionDisposition.PROHIBITED,
            "TriAttention rewrites KV state and explicitly requires prefix/radix caching to be disabled.",
        ),
        _rule(
            "minference",
            "proxima",
            CompositionDisposition.CONDITIONAL,
            "Sparse prefill must write directly into Proxima's calibrated low-rank cache layout.",
            requirements=("A dedicated MInference-to-STAR-KV prefill/store kernel",),
            validation_gates=("Prefill logits, cache layout, and long-context quality parity",),
        ),
        _rule(
            "minference",
            "triattention",
            CompositionDisposition.CONDITIONAL,
            "Prefill sparsity and decode-time KV selection are orthogonal in principle but patch the same serving runtime.",
            requirements=("An exact jointly supported runtime with bounded chunked prefill",),
            validation_gates=("Long-context prefill quality and reasoning retention",),
        ),
        _rule(
            "proxima",
            "triattention",
            CompositionDisposition.PROHIBITED,
            "TriAttention compaction assumes ordinary KV blocks while Proxima replaces them with per-layer low-rank pages.",
        ),
    )
}


def _classify_pair(left: OptimizationDescriptor, right: OptimizationDescriptor) -> CompositionRule:
    pair = tuple(sorted((left.id, right.id)))
    curated = _CURATED_RULES.get(pair)
    if curated is not None:
        return curated

    if (
        left.maturity == OptimizationMaturity.BLOCKED
        or right.maturity == OptimizationMaturity.BLOCKED
        or left.integration_mode == IntegrationMode.RESEARCH_ONLY
        or right.integration_mode == IntegrationMode.RESEARCH_ONLY
    ):
        return _rule(
            *pair,
            CompositionDisposition.PROHIBITED,
            "At least one entry has no eligible maintained runtime integration.",
        )

    explicit_conflict = right.id in left.conflicts or left.id in right.conflicts
    if explicit_conflict:
        return _rule(
            *pair,
            CompositionDisposition.PROHIBITED,
            "The catalog explicitly records this pair as conflicting.",
        )

    transitive_conflicts = tuple(sorted(
        (set(left.requires) & set(right.conflicts))
        | (set(right.requires) & set(left.conflicts))
    ))
    if transitive_conflicts:
        return _rule(
            *pair,
            CompositionDisposition.PROHIBITED,
            "A required dependency conflicts with the other entry: "
            f"{', '.join(transitive_conflicts)}.",
        )

    dependency_pair = right.id in left.requires or left.id in right.requires
    mutually_allowed = (
        right.id in left.allows_composition_with
        and left.id in right.allows_composition_with
    )
    if dependency_pair or mutually_allowed:
        return _rule(
            *pair,
            CompositionDisposition.DIRECT,
            "The reviewed catalog records an explicit dependency or mutual composition profile.",
            validation_gates=tuple(dict.fromkeys(
                left.validation_gates + right.validation_gates
            )),
        )

    shared_domains = tuple(sorted(
        set(left.composition_domains) & set(right.composition_domains)
    ))
    if shared_domains:
        return _rule(
            *pair,
            CompositionDisposition.PROHIBITED,
            f"Both entries modify {', '.join(shared_domains)} without a reviewed joint implementation.",
        )

    shared_engines = tuple(sorted(set(left.supported_engines) & set(right.supported_engines)))
    if not shared_engines:
        return _rule(
            *pair,
            CompositionDisposition.ROUTED,
            "The entries have no common inference runtime and must run in separate processes.",
            requirements=("A request router with explicit backend selection and health checks",),
            validation_gates=("Cross-backend output and service-level regression",),
        )

    return _rule(
        *pair,
        CompositionDisposition.CONDITIONAL,
        f"The entries share {', '.join(shared_engines)} but have no reviewed combined profile.",
        requirements=("An exact-version combined runtime profile",),
        validation_gates=("Joint correctness, quality, memory, and throughput regression",),
    )


def build_composition_matrix(
    descriptors: Iterable[OptimizationDescriptor],
) -> CompositionMatrix:
    """Build a complete deterministic matrix for the supplied descriptors."""
    descriptor_list = tuple(descriptors)
    by_id = {descriptor.id: descriptor for descriptor in descriptor_list}
    if len(by_id) != len(descriptor_list):
        raise ValueError("Composition descriptors must have unique IDs")
    if len(by_id) < 2:
        raise ValueError("At least two unique descriptors are required")
    rules = (
        _classify_pair(by_id[left], by_id[right])
        for left, right in combinations(sorted(by_id), 2)
    )
    return CompositionMatrix(by_id, rules)
