# SPDX-License-Identifier: MIT
"""Capability scanner for llama.cpp-compatible server binaries."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
import subprocess

from .llamacpp_args import GODZILLA_KVARN_CACHE_TYPES, GODZILLA_SPEC_TYPES


KNOWN_CACHE_TYPES = frozenset({
    "f32", "f16", "bf16",
    "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
    "turbo2", "turbo3", "turbo4", "turbo2_tcq", "turbo3_tcq",
    "iso3", "iso4", "planar3", "planar4",
    *GODZILLA_KVARN_CACHE_TYPES.values(),
})

KNOWN_SPEC_TYPES = frozenset({
    *GODZILLA_SPEC_TYPES,
    "draft-simple", "draft-eagle3", "draft-mtp",
    "ngram-simple", "ngram-map-k", "ngram-map-k4v",
    "ngram-mod", "ngram-cache",
})

_FLAG_RE = re.compile(r"(?<![\w-])--[A-Za-z0-9][A-Za-z0-9_-]*")


@dataclass(frozen=True)
class LlamaCppCapabilities:
    """Detected feature surface for a llama.cpp-compatible binary."""

    binary: str = "llama-server"
    scanned: bool = False
    help_returncode: int | None = None
    error: str | None = None
    flags: frozenset[str] = field(default_factory=frozenset)
    cache_types: frozenset[str] = field(default_factory=frozenset)
    speculative_types: frozenset[str] = field(default_factory=frozenset)
    build_info: str | None = None
    gigatoken_identified: bool = False

    def supports_flag(self, flag: str) -> bool:
        normalized = flag if flag.startswith("--") else f"--{flag}"
        return normalized in self.flags

    def supports_cache_type(self, cache_type: str) -> bool:
        return cache_type.lower() in self.cache_types

    def supports_speculative_type(self, spec_type: str) -> bool:
        return spec_type.lower() in self.speculative_types

    @property
    def supports_context_extension(self) -> bool:
        return any(
            self.supports_flag(flag)
            for flag in (
                "--rope-scaling",
                "--rope-scale",
                "--rope-freq-base",
                "--rope-freq-scale",
            )
        )

    @property
    def supports_yarn(self) -> bool:
        return self.supports_flag("--rope-scaling") and self.supports_flag(
            "--yarn-orig-ctx"
        )

    @property
    def supports_triattention(self) -> bool:
        return self.supports_flag("--triattention-stats")

    @property
    def supports_kvarn(self) -> bool:
        return any(value.startswith("kvarn") for value in self.cache_types)

    @property
    def supports_speculative(self) -> bool:
        return self.supports_flag("--spec-type")

    @property
    def supports_dflash(self) -> bool:
        return self.supports_speculative_type("dflash") or self.supports_flag(
            "--spec-dflash-cross-ctx"
        )

    @property
    def supports_props_endpoint(self) -> bool:
        return self.supports_flag("--props")

    @property
    def supports_gigatoken(self) -> bool:
        return self.gigatoken_identified or any(
            "gigatoken" in flag.lower() for flag in self.flags
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "binary": self.binary,
            "scanned": self.scanned,
            "help_returncode": self.help_returncode,
            "error": self.error,
            "build_info": self.build_info,
            "flags": sorted(self.flags),
            "cache_types": sorted(self.cache_types),
            "speculative_types": sorted(self.speculative_types),
            "supports_context_extension": self.supports_context_extension,
            "supports_yarn": self.supports_yarn,
            "supports_triattention": self.supports_triattention,
            "supports_kvarn": self.supports_kvarn,
            "supports_speculative": self.supports_speculative,
            "supports_dflash": self.supports_dflash,
            "supports_props_endpoint": self.supports_props_endpoint,
            "supports_gigatoken": self.supports_gigatoken,
        }


def parse_llamacpp_help(
    help_text: str,
    *,
    binary: str = "llama-server",
    help_returncode: int | None = None,
) -> LlamaCppCapabilities:
    """Parse `llama-server --help` text into a capability summary."""
    flags = frozenset(_FLAG_RE.findall(help_text))
    lowered = help_text.lower()

    cache_types = {
        cache_type
        for cache_type in KNOWN_CACHE_TYPES
        if re.search(rf"(?<![\w-]){re.escape(cache_type)}(?![\w-])", lowered)
    }
    if re.search(r"(?<![\w-])kvarn\s*2\s*(?:\.\.|-|to)\s*kvarn\s*8(?![\w-])", lowered):
        cache_types.update(GODZILLA_KVARN_CACHE_TYPES.values())
    cache_types = frozenset(cache_types)
    speculative_types = frozenset(
        spec_type
        for spec_type in KNOWN_SPEC_TYPES
        if re.search(rf"(?<![\w-]){re.escape(spec_type)}(?![\w-])", lowered)
    )

    build_info = None
    for line in help_text.splitlines():
        stripped = line.strip()
        lowered_line = stripped.lower()
        if lowered_line.startswith("build:") or lowered_line.startswith("version:"):
            build_info = stripped
            break

    return LlamaCppCapabilities(
        binary=binary,
        scanned=True,
        help_returncode=help_returncode,
        flags=flags,
        cache_types=cache_types,
        speculative_types=speculative_types,
        build_info=build_info,
        gigatoken_identified="gigatoken" in lowered,
    )


def scan_llamacpp_binary(
    binary: str = "llama-server",
    *,
    timeout_seconds: float = 10.0,
) -> LlamaCppCapabilities:
    """Run a llama.cpp-compatible binary with `--help` and parse its features."""
    try:
        proc = subprocess.run(
            [binary, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return LlamaCppCapabilities(binary=binary, error=str(exc))
    except subprocess.TimeoutExpired:
        return LlamaCppCapabilities(
            binary=binary,
            error=f"{binary!r} --help timed out after {timeout_seconds:g}s",
        )
    except OSError as exc:
        return LlamaCppCapabilities(binary=binary, error=str(exc))

    if not proc.stdout:
        return LlamaCppCapabilities(
            binary=binary,
            scanned=proc.returncode == 0,
            help_returncode=proc.returncode,
            error=f"{binary!r} --help returned no output",
        )

    capabilities = parse_llamacpp_help(
        proc.stdout,
        binary=binary,
        help_returncode=proc.returncode,
    )
    if proc.returncode != 0:
        return replace(
            capabilities,
            error=f"{binary!r} --help exited with {proc.returncode}",
        )
    return capabilities
