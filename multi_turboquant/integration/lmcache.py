# SPDX-License-Identifier: MIT
"""Conservative command planning for LMCache's documented vLLM connectors."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum


class LMCacheMode(str, Enum):
    IN_PROCESS = "in_process"
    MULTIPROCESS = "multiprocess"


def _version_at_least_020(value: str) -> bool | None:
    match = re.match(r"^\s*v?(\d+)\.(\d+)(?:\.(\d+))?([^+\s]*)", value)
    if not match:
        return None
    release = (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )
    suffix = match.group(4).lower()
    if release < (0, 20, 0):
        return False
    if release == (0, 20, 0) and suffix.startswith(("a", "b", "rc", "dev")):
        return False
    return True


@dataclass(frozen=True)
class LMCacheIntegrationConfig:
    """LMCache options that map directly to its public vLLM quickstart."""

    mode: LMCacheMode | str = LMCacheMode.MULTIPROCESS
    config_file: str | None = None
    mp_host: str = "localhost"
    mp_port: int = 5555
    server_l1_size_gb: float = 5.0
    server_chunk_size: int = 256
    server_eviction_policy: str = "LRU"
    use_lmcache_shipped_connector: bool = False
    vllm_version: str | None = None

    def __post_init__(self) -> None:
        mode = self.mode
        if not isinstance(mode, LMCacheMode):
            try:
                mode = LMCacheMode(str(mode).strip().lower())
            except ValueError as exc:
                allowed = ", ".join(item.value for item in LMCacheMode)
                raise ValueError(f"Unknown LMCache mode {self.mode!r}; use {allowed}") from exc
            object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "mp_host", self.mp_host.strip())
        object.__setattr__(
            self,
            "server_eviction_policy",
            self.server_eviction_policy.strip().upper(),
        )
        if self.config_file is not None:
            object.__setattr__(self, "config_file", self.config_file.strip())

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.mp_host:
            errors.append("LMCache MP host cannot be blank")
        elif "://" in self.mp_host and not self.mp_host.startswith("tcp://"):
            errors.append("LMCache MP host only supports the tcp:// transport")
        else:
            server_host = self.server_host
            if not server_host:
                errors.append("LMCache MP host must include a hostname")
            elif any(character.isspace() for character in server_host) or "/" in server_host:
                errors.append("LMCache MP host must be a hostname without whitespace or a path")
            elif ":" in server_host and not (
                server_host.startswith("[") and server_host.endswith("]")
            ):
                errors.append("LMCache MP host must not include a port; use mp_port")
        if not 1 <= self.mp_port <= 65535:
            errors.append("LMCache MP port must be between 1 and 65535")
        if not math.isfinite(self.server_l1_size_gb) or self.server_l1_size_gb <= 0:
            errors.append("LMCache server L1 size must be positive")
        if self.server_chunk_size <= 0:
            errors.append("LMCache server chunk size must be positive")
        if self.server_eviction_policy != "LRU":
            errors.append("Only LMCache's documented LRU eviction policy is supported")
        if self.config_file is not None:
            if not self.config_file:
                errors.append("LMCache config file cannot be blank")
            elif "\x00" in self.config_file:
                errors.append("LMCache config file cannot contain a NUL byte")

        if self.use_lmcache_shipped_connector:
            if self.mode != LMCacheMode.MULTIPROCESS:
                errors.append("The LMCache-shipped connector module is only valid in MP mode")
            if self.vllm_version is None:
                errors.append("vLLM version is required when selecting the LMCache-shipped connector")
            else:
                supported = _version_at_least_020(self.vllm_version)
                if supported is None:
                    errors.append(f"Could not parse vLLM version {self.vllm_version!r}")
                elif not supported:
                    errors.append("The LMCache-shipped connector requires vLLM 0.20.0 or newer")
        return tuple(errors)

    @property
    def connector_host(self) -> str:
        return self.mp_host if self.mp_host.startswith("tcp://") else f"tcp://{self.mp_host}"

    @property
    def server_host(self) -> str:
        return self.mp_host.removeprefix("tcp://")


@dataclass(frozen=True)
class LMCacheLaunchPlan:
    """Commands and environment to apply explicitly in the serving process."""

    environment: dict[str, str]
    vllm_args: tuple[str, ...]
    server_command: tuple[str, ...] | None
    notes: tuple[str, ...]

    def extend_vllm_command(self, command: list[str] | tuple[str, ...]) -> list[str]:
        if any(
            arg == "--kv-transfer-config" or arg.startswith("--kv-transfer-config=")
            for arg in command
        ):
            raise ValueError("vLLM command already contains --kv-transfer-config")
        return [*command, *self.vllm_args]

    def to_dict(self) -> dict:
        return {
            "environment": dict(self.environment),
            "vllm_args": list(self.vllm_args),
            "server_command": list(self.server_command) if self.server_command else None,
            "notes": list(self.notes),
        }


def build_lmcache_launch_plan(config: LMCacheIntegrationConfig) -> LMCacheLaunchPlan:
    """Build an LMCache/vLLM plan without importing or launching either project."""
    errors = config.validate()
    if errors:
        raise ValueError("Invalid LMCache configuration: " + "; ".join(errors))

    transfer: dict[str, object]
    server_command: tuple[str, ...] | None
    notes: list[str] = []
    if config.mode == LMCacheMode.MULTIPROCESS:
        transfer = {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": config.connector_host,
                "lmcache.mp.port": config.mp_port,
            },
        }
        if config.use_lmcache_shipped_connector:
            transfer["kv_connector_module_path"] = (
                "lmcache.integration.vllm.lmcache_mp_connector"
            )
        server_command = (
            "lmcache",
            "server",
            "--host",
            config.server_host,
            "--port",
            str(config.mp_port),
            "--l1-size-gb",
            f"{config.server_l1_size_gb:g}",
            "--eviction-policy",
            config.server_eviction_policy,
            "--chunk-size",
            str(config.server_chunk_size),
        )
        notes.append("Start the LMCache server before the vLLM process.")
    else:
        transfer = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
        server_command = None
        notes.append("LMCache runs inside the vLLM process in in-process mode.")

    environment = {}
    if config.config_file:
        environment["LMCACHE_CONFIG_FILE"] = config.config_file
    payload = json.dumps(transfer, separators=(",", ":"), sort_keys=True)
    return LMCacheLaunchPlan(
        environment=environment,
        vllm_args=("--kv-transfer-config", payload),
        server_command=server_command,
        notes=tuple(notes),
    )
