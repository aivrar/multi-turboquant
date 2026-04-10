# SPDX-License-Identifier: MIT
"""Drop-in adapter for the Llama_TQ bridge.

Provides the interface that bridge_services.py expects, mapping
Multi-TurboQuant's unified config to the bridge's service start/stop
lifecycle.

Usage in bridge_services.py:
    from multi_turboquant.integration import BridgeAdapter

    adapter = BridgeAdapter(config)
    cmd = adapter.get_start_command(model_path, engine="llamacpp")
    metadata_ready = adapter.ensure_calibration(model_path)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..config import CacheConfig, CacheMethod, MethodFamily, METHOD_FAMILIES


class BridgeAdapter:
    """Adapter between Multi-TurboQuant and the Llama_TQ bridge.

    Handles the bridge's expectations around:
    - Service start commands (llama.cpp or vLLM)
    - Calibration metadata generation
    - Status reporting for the UI
    """

    def __init__(self, config: CacheConfig):
        self.config = config

    def get_start_command(
        self,
        model_path: str,
        *,
        engine: str = "llamacpp",
        host: str = "0.0.0.0",
        port: int = 8080,
        context_size: int = 4096,
        gpu_layers: int = 99,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """Generate the service start command.

        Args:
            model_path: Path to model file (GGUF for llama.cpp, dir for vLLM).
            engine: "llamacpp" or "vllm".
            host: Listen address.
            port: Listen port.
            context_size: Context window.
            gpu_layers: GPU layer offload count.
            extra_args: Additional arguments.

        Returns:
            Command as list of strings.
        """
        if engine == "llamacpp":
            from .llamacpp_args import get_llamacpp_command
            return get_llamacpp_command(
                self.config,
                model_path=model_path,
                host=host,
                port=port,
                context_size=context_size,
                gpu_layers=gpu_layers,
                extra_args=extra_args,
            )
        elif engine == "vllm":
            return self._vllm_command(
                model_path, host=host, port=port,
                context_size=context_size, extra_args=extra_args,
            )
        else:
            raise ValueError(f"Unknown engine: {engine}")

    def _vllm_command(
        self,
        model_path: str,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        context_size: int = 4096,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """Generate vLLM start command."""
        from .vllm_patch import get_vllm_kv_cache_dtype

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--host", host,
            "--port", str(port),
            "--max-model-len", str(context_size),
            "--kv-cache-dtype", get_vllm_kv_cache_dtype(self.config),
        ]

        if self.config.triattention_enabled:
            cmd.extend([
                "--enable-prefix-caching", "false",
            ])
            # TriAttention environment variables
            os.environ["TRIATTENTION_BUDGET"] = str(self.config.triattention_budget)
            os.environ["TRIATTENTION_WINDOW"] = str(self.config.triattention_window)

        if extra_args:
            cmd.extend(extra_args)

        return cmd

    def ensure_calibration(self, model_path: str) -> bool:
        """Ensure all required calibration files exist, generating if needed.

        Args:
            model_path: Path to model directory.

        Returns:
            True if all calibration is ready.
        """
        if not self.config.needs_calibration:
            return True

        from ..calibration.auto_calibrate import auto_calibrate, check_calibration

        # Check existing files
        status = check_calibration(self.config, model_path)
        if all(status.values()):
            return True

        # Generate missing calibration
        try:
            auto_calibrate(self.config, model_path)
            return True
        except Exception:
            return False

    def get_status(self) -> dict[str, Any]:
        """Return status info for the bridge UI."""
        return {
            "k_method": self.config.k_method.value,
            "v_method": self.config.v_method.value,
            "k_compression": f"{self.config.k_compression:.1f}x",
            "v_compression": f"{self.config.v_compression:.1f}x",
            "triattention": self.config.triattention_enabled,
            "triattention_budget": self.config.triattention_budget if self.config.triattention_enabled else None,
            "needs_calibration": self.config.needs_calibration,
            "is_symmetric": self.config.is_symmetric,
            "is_k_only": self.config.is_k_only,
            "preset": self._detect_preset(),
        }

    def _detect_preset(self) -> str | None:
        """Try to match current config to a named preset."""
        from ..presets import PRESETS
        for name, preset in PRESETS.items():
            if (preset.k_method == self.config.k_method
                    and preset.v_method == self.config.v_method
                    and preset.triattention_enabled == self.config.triattention_enabled):
                return name
        return None

    def get_ui_options(self) -> list[dict[str, Any]]:
        """Return cache type options for the bridge UI dropdown."""
        from ..registry import list_methods
        options = []
        for info in list_methods():
            options.append({
                "value": info.method.value,
                "label": f"{info.method.value} ({info.bits:.1f}-bit, "
                         f"{16.0/info.bits:.1f}x)",
                "family": info.family.value,
                "requires_calibration": info.requires_calibration,
                "description": info.description,
            })
        return options
