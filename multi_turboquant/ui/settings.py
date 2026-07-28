# SPDX-License-Identifier: MIT
"""Versioned, persistent settings for the localhost UI."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Mapping


UI_SETTINGS_SCHEMA = 1
MAX_FORM_VALUES = 256
MAX_STRING_LENGTH = 16_384

DEFAULT_UI_SETTINGS = {
    "schema": UI_SETTINGS_SCHEMA,
    "model_root": "",
    "addon_roots": [],
    "flashattention_source": "",
    "environment_root": ".mtq/environments",
    "form_values": {},
}


def default_settings_path() -> Path:
    """Return the per-user settings path without creating it."""
    return Path.home() / ".multi-turboquant" / "ui-settings.json"


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if len(value) > MAX_STRING_LENGTH:
        raise ValueError(f"{field} is too long")
    return value.strip()


def _form_value(value: object, field: str) -> str | int | float | bool | None:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str) and len(value) <= MAX_STRING_LENGTH:
        return value
    raise ValueError(f"{field} must be a JSON scalar")


def validate_ui_settings(value: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize a complete UI settings payload."""
    if not isinstance(value, Mapping):
        raise ValueError("Settings must be a JSON object")
    schema = value.get("schema", UI_SETTINGS_SCHEMA)
    if schema != UI_SETTINGS_SCHEMA:
        raise ValueError(
            f"Unsupported settings schema {schema!r}; expected {UI_SETTINGS_SCHEMA}"
        )

    roots = value.get("addon_roots", [])
    if not isinstance(roots, list) or len(roots) > 32:
        raise ValueError("addon_roots must be a list with at most 32 entries")
    normalized_roots = [_text(item, f"addon_roots[{index}]") for index, item in enumerate(roots)]
    normalized_roots = [item for item in normalized_roots if item]

    form_values = value.get("form_values", {})
    if not isinstance(form_values, Mapping) or len(form_values) > MAX_FORM_VALUES:
        raise ValueError(f"form_values must contain at most {MAX_FORM_VALUES} entries")
    normalized_form: dict[str, object] = {}
    for raw_key, raw_value in form_values.items():
        key = _text(raw_key, "form_values key")
        if not key:
            raise ValueError("form_values keys must not be empty")
        normalized_form[key] = _form_value(raw_value, f"form_values.{key}")

    return {
        "schema": UI_SETTINGS_SCHEMA,
        "model_root": _text(value.get("model_root", ""), "model_root"),
        "addon_roots": normalized_roots,
        "flashattention_source": _text(
            value.get("flashattention_source", ""), "flashattention_source"
        ),
        "environment_root": _text(
            value.get("environment_root", ".mtq/environments"), "environment_root"
        )
        or ".mtq/environments",
        "form_values": normalized_form,
    }


class UISettingsStore:
    """Thread-safe JSON settings store with atomic replacement."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser() if path is not None else default_settings_path()
        self._lock = threading.RLock()

    def load(self) -> dict[str, object]:
        with self._lock:
            if not self.path.is_file():
                return copy.deepcopy(DEFAULT_UI_SETTINGS)
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Could not read UI settings: {exc}") from exc
            try:
                return validate_ui_settings(raw)
            except ValueError as exc:
                raise RuntimeError(f"Invalid UI settings: {exc}") from exc

    def save(self, value: Mapping[str, object]) -> dict[str, object]:
        normalized = validate_ui_settings(value)
        payload = json.dumps(normalized, indent=2, sort_keys=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
        return copy.deepcopy(normalized)

    def reset(self) -> dict[str, object]:
        return self.save(DEFAULT_UI_SETTINGS)

    def public_state(self) -> dict[str, object]:
        return {
            "path": str(self.path.resolve()),
            "settings": self.load(),
        }
