# SPDX-License-Identifier: MIT
"""Path helpers for executable entry points.

Virtual-environment Python executables are commonly symlinks on POSIX.  Code
that launches them must preserve the lexical venv path: resolving the symlink
first bypasses ``pyvenv.cfg`` discovery and runs the base interpreter instead.
"""

from __future__ import annotations

import os
from pathlib import Path


def lexical_absolute_path(value: str | Path) -> Path:
    """Return an absolute path without dereferencing its final symlink."""
    expanded = Path(value).expanduser()
    return Path(os.path.abspath(os.fspath(expanded)))


def same_lexical_path(left: str | Path, right: str | Path) -> bool:
    """Compare executable entry paths without collapsing venv symlinks."""
    left_key = os.path.normcase(os.fspath(lexical_absolute_path(left)))
    right_key = os.path.normcase(os.fspath(lexical_absolute_path(right)))
    return left_key == right_key
