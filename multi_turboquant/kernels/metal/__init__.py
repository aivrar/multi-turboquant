# SPDX-License-Identifier: MIT
"""Apple Silicon Metal kernels for PlanarQuant and IsoQuant.

Metal kernels are available in johndpope/llama-cpp-turboquant for
PlanarQuant and IsoQuant methods. These run on Apple M-series chips
and provide native KV cache compression on macOS.

Not used directly from Python — these are compiled into llama.cpp
when building with -DGGML_METAL=ON.
"""
