# SPDX-License-Identifier: MIT
"""Compiled CUDA kernels for llama.cpp integration.

These are the C/CUDA source files that need to be compiled into
llama.cpp for native KV cache compression support. They are not
used directly from Python — see the integration/llamacpp_args.py
module for generating the correct build and runtime flags.

Source files (from upstream forks):
  TurboQuant:   ggml-cuda/fattn-vec-*.cu (TheTom/llama-cpp-turboquant)
  PlanarQuant:  ggml-cuda/cpy-planar-iso.cu (johndpope/llama-cpp-turboquant)
  IsoQuant:     ggml-cuda/cpy-planar-iso.cu (johndpope/llama-cpp-turboquant)
"""
