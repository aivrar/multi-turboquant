# SPDX-License-Identifier: MIT
from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="multi-turboquant",
    version="0.1.0",
    description=(
        "Unified KV cache compression for LLM inference — "
        "TurboQuant, TCQ, IsoQuant, PlanarQuant, TriAttention"
    ),
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="aivrar",
    license="MIT",
    url="https://github.com/aivrar/multi-turboquant",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
    ],
    extras_require={
        "calibration": [
            "safetensors>=0.4.0",
            "transformers>=4.36.0,<5",
        ],
        "triton": [
            "triton>=2.1.0",
        ],
        "vllm": [
            "vllm>=0.4.0",
        ],
        "benchmark": [
            "datasets",
            "requests",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
        "all": [
            "safetensors>=0.4.0",
            "transformers>=4.36.0,<5",
            "triton>=2.1.0",
            "datasets",
            "requests",
            "pytest>=7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mtq-calibrate=multi_turboquant.calibration.generate_metadata:main",
            "mtq-triattention-stats=multi_turboquant.calibration.generate_stats:main",
            "mtq-godzilla-triattention=multi_turboquant.calibration.godzilla_triattention:main",
            "mtq-godzilla-gigatoken=multi_turboquant.integration.godzilla_gigatoken_cli:main",
            "mtq-godzilla-compose=multi_turboquant.integration.godzilla_composition_cli:main",
            "mtq-weight-share=multi_turboquant.integration.weight_share_cli:main",
            "mtq-benchmark=multi_turboquant.benchmark.run_benchmark:main",
            "mtq-optimizations=multi_turboquant.optimizations.cli:main",
            "mtq-env=multi_turboquant.optimizations.env_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
