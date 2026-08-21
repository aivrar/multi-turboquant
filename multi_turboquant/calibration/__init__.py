# SPDX-License-Identifier: MIT
"""Calibration tooling for methods that require per-model metadata."""

from .generate_metadata import generate_turboquant_metadata
from .generate_stats import generate_triattention_stats
from .text_corpus import CALIBRATION_CORPUS_SCHEMA_VERSION, generate_calibration_text
from .auto_calibrate import auto_calibrate
from .godzilla_triattention import (
    LONG_CALIBRATION_THRESHOLD,
    MAX_CALIBRATION_TOKENS,
    calibrate_official_triattention_for_godzilla,
    calibrate_domvox_triattention_for_godzilla,
    convert_domvox_triattention_stats,
    convert_official_triattention_stats,
    inspect_godzilla_triattention_file,
    inspect_calibration_python,
    inspect_domvox_triattention_calibrator,
    inspect_domvox_triattention_checkout,
    inspect_domvox_triattention_file,
    inspect_official_triattention_calibrator,
    inspect_official_triattention_checkout,
    load_huggingface_model_metadata,
    normalize_calibration_device,
    select_compatible_calibration_python,
    estimate_official_calibration_bytes,
    validate_model_calibration_length,
)
from .gguf_streaming import (
    StreamingQueryStats,
    calibrate_local_gguf_streaming,
    load_local_gguf_metadata,
)

__all__ = [
    "generate_turboquant_metadata",
    "generate_triattention_stats",
    "generate_calibration_text",
    "CALIBRATION_CORPUS_SCHEMA_VERSION",
    "auto_calibrate",
    "calibrate_official_triattention_for_godzilla",
    "LONG_CALIBRATION_THRESHOLD",
    "MAX_CALIBRATION_TOKENS",
    "calibrate_domvox_triattention_for_godzilla",
    "convert_domvox_triattention_stats",
    "convert_official_triattention_stats",
    "inspect_godzilla_triattention_file",
    "inspect_calibration_python",
    "inspect_domvox_triattention_calibrator",
    "inspect_domvox_triattention_checkout",
    "inspect_domvox_triattention_file",
    "inspect_official_triattention_calibrator",
    "inspect_official_triattention_checkout",
    "load_huggingface_model_metadata",
    "normalize_calibration_device",
    "select_compatible_calibration_python",
    "estimate_official_calibration_bytes",
    "validate_model_calibration_length",
    "StreamingQueryStats",
    "calibrate_local_gguf_streaming",
    "load_local_gguf_metadata",
]
