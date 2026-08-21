from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from multi_turboquant.calibration.gguf_streaming import (
    StreamingQueryStats,
    _config_metadata,
    _gguf_quantization,
    _local_gguf_arguments,
    build_parser,
    calibrate_local_gguf_streaming,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("model.IQ4_XS.gguf", "IQ4_XS"),
        ("model-Q6_K_L.gguf", "Q6_K_L"),
        ("model-f16.gguf", "F16"),
        ("model.gguf", "unknown"),
    ],
)
def test_gguf_quantization_comes_from_filename(filename, expected):
    assert _gguf_quantization(filename) == expected


def test_local_gguf_arguments_keep_windows_path_out_of_hub_filename(tmp_path):
    gguf = tmp_path / "model.IQ4_XS.gguf"

    directory, filename = _local_gguf_arguments(gguf)

    assert directory == str(tmp_path.resolve())
    assert filename == "model.IQ4_XS.gguf"


def test_streaming_query_stats_match_direct_half_style_moments():
    projection = torch.nn.Linear(4, 8, bias=False)
    projection.weight.data.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10)
    attention = SimpleNamespace(q_proj=projection)
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
    collector = StreamingQueryStats(num_heads=2, head_dim=4, chunk_tokens=2)

    collector.hook(0)(attention, (hidden,), {})
    payload = collector.payload(metadata={"source_format": "test"})

    projected = projection(hidden).reshape(2, 3, 2, 4).transpose(1, 2)
    expected_real = projected[..., :2]
    expected_imag = projected[..., 2:]
    first = payload["stats"]["layer00_head00"]
    assert torch.allclose(first["q_mean_real"], expected_real[:, 0].mean(dim=(0, 1)))
    assert torch.allclose(first["q_mean_imag"], expected_imag[:, 0].mean(dim=(0, 1)))
    assert torch.allclose(
        first["q_abs_mean"],
        torch.hypot(expected_real[:, 0], expected_imag[:, 0]).mean(dim=(0, 1)),
    )
    assert len(payload["metadata"]["sampled_heads"]) == 2


def test_streaming_query_stats_accumulate_multiple_forwards():
    projection = torch.nn.Linear(4, 4, bias=False)
    projection.weight.data.copy_(torch.eye(4))
    attention = SimpleNamespace(q_proj=projection)
    collector = StreamingQueryStats(num_heads=1, head_dim=4, chunk_tokens=1)

    collector.hook(3)(attention, (torch.ones(1, 2, 4),), {})
    collector.hook(3)(attention, (torch.full((1, 1, 4), 4.0),), {})
    stats = collector.payload(metadata={})["stats"]["layer03_head00"]

    assert torch.allclose(stats["q_mean_real"], torch.full((2,), 2.0))
    assert torch.allclose(stats["q_mean_imag"], torch.full((2,), 2.0))


def test_config_metadata_prefers_nested_rope_theta():
    metadata = _config_metadata(
        SimpleNamespace(
            model_type="qwen2",
            num_attention_heads=16,
            num_key_value_heads=2,
            num_hidden_layers=36,
            hidden_size=2048,
            head_dim=128,
            max_position_embeddings=131072,
            rope_theta=10_000.0,
            rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
        )
    )

    assert metadata["rope_theta"] == 1_000_000.0
    assert metadata["max_position_embeddings"] == 131_072


def test_calibration_requires_explicit_dequantization_acknowledgement(tmp_path):
    gguf = tmp_path / "model.gguf"
    text = tmp_path / "calibration.txt"
    gguf.write_bytes(b"GGUF")
    text.write_text("representative text", encoding="utf-8")

    with pytest.raises(ValueError, match="dequantizes GGUF"):
        calibrate_local_gguf_streaming(
            gguf=gguf,
            input_path=text,
            output_path=tmp_path / "stats.pt",
        )


def test_cli_accepts_optional_godzilla_output():
    args = build_parser().parse_args(
        [
            "--gguf",
            "model.gguf",
            "--input",
            "calibration.txt",
            "--output",
            "stats.pt",
            "--godzilla-output",
            "model.triattention",
            "--confirm-fp32-dequantization",
        ]
    )

    assert args.godzilla_output == "model.triattention"
