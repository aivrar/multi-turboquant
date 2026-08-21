from __future__ import annotations

import sys
from types import SimpleNamespace

from multi_turboquant.calibration import triattention_runner


def test_runner_promotes_nested_rope_theta(monkeypatch):
    config = SimpleNamespace(
        rope_theta=10_000.0,
        rope_parameters={"rope_theta": 1_000_000.0, "rope_type": "default"},
    )
    auto_config = SimpleNamespace(
        from_pretrained=staticmethod(lambda *_args, **_kwargs: config)
    )
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoConfig=auto_config)
    )
    messages: list[str] = []

    triattention_runner.patch_auto_config(
        stderr=SimpleNamespace(
            write=lambda value: messages.append(value)
        )
    )
    loaded = auto_config.from_pretrained("org/model")

    assert loaded.rope_theta == 1_000_000.0
    assert any("rope_parameters.rope_theta=1e+06" in message for message in messages)


def test_runner_selects_requested_cuda_index(monkeypatch):
    import torch

    selected: list[int] = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: selected.append(index))

    triattention_runner._select_cuda_device("cuda:1")

    assert selected == [1]


def test_runner_forces_auto_device_map_to_selected_cuda(monkeypatch):
    observed: list[dict[str, object]] = []
    auto_model = SimpleNamespace(
        from_pretrained=staticmethod(
            lambda *_args, **kwargs: observed.append(kwargs) or object()
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=auto_model),
    )

    triattention_runner.patch_auto_model_device("cuda:1")
    auto_model.from_pretrained("org/model", device_map="auto")

    assert observed[0]["device_map"] == "cuda:1"
