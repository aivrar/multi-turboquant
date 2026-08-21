from __future__ import annotations

from types import SimpleNamespace

from multi_turboquant.calibration import triattention_runner


def test_runner_promotes_nested_rope_theta(monkeypatch):
    import transformers

    config = SimpleNamespace(
        rope_theta=10_000.0,
        rope_parameters={"rope_theta": 1_000_000.0, "rope_type": "default"},
    )
    original = transformers.AutoConfig.from_pretrained
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: config),
    )
    messages: list[str] = []

    try:
        triattention_runner.patch_auto_config(stderr=SimpleNamespace(
            write=lambda value: messages.append(value)
        ))
        loaded = transformers.AutoConfig.from_pretrained("org/model")
    finally:
        transformers.AutoConfig.from_pretrained = original

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
    import transformers

    observed: list[dict[str, object]] = []
    original = transformers.AutoModelForCausalLM.from_pretrained
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(lambda *_args, **kwargs: observed.append(kwargs) or object()),
    )

    try:
        triattention_runner.patch_auto_model_device("cuda:1")
        transformers.AutoModelForCausalLM.from_pretrained(
            "org/model", device_map="auto"
        )
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = original

    assert observed[0]["device_map"] == "cuda:1"
