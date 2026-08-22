from types import SimpleNamespace

import pytest

from multi_turboquant.huggingface import (
    normalize_huggingface_endpoint,
    normalize_huggingface_reference,
    resolve_huggingface_model_reference,
)


@pytest.mark.parametrize(
    "value",
    [
        "mradermacher/Mythos-nano-heretic-i1-GGUF",
        "https://huggingface.co/mradermacher/Mythos-nano-heretic-i1-GGUF",
        "https://huggingface.co/mradermacher/Mythos-nano-heretic-i1-GGUF/tree/main",
        "https://huggingface.co/mradermacher/Mythos-nano-heretic-i1-GGUF?show_file_info=x.gguf",
    ],
)
def test_normalize_model_reference(value):
    assert normalize_huggingface_reference(value) == "mradermacher/Mythos-nano-heretic-i1-GGUF"


def test_normalize_model_reference_rejects_non_models():
    with pytest.raises(ValueError, match="dataset"):
        normalize_huggingface_reference("https://huggingface.co/datasets/owner/name")


def test_endpoint_requires_https_except_localhost():
    assert normalize_huggingface_endpoint("http://localhost:8080/") == "http://localhost:8080"
    with pytest.raises(ValueError, match="HTTPS"):
        normalize_huggingface_endpoint("http://mirror.example")
    with pytest.raises(ValueError, match="credentials"):
        normalize_huggingface_endpoint("https://token@example.test")


def test_resolver_selects_declared_base_model_without_exposing_token():
    calls = {}

    class FakeApi:
        def __init__(self, *, endpoint):
            calls["endpoint"] = endpoint

        def model_info(self, repo_id, *, token):
            calls.update(repo_id=repo_id, token=token)
            return SimpleNamespace(
                card_data=SimpleNamespace(base_model="richardyoung/Mythos-nano-heretic"),
                tags=["gguf"],
                siblings=[],
            )

    result = resolve_huggingface_model_reference(
        "https://huggingface.co/mradermacher/Mythos-nano-heretic-i1-GGUF?show_file_info=x",
        token="hf_secret",
        api_factory=FakeApi,
    )

    assert calls == {
        "endpoint": "https://huggingface.co",
        "repo_id": "mradermacher/Mythos-nano-heretic-i1-GGUF",
        "token": "hf_secret",
    }
    assert result["resolved_model"] == "richardyoung/Mythos-nano-heretic"
    assert result["authenticated"] is True
    assert "hf_secret" not in repr(result)


def test_resolver_keeps_non_gguf_repository():
    def factory(**_kwargs):
        return SimpleNamespace(
            model_info=lambda *_args, **_kwargs: SimpleNamespace(
                card_data={}, tags=["transformers"], siblings=[]
            )
        )

    result = resolve_huggingface_model_reference("owner/model", api_factory=factory)
    assert result["resolved_model"] == "owner/model"
    assert result["authenticated"] is False
