"""Small, security-conscious helpers for Hugging Face model references."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse


DEFAULT_HF_ENDPOINT = "https://huggingface.co"
_HF_HOSTS = frozenset({"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"})


def normalize_huggingface_endpoint(value: str | None) -> str:
    """Validate an explicit Hub endpoint without silently selecting a mirror."""
    endpoint = (value or DEFAULT_HF_ENDPOINT).strip().rstrip("/")
    if any(ord(character) < 32 for character in endpoint):
        raise ValueError("Hugging Face endpoint contains control characters")
    parsed = urlparse(endpoint)
    if parsed.username or parsed.password:
        raise ValueError("Hugging Face endpoint must not contain credentials")
    hostname = (parsed.hostname or "").lower()
    local_http = parsed.scheme == "http" and hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not local_http:
        raise ValueError("Hugging Face endpoint must use HTTPS (HTTP is allowed only locally)")
    if not hostname or parsed.query or parsed.fragment:
        raise ValueError("Hugging Face endpoint must be an origin without query or fragment")
    return endpoint


def normalize_huggingface_reference(value: str) -> str:
    """Turn a Hub repository URL or ``owner/repository`` value into a repo ID."""
    reference = str(value).strip()
    if not reference or any(ord(character) < 32 for character in reference):
        raise ValueError("Hugging Face model reference is empty or contains control characters")
    if "://" not in reference:
        parts = reference.strip("/").split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError("Use a Hugging Face owner/repository ID or repository URL")
        return "/".join(parts)
    parsed = urlparse(reference)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in _HF_HOSTS:
        raise ValueError("Only HTTPS huggingface.co or hf.co model links are accepted")
    parts = [part for part in parsed.path.split("/") if part]
    if parts and parts[0] in {"datasets", "spaces"}:
        raise ValueError("A model repository is required, not a dataset or Space")
    if len(parts) < 2:
        raise ValueError("Hugging Face model URL does not contain owner/repository")
    return "/".join(parts[:2])


def _base_model_from_info(info: object) -> str | None:
    card_data = getattr(info, "card_data", None)
    raw = getattr(card_data, "base_model", None)
    if raw is None and isinstance(card_data, dict):
        raw = card_data.get("base_model")
    candidates = [raw] if isinstance(raw, str) else list(raw or [])
    tags = getattr(info, "tags", None) or []
    candidates.extend(tag.removeprefix("base_model:") for tag in tags if tag.startswith("base_model:"))
    for candidate in candidates:
        try:
            return normalize_huggingface_reference(str(candidate))
        except ValueError:
            continue
    return None


def resolve_huggingface_model_reference(
    value: str,
    *,
    token: str | None = None,
    endpoint: str | None = None,
    api_factory: Callable[..., Any] | None = None,
) -> dict[str, object]:
    """Resolve a public or authenticated model repository to calibration metadata."""
    repo_id = normalize_huggingface_reference(value)
    normalized_endpoint = normalize_huggingface_endpoint(endpoint)
    if token is not None and any(ord(character) < 32 for character in token):
        raise ValueError("Hugging Face token contains control characters")
    if api_factory is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required to resolve model links") from exc
        api_factory = HfApi
    api = api_factory(endpoint=normalized_endpoint)
    info = api.model_info(repo_id, token=token or None)
    base_model = _base_model_from_info(info)
    tags = set(getattr(info, "tags", None) or [])
    siblings = getattr(info, "siblings", None) or []
    is_gguf = "gguf" in {str(tag).lower() for tag in tags} or any(
        str(getattr(item, "rfilename", "")).lower().endswith(".gguf") for item in siblings
    )
    return {
        "repo_id": repo_id,
        "base_model": base_model,
        "resolved_model": base_model if is_gguf and base_model else repo_id,
        "is_gguf_repository": is_gguf,
        "authenticated": bool(token),
        "endpoint": normalized_endpoint,
    }
