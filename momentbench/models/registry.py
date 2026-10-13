"""Model adapter registry."""

from __future__ import annotations

from typing import Any

from momentbench.models.base import ModelAdapter

_REGISTRY: dict[str, type[ModelAdapter]] = {}


def register_adapter(name: str):
    """Decorator to register a model adapter class."""
    def decorator(cls: type[ModelAdapter]):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_adapter_class(name: str) -> type[ModelAdapter]:
    """Look up a registered adapter class by name, importing built-ins on demand."""
    if name not in _REGISTRY:
        _load_builtin(name)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model adapter: {name}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def get_adapter(name: str, **kwargs: Any) -> ModelAdapter:
    """Instantiate a registered model adapter by name."""
    return get_adapter_class(name)(**kwargs)


def _load_builtin(name: str) -> None:
    """Attempt to import a built-in adapter module."""
    builtins = {
        "gemini": "momentbench.models.gemini",
        "openai": "momentbench.models.openai_adapter",
        "claude": "momentbench.models.claude",
        "mixpeek": "momentbench.models.mixpeek_adapter",
        "qwen-vl": "momentbench.models.qwen_vl",
        "e5-omni": "momentbench.models.e5_omni",
        "hf-vlm": "momentbench.models.hf_vlm",
        "twelvelabs": "momentbench.models.twelvelabs_adapter",
        "clip": "momentbench.models.clip_adapter",
        "mixedbread": "momentbench.models.mixedbread_adapter",
        "jina-clip": "momentbench.models.jina_clip_adapter",
    }
    if name in builtins:
        import importlib
        importlib.import_module(builtins[name])
