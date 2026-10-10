"""VLM provider registry — the single place to add a new vision backend.

``VLM_PROVIDER`` selects the engine (see ``config/provider_presets.py``):

* ``moondream`` (default) – local Moondream-2, no network
* ``openai`` / ``groq`` / ``openai_compat`` – any OpenAI-compatible vision
  endpoint (GPT-4o, Groq vision models, Qwen-VL, LLaVA-on-vLLM, …)

Adding your own engine
----------------------

    from src.perception.vlm.registry import register_vlm

    class MyVLM:
        def load(self, force=False): ...
        def ask(self, frame, question, **kw) -> VisualEvidence: ...
        def status(self): ...
        def unload(self): ...

    register_vlm("myvlm", lambda settings: MyVLM(settings))

then set ``VLM_PROVIDER=myvlm``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from config.provider_presets import known_vlm_providers, resolve_vlm_preset
from config.settings import VLMSettings, get_settings
from src.utils.exceptions import ModelNotFoundError
from src.utils.logger import get_logger

log = get_logger("vlm.registry")

#: name → ``factory(settings: VLMSettings) -> engine``
_VLM_FACTORIES: dict[str, Callable[[VLMSettings], Any]] = {}


def register_vlm(name: str, factory: Callable[[VLMSettings], Any]) -> None:
    _VLM_FACTORIES[name.strip().lower()] = factory


def _make_moondream(settings: VLMSettings):
    from src.perception.moondream_engine import get_engine

    return get_engine(settings)


def _make_cloud(settings: VLMSettings, provider_name: str):
    from src.perception.vlm.cloud import CloudVLM

    return CloudVLM(settings, provider_name=provider_name)


def _builtin_factory(settings: VLMSettings):
    preset = resolve_vlm_preset(settings.provider)
    protocol = str(preset.get("protocol", "openai"))
    if protocol == "local":
        return _make_moondream(settings)
    return _make_cloud(settings, settings.provider)


def create_vlm(settings: Optional[VLMSettings] = None) -> Any:
    """Build the vision engine selected by ``VLM_PROVIDER``."""
    settings = settings or get_settings().vlm
    name = (settings.provider or "moondream").strip().lower()
    factory = _VLM_FACTORIES.get(name) or _builtin_factory
    engine = factory(settings)
    log.debug("VLM provider ready: %s", name)
    return engine


def available_vlm_providers() -> list[str]:
    return sorted(set(known_vlm_providers()) | set(_VLM_FACTORIES))
