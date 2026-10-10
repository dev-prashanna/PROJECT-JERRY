"""Known LLM/VLM provider presets.

This module is intentionally dependency-free (no project imports) so both
``config/settings.py`` and ``src/intelligence/llm/registry.py`` can use it
without circular imports.

A preset describes how to reach a provider's API:

* ``protocol``   – ``groq`` (native SDK), ``openai`` (OpenAI-compatible
  chat-completions), or ``anthropic`` (Messages API).
* ``base_url``   – default endpoint.
* ``key_env``    – environment variables checked, in order, for the API key.
* ``model``      – default model id.
* ``fallback``   – default secondary model (same provider).
* ``aliases``    – other spellings accepted in ``LLM_PROVIDER``.

Anything not listed here can still be used through the generic
``openai_compat`` provider: set ``LLM_BASE_URL`` + ``LLM_API_KEY`` (+ or
``LLM_MODEL``) and AURA will talk to any OpenAI-compatible server (vLLM,
Ollama, LM Studio, Together, Mistral, …).
"""

from __future__ import annotations

from typing import Any

#: provider-name → preset (lower-case keys, no aliases).
LLM_PRESETS: dict[str, dict[str, Any]] = {
    "groq": {
        "protocol": "groq",
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": ["GROQ_API_KEY"],
        "model": "openai/gpt-oss-120b",
        "fallback": "qwen/qwen3.8-27b",
        "docs": "https://console.groq.com/keys",
    },
    "openai": {
        "protocol": "openai",
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "key_env": ["OPENAI_API_KEY"],
        "model": "gpt-4o-mini",
        "fallback": "gpt-4o-mini",
        "docs": "https://platform.openai.com/api-keys",
    },
    "anthropic": {
        "protocol": "anthropic",
        "label": "Anthropic (Claude)",
        "base_url": "https://api.anthropic.com",
        "key_env": ["ANTHROPIC_API_KEY"],
        "model": "claude-sonnet-4-5",
        "fallback": "claude-haiku-4-5",
        "docs": "https://console.anthropic.com/settings/keys",
    },
    "moonshot": {
        "protocol": "openai",
        "label": "Moonshot (Kimi)",
        "base_url": "https://api.moonshot.ai/v1",
        "key_env": ["MOONSHOT_API_KEY", "KIMI_API_KEY"],
        "model": "kimi-k2.5",
        "fallback": "moonshot-v1-32k",
        "docs": "https://platform.moonshot.cn/console/api-keys",
    },
    "deepseek": {
        "protocol": "openai",
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "key_env": ["DEEPSEEK_API_KEY"],
        "model": "deepseek-chat",
        "fallback": "deepseek-reasoner",
        "docs": "https://platform.deepseek.com/api_keys",
    },
    "mimo": {
        "protocol": "openai",
        "label": "Xiaomi MiMo",
        "base_url": "https://api.mimo.ai/v1",  # override with LLM_BASE_URL if different
        "key_env": ["MIMO_API_KEY"],
        "model": "mimo-v2-flash",
        "fallback": "",
        "docs": "",
    },
    "openai_compat": {
        "protocol": "openai",
        "label": "OpenAI-compatible server",
        "base_url": "",  # required: LLM_BASE_URL
        "key_env": ["LLM_API_KEY"],
        "model": "",
        "fallback": "",
        "docs": "",
    },
}

#: alias → canonical preset name.
LLM_ALIASES: dict[str, str] = {
    "kimi": "moonshot",
    "claude": "anthropic",
    "chatgpt": "openai",
    "gpt": "openai",
    "mimo-v2": "mimo",
    "local": "openai_compat",
    "ollama": "openai_compat",
    "vllm": "openai_compat",
    "lmstudio": "openai_compat",
}

#: VLM provider presets (same idea, vision-specific defaults).
VLM_PRESETS: dict[str, dict[str, Any]] = {
    "moondream": {
        "protocol": "local",
        "label": "Moondream-2 (local)",
    },
    "openai_compat": {
        "protocol": "openai",
        "label": "OpenAI-compatible vision model",
        "base_url": "",
        "key_env": ["VLM_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"],
        "model": "gpt-4o-mini",
    },
    "openai": {
        "protocol": "openai",
        "label": "OpenAI vision",
        "base_url": "https://api.openai.com/v1",
        "key_env": ["VLM_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"],
        "model": "gpt-4o-mini",
    },
    "groq": {
        "protocol": "openai",
        "label": "Groq vision (e.g. meta-llama/llama-4-scout-17b-16e-instruct)",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": ["VLM_API_KEY", "GROQ_API_KEY", "LLM_API_KEY"],
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
}

VLM_ALIASES: dict[str, str] = {
    "moondream2": "moondream",
    "local": "moondream",
    "cloud": "openai_compat",
    "vision": "openai_compat",
    "gpt4o": "openai",
    "gpt-4o": "openai",
}


def resolve_llm_preset(name: str) -> dict[str, Any]:
    """Return the preset for ``name`` (aliases resolved); default is groq."""
    key = (name or "groq").strip().lower()
    key = LLM_ALIASES.get(key, key)
    if key in LLM_PRESETS:
        return dict(LLM_PRESETS[key])
    # Unknown name: treat as a generic OpenAI-compatible provider whose
    # identity is the name itself (so `LLM_PROVIDER=my-vllm` just works).
    return {
        "protocol": "openai",
        "label": key,
        "base_url": "",
        "key_env": [],
        "model": "",
        "fallback": "",
        "docs": "",
    }


def resolve_vlm_preset(name: str) -> dict[str, Any]:
    key = (name or "moondream").strip().lower()
    key = VLM_ALIASES.get(key, key)
    if key in VLM_PRESETS:
        return dict(VLM_PRESETS[key])
    return {"protocol": "openai", "label": key, "base_url": "", "key_env": [], "model": ""}


def known_llm_providers() -> list[str]:
    return sorted(set(LLM_PRESETS) | set(LLM_ALIASES))


def known_vlm_providers() -> list[str]:
    return sorted(set(VLM_PRESETS) | set(VLM_ALIASES))
