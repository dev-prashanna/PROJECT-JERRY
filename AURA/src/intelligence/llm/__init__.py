"""Pluggable LLM chat providers.

Public surface::

    from src.intelligence.llm import create_llm_client, BaseLLMClient

    client = create_llm_client()          # honours LLM_PROVIDER
    result = client.chat([{"role": "user", "content": "hi"}])
"""

from src.intelligence.llm.base import (
    BaseLLMClient,
    LLMResult,
    classify_provider_error,
    parse_json_reply,
)
from src.intelligence.llm.registry import (
    available_providers,
    create_llm_client,
    describe_providers,
    register_llm,
)

__all__ = [
    "BaseLLMClient",
    "LLMResult",
    "available_providers",
    "classify_provider_error",
    "create_llm_client",
    "describe_providers",
    "parse_json_reply",
    "register_llm",
]
