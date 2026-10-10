"""Pluggable vision-language model backends.

Public surface::

    from src.perception.vlm import create_vlm

    vlm = create_vlm()                      # honours VLM_PROVIDER
    evidence = vlm.ask(frame, "What is visible?")
"""

from src.perception.vlm.base import VLMEngine
from src.perception.vlm.registry import available_vlm_providers, create_vlm, register_vlm

__all__ = ["VLMEngine", "available_vlm_providers", "create_vlm", "register_vlm"]
