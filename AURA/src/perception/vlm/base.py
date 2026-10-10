"""Provider-agnostic VLM (vision-language model) contract.

Any engine AURA can use to answer questions about a camera frame
implements this interface — it is exactly what the pipeline consumes:

* ``load(force=False)``   – warm the model (may be a no-op for cloud APIs)
* ``ask(frame, question) -> VisualEvidence``
* ``status() -> dict``
* ``unload()``            – free resources

The local default is Moondream-2; cloud vision backends (GPT-4o, Groq
vision models, any OpenAI-compatible VLM) plug in through
:func:`src.perception.vlm.registry.create_vlm` with ``VLM_PROVIDER``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VLMEngine(Protocol):
    """What every vision backend must provide."""

    def load(self, force: bool = False) -> Any: ...

    def ask(self, image: Any, question: str, **kwargs: Any) -> Any:
        """Return a :class:`~src.orchestrator.schemas.VisualEvidence`."""
        ...

    def status(self) -> dict[str, Any]: ...

    def unload(self) -> None: ...
