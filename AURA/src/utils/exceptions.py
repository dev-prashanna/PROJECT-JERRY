"""Typed exceptions with actionable messages."""

from __future__ import annotations


class AURAError(Exception):
    """Base class. ``hint`` is surfaced to the user in the GUI/CLI."""

    default_hint = ""

    def __init__(self, message: str, *, hint: str = "", cause: BaseException | None = None):
        super().__init__(message)
        self.hint = hint or self.default_hint
        if cause is not None:
            self.__cause__ = cause

    def user_message(self) -> str:
        msg = str(self)
        return f"{msg}\n→ {self.hint}" if self.hint else msg


class ConfigurationError(AURAError):
    default_hint = "Check AURA/.env and the Settings panel."


class DeviceError(AURAError):
    default_hint = "Verify the device is connected and not used by another application."


class MicrophoneError(DeviceError):
    default_hint = "Check `python3 scripts/inspect_environment.py` for input devices."


class CameraError(DeviceError):
    default_hint = "Close other apps using the webcam or set CAMERA_INDEX."


class ModelNotFoundError(AURAError):
    default_hint = "Run scripts/inspect_environment.py to locate the model weights."


class InferenceError(AURAError):
    default_hint = "Check GPU memory (nvidia-smi) and the logs/aura.log file."


class GPUOutOfMemoryError(InferenceError):
    default_hint = (
        "Free GPU memory or set STT_DEVICE=cpu / TTS_DEVICE=cpu / VLM_DEVICE=cpu."
    )


class CloudServiceError(AURAError):
    default_hint = "Check internet connectivity and the Groq status page."


class RateLimitError(CloudServiceError):
    default_hint = "Wait before retrying; AURA retries with bounded backoff."


class InvalidModelError(CloudServiceError):
    default_hint = "Set GROQ_MODEL to an id listed by `python3 scripts/verify_groq.py --list`."


class AuthenticationError(CloudServiceError):
    default_hint = "Set a valid GROQ_API_KEY in AURA/.env."


class OfflineError(CloudServiceError):
    default_hint = "AURA is in offline degraded mode; disable AURA_OFFLINE to use Groq."


class InvalidToolCallError(AURAError):
    default_hint = "The LLM proposed a tool call that failed local validation; it was refused."


class StaleFrameError(AURAError):
    default_hint = "The camera frame was older than CAMERA_MAX_FRAME_AGE_S."


class QueueFullError(AURAError):
    default_hint = "The pipeline is busy; wait for the current request to finish."


class ShutdownError(AURAError):
    default_hint = ""


def classify_groq_error(exc: BaseException, provider: str = "Groq") -> AURAError:
    """Translate a Groq/OpenAI-style SDK error into an AURA exception.

    ``provider`` only affects the message wording; the exception *type*
    mapping is identical for every chat provider (401→AuthenticationError,
    429→RateLimitError, 404→InvalidModelError, 5xx/network→CloudServiceError).
    """
    label = provider or "Groq"
    status = getattr(exc, "status_code", None)
    text = str(exc)
    if status == 401 or "Invalid API Key" in text or "invalid api key" in text.lower():
        return AuthenticationError(f"{label} rejected the API key.", cause=exc)
    if status == 429 or "rate limit" in text.lower():
        return RateLimitError(f"{label} rate limit reached.", cause=exc)
    if status == 404 or "model_not_found" in text or "does not exist" in text:
        return InvalidModelError(f"Model unavailable: {text[:160]}", cause=exc)
    if status in (500, 502, 503, 504):
        return CloudServiceError(f"{label} server error ({status}).", cause=exc)
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return CloudServiceError(f"Network problem talking to {label}: {text[:160]}", cause=exc)
    return CloudServiceError(f"{label} request failed: {text[:200]}", cause=exc)
