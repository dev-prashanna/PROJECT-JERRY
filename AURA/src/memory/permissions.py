"""Consent and permission checks for biometric and personal memory.

Rules enforced here:

* Face enrollment always requires consent (explicit utterance about oneself,
  or a validated local consent token issued by the GUI/CLI).
* Third-party enrollment ("this is Alice") never auto-persists biometric data.
* Deletion of an enrolled identity revokes bindings and removes embeddings.
"""

from __future__ import annotations

import secrets
import time
from datetime import timedelta
from typing import Any, Optional

from src.memory.exceptions import ConsentRequired, MemoryPermissionDenied
from src.memory.models import ConsentRecord, utcnow
from src.utils.logger import get_logger

log = get_logger("memory.permissions")

DEFAULT_CONSENT_TTL_S = 3600.0

#: Phrases that indicate the speaker is enrolling *themselves*.
_SELF_ENROLL_MARKERS = (
    "my face",
    "me as",
    "i am",
    "i'm ",
    "learn my",
    "remember my",
    "enroll me",
    "my identity",
    "remember me",
)

#: Phrases that point at a *third party* without explicit consent handling.
_THIRD_PARTY_MARKERS = (
    "this is ",
    "that is ",
    "she is ",
    "he is ",
    "they are ",
    "her face",
    "his face",
    "their face",
)


def looks_like_self_enrollment(transcript: str) -> bool:
    lowered = (transcript or "").lower()
    return any(m in lowered for m in _SELF_ENROLL_MARKERS)


def looks_like_third_party_enrollment(transcript: str) -> bool:
    lowered = (transcript or "").lower()
    return any(m in lowered for m in _THIRD_PARTY_MARKERS) and not looks_like_self_enrollment(transcript)


def issue_consent_token(
    store,
    subject_description: str,
    *,
    scope: str = "face_enrollment",
    ttl_s: float = DEFAULT_CONSENT_TTL_S,
    method: str = "local_confirmation",
    metadata: Optional[dict[str, Any]] = None,
) -> ConsentRecord:
    """Create a short-lived consent grant (GUI/CLI confirmation path)."""
    consent = ConsentRecord(
        subject_description=subject_description,
        scope=scope,
        granted=True,
        granted_at=utcnow(),
        expires_at=utcnow() + timedelta(seconds=ttl_s),
        method=method,
        metadata=dict(metadata or {}),
    )
    store.save_consent(consent)
    log.info("issued consent %s for %r", consent.consent_id, subject_description)
    return consent


def validate_consent_token(store, token: str, *, scope: str = "face_enrollment") -> ConsentRecord:
    if not token or not str(token).strip():
        raise ConsentRequired("a validated consent token is required for this operation")
    consent = store.get_consent(str(token).strip())
    if consent is None or not consent.granted:
        raise ConsentRequired("consent token is unknown or not granted")
    if consent.scope != scope:
        raise MemoryPermissionDenied(f"consent scope {consent.scope!r} does not cover {scope!r}")
    if consent.expires_at is not None and consent.expires_at < utcnow():
        raise ConsentRequired("consent token has expired; confirm enrollment again")
    return consent


def authorize_face_enrollment(
    store,
    *,
    transcript: str = "",
    consent_token: Optional[str] = None,
    require_confirmation: bool = True,
    face_count: int = 1,
) -> tuple[Optional[ConsentRecord], str]:
    """Return (consent, mode) or raise.

    Modes:

    * ``token``  – validated GUI/CLI consent token
    * ``self``   – explicit self-enrollment utterance
    """
    if consent_token:
        return validate_consent_token(store, consent_token), "token"

    if not require_confirmation:
        log.warning("enrollment confirmation disabled by configuration")
        return None, "self"

    if face_count > 1:
        raise ConsentRequired(
            "multiple faces are visible; select the person to enroll and confirm"
        )

    if looks_like_self_enrollment(transcript):
        return None, "self"

    if looks_like_third_party_enrollment(transcript):
        raise MemoryPermissionDenied(
            "cannot enroll a third party from speech alone; explicit consent "
            "from that person is required (use the GUI confirmation)"
        )

    raise ConsentRequired(
        "face enrollment requires an explicit request such as "
        "'please remember my face as Alex'"
    )


def authorize_deletion(entity_type: str, *, confirmed: bool) -> None:
    if entity_type == "person" and not confirmed:
        raise MemoryPermissionDenied(
            "deleting a person identity requires explicit confirmation"
        )
