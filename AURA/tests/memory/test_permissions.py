"""Consent rules and permission enforcement."""

from __future__ import annotations

import pytest

from src.memory.exceptions import ConsentRequired, MemoryPermissionDenied
from src.memory.permissions import (
    authorize_face_enrollment,
    issue_consent_token,
    looks_like_self_enrollment,
    looks_like_third_party_enrollment,
    validate_consent_token,
)


def test_self_enrollment_phrases():
    assert looks_like_self_enrollment("Please remember my face as Alex")
    assert looks_like_self_enrollment("Learn my face as Alex")
    assert looks_like_self_enrollment("Remember me as Sam")
    assert not looks_like_self_enrollment("What is the capital of France?")


def test_third_party_detection():
    assert looks_like_third_party_enrollment("This is Alice")
    assert not looks_like_third_party_enrollment("Please remember my face as Alex")


def test_self_enrollment_authorized_without_token(store):
    consent, mode = authorize_face_enrollment(
        store, transcript="Please remember my face as Alex", face_count=1
    )
    assert mode == "self"


def test_third_party_blocked(store):
    with pytest.raises(MemoryPermissionDenied):
        authorize_face_enrollment(store, transcript="This is Alice", face_count=1)


def test_multi_face_blocks_enrollment(store):
    with pytest.raises(ConsentRequired):
        authorize_face_enrollment(
            store, transcript="Please remember my face as Alex", face_count=2
        )


def test_non_enrollment_utterance_blocked(store):
    with pytest.raises(ConsentRequired):
        authorize_face_enrollment(store, transcript="Hello there", face_count=1)


def test_consent_token_lifecycle(store):
    consent = issue_consent_token(store, "Alex", ttl_s=60)
    got = validate_consent_token(store, consent.consent_id)
    assert got.granted and got.subject_description == "Alex"
    with pytest.raises(ConsentRequired):
        validate_consent_token(store, "not-a-token")


def test_expired_consent_rejected(store):
    consent = issue_consent_token(store, "Alex", ttl_s=-1)
    with pytest.raises(ConsentRequired):
        validate_consent_token(store, consent.consent_id)
