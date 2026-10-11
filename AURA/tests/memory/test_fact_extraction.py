"""Fact classification and extraction."""

from __future__ import annotations

from src.memory.fact_extractor import (
    StatementClass,
    classify_statement,
    extract_deterministic,
    extract_facts,
    parse_llm_reply,
)


def test_name_assertion():
    r = extract_deterministic("My name is Alex.")
    assert r.classification == StatementClass.EXPLICIT_ASSERTION
    assert any(f.predicate == "name" and f.object_value == "Alex" for f in r.facts)


def test_call_me():
    r = extract_deterministic("Call me Sam")
    assert any(f.object_value == "Sam" for f in r.facts)


def test_project_name():
    r = extract_deterministic("My robotics project is called NATERIDA")
    assert any(f.object_value == "NATERIDA" for f in r.facts)


def test_hypothetical_not_stored():
    assert classify_statement("If I were called Alex, that would be fun") == StatementClass.HYPOTHETICAL
    r = extract_deterministic("If I were called Alex, that would be fun")
    assert not r.facts and not r.corrections


def test_negation_not_stored():
    assert classify_statement("My name is not Alex") == StatementClass.NEGATION
    r = extract_deterministic("My name is not Alex")
    assert not r.facts


def test_question_not_stored():
    r = extract_deterministic("What is my name?")
    assert r.classification == StatementClass.QUESTION
    assert not r.facts


def test_quoted_speech_not_stored():
    r = extract_deterministic("Someone said Alice built this")
    assert r.classification == StatementClass.QUOTED_SPEECH
    assert not r.facts


def test_correction_detected():
    r = extract_deterministic("Actually the project is now called NATERIDA-X")
    assert r.classification == StatementClass.CORRECTION
    assert any(c.object_value.startswith("NATERIDA") for c in r.corrections)


def test_remember_prefix_stripped():
    r = extract_deterministic("Remember that my project is called HELIOS")
    assert any(f.object_value == "HELIOS" for f in r.facts)


def test_ownership_fact():
    r = extract_deterministic("I built a custom ESP32 motor controller")
    assert r.facts
    assert "esp32" in r.facts[0].object_value.lower() or "controller" in r.facts[0].object_value.lower()


def test_llm_reply_parsing():
    r = parse_llm_reply(
        {"facts": [{"subject": "current_speaker", "predicate": "likes", "object": "tea"}], "correction": False}
    )
    assert r.facts[0].predicate == "likes"
    r2 = parse_llm_reply({"facts": [{"subject": "my_project", "predicate": "name", "object": "X"}], "correction": True})
    assert r2.corrections


def test_extract_facts_without_llm_is_deterministic():
    r = extract_facts("My name is Alex", client=None, use_llm=True)
    assert r.method == "deterministic"
