"""Text → sentence segmentation for incremental TTS.

Splits on sentence boundaries while keeping abbreviations and decimals
intact, and respects a character budget so a single utterance handed to the
voice model stays short enough to synthesise quickly.
"""

from __future__ import annotations

import re
from typing import List

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "e.g",
    "i.e", "fig", "no", "vol", "dept", "est", "approx", "inc", "ltd", "co",
}

_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE = re.compile(r"(?<=[,;:])\s+")


def split_sentences(text: str, max_chars: int = 300) -> list[str]:
    """Split ``text`` into speakable chunks of at most ``max_chars``."""
    text = " ".join((text or "").split())
    if not text:
        return []

    sentences: list[str] = []
    carry = ""
    for part in _BOUNDARY.split(text):
        part = part.strip()
        if not part:
            continue
        if carry:
            part = f"{carry} {part}".strip()
            carry = ""
        if _ends_with_abbreviation(part):
            carry = part
            continue
        sentences.extend(_split_long(part, max_chars))
    if carry:
        sentences.append(carry)

    # Hard-split anything still too long (lists, run-on clauses).
    out: list[str] = []
    for s in sentences:
        if len(s) <= max_chars:
            out.append(s)
        else:
            out.extend(_split_long(s, max_chars))
    return [s for s in (x.strip() for x in out) if s]


def _ends_with_abbreviation(segment: str) -> bool:
    """True when the text ends with an abbreviation/initial, i.e. the period
    that just triggered a split is not a sentence boundary."""
    tail = segment.rstrip()
    if not tail.endswith("."):
        return False
    # single-letter initial, e.g. "Ask J." -> "J. Smith"
    if re.search(r"\b[A-Z]\.$", tail):
        return True
    # dotted abbreviation, e.g. "at 5 p.m."
    if re.search(r"\b(?:[A-Za-z]\.){1,4}$", tail):
        return True
    m = re.search(r"([A-Za-z]+)\.$", tail)
    return bool(m) and m.group(1).lower() in _ABBREV


def _split_long(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    parts: list[str] = []
    current = ""
    for clause in _CLAUSE.split(sentence):
        candidate = f"{current} {clause}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            parts.append(current)
        if len(clause) <= max_chars:
            current = clause
            continue
        # word-level split as a last resort
        words = clause.split(" ")
        current = ""
        for w in words:
            trial = f"{current} {w}".strip()
            if len(trial) > max_chars and current:
                parts.append(current)
                current = w
            else:
                current = trial
    if current:
        parts.append(current)
    return parts


def streaming_chunks(text: str, max_chars: int = 300) -> List[str]:
    """Alias used by the streaming TTS path."""
    return split_sentences(text, max_chars)
