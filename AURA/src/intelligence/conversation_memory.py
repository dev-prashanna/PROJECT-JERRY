"""Bounded, session-scoped conversational memory.

Three distinct kinds of information are kept separate on purpose:

1. conversation history  – what was said (turns),
2. historical visual observations – what the VLM answered about past frames,
3. current observation   – never stored here; produced fresh per request.

A past observation is never silently reused as a "current" camera image; the
memory only ever *references* it by frame id and timestamp.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from src.orchestrator.schemas import ConversationTurn, VisualEvidence
from src.utils.logger import get_logger

log = get_logger("memory")

MAX_VISUAL_EVIDENCE_HISTORY = 6


@dataclass
class MemorySnapshot:
    turns: list[ConversationTurn] = field(default_factory=list)
    visual_observations: list[VisualEvidence] = field(default_factory=list)
    last_response: str = ""


class ConversationMemory:
    def __init__(self, max_turns: int = 8, persist_path: Optional[Path] = None):
        self.max_turns = max(1, int(max_turns))
        self._turns: deque[ConversationTurn] = deque(maxlen=self.max_turns)
        self._visuals: deque[VisualEvidence] = deque(maxlen=MAX_VISUAL_EVIDENCE_HISTORY)
        self._lock = threading.RLock()
        self._persist_path = persist_path

    # -- writes ----------------------------------------------------------
    def record(
        self,
        *,
        user_transcript: str,
        final_response: str,
        intent: Optional[str] = None,
        frame_id: Optional[str] = None,
        visual_observation: Optional[str] = None,
        route_tool: Optional[str] = None,
        error: Optional[str] = None,
    ) -> ConversationTurn:
        turn = ConversationTurn(
            user_transcript=user_transcript,
            final_response=final_response,
            intent=intent,
            frame_id=frame_id,
            visual_observation=visual_observation,
            route_tool=route_tool,
            error=error,
            timestamp=time.time(),
        )
        with self._lock:
            self._turns.append(turn)
        log.debug("memory +turn %s intent=%s", turn.turn_id, intent)
        self._maybe_persist()
        return turn

    def record_evidence(self, evidence: VisualEvidence) -> None:
        with self._lock:
            self._visuals.append(evidence)

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
            self._visuals.clear()
        log.info("conversation memory cleared")

    # -- reads -----------------------------------------------------------
    @property
    def turns(self) -> list[ConversationTurn]:
        with self._lock:
            return list(self._turns)

    @property
    def visual_observations(self) -> list[VisualEvidence]:
        with self._lock:
            return list(self._visuals)

    @property
    def last_response(self) -> str:
        with self._lock:
            return self._turns[-1].final_response if self._turns else ""

    def chat_messages(self) -> list[dict[str, str]]:
        """Compact role/content view for the LLM (bounded by max_turns)."""
        out: list[dict[str, str]] = []
        for turn in self.turns:
            if turn.user_transcript:
                out.append({"role": "user", "content": turn.user_transcript})
            if turn.final_response:
                out.append({"role": "assistant", "content": turn.final_response})
        return out

    def context_summary(self, max_chars: int = 900) -> str:
        """Human/LLM readable context including the most recent observations."""
        with self._lock:
            if not self._turns and not self._visuals:
                return ""
            lines: list[str] = []
            recent = list(self._turns)[-4:]
            for t in recent:
                lines.append(f"User: {t.user_transcript}")
                if t.final_response:
                    lines.append(f"AURA: {t.final_response}")
            if self._visuals:
                last = self._visuals[-1]
                lines.append(
                    f"Previous observation ({last.frame_id}): {last.raw_answer or '(none)'}"
                )
            text = "\n".join(lines).strip()
            if len(text) > max_chars:
                text = "…" + text[-max_chars:]
            return text

    def find_frame_for_reference(self, object_hint: str) -> Optional[VisualEvidence]:
        """Locate the most recent observation mentioning ``object_hint``."""
        hint = object_hint.lower().strip()
        if not hint:
            return None
        with self._lock:
            for ev in reversed(self._visuals):
                if hint in (ev.raw_answer or "").lower():
                    return ev
        return None

    def __iter__(self) -> Iterator[ConversationTurn]:
        return iter(self.turns)

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)

    # -- persistence (explicit opt-in only) ------------------------------
    def _maybe_persist(self) -> None:
        if not self._persist_path:
            return
        try:
            with self._lock:
                payload = [t.model_dump() for t in self._turns]
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            log.warning("could not persist history: %s", exc)

    @classmethod
    def load(cls, path: Path, max_turns: int = 8) -> "ConversationMemory":
        mem = cls(max_turns=max_turns, persist_path=path)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for item in data[-max_turns:]:
                    mem._turns.append(ConversationTurn(**item))
            except Exception as exc:  # noqa: BLE001
                log.warning("could not load history: %s", exc)
        return mem


def snapshot_memory(memory: ConversationMemory) -> MemorySnapshot:
    turns = memory.turns
    return MemorySnapshot(
        turns=turns,
        visual_observations=memory.visual_observations,
        last_response=memory.last_response,
    )
