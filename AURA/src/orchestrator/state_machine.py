"""Validated state machine for the AURA request lifecycle.

States::

    IDLE → LISTENING → TRANSCRIBING → INTERPRETING → [CAPTURING → PERCEIVING]
        → GENERATING → SPEAKING → IDLE

``ERROR`` is reachable from every active state and always returns to
``IDLE``.  A general-knowledge request skips CAPTURING/PERCEIVING.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from src.utils.logger import get_logger

log = get_logger("state")


class State(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    INTERPRETING = "INTERPRETING"
    CAPTURING = "CAPTURING"
    PERCEIVING = "PERCEIVING"
    GENERATING = "GENERATING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"


ALLOWED: dict[State, frozenset[State]] = {
    State.IDLE: frozenset({State.LISTENING, State.TRANSCRIBING, State.INTERPRETING, State.SHUTDOWN}),
    State.LISTENING: frozenset({State.TRANSCRIBING, State.IDLE, State.ERROR, State.SHUTDOWN}),
    State.TRANSCRIBING: frozenset({State.INTERPRETING, State.IDLE, State.ERROR, State.SHUTDOWN}),
    State.INTERPRETING: frozenset(
        {State.CAPTURING, State.GENERATING, State.IDLE, State.ERROR, State.SHUTDOWN}
    ),
    State.CAPTURING: frozenset({State.PERCEIVING, State.GENERATING, State.ERROR, State.SHUTDOWN}),
    State.PERCEIVING: frozenset({State.GENERATING, State.ERROR, State.SHUTDOWN}),
    State.GENERATING: frozenset({State.SPEAKING, State.IDLE, State.ERROR, State.SHUTDOWN}),
    State.SPEAKING: frozenset({State.IDLE, State.LISTENING, State.ERROR, State.SHUTDOWN}),
    State.ERROR: frozenset({State.IDLE, State.SHUTDOWN}),
    State.SHUTDOWN: frozenset(),
}


class InvalidTransition(RuntimeError):
    pass


@dataclass
class Transition:
    previous: State
    current: State
    reason: str = ""
    at: float = field(default_factory=time.time)


class StateMachine:
    def __init__(self, initial: State = State.IDLE):
        self._lock = threading.RLock()
        self._state = initial
        self._history: list[Transition] = []
        self._listeners: list[Callable[[Transition], None]] = []

    # -- properties ------------------------------------------------------
    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    @property
    def history(self) -> list[Transition]:
        with self._lock:
            return list(self._history)

    @property
    def busy(self) -> bool:
        return self.state not in {State.IDLE, State.LISTENING, State.SHUTDOWN}

    def on_transition(self, callback: Callable[[Transition], None]) -> None:
        self._listeners.append(callback)

    # -- transitions -----------------------------------------------------
    def can(self, target: State) -> bool:
        return target in ALLOWED[self.state]

    def transition(self, target: State, reason: str = "") -> Transition:
        with self._lock:
            current = self._state
            if target not in ALLOWED[current]:
                raise InvalidTransition(f"{current.value} -> {target.value} not allowed ({reason})")
            tr = Transition(previous=current, current=target, reason=reason)
            self._state = target
            self._history.append(tr)
            if len(self._history) > 200:
                del self._history[:100]
        log.debug("state %s -> %s (%s)", tr.previous.value, tr.current.value, reason)
        for cb in list(self._listeners):
            try:
                cb(tr)
            except Exception:  # pragma: no cover - listeners must not break transitions
                log.exception("state listener failed")
        return tr

    def try_transition(self, target: State, reason: str = "") -> bool:
        try:
            self.transition(target, reason)
            return True
        except InvalidTransition:
            log.warning("rejected transition %s -> %s (%s)", self.state.value, target.value, reason)
            return False

    def reset(self, reason: str = "reset") -> None:
        """Force back to IDLE from any state (error recovery)."""
        with self._lock:
            if self._state == State.IDLE:
                return
            self.transition(State.ERROR, reason) if State.ERROR in ALLOWED[self._state] else None
        if self.state != State.IDLE:
            self.try_transition(State.IDLE, reason)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self._state.value,
                "last": self._history[-1].current.value if self._history else None,
                "reason": self._history[-1].reason if self._history else "",
            }
