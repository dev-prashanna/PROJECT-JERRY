"""Shared fixtures: hardware- and network-free fakes.

Nothing in this file touches a microphone, a camera, the GPU or the internet,
so the whole suite runs on CI or on a machine without the robot.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from config.settings import Settings, reset_settings_cache  # noqa: E402
from src.intelligence.groq_client import LLMResult  # noqa: E402
from src.intelligence.intent_classifier import IntentRouter  # noqa: E402
from src.intelligence.response_generator import ResponseGenerator  # noqa: E402
from src.orchestrator.schemas import RoutePlan  # noqa: E402
from src.perception.frame_buffer import Frame, FrameBuffer  # noqa: E402
from src.speech.whisper_engine import TranscriptionResult  # noqa: E402
from src.orchestrator.agent import AIAgent  # noqa: E402


# ---------------------------------------------------------------------------
# fake cloud LLM
# ---------------------------------------------------------------------------

class FakeGroqClient:
    """Scripted stand-in for :class:`GroqClient`."""

    def __init__(
        self,
        plan: Optional[dict] = None,
        answer: str = "This is a scripted answer.",
        fail: Optional[Exception] = None,
    ):
        self.plan = plan
        self.answer = answer
        self.fail = fail
        self.offline = False
        self.calls: list[dict[str, Any]] = []
        self.model_chain = ["fake-model"]

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.fail is not None:
            raise self.fail
        return LLMResult(
            text=self.answer,
            model="fake-model",
            latency_s=0.001,
            prompt_tokens=10,
            completion_tokens=5,
        )

    def chat_json(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.fail is not None:
            raise self.fail
        plan = self.plan or {
            "intent": "general_question",
            "requires_camera": False,
            "requires_vlm": False,
            "tool": "respond_directly",
            "tool_arguments": {},
            "confidence": 0.99,
            "rationale": "scripted",
        }
        import json

        return dict(plan), LLMResult(
            text=json.dumps(plan), model="fake-model", latency_s=0.001
        )

    def ping(self) -> bool:
        return self.fail is None


# ---------------------------------------------------------------------------
# fake hardware
# ---------------------------------------------------------------------------

def _test_image(value: int = 90) -> np.ndarray:
    """A non-uniform image so blank-frame detection stays quiet."""
    rng = np.random.default_rng(42)
    img = np.full((48, 64, 3), value, dtype=np.uint8)
    img[:, :, 0] = rng.integers(0, 255, size=(48, 64), dtype=np.uint8)
    return img


class FakeCamera:
    def __init__(self, image: Optional[np.ndarray] = None, fail: bool = False):
        self.buffer = FrameBuffer(max_age_s=5.0)
        self._image = image if image is not None else _test_image()
        self.buffer.put(self._image, source="fake")
        self.fail = fail
        self.started = False
        self.closed = False
        self.snapshots = 0

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def snapshot(self, max_age_s: Optional[float] = None, wait_s: float = 0.0) -> Frame:
        if self.fail:
            from src.utils.exceptions import CameraError

            raise CameraError("fake camera failure")
        self.snapshots += 1
        return self.buffer.latest_fresh(max_age_s or 5.0) or self.buffer.get()

    def info(self) -> dict:
        return {
            "status": "running" if not self.fail else "error",
            "running": not self.fail,
            "device_index": 0,
            "fps": 30.0,
            "resolution": "64x48",
            "frames_seen": 1,
            "last_frame_tag": "frame_0001",
            "last_frame_age_s": 0.0,
            "error": "fake" if self.fail else None,
            "blank_warning": None,
            "buffer": self.buffer.stats,
        }


class FakeVLM:
    """Scripted vision model: no GPU, no weights, deterministic answers."""

    def __init__(self, answer: str = "a laptop and a red bottle", fail: bool = False):
        self.answer = answer
        self.fail = fail
        self.calls: list[dict] = []

    def load(self, force: bool = False):
        return {"loaded": True}

    def ask(self, frame, question: str, **kwargs):
        from src.orchestrator.schemas import VisualEvidence

        self.calls.append({"question": question, "frame_id": getattr(frame, "tag", None)})
        if self.fail:
            return VisualEvidence(
                frame_id=getattr(frame, "tag", "unknown"),
                question=question,
                raw_answer="",
                status="failed",
                error="fake vision failure",
            )
        return VisualEvidence(
            source="moondream2",
            frame_id=getattr(frame, "tag", "unknown"),
            question=question,
            raw_answer=self.answer,
            status="completed",
            inference_time_ms=3.5,
            frame_timestamp=getattr(frame, "timestamp", None),
            image_age_s=round(getattr(frame, "age_s", 0.0), 3),
        )

    def status(self):
        return {"loaded": True, "device": "fake", "questions_answered": len(self.calls)}


class FakeSTT:
    def __init__(self, text: str = "", fail: bool = False):
        self.text = text
        self.fail = fail
        self.calls = 0

    def load(self):
        return {"loaded": True}

    def transcribe(self, samples, sample_rate=16000, metrics=None, **kwargs) -> TranscriptionResult:
        self.calls += 1
        if self.fail:
            return TranscriptionResult(text="", error="fake stt failure")
        duration = len(samples) / float(sample_rate or 16000)
        return TranscriptionResult(
            text=self.text,
            language="en",
            language_probability=1.0,
            latency_s=0.01,
            duration_s=duration,
            model="fake",
            device="fake",
            is_empty=not bool(self.text),
        )

    def status(self):
        return {"loaded": True, "model": "fake"}

    def close(self):
        pass


class FakeTTS:
    def __init__(self):
        self.requests: list[str] = []
        self.sample_rate = 24000

    def load(self):
        return {"loaded": True}

    def synthesize(self, text, voice=None, speed=None, **kwargs) -> np.ndarray:
        self.requests.append(text)
        # 0.02 s of audio per character keeps durations realistic but tiny.
        n = max(1, int(len(text) * 0.02 * self.sample_rate))
        return np.zeros(n, dtype=np.float32)

    def status(self):
        return {"loaded": True, "sentences_spoken": len(self.requests)}


class FakePlayer:
    def __init__(self, fail: bool = False):
        self.played: list[tuple[np.ndarray, int]] = []
        self.fail = fail
        self.is_playing = False
        self.pending = 0
        self.stopped = 0
        self.last_error = None

    def play(self, audio, sample_rate, **kwargs) -> bool:
        if self.fail:
            self.last_error = "fake playback failure"
            return False
        self.played.append((np.asarray(audio), int(sample_rate)))
        return True

    def stop(self):
        self.stopped += 1

    def close(self):
        pass

    def status(self):
        return {"playing": self.is_playing, "pending": self.pending,
                "chunks_played": len(self.played), "last_error": self.last_error}


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    """Deterministic settings for every test; no .env surprises."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_dummy_key_not_real")
    monkeypatch.setenv("GROQ_MODEL", "fake-model")
    monkeypatch.setenv("ROUTER_MODE", "llm")
    monkeypatch.setenv("RESPONSE_STRATEGY", "auto")
    monkeypatch.setenv("AURA_OFFLINE", "0")
    monkeypatch.setenv("MEMORY_TURNS", "8")
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def image() -> np.ndarray:
    return _test_image()


@pytest.fixture
def fake_camera(image) -> FakeCamera:
    return FakeCamera(image)


@pytest.fixture
def fake_stt() -> FakeSTT:
    return FakeSTT(text="hello robot")


@pytest.fixture
def fake_tts() -> FakeTTS:
    return FakeTTS()


@pytest.fixture
def fake_player() -> FakePlayer:
    return FakePlayer()


def make_agent(
    plan: Optional[dict] = None,
    answer: str = "Scripted spoken response.",
    camera=None,
    stt=None,
    tts=None,
    player=None,
    router_mode: str = "llm",
    settings: Optional[Settings] = None,
    fail_client: Optional[Exception] = None,
    response_strategy: str = "auto",
    vlm=None,
) -> AIAgent:
    """Build an AIAgent wired entirely to fakes."""
    settings = settings or Settings()
    if response_strategy != "auto":
        settings = settings.with_overrides(**{"orchestrator.response_strategy": response_strategy})
    if router_mode != "llm":
        settings = settings.with_overrides(**{"orchestrator.router_mode": router_mode})

    client = FakeGroqClient(plan=plan, answer=answer, fail=fail_client)
    router = IntentRouter(client=client, mode=settings.orchestrator.router_mode)
    generator = ResponseGenerator(client=client)

    agent = AIAgent(
        settings=settings,
        camera=camera if camera is not None else FakeCamera(),
        stt=stt if stt is not None else FakeSTT(),
        tts=tts if tts is not None else FakeTTS(),
        player=player if player is not None else FakePlayer(),
        router=router,
        response_generator=generator,
    )
    fake_vlm = vlm if vlm is not None else FakeVLM()
    agent.vlm = fake_vlm
    agent.ctx._vlm = fake_vlm
    return agent


@pytest.fixture
def agent() -> AIAgent:
    a = make_agent()
    a.start(warm_models=False)
    yield a
    a.shutdown()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def plan_dict(intent: str, tool: str = "respond_directly", **extra) -> dict:
    visual = intent in {"visual_question", "scene_description", "object_reasoning"}
    base = {
        "intent": intent,
        "requires_camera": visual or extra.get("requires_camera", False),
        "requires_vlm": visual,
        "tool": tool,
        "tool_arguments": extra.pop("tool_arguments", {}),
        "normalized_question": extra.pop("normalized_question", ""),
        "confidence": 0.99,
        "rationale": "test",
    }
    base.update(extra)
    if base["requires_vlm"]:
        base["requires_camera"] = True
    return base


def recording(duration_s: float = 1.0, sample_rate: int = 16000, level: float = 0.1):
    from src.speech.audio_capture import Recording

    n = int(duration_s * sample_rate)
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(n) * level).astype(np.float32)
    return Recording(
        samples=samples,
        sample_rate=sample_rate,
        duration_s=duration_s,
        peak=float(np.max(np.abs(samples))) if n else 0.0,
        rms=float(np.sqrt(np.mean(samples**2))) if n else 0.0,
    )
