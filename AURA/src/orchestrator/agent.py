"""AURA's multimodal agent: the orchestrator that owns the whole pipeline.

State machine (see :mod:`src.orchestrator.state_machine`)::

    IDLE → LISTENING → TRANSCRIBING → INTERPRETING
         → [CAPTURING → PERCEIVING]           (visual requests only)
         → GENERATING → SPEAKING → IDLE

The cloud LLM only *proposes* a route; this class validates it, executes
only allow-listed tools, and keeps every stage bounded and observable.

Public entry points:

* :meth:`AIAgent.handle_text`     – text in, spoken-ready text out (testable)
* :meth:`AIAgent.handle_recording`– audio in, adds the STT stage
* :meth:`AIAgent.run_voice_turn`  – full listen → speak cycle
* :meth:`AIAgent.status`          – everything the GUI needs to render
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from config.settings import Settings, get_settings
from src.intelligence.conversation_memory import ConversationMemory
from src.intelligence.intent_classifier import IntentRouter, RouteDecision
from src.intelligence.response_generator import ResponseGenerator
from src.orchestrator.schemas import (
    Intent,
    RequestRecord,
    RoutePlan,
    ToolResult,
    VisualEvidence,
    VISUAL_INTENTS,
)
from src.orchestrator.scheduler import Job, PipelineScheduler
from src.orchestrator.state_machine import State, StateMachine
from src.orchestrator.tools import ToolContext, build_registry
from src.utils.exceptions import AURAError, OfflineError, StaleFrameError
from src.utils.logger import get_logger
from src.utils.metrics import REGISTRY, RequestMetrics, StageTimer

log = get_logger("agent")


class AIAgent:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        camera=None,
        stt=None,
        tts=None,
        player=None,
        router: Optional[IntentRouter] = None,
        response_generator: Optional[ResponseGenerator] = None,
    ):
        self.settings = settings or get_settings()
        self.sm = StateMachine()
        self.memory = ConversationMemory(max_turns=self.settings.orchestrator.memory_turns)
        self.camera = camera
        self.stt = stt
        self.tts = tts
        self.player = player

        self.router = router or IntentRouter()
        self.generator = response_generator or ResponseGenerator()
        self.vlm = None  # lazily loaded (GPU)
        self._warm_thread: threading.Thread | None = None

        self.ctx = ToolContext(
            camera=self.camera,
            memory=self.memory,
            tool_timeout=self.settings.orchestrator.tool_timeout_s,
            system_status_fn=self.status,
        )
        self.registry = build_registry(self.ctx)
        self.scheduler = PipelineScheduler(
            handler=self._execute_job,
            queue_size=self.settings.orchestrator.queue_size,
            drop_oldest=True,
        )

        self._last_record: Optional[RequestRecord] = None
        self._last_metrics: Optional[RequestMetrics] = None
        self._lock = threading.RLock()
        self._running = False
        self._speaking = threading.Event()
        self._listeners: dict[str, list[Callable[..., None]]] = {
            "state": [], "transcript": [], "route": [], "response": [],
            "error": [], "log": [],
        }
        self._offline_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    def on(self, event: str, callback: Callable[..., None]) -> None:
        self._listeners.setdefault(event, []).append(callback)

    def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        for cb in self._listeners.get(event, ()):
            try:
                cb(*args, **kwargs)
            except Exception:  # pragma: no cover - listeners must not break the pipeline
                log.exception("listener for %s failed", event)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self, warm_models: bool = True) -> None:
        if self._running:
            return
        self._running = True
        self.scheduler.start()
        if self.sm.state != State.IDLE:
            self.sm.try_transition(State.IDLE, "agent started")
        else:
            self._emit("state", self.sm.snapshot())
        self.sm.on_transition(lambda tr: self._emit("state", tr))

        if self.camera is not None and self.settings.camera.enabled:
            try:
                self.camera.start()
            except AURAError as exc:
                log.warning("camera failed to start: %s", exc)
                self._emit("error", str(exc))

        if warm_models:
            self._warm_thread = threading.Thread(
                target=self._warm_models, name="aura-warmup", daemon=True
            )
            self._warm_thread.start()
        log.info("agent started (mode=%s, router=%s)",
                 self.mode, self.router.mode)

    def _warm_models(self) -> None:
        """Load local models once, off the critical path."""
        try:
            if self.stt is not None:
                self.stt.load()
                log.info("STT ready")
        except Exception as exc:  # noqa: BLE001
            log.warning("STT warmup failed: %s", exc)
        try:
            if self.tts is not None:
                self.tts.load()
                log.info("TTS ready")
        except Exception as exc:  # noqa: BLE001
            log.warning("TTS warmup failed: %s", exc)
        try:
            self.vlm = self._get_vlm()
            self.ctx._vlm = self.vlm
            self.vlm.load()
            log.info("VLM ready")
        except Exception as exc:  # noqa: BLE001
            log.warning("VLM warmup failed: %s", exc)

    def shutdown(self, timeout_s: float = 5.0) -> None:
        if not self._running:
            return
        self._running = False
        log.info("agent shutting down")
        if self._warm_thread is not None and self._warm_thread.is_alive():
            self._warm_thread.join(timeout=timeout_s)
        self._warm_thread = None
        try:
            self.stop_speaking()
        except Exception:  # pragma: no cover
            pass
        self.scheduler.shutdown(timeout_s=timeout_s)
        if self.camera is not None:
            try:
                from src.perception.depth_service import stop_depth_service

                stop_depth_service(self.camera)
            except Exception:  # pragma: no cover - depth is optional
                pass
            try:
                self.camera.close()
            except Exception:  # pragma: no cover
                log.exception("camera close failed")
        if self.stt is not None:
            try:
                self.stt.close()
            except Exception:  # pragma: no cover
                pass
        if self.vlm is not None:
            try:
                self.vlm.unload()
            except Exception:  # pragma: no cover
                pass
            self.vlm = None
        try:
            self.sm.try_transition(State.SHUTDOWN, "shutdown")
        except Exception:  # pragma: no cover
            pass
        log.info("agent stopped")

    def __enter__(self) -> "AIAgent":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # mode / status
    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        if self.settings.orchestrator.offline or not self.settings.llm.available:
            return "offline"
        return "online"

    @property
    def offline(self) -> bool:
        return self.mode == "offline"

    def status(self) -> dict[str, Any]:
        from src.utils.device_manager import gpu_snapshot

        gpu = gpu_snapshot()
        return {
            "state": self.sm.state.value,
            "mode": self.mode,
            "offline_reason": self._offline_reason,
            "router": self.router.mode,
            "response_strategy": self.settings.orchestrator.response_strategy,
            "queue": self.scheduler.status(),
            "camera": self.camera.info() if self.camera else {"status": "not configured"},
            "stt": self.stt.status() if self.stt else {"status": "not configured"},
            "tts": self.tts.status() if self.tts else {"status": "not configured"},
            "vlm": self.vlm.status() if self.vlm else {"status": "not loaded"},
            "player": self.player.status() if self.player else {"status": "not configured"},
            "gpu": gpu.get("nvsmi", {}),
            "vram": gpu.get("cuda", {}),
            "llm": {
                "model": self.settings.llm.model,
                "fallback": self.settings.llm.fallback_model,
                "configured": self.settings.llm.available,
            },
            "memory_turns": len(self.memory),
            "last_error": self._last_record.error if self._last_record else None,
            "last_response": self.memory.last_response,
        }

    # ------------------------------------------------------------------
    # voice pipeline
    # ------------------------------------------------------------------
    def start_listening(self) -> None:
        if self.sm.state == State.SPEAKING:
            self.stop_speaking()
        if self.sm.state == State.IDLE:
            self.sm.transition(State.LISTENING, "user interaction started")
        self._emit("state", self.sm.history[-1])

    def listen_once(self, seconds: Optional[float] = None):
        """Push-to-talk: record one utterance and return the Recording."""
        from src.speech.audio_capture import AudioCapture

        if self.stt is None:
            raise AURAError("no STT engine configured")
        self.start_listening()
        capture = AudioCapture(self.settings.stt)
        try:
            capture.open()
            if seconds:
                rec = capture.record_for(seconds)
            else:
                rec = capture.record_utterance()
        finally:
            capture.close()
        self.sm.try_transition(State.TRANSCRIBING, "utterance captured")
        self._emit("state", self.sm.history[-1])
        return rec

    def handle_recording(self, recording, metrics: Optional[RequestMetrics] = None, speak: bool = True) -> RequestRecord:
        """Audio → text → full text pipeline."""
        if self.stt is None:
            raise AURAError("no STT engine configured")
        metrics = metrics or REGISTRY.new(f"rec_{int(time.time() * 1000)}")
        if self.sm.state in {State.IDLE, State.LISTENING}:
            self.sm.try_transition(State.TRANSCRIBING, "processing recording")
            self._emit("state", self.sm.history[-1])
        with StageTimer(metrics, "whisper_transcribe"):
            result = self.stt.transcribe(recording.samples, recording.sample_rate, metrics=None)
        metrics.stages["whisper_transcribe"] = metrics.stages.get("whisper_transcribe", 0.0)
        metrics.flags["audio_duration_s"] = round(recording.duration_s, 3)
        self._emit("transcript", result.text, result)
        log.info("transcript: %r (%.0f ms)", result.text, result.latency_s * 1000)
        metrics.flags["transcript"] = result.text

        if not result.ok:
            log.info("ignoring empty transcript")
            self.sm.try_transition(State.IDLE, "empty transcript")
            return RequestRecord(transcript="", error="empty transcript")

        return self.handle_text(
            result.text,
            metrics=metrics,
            speech_started_at=recording.started_at,
            speech_ended_at=recording.started_at + recording.duration_s,
            speak=speak,
        )

    def handle_text(
        self,
        transcript: str,
        metrics: Optional[RequestMetrics] = None,
        speech_ended_at: Optional[float] = None,
        speech_started_at: Optional[float] = None,
        speak: bool = True,
    ) -> RequestRecord:
        """Text command → routed, grounded, (optionally) spoken response."""
        metrics = metrics or REGISTRY.new(f"txt_{int(time.time() * 1000)}")
        metrics.flags["speech_ended_at"] = speech_ended_at
        record = RequestRecord(transcript=transcript, mode=self.mode)
        metrics.start("route")

        try:
            # -- routing ------------------------------------------------
            if self.sm.state in {State.IDLE, State.TRANSCRIBING, State.LISTENING}:
                self.sm.try_transition(State.INTERPRETING, "routing")
            decision = self._route(transcript, metrics)
            record.plan = decision.plan
            self._emit("route", decision)
            metrics.stop("route")
            metrics.flags.update(
                {
                    "intent": decision.plan.intent.value,
                    "router": decision.router,
                    "tool": decision.plan.tool,
                    "route_confidence": decision.plan.confidence,
                    "requires_camera": decision.plan.requires_camera,
                }
            )

            # -- tool execution (visual path) --------------------------
            if decision.plan.is_visual:
                self._run_visual_tools(record, decision.plan, metrics)
            elif decision.plan.tool not in {"respond_directly", "analyze_camera", "describe_scene"}:
                self._run_tool(record, decision.plan, metrics)

            if decision.plan.tool == "repeat_last_response" and self.memory.last_response:
                record.response = self.memory.last_response
                record.response_strategy = "repeat"
                metrics.flags["intent"] = Intent.REPEAT_RESPONSE.value

            # -- response ----------------------------------------------
            if not record.response:
                if self.sm.state in {State.INTERPRETING, State.CAPTURING, State.PERCEIVING}:
                    self.sm.transition(State.GENERATING, "compose response")
                with StageTimer(metrics, "generate_response"):
                    resp = self.generator.generate(
                        record,
                        history=self.memory.chat_messages()[-8:],
                        strategy=self.settings.orchestrator.response_strategy,
                    )
                metrics.flags["response_strategy"] = resp.strategy
                metrics.flags["response_model"] = resp.model
                metrics.flags["response_audit"] = resp.audit
                if resp.prompt_tokens:
                    metrics.flags["llm_prompt_tokens"] = resp.prompt_tokens
                    metrics.flags["llm_completion_tokens"] = resp.completion_tokens
                record.response = resp.text
                if resp.error:
                    record.error = resp.error

            # -- speech ------------------------------------------------
            if speak and record.response:
                self._speak(record, metrics, speech_ended_at)

            self._finish_turn(record, metrics)
            return record

        except AURAError as exc:
            metrics.error = str(exc)
            log.warning("request failed: %s", exc)
            self.sm.try_transition(State.ERROR, str(exc))
            record.error = str(exc)
            record.response = _friendly_error(exc)
            self._emit("error", record.response)
            if speak and record.response:
                try:
                    self._speak(record, metrics, speech_ended_at)
                except Exception:  # pragma: no cover
                    log.exception("speaking the error message failed")
            self._finish_turn(record, metrics)
            return record
        except Exception as exc:  # noqa: BLE001 - last line of defence
            metrics.error = str(exc)
            log.exception("unexpected pipeline failure")
            self.sm.try_transition(State.ERROR, "unexpected failure")
            record.error = str(exc)
            record.response = "Sorry, something went wrong inside the pipeline."
            self._emit("error", record.response)
            self._finish_turn(record, metrics)
            return record
        finally:
            metrics.stop("route")

    # ------------------------------------------------------------------
    # stages
    # ------------------------------------------------------------------
    def _route(self, transcript: str, metrics: RequestMetrics) -> RouteDecision:
        if self.offline and self.router.mode == "llm":
            # Cloud is not usable; force the local router.
            decision = self.router.deterministic.route(transcript, self.memory.context_summary())
            if decision.error:
                decision.plan = RoutePlan.model_validate(
                    {
                        "intent": Intent.GENERAL_QUESTION,
                        "tool": "respond_directly",
                        "confidence": 0.3,
                        "rationale": "offline fallback",
                        "needs_offline_notice": True,
                    }
                )
            decision.plan.needs_offline_notice = True
            decision.router = "deterministic(offline)"
            self._offline_reason = "no usable cloud LLM"
            return decision

        decision = self.router.route(
            transcript,
            history_context=self.memory.context_summary(),
            history=self.memory.chat_messages(),
        )
        if decision.error:
            log.warning("routing fell back: %s", decision.error)
        return decision

    def _run_visual_tools(self, record: RequestRecord, plan: RoutePlan, metrics: RequestMetrics) -> None:
        """CAPTURING → PERCEIVING with a stable, id-tagged snapshot."""
        assert self.camera is not None, "camera not configured"
        question = (
            plan.tool_arguments.get("question")
            or plan.normalized_question
            or "Describe everything visible in front of the camera."
        )

        # --- CAPTURING
        self.sm.transition(State.CAPTURING, "snapshot")
        with StageTimer(metrics, "camera_snapshot"):
            frame = self.camera.snapshot(wait_s=2.5)
        metrics.flags["frame_id"] = frame.tag
        metrics.flags["frame_age_s"] = round(frame.age_s, 3)
        self.ctx.pending_snapshot = frame

        # --- PERCEIVING
        self.sm.transition(State.PERCEIVING, "vlm inference")
        capture_mode = plan.tool_arguments.get("capture_mode", "fresh")
        result = self.registry.execute(
            plan.tool if plan.tool in {"analyze_camera", "describe_scene"} else "analyze_camera",
            {"question": question, "capture_mode": capture_mode},
            default_timeout_s=self.settings.orchestrator.tool_timeout_s,
        )
        record.tool_results.append(result)
        evidence = _evidence_from_result(result)
        if evidence is not None:
            record.evidence.append(evidence)
            metrics.flags["vlm_ms"] = evidence.inference_time_ms
        elif not result.ok:
            record.error = result.error
        self._attach_depth(record)

    def _attach_depth(self, record: RequestRecord) -> None:
        """Best-effort monocular depth summary for near/far reasoning."""
        try:
            if not getattr(self.settings, "depth", None) or not self.settings.depth.enabled:
                return
            from src.perception.depth_service import get_depth_service

            svc = get_depth_service(self.camera)
            if not svc.running:
                svc.start()
            if svc.latest() is None:
                svc.wait_until_ready(timeout_s=5.0)
            summary = svc.summarize_for_vlm()
            if summary:
                record.metadata["depth"] = summary
        except Exception as exc:  # noqa: BLE001 - depth is optional enrichment
            log.debug("depth summary skipped: %s", exc)

    def _run_tool(self, record: RequestRecord, plan: RoutePlan, metrics: RequestMetrics) -> None:
        tool = plan.tool
        if not self.registry.get(tool):
            log.info("tool %s not registered; falling back to direct response", tool)
            return
        if not self.registry.authorize(tool, plan.requires_camera):
            record.tool_results.append(
                ToolResult.failure(tool, "tool not authorised for this plan")
            )
            return
        started = time.perf_counter()
        with StageTimer(metrics, f"tool:{tool}"):
            result = self.registry.execute(tool, plan.tool_arguments)
        metrics.stop(f"tool:{tool}")
        record.tool_results.append(result)
        log.info("tool %s -> ok=%s in %.2fs", tool, result.ok, time.perf_counter() - started)

        # Some tools carry the answer themselves.
        if tool == "repeat_last_response" and result.ok and result.payload.get("available"):
            record.response = str(result.payload.get("response", ""))
        if tool == "explain_limitation" and result.ok:
            record.response = result.text

    def _speak(self, record: RequestRecord, metrics: RequestMetrics, speech_ended_at: Optional[float]) -> None:
        if not self.tts or not self.player or not record.response:
            return
        if not self.sm.try_transition(State.SPEAKING, "tts"):
            log.warning("could not enter SPEAKING from %s", self.sm.state.value)
        self._speaking.set()
        try:
            with StageTimer(metrics, "kokoro_synth"):
                audio = self.tts.synthesize(record.response)
            if audio.size == 0:
                log.warning("TTS produced no audio")
                return
            started = time.perf_counter()
            ok = self.player.play(audio, self.tts.sample_rate, tag=record.request_id)
            if not ok:
                log.warning("audio chunk rejected by the player queue")
                return
            if speech_ended_at:
                metrics.stages["time_to_first_audio"] = time.time() - speech_ended_at
            metrics.flags["speech_duration_s"] = round(audio.size / self.tts.sample_rate, 2)
            metrics.stages["playback_overhead"] = time.perf_counter() - started
        finally:
            # Half-duplex: we stay in SPEAKING until the buffer drains.
            self._wait_for_playback()
            self._speaking.clear()

    def _wait_for_playback(self, timeout_s: float = 60.0) -> None:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if not self.player.is_playing and self.player.pending == 0:
                return
            time.sleep(0.05)

    def _finish_turn(self, record: RequestRecord, metrics: RequestMetrics) -> None:
        if record.transcript and record.response:
            self.memory.record(
                user_transcript=record.transcript,
                final_response=record.response,
                intent=record.plan.intent.value if record.plan else None,
                frame_id=metrics.flags.get("frame_id"),
                visual_observation=record.evidence_text or None,
                route_tool=record.plan.tool if record.plan else None,
                error=record.error,
            )
        self._last_record = record
        self._last_metrics = metrics
        metrics.stages["total"] = metrics.total_s
        self._emit("response", record)
        self.sm.try_transition(State.IDLE, "turn complete")

    # ------------------------------------------------------------------
    # interruption
    # ------------------------------------------------------------------
    def stop_speaking(self) -> None:
        """Barge-in: cancel speech and return to listening."""
        if self.player:
            self.player.stop()
        self._speaking.clear()
        if self.scheduler.busy:
            self.scheduler.cancel_current()
        if self.sm.state == State.SPEAKING:
            self.sm.try_transition(State.IDLE, "interrupted")
        if self.sm.history:
            self._emit("state", self.sm.history[-1])

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set() or bool(self.player and self.player.is_playing)

    @property
    def is_busy(self) -> bool:
        return self.sm.busy or self.scheduler.busy

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _get_vlm(self):
        if self.vlm is None:
            from src.perception.vlm.registry import create_vlm

            self.vlm = create_vlm(self.settings.vlm)
            self.ctx._vlm = self.vlm
        return self.vlm

    def submit_text(self, transcript: str) -> Job:
        """Queue a text request (bounded); returns the Job for waiting."""
        self.scheduler.start()
        return self.scheduler.submit(transcript)

    def _execute_job(self, job: Job) -> RequestRecord:
        """Scheduler callback: run one queued text request."""
        if isinstance(job.payload, str):
            return self.handle_text(job.payload, speak=True)
        if isinstance(job.payload, dict):
            return self.handle_text(**job.payload)
        # Recordings produced elsewhere are passed pre-transcribed.
        payload = job.payload
        if hasattr(payload, "samples"):
            return self.handle_recording(payload)
        return RequestRecord(transcript=str(payload), error="unsupported payload")

    @property
    def last_record(self) -> Optional[RequestRecord]:
        return self._last_record

    @property
    def last_metrics(self) -> Optional[RequestMetrics]:
        return self._last_metrics


# ---------------------------------------------------------------------------
# module helpers
# ---------------------------------------------------------------------------

def _evidence_from_result(result: ToolResult) -> Optional[VisualEvidence]:
    raw = (result.payload or {}).get("evidence")
    if not raw:
        return None
    try:
        return VisualEvidence.model_validate(raw)
    except Exception:  # noqa: BLE001
        return None


def _friendly_error(exc: AURAError) -> str:
    text = str(exc)
    lowered = text.lower()
    if "rate limit" in lowered:
        return "I am being rate limited by the language service right now. Please try again in a moment."
    if "api key" in lowered:
        return "My cloud language model is not configured, so I can only handle simple requests."
    if "camera" in lowered:
        return "I cannot use the camera right now, so I cannot answer visual questions."
    if "microphone" in lowered:
        return "I cannot hear you: the microphone is unavailable."
    if "memory" in lowered or "out of memory" in lowered:
        return "I ran out of graphics memory. Please try again in a moment."
    return "I could not complete that request. Please try again."
