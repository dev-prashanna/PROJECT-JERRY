"""Microphone capture: push-to-talk and VAD-endpointed utterances.

Pipeline::

    microphone → bounded frame buffer → (optional) VAD endpointing
              → mono float32 → 16 kHz resample → Whisper

Guarantees
----------
* Recording is always bounded (``MAX_UTTERANCE_SECONDS``); the buffer never
  grows without limit.
* Empty/near-silent utterances are rejected before they reach the STT model.
* Device failures are surfaced as :class:`MicrophoneError` instead of
  crashing the pipeline.
* Captured audio lives in memory only: nothing is written to disk unless a
  caller explicitly asks for it (privacy).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from config.settings import STTSettings, get_settings
from src.speech.vad import EnergyVAD, VADConfig
from src.utils.exceptions import MicrophoneError
from src.utils.logger import get_logger

log = get_logger("capture")

INPUT_KIND = "input"


@dataclass
class Recording:
    samples: np.ndarray                 # mono float32 at `sample_rate`
    sample_rate: int
    duration_s: float
    peak: float = 0.0
    rms: float = 0.0
    truncated: bool = False
    device: str = ""
    started_at: float = field(default_factory=time.time)

    @property
    def is_effectively_empty(self) -> bool:
        return self.samples.size == 0 or self.rms < 1e-4


class AudioCapture:
    def __init__(self, settings: Optional[STTSettings] = None, vad: Optional[EnergyVAD] = None):
        self.settings = settings or get_settings().stt
        self.vad = vad or EnergyVAD(
            VADConfig(threshold=max(0.006, self.settings.vad_threshold * 0.02))
        )
        self._stream = None
        self._device_index: Optional[int] = None
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._recordings = 0

    # -- device management ----------------------------------------------
    @staticmethod
    def list_devices() -> list[dict]:
        from src.utils.device_manager import audio_devices

        inputs, _ = audio_devices()
        return [d.__dict__ for d in inputs]

    def resolve_device(self) -> int:
        from src.utils.device_manager import resolve_audio_device

        index = resolve_audio_device(self.settings.microphone_device, INPUT_KIND)
        if index is None:
            raise MicrophoneError(
                f"microphone {self.settings.microphone_device!r} not found",
                hint="Run scripts/inspect_environment.py to list input devices.",
            )
        return index

    def open(self) -> None:
        import sounddevice as sd

        with self._lock:
            if self._stream is not None:
                return
            index = self.resolve_device()
            try:
                self._stream = sd.InputStream(
                    device=index,
                    channels=1,
                    samplerate=int(self.settings.sample_rate),
                    dtype="float32",
                    blocksize=0,
                )
                self._stream.start()
                self._device_index = index
                self._last_error = None
            except Exception as exc:  # noqa: BLE001
                self._stream = None
                self._last_error = str(exc)
                raise MicrophoneError(
                    f"could not open microphone: {exc}",
                    hint="Another application may be using the device.",
                ) from exc
        log.info("microphone opened (device=%s, %s Hz)", self._device_index, self.settings.sample_rate)

    def close(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # pragma: no cover
                pass
        log.info("microphone closed")

    def __enter__(self) -> "AudioCapture":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reading ---------------------------------------------------------
    def _read_blocks(self, max_seconds: float, on_block: Optional[Callable[[np.ndarray], None]] = None):
        """Read blocks until ``max_seconds``; yields concatenated samples."""
        stream = self._stream
        if stream is None:
            raise MicrophoneError("microphone is not open", hint="Call open() first.")
        sr = int(self.settings.sample_rate)
        chunks: list[np.ndarray] = []
        deadline = time.perf_counter() + max_seconds
        try:
            while time.perf_counter() < deadline:
                remaining = deadline - time.perf_counter()
                block, _overflowed = stream.read(min(sr // 10, max(1, int(sr * remaining))))
                arr = np.asarray(block, dtype=np.float32).reshape(-1)
                if arr.size:
                    chunks.append(arr)
                    if on_block:
                        on_block(arr)
        except Exception as exc:  # noqa: BLE001 - device unplugged etc.
            self._last_error = str(exc)
            raise MicrophoneError(f"microphone read failed: {exc}") from exc
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    # -- public recording APIs ------------------------------------------
    def record_for(self, seconds: float) -> Recording:
        """Push-to-talk: record a fixed duration (bounded by the config)."""
        seconds = float(min(max(0.1, seconds), self.settings.max_utterance_s))
        self.vad.reset()
        samples = self._read_blocks(seconds)
        return self._to_recording(samples, truncated=False)

    def record_utterance(
        self,
        max_seconds: Optional[float] = None,
        silence_timeout_s: Optional[float] = None,
        leading_padding_s: float = 0.25,
        on_level: Optional[Callable[[float], None]] = None,
    ) -> Recording:
        """Record until ``silence_timeout_s`` of quiet follows speech.

        A short leading buffer is kept so the first word is not clipped.
        """
        sr = int(self.settings.sample_rate)
        limit = float(max_seconds or self.settings.max_utterance_s)
        silence_limit = float(
            silence_timeout_s if silence_timeout_s is not None else self.settings.silence_timeout_s
        )
        frame_len = max(1, int(sr * 0.03))

        pre_roll: list[np.ndarray] = []
        pre_roll_samples = int(sr * leading_padding_s)
        collected: list[np.ndarray] = []
        collected_len = 0

        stream = self._stream
        if stream is None:
            raise MicrophoneError("microphone is not open")

        self.vad.reset()
        started = time.perf_counter()
        speech_seen = False
        speech_frames = 0
        silence_frames = 0
        silence_needed = max(1, int(silence_limit / 0.03))
        truncated = False
        pending: np.ndarray = np.zeros(0, dtype=np.float32)

        try:
            while True:
                elapsed = time.perf_counter() - started
                if elapsed >= limit:
                    truncated = True
                    break
                block, _ = stream.read(frame_len)
                frame = np.asarray(block, dtype=np.float32).reshape(-1)
                if frame.size == 0:
                    continue
                pending = np.concatenate([pending, frame]) if pending.size else frame

                while pending.size >= frame_len:
                    piece, pending = pending[:frame_len], pending[frame_len:]
                    decision = self.vad.update(piece, sr)
                    if on_level:
                        on_level(decision["level"])

                    if not speech_seen:
                        pre_roll.append(piece)
                        pre_roll_samples -= piece.size
                        if pre_roll_samples <= 0 and len(pre_roll) > 1:
                            pre_roll.pop(0)
                        if decision["active"]:
                            speech_seen = True
                            collected.extend(pre_roll)
                            collected_len = sum(c.size for c in collected)
                            pre_roll.clear()
                            silence_frames = 0
                        continue

                    collected.append(piece)
                    collected_len += piece.size
                    if decision["speech"]:
                        silence_frames = 0
                        speech_frames += 1
                    else:
                        silence_frames += 1
                        if silence_frames >= silence_needed and speech_frames > 0:
                            break
                else:
                    continue
                break
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            raise MicrophoneError(f"microphone read failed: {exc}") from exc

        if not speech_seen or collected_len == 0:
            samples = np.zeros(0, dtype=np.float32)
        else:
            samples = np.concatenate(collected)
            # trim trailing silence for faster transcription
            keep = _trim_silence(samples, sr)
            samples = keep

        return self._to_recording(samples, truncated=truncated)

    # -- helpers ---------------------------------------------------------
    def _to_recording(self, samples: np.ndarray, truncated: bool) -> Recording:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        self._recordings += 1
        return Recording(
            samples=samples,
            sample_rate=int(self.settings.sample_rate),
            duration_s=samples.size / max(1, self.settings.sample_rate),
            peak=peak,
            rms=rms,
            truncated=truncated,
            device=str(self._device_index),
        )

    # -- status ----------------------------------------------------------
    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def status(self) -> dict:
        return {
            "open": self.is_open,
            "device_index": self._device_index,
            "sample_rate": self.settings.sample_rate,
            "recordings": self._recordings,
            "last_error": self._last_error,
            "vad_enabled": self.settings.vad_enabled,
        }


def _trim_silence(samples: np.ndarray, sr: int, keep_tail_s: float = 0.25) -> np.ndarray:
    if samples.size == 0:
        return samples
    frame = max(1, int(sr * 0.02))
    n = samples.size // frame
    if n == 0:
        return samples
    levels = np.sqrt(np.mean(samples[: n * frame].reshape(n, frame) ** 2, axis=1))
    threshold = max(1e-4, float(levels.max()) * 0.05)
    voiced = np.where(levels > threshold)[0]
    if voiced.size == 0:
        return samples
    start = max(0, int(voiced[0]) * frame - int(sr * 0.1))
    end = min(samples.size, (int(voiced[-1]) + 1) * frame + int(sr * keep_tail_s))
    return samples[start:end]
