"""Voice activity detection.

Default detector: an adaptive energy (RMS) detector with a noise floor that
tracks quiet periods.  It needs no model download, has microsecond cost per
frame, and is good enough for utterance endpointing.

If ``webrtcvad`` happens to be installed it is used as an additional
confirming detector (a frame must pass both).  Optionally, faster-whisper's
built-in Silero VAD filters the audio again during transcription.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

log = get_logger("vad")


@dataclass
class VADConfig:
    threshold: float = 0.012        # RMS above which a frame counts as speech
    noise_floor: float = 0.004      # initial noise floor estimate
    adapt_rate: float = 0.05        # how fast the noise floor follows silence
    silence_frames_required: int = 8
    min_speech_frames: int = 3
    frame_ms: int = 30


class EnergyVAD:
    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self.noise_floor = self.config.noise_floor
        self._speech_run = 0
        self._silence_run = 0
        self._active = False
        self._webrtc = None
        self._init_webrtc()

    def _init_webrtc(self) -> None:
        try:
            import webrtcvad  # type: ignore

            self._webrtc = webrtcvad.Vad(2)
        except Exception:
            self._webrtc = None

    def reset(self) -> None:
        self._speech_run = 0
        self._silence_run = 0
        self._active = False

    @staticmethod
    def rms(frame: np.ndarray) -> float:
        if frame.size == 0:
            return 0.0
        x = frame.astype(np.float32, copy=False)
        return float(np.sqrt(np.mean(np.square(x))))

    def frame_is_speech(self, frame: np.ndarray, sample_rate: int = 16000) -> bool:
        level = self.rms(frame)
        speech = level >= max(self.config.threshold, self.noise_floor * 3.0)

        # Adapt the noise floor only while we believe nothing is speaking.
        if not speech:
            self.noise_floor = (
                (1 - self.config.adapt_rate) * self.noise_floor
                + self.config.adapt_rate * level
            )
        elif level < self.noise_floor * 8:
            # quiet speech: let the floor rise slowly so it stays relevant
            self.noise_floor = (1 - self.config.adapt_rate / 4) * self.noise_floor + (
                self.config.adapt_rate / 4
            ) * level

        if speech and self._webrtc is not None:
            try:
                pcm16 = _to_pcm16(frame, sample_rate)
                speech = bool(self._webrtc.is_speech(pcm16, sample_rate))
            except Exception:  # pragma: no cover - webrtcvad wants 10/20/30 ms frames
                speech = True
        return speech

    def update(self, frame: np.ndarray, sample_rate: int = 16000) -> dict:
        """Feed one frame and return the endpointing decision."""
        speech = self.frame_is_speech(frame, sample_rate)
        if speech:
            self._speech_run += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            self._speech_run = 0

        started = False
        if not self._active and self._speech_run >= self.config.min_speech_frames:
            self._active = True
            started = True

        ended = False
        if self._active and self._silence_run >= self.config.silence_frames_required:
            self._active = False
            ended = True
            self._speech_run = 0

        return {
            "speech": speech,
            "active": self._active,
            "started": started,
            "ended": ended,
            "level": self.rms(frame),
            "noise_floor": self.noise_floor,
        }


def _to_pcm16(frame: np.ndarray, sample_rate: int) -> bytes:
    x = np.clip(frame, -1.0, 1.0)
    return (x * 32767).astype("<i2").tobytes()


def frame_iterator(samples: np.ndarray, sample_rate: int, frame_ms: int = 30):
    """Yield overlapping-free frames of ``frame_ms`` milliseconds."""
    n = max(1, int(sample_rate * frame_ms / 1000))
    for start in range(0, len(samples), n):
        chunk = samples[start : start + n]
        if chunk.size:
            yield chunk
