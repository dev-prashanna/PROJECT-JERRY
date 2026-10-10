"""Kokoro-82M text-to-speech engine (local weights, loaded once).

The engine owns synthesis only; playback lives in
:mod:`src.speech.audio_player`.  Everything runs from memory — no
intermediate MP3/WAV files are created unless the caller asks for one.

Supported interfaces: the installed ``kokoro`` package (``KModel`` /
``KPipeline``) with the local checkpoint directory.
"""

from __future__ import annotations

import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from config.settings import TTSSettings, get_settings
from src.speech.segmentation import split_sentences
from src.utils.exceptions import InferenceError, ModelNotFoundError
from src.utils.logger import get_logger

log = get_logger("kokoro")

ArrayLike = Union[np.ndarray, "Any"]


@dataclass
class SynthesisResult:
    """One chunk of generated speech plus its timing."""

    text: str
    audio: np.ndarray
    sample_rate: int
    synth_ms: float
    index: int = 0

    @property
    def duration_s(self) -> float:
        return float(len(self.audio)) / max(1, self.sample_rate)


@dataclass
class TTSInfo:
    model_path: str
    device: str
    voices: list[str] = field(default_factory=list)
    default_voice: str = ""
    load_seconds: float = 0.0
    sample_rate: int = 24000

    def describe(self) -> str:
        return f"Kokoro-82M · {len(self.voices)} voices · {self.device}"


class KokoroEngine:
    def __init__(self, settings: Optional[TTSSettings] = None):
        self.settings = settings or get_settings().tts
        self._model = None
        self._pipelines: dict[str, Any] = {}
        self._info: Optional[TTSInfo] = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._device = "cpu"
        self._last_error: Optional[str] = None
        self._sentences_spoken = 0

    # -- introspection ---------------------------------------------------
    @property
    def sample_rate(self) -> int:
        return int(self.settings.sample_rate)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def sentences_spoken(self) -> int:
        return self._sentences_spoken

    def info(self) -> Optional[TTSInfo]:
        return self._info

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "device": self._device,
            "voice": self.settings.voice,
            "speed": self.settings.speed,
            "model_path": self.settings.model_path,
            "sentences_spoken": self._sentences_spoken,
            "last_error": self._last_error,
            "describe": self._info.describe() if self._info else "not loaded",
        }

    def list_voices(self) -> list[str]:
        voices_dir = Path(self.settings.model_path) / "voices"
        if not voices_dir.is_dir():
            return []
        return sorted(p.stem for p in voices_dir.glob("*.pt"))

    # -- loading ---------------------------------------------------------
    def load(self) -> TTSInfo:
        with self._load_lock:
            if self._info is not None:
                return self._info
            started = time.perf_counter()
            model_dir = Path(self.settings.model_path)
            weights = model_dir / "kokoro-v1_0.pth"
            if not weights.is_file():
                raise ModelNotFoundError(
                    f"Kokoro weights not found at {weights}",
                    hint="Set TTS_MODEL_PATH to the directory containing kokoro-v1_0.pth.",
                )

            import os

            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            from src.utils.device_manager import resolve_torch_device

            self._device = resolve_torch_device(self.settings.device)
            try:
                from kokoro import KModel

                model = KModel(
                    repo_id="hexgrad/Kokoro-82M",
                    config=str(model_dir / "config.json"),
                    model=str(weights),
                )
                model = model.to(self._device)
                model.eval()
                self._model = model
            except Exception as exc:  # noqa: BLE001
                self._last_error = str(exc)
                raise InferenceError(f"failed to load Kokoro: {exc}", cause=exc) from exc

            info = TTSInfo(
                model_path=str(model_dir),
                device=self._device,
                voices=self.list_voices(),
                default_voice=self.settings.voice,
                load_seconds=time.perf_counter() - started,
                sample_rate=self.sample_rate,
            )
            self._info = info
            log.info("Kokoro ready in %.1fs — %s", info.load_seconds, info.describe())
            return info

    def _pipeline(self, lang_code: str):
        if lang_code in self._pipelines:
            return self._pipelines[lang_code]
        from kokoro import KPipeline

        model_dir = Path(self.settings.model_path)
        engine = self

        class _LocalPipeline(KPipeline):
            def load_single_voice(self, voice: str):
                if voice in self.voices:
                    return self.voices[voice]
                path = Path(voice) if str(voice).endswith(".pt") else model_dir / "voices" / f"{voice}.pt"
                if not path.exists():
                    available = engine.list_voices()
                    raise InferenceError(
                        f"unknown voice {voice!r}",
                        hint=f"available voices: {', '.join(available[:12])}…",
                    )
                import torch

                pack = torch.load(path, weights_only=True)
                self.voices[voice] = pack
                return pack

        self._pipelines[lang_code] = _LocalPipeline(
            lang_code=lang_code, model=self._model, repo_id="hexgrad/Kokoro-82M"
        )
        return self._pipelines[lang_code]

    # -- synthesis -------------------------------------------------------
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        lang_code: Optional[str] = None,
    ) -> np.ndarray:
        """Text → mono float32 waveform at the model's native rate."""
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        voice = voice or self.settings.voice
        speed = speed if speed is not None else self.settings.speed
        lang = lang_code or self.settings.lang_code

        self.load()
        pipeline = self._pipeline(lang)
        chunks: list[np.ndarray] = []
        started = time.perf_counter()
        try:
            with self._infer_lock:
                for result in pipeline(text, voice=voice, speed=speed, split_pattern=None):
                    audio = getattr(result, "audio", None)
                    if audio is None:
                        continue
                    if hasattr(audio, "detach"):
                        audio = audio.detach().cpu().numpy()
                    chunks.append(np.asarray(audio, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            raise InferenceError(f"Kokoro synthesis failed: {exc}", cause=exc) from exc
        elapsed = (time.perf_counter() - started) * 1000
        self._sentences_spoken += 1
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(chunks)
        log.debug("synth %d chars -> %.2fs audio in %.0f ms", len(text), len(audio) / self.sample_rate, elapsed)
        return audio.astype(np.float32, copy=False)

    def synthesize_streaming(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        max_sentence_chars: Optional[int] = None,
    ):
        """Yield :class:`SynthesisResult` per sentence (enables incremental TTS)."""
        budget = max_sentence_chars or self.settings.max_sentence_chars
        sentences = split_sentences(text, max_chars=budget)
        for i, sentence in enumerate(sentences):
            started = time.perf_counter()
            audio = self.synthesize(sentence, voice=voice, speed=speed)
            if audio.size == 0:
                continue
            yield SynthesisResult(
                text=sentence,
                audio=audio,
                sample_rate=self.sample_rate,
                synth_ms=(time.perf_counter() - started) * 1000,
                index=i,
            )

    def duration_s(self, audio: np.ndarray) -> float:
        return float(np.asarray(audio).size) / max(1, self.sample_rate)

    # -- utilities -------------------------------------------------------
    def save_wav(self, audio: np.ndarray, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        pcm = (samples * 32767).astype("<i2")
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(self.sample_rate)
            fh.writeframes(pcm.tobytes())
        return path

    def play(self, audio: np.ndarray, blocking: bool = True) -> bool:
        """Convenience direct playback (used by the verification script)."""
        player = _get_default_player()
        return player.play(audio, self.sample_rate, blocking=blocking)

    def unload(self) -> None:
        with self._load_lock:
            self._model = None
            self._pipelines.clear()
            self._info = None
            try:
                import gc

                gc.collect()
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                pass
            log.info("Kokoro unloaded")


# ---------------------------------------------------------------------------
# Singleton + default player
# ---------------------------------------------------------------------------

_ENGINES: dict[str, KokoroEngine] = {}
_ENGINES_LOCK = threading.Lock()
_DEFAULT_PLAYER: Any = None


def get_tts_engine(settings: Optional[TTSSettings] = None) -> KokoroEngine:
    settings = settings or TTSSettings()
    key = str(settings.model_path)
    with _ENGINES_LOCK:
        engine = _ENGINES.get(key)
        if engine is None:
            engine = KokoroEngine(settings)
            _ENGINES[key] = engine
        return engine


def _get_default_player():
    global _DEFAULT_PLAYER
    if _DEFAULT_PLAYER is None:
        from src.speech.audio_player import AudioPlayer

        _DEFAULT_PLAYER = AudioPlayer()
    return _DEFAULT_PLAYER


def reset_engine() -> None:
    with _ENGINES_LOCK:
        for e in _ENGINES.values():
            e.unload()
        _ENGINES.clear()
