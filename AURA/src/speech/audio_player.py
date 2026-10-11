"""Speaker playback worker.

* Non-blocking: synthesis results are queued and played on a dedicated
  thread so the pipeline never blocks on the sound card.
* Bounded queue (2 chunks) — stale speech is dropped rather than piling up.
* ``stop()`` cancels pending chunks immediately (barge-in / interruption).
* ``is_playing`` drives the half-duplex rule: AURA does not listen while it
  is speaking, so its own voice is never mistaken for a user command.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.utils.logger import get_logger

log = get_logger("player")


@dataclass
class PlaybackJob:
    audio: np.ndarray
    sample_rate: int
    tag: str = ""
    on_done: Optional[Any] = None


class AudioPlayer:
    def __init__(self, device: Optional[int] = None, queue_size: int = 4):
        self._device = device if device is not None else -1
        self._queue: "queue.Queue[Optional[PlaybackJob]]" = queue.Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._playing = False
        self._chunks_played = 0
        self._last_error: Optional[str] = None
        self._started_at = 0.0

    # -- lifecycle -------------------------------------------------------
    def _ensure_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._thread = threading.Thread(target=self._run, name="audio-playback", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_requested.is_set():
            try:
                job = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                self._queue.task_done()
                break
            try:
                self._play_job(job)
            except Exception as exc:  # noqa: BLE001
                self._last_error = str(exc)
                log.warning("playback failed: %s", exc)
            finally:
                self._queue.task_done()

    def _play_job(self, job: PlaybackJob) -> None:
        audio = np.asarray(job.audio, dtype=np.float32)
        if audio.size == 0:
            if job.on_done:
                job.on_done(True)
            return
        # Guard against accidental stereo (2,N) / (N,2) shapes.
        if audio.ndim == 2:
            audio = audio.mean(axis=0) if audio.shape[0] <= 8 else audio.mean(axis=1)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak

        self._playing = True
        self._started_at = time.perf_counter()
        ok = False
        try:
            import sounddevice as sd

            with sd.OutputStream(
                samplerate=job.sample_rate,
                channels=1,
                dtype="float32",
                device=None if self._device in (-1, None) else self._device,
            ) as stream:
                # OutputStream opened without `blocksize` is blocking already;
                # write() has no `blocking=` kwarg (that belongs to sd.play()).
                stream.write(audio)
            ok = True
            self._chunks_played += 1
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            log.warning("sounddevice playback failed (%s); falling back to aplay", exc)
            ok = self._play_via_cli(audio, job.sample_rate)
        finally:
            self._playing = False
            if job.on_done:
                try:
                    job.on_done(ok)
                except Exception:  # pragma: no cover
                    pass

    @staticmethod
    def _play_via_cli(audio: np.ndarray, sample_rate: int) -> bool:
        import shutil
        import subprocess
        import tempfile
        from pathlib import Path

        for player, args in (
            ("aplay", ["-q"]),
            ("paplay", []),
            ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
        ):
            if not shutil.which(player):
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                path = Path(fh.name)
            try:
                import wave

                pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
                with wave.open(str(path), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(sample_rate)
                    w.writeframes(pcm.tobytes())
                subprocess.run([player, *args, str(path)], check=False, timeout=60)
                return True
            except Exception as exc:  # noqa: BLE001
                log.debug("%s failed: %s", player, exc)
            finally:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        return False

    # -- public API ------------------------------------------------------
    def play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        blocking: bool = False,
        tag: str = "",
        on_done: Optional[Any] = None,
        timeout_s: float = 120.0,
    ) -> bool:
        """Queue audio for playback; returns False if the queue rejected it."""
        self._ensure_thread()
        job = PlaybackJob(audio=np.asarray(audio, dtype=np.float32), sample_rate=sample_rate, tag=tag, on_done=on_done)
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            # Drop the oldest queued chunk to make room for the newest.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                log.warning("playback queue full; chunk dropped")
                return False
        if blocking:
            deadline = time.perf_counter() + timeout_s
            while self._playing or not self._queue.empty():
                if time.perf_counter() > deadline:
                    return False
                time.sleep(0.02)
        return True

    def stop(self) -> None:
        """Cancel pending and in-flight playback (speech interruption)."""
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                dropped += 1
            except queue.Empty:
                break
        # The blocking `stream.write` can not be aborted mid-buffer; we wait
        # for the current chunk to finish while suppressing new ones.
        deadline = time.perf_counter() + 2.0
        while self._playing and time.perf_counter() < deadline:
            time.sleep(0.02)
        if dropped:
            log.info("playback stopped (%d queued chunk(s) cancelled)", dropped)

    def close(self) -> None:
        self.stop()
        self._stop_requested.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def status(self) -> dict[str, Any]:
        return {
            "playing": self._playing,
            "pending": self.pending,
            "chunks_played": self._chunks_played,
            "last_error": self._last_error,
            "elapsed_s": round(time.perf_counter() - self._started_at, 2) if self._playing else 0.0,
        }
