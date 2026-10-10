"""TTS tests: streaming synthesis, playback queue, interruption."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from src.speech.audio_player import AudioPlayer, PlaybackJob
from src.speech.kokoro_engine import SynthesisResult
from tests.conftest import FakeTTS


# ---------------------------------------------------------------------------
# player
# ---------------------------------------------------------------------------

class TestAudioPlayer:
    @pytest.fixture
    def player(self, monkeypatch):
        p = AudioPlayer()
        played: list[str] = []

        def fake_play(job):
            time.sleep(0.05)
            played.append(job.tag)

        monkeypatch.setattr(p, "_play_job", fake_play)
        p._played_tags = played
        return p

    def test_play_is_non_blocking(self, player):
        audio = np.zeros(2400, dtype=np.float32)
        start = time.perf_counter()
        assert player.play(audio, 24000, tag="a") is True
        assert time.perf_counter() - start < 0.04  # returns before playback finishes
        player.close()

    def test_chunks_play_in_order(self, player):
        for tag in ("a", "b", "c"):
            player.play(np.zeros(100, dtype=np.float32), 24000, tag=tag)
        deadline = time.time() + 3
        while time.time() < deadline and len(player._played_tags) < 3:
            time.sleep(0.02)
        assert player._played_tags == ["a", "b", "c"]
        player.close()

    def test_stop_cancels_pending_chunks(self, player):
        # fill the queue while the worker is busy
        for i in range(4):
            player.play(np.zeros(100, dtype=np.float32), 24000, tag=f"t{i}")
        player.stop()
        time.sleep(0.3)
        assert len(player._played_tags) < 4
        player.close()

    def test_queue_is_bounded(self, player):
        for i in range(30):
            player.play(np.zeros(10, dtype=np.float32), 24000, tag=str(i))
        assert player.pending <= 4
        player.close()

    def test_status_shape(self, player):
        status = player.status()
        assert {"playing", "pending", "chunks_played", "last_error", "elapsed_s"} <= set(status)
        player.close()

    def test_close_stops_thread(self, player):
        player.play(np.zeros(10, dtype=np.float32), 24000, tag="x")
        player.close()
        assert player._thread is None or not player._thread.is_alive()

    def test_zero_length_audio_is_skipped(self, monkeypatch):
        p = AudioPlayer()
        called = []
        monkeypatch.setattr(p, "_play_job", lambda job: called.append(job.tag))
        p.play(np.zeros(0, dtype=np.float32), 24000, tag="empty")
        time.sleep(0.2)
        p.close()
        assert called == ["empty"]  # handler ran but produced no sound


# ---------------------------------------------------------------------------
# synthesis contract
# ---------------------------------------------------------------------------

class TestTTSContract:
    def test_fake_tts_produces_audio_proportional_to_text(self):
        tts = FakeTTS()
        short = tts.synthesize("Hi.")
        long = tts.synthesize("This is a much longer sentence for the test.")
        assert long.size > short.size > 0
        assert tts.requests == ["Hi.", "This is a much longer sentence for the test."]

    def test_synthesis_result_duration(self):
        res = SynthesisResult(
            text="hello", audio=np.zeros(12000, dtype=np.float32), sample_rate=24000, synth_ms=10
        )
        assert res.duration_s == pytest.approx(0.5)

    def test_engine_status_reports_voice(self):
        from config.settings import TTSSettings

        from src.speech.kokoro_engine import KokoroEngine

        engine = KokoroEngine(TTSSettings())
        status = engine.status()
        assert status["loaded"] is False
        assert "voice" in status and "model_path" in status

    def test_unknown_voice_error_is_actionable(self):
        from config.settings import TTSSettings

        from src.speech.kokoro_engine import KokoroEngine

        engine = KokoroEngine(TTSSettings())
        # list_voices works without loading the network model
        voices = engine.list_voices()
        assert voices, "expected local voice packs"
        assert "af_heart" in voices

    def test_save_wav_roundtrip(self, tmp_path):
        import wave

        from config.settings import TTSSettings

        from src.speech.kokoro_engine import KokoroEngine

        engine = KokoroEngine(TTSSettings())
        audio = np.sin(2 * np.pi * 440 * np.arange(24000) / 24000).astype(np.float32)
        path = engine.save_wav(audio, tmp_path / "out.wav")
        with wave.open(str(path)) as fh:
            assert fh.getframerate() == 24000
            assert fh.getnchannels() == 1
            assert fh.getnframes() == 24000


def test_interrupt_clears_player_and_state():
    from tests.conftest import make_agent

    agent = make_agent()
    agent.start(warm_models=False)
    try:
        agent._speaking.set()
        agent.stop_speaking()
        assert agent.player.stopped >= 1
        assert agent.is_speaking is False
        assert agent.sm.state.value in {"IDLE", "ERROR"}
    finally:
        agent.shutdown()
