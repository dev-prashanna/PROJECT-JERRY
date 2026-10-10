"""Speech unit tests: segmentation, resampling, VAD, empty transcripts."""

from __future__ import annotations

import numpy as np
import pytest

from src.speech.segmentation import split_sentences
from src.speech.vad import EnergyVAD, VADConfig, frame_iterator
from src.speech.whisper_engine import TranscriptionResult, _resample
from src.speech.audio_capture import Recording


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------

class TestSegmentation:
    def test_basic_split(self):
        parts = split_sentences("Hello there. How are you? I am fine!")
        assert parts == ["Hello there.", "How are you?", "I am fine!"]

    def test_abbreviations_not_split(self):
        parts = split_sentences("Dr. Smith arrived at 5 p.m. yesterday.")
        assert len(parts) == 1

    def test_initials_not_split(self):
        parts = split_sentences("Ask J. Smith about it.")
        assert len(parts) == 1

    def test_long_sentence_is_chunked(self):
        text = " ".join(["word"] * 200)
        parts = split_sentences(text, max_chars=120)
        assert all(len(p) <= 120 for p in parts)
        assert "".join(parts).replace(" ", "") == text.replace(" ", "")

    def test_empty_and_whitespace(self):
        assert split_sentences("") == []
        assert split_sentences("   \n  ") == []

    def test_no_terminal_punctuation(self):
        parts = split_sentences("just a fragment without punctuation")
        assert parts == ["just a fragment without punctuation"]


# ---------------------------------------------------------------------------
# audio utilities
# ---------------------------------------------------------------------------

class TestAudioUtils:
    def test_resample_preserves_duration(self):
        sr_src, sr_dst = 44100, 16000
        seconds = 1.0
        tone = np.sin(2 * np.pi * 220 * np.arange(int(sr_src * seconds)) / sr_src).astype(np.float32)
        out = _resample(tone, sr_src, sr_dst)
        assert abs(len(out) / sr_dst - seconds) < 0.02

    def test_resample_identity(self):
        x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
        assert _resample(x, 16000, 16000) is x

    def test_resample_empty(self):
        assert _resample(np.zeros(0, dtype=np.float32), 44100, 16000).size == 0

    def test_recording_empty_detection(self):
        silent = Recording(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000, duration_s=1.0)
        assert silent.is_effectively_empty
        loud = Recording(samples=np.ones(16000, dtype=np.float32) * 0.3,
                         sample_rate=16000, duration_s=1.0, rms=0.3)
        assert not loud.is_effectively_empty


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

class TestVAD:
    def test_silence_is_not_speech(self):
        vad = EnergyVAD(VADConfig(threshold=0.02))
        for _ in range(10):
            d = vad.update(np.zeros(480, dtype=np.float32), 16000)
        assert d["speech"] is False and d["active"] is False

    def test_loud_tone_starts_and_ends(self):
        vad = EnergyVAD(VADConfig(threshold=0.02, min_speech_frames=2, silence_frames_required=3))
        sr = 16000
        tone = (np.sin(2 * np.pi * 440 * np.arange(480) / sr) * 0.5).astype(np.float32)
        silence = np.zeros(480, dtype=np.float32)

        started = ended = False
        for _ in range(4):
            d = vad.update(tone, sr)
            started |= d["started"]
        assert started and vad._active

        for _ in range(5):
            d = vad.update(silence, sr)
            ended |= d["ended"]
        assert ended and not vad._active

    def test_noise_floor_adapts_downward(self):
        vad = EnergyVAD(VADConfig(noise_floor=0.05))
        initial = vad.noise_floor
        for _ in range(50):
            vad.update(np.full(480, 1e-4, dtype=np.float32), 16000)
        assert vad.noise_floor < initial

    def test_frame_iterator_lengths(self):
        frames = list(frame_iterator(np.zeros(16000, dtype=np.float32), 16000, 30))
        assert len(frames) == 34  # 16000 / 480 rounded up
        assert all(f.size > 0 for f in frames)


# ---------------------------------------------------------------------------
# transcription validation
# ---------------------------------------------------------------------------

class TestTranscriptionValidation:
    def test_empty_audio_is_rejected_by_engine_contract(self):
        result = TranscriptionResult(text="", error="no audio")
        assert result.ok is False

    def test_whitespace_transcript_is_rejected(self):
        result = TranscriptionResult(text="   \n ")
        assert result.ok is False and result.is_empty

    def test_normal_transcript_ok(self):
        result = TranscriptionResult(text="hey robot")
        assert result.ok is True and result.is_empty is False


def test_fake_stt_roundtrip():
    from tests.conftest import FakeSTT, recording

    stt = FakeSTT(text="what do you see")
    result = stt.transcribe(recording(duration_s=1.0).samples, 16000)
    assert result.ok
    assert result.text == "what do you see"

    empty = FakeSTT(text="")
    assert empty.transcribe(np.zeros(0, dtype=np.float32), 16000).ok is False
