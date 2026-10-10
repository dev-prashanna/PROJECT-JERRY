"""Factory that wires a fully configured :class:`AIAgent` from settings.

Used by ``main.py``, the GUI, the benchmark runner and the tests so that
exactly one wiring path exists.
"""

from __future__ import annotations

from typing import Optional

from config.settings import Settings, get_settings
from src.orchestrator.agent import AIAgent
from src.utils.logger import get_logger, setup_logging

log = get_logger("bootstrap")


def build_agent(
    settings: Optional[Settings] = None,
    *,
    with_camera: bool = True,
    with_stt: bool = True,
    with_tts: bool = True,
    warm: bool = False,
    speak: bool = True,
) -> AIAgent:
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    camera = None
    if with_camera and settings.camera.enabled:
        from src.perception.camera_manager import CameraManager

        camera = CameraManager(settings.camera)

    stt = None
    if with_stt:
        from src.speech.whisper_engine import get_stt_engine

        stt = get_stt_engine(settings.stt)

    tts = None
    player = None
    if with_tts:
        from src.speech.audio_player import AudioPlayer
        from src.speech.kokoro_engine import get_tts_engine

        tts = get_tts_engine(settings.tts)
        player = AudioPlayer(
            device=None if settings.tts.output_device < 0 else settings.tts.output_device
        )

    agent = AIAgent(
        settings=settings,
        camera=camera,
        stt=stt,
        tts=tts,
        player=player,
    )

    if warm:
        agent.start(warm_models=True)
    return agent
