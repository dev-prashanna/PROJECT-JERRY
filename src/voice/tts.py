"""Kokoro-82M text-to-speech, loaded from the local models/ directory (offline)."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Union

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "kokoro-82M"
DEFAULT_VOICE = "af_heart"
SAMPLE_RATE = 24000

_model = None
_pipelines = {}


def list_voices() -> list:
    return sorted(p.stem for p in (MODEL_DIR / "voices").glob("*.pt"))


def get_model():
    global _model
    if _model is None:
        from kokoro import KModel

        if not (MODEL_DIR / "kokoro-v1_0.pth").exists():
            raise FileNotFoundError(f"Kokoro weights not found at {MODEL_DIR}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = KModel(
            repo_id="hexgrad/Kokoro-82M",
            config=str(MODEL_DIR / "config.json"),
            model=str(MODEL_DIR / "kokoro-v1_0.pth"),
        ).to(device)
        _model.eval()
    return _model


def get_pipeline(lang_code: str = "a"):
    if lang_code in _pipelines:
        return _pipelines[lang_code]

    from kokoro import KPipeline

    class LocalPipeline(KPipeline):
        def load_single_voice(self, voice):
            if voice in self.voices:
                return self.voices[voice]
            path = (
                Path(voice)
                if voice.endswith(".pt")
                else MODEL_DIR / "voices" / f"{voice}.pt"
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"Unknown voice {voice!r}. Available: {', '.join(list_voices())}"
                )
            pack = torch.load(path, weights_only=True)
            self.voices[voice] = pack
            return pack

    _pipelines[lang_code] = LocalPipeline(
        lang_code=lang_code, model=get_model(), repo_id="hexgrad/Kokoro-82M"
    )
    return _pipelines[lang_code]


def synthesize(
    text: str,
    voice: str = DEFAULT_VOICE,
    speed: float = 1.0,
    lang_code: str = "a",
    split_pattern: Union[str, None] = r"\n+",
) -> torch.Tensor:
    """Text -> mono float waveform at 24 kHz."""
    if not text.strip():
        return torch.zeros(0)
    pipeline = get_pipeline(lang_code)
    chunks = [
        r.audio
        for r in pipeline(text, voice=voice, speed=speed, split_pattern=split_pattern)
        if r.audio is not None
    ]
    if not chunks:
        return torch.zeros(0)
    return torch.cat(chunks)


def save_wav(audio: torch.Tensor, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = (audio.detach().cpu().numpy() * 32767).astype("int16")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(samples.tobytes())
    return path


def speak(
    text: str,
    voice: str = DEFAULT_VOICE,
    speed: float = 1.0,
    lang_code: str = "a",
    out: Union[str, Path, None] = None,
    play: bool = False,
) -> Path:
    """Synthesize text to a WAV file; optionally play it."""
    audio = synthesize(text, voice=voice, speed=speed, lang_code=lang_code)
    path = save_wav(audio, out or PROJECT_ROOT / "outputs" / "speech.wav")
    if play:
        for player in (
            ["aplay", "-q"],
            ["paplay"],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
        ):
            if shutil.which(player[0]):
                subprocess.run(player + [str(path)], check=False)
                break
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kokoro-82M text-to-speech")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("-v", "--voice", default=DEFAULT_VOICE, help="Voice name")
    parser.add_argument("-s", "--speed", type=float, default=1.0)
    parser.add_argument("-l", "--lang", default="a", help="a=US English, b=British English")
    parser.add_argument("-o", "--out", default=str(PROJECT_ROOT / "outputs" / "speech.wav"))
    parser.add_argument("--play", action="store_true", help="Play the result after writing")
    parser.add_argument("--voices", action="store_true", help="List available voices and exit")
    args = parser.parse_args()

    if args.voices:
        print("\n".join(list_voices()))
    elif not args.text:
        parser.error("text is required unless --voices is given")
    else:
        out = speak(
            args.text,
            voice=args.voice,
            speed=args.speed,
            lang_code=args.lang,
            out=args.out,
            play=args.play,
        )
        print(f"wrote {out}")
