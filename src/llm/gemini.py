"""Thin wrapper around the Gemini API used to generate questions for the VLM."""

import io
import os
import time
import mimetypes
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
RETRYABLE_CODES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "No Gemini API key found. Add GEMINI_API_KEY or GOOGLE_API_KEY to "
            f"{PROJECT_ROOT / '.env'}"
        )
    return key


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    return genai.Client(api_key=_api_key())


def image_part(image, mime_type: str = "image/jpeg") -> types.Part:
    """Converts a PIL image, file path, or raw bytes into a Gemini part."""
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG")
        return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

    if isinstance(image, (str, os.PathLike)):
        path = Path(image)
        guessed, _ = mimetypes.guess_type(path.name)
        return types.Part.from_bytes(data=path.read_bytes(), mime_type=guessed or mime_type)

    if isinstance(image, (bytes, bytearray)):
        return types.Part.from_bytes(data=bytes(image), mime_type=mime_type)

    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _generate_with_retry(model: str, contents, config, attempts: int = MAX_ATTEMPTS):
    last_error = None
    for attempt in range(attempts):
        try:
            return get_client().models.generate_content(
                model=model, contents=contents, config=config
            )
        except genai_errors.APIError as err:
            if err.code not in RETRYABLE_CODES:
                raise
            last_error = err
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)
    raise last_error


def generate(
    prompt: str,
    *,
    images=None,
    system: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """Sends a prompt (optionally with images and a system instruction) to Gemini."""
    contents = [image_part(image) for image in (images or [])]
    contents.append(prompt)

    config_kwargs = {}
    if system is not None:
        config_kwargs["system_instruction"] = system
    if temperature is not None:
        config_kwargs["temperature"] = temperature

    response = _generate_with_retry(
        model=model or DEFAULT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
    )

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError(f"Empty response from Gemini ({model or DEFAULT_MODEL})")
    return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the Gemini integration")
    parser.add_argument("prompt", nargs="?", default="Say hello in one sentence.")
    parser.add_argument("--image", nargs="*", default=None, help="image path(s) to attach")
    parser.add_argument("--model", default=None)
    parser.add_argument("--system", default=None)
    args = parser.parse_args()

    print(generate(args.prompt, images=args.image, system=args.system, model=args.model))
