import torch
import soundfile as sf
from pathlib import Path
from kokoro import KPipeline

MODEL_DIR = Path("/home/prashanna/Documents/PROJECT VBMD/models/kokoro-82M")
WEIGHTS_PATH = MODEL_DIR / "kokoro-v1_0.pth"
VOICES_DIR = MODEL_DIR / "voices"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Kokoro pipeline on {DEVICE}...")

# Pass local model weights path directly to KPipeline
pipeline = KPipeline(
    lang_code="a",
    model=str(WEIGHTS_PATH),
    repo_id="hexgrad/Kokoro-82M",
    device=DEVICE
)

# Load voice tensor
voice_path = VOICES_DIR / "af_heart.pt"
voice_pack = torch.load(voice_path, map_location=DEVICE, weights_only=True)

# Run inference
text = "The Kokoro model is running completely offline from my local directory!"
generator = pipeline(text, voice=voice_pack, speed=1.0)

for i, (gs, ps, audio) in enumerate(generator):
    output_filename = f"output_{i}.wav"
    sf.write(output_filename, audio, 24000)
    print(f"Saved: {output_filename}")