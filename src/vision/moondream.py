import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "moondream2")

_model = None
_tokenizer = None


def get_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        _model = _model.to(device)
        _model.eval()
    return _model


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    return _tokenizer


@torch.inference_mode()
def caption(path, length="normal"):
    image = Image.open(path).convert("RGB") if isinstance(path, str) else path
    return get_model().caption(image, length=length)["caption"]


@torch.inference_mode()
def query(path, question):
    image = Image.open(path).convert("RGB") if isinstance(path, str) else path
    return get_model().query(image, question)["answer"]


@torch.inference_mode()
def detect(path, object_name):
    image = Image.open(path).convert("RGB") if isinstance(path, str) else path
    return get_model().detect(image, object_name)["objects"]


if __name__ == "__main__":
    image_path = os.path.join(PROJECT_ROOT, "assets", "sample_frame.jpeg")
    print("caption:", caption(image_path))
    print("query  :", query(image_path, "What kind of robot is this?"))
    print("detect :", len(detect(image_path, "robot")), "robot(s)")
