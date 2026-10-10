"""Generates exploratory VLM questions with Gemini, as a curious first-time observer."""

from collections import deque
from pathlib import Path

from PIL import Image

try:
    from .gemini import generate
except ImportError:  # running this file directly instead of as a package
    from gemini import generate

SYSTEM_PROMPT = """\
You are a curious newborn looking at the world through a camera for the very first time.
You do not know names for things yet - you explore by noticing shapes, textures, light,
depth, motion, and how things relate to each other.

You are shown one camera frame. Produce exactly ONE question about what you see that would
genuinely help you understand this scene better next.

Pick a different direction each time:
- purpose: what is that object likely used for?
- space: what is behind, underneath, or beside the main object?
- depth: which object is closest to the camera, which is furthest?
- material: what would the surfaces feel like?
- cause: what seems about to happen, or what would happen if something moved?
- hidden: what might be just outside the frame?
- relation: how are the parts of the scene connected or arranged?
- contrast: which part of the scene is busiest, emptiest, brightest?

Good examples of the style you should produce:
"What is the black box mounted on the arm pointing at?"
"What would happen if the robot moved forward one step?"
"Does the floor look slippery or grippy?"

Rules:
- Output only the single question, nothing else: no preamble, no quotes, no explanation.
- Never ask "who is in the image", and never ask for a caption or a list of objects.
- It must be answerable from this single still image.
- Do not repeat a question that has already been asked; they are listed for you.
- Maximum 20 words.\
"""

USER_PROMPT = "Here is the current camera frame:"

FALLBACK_QUESTION = "What is the most interesting thing in this scene?"
MAX_HISTORY = 5


def _clean(raw: str) -> str:
    question = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    question = question.strip("\"'“”‘’ ")
    lowered = question.lower()
    for prefix in ("question:", "q:"):
        if lowered.startswith(prefix):
            question = question[len(prefix) :].strip()
            break
    if not question:
        return FALLBACK_QUESTION
    if not question.endswith("?"):
        question += "?"
    return question


class QuestionGenerator:
    """Holds the questions already asked so Gemini keeps exploring new angles."""

    def __init__(self, model: str | None = None, history: int = MAX_HISTORY):
        self.model = model
        self._asked = deque(maxlen=history)

    @property
    def asked(self) -> list[str]:
        return list(self._asked)

    def next_question(self, image) -> str:
        prompt = USER_PROMPT
        if self._asked:
            listed = "\n".join(f"- {q}" for q in self._asked)
            prompt = f"{prompt}\n\nQuestions already asked:\n{listed}"

        raw = generate(
            prompt,
            images=[image],
            system=SYSTEM_PROMPT,
            model=self.model,
            temperature=1.0,
        )
        question = _clean(raw)
        self._asked.append(question)
        return question


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample exploratory questions for an image")
    parser.add_argument("image", help="path to an image")
    parser.add_argument("-n", type=int, default=3, help="how many questions to sample")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    generator = QuestionGenerator(model=args.model)
    image = Image.open(Path(args.image)).convert("RGB")
    for i in range(args.n):
        print(f"{i + 1}. {generator.next_question(image)}")
