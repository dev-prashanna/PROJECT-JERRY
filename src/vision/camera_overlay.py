import os
import threading
import time
import cv2
from PIL import Image 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from ..llm.question_gen import QuestionGenerator
except ImportError:  # running this file directly instead of as a package
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from llm.question_gen import QuestionGenerator

def wrap_text(text, font, font_scale, thickness, max_width):
    """Word-wraps text so no line exceeds max_width pixels."""
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if cv2.getTextSize(candidate, font, font_scale, thickness)[0][0] <= max_width:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def draw_multiline_text(frame, text, start_x=20, start_y=40):
    """Draws wrapped text on a dark background box so it stays readable."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 255, 0)  # Green in BGR
    thickness = 2
    line_spacing = 30    # Vertical distance between lines

    max_width = frame.shape[1] - 2 * start_x
    lines = wrap_text(text, font, font_scale, thickness, max_width)

    # Drop lines that would fall below the bottom of the frame.
    max_lines = max(1, (frame.shape[0] - start_y) // line_spacing)
    lines = lines[:max_lines]
    if not lines:
        return frame

    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    box_w = max(w for w, _ in sizes) if sizes else 0
    box_h = line_spacing * len(lines)

    top = start_y - 24
    cv2.rectangle(
        frame,
        (start_x - 8, top),
        (min(start_x + box_w + 8, frame.shape[1] - 1), min(top + box_h, frame.shape[0] - 1)),
        (0, 0, 0),
        -1,
    )

    for i, line in enumerate(lines):
        y = start_y + i * line_spacing
        cv2.putText(frame, line, (start_x, y), font, font_scale, color, thickness)

    return frame


device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "moondream2"
)

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, trust_remote_code=True, dtype=torch.bfloat16
).to(device)
model.eval()
print(f"Model loaded successfully on {device}")

cap=cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam") 
    exit()

CAPTURE_INTERVAL = 5.0  # seconds between automatic captures

question_generator = QuestionGenerator()

answer_text = "Waiting for first capture..."
infer_thread = None


def run_inference(pil_image):
    global answer_text
    try:
        question = question_generator.next_question(pil_image)
        print(f"[{time.strftime('%H:%M:%S')}] Q: {question}")
        enc_img = model.encode_image(pil_image)
        answer = model.query(enc_img, question)["answer"]
        answer_text = f"Q: {question}\n{answer}"
        print(f"A: {answer}")
    except Exception as e:
        answer_text = f"Error during inference: {e}"
        print(answer_text)


def capture_now(frame, trigger):
    """Grab the current frame and run inference on it in the background."""
    global answer_text, infer_thread
    if infer_thread is not None and infer_thread.is_alive():
        return  # previous shot still running; skip instead of stacking jobs

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    answer_text = "Thinking..."
    print(f"[{time.strftime('%H:%M:%S')}] {trigger} capture -> asking Gemini for a question")
    infer_thread = threading.Thread(target=run_inference, args=(pil_image,), daemon=True)
    infer_thread.start()


print("Camera started! Press 'q' in the window to exit.")
print(f"Auto-capturing every {CAPTURE_INTERVAL:.0f}s (press space for an immediate shot).")

next_capture = time.monotonic() + CAPTURE_INTERVAL

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    if answer_text:
        draw_multiline_text(frame, answer_text)

    cv2.imshow("Webcam Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        capture_now(frame, "manual")

    if time.monotonic() >= next_capture:
        next_capture = time.monotonic() + CAPTURE_INTERVAL
        capture_now(frame, "auto")

cap.release()
cv2.destroyAllWindows()
