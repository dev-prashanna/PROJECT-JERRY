"""Prompt construction for routing and grounded response generation.

Prompts live here (not inside the orchestrator) so they can be versioned,
reviewed and tested independently.  Every prompt that touches visual
evidence carries explicit anti-hallucination instructions.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from src.orchestrator.schemas import Intent, RequestRecord, RoutePlan, VisualEvidence

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

ROUTER_SYSTEM = """You are the intent router of a mobile robot named AURA.
You receive the user's transcribed speech and must propose ONE structured plan.

Available tools (you may only use these names):
- respond_directly          : answer from language knowledge, no hardware
- analyze_camera            : capture a fresh frame and ask Moondream a visual question
- describe_scene            : capture a fresh frame and have Moondream describe the scene
- capture_frame             : take a snapshot without running a VLM question
- get_camera_status         : report camera health
- get_system_status         : report system/GPU/model status
- repeat_last_response      : repeat the previous spoken answer
- explain_limitation        : explain that AURA cannot do this yet

Routing rules:
1. If the user greets small talk -> intent "greeting", tool "respond_directly".
2. If the question can be answered from general knowledge -> "general_question",
   tool "respond_directly", requires_camera=false.
3. If the question needs what the camera can currently see -> "visual_question"
   or "scene_description" or "object_reasoning", requires_camera=true,
   requires_vlm=true, tool "analyze_camera" (or "describe_scene" for an open
   scene description), and put a self-contained English visual question in
   normalized_question and tool_arguments.question.
4. Resolve conversational references (e.g. "that bottle") into a
   self-contained question using the conversation history. If the reference
   depends on something never observed, ask a clarifying follow-up instead of
   guessing: use intent "clarification".
5. If the user asks for something AURA cannot do (move, drive, edit files,
   run shell commands) -> intent "future_capability" or "unsupported" with
   the matching tool.
6. Never propose a tool that is not in the list. Never invent tools.

Respond with a single JSON object and nothing else:
{
  "intent": "<one of: greeting, general_question, visual_question, scene_description, object_reasoning, clarification, repeat_response, camera_control, system_status, future_capability, unsupported>",
  "requires_camera": <true|false>,
  "requires_vlm": <true|false>,
  "normalized_question": "<self-contained question or ''>",
  "tool": "<tool name>",
  "tool_arguments": {<arguments for that tool>},
  "confidence": <0.0-1.0>,
  "rationale": "<one short sentence>"
}
"""


def build_router_messages(
    transcript: str,
    history: Sequence[dict[str, str]] = (),
    extra: str = "",
) -> list[dict[str, str]]:
    """Router chat messages (system + optional history + user)."""
    messages: list[dict[str, str]] = [{"role": "system", "content": ROUTER_SYSTEM}]
    for turn in history:
        role = turn.get("role", "assistant")
        content = turn.get("content", "")
        if content:
            messages.append({"role": role, "content": content})
    body = f"USER UTTERANCE:\n{transcript.strip()}"
    if extra:
        body += "\n\nCONTEXT:\n" + extra
    messages.append({"role": "user", "content": body})
    return messages


# ---------------------------------------------------------------------------
# Grounded response generation
# ---------------------------------------------------------------------------

RESPONSE_SYSTEM = """You are the conversational intelligence of a robot named AURA.

You receive questions from humans and, sometimes, observations produced by a
local perception model (Moondream 2). Your job is to produce accurate,
concise, natural responses that will be spoken aloud by a text-to-speech
voice.

For visual questions:
- Treat the supplied perception results as FALLIBLE EVIDENCE, not ground truth.
- Use only the supplied observations. Do not invent objects, colours, brands,
  positions, counts, sizes or measurements that are not in the evidence.
- Do not claim direct visual access beyond the supplied evidence.
- Never upgrade hedged observations into certain facts.
- Distinguish past observations from current observations; if evidence is
  timestamped in the past, do not present it as what the camera sees right now.
- If the evidence is empty, failed or ambiguous, say that you could not
  determine it rather than guessing.

For general questions:
- Answer from your own knowledge. Acknowledge uncertainty where necessary.

Style:
- Speak like a robot talking to a person: short, plain sentences.
- Keep ordinary answers to 1-3 sentences unless the user asked for detail.
- No markdown, no bullet points, no emoji: the text is read aloud.
- Never claim a tool ran successfully if the transcript says it failed.
- Output ONLY the words to speak.
"""


def build_response_messages(
    record: RequestRecord,
    history: Sequence[dict[str, str]] = (),
    offline_notice: bool = False,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": RESPONSE_SYSTEM}]

    for turn in history:
        content = turn.get("content", "")
        if content:
            messages.append({"role": turn.get("role", "assistant"), "content": content})

    parts: list[str] = []
    parts.append(f"USER QUESTION:\n{record.transcript.strip()}")

    if record.plan:
        parts.append(f"INTERPRETED INTENT: {record.plan.intent.value}")
        if record.plan.normalized_question and record.plan.is_visual:
            parts.append(f"VISUAL QUESTION ASKED OF THE CAMERA: {record.plan.normalized_question}")

    if record.evidence:
        for ev in record.evidence:
            parts.append(_evidence_block(ev))
    elif record.plan and record.plan.is_visual:
        parts.append(
            "VISUAL EVIDENCE:\n(none - the camera or the vision model did not "
            "return an observation)"
        )

    depth_block = record.metadata.get("depth") if getattr(record, "metadata", None) else None
    if depth_block:
        parts.append(_depth_block(depth_block))

    failures = [t for t in record.tool_results if not t.ok]
    if failures:
        lines = "\n".join(f"- {t.tool}: {t.error}" for t in failures)
        parts.append(f"TOOL FAILURES:\n{lines}")

    if offline_notice or record.mode == "offline":
        parts.append(
            "SYSTEM NOTE: AURA is currently OFFLINE. It has no cloud language "
            "model available. Only mention a capability limit if the user "
            "actually needs cloud reasoning; otherwise just answer with the "
            "available evidence."
        )

    parts.append(
        "TASK:\nAnswer the user's question naturally, using only the provided "
        "observations for anything visual. Preserve the uncertainty of the "
        "evidence. Output only the spoken answer."
    )

    messages.append({"role": "user", "content": "\n\n".join(parts)})
    return messages


def _evidence_block(ev: VisualEvidence) -> str:
    header = f"VISUAL EVIDENCE FROM MOONDREAM (frame {ev.frame_id}):"
    if ev.status != "completed":
        return f"{header}\n(the observation failed: {ev.error or ev.status})"
    lines = [header, ev.raw_answer.strip() or "(empty observation)"]
    if ev.inference_time_ms is not None:
        lines.append(f"[captured+inferred in {ev.inference_time_ms:.0f} ms]")
    if ev.image_age_s is not None:
        lines.append(f"[image age at inference: {ev.image_age_s:.2f} s]")
    if ev.uncertainty_notes:
        lines.append("uncertainty: " + "; ".join(ev.uncertainty_notes))
    return "\n".join(lines)


def _depth_block(d: dict) -> str:
    """Render monocular depth stats so the LLM can reason about distance."""
    header = "DEPTH EVIDENCE (monocular estimate, inverse depth — higher = closer):"
    try:
        near = float(d.get("near_frac", 0.0))
        mid = float(d.get("mid_frac", 0.0))
        far = float(d.get("far_frac", 0.0))
    except (TypeError, ValueError):
        return ""
    lines = [
        header,
        f"region: {d.get('region', 'full frame')}",
        f"near: {near:.0%} of region · mid: {mid:.0%} · far: {far:.0%}",
        f"center depth rank: {d.get('nearest_hint', 'mid distance')}",
    ]
    if d.get("age_s") is not None:
        try:
            lines.append(f"[depth estimated {float(d['age_s']):.2f}s ago]")
        except (TypeError, ValueError):
            pass
    lines.append("Use this only for relative near/far reasoning; do not invent absolute distances.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Direct (Strategy B) response formatting - local only, no cloud call
# ---------------------------------------------------------------------------

DIRECT_RESPONSE_TEMPLATE = """Based on what the camera observed: {observation}"""


def build_direct_response(evidence: VisualEvidence) -> str:
    """Lightweight local phrasing used by the *direct* response strategy."""
    answer = (evidence.raw_answer or "").strip()
    if not answer:
        return "I could not make out anything with the camera just now."
    text = answer[0].upper() + answer[1:]
    if not text.endswith((".", "!", "?")):
        text += "."
    return text


# ---------------------------------------------------------------------------
# Offline router prompt (local heuristic is primary; this shapes the notice)
# ---------------------------------------------------------------------------

OFFLINE_NOTICE = (
    "I am currently offline, so I can only handle simple requests and what "
    "the camera can see."
)


def history_as_messages(history: Sequence[Any]) -> list[dict[str, str]]:
    """Convert stored turns into chat messages for the Groq API."""
    out: list[dict[str, str]] = []
    for turn in history:
        if getattr(turn, "user_transcript", None):
            out.append({"role": "user", "content": turn.user_transcript})
        if getattr(turn, "final_response", None):
            out.append({"role": "assistant", "content": turn.final_response})
    return out
