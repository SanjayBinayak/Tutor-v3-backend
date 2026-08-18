from fastapi import APIRouter
from pydantic import BaseModel

from app.llm_providers import call_gemini_with_fallback

router = APIRouter(prefix="/diagrams", tags=["diagrams"])

DIAGRAM_PROMPT = """
You are generating a diagram to help a student understand a concept, using
Mermaid.js syntax (flowchart, sequence diagram, or graph — pick whichever
best fits the topic below).

Topic / question the diagram should explain:
{topic}

Rules:
- Respond with ONLY raw Mermaid syntax. No markdown code fences (no ```),
  no explanation, no extra text before or after.
- Prefer a flowchart (`graph TD` for vertical or `graph LR` for horizontal)
  for processes, structures, or concept relationships.
- Use `sequenceDiagram` only if the topic is clearly about a sequence of
  interactions over time (e.g. a protocol, an algorithm's steps between
  actors).
- Keep node labels short (a few words). Keep the whole diagram under 15
  nodes/steps so it stays readable.
""".strip()


class DiagramRequest(BaseModel):
    topic: str  # e.g. the student's question, or a teacher-picked concept name


def _clean_mermaid(raw: str) -> str:
    """Strips markdown code fences in case the model adds them despite instructions."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


@router.post("/generate")
def generate_diagram(req: DiagramRequest):
    """
    Public endpoint (same trust level as /personas/{id}/ask-anonymous) — the
    AI Tutor site's chat is anonymous/stateless, so this stays anonymous too.
    Returns raw Mermaid syntax for the frontend to render with mermaid.js.
    """
    prompt = DIAGRAM_PROMPT.format(topic=req.topic.strip())
    raw = call_gemini_with_fallback(prompt)
    return {"mermaid": _clean_mermaid(raw)}
