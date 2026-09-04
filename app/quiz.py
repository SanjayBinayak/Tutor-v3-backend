"""Quiz section for a persona (tutor) — generate a multiple-choice quiz from
whatever that persona was built from (topics_covered / solved_questions /
teaching_style), then grade attempts server-side so correct answers never
reach the browser until after submission. Mirrors the old
material_quizzes / material_quiz_attempts feature (schema_v4_quiz.sql),
just built from a persona instead of an uploaded document.
"""
import json
import re
import uuid
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.supabase_client import supabase
from app.auth import get_current_student
from app.llm_providers import call_gemini_with_fallback

router = APIRouter(tags=["quiz"])

MARKING_SCHEMES = {
    "basic": {"label": "+1 / 0", "correct": 1, "incorrect": 0},
    "board": {"label": "+1 / \u22120.25", "correct": 1, "incorrect": -0.25},
    "jee": {"label": "+4 / \u22121", "correct": 4, "incorrect": -1},
}
DIFFICULTIES = ("mixed", "easy", "medium", "hard")


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
class GenerateQuizRequest(BaseModel):
    num_questions: int = 8
    difficulty: str = "mixed"
    marking: str = "basic"


def _extract_json(text: str):
    """Gemini sometimes wraps JSON in ```json fences or adds stray prose
    around it — pull out the first {...} or [...] block and parse that."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError("Could not parse quiz JSON from model response")


def _build_quiz_prompt(persona: dict, num_questions: int, difficulty: str) -> str:
    difficulty_instruction = (
        "Mix easy, medium, and hard questions roughly evenly."
        if difficulty == "mixed"
        else f"Every question should be {difficulty} difficulty."
    )
    return f"""
You are writing a {num_questions}-question multiple-choice quiz to test a
student's understanding of material taught by {persona['name']}. Base the
questions on the topics and reference material below — stay within what
was actually covered, don't invent unrelated topics.

TOPICS COVERED:
{persona.get('topics_covered') or '(none on file)'}

REFERENCE MATERIAL (worked examples / solved questions, for context on
depth and notation):
{(persona.get('solved_questions') or '(none on file)')[:12000]}

{difficulty_instruction}

Return ONLY a JSON array (no markdown fences, no commentary) of exactly
{num_questions} objects, each shaped like:
{{
  "topic": "short topic label, e.g. 'Kinematics'",
  "difficulty": "easy" | "medium" | "hard",
  "question": "the question text",
  "options": ["option A", "option B", "option C", "option D"],
  "correct_index": 0,
  "explanation": "one or two sentences on why that option is correct"
}}

Exactly 4 options per question, correct_index is a 0-based index into
options. Return the JSON array and nothing else.
""".strip()


@router.post("/personas/{persona_id}/quiz")
def generate_persona_quiz(
    persona_id: str, req: GenerateQuizRequest, student_id: str = Depends(get_current_student)
):
    persona = supabase.table("personas").select("*").eq("id", persona_id).single().execute().data
    if not persona:
        raise HTTPException(status_code=404, detail="Tutor not found")
    if persona["status"] != "ready":
        raise HTTPException(status_code=409, detail=f"Tutor is not ready (status: {persona['status']})")

    num_questions = max(1, min(req.num_questions, 25))
    difficulty = req.difficulty if req.difficulty in DIFFICULTIES else "mixed"
    marking = req.marking if req.marking in MARKING_SCHEMES else "basic"

    prompt = _build_quiz_prompt(persona, num_questions, difficulty)
    raw = call_gemini_with_fallback(prompt)
    try:
        parsed = _extract_json(raw)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("Expected a non-empty JSON array of questions")
    except Exception:
        raise HTTPException(status_code=502, detail="Couldn't generate a quiz right now — try again.")

    questions = []
    for item in parsed[:num_questions]:
        options = item.get("options") or []
        if len(options) < 2:
            continue
        correct_index = item.get("correct_index", 0)
        if not isinstance(correct_index, int) or not (0 <= correct_index < len(options)):
            correct_index = 0
        questions.append({
            "id": str(uuid.uuid4()),
            "topic": str(item.get("topic") or "General")[:80],
            "difficulty": item.get("difficulty") if item.get("difficulty") in DIFFICULTIES[1:] else "medium",
            "question": item.get("question") or "",
            "options": [str(o) for o in options],
            "correct_index": correct_index,
            "explanation": item.get("explanation") or "",
        })

    if not questions:
        raise HTTPException(status_code=502, detail="Couldn't generate a quiz right now — try again.")

    row = supabase.table("persona_quizzes").insert({
        "persona_id": persona_id,
        "student_id": student_id,
        "num_questions": len(questions),
        "difficulty": difficulty,
        "marking": marking,
        "questions": questions,
    }).execute().data[0]

    public_questions = [
        {"id": q["id"], "topic": q["topic"], "difficulty": q["difficulty"],
         "question": q["question"], "options": q["options"]}
        for q in questions
    ]

    return {
        "quiz_id": row["id"],
        "marking": MARKING_SCHEMES[marking],
        "questions": public_questions,
    }


# ---------------------------------------------------------------------------
# Submit / grade
# ---------------------------------------------------------------------------
class SubmitQuizAttemptRequest(BaseModel):
    answers: dict[str, int] = {}
    time_taken_seconds: int | None = None


@router.post("/personas/{persona_id}/quiz/{quiz_id}/attempt")
def submit_persona_quiz_attempt(
    persona_id: str, quiz_id: str, req: SubmitQuizAttemptRequest,
    student_id: str = Depends(get_current_student),
):
    quiz = supabase.table("persona_quizzes").select("*") \
        .eq("id", quiz_id).eq("persona_id", persona_id).eq("student_id", student_id) \
        .single().execute().data
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")

    scheme = MARKING_SCHEMES.get(quiz["marking"], MARKING_SCHEMES["basic"])
    questions = quiz["questions"]

    score = 0.0
    correct_count = 0
    wrong_count = 0
    unanswered_count = 0
    topic_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
    review = []

    for q in questions:
        your_answer = req.answers.get(q["id"])
        topic_stats[q["topic"]][1] += 1
        if your_answer is None:
            unanswered_count += 1
            correct = False
        else:
            correct = your_answer == q["correct_index"]
            if correct:
                score += scheme["correct"]
                correct_count += 1
                topic_stats[q["topic"]][0] += 1
            else:
                score += scheme["incorrect"]
                wrong_count += 1
        review.append({
            "id": q["id"],
            "topic": q["topic"],
            "question": q["question"],
            "options": q["options"],
            "your_answer": your_answer,
            "correct_index": q["correct_index"],
            "correct": correct,
            "explanation": q["explanation"],
        })

    max_score = len(questions) * scheme["correct"]
    weak_topics = [
        {"topic": t, "correct": c, "total": total, "pct": round(100 * c / total) if total else 0}
        for t, (c, total) in topic_stats.items()
    ]
    weak_topics.sort(key=lambda t: t["pct"])

    attempt = supabase.table("persona_quiz_attempts").insert({
        "quiz_id": quiz_id,
        "persona_id": persona_id,
        "student_id": student_id,
        "answers": req.answers,
        "score": score,
        "max_score": max_score,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "unanswered_count": unanswered_count,
        "time_taken_seconds": req.time_taken_seconds,
    }).execute().data[0]

    return {
        "score": score,
        "max_score": max_score,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "unanswered_count": unanswered_count,
        "weak_topics": weak_topics,
        "review": review,
    }


@router.get("/personas/{persona_id}/quiz-attempts")
def list_persona_quiz_attempts(persona_id: str, student_id: str = Depends(get_current_student)):
    return supabase.table("persona_quiz_attempts").select(
        "id, quiz_id, score, max_score, correct_count, wrong_count, unanswered_count, "
        "time_taken_seconds, created_at"
    ).eq("persona_id", persona_id).eq("student_id", student_id) \
        .order("created_at", desc=True).execute().data
