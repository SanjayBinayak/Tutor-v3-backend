# study.py - Updated with better formatting for mind maps and other sections

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from app.supabase_client import supabase
from app.auth import get_current_user
from app.llm_providers import call_study_gemini_with_fallback, call_study_gemini_vision_with_fallback

router = APIRouter(prefix="/study", tags=["study"])

# ---------------------------------------------------------------------------
# Section definitions — each is generated (and regenerated) independently, so
# the frontend can render every requested section as its own tab and re-roll
# just one without touching the others.
# ---------------------------------------------------------------------------
SECTION_INSTRUCTIONS = {
    "explanation": (
        "Give a clear, well-structured explanation of the concept/question below. "
        "Use plain language first, then introduce notation/formulas where needed. "
        "Format your response with:\n"
        "- Clear headings using ## and ###\n"
        "- Bullet points for key ideas\n"
        "- Code blocks or proper formatting for equations\n"
        "Keep it engaging and easy to read."
    ),
    "summary": (
        "Summarize the concept/question below in a visually structured format. "
        "Use:\n"
        "- ### headings for main topics\n"
        "- Bullet points for key facts\n"
        "- Tables for comparisons where applicable\n"
        "- Bold text for important terms\n"
        "Make it exam-ready and easy to review quickly."
    ),
    "examples": (
        "Work through ONE worked example that illustrates the concept/question "
        "below, step by step. Format as:\n"
        "## Problem Statement\n"
        "## Solution\n"
        "### Step 1: ...\n"
        "### Step 2: ...\n"
        "## Final Answer\n"
        "Show every step of the calculation/reasoning clearly."
    ),
    "key_points": (
        "List the key points a student MUST remember about the concept/question "
        "below for an exam. Format as:\n"
        "- **Key Point**: Brief explanation\n"
        "- **Formula**: Important equations in code blocks\n"
        "- **Common Mistake**: What to avoid\n"
        "Use bold, code blocks, and proper formatting."
    ),
    "flashcards": (
        "Generate 4-6 flashcards for the concept/question below. Format as:\n"
        "## Flashcard 1\n"
        "**Q:** Question here\n"
        "**A:** Answer here\n\n"
        "## Flashcard 2\n"
        "... and so on.\n"
        "Make them clear, concise, and exam-focused."
    ),
    "mind_map": (
        "Create a comprehensive mind map for the concept/question below. Format as:\n\n"
        "# 📌 MAIN TOPIC\n\n"
        "## 🔹 Subtopic 1\n"
        "- **Key Point**: Description\n"
        "- **Formula**: `equation`\n"
        "- **Visual**: Simple ASCII diagram if helpful\n\n"
        "## 🔹 Subtopic 2\n"
        "... and so on.\n\n"
        "Use emojis for visual appeal (📌, 🔹, ⚡, 📊, 🔄, etc.)\n"
        "Include:\n"
        "- Clear hierarchical structure (use #, ##, ###)\n"
        "- Bullet points and sub-bullets\n"
        "- Important formulas in code blocks\n"
        "- Comparisons using tables where helpful\n"
        "- ASCII diagrams or visual elements\n"
        "Make it comprehensive yet scannable for quick revision."
    ),
    "rough_work": (
        "Show the full step-by-step derivation/working for the concept/question "
        "below, as rough working notes — every intermediate step, no skipped algebra. "
        "Format with clear numbering and proper mathematical notation."
    ),
    "real_world": (
        "Give 2-3 real-world applications of the concept/question below. Format as:\n"
        "## Application 1: [Title]\n"
        "- **Context**: ...\n"
        "- **How it works**: ...\n"
        "- **Why it matters**: ...\n"
        "Be concrete and specific, not generic."
    ),
}

# Enhanced mind map prompt for better visual output
MIND_MAP_ENHANCED = """
Create a detailed, visually structured mind map for the concept below.

FORMAT REQUIREMENTS:
- Use # for main topic (large)
- Use ## for subtopics 
- Use ### for sub-subtopics
- Use bullet points (-) for details
- Use ``` for formulas and code
- Use | for tables when comparing items
- Use emojis: 📌 🔹 ⚡ 📊 🔄 🔗 🎯 📈 🛠️ 🔌

STRUCTURE:
1. Start with "# 📌 [MAIN TOPIC]"
2. Break into 5-10 main subtopics with "## 🔹 [Subtopic]"
3. For each subtopic, include:
   - Key concepts as bullet points
   - Important formulas in code blocks
   - Tables for comparisons
   - Simple ASCII diagrams if helpful
4. End with "## 📊 Quick Reference" section with key formulas

Make it comprehensive, exam-ready, and visually scannable.
"""

VALID_SECTIONS = set(SECTION_INSTRUCTIONS)

MODIFIER_INSTRUCTIONS = {
    "simpler": "Rewrite this significantly simpler — assume a much younger/newer student.",
    "more_math": "Add more mathematical rigor/detail/derivation than before.",
    "example": "Add or swap in a fresh worked example.",
}


class AskStudyRequest(BaseModel):
    question: str
    sections: list[str]
    persona_id: Optional[str] = None  # None = generic "Default Tutor"


class RegenerateSectionRequest(BaseModel):
    question: str
    section: str
    modifier: Optional[str] = None  # key into MODIFIER_INSTRUCTIONS, or None
    persona_id: Optional[str] = None


def _validate_sections(sections: list[str]):
    if not sections:
        raise HTTPException(status_code=400, detail="Request at least one section")
    unknown = set(sections) - VALID_SECTIONS
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown section(s): {', '.join(sorted(unknown))}")


def _get_ready_persona_or_404(persona_id: str) -> dict:
    persona = supabase.table("personas").select("*").eq("id", persona_id).single().execute().data
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    if persona["status"] != "ready":
        raise HTTPException(status_code=409, detail=f"Persona is not ready (status: {persona['status']})")
    return persona


def _build_system_context(persona: Optional[dict]) -> str:
    """Persona-flavored framing, or a generic 'Default Tutor' framing when no
    persona_id was given — this is the model choice from the elicitation."""
    if persona is None:
        return (
            "You are a clear, patient, encouraging tutor helping a student "
            "understand a concept. Use accurate terminology but explain it "
            "accessibly, with no particular persona or teaching style."
        )
    return f"""
You are {persona['name']}, a teacher. Answer in your own authentic teaching style,
using the reference material below where relevant. Stay in character as this teacher.

TEACHING STYLE & TOPICS YOU'VE COVERED:
{persona['teaching_style']}

YOUR PROBLEM-SOLVING APPROACH:
{persona['problem_solving_approach']}

REFERENCE MATERIAL (questions you've solved before, for consistent method/notation):
{persona['solved_questions']}
""".strip()


def _build_section_prompt(system_context: str, question: str, section: str, modifier: Optional[str] = None) -> str:
    # Use enhanced prompt for mind_map
    if section == "mind_map":
        instructions = MIND_MAP_ENHANCED
    else:
        instructions = SECTION_INSTRUCTIONS[section]
    
    modifier_line = f"\n\nADDITIONAL INSTRUCTION: {MODIFIER_INSTRUCTIONS[modifier]}" if modifier else ""
    
    return f"""
{system_context}

TASK: {instructions}{modifier_line}

CONCEPT / QUESTION:
{question}

Remember to format your response with proper markdown for a rich visual experience.
""".strip()


# ---------------------------------------------------------------------------
# Text-based ask — a question/topic + which sections (tabs) to generate.
# ---------------------------------------------------------------------------
@router.post("/ask")
def ask_study(req: AskStudyRequest, user_id: str = Depends(get_current_user)):
    _validate_sections(req.sections)
    persona = _get_ready_persona_or_404(req.persona_id) if req.persona_id else None
    system_context = _build_system_context(persona)

    sections = {
        section: call_study_gemini_with_fallback(_build_section_prompt(system_context, req.question, section))
        for section in req.sections
    }
    return {"question": req.question, "persona_id": req.persona_id, "sections": sections}


# ---------------------------------------------------------------------------
# Regenerate a single section — matches "Regenerate / Simpler / More Math /
# Example" per-tab controls, without re-running every other tab.
# ---------------------------------------------------------------------------
@router.post("/section")
def regenerate_section(req: RegenerateSectionRequest, user_id: str = Depends(get_current_user)):
    _validate_sections([req.section])
    if req.modifier and req.modifier not in MODIFIER_INSTRUCTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown modifier: {req.modifier}")

    persona = _get_ready_persona_or_404(req.persona_id) if req.persona_id else None
    system_context = _build_system_context(persona)
    prompt = _build_section_prompt(system_context, req.question, req.section, req.modifier)
    content = call_study_gemini_with_fallback(prompt)
    return {"section": req.section, "content": content}


# ---------------------------------------------------------------------------
# Notes analysis — upload a photo of handwritten notes or a PDF, pick which
# sections to generate from it. Each selected section = its own tab.
# ---------------------------------------------------------------------------
ALLOWED_NOTE_TYPES = ("image/jpeg", "image/png", "image/webp", "application/pdf")


@router.post("/analyze-notes")
async def analyze_notes(
    sections: str = Form(...),  # comma-separated section keys, e.g. "explanation,summary"
    persona_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    section_list = [s.strip() for s in sections.split(",") if s.strip()]
    _validate_sections(section_list)

    if file.content_type not in ALLOWED_NOTE_TYPES:
        raise HTTPException(status_code=400, detail="Upload a JPEG/PNG/WEBP image or a PDF")

    file_bytes = await file.read()
    persona = _get_ready_persona_or_404(persona_id) if persona_id else None
    system_context = _build_system_context(persona)

    placeholder_question = (
        "The concept/question is contained in the attached notes image/PDF — "
        "read it and respond accordingly."
    )

    results = {
        section: call_study_gemini_vision_with_fallback(
            _build_section_prompt(system_context, placeholder_question, section),
            file_bytes,
            file.content_type,
        )
        for section in section_list
    }
    return {"persona_id": persona_id, "sections": results}