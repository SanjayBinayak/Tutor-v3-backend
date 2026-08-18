import re
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel

from app.supabase_client import supabase
from app.auth import get_current_teacher
from app.llm_providers import call_gemini_vision_with_fallback

router = APIRouter(prefix="/homework", tags=["homework"])

HOMEWORK_BUCKET = "homework"  # create this bucket in Supabase Storage first (not public)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------------------------------------------------------------------
# Teacher: define an assignment's requirements
# ---------------------------------------------------------------------------
class CreateAssignmentRequest(BaseModel):
    title: str
    requirements: str  # e.g. "Must show all working for Q1-5, answers to 2dp, ..."


@router.post("/assignments")
def create_assignment(req: CreateAssignmentRequest, teacher_id: str = Depends(get_current_teacher)):
    result = supabase.table("assignments").insert({
        "teacher_id": teacher_id,
        "title": req.title,
        "requirements": req.requirements,
    }).execute()
    return result.data[0]


@router.get("/assignments")
def list_assignments(teacher_id: str = Depends(get_current_teacher)):
    """Teacher's own assignments, for their dashboard."""
    return supabase.table("assignments").select("id, title, teacher_id, created_at") \
        .eq("teacher_id", teacher_id).order("created_at", desc=True).execute().data


@router.get("/assignments/{assignment_id}")
def get_assignment(assignment_id: str):
    """Public — the submission link page needs this to show the assignment title,
    with no login required (that's the whole point of the shareable link)."""
    result = supabase.table("assignments").select("id, title, created_at").eq("id", assignment_id).single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Assignment not found")
    return result.data


# ---------------------------------------------------------------------------
# Student: submit homework via the shareable link — no account needed,
# just name + email typed in on the page.
# ---------------------------------------------------------------------------
def _run_homework_check(submission_id: str, assignment_requirements: str,
                         file_bytes: bytes, mime_type: str):
    try:
        prompt = f"""
You are checking a student's homework submission against their teacher's
requirements. Be specific and constructive.

TEACHER'S REQUIREMENTS:
{assignment_requirements}

Look at the attached homework submission and provide:
1. Whether it meets each requirement (yes/no/partially, with the specific
   reason — quote or describe what you see).
2. Any errors you notice in the actual work (wrong answers, missing steps).
3. Concrete, encouraging suggestions for improvement.

Be honest about gaps, but keep the tone supportive — this is feedback for
a student, not a harsh grade.
""".strip()

        feedback = call_gemini_vision_with_fallback(prompt, file_bytes, mime_type)

        supabase.table("homework_submissions").update({
            "status": "checked",
            "ai_feedback": feedback,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", submission_id).execute()

    except Exception as e:
        supabase.table("homework_submissions").update({
            "status": "failed",
            "error_message": str(e),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", submission_id).execute()


@router.post("/assignments/{assignment_id}/submit", status_code=202)
async def submit_homework(
    assignment_id: str,
    background_tasks: BackgroundTasks,
    student_name: str = Form(...),
    student_email: str = Form(...),
    file: UploadFile = File(...),
):
    assignment = supabase.table("assignments").select("*").eq("id", assignment_id).single().execute().data
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    student_name = student_name.strip()
    student_email = student_email.strip()
    if not student_name:
        raise HTTPException(status_code=400, detail="Name is required")
    if not EMAIL_RE.match(student_email):
        raise HTTPException(status_code=400, detail="Enter a valid email address")

    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "application/pdf"):
        raise HTTPException(status_code=400, detail="Upload a JPEG/PNG/WEBP image or a PDF")

    file_bytes = await file.read()

    storage_path = f"{assignment_id}/{student_email}_{file.filename}"
    supabase.storage.from_(HOMEWORK_BUCKET).upload(
        storage_path, file_bytes, {"content-type": file.content_type, "upsert": "true"}
    )

    submission = supabase.table("homework_submissions").insert({
        "assignment_id": assignment_id,
        "student_name": student_name,
        "student_email": student_email,
        "file_path": storage_path,
        "status": "pending",
    }).execute().data[0]

    background_tasks.add_task(
        _run_homework_check, submission["id"], assignment["requirements"], file_bytes, file.content_type
    )

    return {"id": submission["id"], "status": "pending"}


@router.get("/submissions/{submission_id}")
def get_submission(submission_id: str):
    result = supabase.table("homework_submissions").select("*").eq("id", submission_id).single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Submission not found")
    return result.data


@router.get("/assignments/{assignment_id}/submissions")
def list_submissions_for_assignment(assignment_id: str, teacher_id: str = Depends(get_current_teacher)):
    """Teacher view: every student's submission + AI feedback for one assignment."""
    return supabase.table("homework_submissions").select("*") \
        .eq("assignment_id", assignment_id).order("created_at", desc=True).execute().data
