from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.supabase_client import supabase
from app.auth import get_current_teacher
from app.llm_providers import call_gemini_with_fallback

router = APIRouter(prefix="/announcements", tags=["announcements"])


class DraftAnnouncementRequest(BaseModel):
    notes: str  # rough bullet points / intent from the teacher


class PostAnnouncementRequest(BaseModel):
    content: str  # the final text (either AI-drafted-then-edited, or written from scratch)


@router.post("/draft")
def draft_announcement(req: DraftAnnouncementRequest, teacher_id: str = Depends(get_current_teacher)):
    prompt = f"""
You are helping a teacher write a short, clear announcement for their
students. Turn these rough notes into a polished announcement — friendly
but professional tone, concise, no unnecessary filler.

Teacher's notes:
{req.notes}

Return ONLY the announcement text, nothing else.
""".strip()
    draft = call_gemini_with_fallback(prompt)
    return {"draft": draft}


@router.post("")
def post_announcement(req: PostAnnouncementRequest, teacher_id: str = Depends(get_current_teacher)):
    result = supabase.table("announcements").insert({
        "teacher_id": teacher_id,
        "content": req.content,
    }).execute()
    return result.data[0]


@router.get("")
def list_announcements():
    return supabase.table("announcements").select("*").order("created_at", desc=True).execute().data
