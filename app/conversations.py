"""ChatGPT-style chat history for the persona/tutor chat: list a student's
past conversations (across all their tutors), reopen one to keep asking
follow-ups inside it, or delete one. Conversations + messages are the same
tables /personas/{id}/ask already writes to (see schema_v1.sql /
schema_v7_quiz_and_history.sql) — this router just adds read/list/delete
access to that history.
"""
from fastapi import APIRouter, HTTPException, Depends

from app.supabase_client import supabase
from app.auth import get_current_student

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("")
def list_conversations(student_id: str = Depends(get_current_student)):
    """Most-recently-active first, with the persona's name attached so the
    sidebar can show e.g. 'Mr. Sharma — Physics' next to each chat."""
    rows = supabase.table("conversations").select(
        "id, persona_id, title, created_at, updated_at, personas(name)"
    ).eq("student_id", student_id).order("updated_at", desc=True).execute().data

    return [
        {
            "id": r["id"],
            "persona_id": r["persona_id"],
            "persona_name": (r.get("personas") or {}).get("name") if r.get("personas") else None,
            "title": r.get("title") or "New chat",
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]


@router.get("/{conversation_id}/messages")
def get_conversation_messages(conversation_id: str, student_id: str = Depends(get_current_student)):
    convo = supabase.table("conversations").select("id") \
        .eq("id", conversation_id).eq("student_id", student_id).single().execute().data
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return supabase.table("messages").select("id, role, content, created_at") \
        .eq("conversation_id", conversation_id).order("created_at").execute().data


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, student_id: str = Depends(get_current_student)):
    convo = supabase.table("conversations").select("id") \
        .eq("id", conversation_id).eq("student_id", student_id).single().execute().data
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    supabase.table("conversations").delete().eq("id", conversation_id).execute()
    return None
