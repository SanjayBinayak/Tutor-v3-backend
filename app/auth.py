from fastapi import Header, HTTPException
from app.supabase_client import supabase


def get_current_user(authorization: str = Header(...)) -> str:
    """Verifies the Supabase JWT, returns the user's id."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        user_response = supabase.auth.get_user(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if not user_response or not user_response.user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_response.user.id


def _get_role(user_id: str) -> str:
    row = supabase.table("profiles").select("role").eq("id", user_id).single().execute().data
    if not row:
        raise HTTPException(status_code=403, detail="No profile found for this user")
    return row["role"]


def get_current_student(authorization: str = Header(...)) -> str:
    user_id = get_current_user(authorization)
    if _get_role(user_id) != "student":
        raise HTTPException(status_code=403, detail="This action requires a student account")
    return user_id


def get_current_teacher(authorization: str = Header(...)) -> str:
    user_id = get_current_user(authorization)
    if _get_role(user_id) != "teacher":
        raise HTTPException(status_code=403, detail="This action requires a teacher account")
    return user_id
