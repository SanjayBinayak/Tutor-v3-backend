import tempfile
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import ALLOWED_ORIGINS
from app.supabase_client import supabase
from app.auth import get_current_student
from app.ingestion import (
    fetch_transcript, build_persona_profile, build_persona_profile_from_youtube_chunked,
    transcribe_uploaded_audio, extract_pdf_transcript_via_images,
)
from app.llm_providers import call_gemini_with_fallback
from app.persona_rag import chunk_and_embed_persona, retrieve_relevant_solved_questions
from app.homework import router as homework_router
from app.announcements import router as announcements_router
from app.diagrams import router as diagrams_router
from app.notifications import send_notification_email

app = FastAPI(title="Classroom Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(homework_router)
app.include_router(announcements_router)
app.include_router(diagrams_router)


# ---------------------------------------------------------------------------
# Persona creation (async — ingestion takes 10-15+ min for a full lecture).
#
# Students create personas themselves now, from any of three sources:
#   - a YouTube link              (POST /personas)
#   - an uploaded audio/video file (POST /personas/upload)
#   - an uploaded PDF              (POST /personas/upload-pdf, image-extraction only)
# ---------------------------------------------------------------------------
class CreatePersonaRequest(BaseModel):
    name: str
    youtube_url: str


def _run_ingestion_job(persona_id: str, youtube_url: str):
    try:
        try:
            # Primary path: Gemini watches the video directly via its
            # YouTube URL, processed in ~20-min clipped chunks — no
            # download, no transcription, no MEDIA_RESOLUTION_LOW quality
            # loss, and no length ceiling. Sidesteps yt-dlp's ongoing fight
            # with YouTube's anti-automation blocking entirely.
            profile = build_persona_profile_from_youtube_chunked(youtube_url)
        except Exception as youtube_err:
            # Fallback: download audio + transcribe + 3-provider analysis.
            # Covers cases the direct path can't (video over ~3hrs even at
            # low resolution, or a transient Gemini-side video-fetch issue).
            print(f"[WARN] Direct YouTube analysis failed ({youtube_err}); "
                  f"falling back to download+transcribe pipeline.")
            with tempfile.TemporaryDirectory() as work_dir:
                transcript = fetch_transcript(youtube_url, work_dir)
                profile = build_persona_profile(transcript)

        supabase.table("personas").update({
            "status": "ready",
            "teaching_style": profile["teaching_style"],
            "topics_covered": profile["topics_covered"],
            "problem_solving_approach": profile["problem_solving_approach"],
            "solved_questions": profile["solved_questions"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", persona_id).execute()

        _index_persona_for_retrieval(persona_id, profile["solved_questions"])

    except Exception as e:
        supabase.table("personas").update({
            "status": "failed",
            "error_message": str(e),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", persona_id).execute()


def _index_persona_for_retrieval(persona_id: str, solved_questions: str) -> None:
    """Chunks+embeds solved_questions into persona_chunks right after a
    persona goes ready, so /study/ask and /personas/{id}/ask can retrieve
    just the relevant records instead of resending the whole blob on every
    call (see persona_rag.py). Best-effort: a failure here (e.g. schema_v5
    not migrated yet) shouldn't fail persona ingestion — retrieval just
    falls back to a bounded slice of the full text until this succeeds."""
    try:
        chunk_and_embed_persona(persona_id, solved_questions)
    except Exception as e:
        print(f"[WARN] persona_chunks indexing failed for {persona_id}: {e}")


@app.post("/personas", status_code=202)
def create_persona(req: CreatePersonaRequest, background_tasks: BackgroundTasks,
                    student_id: str = Depends(get_current_student)):
    result = supabase.table("personas").insert({
        "name": req.name,
        "source_youtube_url": req.youtube_url,
        "status": "processing",
    }).execute()

    persona_id = result.data[0]["id"]
    background_tasks.add_task(_run_ingestion_job, persona_id, req.youtube_url)

    return {"id": persona_id, "status": "processing"}


# ---------------------------------------------------------------------------
# Persona creation from an uploaded audio or video file — bypasses yt-dlp
# and Gemini's YouTube-URL fetch entirely, since the file is already on
# hand. Video files go through the same path: ffmpeg pulls just the audio
# track out in transcribe_uploaded_audio (see ingestion.py), so a video
# upload is just as "audio-first" as the YouTube path.
# ---------------------------------------------------------------------------
ALLOWED_AUDIO_CONTENT_TYPES = (
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/mp4", "audio/x-m4a", "audio/m4a", "audio/ogg", "audio/webm",
    "video/mp4", "video/webm", "video/quicktime", "video/x-matroska",
)


def _run_ingestion_job_from_upload(persona_id: str, file_bytes: bytes, filename: str):
    try:
        with tempfile.TemporaryDirectory() as work_dir:
            transcript = transcribe_uploaded_audio(file_bytes, filename, work_dir)
            profile = build_persona_profile(transcript)

        supabase.table("personas").update({
            "status": "ready",
            "teaching_style": profile["teaching_style"],
            "topics_covered": profile["topics_covered"],
            "problem_solving_approach": profile["problem_solving_approach"],
            "solved_questions": profile["solved_questions"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", persona_id).execute()

        _index_persona_for_retrieval(persona_id, profile["solved_questions"])

    except Exception as e:
        supabase.table("personas").update({
            "status": "failed",
            "error_message": str(e),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", persona_id).execute()


@app.post("/personas/upload", status_code=202)
async def create_persona_from_upload(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    source_youtube_url: str | None = Form(None),
    file: UploadFile = File(...),
    student_id: str = Depends(get_current_student),
):
    if file.content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}' — upload an mp3/m4a/wav/ogg/mp4/mov/webm/mkv audio or video file.",
        )

    file_bytes = await file.read()

    result = supabase.table("personas").insert({
        "name": name,
        "source_youtube_url": source_youtube_url,  # optional, just for reference
        "status": "processing",
    }).execute()

    persona_id = result.data[0]["id"]
    background_tasks.add_task(_run_ingestion_job_from_upload, persona_id, file_bytes, file.filename or "audio.mp3")

    return {"id": persona_id, "status": "processing"}


# ---------------------------------------------------------------------------
# Persona creation from an uploaded PDF — image extraction only. Every page
# is rendered to an image and read with Gemini vision (see
# extract_pdf_transcript_via_images in ingestion.py); there's no text-layer
# extraction step anywhere in this path, by design.
# ---------------------------------------------------------------------------
def _run_ingestion_job_from_pdf(persona_id: str, file_bytes: bytes):
    try:
        transcript = extract_pdf_transcript_via_images(file_bytes)
        if not transcript.strip():
            raise ValueError("Couldn't read any content from this PDF — try a clearer scan.")
        profile = build_persona_profile(transcript)

        supabase.table("personas").update({
            "status": "ready",
            "teaching_style": profile["teaching_style"],
            "topics_covered": profile["topics_covered"],
            "problem_solving_approach": profile["problem_solving_approach"],
            "solved_questions": profile["solved_questions"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", persona_id).execute()

        _index_persona_for_retrieval(persona_id, profile["solved_questions"])

    except Exception as e:
        supabase.table("personas").update({
            "status": "failed",
            "error_message": str(e),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", persona_id).execute()


@app.post("/personas/upload-pdf", status_code=202)
async def create_persona_from_pdf(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    file: UploadFile = File(...),
    student_id: str = Depends(get_current_student),
):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}' — upload a PDF file.",
        )

    file_bytes = await file.read()

    result = supabase.table("personas").insert({
        "name": name,
        "status": "processing",
    }).execute()

    persona_id = result.data[0]["id"]
    background_tasks.add_task(_run_ingestion_job_from_pdf, persona_id, file_bytes)

    return {"id": persona_id, "status": "processing"}


@app.get("/personas")
def list_personas():
    result = supabase.table("personas").select(
        "id, name, status, created_at"
    ).order("created_at", desc=True).execute()
    return result.data


@app.get("/personas/{persona_id}")
def get_persona(persona_id: str):
    result = supabase.table("personas").select("*").eq("id", persona_id).single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Persona not found")
    return result.data


# ---------------------------------------------------------------------------
# Ask a tutor persona a question
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    conversation_id: str | None = None


@app.post("/personas/{persona_id}/ask")
def ask_persona(persona_id: str, req: AskRequest, student_id: str = Depends(get_current_student)):
    persona = supabase.table("personas").select("*").eq("id", persona_id).single().execute().data
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    if persona["status"] != "ready":
        raise HTTPException(status_code=409, detail=f"Persona is not ready (status: {persona['status']})")

    if req.conversation_id:
        convo = supabase.table("conversations").select("*") \
            .eq("id", req.conversation_id).eq("student_id", student_id).single().execute().data
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation_id = convo["id"]
    else:
        convo = supabase.table("conversations").insert({
            "persona_id": persona_id, "student_id": student_id
        }).execute().data[0]
        conversation_id = convo["id"]

    history = supabase.table("messages").select("role, content") \
        .eq("conversation_id", conversation_id) \
        .order("created_at", desc=True).limit(10).execute().data
    history.reverse()
    history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)

    answer = call_gemini_with_fallback(_build_persona_prompt(persona, req.question, history_text))

    supabase.table("messages").insert([
        {"conversation_id": conversation_id, "role": "student", "content": req.question},
        {"conversation_id": conversation_id, "role": "tutor", "content": answer},
    ]).execute()

    return {"conversation_id": conversation_id, "answer": answer}


def _build_persona_prompt(persona: dict, question: str, history_text: str) -> str:
    # Retrieval-based, not a full dump of persona['solved_questions'] — see
    # persona_rag.py. This field can be huge for a persona built from a
    # full course, and was previously resent in full on every single
    # question; only the records relevant to THIS question are included now.
    relevant_solved_questions = retrieve_relevant_solved_questions(persona, question)
    return f"""
You are {persona['name']}, a teacher. Answer the student's question in your
own authentic teaching style, using the reference material below where
relevant. Stay in character as this teacher.

TEACHING STYLE & TOPICS YOU'VE COVERED:
{persona['teaching_style']}

YOUR PROBLEM-SOLVING APPROACH:
{persona['problem_solving_approach']}

REFERENCE MATERIAL (questions you've solved before that are closest to this one,
for consistent method/notation):
{relevant_solved_questions or "(no closely matching solved example on file — fall back to your general teaching style above.)"}

RECENT CONVERSATION:
{history_text}

STUDENT'S NEW QUESTION:
{question}

Answer as {persona['name']}, in their teaching style, clearly and helpfully.
""".strip()


# ---------------------------------------------------------------------------
# Public, stateless chat — no login needed (AI Tutor site). The frontend
# keeps conversation history in the browser and resends it each turn,
# ChatGPT-style — nothing is written to the database, so there's no
# per-user history to manage server-side.
# ---------------------------------------------------------------------------
class ChatTurn(BaseModel):
    role: str  # "student" or "tutor"
    content: str


class AskAnonymousRequest(BaseModel):
    question: str
    history: list[ChatTurn] = []


@app.post("/personas/{persona_id}/ask-anonymous")
def ask_persona_anonymous(persona_id: str, req: AskAnonymousRequest):
    persona = supabase.table("personas").select("*").eq("id", persona_id).single().execute().data
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    if persona["status"] != "ready":
        raise HTTPException(status_code=409, detail=f"Persona is not ready (status: {persona['status']})")

    history_text = "\n".join(f"{m.role.upper()}: {m.content}" for m in req.history[-10:])
    answer = call_gemini_with_fallback(_build_persona_prompt(persona, req.question, history_text))
    return {"answer": answer}


# ---------------------------------------------------------------------------
# Public "request a new tutor" — sends you an email instead of letting
# random visitors trigger a real (costly, 10-15+ min) ingestion job.
# ---------------------------------------------------------------------------
class PersonaRequestRequest(BaseModel):
    requested_by_name: str
    requested_by_email: str
    teacher_name: str
    youtube_url: str | None = None
    notes: str | None = None


@app.post("/personas/request", status_code=201)
def request_persona(req: PersonaRequestRequest):
    supabase.table("persona_requests").insert({
        "requested_by_name": req.requested_by_name,
        "requested_by_email": req.requested_by_email,
        "teacher_name": req.teacher_name,
        "youtube_url": req.youtube_url,
        "notes": req.notes,
    }).execute()

    send_notification_email(
        subject=f"New AI tutor request: {req.teacher_name}",
        body=f"""New AI tutor request from your site:

Requested by: {req.requested_by_name} <{req.requested_by_email}>
Teacher name: {req.teacher_name}
YouTube URL: {req.youtube_url or 'not provided'}
Notes: {req.notes or 'none'}
""",
    )
    return {"message": "Request received — thanks! We'll be in touch."}