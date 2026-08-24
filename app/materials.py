import io
import tempfile
from datetime import datetime, timezone
from typing import Optional

import pdfplumber
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel

from app.supabase_client import supabase
from app.auth import get_current_student
from app.ingestion import fetch_transcript, transcribe_uploaded_audio
from app.llm_providers import (
    embed_text, embed_texts_batch,
    call_study_gemini_with_fallback, call_study_gemini_vision_with_fallback,
    map_reduce_analyze,
)
from app.study import SECTION_INSTRUCTIONS

router = APIRouter(prefix="/materials", tags=["materials"])

# ---------------------------------------------------------------------------
# "Materials" are student-owned, self-service knowledge sources — distinct
# from app/main.py's `personas` (teacher-built, from a single lecture video).
# A student can build one from a YouTube link, an uploaded file (downloaded
# lecture video/audio, a PDF, or a photo of notes), or pasted text. Each is
# chunked + embedded for Document Chat (RAG in /{id}/chat) and can also
# produce Smart Summaries & Notes (/{id}/summary) reusing study.py's section
# prompts, grounded in the material's own extracted text instead of a
# persona's teaching style.
# ---------------------------------------------------------------------------

MATERIALS_BUCKET = "materials"  # create this bucket in Supabase Storage first (not public)

ALLOWED_PDF = ("application/pdf",)
ALLOWED_IMAGE = ("image/jpeg", "image/png", "image/webp")
ALLOWED_AUDIO_VIDEO = (
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4",
    "audio/x-m4a", "audio/m4a", "audio/ogg", "audio/webm",
    "video/mp4", "video/webm", "video/quicktime", "video/x-matroska",
)

OCR_PROMPT = """
Transcribe ALL text visible in this image as accurately as possible.
Preserve structure (headings, bullet points, numbered steps, equations)
using plain text/markdown. If it's handwritten, do your best to read it
faithfully — mark only genuinely illegible spots with [unclear]. Return
ONLY the transcribed text, nothing else (no preamble, no commentary).
""".strip()

RAG_CHAT_PROMPT = """
You are a study assistant helping a student understand their own material,
titled "{title}". Answer using ONLY the excerpts below — this is the
student's actual notes/lecture, not general knowledge. If the excerpts
don't contain the answer, say so plainly rather than guessing or filling
the gap from outside knowledge. When you use a fact from an excerpt,
reference it inline like [Source 1].

EXCERPTS FROM THE STUDENT'S MATERIAL:
{context_block}

RECENT CONVERSATION:
{history_text}

STUDENT'S QUESTION:
{question}
""".strip()


class CreateMaterialTextRequest(BaseModel):
    title: str
    pasted_text: str


class ChatRequest(BaseModel):
    question: str


class SummaryRequest(BaseModel):
    sections: list[str] = ["summary", "key_points", "flashcards"]


# ---------------------------------------------------------------------------
# Text extraction + chunking helpers
# ---------------------------------------------------------------------------
def _chunk_with_overlap(text: str, chunk_chars: int = 1200, overlap_chars: int = 150) -> list:
    """
    Splits text into overlapping ~1200-char chunks (breaking on whitespace
    where possible) for embedding. The overlap keeps a fact that straddles a
    chunk boundary retrievable from either side. This is separate from
    llm_providers.chunk_text, which chunks for map-reduce summarization
    (no overlap needed there — every chunk gets folded into one final
    answer either way).
    """
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        if end < n:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def _extract_pdf_pages(file_bytes: bytes) -> list:
    """Returns [(location_label, page_text), ...] for each non-empty page."""
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((f"Page {i}", text))
    return pages


def _store_chunks(material_id: str, labeled_texts: list):
    """labeled_texts: [(location_label_or_None, chunk_text), ...] in order.
    Embeds everything (one batch call, chunked internally) and inserts rows."""
    if not labeled_texts:
        raise RuntimeError(
            "No text could be extracted from this material — nothing to chunk/embed."
        )
    texts = [t for _, t in labeled_texts]
    embeddings = embed_texts_batch(texts, task_type="RETRIEVAL_DOCUMENT")
    rows = [
        {
            "material_id": material_id,
            "chunk_index": i,
            "content": text,
            "location_label": label,
            "embedding": embedding,
        }
        for i, ((label, text), embedding) in enumerate(zip(labeled_texts, embeddings))
    ]
    insert_batch = 50
    for i in range(0, len(rows), insert_batch):
        supabase.table("material_chunks").insert(rows[i:i + insert_batch]).execute()


# ---------------------------------------------------------------------------
# Background ingestion job — mirrors main.py's persona ingestion pattern:
# insert a `processing` row immediately, do the slow work in the
# background, then flip status to `ready`/`failed`.
# ---------------------------------------------------------------------------
def _run_material_ingestion(material_id: str, source_type: str, payload: dict):
    try:
        labeled_texts = []

        if source_type == "pasted_text":
            labeled_texts = [(None, c) for c in _chunk_with_overlap(payload["text"])]

        elif source_type == "youtube":
            with tempfile.TemporaryDirectory() as work_dir:
                transcript = fetch_transcript(payload["youtube_url"], work_dir)
            labeled_texts = [(None, c) for c in _chunk_with_overlap(transcript)]

        elif source_type in ("upload_audio", "upload_video"):
            # transcribe_uploaded_audio's ffmpeg step extracts the audio
            # track regardless of container, so video files work the same
            # way as pure audio files here.
            with tempfile.TemporaryDirectory() as work_dir:
                transcript = transcribe_uploaded_audio(payload["file_bytes"], payload["filename"], work_dir)
            labeled_texts = [(None, c) for c in _chunk_with_overlap(transcript)]

        elif source_type == "upload_pdf":
            pages = _extract_pdf_pages(payload["file_bytes"])
            if not pages:
                raise RuntimeError(
                    "Couldn't extract selectable text from this PDF — it may be a scanned "
                    "image PDF. Try re-uploading the page images instead so they can be OCR'd."
                )
            for label, page_text in pages:
                labeled_texts.extend((label, c) for c in _chunk_with_overlap(page_text))

        elif source_type == "upload_image":
            raw = call_study_gemini_vision_with_fallback(OCR_PROMPT, payload["file_bytes"], payload["mime_type"])
            labeled_texts = [(None, c) for c in _chunk_with_overlap(raw)]

        else:
            raise RuntimeError(f"Unknown source_type: {source_type}")

        _store_chunks(material_id, labeled_texts)

        supabase.table("materials").update({
            "status": "ready",
            "chunk_count": len(labeled_texts),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", material_id).execute()

    except Exception as e:
        supabase.table("materials").update({
            "status": "failed",
            "error_message": str(e),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", material_id).execute()


def _create_material_row(student_id: str, title: str, source_type: str, source_ref: Optional[str] = None) -> dict:
    title = title.strip() or "Untitled material"
    return supabase.table("materials").insert({
        "student_id": student_id,
        "title": title,
        "source_type": source_type,
        "source_ref": source_ref,
        "status": "processing",
    }).execute().data[0]


def _get_owned_material(material_id: str, student_id: str) -> dict:
    material = supabase.table("materials").select("*") \
        .eq("id", material_id).eq("student_id", student_id).single().execute().data
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    return material


def _get_owned_ready_material(material_id: str, student_id: str) -> dict:
    material = _get_owned_material(material_id, student_id)
    if material["status"] != "ready":
        raise HTTPException(status_code=409, detail=f"Material is not ready yet (status: {material['status']})")
    return material


# ---------------------------------------------------------------------------
# Create a material — one endpoint per input shape, since FastAPI/pydantic
# don't mix well with "exactly one of these three, one of which is a file
# upload" in a single body. All three kick off the same background pipeline.
# ---------------------------------------------------------------------------
@router.post("/from-youtube", status_code=202)
def create_material_from_youtube(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    youtube_url: str = Form(...),
    student_id: str = Depends(get_current_student),
):
    material = _create_material_row(student_id, title, "youtube", source_ref=youtube_url)
    background_tasks.add_task(_run_material_ingestion, material["id"], "youtube", {"youtube_url": youtube_url})
    return {"id": material["id"], "status": "processing"}


@router.post("/from-text", status_code=202)
def create_material_from_text(
    req: CreateMaterialTextRequest,
    background_tasks: BackgroundTasks,
    student_id: str = Depends(get_current_student),
):
    if not req.pasted_text.strip():
        raise HTTPException(status_code=400, detail="Pasted text can't be empty")
    material = _create_material_row(student_id, req.title, "pasted_text")
    background_tasks.add_task(_run_material_ingestion, material["id"], "pasted_text", {"text": req.pasted_text})
    return {"id": material["id"], "status": "processing"}


@router.post("/from-file", status_code=202)
async def create_material_from_file(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    file: UploadFile = File(...),
    student_id: str = Depends(get_current_student),
):
    content_type = file.content_type
    if content_type in ALLOWED_PDF:
        source_type = "upload_pdf"
    elif content_type in ALLOWED_IMAGE:
        source_type = "upload_image"
    elif content_type in ALLOWED_AUDIO_VIDEO:
        source_type = "upload_video" if content_type.startswith("video/") else "upload_audio"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{content_type}' — upload a PDF, a photo of notes, "
                   f"or an audio/video lecture file.",
        )

    file_bytes = await file.read()

    # Only keep the original file in Storage for PDFs/images — they're small
    # and worth letting the student re-view later. Audio/video lecture files
    # are read exactly once (transcription, right below) and can be large
    # enough to hit Supabase Storage's per-file size limit for no benefit,
    # so we skip persisting them; `source_ref` just stays the filename.
    if source_type in ("upload_pdf", "upload_image"):
        storage_path = f"{student_id}/{datetime.now(timezone.utc).timestamp()}_{file.filename}"
        supabase.storage.from_(MATERIALS_BUCKET).upload(
            storage_path, file_bytes, {"content-type": content_type, "upsert": "true"}
        )
        source_ref = storage_path
    else:
        source_ref = file.filename

    material = _create_material_row(student_id, title, source_type, source_ref=source_ref)
    background_tasks.add_task(
        _run_material_ingestion, material["id"], source_type,
        {"file_bytes": file_bytes, "filename": file.filename, "mime_type": content_type},
    )
    return {"id": material["id"], "status": "processing"}


@router.get("")
def list_materials(student_id: str = Depends(get_current_student)):
    return supabase.table("materials").select(
        "id, title, source_type, status, chunk_count, created_at"
    ).eq("student_id", student_id).order("created_at", desc=True).execute().data


@router.get("/{material_id}")
def get_material(material_id: str, student_id: str = Depends(get_current_student)):
    return _get_owned_material(material_id, student_id)


@router.delete("/{material_id}", status_code=204)
def delete_material(material_id: str, student_id: str = Depends(get_current_student)):
    material = _get_owned_material(material_id, student_id)
    if material["source_type"] in ("upload_pdf", "upload_image") and material.get("source_ref"):
        try:
            supabase.storage.from_(MATERIALS_BUCKET).remove([material["source_ref"]])
        except Exception:
            pass  # best-effort — don't block deleting the record on storage cleanup
    # material_chunks and material_messages cascade-delete via FK.
    supabase.table("materials").delete().eq("id", material_id).execute()
    return None


# ---------------------------------------------------------------------------
# Document Chat (RAG) — grounded in this material's chunks only, retrieved
# by embedding similarity, so answers stay tied to what the student actually
# uploaded instead of drifting into the model's general knowledge.
# ---------------------------------------------------------------------------
@router.post("/{material_id}/chat")
def chat_with_material(material_id: str, req: ChatRequest, student_id: str = Depends(get_current_student)):
    material = _get_owned_ready_material(material_id, student_id)

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question can't be empty")

    query_embedding = embed_text(req.question, task_type="RETRIEVAL_QUERY")
    matches = supabase.rpc("match_material_chunks", {
        "p_material_id": material_id,
        "p_student_id": student_id,
        "p_query_embedding": query_embedding,
        "p_match_count": 6,
    }).execute().data or []

    if matches:
        context_block = "\n\n".join(
            f"[Source {i + 1}{' — ' + m['location_label'] if m.get('location_label') else ''}]\n{m['content']}"
            for i, m in enumerate(matches)
        )
    else:
        context_block = "(No relevant passages were found in this material for this question.)"

    history = supabase.table("material_messages").select("role, content") \
        .eq("material_id", material_id).eq("student_id", student_id) \
        .order("created_at", desc=True).limit(10).execute().data
    history.reverse()
    history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)

    prompt = RAG_CHAT_PROMPT.format(
        title=material["title"], context_block=context_block,
        history_text=history_text, question=req.question,
    )
    answer = call_study_gemini_with_fallback(prompt)

    sources = [
        {"label": m.get("location_label") or f"Excerpt {i + 1}", "snippet": m["content"][:280]}
        for i, m in enumerate(matches)
    ]

    supabase.table("material_messages").insert([
        {"material_id": material_id, "student_id": student_id, "role": "student", "content": req.question},
        {"material_id": material_id, "student_id": student_id, "role": "assistant", "content": answer,
         "sources": sources},
    ]).execute()

    return {"answer": answer, "sources": sources}


@router.get("/{material_id}/messages")
def get_material_messages(material_id: str, student_id: str = Depends(get_current_student)):
    _get_owned_material(material_id, student_id)  # 404s if not this student's
    return supabase.table("material_messages").select("id, role, content, sources, created_at") \
        .eq("material_id", material_id).eq("student_id", student_id) \
        .order("created_at").execute().data


# ---------------------------------------------------------------------------
# Smart Summaries & Notes — reuses study.py's SECTION_INSTRUCTIONS (summary,
# key_points, flashcards, mind_map, ...) and llm_providers' map-reduce
# pattern, but grounds them in the material's own full extracted text
# instead of a persona's teaching style.
# ---------------------------------------------------------------------------
@router.post("/{material_id}/summary")
def generate_material_summary(
    material_id: str, req: SummaryRequest, student_id: str = Depends(get_current_student)
):
    _get_owned_ready_material(material_id, student_id)

    unknown = set(req.sections) - set(SECTION_INSTRUCTIONS)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown section(s): {', '.join(sorted(unknown))}")
    if not req.sections:
        raise HTTPException(status_code=400, detail="Request at least one section")

    chunks = supabase.table("material_chunks").select("content") \
        .eq("material_id", material_id).order("chunk_index").execute().data
    full_text = "\n\n".join(c["content"] for c in chunks)
    if not full_text.strip():
        raise HTTPException(status_code=409, detail="This material has no extracted text yet")

    sections = {
        section: map_reduce_analyze(call_study_gemini_with_fallback, SECTION_INSTRUCTIONS[section], full_text)
        for section in req.sections
    }

    supabase.table("materials").update({
        "summary_sections": sections,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", material_id).execute()

    return {"sections": sections}


@router.get("/{material_id}/summary")
def get_material_summary(material_id: str, student_id: str = Depends(get_current_student)):
    material = _get_owned_material(material_id, student_id)
    return {"sections": material.get("summary_sections") or {}}
