# persona_rag.py
#
# RAG-style retrieval over a persona's reference material (solved_questions),
# instead of dumping the entire blob into every prompt.
#
# WHY THIS EXISTS
# ----------------
# study.py's /study/ask can generate up to ~9 sections per question, each as
# its own independent LLM call (see _run_sections_concurrently) — and every
# one of those calls previously carried persona['solved_questions'] in full
# inside its system context (see study.py's old _build_system_context).
# For a persona built from a full lecture/course, that field can be tens of
# thousands of characters, and it was being resent, verbatim, N times for a
# single student question. main.py's /personas/{id}/ask has the same
# pattern (once per question there, instead of N times, but still every
# single question resends the whole thing).
#
# This mirrors what materials.py already does correctly for Document Chat
# (chunk -> embed -> match_material_chunks -> only relevant excerpts): here
# a persona's solved_questions is chunked into per-topic/per-question
# "records" once (right after ingestion finishes), embedded, and stored in
# persona_chunks (schema_v5). At ask-time, only the handful of records
# relevant to the student's actual question are retrieved and injected —
# everything else about the persona (teaching_style, problem_solving_approach)
# stays as-is since those are short "style" fields, not the bulky part.
#
# Best-effort by design: any failure here (embedding hiccup, RPC not
# migrated yet, persona created before this existed) just means
# retrieve_relevant_solved_questions() falls back to a bounded slice of the
# full text — grounding never disappears, only the "always send everything"
# cost does.

from app.supabase_client import supabase
from app.llm_providers import embed_text, embed_texts_batch
from app.rag_utils import chunk_by_blocks

TOP_K = 6
# Ceiling for the no-match / no-persona_chunks-yet fallback path, so even a
# persona nobody's re-ingested since this feature shipped never blows up a
# prompt to unbounded size the way the old "always send it all" path could.
MAX_FALLBACK_CHARS = 6000


def chunk_and_embed_persona(persona_id: str, solved_questions: str) -> None:
    """
    (Re)builds persona_chunks for one persona from its solved_questions
    text. Safe to call again on re-ingestion — clears old chunks first so
    stale records never linger alongside fresh ones.

    Call this once, right after a persona's profile is built (main.py's
    ingestion jobs) — not per-request. Wrap the call in try/except at the
    call site: this is an optimization, not a correctness requirement, so a
    failure here should never fail persona ingestion itself.
    """
    solved_questions = (solved_questions or "").strip()
    supabase.table("persona_chunks").delete().eq("persona_id", persona_id).execute()
    if not solved_questions:
        return

    chunks = chunk_by_blocks(solved_questions)
    if not chunks:
        return

    embeddings = embed_texts_batch(chunks, task_type="RETRIEVAL_DOCUMENT")
    rows = [
        {
            "persona_id": persona_id,
            "chunk_index": i,
            "section": "solved_questions",
            "content": chunk,
            "embedding": embedding,
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    insert_batch = 50
    for i in range(0, len(rows), insert_batch):
        supabase.table("persona_chunks").insert(rows[i:i + insert_batch]).execute()


def retrieve_relevant_solved_questions(persona: dict, question: str, top_k: int = TOP_K) -> str:
    """
    Returns just the solved-question records relevant to `question`,
    instead of persona['solved_questions'] in full.

    Falls back to a truncated slice of the full text if retrieval finds
    nothing — a persona ingested before persona_chunks existed, a
    transient embedding/RPC error, or a persona with no solved_questions at
    all — so callers never lose reference-material grounding, only the
    "resend everything every time" cost.
    """
    full_text = (persona.get("solved_questions") or "").strip()
    if not full_text or not (question or "").strip():
        return full_text[:MAX_FALLBACK_CHARS]

    try:
        query_embedding = embed_text(question, task_type="RETRIEVAL_QUERY")
        matches = supabase.rpc("match_persona_chunks", {
            "p_persona_id": persona["id"],
            "p_query_embedding": query_embedding,
            "p_match_count": top_k,
        }).execute().data or []
    except Exception:
        matches = []

    if matches:
        # Restore original (roughly reading) order rather than pure
        # similarity rank — an easier passage for the model to follow.
        matches = sorted(matches, key=lambda m: m.get("chunk_index", 0))
        return "\n\n---\n\n".join(m["content"] for m in matches)

    return full_text[:MAX_FALLBACK_CHARS]
