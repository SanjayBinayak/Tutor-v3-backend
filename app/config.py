# config.py - Complete file with correct embedding model names

import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Separate key for the study-tool endpoints (notes analysis / explain-a-concept
# tabs, app/study.py). Kept independent from GEMINI_API_KEY used by the
# persona/tutor chat system so usage and billing/quota don't share one key.
STUDY_GEMINI_API_KEY = os.getenv("STUDY_GEMINI_API_KEY")

# --- Optional second Gemini key per pool, for automatic key rotation -------
# Gemini's free tier is metered PER GOOGLE CLOUD PROJECT, not per API key —
# so a second key from a second Google account genuinely gets its own
# separate free daily quota (unlike Groq, whose free-tier limits are
# per-organization and are NOT multiplied by adding more keys — see
# ingestion.py's docstring on _transcribe_chunk). If GEMINI_API_KEY_2 /
# STUDY_GEMINI_API_KEY_2 are set, llm_providers.py will automatically fall
# over to them once every model on the first key has failed/hit quota.
# Leave unset to keep using a single key (existing behaviour).
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2", "")
STUDY_GEMINI_API_KEY_2 = os.getenv("STUDY_GEMINI_API_KEY_2", "")

GEMINI_API_KEYS = [k for k in (GEMINI_API_KEY, GEMINI_API_KEY_2) if k]
STUDY_GEMINI_API_KEYS = [k for k in (STUDY_GEMINI_API_KEY, STUDY_GEMINI_API_KEY_2) if k]

_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# Email notifications (for "request a new tutor" on the AI Tutor site).
# Plain SMTP so this works with a free Gmail account — no paid email API
# needed. Gmail requires an "App Password", not your normal password:
# https://myaccount.google.com/apppasswords
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL", "")  # where "request a tutor" emails go

GEMINI_FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3-flash-preview",  # verify exact id if this ever 404s
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
]
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"

# Separate from GEMINI_FALLBACK_MODELS in case the study-tool key is on a
# different tier/quota and you want to tune its fallback order independently.
STUDY_GEMINI_FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3-flash-preview",  # verify exact id if this ever 404s
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
]

# Used by app/materials.py (student-created Document Chat / RAG). Runs on
# the same STUDY_GEMINI_API_KEY as the rest of the study-tool endpoints.
# The embedding models available for embedContent are:
# - "embedding-001" (older, 768 dimensions) 
# - "text-embedding-004" (newer, 768 dimensions, better quality)
# NOTE: Do NOT use "models/" prefix - just the model name
STUDY_GEMINI_EMBEDDING_MODEL = os.getenv("STUDY_GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")
# Fallback to a genuinely different model — this previously defaulted to the
# SAME model as STUDY_GEMINI_EMBEDDING_MODEL, so the "fallback" path never
# actually helped if the primary model was down or quota'd out.
STUDY_GEMINI_EMBEDDING_FALLBACK_MODEL = os.getenv("STUDY_GEMINI_EMBEDDING_FALLBACK_MODEL", "text-embedding-004")

# Must match the `vector(...)` column dimension in
# app/migrations/materials_schema.sql — change both together if you tune
# this. 768 (via output_dimensionality) keeps the ivfflat index cheap; the
# embedding model's native size is larger.
MATERIAL_EMBEDDING_DIM = 768

# yt-dlp cookies, to get past YouTube's "Sign in to confirm you're not a
# bot" block. Set ONE of these (not both) in your .env:
#   YTDLP_COOKIES_BROWSER=chrome   -> reads cookies live from that browser
#                                     (local dev only; browser must be closed;
#                                      Windows Chrome hits a DPAPI decrypt
#                                      error currently — use the file option
#                                      below instead if you see that)
#   YTDLP_COOKIES_FILE=path        -> reads an exported cookies.txt file
#                                     (works on a real server too)
# Leave YTDLP_COOKIES_BROWSER unset/empty in .env to use the file path.
# Default file path matches this project's actual folder structure:
# tutor-backend-v3/app/secrets/youtube_cookies.txt
YTDLP_COOKIES_BROWSER = os.getenv("YTDLP_COOKIES_BROWSER", "")
YTDLP_COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE", "./app/secrets/youtube_cookies.txt")