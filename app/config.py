import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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
