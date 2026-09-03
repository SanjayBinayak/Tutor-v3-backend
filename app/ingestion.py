import io
import os
import glob
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import pymupdf
import yt_dlp
from groq import Groq

from app.config import GROQ_API_KEY, GROQ_WHISPER_MODEL, YTDLP_COOKIES_FILE, YTDLP_COOKIES_BROWSER
from app.llm_providers import (
    call_groq, call_openrouter, call_gemini_with_fallback,
    call_gemini_with_youtube_fallback, call_gemini_video_chunk_with_fallback,
    call_gemini_audio_with_fallback, analyze_section_with_fallback,
    call_gemini_vision_with_fallback,
)

# ---------------------------------------------------------------------------
# PDF ingestion — image extraction only. Every page is rendered to an image
# and read with Gemini vision, never with a text-layer extractor. This is a
# deliberate choice (not just a fallback for scanned pages): PDFs exported
# from slides/handwritten notes often have an unreliable or missing text
# layer, and reading every page as a photo keeps one consistent pipeline
# instead of silently mixing two extraction qualities.
# ---------------------------------------------------------------------------
PDF_PAGE_OCR_PROMPT = """
Transcribe ALL text and content visible on this page as accurately as
possible. Preserve structure (headings, bullet points, numbered steps,
equations, diagram labels) using plain text/markdown. If it's handwritten,
do your best to read it faithfully — mark only genuinely illegible spots
with [unclear]. Return ONLY the transcribed content for this page, nothing
else (no preamble, no commentary).
""".strip()

PDF_OCR_DPI = 200
PDF_OCR_MAX_CONCURRENCY = 5


def _rasterize_pdf_page(file_bytes: bytes, page_index: int, dpi: int = PDF_OCR_DPI) -> bytes:
    """Renders ONE page (0-indexed) of a PDF to PNG bytes."""
    with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
        pix = doc[page_index].get_pixmap(dpi=dpi)
        return pix.tobytes("png")


def extract_pdf_transcript_via_images(file_bytes: bytes) -> str:
    """Turns a PDF into a single "transcript" string, page by page, by
    rendering each page to an image and reading it with Gemini vision —
    no pdfplumber/text-layer extraction anywhere in this path. Runs the
    per-page vision calls concurrently since they're independent."""
    with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
        page_count = doc.page_count

    def _ocr_page(page_index: int) -> tuple:
        try:
            png_bytes = _rasterize_pdf_page(file_bytes, page_index)
            text = call_gemini_vision_with_fallback(PDF_PAGE_OCR_PROMPT, png_bytes, "image/png")
            return page_index, (text or "").strip()
        except Exception:
            # Don't let one bad page (corrupt render, genuinely blank page)
            # sink the whole upload — just skip it and keep the rest.
            return page_index, ""

    pages = {}
    with ThreadPoolExecutor(max_workers=min(PDF_OCR_MAX_CONCURRENCY, max(page_count, 1))) as pool:
        for page_index, text in pool.map(_ocr_page, range(page_count)):
            if text:
                pages[page_index] = text

    return "\n\n".join(f"[Page {i + 1}]\n{pages[i]}" for i in sorted(pages))

TEACHING_STYLE_AND_TOPICS_PROMPT = """
Cover BOTH sections clearly labeled:
## 1. TEACHING STYLE
Tone, pacing, use of examples/analogies, how the teacher engages students,
verbal habits, any repeated phrases or techniques.
## 2. TOPICS COVERED
A clear ordered list of every topic/subtopic the teacher went through, with
a short description of each.
"""

PROBLEM_SOLVING_APPROACH_PROMPT = """
## 3. PROBLEM-SOLVING APPROACH
Describe the general method the teacher uses to approach problems (e.g.
first principles, formula-first, diagram-first, common shortcuts taught).
"""

SOLVED_QUESTIONS_PROMPT = """
## 4. EACH SOLVED QUESTION (in detail, one entry per question)
For every question solved in the lecture, include: the exact question
statement, the step-by-step solution as explained, key formulas/tricks
highlighted, and any mistakes/common errors the teacher flagged.
Be thorough and do not skip any question.
"""

# Used by build_persona_profile_from_youtube_chunked's map step — one call
# per ~20-min video segment, extracting raw notes for ALL FOUR sections at
# once (cheaper than one call per section per chunk). The reduce step below
# then polishes each section separately from the combined notes.
VIDEO_CHUNK_NOTES_PROMPT = """
You are extracting notes from ONE SEGMENT of a longer lecture video — this
segment only. Do not assume or invent anything about what happens outside
this clip. Extract raw facts/details in short note form (no polished
writing yet), covering all of the following that apply to this segment:

1. TEACHING STYLE observed here (tone, pacing, examples/analogies, verbal
   habits, engagement techniques).
2. TOPICS covered here.
3. PROBLEM-SOLVING APPROACH shown here (method used, shortcuts taught).
4. ANY QUESTIONS SOLVED here — exact question statement, the full
   step-by-step solution as explained, key formulas/tricks highlighted,
   and any mistakes/common errors flagged. Be thorough — do not skip any
   question solved in this segment.

If a section has nothing relevant in this segment, write "None in this
segment" for that section instead of guessing or padding.
""".strip()


def _download_audio(youtube_url: str, out_name: str) -> str:
    base_opts = {
        # Audio-ONLY download (never the video stream) — this is what makes
        # this "fast processing" in the first place: we never pull the
        # (often 10-50x larger) video track just to throw it away a moment
        # later. On top of that, prefer a stream at/under ~64kbps when one
        # exists: _split_into_chunks re-encodes everything to 16kHz mono
        # 32kbps anyway for Whisper, so a hi-fi 256kbps download buys us
        # nothing but slower downloads — this trims that for slow/metered
        # connections while still falling back to bestaudio if no
        # low-bitrate stream is offered for this video.
        "format": "bestaudio[abr<=64]/bestaudio/best",
        "outtmpl": f"{out_name}.%(ext)s",
        "quiet": True,
    }
    # Cookies: prefer a live browser session if configured, else fall back
    # to an exported cookies file — matches the ONE-of-these setup
    # documented in config.py.
    if YTDLP_COOKIES_BROWSER:
        base_opts["cookiesfrombrowser"] = (YTDLP_COOKIES_BROWSER,)
    elif YTDLP_COOKIES_FILE and os.path.isfile(YTDLP_COOKIES_FILE):
        base_opts["cookiefile"] = YTDLP_COOKIES_FILE

    # YouTube is currently (as of Aug 2026) in an active back-and-forth with
    # yt-dlp — clients that work one day 403 the next (see yt-dlp#17395,
    # #17456, #17389/#17405, all open). Some users report the *same* client
    # succeeding on a retry a few seconds later, so we try several clients,
    # each with a couple of retries, before giving up. Keep `pip install -U
    # yt-dlp` current — this list may need revisiting as things shift.
    client_attempts = [
        ["default", "web_embedded"],
        ["android_vr"],
        ["mweb"],
        ["ios"],
        ["tv"],
    ]
    retries_per_client = 2
    retry_delay_seconds = 8

    last_error = None
    for clients in client_attempts:
        opts = {**base_opts, "extractor_args": {"youtube": {"player_client": clients}}}
        for attempt in range(1, retries_per_client + 1):
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=True)
                    return ydl.prepare_filename(info)
            except Exception as e:
                last_error = e
                if attempt < retries_per_client:
                    time.sleep(retry_delay_seconds)
                continue

    raise RuntimeError(
        "All yt-dlp player-client attempts failed — this is very likely YouTube's "
        f"current anti-automation blocking (see https://github.com/yt-dlp/yt-dlp/issues, "
        f"search '403 Forbidden'), not a bug in this app. Last error: {last_error}"
    )


def _split_into_chunks(input_path: str, chunk_seconds: int, out_prefix: str) -> list:
    out_pattern = f"{out_prefix}_%03d.ogg"
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-b:a", "32k",
         "-f", "segment", "-segment_time", str(chunk_seconds), out_pattern],
        check=True, capture_output=True,
    )
    chunks = sorted(glob.glob(f"{out_prefix}_*.ogg"))
    if not chunks:
        raise RuntimeError("ffmpeg produced no audio chunks — is ffmpeg installed?")
    return chunks


AUDIO_TRANSCRIBE_PROMPT = (
    "Transcribe this audio exactly as spoken, word for word, in the language "
    "it's spoken in. Return ONLY the transcript text — no preamble, no "
    "timestamps, no speaker labels unless multiple speakers are clearly "
    "distinguishable and it actually helps to mark them."
)

# Chunks are always written as mono 16kHz .ogg by _split_into_chunks.
_CHUNK_MIME_TYPE = "audio/ogg"

# Groq's free Whisper tier (as of 2026) is generous for this use case — 2,000
# requests/day and ~8hrs of audio/day — but it IS a shared, per-organization
# cap: unlike Gemini, spinning up extra Groq keys does not multiply that
# quota, so there's no "rotate Groq keys" fallback worth building. What we
# CAN do is fail over to a different provider entirely when Groq is
# unavailable or that daily cap is hit.
GROQ_TRANSIENT_ERROR_TOKENS = (
    "429", "500", "502", "503", "504", "rate limit", "rate_limit",
    "quota", "timeout", "timed out",
)


def _transcribe_chunk(audio_path: str) -> str:
    """
    Transcribes one audio chunk. Groq Whisper is tried first (purpose-built
    for speech-to-text, and fast) with a couple of quick retries on
    transient errors; if Groq is down OR its free-tier daily quota has run
    out, this automatically falls back to Gemini's native audio
    understanding (call_gemini_audio_with_fallback) — which itself tries
    every model in GEMINI_FALLBACK_MODELS, and every configured
    GEMINI_API_KEY* in turn, before finally giving up. So a single chunk
    only fails if BOTH Groq AND every Gemini model/key combination fail.
    """
    groq_error = None
    for attempt in range(2):
        try:
            client = Groq(api_key=GROQ_API_KEY)
            with open(audio_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    file=f, model=GROQ_WHISPER_MODEL, response_format="text"
                )
            return result if isinstance(result, str) else result.text
        except Exception as e:
            groq_error = e
            if attempt == 0 and any(tok in str(e).lower() for tok in GROQ_TRANSIENT_ERROR_TOKENS):
                time.sleep(2)
                continue
            break

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        return call_gemini_audio_with_fallback(AUDIO_TRANSCRIBE_PROMPT, audio_bytes, _CHUNK_MIME_TYPE)
    except Exception as gemini_error:
        raise RuntimeError(
            f"Transcription failed on both providers for {os.path.basename(audio_path)}. "
            f"Groq: {groq_error}. Gemini: {gemini_error}"
        )


def _transcribe_chunks(chunks: list, max_concurrency: int = 4) -> str:
    """Transcribes every chunk of one lecture. Chunks are independent of
    each other, so they're sent concurrently (bounded by max_concurrency —
    comfortably under Groq's free-tier 20 RPM for Whisper) instead of one
    at a time; a 1-hour lecture (~4 chunks at 15min each) that used to
    transcribe serially now finishes in roughly the time of one chunk."""
    if len(chunks) == 1:
        return _transcribe_chunk(chunks[0])
    with ThreadPoolExecutor(max_workers=min(max_concurrency, len(chunks))) as pool:
        return "\n".join(pool.map(_transcribe_chunk, chunks))


def fetch_transcript(youtube_url: str, work_dir: str) -> str:
    """
    Primary path for turning a YouTube lecture into text: downloads the
    AUDIO TRACK ONLY (never the video) via _download_audio — this is the
    fast-processing step, since audio is a small fraction of a video's
    size/bandwidth — then splits it into ~15-min chunks and transcribes
    each one (see _transcribe_chunks/_transcribe_chunk for the
    Groq-then-Gemini fallback chain).
    """
    audio_name = os.path.join(work_dir, "audio")
    audio_path = _download_audio(youtube_url, audio_name)
    chunks = _split_into_chunks(audio_path, chunk_seconds=900, out_prefix=audio_name)
    return _transcribe_chunks(chunks)


def transcribe_uploaded_audio(file_bytes: bytes, filename: str, work_dir: str) -> str:
    """
    Same chunk+transcribe pipeline as fetch_transcript(), but starting from
    audio/video bytes already on hand (a student's upload, or audio
    manually downloaded through a browser) instead of having yt-dlp fetch
    them. No automated download happens here at all — for a video file,
    ffmpeg pulls just the audio track out of it in _split_into_chunks
    (the video frames are never decoded/processed), so this is just as
    "audio-first" as the YouTube path, it just starts one step later.
    """
    ext = os.path.splitext(filename or "")[1] or ".mp3"
    audio_name = os.path.join(work_dir, "audio")
    input_path = f"{audio_name}{ext}"
    with open(input_path, "wb") as f:
        f.write(file_bytes)

    chunks = _split_into_chunks(input_path, chunk_seconds=900, out_prefix=audio_name)
    return _transcribe_chunks(chunks)


def build_persona_profile(transcript: str) -> dict:
    """Runs the same 3-provider analysis as the standalone script, returns
    a dict ready to insert into the `personas` table."""
    teaching_style_and_topics = analyze_section_with_fallback(
        TEACHING_STYLE_AND_TOPICS_PROMPT, transcript, call_groq, "Groq"
    )
    problem_solving_approach = analyze_section_with_fallback(
        PROBLEM_SOLVING_APPROACH_PROMPT, transcript, call_openrouter, "OpenRouter"
    )
    solved_questions = call_gemini_with_fallback(
        f"You are analyzing a recorded lecture transcript.\n\n{SOLVED_QUESTIONS_PROMPT}\n\nTranscript:\n\n{transcript}"
    )
    return {
        "teaching_style": teaching_style_and_topics,  # contains both style + topics
        "topics_covered": None,  # merged into teaching_style_and_topics above; kept for schema compatibility
        "problem_solving_approach": problem_solving_approach,
        "solved_questions": solved_questions,
    }


def build_persona_profile_from_youtube(youtube_url: str) -> dict:
    """
    Primary ingestion path: skips download + transcription entirely. Gemini
    fetches and watches/listens to the YouTube video directly, so this needs
    no yt-dlp, no ffmpeg, no Groq Whisper — and sidesteps YouTube's
    anti-automation blocking of download tools altogether.

    Trade-offs vs. the transcript-based build_persona_profile():
    - Only Gemini is used (no Groq/OpenRouter split) — one provider, so if
      Gemini itself is down there's no cross-provider fallback for this path
    - Video must be public; videos over ~1hr need MEDIA_RESOLUTION_LOW
      (set in call_gemini_with_youtube_fallback), which trades some visual
      fidelity for fitting longer lectures in context
    """
    teaching_style_and_topics = call_gemini_with_youtube_fallback(
        f"You are analyzing a recorded lecture video.\n\n{TEACHING_STYLE_AND_TOPICS_PROMPT}",
        youtube_url,
    )
    problem_solving_approach = call_gemini_with_youtube_fallback(
        f"You are analyzing a recorded lecture video.\n\n{PROBLEM_SOLVING_APPROACH_PROMPT}",
        youtube_url,
    )
    solved_questions = call_gemini_with_youtube_fallback(
        f"You are analyzing a recorded lecture video.\n\n{SOLVED_QUESTIONS_PROMPT}",
        youtube_url,
    )
    return {
        "teaching_style": teaching_style_and_topics,
        "topics_covered": None,
        "problem_solving_approach": problem_solving_approach,
        "solved_questions": solved_questions,
    }


VIDEO_CHUNK_SECONDS = 20 * 60  # ~20 min/chunk — comfortably under the 1hr
                                 # default-media-resolution limit per request
VIDEO_CHUNK_DELAY_SECONDS = 5   # small gap between chunk calls, same idea as
                                 # map_reduce_analyze's delay_between_calls


def _get_video_duration_seconds(youtube_url: str) -> int:
    """
    Lightweight metadata-only lookup (download=False) — just asks for the
    video's length, doesn't fetch the actual media stream. This is a much
    smaller, less suspicious request than a real download, so it's far less
    likely to hit the same 403/bot-detection blocking as _download_audio.
    """
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        duration = info.get("duration")
        if not duration:
            raise RuntimeError("Could not determine video duration from YouTube metadata.")
        return int(duration)


def _video_chunk_ranges(duration_seconds: int, chunk_seconds: int = VIDEO_CHUNK_SECONDS) -> list:
    ranges = []
    start = 0
    while start < duration_seconds:
        end = min(start + chunk_seconds, duration_seconds)
        ranges.append((start, end))
        start = end
    return ranges


def _reduce_video_notes(section_instructions: str, combined_notes: str, thorough: bool = False) -> str:
    quality_note = (
        "ONE thorough, well-organized, de-duplicated final answer for this "
        "section — do not skip any solved question mentioned in the notes"
        if thorough else
        "ONE polished, well-structured, de-duplicated final answer for this section"
    )
    return call_gemini_with_fallback(
        "Below are raw notes extracted from consecutive segments (in order) of "
        f"one long lecture video. Combine them into {quality_note}:\n\n"
        f"{section_instructions}\n\nNotes from all segments:\n\n{combined_notes}"
    )


def build_persona_profile_from_youtube_chunked(youtube_url: str) -> dict:
    """
    Preferred over build_persona_profile_from_youtube() for any video —
    splits it into ~20-min clipped segments (via video_metadata start/end
    offsets) instead of one giant request. Benefits over the single-shot
    version:
    - Each chunk stays under the 1hr default-media-resolution limit, so we
      never need MEDIA_RESOLUTION_LOW's reduced visual fidelity
    - No ~3hr ceiling — works for videos of any length
    - Shorter individual requests, which have been reported as more
      reliable than single multi-hour video requests

    Known caveat (Aug 2026): see call_gemini_video_chunk_with_fallback's
    docstring — YouTube-URL clipping currently only clips video frames, not
    audio, on Gemini's side. Doesn't break this, just some redundant audio
    reprocessing per chunk until Google ships a fix.
    """
    duration = _get_video_duration_seconds(youtube_url)
    chunk_ranges = _video_chunk_ranges(duration)

    per_chunk_notes = []
    for i, (start, end) in enumerate(chunk_ranges, start=1):
        notes = call_gemini_video_chunk_with_fallback(VIDEO_CHUNK_NOTES_PROMPT, youtube_url, start, end)
        per_chunk_notes.append(f"--- Segment {i}/{len(chunk_ranges)} ({start}s-{end}s) ---\n{notes}")
        if i < len(chunk_ranges):
            time.sleep(VIDEO_CHUNK_DELAY_SECONDS)

    combined_notes = "\n\n".join(per_chunk_notes)

    return {
        "teaching_style": _reduce_video_notes(TEACHING_STYLE_AND_TOPICS_PROMPT, combined_notes),
        "topics_covered": None,
        "problem_solving_approach": _reduce_video_notes(PROBLEM_SOLVING_APPROACH_PROMPT, combined_notes),
        "solved_questions": _reduce_video_notes(SOLVED_QUESTIONS_PROMPT, combined_notes, thorough=True),
    }