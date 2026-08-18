import os
import glob
import subprocess
import yt_dlp
from groq import Groq

from app.config import GROQ_API_KEY, GROQ_WHISPER_MODEL
from app.llm_providers import call_groq, call_openrouter, call_gemini_with_fallback, analyze_section_with_fallback

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


def _download_audio(youtube_url: str, out_name: str) -> str:
    ydl_opts = {"format": "bestaudio/best", "outtmpl": f"{out_name}.%(ext)s", "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        return ydl.prepare_filename(info)


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


def _transcribe_chunk(audio_path: str) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=f, model=GROQ_WHISPER_MODEL, response_format="text"
        )
    return result if isinstance(result, str) else result.text


def fetch_transcript(youtube_url: str, work_dir: str) -> str:
    audio_name = os.path.join(work_dir, "audio")
    audio_path = _download_audio(youtube_url, audio_name)
    chunks = _split_into_chunks(audio_path, chunk_seconds=900, out_prefix=audio_name)
    return "\n".join(_transcribe_chunk(c) for c in chunks)


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
