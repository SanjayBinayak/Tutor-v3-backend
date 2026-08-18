import time
import requests
from google import genai
from groq import Groq

from app.config import (
    GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY,
    GEMINI_FALLBACK_MODELS, GROQ_LLM_MODEL,
)


def call_groq(prompt: str) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def pick_openrouter_free_model() -> str:
    resp = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
    resp.raise_for_status()
    for model in resp.json().get("data", []):
        pricing = model.get("pricing", {})
        if pricing.get("prompt") == "0" and pricing.get("completion") == "0" and model["id"].endswith(":free"):
            return model["id"]
    raise RuntimeError("No free OpenRouter model found — check https://openrouter.ai/models?max_price=0")


def call_openrouter(prompt: str) -> str:
    model_id = pick_openrouter_free_model()
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={"model": model_id, "messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_gemini_with_fallback(prompt: str) -> str:
    """Tries each Gemini Flash-tier model in order until one succeeds.
    Large context window — fine to send full transcripts/personas in one call."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    last_error = None
    for model in GEMINI_FALLBACK_MODELS:
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All Gemini fallback models failed. Last error: {last_error}")


def call_gemini_vision_with_fallback(prompt: str, file_bytes: bytes, mime_type: str) -> str:
    """
    Same fallback pattern as call_gemini_with_fallback, but for requests that
    include an image or PDF (e.g. a photo of homework) alongside text.
    """
    from google.genai import types as genai_types
    client = genai.Client(api_key=GEMINI_API_KEY)
    last_error = None
    for model in GEMINI_FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    genai_types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                    prompt,
                ],
            )
            return response.text
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All Gemini fallback models failed. Last error: {last_error}")


def chunk_text(text: str, max_chars: int = 16000) -> list:
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def map_reduce_analyze(call_fn, section_instructions: str, transcript: str,
                        chunk_chars: int = 16000, delay_between_calls: int = 20) -> str:
    chunks = chunk_text(transcript, chunk_chars)
    if len(chunks) == 1:
        return call_fn(build_prompt(section_instructions, transcript))

    partial_notes = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_prompt = (
            f"You are extracting notes from PART {i} of {len(chunks)} of a longer "
            "lecture transcript. Extract only the raw facts/details relevant to the "
            "section below — short note form, no polished writing yet:\n\n"
            + section_instructions + "\n\nTranscript part:\n\n" + chunk
        )
        partial_notes.append(call_fn(chunk_prompt))
        if i < len(chunks):
            time.sleep(delay_between_calls)

    combined_notes = "\n\n---\n\n".join(partial_notes)
    reduce_prompt = (
        "Below are notes extracted from consecutive parts (in order) of one long "
        "lecture transcript. Combine them into ONE polished, well-structured, "
        "de-duplicated final answer for this section:\n\n"
        + section_instructions + "\n\nNotes from all parts:\n\n" + combined_notes
    )
    return call_fn(reduce_prompt)


def build_prompt(section_instructions: str, transcript: str) -> str:
    return (
        "You are analyzing a recorded lecture transcript.\n\n"
        + section_instructions + "\n\nHere is the lecture transcript:\n\n" + transcript
    )


def analyze_section_with_fallback(section_instructions: str, transcript: str,
                                   primary_call_fn, primary_name: str) -> str:
    try:
        return map_reduce_analyze(primary_call_fn, section_instructions, transcript)
    except Exception:
        return call_gemini_with_fallback(build_prompt(section_instructions, transcript))
