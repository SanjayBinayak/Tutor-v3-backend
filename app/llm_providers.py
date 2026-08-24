# llm_providers.py - Updated with better rate limit handling

import time
import requests
from google import genai
from groq import Groq

from app.config import (
    GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY,
    GEMINI_FALLBACK_MODELS, GROQ_LLM_MODEL,
    STUDY_GEMINI_API_KEY, STUDY_GEMINI_FALLBACK_MODELS,
    STUDY_GEMINI_EMBEDDING_MODEL, STUDY_GEMINI_EMBEDDING_FALLBACK_MODEL,
    MATERIAL_EMBEDDING_DIM,
)


def call_groq(prompt: str) -> str:
    """Call Groq's LLM API with the specified model."""
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def pick_openrouter_free_model() -> str:
    """Pick a free OpenRouter model from the available list."""
    resp = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
    resp.raise_for_status()
    for model in resp.json().get("data", []):
        pricing = model.get("pricing", {})
        if pricing.get("prompt") == "0" and pricing.get("completion") == "0" and model["id"].endswith(":free"):
            return model["id"]
    raise RuntimeError("No free OpenRouter model found — check https://openrouter.ai/models?max_price=0")


def call_openrouter(prompt: str) -> str:
    """Call OpenRouter's API with a free model."""
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


def call_gemini_with_youtube_fallback(prompt: str, youtube_url: str) -> str:
    """
    Same fallback pattern as call_gemini_with_fallback, but passes a YouTube
    URL directly as a file_data Part — Gemini fetches and watches/listens to
    the video itself, no download or transcription needed on our end.

    Requirements (per Gemini's YouTube URL support):
    - Video must be public (not private/unlisted)
    - MEDIA_RESOLUTION_LOW lets a 1M-context model handle up to ~3hr videos
      (vs. ~1hr at default resolution) — trades some visual fidelity
      (e.g. reading whiteboard detail) for being able to fit longer lectures
    - Free tier caps at 8 hours of YouTube video processed per day, total
    """
    from google.genai import types as genai_types
    client = genai.Client(api_key=GEMINI_API_KEY)
    last_error = None
    for model in GEMINI_FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=genai_types.Content(parts=[
                    genai_types.Part(file_data=genai_types.FileData(file_uri=youtube_url)),
                    genai_types.Part(text=prompt),
                ]),
                config=genai_types.GenerateContentConfig(
                    media_resolution="MEDIA_RESOLUTION_LOW",
                ),
            )
            return response.text
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All Gemini fallback models failed on YouTube video. Last error: {last_error}")


def call_gemini_video_chunk_with_fallback(prompt: str, youtube_url: str,
                                           start_seconds: int, end_seconds: int) -> str:
    """
    Same fallback pattern, but scoped to ONE clipped time range of a YouTube
    video via video_metadata start/end offsets. Used to process long videos
    in ~20-min chunks instead of one giant request, so each chunk can use
    default (higher-fidelity) media resolution instead of the LOW setting
    call_gemini_with_youtube_fallback needs for full multi-hour videos.

    Known caveat (Aug 2026): a currently-open Gemini API regression means
    start/end offset clipping on YouTube URLs clips the video frames but
    NOT the audio track — every chunk request receives the full audio.
    This doesn't break anything (audio alone is cheap: ~32 tokens/sec, so
    even 3hrs of audio is ~345K tokens — fine alongside one chunk's
    clipped frames), just some wasted redundant audio processing per
    chunk until Google fixes it.
    """
    from google.genai import types as genai_types
    client = genai.Client(api_key=GEMINI_API_KEY)
    last_error = None
    for model in GEMINI_FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=genai_types.Content(parts=[
                    genai_types.Part(
                        file_data=genai_types.FileData(file_uri=youtube_url),
                        video_metadata=genai_types.VideoMetadata(
                            start_offset=f"{start_seconds}s",
                            end_offset=f"{end_seconds}s",
                        ),
                    ),
                    genai_types.Part(text=prompt),
                ]),
            )
            return response.text
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(
        f"All Gemini fallback models failed on video chunk "
        f"[{start_seconds}s-{end_seconds}s]. Last error: {last_error}"
    )


def call_study_gemini_with_fallback(prompt: str) -> str:
    """Same fallback pattern as call_gemini_with_fallback, but uses the
    study-tool's own API key/quota (STUDY_GEMINI_API_KEY) — kept separate
    from the persona/tutor chat system's key."""
    client = genai.Client(api_key=STUDY_GEMINI_API_KEY)
    last_error = None
    for model in STUDY_GEMINI_FALLBACK_MODELS:
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All study-tool Gemini fallback models failed. Last error: {last_error}")


def call_study_gemini_vision_with_fallback(prompt: str, file_bytes: bytes, mime_type: str) -> str:
    """Vision variant of call_study_gemini_with_fallback, for notes analysis
    (photos of handwritten notes / PDFs)."""
    from google.genai import types as genai_types
    client = genai.Client(api_key=STUDY_GEMINI_API_KEY)
    last_error = None
    for model in STUDY_GEMINI_FALLBACK_MODELS:
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
    raise RuntimeError(f"All study-tool Gemini fallback models failed. Last error: {last_error}")


def embed_texts_batch(texts: list, task_type: str = "RETRIEVAL_DOCUMENT", max_retries: int = 5) -> list:
    """
    Embeds a batch of strings with Gemini's embedding model.
    Uses aggressive rate limit handling with:
    - Smaller batch sizes (5 instead of 10)
    - Longer delays between batches (2 seconds)
    - Exponential backoff with jitter
    - Fallback to text-embedding-004
    
    task_type should be "RETRIEVAL_DOCUMENT" when embedding material chunks
    to store, or "RETRIEVAL_QUERY" when embedding a student's question.
    """
    from google.genai import types as genai_types
    import random
    
    client = genai.Client(api_key=STUDY_GEMINI_API_KEY)
    all_embeddings = []
    
    # Much smaller batch size to reduce TPM spikes
    batch_size = 5  # Reduced from 10
    # Longer delay between batches to stay under TPM limits
    batch_delay = 2.0  # 2 seconds between batches
    
    primary_model = STUDY_GEMINI_EMBEDDING_MODEL
    fallback_model = STUDY_GEMINI_EMBEDDING_FALLBACK_MODEL
    
    # Track token usage to avoid TPM limits
    estimated_tokens_per_text = 100  # Rough estimate for embedding tokens
    
    for attempt in range(max_retries):
        try:
            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                # Estimate token usage for this batch
                batch_tokens = sum(len(t.split()) for t in batch) * 1.3  # Rough token estimate
                print(f"[INFO] Processing batch {batch_num}/{total_batches} (~{batch_tokens:.0f} tokens)")
                
                response = client.models.embed_content(
                    model=primary_model,
                    contents=batch,
                    config=genai_types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=MATERIAL_EMBEDDING_DIM,
                    ),
                )
                all_embeddings.extend([e.values for e in response.embeddings])
                
                # Longer delay between batches with some randomness (jitter)
                if i + batch_size < len(texts):
                    jitter = random.uniform(0.3, 0.7)  # 0.3-0.7s randomness
                    delay = batch_delay + jitter
                    print(f"[INFO] Waiting {delay:.1f}s before next batch...")
                    time.sleep(delay)
            
            print(f"[INFO] Successfully embedded {len(texts)} texts using {primary_model}")
            return all_embeddings
            
        except Exception as e:
            error_str = str(e)
            print(f"[WARN] Embedding attempt {attempt+1}/{max_retries} failed: {error_str[:100]}...")
            
            # Check if it's a rate limit or quota error
            is_rate_limit = (
                "429" in error_str or 
                "RESOURCE_EXHAUSTED" in error_str or 
                "quota" in error_str.lower() or
                "rate" in error_str.lower()
            )
            
            # Check if it's a model not found error
            is_model_not_found = "not found" in error_str.lower() or "404" in error_str
            
            if is_model_not_found:
                print(f"[INFO] Model {primary_model} not found, trying fallback {fallback_model}")
                # Try fallback model with same batch strategy
                try:
                    all_embeddings = []
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        response = client.models.embed_content(
                            model=fallback_model,
                            contents=batch,
                            config=genai_types.EmbedContentConfig(
                                task_type=task_type,
                                output_dimensionality=MATERIAL_EMBEDDING_DIM,
                            ),
                        )
                        all_embeddings.extend([e.values for e in response.embeddings])
                        if i + batch_size < len(texts):
                            time.sleep(batch_delay)
                    print(f"[INFO] Successfully embedded {len(texts)} texts using fallback {fallback_model}")
                    return all_embeddings
                except Exception as fallback_e:
                    print(f"[ERROR] Fallback model {fallback_model} also failed: {fallback_e}")
                    raise RuntimeError(
                        f"Both embedding models failed. Primary ({primary_model}): {error_str[:200]}, "
                        f"Fallback ({fallback_model}): {fallback_e}"
                    )
            
            elif is_rate_limit:
                if attempt < max_retries - 1:
                    # Longer exponential backoff with jitter
                    base_wait = (2 ** attempt) * 3  # 3, 6, 12, 24 seconds
                    jitter = random.uniform(0.8, 1.2)
                    wait_time = base_wait * jitter
                    print(f"[WARN] Rate limit hit, waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Last attempt failed, try fallback model
                    print(f"[INFO] Primary model {primary_model} rate limited, trying fallback {fallback_model}")
                    try:
                        all_embeddings = []
                        for i in range(0, len(texts), batch_size):
                            batch = texts[i:i + batch_size]
                            response = client.models.embed_content(
                                model=fallback_model,
                                contents=batch,
                                config=genai_types.EmbedContentConfig(
                                    task_type=task_type,
                                    output_dimensionality=MATERIAL_EMBEDDING_DIM,
                                ),
                            )
                            all_embeddings.extend([e.values for e in response.embeddings])
                            if i + batch_size < len(texts):
                                time.sleep(batch_delay)
                        print(f"[INFO] Successfully embedded {len(texts)} texts using fallback {fallback_model}")
                        return all_embeddings
                    except Exception as fallback_e:
                        print(f"[ERROR] Fallback model {fallback_model} also failed: {fallback_e}")
                        raise RuntimeError(
                            f"Both embedding models failed. Primary ({primary_model}): {error_str[:200]}, "
                            f"Fallback ({fallback_model}): {fallback_e}"
                        )
            else:
                # Non-retryable error
                print(f"[ERROR] Non-retryable embedding error: {e}")
                raise
    
    raise RuntimeError(f"Failed to embed batch after {max_retries} attempts")


def embed_text(text: str, task_type: str = "RETRIEVAL_QUERY") -> list:
    """Single-string convenience wrapper around embed_texts_batch."""
    return embed_texts_batch([text], task_type=task_type)[0]


def chunk_text(text: str, max_chars: int = 16000) -> list:
    """Split text into chunks of approximately max_chars characters."""
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def map_reduce_analyze(call_fn, section_instructions: str, transcript: str,
                        chunk_chars: int = 16000, delay_between_calls: int = 20) -> str:
    """Use map-reduce pattern for analyzing long texts."""
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
    """Build a prompt for analyzing a lecture transcript."""
    return (
        "You are analyzing a recorded lecture transcript.\n\n"
        + section_instructions + "\n\nHere is the lecture transcript:\n\n" + transcript
    )


def analyze_section_with_fallback(section_instructions: str, transcript: str,
                                   primary_call_fn, primary_name: str) -> str:
    """Try primary provider first, fall back to Gemini if it fails."""
    try:
        return map_reduce_analyze(primary_call_fn, section_instructions, transcript)
    except Exception:
        return call_gemini_with_fallback(build_prompt(section_instructions, transcript))