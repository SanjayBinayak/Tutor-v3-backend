# llm_providers.py - Updated with better rate limit handling

import random
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from google import genai
from groq import Groq

from app.config import (
    GEMINI_API_KEY, GEMINI_API_KEYS, GROQ_API_KEY, OPENROUTER_API_KEY,
    GEMINI_FALLBACK_MODELS, GROQ_LLM_MODEL,
    STUDY_GEMINI_API_KEY, STUDY_GEMINI_API_KEYS, STUDY_GEMINI_FALLBACK_MODELS,
    STUDY_GEMINI_EMBEDDING_MODEL, STUDY_GEMINI_EMBEDDING_FALLBACK_MODEL,
    MATERIAL_EMBEDDING_DIM,
)

# ---------------------------------------------------------------------------
# Shared plumbing used by every call_*_with_fallback() function below:
#   - one genai.Client per API key, reused across requests (a fresh client
#     was previously constructed on every single call for no reason)
#   - a shared thread pool, used both to enforce a hard per-attempt timeout
#     (so one stuck request can't hang a whole chat/summary/ingestion
#     request forever) and to run independent calls concurrently (parallel
#     Study Deck sections, parallel embedding batches, parallel map-reduce
#     chunks) instead of one-at-a-time with fixed sleeps between them.
# ---------------------------------------------------------------------------
_genai_clients: dict = {}
_genai_clients_lock = threading.Lock()
_shared_executor = ThreadPoolExecutor(max_workers=16)
_DEFAULT_TIMEOUT_S = 45  # per model attempt


def _get_genai_client(api_key: str) -> genai.Client:
    client = _genai_clients.get(api_key)
    if client is None:
        with _genai_clients_lock:
            client = _genai_clients.get(api_key)
            if client is None:
                client = genai.Client(api_key=api_key)
                _genai_clients[api_key] = client
    return client


def _is_transient_error(e: Exception) -> bool:
    """Rate limits, 5xx, and timeouts are worth retrying the same model for;
    anything else (bad request, auth, not-found) isn't."""
    s = str(e).lower()
    return any(tok in s for tok in (
        "429", "500", "502", "503", "504",
        "resource_exhausted", "rate limit", "rate_limit",
        "timeout", "timed out", "deadline", "unavailable", "internal error",
    ))


def _is_quota_exhausted_error(e: Exception) -> bool:
    """Narrower than _is_transient_error: specifically a daily/quota cap,
    not just a momentary rate limit or blip. Used only to decide whether
    it's worth trying the SAME model again on the NEXT key (no point doing
    that for a plain timeout or a 500, which a different key can't fix)."""
    s = str(e).lower()
    return any(tok in s for tok in ("429", "resource_exhausted", "quota", "rate limit", "rate_limit"))


def _call_gemini_models(api_keys, models: list, make_request,
                         retries_per_model: int = 2, timeout_s: float = _DEFAULT_TIMEOUT_S) -> str:
    """
    Shared fallback engine for every call_*_with_fallback() function below.

    Tries every model in `models`, in order, on the first key in
    `api_keys`; only once EVERY model has failed on that key does it move
    to the next key and repeat the whole model list. This is what makes
    "credits ran out" recoverable two ways at once:
      1. model-to-model on the same key (e.g. gemini-3.6-flash's quota is
         used up for the day -> try gemini-3-flash-preview on that same key)
      2. key-to-key once a whole key's models are exhausted (e.g. every
         model on your first free Google account is out for the day -> the
         whole chain repeats on a second free-tier key/account, if one is
         configured — see GEMINI_API_KEY_2 in config.py)

    Within each (key, model) pair: retries transient errors (rate limits,
    5xx, timeouts) a couple of times with backoff before giving up on that
    model and moving on, so a single transient blip on the best model
    doesn't immediately drop the request to a weaker fallback. A hard
    per-attempt timeout keeps one stuck call from blocking the whole chain.

    api_keys may be a single key string (back-compat) or a list of keys.
    make_request(client, model) -> str does the actual API call.
    """
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    api_keys = [k for k in (api_keys or []) if k]
    if not api_keys:
        raise RuntimeError("No Gemini API key configured.")

    last_error = None
    for key_idx, api_key in enumerate(api_keys):
        client = _get_genai_client(api_key)
        for model in models:
            for attempt in range(retries_per_model + 1):
                try:
                    future = _shared_executor.submit(make_request, client, model)
                    return future.result(timeout=timeout_s)
                except FutureTimeoutError:
                    last_error = TimeoutError(f"{model} timed out after {timeout_s}s (key #{key_idx + 1})")
                    break  # timeouts aren't worth retrying on the same model
                except Exception as e:
                    last_error = e
                    if attempt < retries_per_model and _is_transient_error(e):
                        time.sleep((2 ** attempt) * 1.0 + random.uniform(0, 0.5))
                        continue
                    break
    raise RuntimeError(
        f"All Gemini fallback models failed across {len(api_keys)} key(s). Last error: {last_error}"
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


def _call_with_cross_provider_fallback(gemini_call, prompt: str) -> str:
    """
    Final safety net layered on top of _call_gemini_models' own
    model-then-key rotation. If EVERY Gemini model has failed on EVERY
    configured key (both models' quotas AND both Google accounts' quotas
    exhausted — not just one blip), fall through to Groq, then to a free
    OpenRouter model, before finally giving up.

    Text-only prompts only — vision/audio/YouTube calls stay Gemini-only
    since Groq/OpenRouter aren't wired up for multimodal input here, and a
    lecture-length prompt likely wouldn't fit Groq's/most free OpenRouter
    models' smaller context windows anyway.
    """
    try:
        return gemini_call()
    except Exception as gemini_error:
        try:
            return call_groq(prompt)
        except Exception as groq_error:
            try:
                return call_openrouter(prompt)
            except Exception as openrouter_error:
                raise RuntimeError(
                    "All providers failed — Gemini (all keys/models), Groq, and "
                    f"OpenRouter.\nGemini: {gemini_error}\nGroq: {groq_error}\n"
                    f"OpenRouter: {openrouter_error}"
                )


def call_gemini_with_fallback(prompt: str) -> str:
    """Tries each Gemini Flash-tier model in order (across every configured
    key) until one succeeds, then Groq, then OpenRouter as a last resort.
    Large context window — fine to send full transcripts/personas in one call."""
    return _call_with_cross_provider_fallback(
        lambda: _call_gemini_models(
            GEMINI_API_KEYS, GEMINI_FALLBACK_MODELS,
            lambda client, model: client.models.generate_content(model=model, contents=prompt).text,
        ),
        prompt,
    )


def call_gemini_vision_with_fallback(prompt: str, file_bytes: bytes, mime_type: str) -> str:
    """
    Same fallback pattern as call_gemini_with_fallback, but for requests that
    include an image or PDF (e.g. a photo of homework) alongside text.
    """
    from google.genai import types as genai_types

    def make_request(client, model):
        response = client.models.generate_content(
            model=model,
            contents=[
                genai_types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                prompt,
            ],
        )
        return response.text

    return _call_gemini_models(GEMINI_API_KEYS, GEMINI_FALLBACK_MODELS, make_request)


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

    def make_request(client, model):
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

    # Watching a full lecture takes longer than the default timeout, so this
    # path gets a longer per-attempt ceiling.
    return _call_gemini_models(GEMINI_API_KEYS, GEMINI_FALLBACK_MODELS, make_request, timeout_s=180)


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

    def make_request(client, model):
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

    try:
        return _call_gemini_models(GEMINI_API_KEYS, GEMINI_FALLBACK_MODELS, make_request, timeout_s=90)
    except RuntimeError as e:
        raise RuntimeError(
            f"All Gemini fallback models failed on video chunk "
            f"[{start_seconds}s-{end_seconds}s]. {e}"
        )


def call_study_gemini_with_fallback(prompt: str) -> str:
    """Same fallback pattern as call_gemini_with_fallback (now including the
    Groq -> OpenRouter cross-provider tail), but uses the study-tool's own
    API key/quota (STUDY_GEMINI_API_KEY) — kept separate from the
    persona/tutor chat system's key."""
    return _call_with_cross_provider_fallback(
        lambda: _call_gemini_models(
            STUDY_GEMINI_API_KEYS, STUDY_GEMINI_FALLBACK_MODELS,
            lambda client, model: client.models.generate_content(model=model, contents=prompt).text,
        ),
        prompt,
    )


def call_study_gemini_vision_with_fallback(prompt: str, file_bytes: bytes, mime_type: str) -> str:
    """Vision variant of call_study_gemini_with_fallback, for notes analysis
    (photos of handwritten notes / PDFs)."""
    from google.genai import types as genai_types

    def make_request(client, model):
        response = client.models.generate_content(
            model=model,
            contents=[
                genai_types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                prompt,
            ],
        )
        return response.text

    return _call_gemini_models(STUDY_GEMINI_API_KEYS, STUDY_GEMINI_FALLBACK_MODELS, make_request)


def call_gemini_audio_with_fallback(prompt: str, audio_bytes: bytes, mime_type: str,
                                     api_keys=None, models=None) -> str:
    """
    Sends an audio clip straight to Gemini (native audio understanding) and
    asks it to transcribe it per `prompt`. This exists as the automatic
    fallback for ingestion.py's _transcribe_chunk() — used when Groq Whisper
    is unavailable or its free-tier daily quota (2,000 requests / 28,800
    audio-seconds per day, shared across ALL your Groq keys, since Groq's
    free limits are per-organization, not per-key) has run out for the day.

    Gemini's free tier, unlike Groq's, is metered per Google Cloud project —
    so a genuinely separate key (GEMINI_API_KEY_2 in config.py, from a
    second Google account) gets its own quota here, and _call_gemini_models
    will automatically roll over to it once the first key's models are all
    exhausted.

    Defaults to the persona-system key pool (GEMINI_API_KEYS); pass
    api_keys=STUDY_GEMINI_API_KEYS explicitly if calling this from the
    student-facing Materials Hub instead.
    """
    from google.genai import types as genai_types

    def make_request(client, model):
        response = client.models.generate_content(
            model=model,
            contents=[
                genai_types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                prompt,
            ],
        )
        return response.text

    return _call_gemini_models(
        api_keys if api_keys is not None else GEMINI_API_KEYS,
        models if models is not None else GEMINI_FALLBACK_MODELS,
        make_request,
        timeout_s=90,
    )


def embed_texts_batch(texts: list, task_type: str = "RETRIEVAL_DOCUMENT",
                       batch_size: int = 20, max_concurrency: int = 4,
                       max_retries: int = 3) -> list:
    """
    Embeds a batch of strings with Gemini's embedding model.

    Previously this sent 5 texts per request and slept ~2-3s between every
    batch, serially — for a 30-40 chunk PDF that's 20-30+ seconds of pure
    waiting before a material could ever go `ready`. Now batches are 4x
    bigger AND sent concurrently (bounded by max_concurrency), with no fixed
    delay — a batch only backs off if it actually gets rate-limited. Each
    batch still retries transient errors with backoff and falls back to a
    different embedding model if the primary keeps failing.

    task_type should be "RETRIEVAL_DOCUMENT" when embedding material chunks
    to store, or "RETRIEVAL_QUERY" when embedding a student's question.
    """
    from google.genai import types as genai_types

    if not texts:
        return []

    client = _get_genai_client(STUDY_GEMINI_API_KEY)
    primary_model = STUDY_GEMINI_EMBEDDING_MODEL
    fallback_model = STUDY_GEMINI_EMBEDDING_FALLBACK_MODEL
    # Guard against a misconfigured fallback that's identical to the primary
    # model — that wouldn't actually help if the primary is down/quota'd.
    models_to_try = [primary_model] if fallback_model == primary_model else [primary_model, fallback_model]

    def embed_one_batch(batch: list) -> list:
        last_error = None
        for model in models_to_try:
            for attempt in range(max_retries):
                try:
                    response = client.models.embed_content(
                        model=model,
                        contents=batch,
                        config=genai_types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=MATERIAL_EMBEDDING_DIM,
                        ),
                    )
                    return [e.values for e in response.embeddings]
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1 and _is_transient_error(e):
                        wait = (2 ** attempt) * 1.5 + random.uniform(0, 0.5)
                        time.sleep(wait)
                        continue
                    break  # this model's exhausted its retries — try the next model
        raise RuntimeError(f"Embedding failed for all models. Last error: {last_error}")

    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    if len(batches) == 1:
        return embed_one_batch(batches[0])

    results = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=min(max_concurrency, len(batches))) as pool:
        future_to_idx = {pool.submit(embed_one_batch, b): i for i, b in enumerate(batches)}
        for future, idx in future_to_idx.items():
            results[idx] = future.result()  # re-raises if that batch ultimately failed

    all_embeddings = []
    for batch_result in results:
        all_embeddings.extend(batch_result)
    return all_embeddings


def embed_text(text: str, task_type: str = "RETRIEVAL_QUERY") -> list:
    """Single-string convenience wrapper around embed_texts_batch."""
    return embed_texts_batch([text], task_type=task_type)[0]


def chunk_text(text: str, max_chars: int = 16000) -> list:
    """Split text into chunks of approximately max_chars characters."""
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def map_reduce_analyze(call_fn, section_instructions: str, transcript: str,
                        chunk_chars: int = 16000, max_concurrency: int = 3) -> str:
    """Use map-reduce pattern for analyzing long texts.

    Map-step chunks are independent of each other, so they're processed
    concurrently (bounded by max_concurrency) instead of serially with a
    fixed 20s sleep between every call — each call_fn already retries/falls
    back on transient errors on its own, so the fixed delay was mostly just
    dead time. For a multi-chunk material summary this cuts wall-clock time
    from minutes down to roughly the time of one call."""
    chunks = chunk_text(transcript, chunk_chars)
    if len(chunks) == 1:
        return call_fn(build_prompt(section_instructions, transcript))

    def process_chunk(indexed_chunk):
        i, chunk = indexed_chunk
        chunk_prompt = (
            f"You are extracting notes from PART {i} of {len(chunks)} of a longer "
            "lecture transcript. Extract only the raw facts/details relevant to the "
            "section below — short note form, no polished writing yet:\n\n"
            + section_instructions + "\n\nTranscript part:\n\n" + chunk
        )
        return call_fn(chunk_prompt)

    with ThreadPoolExecutor(max_workers=min(max_concurrency, len(chunks))) as pool:
        # pool.map preserves input order in its output, even though the
        # calls themselves run concurrently.
        partial_notes = list(pool.map(process_chunk, enumerate(chunks, start=1)))

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