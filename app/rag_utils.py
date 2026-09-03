# rag_utils.py - shared chunking helpers for anything that gets embedded
# and retrieved later (material_chunks in materials.py, persona_chunks in
# persona_rag.py). Kept in one place so the two RAG pipelines don't drift
# out of sync with each other.


def chunk_with_overlap(text: str, chunk_chars: int = 1200, overlap_chars: int = 150) -> list:
    """
    Splits text into overlapping ~chunk_chars-character chunks (breaking on
    whitespace where possible). The overlap keeps a fact that straddles a
    chunk boundary retrievable from either side. Good for continuous prose
    (lecture transcripts, pasted notes) with no other natural boundaries.
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


def chunk_by_blocks(text: str, min_chars: int = 200, max_chars: int = 1800) -> list:
    """
    Splits text on blank-line boundaries first, then folds tiny blocks
    together and hard-splits any block that's still too long.

    This is the "divide into records by topic/question" chunking that suits
    persona reference material (a teacher's solved questions, teaching
    notes) better than chunk_with_overlap: that material already has
    natural block boundaries (a blank line between one solved question and
    the next), and breaking on those keeps each retrieved record a whole,
    coherent question+method instead of an arbitrary character-count slice
    that might cut a worked example in half.
    """
    text = (text or "").strip()
    if not text:
        return []
    raw_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if not raw_blocks:
        return chunk_with_overlap(text, max_chars, 0)

    blocks = []
    buffer = ""
    for block in raw_blocks:
        candidate = f"{buffer}\n\n{block}" if buffer else block
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                blocks.append(buffer)
            if len(block) > max_chars:
                # A single block (one "record") is itself too long — fall
                # back to a hard character split just for this one block.
                blocks.extend(chunk_with_overlap(block, max_chars, 100))
                buffer = ""
            else:
                buffer = block
    if buffer:
        blocks.append(buffer)

    # Fold any block under min_chars into the previous one rather than
    # storing lots of tiny, low-signal records that cost a retrieval slot
    # each without adding much.
    merged = []
    for block in blocks:
        if merged and len(block) < min_chars:
            merged[-1] = f"{merged[-1]}\n\n{block}"
        else:
            merged.append(block)
    return merged
