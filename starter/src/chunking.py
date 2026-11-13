"""Fixed-window and section-preserving chunking strategies."""
import hashlib

from .models import Chunk, DocumentPage


def _chunk(page: DocumentPage, strategy: str, start: int, end: int) -> Chunk:
    text = page.text[start:end].strip()
    digest = hashlib.sha256(
        f"{strategy}|{page.document_id}|{page.page}|{start}|{end}|{text}".encode("utf-8")
    ).hexdigest()
    return Chunk(
        chunk_id=digest,
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata={**page.metadata, "start_char": start, "end_char": end},
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split one source section into overlapping character windows."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be in [0, size).")
    chunks: list[Chunk] = []
    start = 0
    while start < len(page.text):
        end = min(start + size, len(page.text))
        if page.text[start:end].strip():
            chunks.append(_chunk(page, "fixed", start, end))
        if end == len(page.text):
            break
        start = end - overlap
    return chunks


def by_section(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Keep each heading section intact when possible; split only oversized bodies."""
    if max_size <= 0:
        raise ValueError("max_size must be positive.")
    if len(page.text) <= max_size:
        return [_chunk(page, "section", 0, len(page.text))] if page.text.strip() else []

    chunks: list[Chunk] = []
    start = 0
    while start < len(page.text):
        end = min(start + max_size, len(page.text))
        if end < len(page.text):
            boundary = page.text.rfind("\n\n", start, end)
            if boundary > start:
                end = boundary
        if page.text[start:end].strip():
            chunks.append(_chunk(page, "section", start, end))
        if end == len(page.text):
            break
        start = end
        while start < len(page.text) and page.text[start].isspace():
            start += 1
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Backward-compatible name for the section-aware strategy."""
    return by_section(page, max_size)
