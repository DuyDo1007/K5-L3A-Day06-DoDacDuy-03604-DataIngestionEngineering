"""Read and normalize source data while retaining a stable source location."""
import re
import unicodedata
from pathlib import Path
from .models import DocumentPage


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page; OCR is intentionally not performed."""
    from pypdf import PdfReader

    pages: list[DocumentPage] = []
    for page_number, pdf_page in enumerate(PdfReader(path).pages, start=1):
        text = normalize_text(pdf_page.extract_text() or "")
        metadata = {"extraction_status": "ok" if text else "empty"}
        pages.append(DocumentPage(document_id, source, page_number, text, metadata))
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph boundaries."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    normalized = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def load_markdown_sections(
    path: Path,
    document_id: str,
    source: str,
    metadata: dict[str, object],
    page_manifest: list[dict] | None = None,
) -> list[DocumentPage]:
    """Treat each level-two heading and its body as a traceable corpus unit."""
    sections: list[tuple[int, str, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    start_line = 0

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("## "):
            if current_title is not None:
                sections.append((start_line, current_title, current_lines))
            current_title = line[3:].strip()
            current_lines = []
            start_line = line_number
        elif current_title is not None and not line.startswith("# "):
            current_lines.append(line)
    if current_title is not None:
        sections.append((start_line, current_title, current_lines))

    if page_manifest is not None and len(sections) != len(page_manifest):
        raise ValueError(f"{path}: section count differs from manifest")

    pages: list[DocumentPage] = []
    for index, (first_line, section_title, lines) in enumerate(sections, start=1):
        page_id = f"{document_id}-{index:02d}"
        citations = [line for line in lines if line.startswith(("Nguồn chính:", "Nguồn tham khảo:"))]
        body = normalize_text("\n".join(line for line in lines if line not in citations))
        if page_manifest is None:
            unit_metadata = {}
            primary_source = source
        else:
            unit = page_manifest[index - 1]
            end_line = first_line + max((i for i, line in enumerate(lines, 1) if line.strip()), default=0)
            actual = (page_id, section_title, first_line, end_line, len(body))
            expected = tuple(unit[key] for key in ("id", "heading", "start_line", "end_line", "normalized_content_characters"))
            if actual != expected or len(body) < 1500:
                raise ValueError(f"{path}:{first_line}: content or boundaries for {page_id} differ from manifest")
            source_urls = re.findall(r"\]\((https?://[^)]+)\)", "\n".join(citations))
            if unit["source_urls"] != source_urls or unit["source"] not in source_urls:
                raise ValueError(f"{path}:{first_line}: sources for {page_id} differ from manifest")
            primary_source = unit["source"]
            unit_metadata = {key: unit[key] for key in ("source_urls", "source_type", "start_line", "end_line")}
        if not body:
            continue
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=primary_source,
                page=page_id,
                text=f"{section_title}\n\n{body}",
                metadata={**metadata, **unit_metadata, "section": section_title},
            )
        )
    return pages
