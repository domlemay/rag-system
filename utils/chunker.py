"""Split ParsedDocument bodies into token-bounded chunks ready for embedding."""

import hashlib
import re
from typing import List

import tiktoken

from .markdown_parser import ParsedDocument

# Reuse one encoder instance — initialising it is expensive
_ENCODER = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(_ENCODER.encode(text))


def chunk_document(doc: ParsedDocument, chunk_size: int, overlap: int) -> List[dict]:
    """
    Split doc.content into overlapping chunks and attach metadata.

    Returns a list of dicts:
        { "id": str, "text": str, "metadata": dict }
    """
    sections = _split_by_headers(doc.content)
    raw_chunks = _pack_sections(sections, chunk_size, overlap)

    result = []
    for i, text in enumerate(raw_chunks):
        text = text.strip()
        if not text:
            continue
        result.append({
            "id": _chunk_id(doc.file_path, i),
            "text": text,
            "metadata": {
                "source_file": doc.file_path,
                "folder": doc.folder,
                "title": doc.title,
                "tags": ",".join(doc.tags),
                "tech": ",".join(doc.tech),
                "level": doc.level,
                "date": doc.date,
                "chunk_index": str(i),
            },
        })
    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_by_headers(content: str) -> List[str]:
    """Split on H1–H3 headings, but never inside a code fence (``` or ~~~)."""
    lines = content.split("\n")
    sections: List[str] = []
    current: List[str] = []
    in_fence = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence

        if not in_fence and re.match(r"^#{1,3} ", line) and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current))

    return [s.strip() for s in sections if s.strip()]


def _pack_sections(sections: List[str], chunk_size: int, overlap: int) -> List[str]:
    """
    Merge small sections into one chunk and split oversized sections.
    This keeps context cohesive while respecting the token budget.
    """
    chunks: List[str] = []
    buffer = ""

    for section in sections:
        if token_count(section) > chunk_size:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_sliding_window(section, chunk_size, overlap))
        elif token_count(buffer + "\n\n" + section) > chunk_size:
            if buffer:
                chunks.append(buffer)
            buffer = section
        else:
            buffer = (buffer + "\n\n" + section).strip() if buffer else section

    if buffer:
        chunks.append(buffer)

    return chunks


def _sliding_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Word-level sliding window for sections that exceed chunk_size."""
    words = text.split()
    chunks: List[str] = []
    start = 0

    while start < len(words):
        end, tokens = start, 0
        while end < len(words) and tokens < chunk_size:
            tokens += token_count(words[end])
            end += 1
        chunks.append(" ".join(words[start:end]))
        # Slide back by overlap_words (rough token-to-word approximation)
        overlap_words = max(1, overlap // 6)
        start = max(start + 1, end - overlap_words)

    return chunks


def _chunk_id(file_path: str, index: int) -> str:
    """Stable, filesystem-safe ID for a chunk."""
    h = hashlib.md5(file_path.encode()).hexdigest()[:10]
    return f"{h}_{index}"
