"""Context builder: retrieves relevant vault chunks and formats them for LLM injection.

Runs the full retrieval pipeline (intent detection → hybrid search → scoring)
without calling the LLM. Returns a clean, structured context block.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from utils import scorer

_SOURCE_ORDER = ["personal", "project", "official", "blog"]
_SECTION_LABELS: dict[str, str] = {
    "personal": "Personal Notes",
    "project":  "Project Notes",
    "official": "Official Documentation",
    "blog":     "External Resources",
}


@dataclass
class RetrievedContext:
    """Result of a context retrieval operation."""
    context: str
    sources: list[dict] = field(default_factory=list)
    chunks_used: int = 0
    query: str = ""


def build(
    query: str,
    top_k: int = config.TOP_K,
    collection: Any = None,
) -> RetrievedContext:
    """Retrieve and format context for a given query.

    Runs intent detection → hybrid search → scoring → formatting.
    Does NOT call the LLM — returns only the context block.

    Args:
        query:      The user's question or search term.
        top_k:      Number of chunks to include in the context.
        collection: ChromaDB collection. If None, loads from config (requires index).
    """
    from query import hybrid_search, score_and_rank, extract_intent

    if collection is None:
        from query import get_collection
        collection = get_collection()

    tag, tech = extract_intent(query)
    raw = hybrid_search(collection, query, tag, tech)

    if not raw["documents"][0]:
        return RetrievedContext(context="", sources=[], chunks_used=0, query=query)

    stats = scorer.load_stats()
    ranked, _ = score_and_rank(raw, top_k, stats)
    scorer.update_stats([r[1] for r in ranked])

    context = _format_context(ranked)
    sources = [
        {
            "title":       meta.get("title", "Untitled"),
            "folder":      meta.get("folder", ""),
            "score":       round(score, 3),
            "source_type": scorer.derive_source(meta),
        }
        for score, _, _, meta in ranked
    ]

    return RetrievedContext(
        context=context,
        sources=sources,
        chunks_used=len(ranked),
        query=query,
    )


def _format_context(ranked: list[tuple]) -> str:
    """Group ranked chunks by source type and return a structured context string."""
    buckets: dict[str, list] = {s: [] for s in _SOURCE_ORDER}
    for score, _chunk_id, doc, meta in ranked:
        src = scorer.derive_source(meta)
        buckets.setdefault(src, []).append((score, doc, meta))

    active = [s for s in _SOURCE_ORDER if buckets[s]]
    multi_source = len(active) > 1

    sections: list[str] = []
    counter = 1

    for src in active:
        parts: list[str] = []
        for _score, doc, meta in buckets[src]:
            title = meta.get("title", "Untitled")
            parts.append(f"[{counter}] {title}\n\n{doc}")
            counter += 1

        block = "\n\n---\n\n".join(parts)
        if multi_source:
            sections.append(f"## {_SECTION_LABELS.get(src, src.title())}\n\n{block}")
        else:
            sections.append(block)

    return "\n\n".join(sections)
