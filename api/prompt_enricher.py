"""Prompt enricher: wraps a user prompt with retrieved context.

The output format is designed for direct injection into any LLM or AI assistant.
When no relevant context is found, the prompt is returned unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from api.context_builder import build, RetrievedContext

_ENRICHED_TEMPLATE = """\
=== CONTEXT FROM DEVELOPER BRAIN ===

{context}

=== SOURCES ===
{sources}

=== USER REQUEST ===

{prompt}"""


def enrich(
    prompt: str,
    top_k: int = 5,
    collection: Any = None,
) -> tuple[str, RetrievedContext]:
    """Enrich a user prompt with relevant vault context.

    Returns:
        enriched_prompt: Context block + original prompt, ready for any LLM.
        ctx:             RetrievedContext with sources and metadata.

    When no relevant context is found, enriched_prompt == prompt.
    """
    ctx = build(prompt, top_k=top_k, collection=collection)

    if not ctx.context:
        return prompt, ctx

    sources_text = "\n".join(
        f"- [{s['title']}] ({s['source_type']}, score: {s['score']})"
        for s in ctx.sources
    )

    enriched = _ENRICHED_TEMPLATE.format(
        context=ctx.context,
        sources=sources_text,
        prompt=prompt,
    )
    return enriched, ctx
