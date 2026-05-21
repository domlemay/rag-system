"""Local RAG API server.

Endpoints:
    GET  /health   — Server status, index document count, config
    POST /query    — Returns structured context for a query (retrieval only, no LLM)
    POST /enrich   — Returns enriched prompt (context block + user request)
    GET  /stats    — Top-used chunks and feedback loop stats

Run from the rag-system/ directory:
    uvicorn api.server:app --host 127.0.0.1 --port 8765
    python -m api.server
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chromadb
import uvicorn
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from api.context_builder import build
from api.prompt_enricher import enrich
from utils import scorer

app = FastAPI(
    title="Developer Brain API",
    description="Local RAG API — semantic search over your developer knowledge base.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None,
)

# ── Shared ChromaDB collection (lazy-loaded singleton) ────────────────────────

_collection: Any = None


def _get_collection() -> Any:
    global _collection
    if _collection is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
        client = chromadb.PersistentClient(path=str(config.DB_PATH))
        embedding_fn = OpenAIEmbeddingFunction(
            api_key=config.OPENAI_API_KEY,
            model_name=config.EMBEDDING_MODEL,
        )
        try:
            _collection = client.get_collection(
                name=config.COLLECTION_NAME,
                embedding_function=embedding_fn,
            )
        except Exception as exc:
            raise RuntimeError(
                "Index not found. Run 'python index.py' first to build the vector DB."
            ) from exc
    return _collection


# ── Pydantic models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question or search term.")
    top_k: int = Field(default=config.TOP_K, ge=1, le=20, description="Chunks to retrieve.")


class EnrichRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt to enrich.")
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)


class SourceItem(BaseModel):
    title: str
    folder: str
    score: float
    source_type: str


class QueryResponse(BaseModel):
    context: str
    sources: list[SourceItem]
    chunks_used: int
    query: str


class EnrichResponse(BaseModel):
    enriched_prompt: str
    context_added: bool
    chunks_used: int
    sources: list[SourceItem]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    """Server status, index info, and active config."""
    index_ok = True
    doc_count = 0
    try:
        col = _get_collection()
        doc_count = col.count()
    except RuntimeError:
        index_ok = False

    return {
        "status":          "ok",
        "index_ready":     index_ok,
        "document_count":  doc_count,
        "embedding_model": config.EMBEDDING_MODEL,
        "chat_model":      config.CHAT_MODEL,
        "top_k_default":   config.TOP_K,
        "vault_path":      str(config.VAULT_PATH),
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    """Retrieve relevant context from the knowledge base.

    Pure retrieval — no LLM call. Fast. Use this when you need raw context
    to inject into your own LLM pipeline.
    """
    try:
        collection = _get_collection()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        result = build(req.query, top_k=req.top_k, collection=collection)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")

    return QueryResponse(
        context=result.context,
        sources=[SourceItem(**s) for s in result.sources],
        chunks_used=result.chunks_used,
        query=result.query,
    )


@app.post("/enrich", response_model=EnrichResponse)
def enrich_endpoint(req: EnrichRequest) -> EnrichResponse:
    """Enrich a user prompt with relevant knowledge base context.

    Returns the full enriched prompt ready for injection into any LLM.
    Format:
        === CONTEXT FROM DEVELOPER BRAIN ===
        ...retrieved chunks...
        === USER REQUEST ===
        ...original prompt...
    """
    try:
        collection = _get_collection()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        enriched_prompt, ctx = enrich(req.prompt, top_k=req.top_k, collection=collection)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Enrichment failed: {exc}")

    return EnrichResponse(
        enriched_prompt=enriched_prompt,
        context_added=bool(ctx.context),
        chunks_used=ctx.chunks_used,
        sources=[SourceItem(**s) for s in ctx.sources],
    )


@app.get("/stats")
def stats_endpoint() -> dict:
    """Usage statistics for the feedback loop (most-retrieved chunks)."""
    usage = scorer.load_stats()
    if not usage:
        return {"total_chunks_tracked": 0, "top_chunks": []}

    top = sorted(usage.items(), key=lambda x: x[1].get("usage_count", 0), reverse=True)
    return {
        "total_chunks_tracked": len(usage),
        "top_chunks": [
            {"chunk_id": cid[:20] + "…", **data}
            for cid, data in top[:10]
        ],
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        log_level="info",
    )
