"""
query.py — Ask questions against your indexed Obsidian vault (RAG pipeline).

Usage:
    python query.py "How do I use async/await in Python?"
    python query.py                              # interactive REPL
    python query.py --tag async "event loop"
    python query.py --tech sqlalchemy --top-k 8 "how to do joins"
    python query.py --verbose "repository pattern"
    python query.py --learn "how do I cancel async tasks?"
    python query.py --learn --yes "asyncio timeout pattern"   # auto-save, no prompt

Retrieval pipeline:
    1. Hybrid search  — semantic pass (top 20) + keyword-boosted pass
    2. Scoring        — semantic × 0.5 + metadata score × 0.5
    3. Re-ranking     — sort by final score, keep top_k
    4. Context fusion — group by source (personal → official → blog)
    5. Feedback loop  — increment usage_count for every retrieved chunk
    6. Auto-learning  — optionally extract and save a vault note (--learn)
"""

import argparse
import re
import sys
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

import config
from utils import get_logger
from utils import scorer
from utils import learner

console = Console()
log = get_logger("query")

# ── Retrieval constant ────────────────────────────────────────────────────────

RETRIEVAL_K = 20   # candidates fetched before scoring

# ── Intent detection lookup tables ───────────────────────────────────────────

_TECH_KEYWORDS: frozenset = frozenset({
    "python", "javascript", "typescript", "js", "ts",
    "react", "vue", "angular", "svelte",
    "node", "nodejs", "deno", "bun",
    "fastapi", "django", "flask", "express", "nextjs",
    "postgres", "postgresql", "mysql", "sqlite", "mongodb", "redis",
    "docker", "kubernetes", "k8s", "terraform",
    "sqlalchemy", "prisma", "drizzle",
    "asyncio", "celery", "pandas", "numpy", "pytorch",
    "git", "graphql", "aws", "azure", "gcp",
})

_TAG_KEYWORDS: frozenset = frozenset({
    "async", "concurrency", "threading",
    "pattern", "patterns", "architecture",
    "testing", "test", "debug", "debugging",
    "performance", "optimization", "cache", "caching",
    "security", "auth", "authentication",
    "deployment", "devops", "error", "errors", "exception",
})

_STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall",
    "i", "you", "he", "she", "it", "we", "they",
    "what", "how", "why", "when", "where", "which", "who",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "up", "about", "into", "through", "during",
    "and", "or", "but", "not", "if", "then", "else",
    "my", "your", "its", "our", "their",
})

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior developer mentor helping a developer search their personal knowledge base.
You have been given notes retrieved from their Obsidian vault, ranked by semantic relevance.

Rules:
- Use the retrieved notes as your PRIMARY source of truth.
- Reference specific note titles when you draw from them.
- Be direct and practical — no filler, no disclaimers.
- Use markdown: headers, bullet points, code blocks.
- If the notes don't cover the question fully, say so briefly, then supplement with your own knowledge.
- Answer at the level of a senior developer explaining to an intermediate one."""

USER_PROMPT = """\
## Retrieved Notes from Personal Knowledge Base

{context}

## Sources

{sources}

---

## My Question

{question}
"""


# ── ChromaDB connection ───────────────────────────────────────────────────────

def get_collection() -> Any:
    """Open the persisted ChromaDB collection. Exits with a clear message if not found."""
    _require_api_key()
    client = chromadb.PersistentClient(path=str(config.DB_PATH))
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=config.OPENAI_API_KEY,
        model_name=config.EMBEDDING_MODEL,
    )
    try:
        return client.get_collection(name=config.COLLECTION_NAME, embedding_function=embedding_fn)
    except Exception:
        console.print("[bold red]ERROR:[/] No index found. Run [bold]python index.py[/] first.")
        sys.exit(1)


# ── Intent detection ──────────────────────────────────────────────────────────

def extract_intent(question: str) -> tuple[str, str]:
    """Detect tech and tag keywords from the question text (no API call).

    Returns (tag, tech) — empty strings if nothing is found.
    """
    words = set(re.sub(r"[^\w\s]", " ", question.lower()).split())
    detected_tech = next((w for w in words if w in _TECH_KEYWORDS), "")
    detected_tag  = next((w for w in words if w in _TAG_KEYWORDS), "")
    return detected_tag, detected_tech


def _extract_key_terms(question: str) -> list[str]:
    """Return meaningful words from a question with stopwords removed."""
    words = re.sub(r"[^\w\s]", " ", question.lower()).split()
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


# ── Hybrid search ─────────────────────────────────────────────────────────────

def hybrid_search(collection: Any, question: str, tag: str, tech: str) -> dict:
    """Semantic search (RETRIEVAL_K candidates) + keyword-boosted pass.

    Includes chunk IDs so the scorer can track usage per chunk.
    Returns dict with keys: ids, documents, metadatas, distances.
    """
    where = _build_where(tag, tech)

    kwargs: dict = dict(
        query_texts=[question],
        n_results=RETRIEVAL_K,
        include=["ids", "documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    seen_docs: set[str] = set(results["documents"][0])
    extra_ids, extra_docs, extra_metas, extra_dists = [], [], [], []

    for term in _extract_key_terms(question)[:3]:
        try:
            kw_kwargs: dict = dict(
                query_texts=[question],
                n_results=config.TOP_K,
                include=["ids", "documents", "metadatas", "distances"],
                where_document={"$contains": term},
            )
            if where:
                kw_kwargs["where"] = where
            kw = collection.query(**kw_kwargs)
            for cid, doc, meta, dist in zip(
                kw["ids"][0], kw["documents"][0], kw["metadatas"][0], kw["distances"][0]
            ):
                if doc not in seen_docs:
                    extra_ids.append(cid)
                    extra_docs.append(doc)
                    extra_metas.append(meta)
                    extra_dists.append(dist)
                    seen_docs.add(doc)
        except Exception:
            pass

    return {
        "ids":       [results["ids"][0]       + extra_ids],
        "documents": [results["documents"][0]  + extra_docs],
        "metadatas": [results["metadatas"][0]  + extra_metas],
        "distances": [results["distances"][0]  + extra_dists],
    }


def _build_where(tag: str, tech: str) -> dict:
    where: dict = {}
    if tag:
        where["tags"] = {"$contains": tag}
    if tech:
        where["tech"] = {"$contains": tech}
    return where


# ── Scoring & re-ranking ──────────────────────────────────────────────────────

def score_and_rank(
    raw: dict,
    top_k: int,
    stats: dict,
) -> tuple[list[tuple[float, str, str, dict]], dict]:
    """Score all candidates and return the top_k.

    Returns:
        ranked     — list of (score, chunk_id, doc, meta), best first
        breakdowns — {chunk_id: breakdown_dict} for verbose logging
    """
    ids   = raw["ids"][0]
    docs  = raw["documents"][0]
    metas = raw["metadatas"][0]
    dists = raw["distances"][0]

    scored: list[tuple[float, str, str, dict]] = []
    breakdowns: dict[str, dict] = {}

    for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
        semantic = max(0.0, 1.0 - dist)
        final, bd = scorer.compute_score(chunk_id, meta, semantic, stats)
        scored.append((final, chunk_id, doc, meta))
        breakdowns[chunk_id] = bd

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k], breakdowns


# ── Multi-source context fusion ───────────────────────────────────────────────

_SECTION_LABELS: dict[str, str] = {
    "personal": "Personal Notes",
    "project":  "Project Notes",
    "official": "Official Documentation",
    "blog":     "External Resources",
}
_SOURCE_ORDER = ["personal", "project", "official", "blog"]


def build_context(ranked: list[tuple[float, str, str, dict]]) -> tuple[str, str]:
    """Group ranked chunks by source type and build a structured context.

    Section headers are only shown when chunks come from multiple source types.
    """
    buckets: dict[str, list] = {s: [] for s in _SOURCE_ORDER}
    for score, chunk_id, doc, meta in ranked:
        src = scorer.derive_source(meta)
        buckets.setdefault(src, []).append((score, doc, meta))

    active = [s for s in _SOURCE_ORDER if buckets[s]]
    multi_source = len(active) > 1

    context_sections: list[str] = []
    source_lines: list[str] = []
    counter = 1

    for src in active:
        chunk_parts: list[str] = []
        for score, doc, meta in buckets[src]:
            title  = meta.get("title", "Untitled")
            folder = meta.get("folder", "")
            chunk_parts.append(f"**[{counter}] {title}**\n\n{doc}")
            source_lines.append(
                f"{counter}. **{title}** — `{folder}` "
                f"(score: {score:.0%} · {src})"
            )
            counter += 1
        block = "\n\n".join(chunk_parts)
        if multi_source:
            context_sections.append(
                f"### {_SECTION_LABELS.get(src, src.title())}\n\n{block}"
            )
        else:
            context_sections.append(block)

    return "\n\n---\n\n".join(context_sections), "\n".join(source_lines)


# ── RAG pipeline ──────────────────────────────────────────────────────────────

def ask(
    question: str,
    top_k: int,
    tag: str,
    tech: str,
    verbose: bool,
    collection: Any = None,
) -> tuple[str, list, Any]:
    """Full RAG pipeline: intent → hybrid search → score → rank → fuse → answer.

    Returns (answer_text, ranked_chunks, collection).
    ranked_chunks and collection are passed to the learner when --learn is on.
    """
    if collection is None:
        collection = get_collection()
    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

    # 1. Auto-detect tech/tag when the user hasn't set explicit filters
    if not tag and not tech:
        auto_tag, auto_tech = extract_intent(question)
        if auto_tag or auto_tech:
            hints = " + ".join(filter(None, [
                f"tag:{auto_tag}"   if auto_tag  else "",
                f"tech:{auto_tech}" if auto_tech else "",
            ]))
            console.print(f"[dim]Intent detected: {hints}[/dim]")
        tag, tech = auto_tag, auto_tech

    # 2. Hybrid search (always RETRIEVAL_K=20 candidates)
    console.print(f"[dim]Searching (top {RETRIEVAL_K} candidates)...[/dim]")
    raw = hybrid_search(collection, question, tag, tech)
    n_candidates = len(raw["documents"][0])

    ranked: list = []

    if not n_candidates:
        console.print("[yellow]No relevant notes found — answering from general knowledge.[/yellow]")
        log.warning(f"No results for: {question!r}")
        context, sources = "No relevant notes found in the vault.", ""
    else:
        # 3. Score and re-rank
        stats = scorer.load_stats()
        ranked, breakdowns = score_and_rank(raw, top_k, stats)

        log.info(
            f"Scored {n_candidates} candidates → kept {len(ranked)} | "
            f"top: {ranked[0][3].get('title', '?')!r} ({ranked[0][0]:.3f})"
        )

        # 4. Verbose scoring table
        if verbose:
            _print_scoring_table(ranked, breakdowns, n_candidates, top_k)

        # 5. Multi-source context fusion
        context, sources = build_context(ranked)

        # 6. Feedback loop — update usage stats for retrieved chunks
        scorer.update_stats([r[1] for r in ranked])

    # 7. Call OpenAI
    user_message = USER_PROMPT.format(context=context, sources=sources, question=question)
    console.print(f"[dim]Generating answer ({config.CHAT_MODEL})...[/dim]\n")

    response = openai_client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.3,
    )
    answer = response.choices[0].message.content
    return answer, ranked, collection


def _print_scoring_table(
    ranked: list,
    breakdowns: dict,
    n_candidates: int,
    top_k: int,
) -> None:
    table = Table(
        title=f"Scored {n_candidates} candidates — top {top_k} selected",
        header_style="bold dim",
        show_edge=False,
        padding=(0, 1),
    )
    table.add_column("#",                    style="dim", width=3,  justify="right")
    table.add_column("Title",                             min_width=26)
    table.add_column("Source",                            width=9)
    table.add_column("Final",                             width=7,  justify="right")
    table.add_column("sem / meta",           style="dim", width=11, justify="right")
    table.add_column("conf / fresh / use",   style="dim", width=18, justify="right")

    for rank, (final, cid, _doc, meta) in enumerate(ranked, 1):
        bd = breakdowns.get(cid, {})
        table.add_row(
            str(rank),
            (meta.get("title") or "?")[:28],
            bd.get("source", "?"),
            f"{final:.3f}",
            f"{bd.get('semantic', 0):.2f} / {bd.get('metadata', 0):.2f}",
            f"{bd.get('confidence', 0):.2f} / {bd.get('freshness', 0):.2f} / {bd.get('usage', 0):.2f}",
        )
    console.print(table)
    console.print()


# ── CLI helpers ───────────────────────────────────────────────────────────────

def run_query(question: str, args: argparse.Namespace) -> None:
    console.print(Rule(f"[bold]{question}[/bold]"))

    answer, ranked, collection = ask(
        question=question,
        top_k=args.top_k,
        tag=args.tag or "",
        tech=args.tech or "",
        verbose=args.verbose,
    )
    console.print(Markdown(answer))
    console.print()

    # ── Auto-learning hook ────────────────────────────────────────────────────
    if args.learn:
        learner.process_interaction(
            question=question,
            answer=answer,
            ranked=ranked,
            collection=collection,
            confirm=not args.yes,
        )


def interactive_mode(args: argparse.Namespace) -> None:
    console.print(Panel(
        "[bold cyan]Developer Second Brain[/bold cyan]\n"
        "[dim]Ask anything from your vault. Type [bold]exit[/bold] to quit.[/dim]"
        + ("\n[dim](auto-learn enabled)[/dim]" if args.learn else ""),
        border_style="cyan",
    ))
    while True:
        try:
            question = console.input("\n[bold green]>[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break
        run_query(question, args)


def _require_api_key() -> None:
    if not config.OPENAI_API_KEY:
        console.print("[bold red]ERROR:[/] OPENAI_API_KEY is not set.")
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query your Developer Second Brain.")
    parser.add_argument("question", nargs="?",
                        help="Your question (omit for interactive mode).")
    parser.add_argument("--top-k",  type=int, default=config.TOP_K,
                        help=f"Chunks to keep after scoring (default: {config.TOP_K}).")
    parser.add_argument("--tag",    help="Override auto-detection: filter by tag.")
    parser.add_argument("--tech",   help="Override auto-detection: filter by technology.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show scoring table and retrieved context.")
    # ── Learning flags ────────────────────────────────────────────────────────
    parser.add_argument("--learn",  action="store_true", default=False,
                        help="Enable auto-learning: evaluate and optionally save this interaction.")
    parser.add_argument("--yes", "-y", action="store_true", default=False,
                        help="Auto-confirm saving when --learn is on (no prompt).")
    args = parser.parse_args()

    if args.question:
        run_query(args.question, args)
    else:
        interactive_mode(args)
