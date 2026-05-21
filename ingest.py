"""
ingest.py — Ingest a web page into your Developer Second Brain.

Usage:
    python ingest.py <url>
    python ingest.py https://docs.python.org/3/library/asyncio-task.html

The page is fetched, summarised by AI, and saved as a structured Markdown note
in dev-brain/08-web-knowledge/. Already-ingested URLs are skipped automatically.
After ingestion, run `python index.py` to add the new note to the vector DB.
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

import config
from utils import get_logger
from utils.web_fetcher import fetch

console = Console()
log = get_logger("ingest")

# ── Paths ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR: Path = config.VAULT_PATH / "08-web-knowledge"
INGESTED_LOG: Path = config.BASE_DIR / "db" / "ingested_urls.txt"

# ── AI extraction prompt ──────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are a developer knowledge curator. Given the extracted text of a web page, produce a
structured JSON summary that a developer can store in their personal knowledge base.

Return ONLY a valid JSON object — no markdown, no explanation — with these exact fields:

{
  "title": "topic_in_snake_case (2-6 words, used as filename)",
  "tags": ["tag1", "tag2"],
  "tech": ["python", "asyncio"],
  "summary": "Two or three paragraphs. What is this page about? What is actionable?",
  "key_concepts": ["Concept name: one-sentence explanation", "..."],
  "code_examples": ["```python\\ncode here\\n```", "..."],
  "warnings": ["Gotcha or deprecation mentioned on the page", "..."],
  "source_type": "official_docs | tutorial | blog | stackoverflow | low_quality",
  "confidence": 85
}

Field rules:
- title: snake_case, e.g. "asyncio_task_cancellation"
- tags / tech: 2-6 items each, lowercase
- key_concepts: 3-8 items, each starts with a concept name followed by a colon
- code_examples: verbatim code blocks from the page; empty list if none
- warnings: real gotchas/deprecations mentioned; empty list if none
- confidence: 90+ official docs, 70-89 quality tutorials, 50-69 blogs, <50 low quality
"""


# ── Main pipeline ──────────────────────────────────────────────────────────────

def ingest(url: str) -> None:
    _require_api_key()

    # Safety: skip already-ingested URLs
    if _already_ingested(url):
        console.print(f"[yellow]Skipped (already ingested):[/] {url}")
        return

    console.print(f"\n[bold cyan]Ingesting[/] {url}\n")

    # 1. Fetch + extract readable text
    console.print("[dim]Fetching page...[/dim]")
    try:
        page = fetch(url)
    except ValueError as exc:
        console.print(f"[bold red]Fetch error:[/] {exc}")
        sys.exit(1)

    console.print(f"[dim]Extracted {len(page.raw_text):,} chars from {page.domain} "
                  f"({page.url_source_hint})[/dim]")

    # 2. AI extraction
    console.print("[dim]Processing with AI...[/dim]")
    try:
        data = _extract_with_openai(page.raw_text, page.url_source_hint)
    except ValueError as exc:
        console.print(f"[bold red]AI error:[/] {exc}")
        sys.exit(1)

    # 3. Write note
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    note_path = _safe_output_path(data["title"])
    note_path.write_text(_render_note(data, url, page.domain), encoding="utf-8")

    # 4. Log ingestion
    _record_ingestion(url)
    log.info(f"Ingested {url} → {note_path}")

    console.print(Panel(
        f"[bold green]Saved![/]\n"
        f"File      : [bold]{note_path.name}[/]\n"
        f"Source    : {data.get('source_type', '?')}  |  "
        f"Confidence: [bold]{data.get('confidence', 0)}%[/]\n"
        f"Tags      : {', '.join(data.get('tags', [])[:5])}\n\n"
        f"[dim]Run [bold]python index.py[/bold] to add this note to the vector DB.[/dim]",
        border_style="green",
        title="[bold]Web Note Created[/bold]",
    ))


# ── AI processing ─────────────────────────────────────────────────────────────

def _extract_with_openai(text: str, url_hint: str) -> dict:
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    # Keep first 8 000 chars — enough for the key content without burning tokens
    trimmed = text[:8000] + ("\n\n[content truncated]" if len(text) > 8000 else "")
    user_msg = f"URL classification hint: {url_hint}\n\n---\n\n{trimmed}"

    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"AI returned invalid JSON: {raw[:300]}") from exc


# ── Note rendering ────────────────────────────────────────────────────────────

def _render_note(data: dict, url: str, domain: str) -> str:
    today = date.today().isoformat()
    source_type = data.get("source_type", "unknown")
    confidence = data.get("confidence", 0)

    tags_yaml = json.dumps(data.get("tags", []))
    tech_yaml = json.dumps(data.get("tech", []))

    concepts_md = "\n".join(f"- {c}" for c in data.get("key_concepts", []))

    code_blocks = data.get("code_examples", [])
    code_md = "\n\n".join(code_blocks) if code_blocks else "_No code examples extracted._"

    warnings = data.get("warnings", [])
    warnings_md = "\n".join(f"- {w}" for w in warnings) if warnings else "_None noted._"

    display_title = data.get("title", "untitled").replace("_", " ").title()

    return f"""\
---
tags: {tags_yaml}
tech: {tech_yaml}
level: intermediate
source: {url}
source_type: {source_type}
confidence: {confidence}
date: {today}
---

# {display_title}

> **Source**: [{domain}]({url}) | Type: `{source_type}` | Confidence: **{confidence}%**

## Summary

{data.get("summary", "")}

## Key Concepts

{concepts_md}

## Code Examples

{code_md}

## Warnings

{warnings_md}

## My Understanding

<!-- Add your own notes and insights here. What surprised you? What would you do differently? -->

"""


# ── Safety helpers ────────────────────────────────────────────────────────────

def _safe_output_path(title: str) -> Path:
    """Return a non-existing path, appending _2, _3, … if the file already exists."""
    slug = re.sub(r"[^\w]+", "_", title.lower()).strip("_")[:60]
    candidate = OUTPUT_DIR / f"{slug}.md"
    if not candidate.exists():
        return candidate
    for i in range(2, 1000):
        candidate = OUTPUT_DIR / f"{slug}_{i}.md"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Too many notes named '{slug}' — clean up manually.")


def _already_ingested(url: str) -> bool:
    if not INGESTED_LOG.exists():
        return False
    return url in INGESTED_LOG.read_text(encoding="utf-8").splitlines()


def _record_ingestion(url: str) -> None:
    INGESTED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with INGESTED_LOG.open("a", encoding="utf-8") as f:
        f.write(url + "\n")


def _require_api_key() -> None:
    if not config.OPENAI_API_KEY:
        console.print("[bold red]ERROR:[/] OPENAI_API_KEY is not set.")
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a web page into your Developer Second Brain."
    )
    parser.add_argument("url", help="URL of the page to ingest.")
    args = parser.parse_args()
    ingest(args.url)
