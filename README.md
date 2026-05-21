# Developer Second Brain — RAG System

A local RAG (Retrieval-Augmented Generation) system that turns your Obsidian vault into a queryable knowledge base powered by OpenAI embeddings and ChromaDB.

## Architecture

```text
dev-brain/              ← Obsidian vault (your notes)
rag-system/
  config.py             ← all settings (paths, models, chunk sizes)
  index.py              ← scans vault → chunks → embeddings → ChromaDB
  query.py              ← question → similarity search → OpenAI → answer
  utils/
    markdown_parser.py  ← parses YAML frontmatter + body text
    chunker.py          ← splits docs into token-bounded overlapping chunks
    logger.py           ← structured logging helper
  db/chroma/            ← persisted vector database (git-ignored)
```

**Flow:**

```text
Obsidian .md files
  → parse frontmatter (tags, tech, level…)
  → split into chunks (≤800 tokens, 150 overlap)
  → embed with text-embedding-3-small
  → store in ChromaDB (cosine similarity)
  → query: top-5 chunks → GPT-4o-mini → structured answer
```

## Setup

### 1. Prerequisites

- Python 3.11+
- An OpenAI API key

### 2. Install dependencies

```bash
cd rag-system
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
copy .env.example .env      # Windows
# cp .env.example .env      # Mac/Linux
```

Edit `.env` and add your key:

```env
OPENAI_API_KEY=sk-...
```

### 4. Index your vault

```bash
# First time — index everything
python index.py

# After adding/changing notes — only indexes new files
python index.py

# Wipe the DB and re-index from scratch
python index.py --reset

# Index one folder only
python index.py --folder 01-concepts
```

### 5. Query

```bash
# One-shot question
python query.py "How do I use async/await in Python?"

# Interactive REPL (keep asking questions)
python query.py

# Filter by tag
python query.py --tag async "how does the event loop work?"

# Filter by technology
python query.py --tech sqlalchemy "how to do joins"

# Show retrieved context before the answer
python query.py --verbose "repository pattern"

# Retrieve more chunks (default: 5)
python query.py --top-k 8 "dependency injection"
```

## Vault Structure

```text
dev-brain/
  _templates/
    note-template.md    ← copy this for every new note
  00-inbox/             ← quick captures, unsorted
  01-concepts/          ← fundamentals, theory, language features
  02-patterns/          ← design patterns, architecture
  03-snippets/          ← reusable code blocks
  04-errors/            ← error messages + fixes
  05-projects/          ← per-project notes and decisions
  06-resources/         ← links, book notes, references
  08-web-knowledge/     ← auto-generated notes from web ingestion
```

## Writing Notes

Copy `dev-brain/_templates/note-template.md` and fill in the frontmatter:

```yaml
---
tags: [async, python, concurrency]
tech: [python, asyncio]
level: intermediate        # beginner | intermediate | advanced
related: [threading, generators]
date: 2026-05-20
---
```

The `tags` and `tech` fields enable filtered queries:

```bash
python query.py --tag concurrency "how to cancel tasks?"
python query.py --tech asyncio "timeout pattern"
```

## Intelligent Query Pipeline

The query engine does three things automatically before calling OpenAI:

### 1. Intent Detection

When you don't pass `--tag` or `--tech`, the system scans your question for known technology and tag keywords and applies filters automatically:

```bash
python query.py "how do async timeouts work in asyncio?"
# → Intent detected: tag:async + tech:asyncio
```

You can always override with explicit flags:

```bash
python query.py --tech python "timeout pattern"   # explicit override
```

### 2. Hybrid Search

Every query runs two passes:

1. **Semantic pass** — fetches 3× `top_k` chunks by cosine similarity
2. **Keyword pass** — finds additional chunks that literally contain key terms from your question and aren't already in the semantic results

Both sets are merged and deduplicated before ranking.

### 3. Context Re-ranking

After retrieval, chunks are re-scored:

| Signal | Weight |
| --- | --- |
| Cosine similarity (embedding) | base score |
| Keyword overlap with your question | up to +15% |
| Recency (notes from the past year) | up to +5% |

The top `top_k` chunks after re-ranking are sent to the model.

## Web Ingestion

Ingest any web page directly into your vault as a structured Markdown note.

```bash
# Install new dependency first (one time)
pip install -r requirements.txt

# Ingest a page
python ingest.py https://docs.python.org/3/library/asyncio-task.html
python ingest.py https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function
python ingest.py https://fastapi.tiangolo.com/tutorial/background-tasks/
```

The pipeline:

1. Fetches the page and strips ads/navigation/noise with `trafilatura`
2. Sends the extracted text to OpenAI for structured extraction:
   - title, tags, tech
   - summary (2-3 paragraphs)
   - key concepts
   - code examples (verbatim from the page)
   - warnings and gotchas
   - source type + confidence score
3. Saves a Markdown note to `dev-brain/08-web-knowledge/<topic>.md`
4. Logs the URL to `db/ingested_urls.txt` — re-running the same URL is silently skipped

Then index the new note:

```bash
python index.py    # incremental — only picks up the new file
```

### Source confidence

| Source type | Confidence range |
| --- | --- |
| Official documentation | 90–100% |
| Quality tutorial | 70–89% |
| Blog post | 50–69% |
| Low quality / unknown | < 50% |

The confidence score is stored in the note frontmatter and visible in the note header.

### Generated note format

```markdown
---
tags: [async, python]
tech: [python, asyncio]
source: https://...
source_type: official_docs
confidence: 95
date: 2026-05-20
---

# Asyncio Task Cancellation

> Source: docs.python.org | Type: official_docs | Confidence: 95%

## Summary
## Key Concepts
## Code Examples
## Warnings
## My Understanding   ← fill this in yourself
```

## Updating the Index

The indexer is **incremental by default** — re-running `python index.py` only processes files that haven't been embedded yet (identified by a stable hash-based chunk ID). Use `--reset` to rebuild from scratch after bulk edits.

## Configuration

All tunable values live in `config.py`:

| Variable | Default | Description |
| --- | --- | --- |
| `VAULT_PATH` | `../dev-brain` | Path to your Obsidian vault |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `CHUNK_SIZE` | `800` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `150` | Token overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |

Override any value via `.env` (for API keys and model names) or edit `config.py` directly.
