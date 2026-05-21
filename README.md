# Developer Second Brain — RAG System

A local AI augmentation system that turns your Obsidian knowledge base into an active coding assistant. It connects to any IDE, enriches prompts automatically, and learns from every session.

---

## System Overview

```text
dev-brain/                    ← Your markdown knowledge base
  core/                       ← Principles + coding standards
  knowledge/                  ← Backend, frontend, DB patterns
  patterns/                   ← Architecture + API conventions
  bugs/                       ← Known errors + fixes
  learning/                   ← Lessons learned
  decision/                   ← Tech choices + tradeoffs
  context/                    ← Active stack + projects
  00-08-*/                    ← RAG-indexed note vault

rag-system/                   ← This repository
  api/
    server.py                 ← FastAPI server (port 8765)
    context_builder.py        ← Retrieval pipeline
    prompt_enricher.py        ← Prompt enrichment
  middleware/
    enrich.py                 ← CLI middleware (pipe-friendly)
  scripts/
    auto_learn.py             ← Standalone learning CLI
    install_git_hook.py       ← Git hook installer
  utils/
    chunker.py                ← Token-bounded document splitter
    scorer.py                 ← Advanced re-ranking with metadata
    learner.py                ← Q&A → vault note extractor
    web_fetcher.py            ← Web page ingestion
    markdown_parser.py        ← YAML frontmatter parser
  docs/
    ide-integration.md        ← Full IDE integration guide
  config.py                   ← All settings
  index.py                    ← Build/update vector index
  query.py                    ← CLI query interface
  ingest.py                   ← Web page ingestion CLI
```

### Data Flow

```text
Question
  ↓
Intent detection (auto-tag, auto-tech)
  ↓
Hybrid search (semantic × 20 + keyword boost)
  ↓
Re-ranking (semantic × 0.5 + metadata × 0.5)
  ↓
Context fusion (grouped by source type)
  ↓
  ├─→ /query endpoint   → raw context (no LLM)
  ├─→ /enrich endpoint  → enriched prompt
  └─→ query.py CLI      → OpenAI answer
```

---

## Quick Start (5 minutes)

### 1. Prerequisites

- Python 3.11+
- OpenAI API key (get one at platform.openai.com)

### 2. Install

```bash
cd rag-system
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure

```bash
copy .env.example .env    # Windows
cp .env.example .env      # macOS/Linux
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...
```

### 4. Index your vault

```bash
python index.py
```

Output: `Indexed 342 chunks from 89 files.`

### 5. Test it

```bash
# Ask a question
python query.py "How do I handle async timeouts in Python?"

# Start the API server
uvicorn api.server:app --host 127.0.0.1 --port 8765

# Enrich a prompt
python middleware/enrich.py "How do I handle async timeouts?"
```

---

## RAG Query Interface

### CLI Usage

```bash
# One-shot question
python query.py "How do I use the repository pattern?"

# Interactive REPL
python query.py

# Filter by tag
python query.py --tag async "event loop patterns"

# Filter by technology
python query.py --tech sqlalchemy "how to do joins"

# Show scoring details
python query.py --verbose "dependency injection"

# Auto-extract learnings from the response
python query.py --learn "how do I cancel async tasks?"

# Auto-learn without prompting (for scripts)
python query.py --learn --yes "asyncio timeout pattern"

# Retrieve more context
python query.py --top-k 10 "architecture patterns"
```

### Retrieval Pipeline

Every query runs:

1. **Intent detection** — scans for tech/tag keywords, auto-applies filters
2. **Hybrid search** — semantic pass (top 20) + keyword-boosted pass
3. **Re-ranking** — `final = semantic × 0.5 + metadata × 0.5`
4. **Context fusion** — groups results by source (personal → official → blog)
5. **Feedback loop** — increments usage count per chunk (improves future ranking)

**Scoring formula:**

```text
metadata_score = confidence×0.4 + freshness×0.2 + usage×0.2 + quality×0.2 + source_boost
final_score    = semantic_similarity×0.5 + min(metadata_score, 1.0)×0.5

Source boosts: personal +0.20 | project +0.15 | official +0.10 | blog +0.00
```

---

## Local API Server

### Start

```bash
uvicorn api.server:app --host 127.0.0.1 --port 8765
```

Visit `http://127.0.0.1:8765/docs` for interactive API documentation.

### Endpoints

#### `GET /health`

```json
{
  "status": "ok",
  "index_ready": true,
  "document_count": 342,
  "embedding_model": "text-embedding-3-small"
}
```

#### `POST /query` — Retrieval only (no LLM)

```bash
curl -s http://127.0.0.1:8765/query \
  -H "Content-Type: application/json" \
  -d '{"query": "async timeout pattern", "top_k": 5}'
```

```json
{
  "context": "[1] Asyncio Timeout Pattern\n\nUse asyncio.wait_for()...",
  "sources": [{"title": "Asyncio Timeout Pattern", "score": 0.87, "source_type": "personal"}],
  "chunks_used": 3,
  "query": "async timeout pattern"
}
```

#### `POST /enrich` — Full enriched prompt

```bash
curl -s http://127.0.0.1:8765/enrich \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do I implement retry logic for HTTP calls?"}'
```

```json
{
  "enriched_prompt": "=== CONTEXT FROM DEVELOPER BRAIN ===\n\n...\n\n=== USER REQUEST ===\n\nHow do I implement...",
  "context_added": true,
  "chunks_used": 4,
  "sources": [...]
}
```

---

## CLI Middleware

The middleware script enriches any prompt with vault context, with no server required.

```bash
# Direct (imports in-process — no server needed)
python middleware/enrich.py "How do I handle database transactions?"

# Via API (requires server running)
python middleware/enrich.py --api "repository pattern"

# Context block only
python middleware/enrich.py --context-only "async patterns"

# JSON output
python middleware/enrich.py --json "error handling"

# Pipe from stdin
echo "How do I use Pydantic validators?" | python middleware/enrich.py

# Copy to clipboard
python middleware/enrich.py "your question" | clip        # Windows
python middleware/enrich.py "your question" | pbcopy      # macOS
```

**Shell aliases (add to your profile):**

```bash
# Bash/Zsh — ~/.bashrc or ~/.zshrc
alias rag='python ~/rag-system/middleware/enrich.py'
alias ragclip='python ~/rag-system/middleware/enrich.py "$@" | pbcopy'
```

```powershell
# PowerShell profile
function rag { python $HOME\rag-system\middleware\enrich.py @args }
function ragclip { python $HOME\rag-system\middleware\enrich.py @args | Set-Clipboard }
```

---

## IDE Integration

Full integration guides in [`docs/ide-integration.md`](docs/ide-integration.md).

### Quick Summary

| IDE | Method | Automation Level |
| --- | --- | --- |
| VS Code + Continue.dev | HTTP context provider | Automatic — runs on every message |
| Cursor | `.cursorrules` + middleware | Semi-auto — rules are static, use middleware for dynamic |
| JetBrains | HTTP context provider (AI Pro) or file-based | Auto with AI Pro / manual otherwise |
| Visual Studio | File-based + Copilot Chat | Manual copy-paste |
| Claude Code | `CLAUDE.md` | Per-session static context |
| **Any IDE** | Clipboard middleware (`ragclip`) | Manual — one command |

---

## Auto-Learning System

The system can extract structured knowledge from any input and save it to your vault.

### Standalone CLI

```bash
# From text
python scripts/auto_learn.py "I learned that asyncio.gather cancels on first exception by default."

# From a file (conversation export, session notes)
python scripts/auto_learn.py --from-file conversation.txt

# From current git diff
python scripts/auto_learn.py --from-diff

# From last commit (message + diff)
python scripts/auto_learn.py --from-commit

# Auto-save without prompting
python scripts/auto_learn.py --yes --from-commit

# Preview extraction without saving
python scripts/auto_learn.py --dry-run "some explanation..."
```

### During RAG Queries

```bash
# Add --learn to any query — extracts knowledge from the Q&A
python query.py --learn "how do I cancel async tasks?"

# Auto-save without confirmation
python query.py --learn --yes "repository pattern"
```

### Git Hook (Automatic)

Install the post-commit hook to auto-extract from every commit:

```bash
# Install in dev-brain repo
python scripts/install_git_hook.py --target ../dev-brain

# Install in any other project
python scripts/install_git_hook.py --target /path/to/project

# Skip for one commit
DEV_BRAIN_LEARN=0 git commit -m "chore: ..."  # Unix
$env:DEV_BRAIN_LEARN=0; git commit -m "..."   # PowerShell
```

### Learning Pipeline

```text
Input (text/diff/commit)
  ↓
Heuristic gate (code blocks, PS terms, length — no API call)
  ↓
OpenAI extraction (title, category, tags, summary, code, insights)
  ↓
Duplicate detection (ChromaDB similarity > 0.88 → skip)
  ↓
User confirmation (or --yes to skip)
  ↓
Vault note → correct folder (01-concepts/ 02-patterns/ 03-snippets/ 04-errors/ 07-ai-learnings/)
  ↓
python index.py  ← run to add the note to the vector DB
```

### After Learning

Always re-index after new notes are saved:

```bash
python index.py   # incremental — only indexes new files
```

---

## Web Ingestion

Add any web page to your vault as a structured note:

```bash
python ingest.py https://docs.python.org/3/library/asyncio-task.html
python ingest.py https://fastapi.tiangolo.com/tutorial/background-tasks/
python ingest.py https://martinfowler.com/articles/injection.html
```

The pipeline:

1. Fetches and strips noise (ads, nav, footers) with `trafilatura`
2. Sends to OpenAI for structured extraction: title, tags, summary, code, warnings
3. Saves to `dev-brain/08-web-knowledge/<slug>.md`
4. Logs URL to `db/ingested_urls.txt` — re-running the same URL is silently skipped

Then index:

```bash
python index.py
```

---

## Vault Structure

### RAG-Indexed Folders

```text
dev-brain/
  00-inbox/         Quick captures, unsorted
  01-concepts/      Fundamentals, language features, theory
  02-patterns/      Design patterns, architecture patterns
  03-snippets/      Reusable code blocks
  04-errors/        Error messages + fixes
  05-projects/      Per-project notes (boosted in scoring)
  06-resources/     Links, book notes, external references
  07-ai-learnings/  Auto-generated from Q&A sessions
  08-web-knowledge/ Auto-generated from web ingestion
```

### Note Format

Every note uses YAML frontmatter for filtering and scoring:

```markdown
---
tags: [async, python, concurrency]
tech: [python, asyncio]
level: intermediate
date: 2026-05-20
---

# Title

Content here...
```

### Writing Good Notes

- **Searchable titles** — use the exact error message or function name
- **Short entries** — < 20 lines per concept
- **Code examples** — a 3-line snippet beats a paragraph
- **Tags and tech** — enable filtered queries (`--tag async --tech python`)

---

## Indexing

```bash
# First time or after bulk changes
python index.py

# Incremental (default) — only new files
python index.py

# Wipe and rebuild from scratch
python index.py --reset

# Index one folder only
python index.py --folder 01-concepts
```

The indexer uses stable hash-based chunk IDs — re-running on unchanged files is free.

---

## Configuration

All settings in `config.py` (override via `.env`):

| Variable | Default | Override in |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | `.env` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | `.env` |
| `CHAT_MODEL` | `gpt-4o-mini` | `.env` |
| `VAULT_PATH` | `../dev-brain` | `config.py` |
| `CHUNK_SIZE` | `800` tokens | `config.py` |
| `CHUNK_OVERLAP` | `150` tokens | `config.py` |
| `TOP_K` | `5` chunks | `config.py` |

---

## Best Practices

**Keep notes short and searchable.** An accurate 50-line vault beats a bloated 500-line one. Cut anything unused for 3 months.

**Tag consistently.** Use the same tag vocabulary everywhere. Inconsistent tags fragment retrieval.

**Re-index after every batch of new notes.** `python index.py` is fast (incremental).

**Use `--learn` on productive query sessions.** Let the system capture what you just learned.

**Ingest official docs.** `python ingest.py <url>` on any doc page gives you queryable knowledge instantly.

**Monthly review.** Check `07-ai-learnings/` — promote good notes, delete stale ones, run `python index.py --reset`.

---

## Limitations

| Limitation | Impact | Workaround |
| --- | --- | --- |
| Requires OpenAI API key | Costs money (~$0.002 per 1M embedding tokens) | Use `-small` model; costs are minimal for personal use |
| ChromaDB is local only | No sync across machines | Commit `db/chroma/` or re-index on each machine |
| API server not persistent | Restarts needed after machine reboot | Add to startup tasks or use `--direct` mode in middleware |
| Learning requires manual re-index | New notes aren't searchable until indexed | Run `python index.py` after `--learn` sessions |
| Web ingestion quality varies | Some pages extract poorly | Check `08-web-knowledge/` notes and edit if needed |
