# IDE Integration Guide

Connect your Developer Brain to any IDE or AI coding assistant.

---

## How It Works

The RAG system exposes two integration surfaces:

| Surface | How | Best for |
|---|---|---|
| **HTTP API** (`/enrich`) | POST request to `localhost:8765` | Continue.dev, custom scripts |
| **CLI middleware** (`middleware/enrich.py`) | Pipe stdin → enriched stdout | Shell-based workflows, any IDE with terminal tasks |
| **Direct Python import** | `from api.prompt_enricher import enrich` | Python-based extensions |

---

## Quick Start (Universal)

The middleware script works with any IDE that can run a terminal command:

```bash
# From rag-system/ root
python middleware/enrich.py "How do I handle async timeouts in Python?"
```

Output: enriched prompt with your personal knowledge context injected.

```bash
# Pipe to clipboard and paste into any AI tool
python middleware/enrich.py "your question" | clip          # Windows
python middleware/enrich.py "your question" | pbcopy        # macOS
python middleware/enrich.py "your question" | xclip -sel c  # Linux
```

---

## VS Code + Continue.dev

**What this does:** Continue.dev intercepts your chat messages and automatically calls the RAG API to inject context before sending to the LLM.

### Setup

**1. Install Continue.dev**

Install from the [VS Code marketplace](https://marketplace.visualstudio.com/items?itemName=Continue.continue).

**2. Start the RAG API server**

```bash
cd rag-system
uvicorn api.server:app --host 127.0.0.1 --port 8765
```

**3. Configure Continue.dev**

Open `~/.continue/config.json` (or `%USERPROFILE%\.continue\config.json` on Windows):

```json
{
  "models": [
    {
      "title": "Claude Sonnet",
      "provider": "anthropic",
      "model": "claude-sonnet-4-5",
      "apiKey": "$ANTHROPIC_API_KEY"
    }
  ],
  "contextProviders": [
    {
      "name": "http",
      "params": {
        "url": "http://127.0.0.1:8765/query",
        "title": "Developer Brain",
        "description": "Your personal engineering knowledge base",
        "displayTitle": "Dev Brain"
      }
    },
    {
      "name": "file",
      "params": {}
    },
    {
      "name": "codebase",
      "params": {}
    }
  ],
  "slashCommands": [
    {
      "name": "enrich",
      "description": "Enrich current prompt with Developer Brain context",
      "step": "HttpSlashCommand",
      "params": {
        "url": "http://127.0.0.1:8765/enrich"
      }
    }
  ]
}
```

**4. Usage in VS Code**

- In the Continue.dev chat panel, type `@Dev Brain your question`
- Or use `/enrich` to manually trigger context injection
- The RAG context appears automatically in the chat context

### Auto-start the API server

Add a VS Code task to start the server automatically (`Ctrl+Shift+P` → `Tasks: Configure Task`):

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Start Developer Brain API",
      "type": "shell",
      "command": "cd ${workspaceFolder}/../rag-system && uvicorn api.server:app --host 127.0.0.1 --port 8765",
      "isBackground": true,
      "problemMatcher": [],
      "group": "build",
      "presentation": {
        "reveal": "silent",
        "panel": "dedicated"
      },
      "runOptions": {
        "runOn": "folderOpen"
      }
    },
    {
      "label": "Enrich Prompt (RAG)",
      "type": "shell",
      "command": "python ${workspaceFolder}/../rag-system/middleware/enrich.py",
      "args": ["${input:userPrompt}"],
      "presentation": {
        "reveal": "always"
      }
    }
  ],
  "inputs": [
    {
      "id": "userPrompt",
      "type": "promptString",
      "description": "Enter your question to enrich with Dev Brain context"
    }
  ]
}
```

### Limitations

- Continue.dev's HTTP context provider sends a GET with `?query=...` — the `/query` endpoint handles this.
- The server must be running before VS Code opens. Use the task above to auto-start.
- Context is injected as a separate context block, not silently — you'll see it in the chat.

---

## Cursor

**What this does:** Cursor reads `.cursorrules` as persistent context for every chat and Cmd+K interaction. We inject your knowledge summaries there.

### Setup

**1. Create `.cursorrules` at your project root**

```
# Developer Brain Context

You have access to this developer's personal engineering knowledge base.
Apply these rules to every response:

## Coding Standards
- Python: type hints on all functions, async for all I/O, ruff + black formatting
- Commits: Conventional Commits (feat/fix/refactor/docs/chore)
- PRs: max 300 lines, never leave a branch open > 3 days
- Tests: test behavior, not implementation — tests must pass after refactors

## Architecture
- Layered: routes → services → repositories → models
- No raw DB queries in services — always use repository pattern
- Validate at boundaries only (HTTP routes, external APIs)
- Use Repository pattern for DB access

## Error Handling
- FastAPI: raise HTTPException with structured error body
- Error format: { "error": "code", "message": "human", "detail": {...} }
- Log at boundaries, not inside internal functions

## Key Stack
- Backend: Python/FastAPI/SQLAlchemy 2.0/Pydantic 2.0/Alembic
- Frontend: React 18/TypeScript/TanStack Query/Zustand/Tailwind
- DB: PostgreSQL with asyncpg

## When Asked About Patterns
Reference the developer's existing patterns before suggesting new ones.
Flag if a suggestion contradicts their established conventions.
```

**2. For per-session dynamic context**

Run the middleware and paste into Cursor chat:

```bash
python rag-system/middleware/enrich.py "your question" | clip
```

Then paste (`Ctrl+V`) as the first message in Cursor chat.

**3. Reference dev-brain files directly**

In Cursor chat:
```
@../dev-brain/patterns/api.md design a new POST /payments endpoint
@../dev-brain/knowledge/backend.md what service layer pattern should I use?
```

### Advanced: Cursor Docs Integration

Add your dev-brain files as a "Doc" in Cursor:

1. Open Cursor Settings → Docs
2. Add local path: `../dev-brain/` (if supported) or copy key files to project
3. Cursor will index them for `@Docs` references

### Limitations

- `.cursorrules` is static — it doesn't run the RAG pipeline dynamically.
- For dynamic context, use the middleware script + paste manually.
- Cursor doesn't natively support HTTP context providers (as of 2025).

---

## JetBrains (IntelliJ, PyCharm, WebStorm, Android Studio)

**What this does:** JetBrains AI Assistant can use HTTP context providers and custom prompts. We configure it to call the RAG API.

### Method 1 — AI Assistant HTTP Context (JetBrains AI Pro)

**1. Start the API server**

```bash
cd rag-system
uvicorn api.server:app --host 127.0.0.1 --port 8765
```

**2. Configure AI Assistant**

Go to `Settings → Tools → AI Assistant → Context`:

- Add a custom HTTP context provider:
  - URL: `http://127.0.0.1:8765/query`
  - Method: POST
  - Body template: `{"query": "{userInput}", "top_k": 5}`

This injects relevant vault context into every AI chat message automatically.

**3. Add a custom prompt template**

Go to `Settings → Tools → AI Assistant → Prompts`:

```
You are assisting a senior developer. Before answering, context from their
personal knowledge base has been injected above. Use it as primary reference.
Apply their exact coding standards, patterns, and architecture conventions.
If the context doesn't cover the question, say so and answer from general knowledge.
```

### Method 2 — File-Based Context

Open dev-brain files in editor tabs alongside your code — JetBrains AI considers
all open files as context.

```bash
# Open key files in your IDE
# In IntelliJ/PyCharm terminal:
idea ../dev-brain/core/coding-standards.md
idea ../dev-brain/patterns/api.md
```

Then ask AI Assistant: *"Apply coding-standards.md to this code."*

### Method 3 — Shell External Tool

Configure as an External Tool (`Settings → Tools → External Tools`):

| Field | Value |
|---|---|
| Name | `Enrich with Dev Brain` |
| Program | `python` |
| Arguments | `$ProjectFileDir$/../rag-system/middleware/enrich.py "$Prompt$"` |
| Working dir | `$ProjectFileDir$/../rag-system` |

Assign a keyboard shortcut (`Settings → Keymap → External Tools`).

### Limitations

- HTTP context provider requires JetBrains AI Pro (paid).
- File-based context doesn't do semantic search — it's full file injection.
- The External Tool approach requires manual invocation.

---

## Visual Studio (Windows)

**What this does:** Visual Studio uses GitHub Copilot for AI assistance. We use file-based context and VS tasks.

### Method 1 — GitHub Copilot Context Files

Keep dev-brain files open in editor tabs:

1. `File → Open → File...`
2. Open `../dev-brain/core/coding-standards.md`
3. Open `../dev-brain/patterns/api.md`

GitHub Copilot treats open files as context. It will apply your standards
when those files are visible.

### Method 2 — Copilot Chat with Manual Enrichment

**Setup a terminal task:**

1. `Tools → External Tools → Add`
2. Configure:
   - Title: `Enrich with Dev Brain`
   - Command: `python`
   - Arguments: `..\rag-system\middleware\enrich.py`
   - Initial directory: `..\rag-system`

**Usage:**

1. Run the External Tool → a terminal opens
2. Type your question → copy the enriched output
3. Paste into Copilot Chat: `Ctrl+Alt+I`

### Method 3 — `.github/copilot-instructions.md`

Create this file in your project root to give Copilot persistent instructions:

```markdown
# Copilot Instructions

## Developer Standards

This developer uses the following conventions — always apply them:

### Code Style
- C#: nullable reference types enabled, async/await everywhere for I/O
- Follow existing patterns in the codebase exactly
- No comments explaining what code does — only why (non-obvious decisions)

### Architecture
- Clean Architecture: Domain → Application → Infrastructure → Presentation
- Repository pattern for all data access
- Validate at service boundaries only

### Naming
- Handlers: {Action}{Entity}Handler
- Commands: {Action}{Entity}Command
- Queries: {Action}{Entity}Query
```

### Limitations

- Visual Studio's Copilot doesn't support external HTTP context providers.
- File injection and instructions are the only reliable methods.
- The enrichment task requires manual copy-paste.

---

## Claude Code (This Tool)

**What this does:** Claude Code reads `CLAUDE.md` automatically at session start. We configure it to load dev-brain context directly.

### Setup

Create `CLAUDE.md` in each project root:

```markdown
# Project Context

## Developer Brain
Personal engineering standards are in `../dev-brain/`.

Apply these files to all responses:
- `../dev-brain/SYSTEM_PROMPT.md` — behavior rules
- `../dev-brain/core/coding-standards.md` — all code must follow these
- `../dev-brain/patterns/api.md` — all API endpoints use these conventions
- `../dev-brain/knowledge/backend.md` — backend patterns for this project

## RAG System
For deep knowledge search:
```bash
python ../rag-system/query.py "your question"
```

## This Project
[describe your project here]
```

### Advanced: Claude Hooks

Add a hook to auto-enrich prompts before they reach Claude:

In `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'Hook active'"
          }
        ]
      }
    ]
  }
}
```

### Limitations

- Claude Code reads CLAUDE.md once per session — context is static per session.
- For real-time RAG, run `python ../rag-system/query.py "question"` from the terminal.

---

## Any IDE — Universal Clipboard Workflow

This workflow works with **any** AI tool, no configuration needed.

**1. Create a shell alias** (add to `~/.bashrc`, `~/.zshrc`, or PowerShell profile):

```bash
# Bash/Zsh
alias rag='python ~/rag-system/middleware/enrich.py'
alias ragclip='python ~/rag-system/middleware/enrich.py "$1" | pbcopy'  # macOS
alias ragclip='python ~/rag-system/middleware/enrich.py "$1" | clip'    # Windows
```

```powershell
# PowerShell
function rag { python $HOME\rag-system\middleware\enrich.py @args }
function ragclip { python $HOME\rag-system\middleware\enrich.py @args | Set-Clipboard }
```

**2. Usage:**

```bash
ragclip "How do I implement pagination in FastAPI?"
# → enriched prompt copied to clipboard

# Paste into any AI tool: Claude.ai, ChatGPT, Gemini, Copilot Chat
```

---

## API Reference

The server runs at `http://127.0.0.1:8765`.

### `GET /health`

```json
{
  "status": "ok",
  "index_ready": true,
  "document_count": 342,
  "embedding_model": "text-embedding-3-small",
  "top_k_default": 5
}
```

### `POST /query`

Request:
```json
{
  "query": "how do I handle async timeouts?",
  "top_k": 5
}
```

Response:
```json
{
  "context": "[1] Asyncio Timeout Pattern\n\nUse asyncio.wait_for()...",
  "sources": [
    { "title": "Asyncio Timeout Pattern", "folder": "01-concepts", "score": 0.87, "source_type": "personal" }
  ],
  "chunks_used": 3,
  "query": "how do I handle async timeouts?"
}
```

### `POST /enrich`

Request:
```json
{
  "prompt": "How do I implement a retry mechanism for HTTP calls?",
  "top_k": 5
}
```

Response:
```json
{
  "enriched_prompt": "=== CONTEXT FROM DEVELOPER BRAIN ===\n\n...\n\n=== USER REQUEST ===\n\nHow do I implement...",
  "context_added": true,
  "chunks_used": 4,
  "sources": [...]
}
```

### `GET /stats`

```json
{
  "total_chunks_tracked": 89,
  "top_chunks": [
    { "chunk_id": "abc123...", "usage_count": 14, "last_used": "2026-05-20T..." }
  ]
}
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Connection refused` on port 8765 | Start the server: `uvicorn api.server:app --port 8765` |
| `Index not found` error | Run `python index.py` first |
| `OPENAI_API_KEY not set` | Add key to `rag-system/.env` |
| Empty context returned | Add more notes to vault or re-run `python index.py` |
| Slow response | Reduce `top_k`, or use `--context-only` in middleware |
| Hook not triggering | Check that the hook file is executable: `chmod +x .git/hooks/post-commit` |
