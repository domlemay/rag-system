"""Auto-learning loop: turn Q&A interactions into structured vault notes.

Pipeline per interaction:
  1. Heuristic relevance check  (no API call — fast gate)
  2. OpenAI extraction          (single call: score + structure)
  3. Duplicate detection        (ChromaDB similarity check)
  4. User confirmation          (optional, skipped with --yes)
  5. Note saved to vault        (category-aware folder routing)
  6. Log updated + pattern check (suggest concept note at threshold)
"""

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from rich.console import Console

import config
from utils.logger import get_logger

console = Console()
log = get_logger("learner")

# ── Tuning ────────────────────────────────────────────────────────────────────

_HEURISTIC_THRESHOLD = 0.55   # minimum heuristic score to trigger OpenAI call
_AI_THRESHOLD        = 0.65   # minimum AI-judged relevance to proceed
_DUPE_THRESHOLD      = 0.88   # cosine similarity above which note is a duplicate
_PATTERN_THRESHOLD   = 3      # tag occurrences before suggesting a concept note

# ── Paths ─────────────────────────────────────────────────────────────────────

_LOG_PATH: Path = config.BASE_DIR / "db" / "ai_learnings_log.json"

_FOLDERS: dict[str, Path] = {
    "concept": config.VAULT_PATH / "01-concepts",
    "snippet": config.VAULT_PATH / "03-snippets",
    "error":   config.VAULT_PATH / "04-errors",
    "pattern": config.VAULT_PATH / "02-patterns",
    "learning": config.VAULT_PATH / "07-ai-learnings",
}

# ── OpenAI extraction prompt ──────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are a developer knowledge extractor. Given a Q&A developer interaction,
decide whether it is worth saving and extract the structured knowledge.

Return ONLY valid JSON with these exact fields:

{
  "title": "topic_in_snake_case (2-6 words)",
  "category": "concept | snippet | error | pattern | learning",
  "tags": ["tag1", "tag2"],
  "tech": ["python", "asyncio"],
  "summary": "One paragraph of the key knowledge gained",
  "problem": "What question or problem was being addressed",
  "solution": "The key solution, approach, or explanation",
  "code": "The most important code snippet (empty string if none)",
  "key_insights": ["One sentence insight worth remembering", "..."],
  "confidence": 80,
  "relevance_score": 0.85
}

Category rules:
- concept:  a concept, theory, or language feature explained
- snippet:  a reusable code pattern or recipe
- error:    an error message / gotcha and how to fix it
- pattern:  a design pattern or architectural decision
- learning: anything valuable that doesn't fit the above

relevance_score guidance:
  0.8+  → definitely save (new concept, code pattern, error fix)
  0.5–0.8 → borderline (general explanation)
  <0.5  → not worth saving (trivial, too generic)

Output only JSON — no markdown, no explanation."""


# ── Public API ────────────────────────────────────────────────────────────────

def process_interaction(
    question: str,
    answer: str,
    ranked: list,        # list of (score, chunk_id, doc, meta) from query.py
    collection: Any,     # ChromaDB collection for duplicate check
    confirm: bool = True,
) -> dict | None:
    """Evaluate a Q&A pair and optionally save it as a structured vault note.

    Returns a dict with {title, category, path} if saved, else None.
    """
    # ── Step 1: heuristic gate (no API call) ──────────────────────────────────
    h_score, worth_checking = _detect_relevance(answer)
    log.debug(f"Heuristic relevance: {h_score:.2f} (threshold={_HEURISTIC_THRESHOLD})")
    if not worth_checking:
        return None

    # ── Step 2: OpenAI extraction ─────────────────────────────────────────────
    try:
        data = _extract_knowledge(question, answer, _format_context(ranked))
    except Exception as exc:
        log.warning(f"Extraction failed: {exc}")
        return None

    ai_score = data.get("relevance_score", 0)
    log.debug(f"AI relevance: {ai_score:.2f} (threshold={_AI_THRESHOLD})")
    if ai_score < _AI_THRESHOLD:
        return None

    # ── Step 3: duplicate detection ───────────────────────────────────────────
    if _is_duplicate(data.get("summary", ""), collection):
        console.print("[dim]Similar note already in vault — skipping.[/dim]")
        return None

    # ── Step 4: user confirmation ─────────────────────────────────────────────
    if confirm and not _user_confirms(data):
        return None

    # ── Step 5: save note ─────────────────────────────────────────────────────
    try:
        note_path = _save_note(data, question)
    except Exception as exc:
        log.error(f"Failed to save note: {exc}")
        return None

    # ── Step 6: update log + pattern check ────────────────────────────────────
    note_info: dict[str, Any] = {
        "title":    data.get("title", ""),
        "category": data.get("category", "learning"),
        "tags":     data.get("tags", []),
        "path":     str(note_path),
        "date":     datetime.utcnow().isoformat(),
    }
    _update_log(note_info)
    log.info(f"Saved learning: {note_path.name}")
    console.print(f"[bold green]Saved![/] → [dim]{note_path}[/dim]")
    console.print(
        "[dim]Run [bold]python index.py[/bold] to add this note to the vector DB.[/dim]"
    )

    # Bonus: pattern detection
    suggestion = _check_patterns(data.get("tags", []))
    if suggestion:
        console.print(
            f"\n[bold yellow]Pattern detected:[/] You have {_PATTERN_THRESHOLD}+ notes "
            f"about [bold]{suggestion}[/bold]. "
            f"Consider creating a high-level concept note!"
        )

    return note_info


# ── Relevance detection (heuristic) ──────────────────────────────────────────

def _detect_relevance(answer: str) -> tuple[float, bool]:
    """Fast heuristic scoring — no API call. Returns (score, should_extract)."""
    score = 0.0

    # Code blocks are the strongest signal
    code_count = len(re.findall(r"```[\s\S]+?```", answer))
    score += min(0.35, code_count * 0.15)

    # Structured headings (organized, substantial response)
    heading_count = len(re.findall(r"^#{1,3} .+", answer, re.MULTILINE))
    score += min(0.20, heading_count * 0.06)

    # Problem-solving vocabulary
    ps_terms = [
        "fix", "solve", "error", "issue", "avoid", "warning",
        "pattern", "best practice", "important", "instead", "gotcha",
        "pitfall", "common mistake", "note that", "be careful",
    ]
    ps_hits = sum(1 for t in ps_terms if t in answer.lower())
    score += min(0.20, ps_hits * 0.04)

    # Response length (longer = more substantial)
    score += 0.10 if len(answer) > 800  else 0.0
    score += 0.10 if len(answer) > 1500 else 0.0

    # Bullet lists (structured knowledge)
    list_items = len(re.findall(r"^[-*] .+", answer, re.MULTILINE))
    score += min(0.10, list_items * 0.02)

    score = min(1.0, score)
    return score, score >= _HEURISTIC_THRESHOLD


# ── Knowledge extraction ──────────────────────────────────────────────────────

def _extract_knowledge(question: str, answer: str, context_text: str) -> dict:
    """Single OpenAI call: judge relevance + extract structured knowledge."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    answer_trimmed = answer[:5000] + ("\n[truncated]" if len(answer) > 5000 else "")

    user_msg = (
        f"## User Question\n\n{question}\n\n"
        f"## AI Answer\n\n{answer_trimmed}\n\n"
        f"## Retrieved Context (from RAG system)\n\n{context_text[:1500]}"
    )

    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from OpenAI: {raw[:200]}") from exc


def _format_context(ranked: list) -> str:
    """Summarise ranked chunks for the extractor prompt."""
    if not ranked:
        return "No documents retrieved (AI answered from general knowledge)."
    parts = []
    for _score, _cid, doc, meta in ranked[:3]:
        title = meta.get("title", "Untitled")
        parts.append(f"[{title}]: {doc[:300]}…")
    return "\n\n".join(parts)


# ── Duplicate detection ───────────────────────────────────────────────────────

def _is_duplicate(summary: str, collection: Any) -> bool:
    """Return True if a note with very similar content already exists."""
    if not summary:
        return False
    try:
        results = collection.query(
            query_texts=[summary],
            n_results=1,
            include=["distances"],
        )
        if results["distances"] and results["distances"][0]:
            similarity = 1.0 - results["distances"][0][0]
            log.debug(f"Nearest existing note similarity: {similarity:.3f}")
            return similarity > _DUPE_THRESHOLD
    except Exception as exc:
        log.debug(f"Duplicate check skipped: {exc}")
    return False


# ── User confirmation ─────────────────────────────────────────────────────────

def _user_confirms(data: dict) -> bool:
    """Show note preview and ask the user whether to save."""
    title_display = data.get("title", "untitled").replace("_", " ").title()
    category      = data.get("category", "learning")
    folder        = _FOLDERS.get(category, _FOLDERS["learning"])
    tags          = ", ".join(data.get("tags", []))

    console.print(f"\n[bold cyan]Auto-learn:[/] Interesting response detected!")
    console.print(f"  Title    : [bold]{title_display}[/]")
    console.print(f"  Category : {category}  →  [dim]{folder.name}/[/dim]")
    console.print(f"  Tags     : {tags}")
    console.print(f"  Confidence: {data.get('confidence', 0)}%")
    try:
        reply = console.input("  Save? [Y/n] ").strip().lower()
        return reply not in ("n", "no")
    except (KeyboardInterrupt, EOFError):
        return False


# ── Note rendering ────────────────────────────────────────────────────────────

def _render_note(data: dict, question: str) -> str:
    today         = date.today().isoformat()
    tags_yaml     = json.dumps(data.get("tags", []))
    tech_yaml     = json.dumps(data.get("tech", []))
    title_display = data.get("title", "untitled").replace("_", " ").title()

    code = data.get("code", "").strip()
    code_md = f"```\n{code}\n```" if code else "_No code extracted._"

    insights_md = "\n".join(f"- {i}" for i in data.get("key_insights", []))

    return f"""\
---
tags: {tags_yaml}
tech: {tech_yaml}
level: intermediate
source: ai-generated
confidence: {data.get('confidence', 70)}
date: {today}
---

# {title_display}

> **Auto-generated** from AI interaction | Confidence: {data.get('confidence', 70)}%

## Summary

{data.get('summary', '')}

## Problem

{data.get('problem', '')}

## Solution

{data.get('solution', '')}

## Code

{code_md}

## Key Insights

{insights_md}

## Original Query

> {question}

## My Notes

<!-- Add your own thoughts, corrections, or extensions here -->

"""


# ── Note storage ──────────────────────────────────────────────────────────────

def _save_note(data: dict, question: str) -> Path:
    """Write the note to the right vault folder, never overwriting existing files."""
    category = data.get("category", "learning")
    folder   = _FOLDERS.get(category, _FOLDERS["learning"])
    folder.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^\w]+", "_", data.get("title", "untitled").lower()).strip("_")[:60]
    path = folder / f"{slug}.md"

    if path.exists():
        for i in range(2, 1000):
            path = folder / f"{slug}_{i}.md"
            if not path.exists():
                break

    path.write_text(_render_note(data, question), encoding="utf-8")
    return path


# ── Log management ────────────────────────────────────────────────────────────

def _load_log() -> dict:
    if not _LOG_PATH.exists():
        return {"tag_counts": {}, "entries": []}
    try:
        return json.loads(_LOG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"tag_counts": {}, "entries": []}


def _update_log(note_info: dict) -> None:
    log_data = _load_log()
    log_data["entries"].append(note_info)
    for tag in note_info.get("tags", []):
        log_data["tag_counts"][tag] = log_data["tag_counts"].get(tag, 0) + 1
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LOG_PATH.write_text(json.dumps(log_data, indent=2), encoding="utf-8")


# ── Pattern detection (bonus) ─────────────────────────────────────────────────

def _check_patterns(tags: list[str]) -> str | None:
    """Return a tag if it has reached _PATTERN_THRESHOLD in the log.

    Called after _update_log so counts already include the just-saved note.
    """
    log_data   = _load_log()
    tag_counts = log_data.get("tag_counts", {})
    for tag in tags:
        if tag_counts.get(tag, 0) >= _PATTERN_THRESHOLD:
            return tag
    return None
