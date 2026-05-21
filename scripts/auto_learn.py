"""Standalone auto-learning CLI.

Extracts knowledge from various inputs and saves structured notes to the developer vault.
This is the manual trigger for the auto-learning system — use it after any session,
conversation, or code change that produced knowledge worth keeping.

Usage:
    # From raw text
    python scripts/auto_learn.py "I learned today that asyncio.gather cancels remaining tasks on first exception by default."

    # From a file (conversation export, session notes)
    python scripts/auto_learn.py --from-file session_notes.txt

    # From current git diff (what changed + why it changed)
    python scripts/auto_learn.py --from-diff

    # From last git commit (message + diff)
    python scripts/auto_learn.py --from-commit

    # Auto-save without prompting (for hooks/automation)
    python scripts/auto_learn.py --yes --from-commit

    # Preview extraction without saving
    python scripts/auto_learn.py --dry-run "explanation of some pattern..."

    # Set minimum relevance threshold
    python scripts/auto_learn.py --min-relevance 0.7 "borderline content..."
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

console = Console()

# ── Tuning ────────────────────────────────────────────────────────────────────

_DEFAULT_MIN_RELEVANCE = 0.60

_FOLDERS: dict[str, Path] = {
    "concept":  config.VAULT_PATH / "01-concepts",
    "snippet":  config.VAULT_PATH / "03-snippets",
    "error":    config.VAULT_PATH / "04-errors",
    "pattern":  config.VAULT_PATH / "02-patterns",
    "learning": config.VAULT_PATH / "07-ai-learnings",
}

# ── Extraction prompt ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a developer knowledge extractor. Given text from a developer (notes, conversation, commit, diff),
extract any knowledge worth keeping in a personal engineering knowledge base.

Return ONLY valid JSON with these exact fields:
{
  "title": "topic_in_snake_case (2-6 words)",
  "category": "concept | snippet | error | pattern | learning",
  "tags": ["tag1", "tag2"],
  "tech": ["python", "asyncio"],
  "summary": "One paragraph summary of the key knowledge",
  "problem": "What problem or question was addressed (empty string if not applicable)",
  "solution": "The key solution, approach, or insight",
  "code": "Most important code snippet (empty string if none)",
  "key_insights": ["One sentence insight", "..."],
  "confidence": 80,
  "relevance_score": 0.85,
  "worth_saving": true
}

Category rules:
  concept  — a concept, theory, or language feature explained
  snippet  — a reusable code pattern or recipe
  error    — an error message / gotcha and how to fix it
  pattern  — a design pattern or architectural decision
  learning — anything valuable that doesn't fit the above

worth_saving = false when content is too generic, trivial, or already common knowledge.
relevance_score guidance:
  0.8+      → definitely save (new concept, code pattern, error fix)
  0.5–0.8   → borderline (general explanation worth keeping)
  below 0.5 → skip

Output only JSON — no markdown fences, no explanation."""


# ── Git helpers ───────────────────────────────────────────────────────────────

def _git_diff() -> str:
    """Return staged + unstaged diff from the dev-brain repo."""
    try:
        staged   = subprocess.check_output(["git", "diff", "--cached"], cwd=_ROOT.parent, text=True)
        unstaged = subprocess.check_output(["git", "diff"],             cwd=_ROOT.parent, text=True)
        result   = (staged + unstaged).strip()
        if not result:
            # Try rag-system itself
            staged   = subprocess.check_output(["git", "diff", "--cached"], cwd=_ROOT, text=True)
            unstaged = subprocess.check_output(["git", "diff"],             cwd=_ROOT, text=True)
            result   = (staged + unstaged).strip()
        return result
    except subprocess.CalledProcessError:
        return ""


def _git_commit() -> str:
    """Return last commit message + diff."""
    for repo in [_ROOT.parent, _ROOT]:
        try:
            msg  = subprocess.check_output(
                ["git", "log", "-1", "--pretty=%B"], cwd=repo, text=True
            ).strip()
            diff = subprocess.check_output(
                ["git", "diff", "HEAD~1", "HEAD"], cwd=repo, text=True
            )
            if msg:
                return f"Commit message:\n{msg}\n\nDiff (first 3000 chars):\n{diff[:3000]}"
        except subprocess.CalledProcessError:
            continue
    return ""


# ── Core extraction ───────────────────────────────────────────────────────────

def _extract(text: str) -> dict:
    """Call OpenAI to extract structured knowledge from text."""
    if not config.OPENAI_API_KEY:
        console.print("[bold red]ERROR:[/] OPENAI_API_KEY is not set. Check .env file.")
        sys.exit(1)

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": text[:6000]},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ── Note rendering ────────────────────────────────────────────────────────────

def _render_note(data: dict) -> str:
    today         = date.today().isoformat()
    tags_yaml     = json.dumps(data.get("tags", []))
    tech_yaml     = json.dumps(data.get("tech", []))
    title_display = data.get("title", "untitled").replace("_", " ").title()
    code          = data.get("code", "").strip()
    code_md       = f"```\n{code}\n```" if code else "_No code extracted._"
    insights_md   = "\n".join(f"- {i}" for i in data.get("key_insights", []))

    return f"""\
---
tags: {tags_yaml}
tech: {tech_yaml}
level: intermediate
source: auto-learned
confidence: {data.get('confidence', 70)}
date: {today}
---

# {title_display}

> **Auto-generated** | Confidence: {data.get('confidence', 70)}%

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

## My Notes

<!-- Add your own thoughts, corrections, or extensions here -->
"""


def _save_note(data: dict) -> Path:
    """Write the note to the correct vault folder. Never overwrites existing files."""
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

    path.write_text(_render_note(data), encoding="utf-8")
    return path


# ── Display ───────────────────────────────────────────────────────────────────

def _preview(data: dict) -> None:
    title_display = data.get("title", "untitled").replace("_", " ").title()
    category      = data.get("category", "learning")
    folder        = _FOLDERS.get(category, _FOLDERS["learning"])
    tags          = ", ".join(data.get("tags", []))

    summary_short = data.get("summary", "")[:200]
    if len(data.get("summary", "")) > 200:
        summary_short += "…"

    console.print(Panel(
        f"[bold]{title_display}[/bold]\n"
        f"Category  : [cyan]{category}[/cyan]  →  [dim]{folder.name}/[/dim]\n"
        f"Tags      : {tags}\n"
        f"Confidence: {data.get('confidence', 0)}%\n"
        f"Relevance : {data.get('relevance_score', 0):.0%}\n\n"
        f"[dim]{summary_short}[/dim]",
        title="[bold cyan]Extracted Knowledge[/bold cyan]",
        border_style="cyan",
    ))


# ── Main logic ────────────────────────────────────────────────────────────────

def run(text: str, auto_save: bool, dry_run: bool, min_relevance: float) -> bool:
    """Extract knowledge from text and optionally save it. Returns True if saved."""
    if not text.strip():
        console.print("[yellow]Empty input — nothing to extract.[/yellow]")
        return False

    console.print("[dim]Extracting knowledge...[/dim]")

    try:
        data = _extract(text)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid JSON from OpenAI:[/] {exc}")
        return False
    except Exception as exc:
        console.print(f"[red]Extraction failed:[/] {exc}")
        return False

    if not data.get("worth_saving", False):
        console.print("[dim]Content deemed not worth saving (too generic or trivial).[/dim]")
        return False

    relevance = data.get("relevance_score", 0)
    if relevance < min_relevance:
        console.print(
            f"[dim]Relevance {relevance:.0%} below threshold {min_relevance:.0%} — skipping.[/dim]"
        )
        return False

    _preview(data)

    if dry_run:
        console.print("[yellow]Dry run — nothing saved.[/yellow]")
        return False

    if not auto_save:
        try:
            reply = console.input("  Save? [Y/n] ").strip().lower()
            if reply in ("n", "no"):
                console.print("[dim]Skipped.[/dim]")
                return False
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Aborted.[/dim]")
            return False

    path = _save_note(data)
    console.print(f"[bold green]Saved![/] → [dim]{path}[/dim]")
    console.print("[dim]Run [bold]python index.py[/bold] to add to the vector DB.[/dim]")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and save developer knowledge to the vault.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("text",           nargs="?", help="Text to extract knowledge from.")
    parser.add_argument("--from-file",    metavar="FILE", help="Read input from a file.")
    parser.add_argument("--from-diff",    action="store_true",
                        help="Extract from current git diff (staged + unstaged).")
    parser.add_argument("--from-commit",  action="store_true",
                        help="Extract from last git commit message + diff.")
    parser.add_argument("--yes", "-y",    action="store_true",
                        help="Auto-save without prompting.")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Show extracted content without saving.")
    parser.add_argument("--min-relevance", type=float, default=_DEFAULT_MIN_RELEVANCE,
                        help=f"Minimum relevance score (default: {_DEFAULT_MIN_RELEVANCE}).")
    args = parser.parse_args()

    if args.from_diff:
        text = _git_diff()
        if not text:
            console.print("[yellow]No changes found in git diff.[/yellow]")
            sys.exit(0)
        console.print("[dim]Analyzing git diff...[/dim]")
    elif args.from_commit:
        text = _git_commit()
        if not text:
            console.print("[yellow]Could not read git commit info.[/yellow]")
            sys.exit(0)
        console.print("[dim]Analyzing last commit...[/dim]")
    elif args.from_file:
        p = Path(args.from_file)
        if not p.exists():
            console.print(f"[red]File not found:[/] {args.from_file}")
            sys.exit(1)
        text = p.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("Provide text as argument, --from-file, --from-diff, or --from-commit.")

    run(text, auto_save=args.yes, dry_run=args.dry_run, min_relevance=args.min_relevance)


if __name__ == "__main__":
    main()
