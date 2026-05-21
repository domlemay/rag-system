"""Parse Obsidian markdown files: extract YAML frontmatter and body text."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ParsedDocument:
    file_path: str
    folder: str      # relative folder inside the vault
    title: str
    content: str     # body text (frontmatter removed)
    tags: list = field(default_factory=list)
    tech: list = field(default_factory=list)
    level: str = ""
    date: str = ""
    related: list = field(default_factory=list)


def parse_markdown_file(file_path: Path, vault_path: Path) -> Optional[ParsedDocument]:
    """Read and parse one markdown file. Returns None on read/decode failure."""
    try:
        raw = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    metadata, content = _split_frontmatter(raw)
    if not content.strip():
        return None

    return ParsedDocument(
        file_path=str(file_path),
        folder=_relative_folder(file_path, vault_path),
        title=_extract_title(content) or file_path.stem,
        content=content.strip(),
        tags=_as_list(metadata.get("tags")),
        tech=_as_list(metadata.get("tech")),
        level=str(metadata.get("level", "")),
        date=str(metadata.get("date", "")),
        related=_as_list(metadata.get("related")),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Return (metadata_dict, body) from a markdown string with optional YAML front matter."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}, text
    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        metadata = {}
    return metadata, text[match.end():]


def _extract_title(content: str) -> Optional[str]:
    """Return text of the first H1 heading, or None."""
    m = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    return m.group(1).strip() if m else None


def _relative_folder(file_path: Path, vault_path: Path) -> str:
    try:
        return str(file_path.relative_to(vault_path).parent)
    except ValueError:
        return ""


def _as_list(value) -> list:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)] if value else []
