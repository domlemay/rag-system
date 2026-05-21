"""Advanced document scoring and usage tracking for RAG re-ranking.

Scoring formula (per prompt spec):
    metadata_score = (confidence * 0.4) + (freshness * 0.2)
                   + (usage_norm * 0.2) + (quality * 0.2)
                   + source_boost

    final_score = semantic_similarity * 0.5 + clamp(metadata_score, 0, 1) * 0.5

Source boosts:  personal +0.20 | project +0.15 | official +0.10 | blog +0.00
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import config

# ── Persistence ───────────────────────────────────────────────────────────────

_STATS_PATH: Path = config.BASE_DIR / "db" / "usage_stats.json"

# ── Lookup tables ─────────────────────────────────────────────────────────────

_SOURCE_BOOSTS: dict[str, float] = {
    "personal": 0.20,
    "project":  0.15,
    "official": 0.10,
    "blog":     0.00,
}

# Default quality when the note has no explicit quality field
_QUALITY_DEFAULTS: dict[str, float] = {
    "personal": 1.00,   # user wrote it → high trust
    "project":  0.90,
    "official": 0.85,
    "blog":     0.65,
}

# Default confidence when the note has no explicit confidence field
_CONFIDENCE_DEFAULTS: dict[str, float] = {
    "personal": 1.00,
    "project":  0.90,
    "official": 0.90,
    "blog":     0.65,
}


# ── Source derivation ─────────────────────────────────────────────────────────

def derive_source(meta: dict) -> str:
    """Map chunk metadata to one of: personal | project | official | blog.

    Rules (in order):
    1. Folder contains '05-projects'  → project
    2. Folder contains '08-web-knowledge':
         source_type == 'official_docs'  → official
         otherwise                       → blog
    3. Everything else                  → personal
    """
    folder = str(meta.get("folder", ""))
    source_type = str(meta.get("source_type", ""))

    if "05-projects" in folder:
        return "project"
    if "08-web-knowledge" in folder:
        return "official" if source_type == "official_docs" else "blog"
    return "personal"


# ── Core scoring ──────────────────────────────────────────────────────────────

def compute_score(
    chunk_id: str,
    meta: dict,
    semantic_similarity: float,
    stats: dict,
) -> tuple[float, dict[str, Any]]:
    """Score a single chunk. Returns (final_score, breakdown_dict).

    breakdown_dict contains all intermediate values for logging/debugging.
    """
    source = derive_source(meta)

    # confidence: web notes store 0-100 int; personal notes default to 1.0
    raw_conf = meta.get("confidence", None)
    if raw_conf is not None:
        confidence = float(raw_conf) / 100.0 if float(raw_conf) > 1.0 else float(raw_conf)
    else:
        confidence = _CONFIDENCE_DEFAULTS.get(source, 0.80)

    freshness = _compute_freshness(str(meta.get("date", "")))

    usage_count = stats.get(chunk_id, {}).get("usage_count", 0)
    usage_norm = min(1.0, usage_count / 20.0)   # 20 uses = max score

    quality = float(meta.get("quality", _QUALITY_DEFAULTS.get(source, 0.75)))
    source_boost = _SOURCE_BOOSTS.get(source, 0.0)

    metadata_score = (
        confidence * 0.4
        + freshness  * 0.2
        + usage_norm * 0.2
        + quality    * 0.2
        + source_boost
    )
    metadata_score = min(1.0, metadata_score)

    final = semantic_similarity * 0.5 + metadata_score * 0.5

    breakdown: dict[str, Any] = {
        "source":     source,
        "semantic":   round(semantic_similarity, 3),
        "confidence": round(confidence, 3),
        "freshness":  round(freshness, 3),
        "usage":      round(usage_norm, 3),
        "quality":    round(quality, 3),
        "boost":      source_boost,
        "metadata":   round(metadata_score, 3),
        "final":      round(final, 3),
    }
    return final, breakdown


def _compute_freshness(date_str: str) -> float:
    """Linear decay: 1.0 today → 0.0 after 2 years. Returns 0.5 if date unknown."""
    if not date_str:
        return 0.5
    try:
        note_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        days_old = (date.today() - note_date).days
        return max(0.0, 1.0 - days_old / 730)
    except ValueError:
        return 0.5


# ── Usage stats (feedback loop) ───────────────────────────────────────────────

def load_stats() -> dict:
    """Return {chunk_id: {"usage_count": int, "last_used": str|None}} from disk."""
    if not _STATS_PATH.exists():
        return {}
    try:
        return json.loads(_STATS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def update_stats(chunk_ids: list[str]) -> None:
    """Increment usage_count and set last_used for every chunk in chunk_ids."""
    stats = load_stats()
    now = datetime.utcnow().isoformat()
    for cid in chunk_ids:
        entry = stats.setdefault(cid, {"usage_count": 0, "last_used": None})
        entry["usage_count"] += 1
        entry["last_used"] = now
    _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
