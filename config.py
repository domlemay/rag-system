"""Central configuration — reads from .env and sets project-wide constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
VAULT_PATH = BASE_DIR.parent / "dev-brain"
DB_PATH = BASE_DIR / "db" / "chroma"

# ── OpenAI ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# ── ChromaDB ───────────────────────────────────────────────────────────────────
COLLECTION_NAME = "dev_brain"

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800    # max tokens per chunk
CHUNK_OVERLAP = 150  # token overlap between adjacent chunks

# ── Query ──────────────────────────────────────────────────────────────────────
TOP_K = 5  # number of chunks to retrieve per query
