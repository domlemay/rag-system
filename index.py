"""
index.py — Scan the Obsidian vault and store all notes as vector embeddings in ChromaDB.

Usage:
    python index.py              # index only new/changed files (incremental)
    python index.py --reset      # wipe the DB and re-index everything from scratch
    python index.py --folder 01-concepts  # index one folder only
"""

import argparse
import sys

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

import config
from utils import chunk_document, get_logger, parse_markdown_file

console = Console()
log = get_logger("indexer")


def build_collection(reset: bool):
    """Connect to (or create) the ChromaDB collection with OpenAI embeddings."""
    _require_api_key()

    config.DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.DB_PATH))

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=config.OPENAI_API_KEY,
        model_name=config.EMBEDDING_MODEL,
    )

    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
            console.print(f"[yellow]Collection cleared:[/] {config.COLLECTION_NAME}")
        except Exception:
            pass  # collection didn't exist yet

    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def get_indexed_ids(collection) -> set[str]:
    """Return the set of chunk IDs already present in the collection."""
    try:
        return set(collection.get(include=[])["ids"])
    except Exception:
        return set()


def run(reset: bool = False, folder_filter: str = "") -> None:
    """Main indexing pipeline."""
    vault = config.VAULT_PATH
    if not vault.exists():
        console.print(f"[bold red]ERROR:[/] Vault not found at {vault}")
        sys.exit(1)

    md_files = sorted(vault.rglob("*.md"))
    if folder_filter:
        md_files = [f for f in md_files if folder_filter in str(f)]

    if not md_files:
        console.print("[yellow]No markdown files found.[/]")
        return

    console.print(f"\n[bold cyan]Developer Second Brain — Indexer[/]")
    console.print(f"Vault  : [dim]{vault}[/]")
    console.print(f"Files  : [bold]{len(md_files)}[/] markdown files found\n")

    collection = build_collection(reset=reset)
    existing_ids = get_indexed_ids(collection)

    total_chunks = skipped = failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing...", total=len(md_files))

        for md_file in md_files:
            progress.update(task, description=f"[dim]{md_file.name[:50]}[/dim]")

            doc = parse_markdown_file(md_file, vault)
            if doc is None:
                failed += 1
                progress.advance(task)
                continue

            chunks = chunk_document(doc, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            new_chunks = [c for c in chunks if c["id"] not in existing_ids]

            if not new_chunks:
                skipped += 1
                progress.advance(task)
                continue

            try:
                collection.upsert(
                    ids=[c["id"] for c in new_chunks],
                    documents=[c["text"] for c in new_chunks],
                    metadatas=[c["metadata"] for c in new_chunks],
                )
                total_chunks += len(new_chunks)
            except Exception as e:
                log.warning(f"Failed to upsert {md_file.name}: {e}")
                failed += 1

            progress.advance(task)

    console.print(f"\n[bold green]Done![/]")
    console.print(f"  Chunks added   : [bold]{total_chunks}[/]")
    console.print(f"  Files skipped  : [dim]{skipped}[/] (already indexed)")
    if failed:
        console.print(f"  Files failed   : [bold red]{failed}[/]")
    console.print(f"  DB path        : [dim]{config.DB_PATH}[/]\n")


def _require_api_key() -> None:
    if not config.OPENAI_API_KEY:
        console.print("[bold red]ERROR:[/] OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index your Obsidian vault into ChromaDB.")
    parser.add_argument("--reset", action="store_true", help="Clear the DB and re-index everything.")
    parser.add_argument("--folder", default="", help="Only index files under this folder name.")
    args = parser.parse_args()

    run(reset=args.reset, folder_filter=args.folder)
