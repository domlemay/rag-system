"""CLI middleware: reads a user prompt, enriches it with RAG context, prints to stdout.

Designed to be piped into any AI tool, IDE extension, or shell script.
Works in two modes:
  - Direct (default): imports the API layer in-process — no server needed.
  - HTTP (--api):     calls the local API server — requires it to be running.

Usage:
    # Pipe a prompt
    echo "How do I handle async timeouts in Python?" | python middleware/enrich.py

    # Pass as argument
    python middleware/enrich.py "How do I use the repository pattern?"

    # Context block only (no user prompt in output)
    python middleware/enrich.py --context-only "async timeout pattern"

    # JSON output for programmatic consumers
    python middleware/enrich.py --json "repository pattern"

    # Via HTTP API (server must be running on port 8765)
    python middleware/enrich.py --api "How do I handle async timeouts?"

    # Control chunk count
    python middleware/enrich.py --top-k 8 "dependency injection patterns"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

_API_BASE = "http://127.0.0.1:8765"


# ── Transport layer ───────────────────────────────────────────────────────────

def _call_api(prompt: str, top_k: int) -> dict:
    """POST to the local HTTP server. Raises RuntimeError if unreachable."""
    payload = json.dumps({"prompt": prompt, "top_k": top_k}).encode()
    req = urllib.request.Request(
        f"{_API_BASE}/enrich",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach API at {_API_BASE}.\n"
            f"Start it first:  uvicorn api.server:app --host 127.0.0.1 --port 8765\n"
            f"Error: {exc}"
        ) from exc


def _call_direct(prompt: str, top_k: int) -> dict:
    """Import API layer directly — no server needed."""
    from api.prompt_enricher import enrich
    enriched, ctx = enrich(prompt, top_k=top_k)
    return {
        "enriched_prompt": enriched,
        "context_added":   bool(ctx.context),
        "chunks_used":     ctx.chunks_used,
        "sources":         ctx.sources,
    }


# ── Output formatting ─────────────────────────────────────────────────────────

def _extract_context_block(enriched: str) -> str:
    """Pull just the CONTEXT block out of an enriched prompt."""
    lines = enriched.split("\n")
    in_context = False
    out: list[str] = []
    for line in lines:
        if line.startswith("=== CONTEXT"):
            in_context = True
            continue
        if line.startswith("===") and in_context:
            break
        if in_context:
            out.append(line)
    return "\n".join(out).strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich a prompt with RAG context from the Developer Brain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "prompt", nargs="?",
        help="The user prompt. Omit to read from stdin.",
    )
    parser.add_argument("--top-k", type=int, default=config.TOP_K,
                        help=f"Chunks to retrieve (default: {config.TOP_K}).")
    parser.add_argument("--context-only", action="store_true",
                        help="Output only the context block, not the full enriched prompt.")
    parser.add_argument("--json", action="store_true",
                        help="Output full JSON (enriched_prompt, sources, metadata).")
    parser.add_argument("--api", action="store_true",
                        help="Use HTTP API mode (server must be running).")
    args = parser.parse_args()

    # ── Read prompt ──────────────────────────────────────────────────────────
    prompt = args.prompt
    if not prompt:
        if sys.stdin.isatty():
            parser.error("Provide a prompt as an argument or pipe via stdin.")
        prompt = sys.stdin.read().strip()
    if not prompt:
        parser.error("Empty prompt.")

    # ── Retrieve context ─────────────────────────────────────────────────────
    try:
        result = _call_api(prompt, args.top_k) if args.api else _call_direct(prompt, args.top_k)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    enriched      = result.get("enriched_prompt", prompt)
    chunks_used   = result.get("chunks_used", 0)
    sources       = result.get("sources", [])
    context_added = result.get("context_added", False)

    # ── Output ────────────────────────────────────────────────────────────────
    if args.json:
        print(json.dumps({
            "enriched_prompt": enriched,
            "context_added":   context_added,
            "chunks_used":     chunks_used,
            "sources":         sources,
        }, indent=2))
        return

    if args.context_only:
        print(_extract_context_block(enriched))
        return

    print(enriched)

    # Metadata to stderr so it doesn't pollute piped output
    if chunks_used > 0:
        print(f"\n[RAG] {chunks_used} chunks injected from Developer Brain.", file=sys.stderr)
    else:
        print("[RAG] No relevant context found — prompt sent unchanged.", file=sys.stderr)


if __name__ == "__main__":
    main()
