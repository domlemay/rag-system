"""Install the auto-learning post-commit git hook.

Creates a post-commit hook in the target repo's .git/hooks/ directory.
After each commit, it silently extracts and saves knowledge to the vault.

Usage:
    # Install in dev-brain repo (most common)
    python scripts/install_git_hook.py --target ../dev-brain

    # Install in rag-system repo
    python scripts/install_git_hook.py

    # Install in any repo
    python scripts/install_git_hook.py --target /path/to/any/repo

    # Remove the hook
    python scripts/install_git_hook.py --target ../dev-brain --uninstall

Safety:
    - Existing post-commit hooks are backed up before overwriting.
    - The hook runs in the background (& on Unix, Start-Process on Windows).
    - Set DEV_BRAIN_LEARN=0 before a commit to skip learning for that commit.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import stat
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent

# ── Hook templates ────────────────────────────────────────────────────────────

_HOOK_UNIX = """\
#!/bin/sh
# Developer Brain — auto-learning post-commit hook
# Extracts knowledge from each commit and saves to the vault.
# Skip for one commit: DEV_BRAIN_LEARN=0 git commit -m "..."

[ "$DEV_BRAIN_LEARN" = "0" ] && exit 0

RAG_DIR="{rag_dir}"
SCRIPT="$RAG_DIR/scripts/auto_learn.py"

[ -f "$SCRIPT" ] && python "$SCRIPT" --from-commit --yes > /dev/null 2>&1 &
exit 0
"""

_HOOK_WIN = """\
#!/bin/sh
# Developer Brain — auto-learning post-commit hook (Windows/Git Bash)
# Skip for one commit: set DEV_BRAIN_LEARN=0 && git commit -m "..."

[ "$DEV_BRAIN_LEARN" = "0" ] && exit 0

RAG_DIR="{rag_dir}"
SCRIPT="$RAG_DIR/scripts/auto_learn.py"

[ -f "$SCRIPT" ] && python "$SCRIPT" --from-commit --yes > /dev/null 2>&1 &
exit 0
"""


def install(repo_path: Path, rag_dir: Path) -> None:
    hooks_dir = repo_path / ".git" / "hooks"
    if not hooks_dir.is_dir():
        print(f"ERROR: Not a git repo (no .git/hooks): {repo_path}")
        sys.exit(1)

    hook_path = hooks_dir / "post-commit"
    rag_dir_str = str(rag_dir).replace("\\", "/")

    template = _HOOK_WIN if platform.system() == "Windows" else _HOOK_UNIX
    content  = template.format(rag_dir=rag_dir_str)

    # Back up existing hook
    if hook_path.exists():
        backup = hooks_dir / "post-commit.bak"
        shutil.copy2(hook_path, backup)
        print(f"Backed up existing hook → {backup}")

    hook_path.write_text(content, encoding="utf-8")

    # Make executable (Unix/macOS)
    if platform.system() != "Windows":
        current = stat.S_IMODE(hook_path.stat().st_mode)
        hook_path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"\nInstalled post-commit hook → {hook_path}")
    print(f"RAG system: {rag_dir}")
    print("\nUsage:")
    print("  git commit -m 'feat: ...'      # auto-learning runs silently after commit")
    print("  DEV_BRAIN_LEARN=0 git commit   # skip learning for this commit")


def uninstall(repo_path: Path) -> None:
    hook_path = repo_path / ".git" / "hooks" / "post-commit"
    if hook_path.exists():
        hook_path.unlink()
        print(f"Removed post-commit hook: {hook_path}")
    else:
        print("No post-commit hook found.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the auto-learning git hook.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--target",    default=".",
                        help="Target git repo path (default: current directory).")
    parser.add_argument("--rag-dir",   default=str(_ROOT),
                        help=f"Path to rag-system directory (default: {_ROOT}).")
    parser.add_argument("--uninstall", action="store_true",
                        help="Remove the hook instead of installing.")
    args = parser.parse_args()

    repo_path = Path(args.target).resolve()
    rag_dir   = Path(args.rag_dir).resolve()

    if args.uninstall:
        uninstall(repo_path)
    else:
        install(repo_path, rag_dir)


if __name__ == "__main__":
    main()
