"""Create a deterministic SHA-256 manifest for the public release candidate.

Transient build files and caches are intentionally excluded.  The manifest is
for release auditing; it is not a substitute for a Git tag or an archive DOI.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "RELEASE_MANIFEST.sha256"
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    ".venv",
}
SKIP_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".synctex.gz",
}


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    if path.name == OUT.name:
        return True
    return False


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    files = sorted(
        p for p in ROOT.rglob("*") if p.is_file() and not should_skip(p)
    )
    lines = [
        "# VALID-OT public release candidate SHA-256 manifest",
        "# Paths are repository-relative and use forward slashes.",
    ]
    lines.extend(
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in files
    )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(files)} files)")


if __name__ == "__main__":
    main()
