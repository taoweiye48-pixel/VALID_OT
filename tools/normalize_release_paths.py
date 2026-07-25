"""Replace machine-local paths in copied text artifacts with portable paths."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".R",
    ".sha256",
    ".tab",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}

BS = chr(92)
REPLACEMENTS = {
    f"C:{BS}Users{BS}Administrator{BS}Desktop{BS}VALIDOT{BS}VALID_OT_BENCHMARK{BS}03_data_processed{BS}": "data/processed/",
    f"C:{BS}Users{BS}Administrator{BS}Desktop{BS}VALIDOT{BS}VALID_OT_BENCHMARK{BS}02_data_raw{BS}": "data/raw/",
    f"C:{BS}Users{BS}Administrator{BS}Desktop{BS}VALIDOT{BS}VALID_OT_BENCHMARK{BS}": ".",
    f"C:{BS}Users{BS}Administrator{BS}Desktop{BS}MHAgent_Auditing finite-intervention m_20260717_163616{BS}workspace{BS}": ".",
    f"C:{BS}Users{BS}Administrator{BS}Desktop{BS}VALID_OT_MANUSCRIPT_LATEX_20260719{BS}": "paper/",
}


def normalize(path: Path) -> bool:
    if path.suffix not in TEXT_SUFFIXES:
        return False
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    updated = content
    for source, target in REPLACEMENTS.items():
        updated = updated.replace(source, target)
    if updated == content:
        return False
    path.write_text(updated, encoding="utf-8", newline="")
    return True


def main() -> None:
    changed = 0
    for path in ROOT.rglob("*"):
        if path.is_file() and normalize(path):
            changed += 1
    print(f"normalized {changed} text artifacts")


if __name__ == "__main__":
    main()
