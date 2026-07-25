#!/usr/bin/env python3
"""Validate the SocioFM release layout and Python syntax."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "DATA_CARD.md",
    "MODEL_CARD.md",
    "LICENSE",
    "pyproject.toml",
    "SHA256SUMS.txt",
    "release_manifest.json",
    "src/sociofm/data/build_dataset.py",
    "src/sociofm/training/train_compact.py",
    "src/sociofm/evaluation/run_downstream_suite.py",
    "src/sociofm/evaluation/run_representation_benchmark.py",
]


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).exists()]
    if missing:
        raise SystemExit(f"Missing required release files: {missing}")
    python_files = sorted((ROOT / "src").rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"Release layout valid; parsed {len(python_files)} Python files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
