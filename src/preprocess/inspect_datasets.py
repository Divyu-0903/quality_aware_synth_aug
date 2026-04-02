from __future__ import annotations

from pathlib import Path
from itertools import islice

from src.utils.io import load_jsonl


def preview_jsonl(path: str | Path, n: int = 2) -> None:
    records = load_jsonl(path)
    print(f"\nPreview: {path}")
    for rec in islice(records, n):
        print(rec)


def main() -> None:
    preview_jsonl("data/raw/ner/train.jsonl")
    preview_jsonl("data/raw/sentiment/train.jsonl")


if __name__ == "__main__":
    main()
