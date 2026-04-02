from __future__ import annotations

from itertools import islice

from src.utils.io import load_jsonl


def preview(path: str, n: int = 3) -> None:
    print(f"\nPreview: {path}")
    records = load_jsonl(path)
    for rec in islice(records, n):
        print(rec)


def main() -> None:
    preview("data/synthetic/ner/synthetic_ner.jsonl")
    preview("data/synthetic/sentiment/synthetic_sentiment.jsonl")


if __name__ == "__main__":
    main()
